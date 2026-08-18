"""
orchestrator.py — wires Stage 1 -> 2 -> 3 -> 4 -> 5 exactly in the order
drawn in the architecture diagram, and produces every output box at the
bottom (SRT + VTT + Job Report always; Burned-in MP4 / Voiceover MP4 opt-in).
"""
import os
import shutil
import uuid
from typing import Callable, Optional

from config import JOBS_DIR, ALLOWED_VIDEO_EXT, ALLOWED_AUDIO_EXT
from pipeline import stage1_preprocess, stage2_asr, stage3_segment_tm, stage4_subtitle
from pipeline import stage5_archive, translate, report, delivery


class JobError(Exception):
    pass


def _progress(cb: Optional[Callable], stage: str, message: str, pct: int):
    if cb:
        cb({"stage": stage, "message": message, "pct": pct})


def run_job(input_path: str, source_lang_hint: Optional[str], target_lang: str,
            want_burned_in: bool, want_voiceover: bool,
            progress_cb: Optional[Callable] = None,
            engine_override: str = "auto",
            job_id: Optional[str] = None) -> dict:
    """
    Full pipeline run for one uploaded file. Returns a dict describing all
    output file paths + stats, ready to hand back to the Flask route.

    job_id: the caller's job id (e.g. app.py's JOBS-dict key, used for
    status polling/downloads) — the `jobs/<job_id>/` folder and the
    returned outputs["job_id"] use this same value, so everything the
    caller sees stays consistent. Generates one if not provided (e.g. for
    standalone/CLI use).
    """
    if job_id is None:
        job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    work_dir = os.path.join(job_dir, "work")
    out_dir = os.path.join(job_dir, "output")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()
    if ext in ALLOWED_VIDEO_EXT:
        input_kind = "video"
    elif ext in ALLOWED_AUDIO_EXT:
        input_kind = "audio"
    else:
        raise JobError(f"Unsupported file extension: {ext}")

    local_input = os.path.join(work_dir, f"input{ext}")
    shutil.copy(input_path, local_input)

    # ---------------- Stage 1 — Pre-Processing ----------------
    _progress(progress_cb, "stage1", "Extracting & normalising audio...", 5)
    pre = stage1_preprocess.run_stage1(local_input, work_dir, input_kind)

    # ---------------- Subtitle-found branch OR Stage 2 — ASR ----------------
    if pre.subtitle_track_found:
        _progress(progress_cb, "skip_asr", "Embedded subtitles found — skipping ASR.", 25)
        asr_segments, detected_lang = _segments_from_srt(pre.subtitle_srt_path)
    else:
        _progress(progress_cb, "stage2", "Running speech recognition (faster-whisper)...", 20)
        asr_result = stage2_asr.run_stage2(
            pre.audio_wav_path, language_hint=_to_whisper_lang(source_lang_hint)
        )
        asr_segments = asr_result.segments
        detected_lang = source_lang_hint or asr_result.detected_language

    source_lang = _normalize_lang(source_lang_hint or detected_lang)

    # ---------------- Stage 3 — Segmentation + TM check ----------------
    _progress(progress_cb, "stage3", "Segmenting & checking translation memory...", 45)
    glossary_version = translate.glossary_version_tag()
    model_version = translate.model_version_tag(source_lang, target_lang, engine_override)

    segments = stage3_segment_tm.chunk_segments(asr_segments, source_lang, target_lang)

    # Group ASR-timed fragments up to sentence boundaries before translating,
    # so the engine sees full sentences instead of pause-cut fragments (e.g.
    # "and pull into an auto repair shop." with no subject) — subtitle cue
    # count/timing is unaffected, Stage 4 still renders one cue per Segment.
    groups = stage3_segment_tm.group_sentences(segments)
    miss_groups = stage3_segment_tm.apply_translation_memory(
        groups, source_lang, target_lang, glossary_version, model_version
    )

    # ---------------- Translation engine (indic-indic vs en-indic/indic-en) ----------------
    engine_name = "Translation Memory (cache only)"
    if miss_groups:
        _progress(progress_cb, "translate",
                   f"Translating {len(miss_groups)} new sentence(s)...", 60)
        # Prepend each sentence's predecessor as extra (discarded-after)
        # context, so isolated one-liners like "She yells." have something
        # to anchor cross-sentence agreement (e.g. pronoun gender) against.
        context_batch = stage3_segment_tm.build_context_batch(groups, miss_groups)
        texts = [text for text, _ in context_batch]
        translated_texts, engine_name = translate.translate_batch(
            texts, source_lang, target_lang, engine_override
        )
        translated_texts = [
            stage3_segment_tm.strip_context_prefix(translated, had_context)
            for translated, (_, had_context) in zip(translated_texts, context_batch)
        ]
        stage3_segment_tm.assign_group_translations(miss_groups, translated_texts)
    else:
        _progress(progress_cb, "translate", "All segments served from translation memory.", 60)

    segments = stage3_segment_tm.merge_empty_members(segments)

    # ---------------- Stage 4 — Subtitle Generation ----------------
    _progress(progress_cb, "stage4", "Generating SRT/VTT subtitle files...", 75)
    srt_path = os.path.join(out_dir, f"{job_id}_subtitles.srt")
    vtt_path = os.path.join(out_dir, f"{job_id}_subtitles.vtt")
    stage4_subtitle.generate_srt(segments, srt_path)
    stage4_subtitle.generate_vtt(segments, vtt_path)

    # ---------------- Job report (always generated) ----------------
    docx_path = os.path.join(out_dir, f"{job_id}_job_report.docx")
    report.generate_job_report(
        job_id, segments, source_lang, target_lang, engine_name,
        pre.to_dict(), docx_path
    )

    # ---------------- Stage 5 — Archive & Reuse ----------------
    _progress(progress_cb, "stage5", "Archiving segments for future reuse...", 85)
    archive_stats = stage5_archive.archive_job(
        job_id, segments, groups, source_lang, target_lang, glossary_version, model_version
    )

    outputs = {
        "job_id": job_id,
        "srt": srt_path,
        "vtt": vtt_path,
        "job_report_docx": docx_path,
        "burned_in_mp4": None,
        "voiceover_mp4": None,
    }

    # ---------------- User opt-in delivery ----------------
    # If both are selected, burn subtitles onto the voiceover output (not the
    # original) so the result is ONE file with dubbed audio + hardcoded
    # subtitles — previously these ran independently and produced two
    # separate files, neither of which had both features.
    voiceover_source_for_burn = None

    if input_kind == "video" and want_voiceover:
        _progress(progress_cb, "voiceover", "Synthesising voiceover (MMS-TTS)...", 92)
        voiceover_wav = delivery.build_voiceover_track(segments, target_lang, work_dir)
        voiceover_path = os.path.join(out_dir, f"{job_id}_voiceover.mp4")
        delivery.mux_voiceover_onto_video(local_input, voiceover_wav, voiceover_path)
        outputs["voiceover_mp4"] = voiceover_path
        voiceover_source_for_burn = voiceover_path

    if input_kind == "video" and want_burned_in:
        _progress(progress_cb, "burned_in", "Burning subtitles into video...", 96)
        burned_path = os.path.join(out_dir, f"{job_id}_burned_in.mp4")
        source_video = voiceover_source_for_burn or local_input
        delivery.burn_in_subtitles(source_video, srt_path, burned_path)
        outputs["burned_in_mp4"] = burned_path
        if voiceover_source_for_burn:
            # Both were requested: burned_path already has dub audio + subtitles.
            # Point BOTH download keys at the same finished file, so whichever
            # link is clicked ("Voiceover" or "Burned-in") gives the complete
            # result — previously these were two different files and it was
            # easy to open the one that only had half the requested features.
            outputs["voiceover_mp4"] = burned_path

    _progress(progress_cb, "done", "Done — ready for USB transfer / offline playback.", 100)

    outputs["stats"] = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "engine": engine_name,
        "segment_count": archive_stats["segment_count"],
        "cache_hit_count": archive_stats["cache_hit_count"],
        "preprocess": pre.to_dict(),
    }
    return outputs


_TWO_TO_THREE_LANG = {
    "hi": "hin", "en": "eng", "mr": "mar", "bn": "ben", "ta": "tam", "te": "tel",
    "kn": "kan", "ml": "mal", "gu": "guj", "pa": "pan", "ur": "urd", "or": "ori",
    "as": "asm", "ne": "nep", "sa": "san", "fr": "fra", "es": "spa", "de": "deu",
    "zh": "zho", "ar": "ara", "pt": "por", "ru": "rus", "ja": "jpn",
}
_THREE_TO_TWO_LANG = {v: k for k, v in _TWO_TO_THREE_LANG.items()}


def _normalize_lang(code: str) -> str:
    """faster-whisper returns ISO 639-1 (e.g. 'hi'); normalize common ones to our 3-letter set."""
    return _TWO_TO_THREE_LANG.get(code, code)


def _to_whisper_lang(code: Optional[str]) -> Optional[str]:
    """
    Our UI/internal codes are 3-letter (e.g. 'eng', 'hin'); faster-whisper's
    `language=` hint wants ISO 639-1 2-letter codes (e.g. 'en', 'hi') and
    raises ValueError on anything else. None (auto-detect) passes through.
    """
    if not code:
        return None
    return _THREE_TO_TWO_LANG.get(code, code)


def _segments_from_srt(srt_path: str):
    """Parse an already-extracted embedded .srt into ASR-like segments (skips Stage 2 entirely)."""
    import re
    from pipeline.stage2_asr import TranscriptSegment

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []
    time_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})"
    )
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        m = time_re.search(lines[1]) if time_re.search(lines[1]) else time_re.search(lines[0])
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        text_lines = lines[2:] if time_re.search(lines[1]) else lines[1:]
        text = " ".join(text_lines).strip()
        if text:
            segments.append(TranscriptSegment(start=start, end=end, text=text, confidence=1.0))

    return segments, None