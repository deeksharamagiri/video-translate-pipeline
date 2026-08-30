"""
orchestrator.py

Wires:

Stage 1
   ↓
Stage 2 ASR
   ↓
Stage 3 segmentation / TM
   ↓
Translation
   ↓
Automatic Translation Quality
   ↓
Stage 4 subtitles
   ↓
Voiceover / burned-in delivery
   ↓
Automatic Voiceover Quality
   ↓
Stage 5 archive

Every generated job receives a quality report.

TIMING INSTRUMENTATION
-----------------------
Every major stage is wrapped in a `_Stage(...)` context manager, which
records wall-clock seconds for that stage into a per-job `timings` dict.
This exists purely to answer "why did an N-minute video take M minutes" --
it doesn't change any pipeline behavior. The full breakdown is printed to
stdout when the job finishes and is also returned as
`outputs["stats"]["timing_seconds"]` for programmatic inspection.
"""

import os
import shutil
import time
import uuid
from contextlib import contextmanager
from typing import Callable, Optional

from config import (
    JOBS_DIR,
    ALLOWED_VIDEO_EXT,
    ALLOWED_AUDIO_EXT,
)

from pipeline import (
    stage1_preprocess,
    stage2_asr,
    stage3_segment_tm,
    stage4_subtitle,
    stage5_archive,
    translate,
    report,
    delivery,
)
from pipeline.logging_setup import get_logger

log = get_logger("orchestrator")


class JobError(Exception):
    pass


def _progress(
    cb: Optional[Callable],
    stage: str,
    message: str,
    pct: int,
):

    if cb:

        cb({
            "stage": stage,
            "message": message,
            "pct": pct,
        })


@contextmanager
def _Stage(timings: dict, name: str):
    """
    Records wall-clock seconds spent inside the `with` block under
    timings[name]. If the same name is used more than once in a job
    (shouldn't normally happen), later calls add to the existing total
    rather than overwriting it.
    """

    start = time.perf_counter()

    try:
        yield

    finally:
        elapsed = time.perf_counter() - start
        timings[name] = timings.get(name, 0.0) + elapsed


def _print_timing_summary(job_id: str, timings: dict, total_elapsed: float):

    print(f"\n[orchestrator] Job {job_id} timing breakdown:")

    # Sort slowest-first so the dominant cost is immediately visible.
    for name, seconds in sorted(
        timings.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        pct_of_total = (
            (seconds / total_elapsed) * 100.0
            if total_elapsed > 0
            else 0.0
        )

        print(
            f"[orchestrator]   {name:<24} "
            f"{seconds:8.2f}s  ({pct_of_total:5.1f}%)"
        )

    accounted = sum(timings.values())
    unaccounted = total_elapsed - accounted

    print(
        f"[orchestrator]   {'(unaccounted)':<24} "
        f"{unaccounted:8.2f}s  "
        f"({(unaccounted / total_elapsed * 100.0) if total_elapsed > 0 else 0.0:5.1f}%)"
    )

    print(
        f"[orchestrator]   {'TOTAL':<24} "
        f"{total_elapsed:8.2f}s\n"
    )


def run_job(
    input_path: str,
    source_lang_hint: Optional[str],
    target_lang: str,
    want_burned_in: bool,
    want_voiceover: bool,
    progress_cb: Optional[Callable] = None,
    engine_override: str = "auto",
    asr_engine: str = "whisper",
    tts_speaker: Optional[str] = None,
) -> dict:
    """
    Runs one end-to-end job. On any failure, partial output under
    jobs/<job_id>/ is rolled back (deleted) rather than left half-written,
    and the failure is logged (console + jobs/pipeline.log) with a full
    traceback so it can be diagnosed after the fact.
    """

    job_id = uuid.uuid4().hex[:12]

    job_dir = os.path.join(
        JOBS_DIR,
        job_id,
    )

    log.info(
        f"Job {job_id} starting "
        f"(target_lang={target_lang}, engine={engine_override}, "
        f"burned_in={want_burned_in}, voiceover={want_voiceover})"
    )

    try:
        result = _run_job(
            job_id,
            job_dir,
            input_path,
            source_lang_hint,
            target_lang,
            want_burned_in,
            want_voiceover,
            progress_cb,
            engine_override,
            asr_engine,
            tts_speaker,
        )

    except Exception:
        log.exception(
            f"Job {job_id} failed -- rolling back partial output at {job_dir}"
        )
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    log.info(f"Job {job_id} finished ok")
    return result


def _run_job(
    job_id: str,
    job_dir: str,
    input_path: str,
    source_lang_hint: Optional[str],
    target_lang: str,
    want_burned_in: bool,
    want_voiceover: bool,
    progress_cb: Optional[Callable] = None,
    engine_override: str = "auto",
    asr_engine: str = "whisper",
    tts_speaker: Optional[str] = None,
) -> dict:

    job_start = time.perf_counter()
    timings: dict = {}

    work_dir = os.path.join(
        job_dir,
        "work",
    )

    out_dir = os.path.join(
        job_dir,
        "output",
    )

    os.makedirs(
        work_dir,
        exist_ok=True,
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    ext = os.path.splitext(
        input_path
    )[1].lower()

    if ext in ALLOWED_VIDEO_EXT:

        input_kind = "video"

    elif ext in ALLOWED_AUDIO_EXT:

        input_kind = "audio"

    else:

        raise JobError(
            f"Unsupported file extension: {ext}"
        )

    local_input = os.path.join(
        work_dir,
        f"input{ext}",
    )

    shutil.copy(
        input_path,
        local_input,
    )

    # =====================================================
    # Stage 1
    # =====================================================

    _progress(
        progress_cb,
        "stage1",
        "Extracting & normalising audio...",
        5,
    )

    with _Stage(timings, "stage1_preprocess"):

        pre = (
            stage1_preprocess.run_stage1(
                local_input,
                work_dir,
                input_kind,
            )
        )

    # =====================================================
    # Stage 2
    # =====================================================

    if pre.subtitle_track_found:

        _progress(
            progress_cb,
            "skip_asr",
            "Embedded subtitles found — skipping ASR.",
            25,
        )

        with _Stage(timings, "stage2_asr"):

            (
                asr_segments,
                detected_lang,
            ) = _segments_from_srt(
                pre.subtitle_srt_path
            )

        asr_dropped_ranges = []

    else:

        _progress(
            progress_cb,
            "stage2",
            "Running speech recognition (whisper.cpp)...",
            20,
        )

        with _Stage(timings, "stage2_asr"):

            asr_result = (
                stage2_asr.run_stage2(
                    pre.audio_wav_path,
                    language_hint=source_lang_hint,
                )
            )

            asr_segments = (
                asr_result.segments
            )

            asr_dropped_ranges = (
                asr_result.dropped_ranges
            )

            detected_lang = (
                source_lang_hint
                or asr_result.detected_language
            )

    source_lang = _normalize_lang(
        source_lang_hint
        or detected_lang
    )

    if not asr_segments:

        raise JobError(
            "No usable speech was transcribed from this file "
            f"(auto-detected source language: {detected_lang or 'unknown'}). "
            "A common cause: whisper.cpp misdetected the source language, "
            "so real speech gets decoded through the wrong language's "
            "model -- this degrades into repeated-token hallucination over "
            "a long clip, which the hallucination filter then correctly "
            "drops, leaving nothing. Other causes: the audio may be mostly "
            "silent, non-speech (music/noise), or too quiet. Try again "
            "with the source language set explicitly instead of "
            "auto-detect if you know it."
        )

    # =====================================================
    # Optional IndicConformer refinement
    # =====================================================

    if asr_engine == "indic_conformer":

        _progress(
            progress_cb,
            "asr_refine",
            "Refining transcript with IndicConformer...",
            30,
        )

        with _Stage(timings, "asr_refine_indic_conformer"):

            asr_segments = (
                stage2_asr
                .refine_segments_with_indic_conformer(
                    asr_segments,
                    pre.audio_wav_path,
                    source_lang,
                )
            )

    # =====================================================
    # Stage 3
    # =====================================================

    _progress(
        progress_cb,
        "stage3",
        "Segmenting & checking translation memory...",
        45,
    )

    with _Stage(timings, "stage3_segment_tm"):

        glossary_version = (
            translate.glossary_version_tag()
        )

        model_version = (
            translate.model_version_tag(
                source_lang,
                target_lang,
                engine_override,
            )
        )

        segments = (
            stage3_segment_tm.chunk_segments(
                asr_segments,
                source_lang,
                target_lang,
            )
        )

        misses = (
            stage3_segment_tm
            .apply_translation_memory(
                segments,
                source_lang,
                target_lang,
                glossary_version,
                model_version,
            )
        )

    # =====================================================
    # Translation
    # =====================================================

    engine_name = (
        "Translation Memory (cache only)"
    )

    with _Stage(timings, "translation"):

        if misses:

            total_misses = len(misses)

            def _translate_progress(done: int, total: int):
                frac = done / total if total else 1.0
                # Translation occupies the 60-75% band of the overall job.
                pct = 60 + round(frac * 15)
                _progress(
                    progress_cb,
                    "translate",
                    f"Translating segment {done} of {total} ({round(frac * 100)}%)...",
                    pct,
                )

            _progress(
                progress_cb,
                "translate",
                f"Translating segment 0 of {total_misses} (0%)...",
                60,
            )

            texts = [
                s.text
                for s in misses
            ]

            (
                translated_texts,
                engine_name,
            ) = translate.translate_batch(
                texts,
                source_lang,
                target_lang,
                engine_override,
                progress_cb=_translate_progress,
            )

            for seg, tr in zip(
                misses,
                translated_texts,
            ):

                seg.translated_text = tr

        else:

            _progress(
                progress_cb,
                "translate",
                f"All {len(segments)} segment(s) served from translation memory.",
                60,
            )

    # =====================================================
    # Automatic Translation Quality
    # =====================================================

    _progress(
        progress_cb,
        "translation_qc",
        "Calculating automatic translation quality...",
        68,
    )

    with _Stage(timings, "translation_qc"):

        translation_quality = (
            delivery.calculate_translation_quality(
                segments
            )
        )

    # =====================================================
    # Stage 4
    # =====================================================

    _progress(
        progress_cb,
        "stage4",
        "Generating SRT/VTT subtitle files...",
        75,
    )

    srt_path = os.path.join(
        out_dir,
        f"{job_id}_subtitles.srt",
    )

    vtt_path = os.path.join(
        out_dir,
        f"{job_id}_subtitles.vtt",
    )

    with _Stage(timings, "stage4_subtitle"):

        stage4_subtitle.generate_srt(
            segments,
            srt_path,
        )

        stage4_subtitle.generate_vtt(
            segments,
            vtt_path,
        )

    # =====================================================
    # Quality information
    # =====================================================

    quality_info = {
        # -----------------------------
        # Translation quality
        # -----------------------------

        "overall_translation_quality_score":
            translation_quality[
                "overall_translation_quality_score"
            ],

        "average_asr_confidence":
            translation_quality[
                "average_asr_confidence"
            ],

        "translation_completeness_score":
            translation_quality[
                "translation_completeness_score"
            ],

        "numeric_preservation_score":
            translation_quality[
                "numeric_preservation_score"
            ],

        "translation_segments":
            translation_quality[
                "translation_segments"
            ],

        "translation_warnings":
            translation_quality[
                "translation_warnings"
            ],

        # -----------------------------
        # Voiceover quality
        # -----------------------------

        "voiceover_requested":
            want_voiceover,

        "tts_success_segments": 0,

        "tts_failed_segments": 0,

        "tts_coverage_percent": (
            100.0
            if not want_voiceover
            else 0.0
        ),

        "numeric_segments": 0,

        "numeric_expressions": 0,

        "decimal_expressions": 0,

        "numeric_normalization_success": 0,

        "numeric_normalization_failures": 0,

        "numeric_normalization_percent": (
            100.0
            if not want_voiceover
            else 0.0
        ),

        "average_alignment_score": (
            100.0
            if not want_voiceover
            else 0.0
        ),

        "overall_quality_score": (
            100.0
            if not want_voiceover
            else 0.0
        ),

        "alignment_scores": [],

        "segment_details": [],

        "warnings": [],
    }

    if asr_dropped_ranges:

        gap_desc = ", ".join(
            f"{s:.0f}s-{e:.0f}s"
            for s, e in asr_dropped_ranges
        )

        total_dropped_sec = sum(
            e - s
            for s, e in asr_dropped_ranges
        )

        quality_info["warnings"].append(
            f"{total_dropped_sec:.0f}s of audio ({gap_desc}) could not be "
            "transcribed reliably (whisper.cpp produced repeated-token "
            "hallucination there, which was dropped) and has no subtitles, "
            "translation, or voiceover as a result. Try re-running with an "
            "explicit source language, or a different WHISPER_BEAM_SIZE/"
            "WHISPER_BEST_OF, if this stretch has real speech."
        )

    # =====================================================
    # Voiceover
    # =====================================================

    voiceover_source_for_burn = None

    if (
        input_kind == "video"
        and want_voiceover
    ):

        if not delivery.voiceover_available(
            target_lang
        ):

            message = (
                f"No Indic-TTS support for "
                f"'{target_lang}' -- "
                "skipping voiceover."
            )

            _progress(
                progress_cb,
                "voiceover",
                message,
                92,
            )

            quality_info[
                "voiceover_requested"
            ] = True

            quality_info[
                "warnings"
            ].append(
                message
            )

            # Voiceover was requested but unavailable.
            quality_info[
                "tts_coverage_percent"
            ] = 0.0

            quality_info[
                "overall_quality_score"
            ] = 0.0

            # Without this, `outputs_voiceover` is referenced further
            # down (building the `outputs` dict) but was never assigned
            # on this path -- a pre-existing NameError landmine that
            # would crash the whole job the moment a target language
            # with no Indic-TTS support requested voiceover.
            outputs_voiceover = None

        else:

            _progress(
                progress_cb,
                "voiceover",
                "Synthesising voiceover (Indic-TTS)...",
                92,
            )

            try:

                with _Stage(timings, "voiceover_tts"):

                    (
                        voiceover_wav,
                        voiceover_quality,
                    ) = delivery.build_voiceover_track(
                        segments,
                        target_lang,
                        work_dir,
                        speaker=tts_speaker,
                    )

            except delivery.NoVoiceoverContentError as exc:

                # A segment producing no synthesizable content (already
                # counted as "SKIPPED"/"FAILED" in per-segment quality
                # details) is one thing; EVERY segment being like that
                # is a whole-video edge case, usually downstream of
                # severe ASR hallucination leaving only fragments behind
                # (verified directly: a real job where hallucination
                # dropped most chunks and every surviving segment's
                # translated text was just punctuation). That shouldn't
                # take the whole job down with it -- subtitles/SRT/VTT
                # from whatever ASR DID produce are still a real,
                # deliverable result, so degrade the same way an
                # unsupported target language already does (skip
                # voiceover, keep going) rather than rolling everything
                # back.
                message = (
                    f"Voiceover could not be generated for any segment "
                    f"({exc}) -- skipping voiceover. Subtitles are "
                    f"still produced from whatever speech was "
                    f"transcribed."
                )

                _progress(
                    progress_cb,
                    "voiceover",
                    message,
                    92,
                )

                quality_info[
                    "voiceover_requested"
                ] = True

                quality_info[
                    "warnings"
                ].append(
                    message
                )

                quality_info[
                    "tts_coverage_percent"
                ] = 0.0

                quality_info[
                    "overall_quality_score"
                ] = 0.0

                voiceover_wav = None

            if voiceover_wav is not None:

                # Preserve the translation metrics that were
                # calculated before voiceover generation.
                #
                # quality_info["warnings"] may already hold the ASR
                # dropped-audio warning (appended above, before this
                # stage ever runs). voiceover_quality also has its own
                # "warnings" key -- dict.update() would silently
                # replace, not merge, that key, discarding the ASR
                # warning even though it was already fully populated.
                # Merge explicitly instead.
                warnings_before_voiceover = quality_info["warnings"]

                quality_info.update(
                    voiceover_quality
                )

                quality_info["warnings"] = (
                    warnings_before_voiceover
                    + voiceover_quality.get("warnings", [])
                )

                quality_info.update({
                    "overall_translation_quality_score":
                        translation_quality[
                            "overall_translation_quality_score"
                        ],

                    "average_asr_confidence":
                        translation_quality[
                            "average_asr_confidence"
                        ],

                    "translation_completeness_score":
                        translation_quality[
                            "translation_completeness_score"
                        ],

                    "numeric_preservation_score":
                        translation_quality[
                            "numeric_preservation_score"
                        ],

                    "translation_segments":
                        translation_quality[
                            "translation_segments"
                        ],

                    "translation_warnings":
                        translation_quality[
                            "translation_warnings"
                        ],
                })

                voiceover_path = os.path.join(
                    out_dir,
                    f"{job_id}_voiceover.mp4",
                )

                with _Stage(timings, "voiceover_mux"):

                    delivery.mux_voiceover_onto_video(
                        local_input,
                        voiceover_wav,
                        voiceover_path,
                    )

                outputs_voiceover = (
                    voiceover_path
                )

                voiceover_source_for_burn = (
                    voiceover_path
                )

            else:

                outputs_voiceover = None

    else:

        outputs_voiceover = None

    # =====================================================
    # Burned-in output
    # =====================================================

    burned_path = None

    if (
        input_kind == "video"
        and want_burned_in
    ):

        _progress(
            progress_cb,
            "burned_in",
            "Burning subtitles into video...",
            96,
        )

        burned_path = os.path.join(
            out_dir,
            f"{job_id}_burned_in.mp4",
        )

        source_video = (
            voiceover_source_for_burn
            or local_input
        )

        with _Stage(timings, "burn_in_subtitles"):

            delivery.burn_in_subtitles(
                source_video,
                srt_path,
                burned_path,
            )

    # =====================================================
    # Outputs
    # =====================================================

    outputs = {
        "job_id": job_id,

        "srt": srt_path,

        "vtt": vtt_path,

        "job_report_docx": None,

        "burned_in_mp4": burned_path,

        "voiceover_mp4":
            outputs_voiceover,

        "quality":
            quality_info,
    }

    # NOTE: when both burned_in and voiceover are requested,
    # burn_in_subtitles() above already burns the captions onto the
    # voiceover (dubbed-audio) video rather than the plain source, so
    # outputs["burned_in_mp4"] is a combined dub+captions file and
    # outputs["voiceover_mp4"] (outputs_voiceover, set above) is the
    # plain dub with no captions -- two genuinely distinct deliverables.
    # This used to be collapsed into one file here by reassigning
    # outputs["voiceover_mp4"] = burned_path, which silently discarded
    # the plain-dub file from the API/UI's output listing entirely (the
    # file itself is still written to jobs/<id>/output/, just orphaned)
    # even though "voiceover" and "burned-in" are offered as two
    # independently downloadable results.

    # =====================================================
    # Stage 5
    # =====================================================

    _progress(
        progress_cb,
        "stage5",
        "Archiving segments for future reuse...",
        85,
    )

    with _Stage(timings, "stage5_archive"):

        archive_stats = (
            stage5_archive.archive_job(
                job_id,
                segments,
                source_lang,
                target_lang,
                glossary_version,
                model_version,
            )
        )

    # =====================================================
    # Job report
    # =====================================================

    docx_path = os.path.join(
        out_dir,
        f"{job_id}_job_report.docx",
    )

    with _Stage(timings, "job_report_docx"):

        report.generate_job_report(
            job_id,
            segments,
            source_lang,
            target_lang,
            engine_name,
            pre.to_dict(),
            docx_path,
            quality_info=quality_info,
        )

    outputs[
        "job_report_docx"
    ] = docx_path

    # =====================================================
    # Finish
    # =====================================================

    _progress(
        progress_cb,
        "done",
        "Done — ready for USB transfer / offline playback.",
        100,
    )

    total_elapsed = time.perf_counter() - job_start

    _print_timing_summary(
        job_id,
        timings,
        total_elapsed,
    )

    outputs["stats"] = {

        "source_lang":
            source_lang,

        "target_lang":
            target_lang,

        "engine":
            engine_name,

        "segment_count":
            archive_stats[
                "segment_count"
            ],

        "cache_hit_count":
            archive_stats[
                "cache_hit_count"
            ],

        "preprocess":
            pre.to_dict(),

        "quality":
            quality_info,

        "timing_seconds": {
            **timings,
            "total": total_elapsed,
        },
    }

    return outputs


def _normalize_lang(
    code: str,
) -> str:

    two_to_three = {
        "hi": "hin",
        "en": "eng",
        "mr": "mar",
        "bn": "ben",
        "ta": "tam",
        "te": "tel",
        "kn": "kan",
        "ml": "mal",
        "gu": "guj",
        "pa": "pan",
        "ur": "urd",
        "or": "ori",
        "as": "asm",
        "ne": "nep",
        "sa": "san",
        "fr": "fra",
        "es": "spa",
        "de": "deu",
        "zh": "zho",
        "ar": "ara",
        "pt": "por",
        "ru": "rus",
        "ja": "jpn",
    }

    return two_to_three.get(
        code,
        code,
    )


def _segments_from_srt(
    srt_path: str,
):

    import re

    from pipeline.stage2_asr import (
        TranscriptSegment,
    )

    with open(
        srt_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        content = f.read()

    blocks = re.split(
        r"\n\s*\n",
        content.strip(),
    )

    segments = []

    time_re = re.compile(
        r"(\d{2}):"
        r"(\d{2}):"
        r"(\d{2})[,\.]"
        r"(\d{3})\s*-->\s*"
        r"(\d{2}):"
        r"(\d{2}):"
        r"(\d{2})[,\.]"
        r"(\d{3})"
    )

    for block in blocks:

        lines = (
            block
            .strip()
            .splitlines()
        )

        if len(lines) < 2:
            continue

        m = (
            time_re.search(
                lines[1]
            )
            if time_re.search(
                lines[1]
            )
            else time_re.search(
                lines[0]
            )
        )

        if not m:
            continue

        (
            h1,
            m1,
            s1,
            ms1,
            h2,
            m2,
            s2,
            ms2,
        ) = map(
            int,
            m.groups(),
        )

        start = (
            h1 * 3600
            + m1 * 60
            + s1
            + ms1 / 1000
        )

        end = (
            h2 * 3600
            + m2 * 60
            + s2
            + ms2 / 1000
        )

        text_lines = (
            lines[2:]
            if time_re.search(
                lines[1]
            )
            else lines[1:]
        )

        text = " ".join(
            text_lines
        ).strip()

        if text:

            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    confidence=1.0,
                )
            )

    return segments, None