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

Outputs:
    - SRT subtitles
    - VTT subtitles
    - PDF subtitle transcript
    - Job report DOCX
    - Optional voiceover MP4
    - Optional burned-in MP4

TIMING INSTRUMENTATION
-----------------------
Every major stage is wrapped in a `_Stage(...)` context manager, which
records wall-clock seconds for that stage into a per-job `timings` dict.

The full breakdown is printed to stdout when the job finishes and is also
returned as:

    outputs["stats"]["timing_seconds"]
"""

import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from typing import Callable, Optional

# ============================================================
# ReportLab imports for subtitle PDF generation
# ============================================================

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

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
from pipeline.cancellation import JobCancelled, check_cancelled


log = get_logger("orchestrator")


class JobError(Exception):
    pass


# ============================================================
# Progress
# ============================================================

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


# ============================================================
# Timing instrumentation
# ============================================================

@contextmanager
def _Stage(timings: dict, name: str):
    """
    Records wall-clock seconds spent inside the `with` block under
    timings[name].

    If the same name is used more than once in a job, later calls add
    to the existing total rather than overwriting it.
    """

    start = time.perf_counter()

    try:
        yield

    finally:
        elapsed = time.perf_counter() - start

        timings[name] = (
            timings.get(name, 0.0)
            + elapsed
        )


def _print_timing_summary(
    job_id: str,
    timings: dict,
    total_elapsed: float,
):
    print(
        f"\n[orchestrator] Job {job_id} timing breakdown:"
    )

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

    accounted = sum(
        timings.values()
    )

    unaccounted = (
        total_elapsed
        - accounted
    )

    print(
        f"[orchestrator]   {'(unaccounted)':<24} "
        f"{unaccounted:8.2f}s  "
        f"({(unaccounted / total_elapsed * 100.0) if total_elapsed > 0 else 0.0:5.1f}%)"
    )

    print(
        f"[orchestrator]   {'TOTAL':<24} "
        f"{total_elapsed:8.2f}s\n"
    )


# ============================================================
# Quality warnings
# ============================================================

def _print_quality_warnings(
    job_id: str,
    warnings: list,
):
    """
    Quality warnings are printed to the terminal/log rather than
    cluttering the web UI.

    Full details remain available in the DOCX Job Report.
    """

    if not warnings:
        return

    print(
        f"\n[orchestrator] Job {job_id} quality warnings:"
    )

    for w in warnings:
        print(
            f"[orchestrator]   ⚠ {w}"
        )

    print()


# ============================================================
# Subtitle PDF helpers
# ============================================================

def _find_subtitle_pdf_font() -> Optional[str]:
    """
    Find a Unicode font suitable for subtitle PDF generation.

    macOS Homebrew font casks install fonts into ~/Library/Fonts.
    Noto Sans Devanagari is preferred for Hindi/Marathi.
    """

    home = os.path.expanduser("~")

    font_dirs = [
        os.path.join(home, "Library", "Fonts"),
        "/Library/Fonts",
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        "/opt/homebrew/share/fonts",
        "/usr/local/share/fonts",
        "/usr/share/fonts",
        "/usr/share/fonts/truetype",
        "/usr/share/fonts/opentype",
    ]

    preferred_names = [
        # Hindi / Marathi
        "NotoSansDevanagari-Regular.ttf",
        "NotoSansDevanagari[wdth,wght].ttf",

        # Other Indic scripts
        "NotoSansBengali-Regular.ttf",
        "NotoSansTamil-Regular.ttf",
        "NotoSansTelugu-Regular.ttf",
        "NotoSansKannada-Regular.ttf",
        "NotoSansMalayalam-Regular.ttf",
        "NotoSansGujarati-Regular.ttf",
        "NotoSansGurmukhi-Regular.ttf",
        "NotoSansOriya-Regular.ttf",

        # General Unicode fallbacks
        "NotoSans-Regular.ttf",
        "DejaVuSans.ttf",
    ]

    # First: exact known filenames.
    for directory in font_dirs:
        for filename in preferred_names:
            path = os.path.join(directory, filename)

            if os.path.isfile(path):
                log.info(
                    "Subtitle PDF font selected: %s",
                    path,
                )
                return path

    # Second: recursive search for the Devanagari variable font.
    for directory in font_dirs:
        if not os.path.isdir(directory):
            continue

        try:
            for root, _, files in os.walk(directory):
                for filename in files:
                    lower = filename.lower()

                    if (
                        "notosansdevanagari" in lower
                        and lower.endswith((".ttf", ".otf"))
                    ):
                        path = os.path.join(root, filename)

                        log.info(
                            "Subtitle PDF font selected: %s",
                            path,
                        )
                        return path

        except OSError:
            continue

    # NOTE: there used to be a third tier here matching *any*
    # "notosans*.ttf" file recursively. Removed -- verified directly that
    # it's actively dangerous, not just imprecise: on a real machine with
    # no Devanagari-capable font installed, it matched
    # "NotoSansLepcha-Regular.ttf" (a real font, for the Lepcha script
    # used in Sikkim -- nothing to do with Hindi/Marathi) ahead of the
    # DejaVu/Arial Unicode fallback below, and reportlab silently
    # rendered a completely blank page for it -- no error, no missing-
    # glyph boxes, just empty subtitle PDFs shipped with no indication
    # anything was wrong. Tier 1 above already covers the one legitimate
    # generic case ("NotoSans-Regular.ttf", in `preferred_names`) and
    # tier 2 already covers any correctly-named Devanagari variant --
    # this tier only ever added the risk of a wrong-script silent match
    # for no real coverage it didn't already provide.

    # Final fallback.
    for directory in font_dirs:
        for filename in ("DejaVuSans.ttf", "Arial Unicode.ttf"):
            path = os.path.join(directory, filename)

            if os.path.isfile(path):
                log.warning(
                    "Using fallback Unicode font for subtitle PDF: %s",
                    path,
                )
                return path

    return None


def _register_subtitle_pdf_font():
    """
    Register the Unicode font used by subtitle PDFs.

    The font is registered only once per Python process.
    """

    font_name = "SubtitleUnicodeFont"

    if (
        font_name
        in pdfmetrics.getRegisteredFontNames()
    ):
        return font_name

    font_path = (
        _find_subtitle_pdf_font()
    )

    log.info(
        f"Registering subtitle PDF font: {font_path}"
    )

    try:

        pdfmetrics.registerFont(
            TTFont(
                font_name,
                font_path,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "The subtitle PDF font was found but could not be "
            f"loaded by ReportLab: {font_path}. "
            f"Original error: {exc}"
        ) from exc

    return font_name


def _format_pdf_timestamp(
    seconds: float,
) -> str:
    """
    Format seconds as:

        HH:MM:SS.mmm
    """

    try:
        seconds = float(seconds)

    except (
        TypeError,
        ValueError,
    ):
        seconds = 0.0

    seconds = max(
        0.0,
        seconds,
    )

    total_ms = round(
        seconds * 1000
    )

    hours = (
        total_ms
        // 3_600_000
    )

    total_ms %= 3_600_000

    minutes = (
        total_ms
        // 60_000
    )

    total_ms %= 60_000

    secs = (
        total_ms
        // 1000
    )

    millis = (
        total_ms
        % 1000
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{millis:03d}"
    )


def _escape_pdf_text(
    text: str,
) -> str:
    """
    Escape text for ReportLab Paragraph XML/HTML parsing.
    """

    return (
        str(text)
        .replace(
            "&",
            "&amp;",
        )
        .replace(
            "<",
            "&lt;",
        )
        .replace(
            ">",
            "&gt;",
        )
    )


def _generate_subtitles_pdf(
    segments,
    pdf_path: str,
    source_lang: str,
    target_lang: str,
):
    """
    Generate a human-readable PDF containing the translated subtitles.

    The PDF is a document representation of the subtitles.

    It does NOT replace SRT/VTT for video playback.

    Each subtitle entry contains:

        Subtitle number
        Timestamp
        Translated subtitle text

    Translation text is preferred. If translated_text is missing,
    the original ASR text is used as a fallback.
    """

    font_name = (
        _register_subtitle_pdf_font()
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Subtitle Transcript",
        author="Video Translation Pipeline",
    )

    styles = (
        getSampleStyleSheet()
    )

    # ========================================================
    # Title
    # ========================================================

    title_style = styles[
        "Title"
    ].clone(
        "SubtitlePDFTitle"
    )

    title_style.fontName = (
        font_name
    )

    title_style.fontSize = 18

    title_style.leading = 22

    title_style.alignment = (
        TA_LEFT
    )

    title_style.spaceAfter = (
        8 * mm
    )

    # ========================================================
    # Metadata
    # ========================================================

    info_style = styles[
        "Normal"
    ].clone(
        "SubtitlePDFInfo"
    )

    info_style.fontName = (
        font_name
    )

    info_style.fontSize = 9

    info_style.leading = 13

    info_style.spaceAfter = (
        2 * mm
    )

    # ========================================================
    # Timestamp
    # ========================================================

    timestamp_style = styles[
        "Normal"
    ].clone(
        "SubtitlePDFTimestamp"
    )

    timestamp_style.fontName = (
        font_name
    )

    timestamp_style.fontSize = 9

    timestamp_style.leading = 13

    timestamp_style.spaceBefore = (
        4 * mm
    )

    timestamp_style.spaceAfter = (
        1 * mm
    )

    # ========================================================
    # Subtitle text
    # ========================================================

    subtitle_style = styles[
        "Normal"
    ].clone(
        "SubtitlePDFText"
    )

    subtitle_style.fontName = (
        font_name
    )

    subtitle_style.fontSize = 12

    subtitle_style.leading = 17

    subtitle_style.spaceAfter = (
        2 * mm
    )

    # ========================================================
    # Build document
    # ========================================================

    story = []

    story.append(
        Paragraph(
            "Subtitle Transcript",
            title_style,
        )
    )

    story.append(
        Paragraph(
            (
                f"Source language: "
                f"{_escape_pdf_text(source_lang)}"
                f"<br/>"
                f"Target language: "
                f"{_escape_pdf_text(target_lang)}"
                f"<br/>"
                f"Total subtitles: "
                f"{len(segments)}"
            ),
            info_style,
        )
    )

    story.append(
        Spacer(
            1,
            6 * mm,
        )
    )

    # ========================================================
    # Subtitle entries
    # ========================================================

    for number, seg in enumerate(
        segments,
        start=1,
    ):

        start = (
            _format_pdf_timestamp(
                seg.start
            )
        )

        end = (
            _format_pdf_timestamp(
                seg.end
            )
        )

        timestamp = (
            f"{number}. "
            f"{start} --> {end}"
        )

        text = (
            getattr(
                seg,
                "translated_text",
                None,
            )
            or getattr(
                seg,
                "text",
                "",
            )
            or ""
        )

        text = (
            _escape_pdf_text(
                text
            )
        )

        story.append(
            Paragraph(
                _escape_pdf_text(
                    timestamp
                ),
                timestamp_style,
            )
        )

        story.append(
            Paragraph(
                text,
                subtitle_style,
            )
        )

        story.append(
            Spacer(
                1,
                2 * mm,
            )
        )

    # ========================================================
    # Build PDF
    # ========================================================

    doc.build(
        story
    )


# ============================================================
# Main public job function
# ============================================================

def run_job(
    input_path: str,
    source_lang_hint: Optional[str],
    target_lang: str,
    want_burned_in: bool,
    want_voiceover: bool,
    progress_cb: Optional[Callable] = None,
    engine_override: str = "auto",
    asr_engine: str = "indic_conformer",
    tts_speaker: Optional[str] = None,
    cancel_event=None,
) -> dict:
    """
    Runs one end-to-end job.

    On any failure, partial output under
    jobs/<job_id>/ is rolled back rather than left half-written.

    The failure is logged with a full traceback.
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
            cancel_event,
        )

    except JobCancelled:

        log.info(
            f"Job {job_id} cancelled by user -- "
            f"rolling back partial output at {job_dir}"
        )

        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise

    except Exception:

        log.exception(
            f"Job {job_id} failed -- "
            f"rolling back partial output at {job_dir}"
        )

        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise

    log.info(
        f"Job {job_id} finished ok"
    )

    return result


# ============================================================
# Internal job implementation
# ============================================================

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
    asr_engine: str = "indic_conformer",
    tts_speaker: Optional[str] = None,
    cancel_event=None,
) -> dict:

    job_start = (
        time.perf_counter()
    )

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

    # ========================================================
    # Determine input type
    # ========================================================

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

    # ========================================================
    # Stage 1
    # ========================================================

    _progress(
        progress_cb,
        "stage1",
        "Extracting & normalising audio...",
        5,
    )

    with _Stage(
        timings,
        "stage1_preprocess",
    ):

        pre = (
            stage1_preprocess.run_stage1(
                local_input,
                work_dir,
                input_kind,
                cancel_event=cancel_event,
            )
        )

    # ========================================================
    # Stage 2
    # ========================================================

    if pre.subtitle_track_found:

        _progress(
            progress_cb,
            "skip_asr",
            "Embedded subtitles found — skipping ASR.",
            25,
        )

        with _Stage(
            timings,
            "stage2_asr",
        ):

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

        with _Stage(
            timings,
            "stage2_asr",
        ):

            asr_result = (
                stage2_asr.run_stage2(
                    pre.audio_wav_path,
                    language_hint=source_lang_hint,
                    cancel_event=cancel_event,
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
            f"(auto-detected source language: "
            f"{detected_lang or 'unknown'}). "
            "A common cause: whisper.cpp misdetected the source "
            "language, so real speech gets decoded through the "
            "wrong language's model -- this degrades into "
            "repeated-token hallucination over a long clip, which "
            "the hallucination filter then correctly drops, "
            "leaving nothing. Other causes: the audio may be mostly "
            "silent, non-speech (music/noise), or too quiet. "
            "Try again with the source language set explicitly "
            "instead of auto-detect if you know it."
        )

    # ========================================================
    # Optional IndicConformer refinement
    # ========================================================

    indic_conformer_warning = None

    if asr_engine == "indic_conformer":

        _progress(
            progress_cb,
            "asr_refine",
            "Refining transcript with IndicConformer...",
            30,
        )

        try:

            with _Stage(
                timings,
                "asr_refine_indic_conformer",
            ):

                asr_segments = (
                    stage2_asr
                    .refine_segments_with_indic_conformer(
                        asr_segments,
                        pre.audio_wav_path,
                        source_lang,
                        cancel_event=cancel_event,
                    )
                )

        except JobCancelled:

            raise

        except Exception as exc:

            # IndicConformer is a refinement on top of an already-usable
            # whisper.cpp transcript, not a hard requirement -- a failure
            # here (missing optional deps, no HF access to the gated
            # checkpoint yet, an OOM on a memory-constrained machine, a
            # network hiccup on first download, etc.) shouldn't discard a
            # perfectly good whisper transcript and roll back the whole
            # job. Fall back to it and surface why in the job report,
            # the same way a missing voiceover-language mapping degrades
            # to subtitles-only instead of failing outright.
            log.warning(
                f"Job {job_id}: IndicConformer refinement failed, "
                f"falling back to the whisper.cpp transcript -- {exc}"
            )

            indic_conformer_warning = (
                "IndicConformer refinement was requested but failed "
                f"({exc}); used the whisper.cpp transcript instead."
            )

    # ========================================================
    # Stage 3
    # ========================================================

    _progress(
        progress_cb,
        "stage3",
        "Segmenting & checking translation memory...",
        45,
    )

    with _Stage(
        timings,
        "stage3_segment_tm",
    ):

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

    # ========================================================
    # Translation
    # ========================================================

    engine_name = (
        "Translation Memory (cache only)"
    )

    with _Stage(
        timings,
        "translation",
    ):

        if misses:

            total_misses = len(
                misses
            )

            def _translate_progress(
                done: int,
                total: int,
            ):

                frac = (
                    done / total
                    if total
                    else 1.0
                )

                pct = (
                    60
                    + round(
                        frac * 15
                    )
                )

                _progress(
                    progress_cb,
                    "translate",
                    (
                        f"Translating segment "
                        f"{done} of {total} "
                        f"({round(frac * 100)}%)..."
                    ),
                    pct,
                )

            _progress(
                progress_cb,
                "translate",
                (
                    f"Translating segment "
                    f"0 of {total_misses} (0%)..."
                ),
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
                (
                    f"All {len(segments)} "
                    f"segment(s) served from "
                    f"translation memory."
                ),
                60,
            )

    # ========================================================
    # Automatic Translation Quality
    # ========================================================

    _progress(
        progress_cb,
        "translation_qc",
        "Calculating automatic translation quality...",
        68,
    )

    with _Stage(
        timings,
        "translation_qc",
    ):

        translation_quality = (
            delivery.calculate_translation_quality(
                segments
            )
        )

    # ========================================================
    # Stage 4 - subtitles
    # ========================================================

    _progress(
        progress_cb,
        "stage4",
        "Generating SRT/VTT/PDF subtitle files...",
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

    pdf_path = os.path.join(
        out_dir,
        f"{job_id}_subtitles.pdf",
    )

    with _Stage(
        timings,
        "stage4_subtitle",
    ):

        # ----------------------------------------------------
        # SRT
        # ----------------------------------------------------

        stage4_subtitle.generate_srt(
            segments,
            srt_path,
        )

        # ----------------------------------------------------
        # VTT
        # ----------------------------------------------------

        stage4_subtitle.generate_vtt(
            segments,
            vtt_path,
        )

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        _generate_subtitles_pdf(
            segments,
            pdf_path,
            source_lang,
            target_lang,
        )

    # ========================================================
    # Quality information
    # ========================================================

    quality_info = {

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

    # ========================================================
    # IndicConformer refinement fallback
    # ========================================================

    if indic_conformer_warning:

        quality_info[
            "warnings"
        ].append(
            indic_conformer_warning
        )

    # ========================================================
    # ASR dropped ranges
    # ========================================================

    if asr_dropped_ranges:

        gap_desc = ", ".join(
            f"{s:.0f}s-{e:.0f}s"
            for s, e in asr_dropped_ranges
        )

        total_dropped_sec = sum(
            e - s
            for s, e in asr_dropped_ranges
        )

        quality_info[
            "warnings"
        ].append(
            (
                f"{total_dropped_sec:.0f}s of audio "
                f"({gap_desc}) could not be transcribed "
                "reliably (whisper.cpp produced repeated-token "
                "hallucination there, which was dropped) and has "
                "no subtitles, translation, or voiceover as a "
                "result. Try re-running with an explicit source "
                "language, or a different WHISPER_BEAM_SIZE/"
                "WHISPER_BEST_OF, if this stretch has real speech."
            )
        )

    # ========================================================
    # Voiceover
    # ========================================================

    voiceover_source_for_burn = None

    voiceover_quality = {}

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

            quality_info[
                "tts_coverage_percent"
            ] = 0.0

            quality_info[
                "overall_quality_score"
            ] = 0.0

            outputs_voiceover = None

        else:

            _progress(
                progress_cb,
                "voiceover",
                "Synthesising voiceover (Indic-TTS)...",
                92,
            )

            voiceover_wav = None

            try:

                with _Stage(
                    timings,
                    "voiceover_tts",
                ):

                    (
                        voiceover_wav,
                        voiceover_quality,
                    ) = delivery.build_voiceover_track(
                        segments,
                        target_lang,
                        work_dir,
                        speaker=tts_speaker,
                        cancel_event=cancel_event,
                    )

            except delivery.NoVoiceoverContentError as exc:

                message = (
                    f"Voiceover could not be generated "
                    f"for any segment ({exc}) -- "
                    "skipping voiceover. Subtitles are "
                    "still produced from whatever speech "
                    "was transcribed."
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

                warnings_before_voiceover = (
                    quality_info[
                        "warnings"
                    ]
                )

                quality_info.update(
                    voiceover_quality
                )

                quality_info[
                    "warnings"
                ] = (
                    warnings_before_voiceover
                    + voiceover_quality.get(
                        "warnings",
                        [],
                    )
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

                with _Stage(
                    timings,
                    "voiceover_mux",
                ):

                    delivery.mux_voiceover_onto_video(
                        local_input,
                        voiceover_wav,
                        voiceover_path,
                        cancel_event=cancel_event,
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

    # ========================================================
    # Burned-in output
    # ========================================================

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

        burn_in_srt_path = (
            srt_path
        )

        if voiceover_source_for_burn:

            burn_in_srt_path = (
                _write_voiceover_synced_srt(
                    segments,
                    voiceover_quality.get(
                        "segment_details",
                        [],
                    ),
                    work_dir,
                    job_id,
                )
            )

        with _Stage(
            timings,
            "burn_in_subtitles",
        ):

            delivery.burn_in_subtitles(
                source_video,
                burn_in_srt_path,
                burned_path,
                cancel_event=cancel_event,
            )

    # ========================================================
    # Outputs
    # ========================================================

    outputs = {

        "job_id":
            job_id,

        # ----------------------------------------------------
        # Subtitle files
        # ----------------------------------------------------

        "srt":
            srt_path,

        "vtt":
            vtt_path,

        "subtitles_pdf":
            pdf_path,

        # ----------------------------------------------------
        # Reports
        # ----------------------------------------------------

        "job_report_docx":
            None,

        # ----------------------------------------------------
        # Video outputs
        # ----------------------------------------------------

        "burned_in_mp4":
            burned_path,

        "voiceover_mp4":
            outputs_voiceover,

        # ----------------------------------------------------
        # Quality
        # ----------------------------------------------------

        "quality":
            quality_info,
    }

    # ========================================================
    # Stage 5
    # ========================================================

    _progress(
        progress_cb,
        "stage5",
        "Archiving segments for future reuse...",
        85,
    )

    with _Stage(
        timings,
        "stage5_archive",
    ):

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

    # ========================================================
    # Job report
    # ========================================================

    docx_path = os.path.join(
        out_dir,
        f"{job_id}_job_report.docx",
    )

    with _Stage(
        timings,
        "job_report_docx",
    ):

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

    # ========================================================
    # Finish
    # ========================================================

    _progress(
        progress_cb,
        "done",
        "Done — ready for USB transfer / offline playback.",
        100,
    )

    total_elapsed = (
        time.perf_counter()
        - job_start
    )

    _print_timing_summary(
        job_id,
        timings,
        total_elapsed,
    )

    _print_quality_warnings(
        job_id,
        quality_info[
            "warnings"
        ],
    )

    outputs[
        "stats"
    ] = {

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
            "total":
                total_elapsed,
        },
    }

    return outputs


# ============================================================
# Voiceover-synchronised subtitle SRT
# ============================================================

def _write_voiceover_synced_srt(
    segments,
    segment_details,
    work_dir,
    job_id,
):
    """
    Write a temporary SRT whose timestamps match where the dubbed
    voiceover audio actually landed.

    The main SRT/VTT deliverables are untouched.

    Segments without a matching placed_start/placed_end keep their
    original ASR timing.
    """

    placement_by_index = {

        d["segment"]:
            d

        for d in segment_details

        if "placed_start" in d
        and "placed_end" in d
    }

    resynced = []

    for seg in segments:

        placement = (
            placement_by_index.get(
                seg.index
            )
        )

        if (
            placement
            and placement[
                "placed_end"
            ]
            > placement[
                "placed_start"
            ]
        ):

            resynced.append(
                replace(
                    seg,
                    start=placement[
                        "placed_start"
                    ],
                    end=placement[
                        "placed_end"
                    ],
                )
            )

        else:

            resynced.append(
                seg
            )

    out_path = os.path.join(
        work_dir,
        f"{job_id}_voiceover_synced.srt",
    )

    stage4_subtitle.generate_srt(
        resynced,
        out_path,
    )

    return out_path


# ============================================================
# Language normalisation
# ============================================================

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


# ============================================================
# Embedded SRT → TranscriptSegments
# ============================================================

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
