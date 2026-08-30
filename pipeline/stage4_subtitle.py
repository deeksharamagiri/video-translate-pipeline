"""
Stage 4 — Subtitle Generation (Python)
  - Translated text + timestamps -> SRT / VTT
  - 42 char/line enforced
  - 2-line max
  - Gap enforcement between subtitles
"""
import textwrap
from dataclasses import replace
from typing import List

from config import MAX_CHARS_PER_LINE, MAX_LINES_PER_SUBTITLE, MIN_GAP_BETWEEN_SUBTITLES_SEC
from pipeline.stage3_segment_tm import Segment


def _wrap_text(text: str) -> List[str]:
    """Wrap translated text to MAX_CHARS_PER_LINE, capped at MAX_LINES_PER_SUBTITLE lines."""
    wrapped = textwrap.wrap(text, width=MAX_CHARS_PER_LINE, break_long_words=False)
    if not wrapped:
        return [""]
    if len(wrapped) <= MAX_LINES_PER_SUBTITLE:
        return wrapped
    # Too long for max lines: keep first (N-1) lines, cram + ellipsis-truncate the rest
    kept = wrapped[: MAX_LINES_PER_SUBTITLE - 1]
    remainder = " ".join(wrapped[MAX_LINES_PER_SUBTITLE - 1:])
    if len(remainder) > MAX_CHARS_PER_LINE:
        remainder = remainder[: MAX_CHARS_PER_LINE - 1].rstrip() + "…"
    kept.append(remainder)
    return kept


def _enforce_gaps(segments: List[Segment]) -> List[Segment]:
    """
    Ensure at least MIN_GAP_BETWEEN_SUBTITLES_SEC between consecutive cues.

    Returns a new list with `replace()`-built copies for any segment whose
    start gets pushed forward, rather than mutating the input segments in
    place. `segments` here is the same shared list orchestrator.py later
    passes to build_voiceover_track() and the job report/archive stages --
    in-place mutation here was previously leaking subtitle-only gap
    adjustments into the ASR timestamps voiceover placement is computed
    from, before those stages even ran, silently shifting dubbed-audio
    placement away from the original ASR timing it was supposed to track.
    """
    out = list(segments)
    for i in range(1, len(out)):
        prev, cur = out[i - 1], out[i]
        if cur.start < prev.end + MIN_GAP_BETWEEN_SUBTITLES_SEC:
            cur_start = prev.end + MIN_GAP_BETWEEN_SUBTITLES_SEC
            if cur_start < cur.end:
                out[i] = replace(cur, start=cur_start)
            # if it would invert start>end, leave as-is (extremely tight ASR timing edge case)
    return out


def _format_timestamp_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _drop_zero_duration(segments: List[Segment]) -> List[Segment]:
    """
    Drop segments with end <= start before writing them out.

    These are never visible to a real subtitle renderer (zero or negative
    display time) -- verified directly against real job output, e.g. an
    entry with start == end == 72.000s written as
    "00:01:12,000 --> 00:01:12,000". Filtered here rather than upstream so
    the shared `segments` list (also used by the job report and
    translation-memory archiving) is left untouched -- only what actually
    gets written to the subtitle file changes.
    """
    return [s for s in segments if s.end > s.start]


def _merge_duplicate_consecutive(segments: List[Segment]) -> List[Segment]:
    """
    Merge adjacent segments whose translated text is byte-identical.

    Verified directly against real job output: two consecutive SRT
    entries with the exact same text, ~4s apart -- almost certainly a
    segmentation artifact (one utterance split across a chunk boundary)
    rather than the speaker deliberately repeating the line twice in a
    row. Merging (not dropping) preserves full time coverage: the kept
    entry spans from the first segment's start to the last's end.
    """
    if not segments:
        return segments

    # Builds a new list of (possibly replaced) Segment objects rather than
    # mutating in place -- `segments` here is the same shared list used
    # elsewhere (job report, translation-memory archiving), and this
    # function must not alter it, only what generate_srt/generate_vtt go
    # on to write out.
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        prev_text = (prev.translated_text if prev.translated_text is not None else prev.text).strip()
        seg_text = (seg.translated_text if seg.translated_text is not None else seg.text).strip()
        if seg_text and seg_text == prev_text:
            merged[-1] = replace(prev, end=seg.end)
        else:
            merged.append(seg)
    return merged


def generate_srt(segments: List[Segment], out_path: str) -> str:
    segments = _drop_zero_duration(segments)
    segments = _merge_duplicate_consecutive(segments)
    segments = _enforce_gaps(segments)
    lines = []
    for i, seg in enumerate(segments, start=1):
        text = seg.translated_text if seg.translated_text is not None else seg.text
        wrapped_lines = _wrap_text(text)
        lines.append(str(i))
        lines.append(f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}")
        lines.extend(wrapped_lines)
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def generate_vtt(segments: List[Segment], out_path: str) -> str:
    segments = _drop_zero_duration(segments)
    segments = _merge_duplicate_consecutive(segments)
    segments = _enforce_gaps(segments)
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = seg.translated_text if seg.translated_text is not None else seg.text
        wrapped_lines = _wrap_text(text)
        lines.append(f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}")
        lines.extend(wrapped_lines)
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
