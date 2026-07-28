"""
Stage 4 — Subtitle Generation (Python)
  - Translated text + timestamps -> SRT / VTT
  - 42 char/line enforced
  - 2-line max
  - Gap enforcement between subtitles
"""
import textwrap
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
    """Ensure at least MIN_GAP_BETWEEN_SUBTITLES_SEC between consecutive cues."""
    for i in range(1, len(segments)):
        prev, cur = segments[i - 1], segments[i]
        if cur.start < prev.end + MIN_GAP_BETWEEN_SUBTITLES_SEC:
            cur_start = prev.end + MIN_GAP_BETWEEN_SUBTITLES_SEC
            if cur_start < cur.end:
                cur.start = cur_start
            # if it would invert start>end, leave as-is (extremely tight ASR timing edge case)
    return segments


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


def generate_srt(segments: List[Segment], out_path: str) -> str:
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
