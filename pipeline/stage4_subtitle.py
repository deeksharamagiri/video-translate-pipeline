"""
Stage 4 — Subtitle Generation (Python)
  - Translated text + timestamps -> SRT / VTT
  - 42 char/line, 2-line max per cue -- longer text becomes multiple
    contiguous cues (proportional to each cue's share of the text)
    rather than being truncated
  - Gap enforcement between subtitles
"""
import textwrap
from dataclasses import replace
from typing import List

from config import MAX_CHARS_PER_LINE, MAX_LINES_PER_SUBTITLE, MIN_GAP_BETWEEN_SUBTITLES_SEC
from pipeline.stage3_segment_tm import Segment


def _wrap_text(text: str) -> List[List[str]]:
    """
    Wrap translated text to MAX_CHARS_PER_LINE, then group those lines
    into successive groups of up to MAX_LINES_PER_SUBTITLE lines each --
    one group per subtitle cue.

    Previously, text longer than MAX_LINES_PER_SUBTITLE lines got the
    excess cram+ellipsis-truncated into the last line instead of shown --
    on real job output, translated segments routinely ran 150-300+
    characters (whole merged sentences), so the vast majority of a long
    segment's translation was silently dropped from the subtitle despite
    the full text being spoken in the voiceover audio -- reported as
    subtitles "ending with three dots" instead of showing the full
    translation. Returning multiple line-groups here (one per would-be
    cue) instead of truncating lets the caller (generate_srt/generate_vtt)
    emit as many time-sliced cues as needed to show all of it -- see
    _timed_cues_for_segment.
    """
    wrapped = textwrap.wrap(text, width=MAX_CHARS_PER_LINE, break_long_words=False)
    if not wrapped:
        return [[""]]
    return [
        wrapped[i : i + MAX_LINES_PER_SUBTITLE]
        for i in range(0, len(wrapped), MAX_LINES_PER_SUBTITLE)
    ]


def _timed_cues_for_segment(seg) -> List[tuple]:
    """
    Split a segment's (possibly multi-cue) text across its own
    [start, end] window, proportioned by each cue's character count --
    a segment whose translation needed 3 cues gets its display time
    divided across them roughly in proportion to how much is read aloud
    in each, rather than one cue sitting on screen for the segment's
    entire duration while the dubbed voice moves on to content that
    cue's text doesn't include at all (previously reported as
    subtitles "should be exactly in sync with audio").

    Returns a list of (start, end, lines) tuples -- one entry unless the
    text needed multiple cues, in which case entries are contiguous and
    together span exactly [seg.start, seg.end].
    """
    text = seg.translated_text if seg.translated_text is not None else seg.text
    chunks = _wrap_text(text)

    if len(chunks) <= 1:
        return [(seg.start, seg.end, chunks[0] if chunks else [""])]

    weights = [sum(len(line) for line in chunk) or 1 for chunk in chunks]
    total_weight = sum(weights)
    total_duration = max(seg.end - seg.start, 0.0)

    cues = []
    cursor = seg.start

    for i, (chunk, weight) in enumerate(zip(chunks, weights)):
        if i == len(chunks) - 1:
            end = seg.end
        else:
            end = cursor + total_duration * (weight / total_weight)

        cues.append((cursor, end, chunk))
        cursor = end

    return cues


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



# Maximum gap, in seconds, between two identical-text segments for them to
# still be treated as a split-utterance artifact worth merging. Verified
# directly against real job output that this needs a bound: after adding
# finer-grained ASR segmentation (splitting on internal pauses -- see
# stage2_asr.py), a short recurring phrase spoken twice at genuinely
# different points in a video started landing as consecutive list entries
# (any intervening segment had been dropped/filtered), and merging them
# unconditionally produced a single caption spanning 72 real seconds for
# a phrase that took a few seconds to say either time it was spoken.
MERGE_DUPLICATE_MAX_GAP_SEC = 8.0


def _merge_duplicate_consecutive(segments: List[Segment]) -> List[Segment]:
    """
    Merge near-adjacent segments whose translated text is byte-identical.

    Verified directly against real job output: two consecutive SRT
    entries with the exact same text, ~4s apart -- almost certainly a
    segmentation artifact (one utterance split across a chunk boundary)
    rather than the speaker deliberately repeating the line twice in a
    row. Merging (not dropping) preserves full time coverage: the kept
    entry spans from the first segment's start to the last's end. Only
    merges when the gap between them is small (see
    MERGE_DUPLICATE_MAX_GAP_SEC) -- the same text recurring far apart in
    time is the speaker genuinely repeating themselves later on, not a
    segmentation artifact, and must stay as separate captions.
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
        gap = seg.start - prev.end
        if seg_text and seg_text == prev_text and gap <= MERGE_DUPLICATE_MAX_GAP_SEC:
            merged[-1] = replace(prev, end=seg.end)
        else:
            merged.append(seg)
    return merged


def generate_srt(segments: List[Segment], out_path: str) -> str:
    segments = _drop_zero_duration(segments)
    segments = _merge_duplicate_consecutive(segments)
    segments = _enforce_gaps(segments)
    lines = []
    cue_number = 1
    for seg in segments:
        for start, end, cue_lines in _timed_cues_for_segment(seg):
            lines.append(str(cue_number))
            lines.append(f"{_format_timestamp_srt(start)} --> {_format_timestamp_srt(end)}")
            lines.extend(cue_lines)
            lines.append("")
            cue_number += 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def generate_vtt(segments: List[Segment], out_path: str) -> str:
    segments = _drop_zero_duration(segments)
    segments = _merge_duplicate_consecutive(segments)
    segments = _enforce_gaps(segments)
    lines = ["WEBVTT", ""]
    for seg in segments:
        for start, end, cue_lines in _timed_cues_for_segment(seg):
            lines.append(f"{_format_timestamp_vtt(start)} --> {_format_timestamp_vtt(end)}")
            lines.extend(cue_lines)
            lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
