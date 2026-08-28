"""
Test evidence for the small pure-logic helpers in pipeline/orchestrator.py
that don't require running the full pipeline: language-code normalization
and SRT parsing (the "video with embedded subtitles -> skip ASR" path from
the README's manual testing checklist).
"""
from pipeline.orchestrator import _normalize_lang, _segments_from_srt


def test_normalize_lang_maps_two_letter_to_three_letter():
    assert _normalize_lang("hi") == "hin"
    assert _normalize_lang("en") == "eng"
    assert _normalize_lang("mr") == "mar"


def test_normalize_lang_passthrough_for_unknown_or_already_three_letter():
    assert _normalize_lang("hin") == "hin"
    assert _normalize_lang("xyz") == "xyz"


def test_segments_from_srt_parses_basic_block(tmp_path):
    srt_content = (
        "1\n"
        "00:00:01,000 --> 00:00:03,500\n"
        "Hello there.\n"
        "\n"
        "2\n"
        "00:00:04,000 --> 00:00:06,000\n"
        "Second line one\n"
        "Second line two\n"
    )
    srt_path = tmp_path / "sample.srt"
    srt_path.write_text(srt_content, encoding="utf-8")

    segments, detected_lang = _segments_from_srt(str(srt_path))

    assert detected_lang is None
    assert len(segments) == 2
    assert segments[0].start == 1.0
    assert segments[0].end == 3.5
    assert segments[0].text == "Hello there."
    # Multi-line subtitle text should be joined.
    assert segments[1].text == "Second line one Second line two"


def test_segments_from_srt_skips_malformed_blocks(tmp_path):
    # A block missing a timecode line entirely should be skipped, not crash
    # the whole parse (embedded-subtitle files from the field are not
    # guaranteed to be perfectly well-formed).
    srt_content = (
        "1\n"
        "not a timecode\n"
        "Orphaned text\n"
        "\n"
        "2\n"
        "00:00:04,000 --> 00:00:06,000\n"
        "Valid segment\n"
    )
    srt_path = tmp_path / "malformed.srt"
    srt_path.write_text(srt_content, encoding="utf-8")

    segments, _ = _segments_from_srt(str(srt_path))
    assert len(segments) == 1
    assert segments[0].text == "Valid segment"
