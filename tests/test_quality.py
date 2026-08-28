"""
Test evidence for pipeline/quality.py -- the numeric-preservation scoring
used in the Job Report DOCX (report.py) to flag translations that dropped
or mangled numbers/decimals from the source.
"""
from pipeline.quality import (
    extract_numbers,
    normalize_number_for_comparison,
    numeric_preservation_score,
    translation_coverage_score,
)


def test_extract_numbers_plain_and_decimal():
    assert extract_numbers("There were 25 cows and 3.5 acres.") == ["25", "3.5"]


def test_extract_numbers_thousands_separator():
    assert extract_numbers("Budget of 1,000 rupees") == ["1,000"]


def test_extract_numbers_none_present():
    assert extract_numbers("No numbers here at all.") == []


def test_extract_numbers_empty_string():
    assert extract_numbers("") == []


def test_normalize_strips_thousands_separator():
    assert normalize_number_for_comparison("1,000") == "1000"


def test_normalize_equates_trailing_zero_decimals():
    # 25.50 and 25.5 should compare equal after normalization.
    assert normalize_number_for_comparison("25.50") == normalize_number_for_comparison("25.5")


def test_normalize_non_numeric_passthrough():
    # Defensive: a malformed "number" shouldn't crash the scorer.
    assert normalize_number_for_comparison("12-34") == "12-34"


def test_numeric_preservation_full_match():
    result = numeric_preservation_score(
        "There were 25 cows and 3.5 acres.",
        "गाय 25 और 3.5 एकड़ थे।",
    )
    assert result["score"] == 1.0
    assert result["matched_numbers"] == 2
    assert result["total_source_numbers"] == 2


def test_numeric_preservation_dropped_number():
    # Edge case this metric exists to catch: a number silently dropped
    # during translation.
    result = numeric_preservation_score(
        "Pay 1,000 rupees for 5 acres.",
        "एकड़ के लिए भुगतान करें।",  # both numbers missing
    )
    assert result["matched_numbers"] == 0
    assert result["total_source_numbers"] == 2
    assert result["score"] == 0.0


def test_numeric_preservation_no_numbers_scores_perfect():
    # No numbers in source -> nothing to preserve -> should not be
    # penalized (score defined as 1.0, not divide-by-zero).
    result = numeric_preservation_score("no numbers here", "यहां कोई संख्या नहीं")
    assert result["score"] == 1.0
    assert result["total_source_numbers"] == 0


def test_translation_coverage_score_partial():
    class FakeSeg:
        def __init__(self, translated_text):
            self.translated_text = translated_text

    segments = [FakeSeg("hi"), FakeSeg(""), FakeSeg(None), FakeSeg("ok")]
    result = translation_coverage_score(segments)
    assert result["total_segments"] == 4
    assert result["translated_segments"] == 2
    assert result["score"] == 0.5


def test_translation_coverage_score_empty_list():
    result = translation_coverage_score([])
    assert result["score"] == 0
