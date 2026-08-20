"""
Quality metrics for the offline field translation pipeline.

These metrics are intentionally lightweight and work offline.
"""

import re
from typing import List, Dict, Any


NUMBER_RE = re.compile(
    r"(?<![\w])"
    r"(?:\d+(?:[.,]\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?)"
    r"(?![\w])"
)


def extract_numbers(text: str) -> List[str]:
    """Extract numeric expressions from text."""
    if not text:
        return []

    return NUMBER_RE.findall(text)


def normalize_number_for_comparison(value: str) -> str:
    """
    Normalize formatting differences for numeric comparison.

    Examples:
        25.50 -> 25.5
        1,000 -> 1000
    """
    value = value.replace(",", "").strip()

    try:
        return str(float(value)).rstrip("0").rstrip(".")
    except ValueError:
        return value


def numeric_preservation_score(
    source_text: str,
    translated_text: str,
) -> Dict[str, Any]:
    """
    Measure whether numbers in the source are represented in the
    translated text.

    This is a text-level metric, not an audio pronunciation metric.
    """

    source_numbers = extract_numbers(source_text)
    translated_numbers = extract_numbers(translated_text)

    source_norm = [
        normalize_number_for_comparison(x)
        for x in source_numbers
    ]

    translated_norm = [
        normalize_number_for_comparison(x)
        for x in translated_numbers
    ]

    matched = 0
    used = set()

    for number in source_norm:
        for i, candidate in enumerate(translated_norm):
            if i in used:
                continue

            if number == candidate:
                matched += 1
                used.add(i)
                break

    total = len(source_norm)

    score = (
        1.0
        if total == 0
        else matched / total
    )

    return {
        "source_numbers": source_numbers,
        "translated_numbers": translated_numbers,
        "matched_numbers": matched,
        "total_source_numbers": total,
        "score": round(score, 4),
    }


def segment_timing_score(segments) -> Dict[str, Any]:
    """
    Measure how well translated segments fit their original time slots.
    """

    if not segments:
        return {
            "segments": 0,
            "average_slot_duration": 0,
            "score": 0,
        }

    valid = 0
    total_ratio = 0.0

    for seg in segments:
        duration = max(0.0, seg.end - seg.start)

        if duration > 0:
            valid += 1
            total_ratio += min(1.0, duration / max(duration, 0.001))

    score = valid / len(segments)

    return {
        "segments": len(segments),
        "valid_timing_segments": valid,
        "score": round(score, 4),
    }


def translation_coverage_score(segments) -> Dict[str, Any]:
    """Measure how many source segments received translated text."""

    if not segments:
        return {
            "total_segments": 0,
            "translated_segments": 0,
            "score": 0,
        }

    translated = sum(
        1
        for seg in segments
        if getattr(seg, "translated_text", None)
        and seg.translated_text.strip()
    )

    return {
        "total_segments": len(segments),
        "translated_segments": translated,
        "score": round(translated / len(segments), 4),
    }


def calculate_quality_score(
    translation_score: float,
    numeric_score: float,
    timing_score: float,
) -> float:
    """
    Lightweight overall score.

    Translation gets the highest weight because it is the main
    transformation performed by the system.
    """

    score = (
        translation_score * 0.50
        + numeric_score * 0.30
        + timing_score * 0.20
    )

    return round(score * 100, 2)