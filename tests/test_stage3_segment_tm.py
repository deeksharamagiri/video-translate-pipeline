"""
Test evidence for pipeline/stage3_segment_tm.py -- segment chunking and
the translation-memory cache hit/miss path (README's "Cache reuse" manual
test case, automated). Uses a throwaway sqlite DB so it never touches the
real data/translation_memory.db.
"""
import os
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class FakeAsrSegment:
    start: float
    end: float
    text: str
    confidence: float = 0.9


@pytest.fixture
def stage3(tmp_path, monkeypatch):
    """Import stage3_segment_tm with TM_DB_PATH pointed at a scratch file."""
    import config
    monkeypatch.setattr(config, "TM_DB_PATH", str(tmp_path / "tm_test.db"))

    import importlib
    from pipeline import stage3_segment_tm as mod
    importlib.reload(mod)
    monkeypatch.setattr(mod, "TM_DB_PATH", str(tmp_path / "tm_test.db"))
    return mod


def test_chunk_segments_keeps_short_text_whole(stage3):
    asr_segments = [FakeAsrSegment(0.0, 2.0, "Short sentence.")]
    chunks = stage3.chunk_segments(asr_segments, "hin", "eng")
    assert len(chunks) == 1
    assert chunks[0].text == "Short sentence."


def test_chunk_segments_splits_long_text(stage3):
    long_text = "This is a very long sentence. " * 20  # well over SEGMENT_MAX_CHARS
    asr_segments = [FakeAsrSegment(0.0, 10.0, long_text)]
    chunks = stage3.chunk_segments(asr_segments, "hin", "eng")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= stage3.SEGMENT_MAX_CHARS
    # Time span should be proportionally distributed, not overlapping/reversed.
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start + 1e-6


def test_hash_segment_is_deterministic_and_case_insensitive(stage3):
    h1 = stage3.hash_segment("Hello World", "eng", "hin")
    h2 = stage3.hash_segment("  hello   world  ", "eng", "hin")
    assert h1 == h2  # normalized (case/whitespace) before hashing


def test_hash_segment_differs_by_language_pair(stage3):
    h1 = stage3.hash_segment("Hello", "eng", "hin")
    h2 = stage3.hash_segment("Hello", "eng", "mar")
    assert h1 != h2


def test_translation_memory_cache_hit_after_store(stage3):
    asr_segments = [FakeAsrSegment(0.0, 2.0, "Cache me please.")]
    segments = stage3.chunk_segments(asr_segments, "eng", "hin")

    # First pass: nothing cached yet -> miss.
    misses = stage3.apply_translation_memory(
        segments, "eng", "hin", "gloss-v1", "model-v1"
    )
    assert len(misses) == 1
    assert misses[0].from_cache is False

    misses[0].translated_text = "मुझे कैश करें।"
    stage3.store_translations(segments, "eng", "hin", "gloss-v1", "model-v1")

    # Second pass with a fresh Segment (same hash) -> hit.
    asr_segments_2 = [FakeAsrSegment(0.0, 2.0, "Cache me please.")]
    segments_2 = stage3.chunk_segments(asr_segments_2, "eng", "hin")
    misses_2 = stage3.apply_translation_memory(
        segments_2, "eng", "hin", "gloss-v1", "model-v1"
    )
    assert misses_2 == []
    assert segments_2[0].from_cache is True
    assert segments_2[0].translated_text == "मुझे कैश करें।"


def test_translation_memory_miss_when_glossary_version_changes(stage3):
    # A cached translation must NOT be reused once glossary_version_tag()
    # changes -- otherwise editing data/glossary.json wouldn't actually
    # take effect on already-seen text.
    asr_segments = [FakeAsrSegment(0.0, 2.0, "Version sensitive text.")]
    segments = stage3.chunk_segments(asr_segments, "eng", "hin")
    stage3.apply_translation_memory(segments, "eng", "hin", "gloss-v1", "model-v1")
    segments[0].translated_text = "translated"
    stage3.store_translations(segments, "eng", "hin", "gloss-v1", "model-v1")

    asr_segments_2 = [FakeAsrSegment(0.0, 2.0, "Version sensitive text.")]
    segments_2 = stage3.chunk_segments(asr_segments_2, "eng", "hin")
    misses = stage3.apply_translation_memory(
        segments_2, "eng", "hin", "gloss-v2", "model-v1"
    )
    assert len(misses) == 1  # miss, because glossary_version differs
