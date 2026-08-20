"""
Stage 3 — Segmentation + Translation Memory Check (SQLite)
  - Chunk segments
  - Hash each
  - Cache hit -> reuse stored translation
  - New segments only -> pass to translation engine

This module owns the `translation_memory` table that both Stage 3 (read)
and Stage 5 (write / archive) use, so the schema lives here.
"""
import hashlib
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from config import TM_DB_PATH, SEGMENT_MAX_CHARS


SCHEMA = """
CREATE TABLE IF NOT EXISTS translation_memory (
    segment_hash TEXT NOT NULL,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    glossary_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    source_text  TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (segment_hash, source_lang, target_lang, glossary_version, model_version)
);

CREATE TABLE IF NOT EXISTS job_archive (
    job_id TEXT PRIMARY KEY,
    source_lang TEXT,
    target_lang TEXT,
    glossary_version TEXT,
    model_version TEXT,
    segment_count INTEGER,
    cache_hit_count INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection():
    conn = sqlite3.connect(TM_DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def hash_segment(text: str, source_lang: str, target_lang: str) -> str:
    """Stable hash used as the TM cache key for a single segment's text."""
    normalized = " ".join(text.strip().lower().split())
    key = f"{source_lang}|{target_lang}|{normalized}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str
    confidence: float
    segment_hash: str
    translated_text: Optional[str] = None
    from_cache: bool = False


def chunk_segments(asr_segments, source_lang: str, target_lang: str) -> List[Segment]:
    """
    Convert raw ASR segments into TM-hashed Segment objects, splitting any
    segment longer than SEGMENT_MAX_CHARS at the nearest sentence boundary
    so downstream subtitle line-length rules (Stage 4) have clean input.
    """
    chunks: List[Segment] = []
    idx = 0
    for seg in asr_segments:
        text = seg.text.strip()
        if len(text) <= SEGMENT_MAX_CHARS:
            pieces = [(seg.start, seg.end, text)]
        else:
            pieces = _split_long_text(text, seg.start, seg.end)

        for start, end, piece_text in pieces:
            if not piece_text.strip():
                continue
            h = hash_segment(piece_text, source_lang, target_lang)
            chunks.append(Segment(
                index=idx, start=start, end=end, text=piece_text,
                confidence=seg.confidence, segment_hash=h
            ))
            idx += 1
    return chunks


def _hard_wrap(text: str, max_chars: int) -> List[str]:
    """Split text into <=max_chars pieces at word boundaries, falling back
    to a character-level cut for any single "word" that's still over the
    cap on its own -- e.g. a run of text with no spaces at all (a known
    Whisper hallucination failure mode: repeated/garbled tokens transcribed
    from silence or noise with no normal word breaks). Without the
    character-level fallback, splitting on " " alone treats such a run as
    one unsplittable word and lets it through at full length -- confirmed
    to be exactly what let a several-thousand-char chunk reach TTS synthesis
    and blow past FastPitch's fixed positional-encoding limit."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    pieces = []
    current = ""
    for w in words:
        while len(w) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(w[:max_chars])
            w = w[max_chars:]
        candidate = f"{current} {w}".strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = w
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _split_long_text(text, start, end):
    """Proportionally split a long segment's time span across chunks, each
    capped at SEGMENT_MAX_CHARS. Splits at sentence boundaries first; any
    resulting piece still over the cap is hard-wrapped at word boundaries so
    every returned piece is guaranteed <= SEGMENT_MAX_CHARS."""
    import re
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    sentences = [s for s in sentences if s.strip()] or [text]

    pieces_text = []
    for s in sentences:
        pieces_text.extend(_hard_wrap(s, SEGMENT_MAX_CHARS))

    total_len = sum(len(p) for p in pieces_text) or 1
    duration = max(end - start, 0.01)

    pieces = []
    cursor = start
    for p in pieces_text:
        frac = len(p) / total_len
        seg_dur = duration * frac
        pieces.append((cursor, cursor + seg_dur, p.strip()))
        cursor += seg_dur
    return pieces


def apply_translation_memory(segments: List[Segment], source_lang: str,
                              target_lang: str, glossary_version: str,
                              model_version: str):
    """
    Look up each segment's hash in the TM cache.
    Mutates segments in place: sets .translated_text + .from_cache for hits.
    Returns list of segments that still need translation (cache misses).
    """
    conn = get_connection()
    cur = conn.cursor()
    misses = []
    for seg in segments:
        cur.execute(
            """SELECT translated_text FROM translation_memory
               WHERE segment_hash=? AND source_lang=? AND target_lang=?
                     AND glossary_version=? AND model_version=?""",
            (seg.segment_hash, source_lang, target_lang, glossary_version, model_version)
        )
        row = cur.fetchone()
        if row:
            seg.translated_text = row[0]
            seg.from_cache = True
        else:
            misses.append(seg)
    conn.close()
    return misses


def store_translations(segments: List[Segment], source_lang: str, target_lang: str,
                        glossary_version: str, model_version: str):
    """Persist newly-translated segments back into the TM cache (used by Stage 5 too)."""
    conn = get_connection()
    cur = conn.cursor()
    for seg in segments:
        if seg.translated_text is None:
            continue
        cur.execute(
            """INSERT OR REPLACE INTO translation_memory
               (segment_hash, source_lang, target_lang, glossary_version, model_version,
                source_text, translated_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (seg.segment_hash, source_lang, target_lang, glossary_version, model_version,
             seg.text, seg.translated_text)
        )
    conn.commit()
    conn.close()
