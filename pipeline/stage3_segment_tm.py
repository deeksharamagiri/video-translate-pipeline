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
import re
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from config import TM_DB_PATH, SEGMENT_MAX_CHARS, TRANSLATE_WITH_CONTEXT

# Sentence-final punctuation a fragment must end with to close a group in
# group_sentences() below (English . ! ? plus Devanagari danda ।, optionally
# followed by a closing quote).
_SENTENCE_END_RE = re.compile(r'[.!?।][\'"”]?\s*$')
MAX_GROUP_MEMBERS = 8   # safety cap if audio has a long unpunctuated run
MAX_GROUP_CHARS = 400


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


def _split_long_text(text, start, end):
    """Proportionally split a long segment's time span across sentence chunks."""
    import re
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    sentences = [s for s in sentences if s.strip()] or [text]
    total_len = sum(len(s) for s in sentences)
    duration = max(end - start, 0.01)

    pieces = []
    cursor = start
    for s in sentences:
        frac = len(s) / total_len if total_len else 1 / len(sentences)
        seg_dur = duration * frac
        pieces.append((cursor, cursor + seg_dur, s.strip()))
        cursor += seg_dur
    return pieces


@dataclass
class SentenceGroup:
    """
    One or more consecutive Segments merged up to a sentence boundary. This
    is the actual translation unit — Stage 4 still renders one subtitle cue
    per member Segment (timing untouched), but the translation engine sees
    full-sentence text instead of ASR-pause fragments. Without this, a
    fragment like "and pull into an auto repair shop." (no subject — Whisper
    happened to cut the sentence there) gets translated with no context and
    the target-language verb ends up in the wrong mood/tense.
    """
    members: List[Segment]
    source_text: str
    translated_text: Optional[str] = None
    from_cache: bool = False


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _make_group(members: List[Segment]) -> SentenceGroup:
    return SentenceGroup(members=members, source_text=" ".join(m.text.strip() for m in members))


def group_sentences(segments: List[Segment]) -> List[SentenceGroup]:
    """Group consecutive Segments into sentence-boundary units for translation."""
    groups: List[SentenceGroup] = []
    current: List[Segment] = []
    current_len = 0
    for seg in segments:
        current.append(seg)
        current_len += len(seg.text) + 1
        if (_ends_sentence(seg.text)
                or len(current) >= MAX_GROUP_MEMBERS
                or current_len >= MAX_GROUP_CHARS):
            groups.append(_make_group(current))
            current, current_len = [], 0
    if current:
        groups.append(_make_group(current))
    return groups


def _distribute_translation(translated_text: str, members: List[Segment]) -> List[str]:
    """
    Split one sentence-level translation back across the ASR-timed cues it
    was merged from. There's no word alignment available, so cues get words
    proportional to their share of the original (source) text length — the
    same class of approximation _split_long_text (above) uses for the
    reverse case (splitting one long segment's timespan across pieces).
    """
    if len(members) == 1:
        return [translated_text.strip()]

    words = translated_text.split()
    if not words:
        return ["" for _ in members]

    src_lens = [max(len(m.text.strip()), 1) for m in members]
    total_src_len = sum(src_lens)
    raw_counts = [len(words) * n / total_src_len for n in src_lens]
    counts = [int(c) for c in raw_counts]
    remainder = len(words) - sum(counts)
    # largest-remainder method so counts sum exactly to len(words)
    order = sorted(range(len(members)), key=lambda i: raw_counts[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1

    pieces = []
    cursor = 0
    for c in counts:
        pieces.append(" ".join(words[cursor:cursor + c]))
        cursor += c
    return pieces


def _apply_group_result(group: SentenceGroup, translated_text: str):
    group.translated_text = translated_text
    for member, piece in zip(group.members, _distribute_translation(translated_text, group.members)):
        member.translated_text = piece


def merge_empty_members(segments: List[Segment]) -> List[Segment]:
    """
    A member can end up with an empty translated_text if the sentence's
    translated word count is smaller than its fragment count (common when
    translating into a more compact target language). Fold its timespan
    into the previous cue instead of emitting a blank subtitle.
    """
    merged: List[Segment] = []
    for seg in segments:
        if seg.translated_text == "":
            if merged:
                merged[-1].end = seg.end
                continue
            seg.translated_text = seg.text  # first cue in the job; keep something visible
        merged.append(seg)
    return merged


def build_context_batch(all_groups: List[SentenceGroup],
                         target_groups: List[SentenceGroup]) -> List[tuple]:
    """
    Returns a list of (text_to_translate, had_context) pairs for the given
    target_groups. A single isolated sentence sometimes gives the model
    nothing to resolve cross-sentence agreement against — e.g. "She yells."
    translated alone can come back with a masculine verb despite "she"
    being right there, because a lone short sentence carries little
    statistical signal. So when the immediately preceding group (from the
    *full* ordered sequence, cache hit or not) exists, its source text is
    prepended purely as extra context; only the target sentence's own
    translation is kept afterwards (see strip_context_prefix).
    """
    if not TRANSLATE_WITH_CONTEXT:
        return [(g.source_text, False) for g in target_groups]

    index_of = {id(g): i for i, g in enumerate(all_groups)}
    batch = []
    for g in target_groups:
        i = index_of[id(g)]
        if i > 0:
            context = all_groups[i - 1].source_text
            batch.append((f"{context} {g.source_text}", True))
        else:
            batch.append((g.source_text, False))
    return batch


def strip_context_prefix(translated_text: str, had_context: bool) -> str:
    """
    Undo build_context_batch's prepending: split the combined translation
    back into sentences and keep only the last one. A no-op when had_context
    is False, so a target sentence that already legitimately contains
    multiple embedded clauses of its own (e.g. a short multi-sentence ASR
    fragment) is never touched — only text we ourselves prefixed gets split.
    """
    if not had_context:
        return translated_text.strip()
    parts = [p for p in re.split(r"(?<=[.!?।])\s+", translated_text.strip()) if p.strip()]
    return parts[-1] if parts else translated_text.strip()


def apply_translation_memory(groups: List[SentenceGroup], source_lang: str,
                              target_lang: str, glossary_version: str,
                              model_version: str) -> List[SentenceGroup]:
    """
    Look up each sentence group's merged text in the TM cache (this is why
    groups, not raw ASR fragments, are the cache key — the group is what
    actually gets sent to the translation engine).
    Mutates member Segments in place: sets .translated_text + .from_cache
    for hits. Returns the groups that still need translation.
    """
    conn = get_connection()
    cur = conn.cursor()
    misses = []
    for group in groups:
        h = hash_segment(group.source_text, source_lang, target_lang)
        cur.execute(
            """SELECT translated_text FROM translation_memory
               WHERE segment_hash=? AND source_lang=? AND target_lang=?
                     AND glossary_version=? AND model_version=?""",
            (h, source_lang, target_lang, glossary_version, model_version)
        )
        row = cur.fetchone()
        if row:
            _apply_group_result(group, row[0])
            group.from_cache = True
            for m in group.members:
                m.from_cache = True
        else:
            misses.append(group)
    conn.close()
    return misses


def assign_group_translations(groups: List[SentenceGroup], translated_texts: List[str]):
    """Attach freshly-translated text to each group and redistribute across its member cues."""
    for group, translated in zip(groups, translated_texts):
        _apply_group_result(group, translated)


def store_translations(groups: List[SentenceGroup], source_lang: str, target_lang: str,
                        glossary_version: str, model_version: str):
    """Persist newly-translated sentence groups back into the TM cache (used by Stage 5 too)."""
    conn = get_connection()
    cur = conn.cursor()
    for group in groups:
        if not group.translated_text:
            continue
        h = hash_segment(group.source_text, source_lang, target_lang)
        cur.execute(
            """INSERT OR REPLACE INTO translation_memory
               (segment_hash, source_lang, target_lang, glossary_version, model_version,
                source_text, translated_text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (h, source_lang, target_lang, glossary_version, model_version,
             group.source_text, group.translated_text)
        )
    conn.commit()
    conn.close()
