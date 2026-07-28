"""
Stage 5 — Archive & Reuse (SQLite)
  - Stores: segment hashes, translated text, glossary version, model version
  - Next job with same content -> >30% faster (via translation_memory cache
    already populated in Stage 3 + logged here in job_archive for reporting)
"""
from typing import List

from pipeline.stage3_segment_tm import get_connection, store_translations, Segment


def archive_job(job_id: str, segments: List[Segment], source_lang: str, target_lang: str,
                 glossary_version: str, model_version: str):
    """
    1. Persist any newly-translated segments into the shared TM cache
       (so future jobs with overlapping content hit the cache).
    2. Log this job's stats into job_archive for the job report / dashboard.
    """
    store_translations(segments, source_lang, target_lang, glossary_version, model_version)

    cache_hits = sum(1 for s in segments if s.from_cache)
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO job_archive
           (job_id, source_lang, target_lang, glossary_version, model_version,
            segment_count, cache_hit_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (job_id, source_lang, target_lang, glossary_version, model_version,
         len(segments), cache_hits)
    )
    conn.commit()
    conn.close()
    return {"segment_count": len(segments), "cache_hit_count": cache_hits}


def get_archive_stats():
    """Used by the UI to show a small 'reuse dashboard'."""
    conn = get_connection()
    cur = conn.execute(
        "SELECT job_id, source_lang, target_lang, segment_count, cache_hit_count, created_at "
        "FROM job_archive ORDER BY created_at DESC LIMIT 20"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "job_id": r[0], "source_lang": r[1], "target_lang": r[2],
            "segment_count": r[3], "cache_hit_count": r[4], "created_at": r[5],
            "cache_hit_pct": round(100 * r[4] / r[3], 1) if r[3] else 0.0,
        }
        for r in rows
    ]
