"""
Stage 5 — Archive & Reuse (SQLite)
  - Stores: segment hashes, translated text, glossary version, model version
  - Next job with same content -> >30% faster (via translation_memory cache
    already populated in Stage 3 + logged here in job_archive for reporting)
"""
from typing import List

from pipeline.stage3_segment_tm import get_connection, store_translations, Segment, SentenceGroup


def archive_job(job_id: str, segments: List[Segment], groups: List[SentenceGroup],
                 source_lang: str, target_lang: str,
                 glossary_version: str, model_version: str):
    """
    1. Persist the job's sentence groups (Stage 3's actual translation/cache
       unit — see stage3_segment_tm.group_sentences) into the shared TM
       cache, so future jobs with overlapping content hit the cache.
    2. Log this job's stats into job_archive for the job report / dashboard.
    """
    store_translations(groups, source_lang, target_lang, glossary_version, model_version)

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
