# Handover & Runbook

This document is for whoever owns/operates this tool after the build team
hands it off — it assumes no prior context beyond `README.md` (setup) and
`DEMO.md` (a guided walkthrough). For deploying to a separate, offline
machine (no native Python/ffmpeg/whisper.cpp setup on that machine, and no
internet there at all), see `DOCKER.md` instead of the steps below, which
describe a native (non-Docker) install.

## What this is

A local, offline-first web app that takes a video/audio recording and
produces: subtitles (SRT/VTT), an optional dubbed voiceover MP4, an
optional burned-in-subtitle MP4, and a DOCX quality report — for one of 22
Indian languages. It's designed to run on a single laptop in the field with
no internet connection required *after* first-time model downloads.

## Ownership model

- **Runs locally, per machine.** There is no shared server or database —
  each install has its own `data/translation_memory.db` (cache + reuse
  history) and `jobs/` directory (per-job output + logs). Nothing is
  synced between machines.
- **No accounts, no auth.** `app.py` binds to `127.0.0.1` only
  (`config.HOST`) — it is not exposed to the network by default. If you
  need it reachable from other devices on the same LAN, change `HOST` in
  `config.py`, but understand there is no authentication layer at all.
- **Models are cached locally** under `models/` and inside each Python
  environment's Hugging Face cache — reinstalling the app on a new machine
  means re-downloading them (multiple GB).

## Support runbook

**Where to look when something goes wrong:**

1. `jobs/pipeline.log` — job lifecycle events (start/finish/fail) with full
   tracebacks on failure. Rotates automatically (5MB × 3 backups).
2. The terminal running `python app.py` — same information, plus the
   per-stage timing breakdown for the most recent job and the detailed
   per-stage `print()` diagnostics from ASR/translation/TTS.
3. The Job Report DOCX for a specific job (`jobs/<job_id>/output/*_job_report.docx`)
   — per-segment quality/warnings for that job specifically.

**Common failure modes and fixes:**

| Symptom | Likely cause | Fix |
|---|---|---|
| Job fails immediately with a gated-model error | IndicTrans2 access not accepted / `HF_TOKEN` not set | See README "The one manual step" |
| Voiceover silently skipped, warning in report | Target language has no Indic-TTS checkpoint yet | Expected for 9 of the 22 languages — see README "Adding more Indic-TTS voices" |
| Voiceover fails outright (not skipped) | `.venv-tts` never set up | Run the one-time `.venv-tts` setup in README |
| Upload rejected | File over size/duration/format limits | Limits are in `config.py` — raise `MAX_VIDEO_SIZE_MB` etc. if field files are routinely larger |
| Whole job fails with a traceback | Check `jobs/pipeline.log` for the actual exception | Partial output for that job is auto-deleted (see Rollback below) — safe to just retry |
| App won't start | Python/ffmpeg/venv issue | Re-run `setup-mac.sh` (macOS) or the manual `Setup` steps in README |

**Rollback behavior:** if any pipeline stage raises an exception, the
partially-written `jobs/<job_id>/` directory for that job is automatically
deleted (`pipeline/orchestrator.py`) — a failed run never leaves half-built
output behind that could be mistaken for a completed job. The failure
itself is still fully logged (see above) before the directory is removed,
so the cause isn't lost, only the incomplete artifacts are.

**Restarting the app:** it has no persistent server state beyond the
SQLite cache and `jobs/` output — killing and re-running `python app.py`
(or `./run-mac.sh`) is always safe. In-flight jobs at the time of a
restart are lost (not resumed) and should just be re-run.

**Backing up / migrating to a new machine:** copy `data/` (translation
memory + glossary) and, if you want job history preserved, `jobs/`. Models
under `models/` and the Hugging Face caches will simply re-download on
first use on the new machine — nothing there needs to be copied unless
avoiding re-download matters more than disk space.

## Adoption approach for BAIF

1. **Pilot on already-available clips** — run 3-5 real field clips through
   using the manual testing checklist in `README.md` before wider rollout,
   so any language- or content-specific issues surface early.
2. **One machine, one operator, per field site** — the tool is designed
   for a single laptop; it does not currently support multiple concurrent
   operators sharing one install over a network.
3. **Train operators on**: the upload → language selection → download flow
   (see `DEMO.md`), reading the Job Report's warnings section, and where
   to find `jobs/pipeline.log` if a job fails and needs escalation.
4. **Extend coverage over time**: adding a new voiceover language or a
   glossary term (see README) doesn't require a code change — both are
   config/data edits an operator's technical point of contact can make
   without redeploying.

## Automated test coverage

`tests/` (run via `pytest`, see README "Running tests") covers the logic
most likely to silently regress: translation-engine routing rules,
translation-memory cache hit/miss behavior, numeric-preservation quality
scoring, SRT parsing for the "skip ASR" path, and the rollback behavior
described above. It intentionally does not run ASR/translation/TTS model
inference itself (that needs multi-GB model downloads and is exercised by
the manual checklist instead) — so a green test run means the routing and
bookkeeping logic is intact, not that translation quality on a given clip
is good.
