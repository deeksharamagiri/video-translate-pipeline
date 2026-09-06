## What's actually in this build

| Diagram box | File |
|---|---|
| Video/Audio input + limits | `config.py` + `app.py` upload validation |
| Stage 1 — Pre-Processing (FFmpeg) | `pipeline/stage1_preprocess.py` |
| Subtitle-found / Stage 2 — ASR (whisper.cpp + IndicConformer refinement) | `pipeline/stage1_preprocess.py` + `pipeline/stage2_asr.py` |
| Stage 3 — Segmentation + TM Check (SQLite) | `pipeline/stage3_segment_tm.py` |
| IndicTrans2 / NLLB-200 routing | `pipeline/translate.py` |
| Stage 4 — Subtitle Generation | `pipeline/stage4_subtitle.py` |
| SRT + VTT + Job Report (DOCX) | `pipeline/report.py` |
| Burned-in MP4 / Voiceover MP4 (Indic-TTS + FFmpeg) | `pipeline/delivery.py`, `tts_worker/` |
| Stage 5 — Archive & Reuse | `pipeline/stage5_archive.py` |
| Output folder → USB / offline playback | `jobs/<job_id>/output/` |
| Everything wired together | `pipeline/orchestrator.py` |
| Job-level logging (console + file) | `pipeline/logging_setup.py` → `jobs/pipeline.log` |
| Web UI + upload + progress + downloads | `app.py`, `static/` |

See [`HANDOVER.md`](HANDOVER.md) for ownership/runbook/support details,
[`DEMO.md`](DEMO.md) for a timed walkthrough script, and
[`DOCKER.md`](DOCKER.md) to package this for a different, offline machine.

---

## Setup

```bash
cd video-translate-pipeline
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That's it for subtitle-only jobs — no manual ffmpeg install, no manual
model downloads. Everything else happens automatically the first time it's
needed:

- **FFmpeg/FFprobe** — provisioned automatically by the `static-ffmpeg`
  package the first time `config.py` is imported (i.e. the first time you
  run the app). No `apt install` / `brew install` required.
- **whisper.cpp (`ggml-medium-q5_0.bin`), NLLB-200-distilled-600M** —
  downloaded automatically the first time a job actually needs ASR or
  non-Indic translation. `whisper.cpp` itself (the `whisper-cli` binary)
  must already be on `PATH`, or point `WHISPER_CPP_BINARY` at it — it is
  not pip-installable.
- **IndicTrans2** — downloaded automatically the first time a job needs an
  Indic↔Indic translation (e.g. Hindi→Marathi) — **with one caveat below.**
- **IndicConformer** — downloaded automatically the first time a job runs
  (it's the default ASR engine, refining whisper.cpp's transcript with a
  pass from a model trained specifically on Indian languages) —
  **also gated, same caveat below.** whisper.cpp's own transcript is
  still what drives subtitle timing; IndicConformer only replaces the
  text. If it isn't set up yet, jobs still complete fine on the
  whisper.cpp transcript alone — see "Troubleshooting" below.

**Voiceover (dubbed MP4) needs one extra one-time step**, because Indic-TTS
requires a newer `transformers` than IndicTrans2/NLLB are pinned to, so it
runs in its own virtualenv:

```bash
python3 -m venv .venv-tts
.venv-tts/bin/pip install -r tts_worker/requirements.txt
```

Skip this if you only need subtitles (SRT/VTT) — voiceover generation will
just be unavailable until it's set up, everything else works fine.
Indic-TTS checkpoints (~1.5GB per language) then download automatically the
first time that language is used for a voiceover.

## Run it

```bash
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:5000**. Upload a file, pick languages, tick the boxes
you want, hit **Run Pipeline**. The first job that touches a given
model/voice will pause while it downloads (progress prints to the terminal
running `python app.py`); every job after that is fast and fully offline.

## The one manual step: gated model access (IndicTrans2 + IndicConformer)

IndicTrans2's and IndicConformer's Hugging Face repos are free, but
"gated" — Hugging Face requires you to accept each repo's terms once,
while logged in, before any download (automatic or not) is allowed.
IndicTrans2 only matters for Indic↔Indic translation pairs (e.g.
Hindi→Marathi; translating to/from English via NLLB-200 needs none of
this). IndicConformer matters for every job, since it's the default ASR
refinement engine.

1. Create a free account: https://huggingface.co/join
2. While logged in, open each page and click **Agree and access repository**:
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
   - https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M
   - https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-1B
   - https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
3. Create an access token (Read access is enough): https://huggingface.co/settings/tokens
4. Before running `python app.py`, set it as an environment variable:
   ```bash
   export HF_TOKEN=     # macOS/Linux
   # or, Windows PowerShell:  $env:HF_TOKEN=""
   ```
   `transformers`/`huggingface_hub` picks this up automatically — no login
   command needed.

If you skip this, everything else still works: Indic↔Indic translation
jobs fail with a clear message pointing back to these steps (instead of
a raw stack trace), and IndicConformer refinement silently falls back to
the whisper.cpp transcript alone (with a warning in the Job Report)
rather than failing the job.

## Testing checklist

1. **Basic subtitle-only run**: upload a short MP3/WAV with no subtitle
   track, leave both checkboxes off. Confirm you get SRT + VTT + DOCX report.
2. **Video with embedded subtitles**: upload an MKV that already has a
   subtitle stream — confirm the UI shows "Skip ASR" and the process is
   much faster.
3. **Burned-in + voiceover**: upload a short MP4, tick both boxes, confirm
   you get 5 output files including two playable MP4s.
4. **Cache reuse**: run the *same* file through twice with the same
   language pair — the second run's "Cache hits" stat should jump toward
   100% and finish noticeably faster. Check the "Translation Memory — Recent
   Jobs" table at the bottom of the page.
5. **Low-SNR audio**: try a noisy/quiet recording — the Job Report should
   show a denoise warning.

## Adding more Indic-TTS voices / languages

Voiceover uses AI4Bharat/Indic-TTS. Supported languages are declared in
`config.INDIC_TTS_LANG_ZIP_MAP` (internal 3-letter code → release asset
code); add an entry there for any additional language AI4Bharat publishes
a checkpoint for (browse releases at
https://github.com/AI4Bharat/Indic-TTS/releases). The checkpoint downloads
automatically the first time that language is used for a voiceover.
Requesting voiceover for a target language *not* in that map doesn't fail
the job — it's skipped with a warning in the Job Report, and subtitles are
still produced normally.

## Running tests

```bash
.venv/bin/pip install -r requirements-dev.txt   # one-time, adds pytest
.venv/bin/python -m pytest
```

Tests cover pure logic that doesn't need model downloads: translation
engine routing (IndicTrans2 vs NLLB), translation-memory hashing/cache
hit-miss behavior, numeric-preservation quality scoring, SRT parsing, and
the job-failure rollback path. They don't exercise ASR/translation/TTS
model inference itself — that's covered by the manual checklist above.

## Adding glossary terms

Edit `data/glossary.json` — any term you add there will never be freely
translated by the model; it's protected and force-substituted with your
exact translation for each language.

## Speeding it up on CPU

Translation (IndicTrans2/NLLB) and voiceover (Indic-TTS) always run on
CPU regardless of GPU availability (see the GPU note below) — on a machine
without Apple Silicon Metal, or with `WHISPER_NO_GPU=1` set, translation is
typically the single largest chunk of a job's total time. Two knobs, both
on by default:

- **`TRANSLATION_NUM_BEAMS`** (`config.py`, default `1`) — beam width for
  both IndicTrans2 and NLLB's `generate()`. Previously hardcoded to `5`.
  Measured directly on this project's real cached NLLB checkpoint on an
  8-segment batch: beam=5 took 23.1s, beam=1 (greedy) took 7.8s — **~3x
  faster, identical output** on that batch. This mirrors the same
  trade-off already made for `whisper.cpp` ASR decoding. If a specific
  job's translation quality looks worse than expected, compare against
  wider beam search without touching code:
  ```bash
  TRANSLATION_NUM_BEAMS=5 python app.py
  ```
  Changing this changes the translation-memory cache key
  (`model_version_tag()`), so switching it never silently serves a
  translation made at a different beam width.
- **`BURN_IN_ENCODE_PRESET`** (`config.py`, default `"veryfast"`) — libx264
  preset used only for the burned-in-subtitles MP4 (subtitle burn-in can't
  be a stream copy, unlike voiceover muxing, which already uses `-c:v
  copy`). Measured ~35% faster encode than the ffmpeg default (`medium`)
  with no visible quality difference at the same CRF.

## Troubleshooting

- **First run is slow** — expected; it's downloading ffmpeg binaries and/or
  ML models. Watch the terminal running `python app.py` for progress.
- **GPU (Apple Silicon / Metal)** — ASR (`whisper.cpp`) uses Apple Metal
  automatically when available; no config needed. To force CPU-only (e.g.
  to A/B time a run without the GPU), set `WHISPER_NO_GPU=1` as an
  environment variable before running `app.py`. Translation
  (IndicTrans2/NLLB) and voiceover (Indic-TTS) already run CPU-only —
  Metal only affects the ASR stage. See "Speeding it up on CPU" above for
  the translation/encode-side knobs.
- **IndicTrans2 401/gated errors** — see "The one manual step" above.
- **IndicConformer not improving transcripts / job report warns it fell
  back** — usually means the gated repo access or `HF_TOKEN` step above
  hasn't been done yet on this machine; the job still completes on the
  whisper.cpp transcript alone. Uncheck "Refine transcript with
  IndicConformer" in the UI to skip the extra pass entirely.
- **Large files rejected** — limits are in `config.py` (`MAX_VIDEO_SIZE_MB`, `MAX_AUDIO_SIZE_MB`, durations) — raise them if your field files run longer than 15/30 minutes.
- **Something failed mid-job** — check `jobs/pipeline.log` (also printed to
  the terminal) for the full traceback. A failed job's partial output
  under `jobs/<job_id>/` is automatically deleted rather than left behind
  half-written.
