## What's actually in this build

| Diagram box | File |
|---|---|
| Video/Audio input + limits | `config.py` + `app.py` upload validation |
| Stage 1 — Pre-Processing (FFmpeg) | `pipeline/stage1_preprocess.py` |
| Subtitle-found / Stage 2 — ASR (faster-whisper) | `pipeline/stage1_preprocess.py` + `pipeline/stage2_asr.py` |
| Stage 3 — Segmentation + TM Check (SQLite) | `pipeline/stage3_segment_tm.py` |
| IndicTrans2 / NLLB-200 routing | `pipeline/translate.py` |
| Stage 4 — Subtitle Generation | `pipeline/stage4_subtitle.py` |
| SRT + VTT + Job Report (DOCX) | `pipeline/report.py` |
| Burned-in MP4 / Voiceover MP4 (Piper + FFmpeg) | `pipeline/delivery.py` |
| Stage 5 — Archive & Reuse | `pipeline/stage5_archive.py` |
| Output folder → USB / offline playback | `jobs/<job_id>/output/` |
| Everything wired together | `pipeline/orchestrator.py` |
| Web UI + upload + progress + downloads | `app.py`, `static/` |

---

## Setup

```bash
cd video-translate-pipeline
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That's it — no shell script, no manual ffmpeg install, no manual model
downloads. Everything else happens automatically the first time it's needed:

- **FFmpeg/FFprobe** — provisioned automatically by the `static-ffmpeg`
  package the first time `config.py` is imported (i.e. the first time you
  run the app). No `apt install` / `brew install` required.
- **faster-whisper, NLLB-200-distilled-600M** — downloaded automatically the
  first time a job actually needs ASR or non-Indic translation.
- **Piper TTS voices** — downloaded automatically the first time you request
  a voiceover MP4 for a given language.
- **IndicTrans2** — downloaded automatically the first time a job needs an
  Indic↔Indic translation (e.g. Hindi→Marathi) — **with one caveat below.**

## Run it

```bash
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:5000**. Upload a file, pick languages, tick the boxes
you want, hit **Run Pipeline**. The first job that touches a given
model/voice will pause while it downloads (progress prints to the terminal
running `python app.py`); every job after that is fast and fully offline.

## The one manual step: IndicTrans2 access

IndicTrans2's Hugging Face repos are free and MIT-licensed, but "gated" —
Hugging Face requires you to accept the terms once, while logged in, before
any download (automatic or not) is allowed. This only matters for
Indic↔Indic pairs (e.g. Hindi↔Marathi, Tamil↔Telugu); translating to/from
English via NLLB-200 needs none of this.

1. Create a free account: https://huggingface.co/join
2. While logged in, open each page and click **Agree and access repository**:
   - https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-1B
3. Create an access token (Read access is enough): https://huggingface.co/settings/tokens
4. Before running `python app.py`, set it as an environment variable:
   ```bash
   export HF_TOKEN=     # macOS/Linux
   # or, Windows PowerShell:  $env:HF_TOKEN=""
   ```
   `transformers`/`huggingface_hub` picks this up automatically — no login
   command needed.

If you skip this, everything else works fine; only Indic↔Indic jobs will
fail with a clear message pointing back to these steps (instead of a raw
stack trace).

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

## Adding more Piper voices / languages

Add a language code → filename in `config.PIPER_VOICE_MAP`, and the matching
download URLs (`.onnx` + `.onnx.json`) in `config.PIPER_VOICE_URLS` (browse
available voices at https://huggingface.co/rhasspy/piper-voices). It'll
auto-download the first time that language is used for a voiceover.

## Adding glossary terms

Edit `data/glossary.json` — any term you add there will never be freely
translated by the model; it's protected and force-substituted with your
exact translation for each language.

## Troubleshooting

- **First run is slow** — expected; it's downloading ffmpeg binaries and/or
  ML models. Watch the terminal running `python app.py` for progress.
- **CUDA / GPU** — set `WHISPER_DEVICE = "cuda"` and `WHISPER_COMPUTE_TYPE = "float16"` in `config.py` if you have an NVIDIA GPU; CPU (`int8`) is the safe default.
- **IndicTrans2 401/gated errors** — see "The one manual step" above.
- **Large files rejected** — limits are in `config.py` (`MAX_VIDEO_SIZE_MB`, `MAX_AUDIO_SIZE_MB`, durations) — raise them if your field files run longer than 15/30 minutes.
