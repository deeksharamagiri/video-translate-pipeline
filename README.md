## What's actually in this build

| Diagram box | File |
|---|---|
| Video/Audio input + limits | `config.py` + `app.py` upload validation |
| Stage 1 — Pre-Processing (FFmpeg) | `pipeline/stage1_preprocess.py` |
| Subtitle-found / Stage 2 — ASR (faster-whisper) | `pipeline/stage1_preprocess.py` + `pipeline/stage2_asr.py` |
| Stage 3 — Segmentation + TM Check (SQLite) | `pipeline/stage3_segment_tm.py` |
| IndicTrans2 (indic-indic) / Krutrim-Translate routing | `pipeline/translate.py` |
| Stage 4 — Subtitle Generation | `pipeline/stage4_subtitle.py` |
| SRT + VTT + Job Report (DOCX) | `pipeline/report.py` |
| Burned-in MP4 / Voiceover MP4 (MMS-TTS + FFmpeg) | `pipeline/delivery.py` |
| Stage 5 — Archive & Reuse | `pipeline/stage5_archive.py` |
| Output folder → USB / offline playback | `jobs/<job_id>/output/` |
| Everything wired together | `pipeline/orchestrator.py` |
| Web UI + upload + progress + downloads | `app.py`, `static/` |

Two translation models — **IndicTrans2 indic-indic-1B** for Indic↔Indic,
**Krutrim-Translate** for English↔Indic — and **one voiceover model**
(MMS-TTS). NLLB-200 and Indic Parler-TTS were both removed to keep memory
footprint down on an 8GB CPU-only demo machine.

Krutrim-Translate, not IndicTrans2's own en-indic-1B/indic-en-1B
checkpoints, handles English↔Indic: it's a distilled IndicTrans2 derivative
built specifically for fast CPU inference. Measured on the same 22
sentences, CPU-only: **Krutrim-Translate 3.9s (0.18s/sentence) vs.
IndicTrans2 en-indic-1B 476.8s (21.7s/sentence) — ~122x slower** for the
full 1B-parameter checkpoint doing beam-search with no GPU. The en-indic-1B/
indic-en-1B checkpoints stay available via the "indictrans2-en" forced
engine for A/B comparison, but "auto" routing uses Krutrim-Translate.

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
- **faster-whisper** — downloaded automatically the first time a job
  actually needs ASR (i.e. the input has no embedded subtitle track).
- **IndicTrans2 indic-indic-1B** — downloaded automatically the first time a
  job needs an Indic↔Indic translation (e.g. Hindi→Marathi) — **with one
  caveat below.**
- **Krutrim-Translate** — downloaded automatically the first time a job
  needs an English↔Indic translation in one of 9 languages (Hindi, Bengali,
  Kannada, Marathi, Malayalam, Gujarati, Punjabi, Telugu, Tamil) — **same
  caveat below.** Fast (CTranslate2, distilled specifically for CPU
  English↔Indic inference — see benchmark above). English↔Indic pairs
  outside those 9 still route here under "auto" (it's the only
  English↔Indic engine in the default path), just without a coverage
  guarantee.
- **IndicTrans2 en-indic-1B / indic-en-1B** — not used by "auto" routing;
  only downloads if you explicitly pick "Force IndicTrans2 en-indic-1B"
  from the engine dropdown, for A/B comparison against Krutrim-Translate.
  Two separate directional checkpoints, both far slower on CPU than
  Krutrim-Translate for the same sentences (see benchmark above) — same
  gated-access caveat below.
- **MMS-TTS** (Meta Massively Multilingual Speech) — downloaded
  automatically the first time a job needs voiceover, one small checkpoint
  per language (`facebook/mms-tts-<code>`). Covers 10 languages: Assamese,
  Bengali, Gujarati, **Hindi**, Maithili, Malayalam, **Marathi**, Punjabi,
  Tamil, Telugu — verified against the live Hugging Face Hub, not assumed;
  the rest of `INDIC_LANGS` simply don't have a checkpoint published under
  that naming, so voiceover is unavailable for them. Not gated — no
  HF_TOKEN step needed for this one. Non-autoregressive (VITS), so it
  generates each segment's audio in one pass rather than token-by-token —
  fast even without a GPU, at the cost of one fixed voice per language (no
  style/emotion steering) and noticeably flatter/less clear audio than
  Indic Parler-TTS (the model this replaced — see "Known limitations"
  below on why we're not going back to it, and what was explored instead).

Each generated voiceover segment is placed on a timeline derived from —
but not strictly bound to — the original video's subtitle timestamps
(`delivery.py`'s `_get_sequential_timeline`): translated/synthesized
speech essentially never matches the original audio's pacing, so without
this, nearly every segment left an audible silent gap before it (however
long the original video's pause happened to be). Gaps are now capped at
`max_gap_sec` (0.4s default) so narration sounds continuous; segments that
run long still get pushed forward with no overlap, unchanged.

## Run it

```bash
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:5000**. Upload a file, pick languages, tick the boxes
you want, hit **Run Pipeline**. The first job that touches a given
model/voice will pause while it downloads (progress prints to the terminal
running `python app.py`); every job after that is fast and fully offline.

## The one manual step: IndicTrans2 / Krutrim-Translate access

Both IndicTrans2's (all three checkpoints) and Krutrim-Translate's Hugging
Face repos are free but "gated" — Hugging Face requires you to accept the
terms once, while logged in, before any download (automatic or not) is
allowed. Voiceover via MMS-TTS is not gated and needs none of this.

1. Create a free account: https://huggingface.co/join
2. While logged in, open each page and click **Agree and access repository**:
   - https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-1B
   - https://huggingface.co/krutrim-ai-labs/Krutrim-Translate (Krutrim
     Community License, not MIT — read it before accepting)
3. Create an access token (Read access is enough): https://huggingface.co/settings/tokens
4. Before running `python app.py`, set it as an environment variable:
   ```bash
   export HF_TOKEN=     # macOS/Linux
   # or, Windows PowerShell:  $env:HF_TOKEN=""
   ```
   `transformers`/`huggingface_hub` picks this up automatically — no login
   command needed.

If you skip this, everything else works fine; only translation jobs will
fail, with a clear message pointing back to these steps (instead of a raw
stack trace).

Krutrim-Translate itself is not a standard `transformers` checkpoint — its HF
repo ships CTranslate2-exported weights (two directional models, English→
Indic and Indic→English) plus a sentencepiece vocab, no `AutoModel`-loadable
files. `pipeline/krutrim_engine/` vendors the model's own reference inference
wrapper (from https://github.com/ola-krutrim/KrutrimTranslate) to run it —
this is why it needs `ctranslate2` rather than just `transformers`.

**Note on MMS-TTS verification:** its language coverage (the 10 codes
listed above) was checked against the live Hugging Face Hub via
`huggingface_hub.HfApi.model_info()` — not assumed from documentation — and
the actual synthesis call was run end-to-end against the real weights for
both Hindi and Marathi, producing real audio (2.37s / 1.95s clips from a
short test sentence each).

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
   100% and finish noticeably faster.
5. **Low-SNR audio**: try a noisy/quiet recording — the Job Report should
   show a denoise warning.

## Adding more voiceover languages

MMS-TTS has one fixed voice per language, no natural-language steering —
if Meta publishes a checkpoint for a language not currently in
`config.MMS_TTS_LANGS`, add its code there (matching the
`facebook/mms-tts-<code>` naming) to pick it up; it'll download
automatically the next time that language is used for voiceover. There's
no fallback engine anymore, so languages outside `MMS_TTS_LANGS` simply
can't produce voiceover — the UI's target-language dropdown is deliberately
limited to that set.

## Adding glossary terms

Edit `data/glossary.json` — any term you add there will never be freely
translated by the model; it's protected and force-substituted with your
exact translation for each language.

## Known limitations

- **Voiceover clarity vs. speed.** MMS-TTS sounds noticeably flatter than
  Indic Parler-TTS (the model it replaced), which is the honest tradeoff
  for going from ~50s/segment down to under a second. The one alternative
  that looked promising on paper — AI4Bharat's Indic-TTS (FastPitch +
  HiFi-GAN, same non-autoregressive speed class) — was investigated and
  rejected: the project has had no commits since November 2024, has no
  `pip`/`transformers` install path (requires a forked, legacy Coqui-TTS
  package), and its checkpoints are ~1.5GB *per language* — about 10x
  MMS-TTS's footprint, working against the whole point of this setup on an
  8GB machine. Revisit if a GPU-equipped or higher-RAM target environment
  ever becomes available.
- **Translation naturalness vs. speed**, same shape of tradeoff: Krutrim-
  Translate is a distilled model, so it occasionally makes different word
  or idiom choices than the full IndicTrans2 en-indic-1B checkpoint it was
  distilled from — not uniformly worse (it's actually gotten at least one
  hard grammatical case right that the bigger model got wrong in direct
  comparison), just a different quality ceiling in exchange for ~122x the
  speed on CPU. The bigger checkpoints stay available via the
  "indictrans2-en" forced engine for spot-checking.
- **Video/audio duration limits aren't actually enforced.** `config.py`
  declares `MAX_VIDEO_DURATION_SEC` (15 min) and `MAX_AUDIO_DURATION_SEC`
  (30 min), and the UI's dropzone hint text advertises them, but nothing
  in `app.py` or `pipeline/stage1_preprocess.py` actually checks a file's
  duration against them — only file *size* (`MAX_VIDEO_SIZE_MB` /
  `MAX_AUDIO_SIZE_MB`) is enforced on upload. A long, small-file-size
  video (e.g. low bitrate) would currently be accepted and processed in
  full regardless of the advertised cap. Not fixed here since it wasn't
  in scope for this pass — worth wiring up `stage1_preprocess.probe_streams()`
  (already computes duration internally) into the upload check in `app.py`
  if this matters for your use case.

## Troubleshooting

- **First run is slow** — expected; it's downloading ffmpeg binaries and/or
  ML models. Watch the terminal running `python app.py` for progress.
- **CUDA / GPU** — set `WHISPER_DEVICE = "cuda"` and `WHISPER_COMPUTE_TYPE = "float16"` in `config.py` if you have an NVIDIA GPU; CPU (`int8`) is the safe default.
- **Apple Silicon (M1/M2/etc.) / MPS** — `TRANSLATION_DEVICE` (IndicTrans2 checkpoints) and `MMS_TTS_DEVICE` default to `"cpu"` on purpose, to match the no-GPU demo machine exactly. Set either to `"mps"` if you want Apple Silicon GPU acceleration during development — both `resolve_torch_device()` calls fall back to CPU automatically if MPS isn't available. Note MPS shares the same unified memory as the rest of the system (not separate VRAM) — on an 8GB Mac, loading both IndicTrans2 1B checkpoints + MMS-TTS at once can exceed MPS's own allocation ceiling ("MPS backend out of memory") well before system RAM is exhausted. `WHISPER_DEVICE` and `KRUTRIM_TRANSLATE_DEVICE` stay `"cpu"` unconditionally — both run on CTranslate2, which has no MPS backend at all.
- **IndicTrans2 / Krutrim-Translate 401/gated errors** — see "The one manual step" above.
- **Large files rejected** — limits are in `config.py` (`MAX_VIDEO_SIZE_MB`, `MAX_AUDIO_SIZE_MB`, durations) — raise them if your field files run longer than 15/30 minutes.
- **Long files still slow / system under memory pressure** — this pipeline loads several multi-GB models into one process (ASR + translation + optionally TTS), which adds up fast on 8GB-RAM machines. If you're processing long files regularly, close other memory-heavy apps first, or consider a smaller `WHISPER_MODEL_SIZE` (e.g. `"base"` instead of `"small"`).
