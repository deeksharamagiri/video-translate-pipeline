"""
config.py — single source of truth for the whole pipeline.
Every limit / model name here maps 1:1 to a box in the architecture slide.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Auto-provision FFmpeg/FFprobe binaries via the `static-ffmpeg` pip package.
# No system install / shell script required: the first time this runs, it
# downloads the right binaries for your OS and puts them on PATH. Every run
# after that is instant (binaries are cached under this package's data dir).
# ---------------------------------------------------------------------------
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception as _e:  # pragma: no cover - defensive; real ffmpeg on PATH still works
    print(f"[config] static-ffmpeg auto-provisioning skipped/failed ({_e}). "
          f"Falling back to any ffmpeg/ffprobe already on PATH.")

# ---------- VIDEO INPUT / AUDIO INPUT limits (top boxes) ----------
MAX_VIDEO_DURATION_SEC = 15 * 60          # "Up to 15 min"
MAX_VIDEO_SIZE_MB = 200
MAX_AUDIO_DURATION_SEC = 30 * 60          # "Up to 30 min"
MAX_AUDIO_SIZE_MB = 150

ALLOWED_VIDEO_EXT = {".mp4", ".mkv"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".ogg"}

# ---------- Stage 1 — Pre-Processing ----------
TARGET_SAMPLE_RATE = 16000                # 16 kHz mono WAV
SNR_DENOISE_THRESHOLD_DB = 15.0           # below this -> auto-denoise + warn user

# ---------- Stage 2 — ASR (faster-whisper) ----------
WHISPER_MODEL_SIZE = "small"              # tiny/base/small/medium/large-v3 — swap as needed
WHISPER_DEVICE = "cpu"                    # "cuda" if you have a GPU
WHISPER_COMPUTE_TYPE = "int8"             # int8 for CPU, float16 for GPU

# ---------- Stage 3 — Segmentation + Translation Memory ----------
TM_DB_PATH = os.path.join(DATA_DIR, "translation_memory.db")
SEGMENT_MAX_CHARS = 200                   # chunking granularity before hashing

# ---------- Translation engines ----------
INDIC_LANGS = {
    "asm", "ben", "brx", "doi", "guj", "hin", "kan", "kas", "kok", "mai",
    "mal", "mni", "mar", "nep", "ori", "pan", "san", "sat", "snd", "tam",
    "tel", "urd"  # 22 official Indian languages, per the diagram exactly.
    # NOTE: "eng" was previously included here as an "English pivot" — that
    # made English source/target count as an Indic language, so English->
    # Hindi was routing to IndicTrans2 instead of NLLB as the diagram
    # specifies ("BOTH Indian languages" -> IndicTrans2, anything else ->
    # NLLB). Worse, the IndicTrans2 call always used the indic-indic
    # checkpoint regardless, which isn't trained for English source — that's
    # very likely why English->Hindi output looked off. Removed.
}
ENGINE_CHOICES = {"auto", "indictrans2", "nllb"}  # "auto" = diagram's routing logic

# AI4Bharat publishes three direction-specific checkpoints. Previously only
# indic-indic was wired up, so English<->Indic silently fell through to
# NLLB (bug). Each direction has a "dist" (distilled, ~200-320M, default --
# lighter/faster on constrained CPU hardware) and "1B" (full-size) variant.
INDICTRANS2_MODEL_SIZE = "dist"  # "dist" (default) or "1B"

INDICTRANS2_MODELS = {
    "en_indic":    {"dist": "ai4bharat/indictrans2-en-indic-dist-200M",
                     "1B":   "ai4bharat/indictrans2-en-indic-1B"},
    "indic_en":    {"dist": "ai4bharat/indictrans2-indic-en-dist-200M",
                     "1B":   "ai4bharat/indictrans2-indic-en-1B"},
    "indic_indic": {"dist": "ai4bharat/indictrans2-indic-indic-dist-320M",
                     "1B":   "ai4bharat/indictrans2-indic-indic-1B"},
}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
GLOSSARY_PATH = os.path.join(DATA_DIR, "glossary.json")

# ---------- CPU speedup: dynamic INT8 quantization of translation models ----------
# torch.quantization.quantize_dynamic over nn.Linear layers, applied once at
# model-load time in translate.py. Applies to IndicTrans2 + NLLB only (not
# faster-whisper, which already runs int8 via WHISPER_COMPUTE_TYPE, and not
# the optional IndicConformer ASR model, and not Indic-TTS which runs in its
# own separate venv/process entirely). Default on; flip off to
# isolate a quality regression if one is ever suspected.
QUANTIZE_TRANSLATION_MODELS = True

# ---------- Optional: IndicConformer ASR (off by default) ----------
# MIT licensed, not gated. Whole-buffer transcription only -- no built-in
# segmentation/timestamps -- so it is used as an optional per-segment TEXT
# re-transcription pass layered on top of faster-whisper's timestamps, never
# a replacement for Stage 2's VAD-based segmentation.
INDIC_CONFORMER_MODEL = "ai4bharat/indic-conformer-600m-multilingual"
INDIC_CONFORMER_DECODING = "ctc"          # "ctc" (faster) or "rnnt" (slower/more accurate)
ASR_ENGINE_CHOICES = {"whisper", "indic_conformer"}  # "whisper" = always-on base engine
# IndicConformer's language codes diverge from both our internal 3-letter
# codes and faster-whisper's 2-letter codes in a few places (e.g. "ks" not
# "kas", "sd" not "snd") -- verified against the model card's supported list.
INDIC_CONFORMER_LANG_MAP = {
    "hin": "hi", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu",
    "kan": "kn", "pan": "pa", "ori": "or", "mal": "ml", "urd": "ur", "asm": "as",
    "brx": "brx", "doi": "doi", "kok": "kok", "kas": "ks", "mai": "mai",
    "mni": "mni", "san": "sa", "sat": "sat", "snd": "sd", "nep": "ne",
}

# ---------- Voiceover TTS: AI4Bharat/Indic-TTS (FastPitch + HiFi-GAN) ----------
# Lightweight, language-specific FastPitch (acoustic) + HiFi-GAN (vocoder)
# models per AI4Bharat/IIT Madras -- MIT licensed, not gated. Replaces
# Indic Parler-TTS (large generative model) and, before that, IndicF5 and
# Piper. Checkpoints are ~1.5GB per language, downloaded lazily on first use
# from GitHub Releases: https://github.com/AI4Bharat/Indic-TTS
#
# Runs in a SEPARATE venv (.venv-tts), invoked via subprocess
# (tts_worker/synthesize.py) -- coqui-tts (needed to load these checkpoints)
# requires a much newer `transformers` than transformers==4.44.2, which
# this project pins for IndicTrans2/NLLB. Installing both in one
# environment risks breaking translation, so they're kept isolated. Set up
# with (from the project root):
#   python3 -m venv .venv-tts
#   .venv-tts/bin/pip install -r tts_worker/requirements.txt
INDIC_TTS_VENV_DIR = os.path.join(BASE_DIR, ".venv-tts")
INDIC_TTS_WORKER_SCRIPT = os.path.join(BASE_DIR, "tts_worker", "synthesize.py")
INDIC_TTS_CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "indic_tts_checkpoints")
INDIC_TTS_RELEASE_BASE_URL = "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release"

# Our internal 3-letter code -> Indic-TTS's release asset language code.
# Only languages confirmed present in the v1-checkpoints-release assets are
# listed (verified against the actual GitHub Release asset list at time of
# writing). Rajasthani ("raj") is available upstream but isn't one of our
# 22 INDIC_LANGS, so it's omitted.
INDIC_TTS_LANG_ZIP_MAP = {
    "asm": "as", "ben": "bn", "brx": "brx", "eng": "en", "guj": "gu", "hin": "hi",
    "kan": "kn", "mal": "ml", "mni": "mni", "mar": "mr", "ori": "or", "pan": "pa",
    "tam": "ta", "tel": "te",
}
INDIC_TTS_DEFAULT_SPEAKER = "female"  # "male" or "female"; brx has no male speaker upstream

# ---------- Delivery: duration-aware voiceover alignment ----------
# After Indic-TTS synthesizes a segment, if its duration doesn't fit the
# segment's original time slot (seg.end - seg.start), time-stretch it with
# ffmpeg's atempo filter before stitching (existing adelay/amix collision
# avoidance stays as the fallback for whatever residual drift remains).
VOICEOVER_TIME_STRETCH = True
VOICEOVER_MAX_TEMPO_RATIO = 2.5   # clamp; atempo chains 2+ stages beyond its native 0.5-2.0 range

# ---------- Stage 4 — Subtitle Generation ----------
MAX_CHARS_PER_LINE = 42
MAX_LINES_PER_SUBTITLE = 2
MIN_GAP_BETWEEN_SUBTITLES_SEC = 0.08

# ---------- Stage 5 — Archive & Reuse ----------
ARCHIVE_DB_PATH = os.path.join(DATA_DIR, "translation_memory.db")  # same DB, different tables

# ---------- Server ----------
HOST = "127.0.0.1"
PORT = 5000