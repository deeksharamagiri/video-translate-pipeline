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

INDICTRANS2_MODEL = "ai4bharat/indictrans2-indic-indic-1B"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
GLOSSARY_PATH = os.path.join(DATA_DIR, "glossary.json")

# ---------- Stage 4 — Subtitle Generation ----------
MAX_CHARS_PER_LINE = 42
MAX_LINES_PER_SUBTITLE = 2
MIN_GAP_BETWEEN_SUBTITLES_SEC = 0.08

# ---------- Delivery: Burned-in / Voiceover (user opt-in) ----------
PIPER_VOICES_DIR = os.path.join(MODELS_DIR, "piper_voices")

# IMPORTANT: Piper only has real published voices for a handful of Indic
# languages — most of the 22 in INDIC_LANGS have NO offline Piper voice at
# all. This map only lists languages that were verified against Piper's
# actual voice catalog (https://huggingface.co/rhasspy/piper-voices/tree/main)
# as of this writing. Previously "mar" (Marathi) pointed at a URL that
# doesn't exist in that catalog — it would 404 at runtime and crash the
# whole job. Removed rather than left broken.
#
# Languages WITHOUT a Piper voice (voiceover unavailable, subtitles/burned-in
# still work fine): tam, kan, ben, guj, pan, ori, asm, san, kok, mai, mni,
# snd, doi, brx, sat, kas, mar. If you need voiceover for one of these,
# you'll need a different offline TTS engine (e.g. Coqui TTS, or AI4Bharat's
# Indic Parler-TTS which covers ~20 Indic languages) — Piper just doesn't
# have the voice.
PIPER_VOICE_MAP = {
    "hin": "hi_IN-pratham-medium.onnx",
    "eng": "en_US-lessac-medium.onnx",
    "mal": "ml_IN-arjun-medium.onnx",
    "nep": "ne_NP-chitwan-medium.onnx",
    "tel": "te_IN-maya-medium.onnx",
    "urd": "ur_PK-fasih-medium.onnx",
}
# Where to fetch each voice from if it's not already present locally.
# Source: https://huggingface.co/rhasspy/piper-voices (verify a language's
# folder exists there before adding a new entry — a guessed URL that 404s
# is exactly the bug this map previously had).
PIPER_VOICE_URLS = {
    "hi_IN-pratham-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
    "hi_IN-pratham-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json",
    "en_US-lessac-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-lessac-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    "ml_IN-arjun-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ml/ml_IN/arjun/medium/ml_IN-arjun-medium.onnx",
    "ml_IN-arjun-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ml/ml_IN/arjun/medium/ml_IN-arjun-medium.onnx.json",
    "ne_NP-chitwan-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ne/ne_NP/chitwan/medium/ne_NP-chitwan-medium.onnx",
    "ne_NP-chitwan-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ne/ne_NP/chitwan/medium/ne_NP-chitwan-medium.onnx.json",
    "te_IN-maya-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/maya/medium/te_IN-maya-medium.onnx",
    "te_IN-maya-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/maya/medium/te_IN-maya-medium.onnx.json",
    "ur_PK-fasih-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx",
    "ur_PK-fasih-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/ur/ur_PK/fasih/medium/ur_PK-fasih-medium.onnx.json",
}

# ---------- Stage 5 — Archive & Reuse ----------
ARCHIVE_DB_PATH = os.path.join(DATA_DIR, "translation_memory.db")  # same DB, different tables

# ---------- Server ----------
HOST = "127.0.0.1"
PORT = 5000