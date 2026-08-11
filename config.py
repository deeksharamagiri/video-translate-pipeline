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
#
# FFMPEG_BIN / FFPROBE_BIN hold the *exact* path to that downloaded binary,
# and every subprocess call in the pipeline uses these rather than the bare
# "ffmpeg"/"ffprobe" command names. That matters because add_paths() only
# *prepends* to PATH — if the system already has an ffmpeg earlier on PATH
# (e.g. a Homebrew build compiled without libass), plain "ffmpeg" can still
# resolve to that one instead, and it'll be missing filters/codecs this
# pipeline needs (the "subtitles" burn-in filter requires libass, which not
# every ffmpeg build includes). Calling the resolved path directly guarantees
# the full-featured static build is what actually runs, regardless of PATH.
# ---------------------------------------------------------------------------
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"
try:
    import static_ffmpeg
    from static_ffmpeg import run as _static_ffmpeg_run
    static_ffmpeg.add_paths()
    FFMPEG_BIN, FFPROBE_BIN = _static_ffmpeg_run.get_or_fetch_platform_executables_else_raise()
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
# Indic Parler-TTS (ai4bharat/indic-parler-tts) replaces Piper here. Piper
# only ever had real published voices for a handful of Indic languages —
# most of the 22 in INDIC_LANGS had NO offline Piper voice at all, so
# voiceover silently degraded to "unavailable" for most languages. Indic
# Parler-TTS is a single Hugging Face model that officially covers 20 Indic
# languages + English (plus unofficial support for a few more, e.g. Punjabi,
# Kashmiri), so one model now serves nearly every language in INDIC_LANGS
# instead of a handful of separately-downloaded per-language voice files.
INDIC_PARLER_TTS_MODEL = "ai4bharat/indic-parler-tts"
INDIC_PARLER_TTS_DEVICE = "cpu"           # "cuda" if you have a GPU

# Indic Parler-TTS doesn't take a language code — it's steered per segment by
# a natural-language "voice description" caption fed in alongside the text,
# and it auto-detects the language from the segment text/script itself. The
# speaker names below are the per-language "Recommended Speakers" from the
# model card's voice table
# (https://huggingface.co/ai4bharat/indic-parler-tts#-using-a-specific-speaker),
# which keeps the same voice consistent across every segment/job for a given
# language. A handful of languages (kas, kok, mai, sat, snd, urd) have no
# named recommended speaker on the model card — those fall back to a generic
# high-quality-voice description; the model still auto-detects the language
# and picks an appropriate voice, it's just not locked to one named speaker.
def _voice(speaker: str) -> str:
    return (f"{speaker}'s voice is clear and natural, delivered at a "
            f"moderate pace and pitch. The recording is of very high "
            f"quality, with no background noise.")


_GENERIC_VOICE = ("A clear, natural voice speaks at a moderate pace and "
                   "pitch. The recording is of very high quality, with no "
                   "background noise.")

INDIC_PARLER_VOICE_DESCRIPTIONS = {
    "asm": _voice("Amit"),
    "ben": _voice("Arjun"),
    "brx": _voice("Bikram"),
    "doi": _voice("Karan"),
    "guj": _voice("Yash"),
    "hin": _voice("Rohit"),
    "kan": _voice("Suresh"),
    "kas": _GENERIC_VOICE,   # unofficial support, no named recommended speaker
    "kok": _GENERIC_VOICE,   # unofficial support, no named recommended speaker
    "mai": _GENERIC_VOICE,   # unofficial support, no named recommended speaker
    "mal": _voice("Anjali"),
    "mni": _voice("Laishram"),
    "mar": _voice("Sanjay"),
    "nep": _voice("Amrita"),
    "ori": _voice("Manas"),
    "pan": _voice("Divjot"),  # unofficial support, but has a named speaker
    "san": _voice("Aryan"),
    "sat": _GENERIC_VOICE,   # unofficial support, no named recommended speaker
    "snd": _GENERIC_VOICE,   # unofficial support, no named recommended speaker
    "tam": _voice("Jaya"),
    "tel": _voice("Prakash"),
    "urd": _GENERIC_VOICE,   # unofficial support, no named recommended speaker
    "eng": _voice("Thoma"),
}

# ---------- Stage 5 — Archive & Reuse ----------
ARCHIVE_DB_PATH = os.path.join(DATA_DIR, "translation_memory.db")  # same DB, different tables

# ---------- Server ----------
HOST = "127.0.0.1"
PORT = 5000