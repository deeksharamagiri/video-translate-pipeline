"""
config.py — single source of truth for the whole pipeline.
Every limit / model name here maps 1:1 to a box in the architecture slide.
"""
import os

# Every TTS/translation model here loads a `tokenizers`-backed HF tokenizer,
# and the pipeline also shells out to ffmpeg via subprocess (which forks).
# Forking after tokenizers' internal thread pool has started makes it print
# a "process just got forked" warning and disable itself defensively — set
# this before anything imports transformers/tokenizers so it never spins
# the pool up in the first place. setdefault() so an explicit env var
# (e.g. in a shell profile) still wins.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def resolve_torch_device(requested_device: str) -> str:
    """
    Shared CUDA/MPS/CPU fallback for every `transformers`/torch-based model
    loader (translate.py's IndicTrans2 checkpoints, delivery.py's MMS-TTS).
    Falls back to CPU with a printed note if the requested accelerator
    isn't actually available.

    NOT used by WHISPER_DEVICE below — that runs on CTranslate2, which only
    supports "cpu"/"cuda", no Apple Silicon MPS backend — so it stays
    CPU-only regardless of what hardware is present. Imports torch lazily
    so importing config.py (done by every module at startup, including
    plain Flask routes) doesn't pay torch's import cost up front.
    """
    import torch

    requested_device = str(requested_device).strip().lower()

    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            return requested_device
        print("[config] CUDA was requested, but CUDA is unavailable. Using CPU.")
        return "cpu"

    if requested_device in {"mps", "mps:0"}:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print("[config] MPS was requested, but MPS is unavailable. Using CPU.")
        return "cpu"

    return "cpu"

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
# CTranslate2-backed (faster-whisper's engine) — only supports "cpu"/"cuda",
# no Apple Silicon MPS backend, so this stays "cpu" even on a Mac with a GPU.
WHISPER_DEVICE = "cpu"                    # "cuda" if you have an NVIDIA GPU
WHISPER_COMPUTE_TYPE = "int8"             # int8 for CPU, float16 for GPU

# ---------- Stage 3 — Segmentation + Translation Memory ----------
TM_DB_PATH = os.path.join(DATA_DIR, "translation_memory.db")
SEGMENT_MAX_CHARS = 200                   # chunking granularity before hashing

# When translating a sentence, prepend the immediately preceding sentence as
# extra (discarded-after) context — helps cross-sentence agreement (e.g.
# pronoun gender: a lone "She yells." can come back with a masculine verb,
# since a single isolated sentence gives the model little to anchor on)
# that dedicated NMT models routinely get wrong on short sentences. Costs
# roughly 2x the input tokens per translated sentence; set False to disable.
TRANSLATE_WITH_CONTEXT = True

# ---------- Translation engines ----------
# BOTH SIDES ARE INDIC LANGUAGES -> IndicTrans2 indic-indic-1B
# ONE SIDE IS ENGLISH            -> Krutrim-Translate
#
# Krutrim-Translate is a distilled IndicTrans2 derivative built specifically
# for fast CPU English<->Indic inference. It replaced IndicTrans2's own
# en-indic-1B/indic-en-1B checkpoints as the "auto" default after a real
# CPU benchmark: 22 sentences took 476.8s (21.7s/sentence) through the full
# 1B-param en-indic-1B checkpoint with beam=5, vs. 4.2s (0.19s/sentence)
# through Krutrim for the exact same sentences — ~114x faster, and it's the
# reason Krutrim was distilled from IndicTrans2 in the first place. The
# en-indic-1B/indic-en-1B checkpoints are still available via the
# "indictrans2-en" forced engine for A/B comparison.
INDIC_LANGS = {
    "asm", "ben", "brx", "doi", "guj", "hin", "kan", "kas", "kok", "mai",
    "mal", "mni", "mar", "nep", "ori", "pan", "san", "sat", "snd", "tam",
    "tel", "urd"  # 22 official Indian languages, per the diagram exactly.
    # NOTE: "eng" was previously included here as an "English pivot" — that
    # made English source/target count as an Indic language, so English->
    # Hindi was routing to the indic-indic checkpoint instead of the
    # English<->Indic model — which isn't trained for English source —
    # that's very likely why English->Hindi output looked off. Removed.
}
ENGINE_CHOICES = {"auto", "indictrans2", "indictrans2-en", "krutrim"}

INDICTRANS2_MODEL = "ai4bharat/indictrans2-indic-indic-1B"
# Dedicated English<->Indic checkpoints (full-size, non-distilled — the
# model Krutrim-Translate was distilled from). Gated on HF like
# INDICTRANS2_MODEL; not used by "auto" routing (Krutrim-Translate is the
# default for English<->Indic — see above) but available via the
# "indictrans2-en" forced engine for A/B comparison.
INDICTRANS2_EN_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-1B"
INDICTRANS2_INDIC_EN_MODEL = "ai4bharat/indictrans2-indic-en-1B"
GLOSSARY_PATH = os.path.join(DATA_DIR, "glossary.json")

# Krutrim-Translate (krutrim-ai-labs) — English<->Indic only, 9 languages.
# Gated on HF like IndicTrans2 (same HF_TOKEN / "accept terms" flow — see
# README.md). Ships as CTranslate2-exported weights (two directional
# checkpoints, no transformers/AutoModel format), so it's loaded via the
# vendored pipeline/krutrim_engine wrapper (the model's own reference
# inference code) rather than `transformers`.
KRUTRIM_TRANSLATE_REPO = "krutrim-ai-labs/Krutrim-Translate"
KRUTRIM_TRANSLATE_DIR = os.path.join(MODELS_DIR, "krutrim-translate")
# CTranslate2-backed — only supports "cpu"/"cuda", no Apple Silicon MPS
# backend, so this stays "cpu" even on a Mac with a GPU (same limitation as
# WHISPER_DEVICE). This is also the fast path, so it being CPU-only is not
# a practical concern the way it was for the 1B-param transformers models.
KRUTRIM_TRANSLATE_DEVICE = "cpu"          # "cuda" if you have an NVIDIA GPU
KRUTRIM_INDIC_LANGS = {"hin", "ben", "kan", "mar", "mal", "guj", "pan", "tel", "tam"}

# IndicTrans2 checkpoints are plain `transformers` models and CAN use MPS
# (measured faster in isolation on an M1). Set to "cpu" here on purpose:
# the demo machine has no GPU, and even on this M1, MPS ran out of its
# shared memory pool once both IndicTrans2 1B checkpoints + MMS-TTS were
# resident at once ("MPS backend out of memory... 9.04 GiB allocated...
# max allowed 9.07 GiB" — MPS shares the same 8GB unified memory as
# everything else, it isn't separate VRAM). Testing CPU-only here matches
# the real demo environment exactly. With Krutrim now handling the common
# English<->Indic path, this setting mostly only affects the (less
# frequently used, and not yet re-benchmarked) indic-indic-1B checkpoint.
TRANSLATION_DEVICE = "cpu"                # "mps" for Apple Silicon GPU, "cuda" if you have an NVIDIA GPU

# ---------- Stage 4 — Subtitle Generation ----------
MAX_CHARS_PER_LINE = 42
MAX_LINES_PER_SUBTITLE = 2
MIN_GAP_BETWEEN_SUBTITLES_SEC = 0.08

# ---------- Delivery: Burned-in / Voiceover (user opt-in) ----------
# facebook/mms-tts-<lang> (Meta Massively Multilingual Speech) — VITS
# (non-autoregressive), ~36M params, ONE CHECKPOINT PER LANGUAGE. The sole
# voiceover engine for the demo build (Indic Parler-TTS was dropped —
# broader language coverage, but autoregressive and ~50s/segment even with
# a GPU, vs. MMS-TTS's 341ms/segment on MPS / ~951ms on CPU-only). Not
# gated — no HF_TOKEN/approval needed.
#
# Coverage was verified against the live HF Hub (not assumed) for every
# INDIC_LANGS code — only these 10 repos actually exist under the
# `facebook/mms-tts-<code>` naming as of this writing; the rest 404
# (brx, doi, kas, kok, mni, nep, ori, san, sat, snd, urd all NOT FOUND).
# If Meta publishes more later, add the code here to pick it up.
# "cpu" to match the no-GPU demo machine exactly — see TRANSLATION_DEVICE
# above for why MPS was disabled here too (shared unified-memory pressure).
MMS_TTS_DEVICE = "cpu"                    # "mps" for Apple Silicon GPU, "cuda" if you have an NVIDIA GPU
MMS_TTS_LANGS = {"asm", "ben", "guj", "hin", "mai", "mal", "mar", "pan", "tam", "tel"}

# ---------- Stage 5 — Archive & Reuse ----------
ARCHIVE_DB_PATH = os.path.join(DATA_DIR, "translation_memory.db")  # same DB, different tables

# ---------- Server ----------
HOST = "127.0.0.1"
PORT = 5000