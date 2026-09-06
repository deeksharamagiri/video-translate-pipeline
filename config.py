"""
config.py — single source of truth for the whole pipeline.
Every limit / model name here maps 1:1 to a box in the architecture slide.
"""
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# A real system ffmpeg (apt/brew/etc.), if one is already on PATH before we
# add static-ffmpeg's own bundled binaries below, is far more likely to
# link a modern libass than the specific static-ffmpeg pip package build
# this project pins -- see the libass 0.15.2 note further down. Captured
# before add_paths() runs, since that prepends the bundled binaries' own
# directory onto PATH and would otherwise shadow a perfectly good system
# ffmpeg that was already there.
_preexisting_system_ffmpeg = shutil.which("ffmpeg")
_preexisting_system_ffprobe = shutil.which("ffprobe")

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

# ---------------------------------------------------------------------------
# FFmpeg binary selection. The auto-provisioned static-ffmpeg binary above
# bundles libass 0.15.2, which has a confirmed rendering bug: complex-script
# text shaping (Devanagari, Thai, Arabic conjuncts) falls back to a "simple"
# shaper that doesn't reorder vowel signs -- burned-in Hindi/Marathi/etc.
# subtitles come out with matras in the wrong position even though the
# underlying SRT/VTT text is correct (verified by direct rendering test;
# fixed upstream in libass 0.17+).
#
# Set FFMPEG_BINARY / FFPROBE_BINARY to a build with libass 0.17+ to fix
# this. The Docker image (see Dockerfile) sets these to the distro's
# packaged ffmpeg -- verified directly that Debian bookworm's ffmpeg links
# libass 0.17.1. Left unset, this falls back to whatever "ffmpeg"/"ffprobe"
# resolve to on PATH (the static-ffmpeg-provisioned binary, if nothing
# better is on PATH already).
# ---------------------------------------------------------------------------


def _ffmpeg_has_subtitles_filter(ffmpeg_path: str) -> bool:
    # A system ffmpeg being present on PATH is not by itself proof it can
    # burn subtitles -- some distro/Homebrew builds ship with libass
    # entirely compiled out (verified directly: a Homebrew ffmpeg 9.0.1
    # build on this machine has NO "subtitles" entry in `ffmpeg -filters`
    # at all and fails every burn-in job outright with "No option name",
    # which is a strictly worse outcome than static-ffmpeg's merely
    # imperfect libass 0.15.2). Actually querying the candidate binary's
    # own registered filters is the only reliable way to know.
    import subprocess
    try:
        proc = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return " subtitles " in proc.stdout or "\nsubtitles " in proc.stdout
    except Exception:
        return False


def _homebrew_ffmpeg_full_candidate(binary_name: str):
    # On macOS, Homebrew's plain `ffmpeg` formula does NOT link libass at
    # all (confirmed via its own formula: libass isn't in its dependency
    # list) -- subtitle-filter support only ships in the separate
    # `ffmpeg-full` formula, which pulls in libass 0.17+ but is
    # deliberately "keg-only" (installed, but not symlinked onto PATH) so
    # it doesn't collide with the plain `ffmpeg` formula's binaries. That
    # means a machine can have a fully working, modern-libass ffmpeg
    # installed and this pipeline would still never find it via PATH
    # alone. Ask brew directly for that keg's location rather than
    # requiring an operator to set FFMPEG_BINARY by hand.
    import subprocess
    try:
        proc = subprocess.run(
            ["brew", "--prefix", "ffmpeg-full"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        candidate = os.path.join(proc.stdout.strip(), "bin", binary_name)
        return candidate if os.path.exists(candidate) else None
    except Exception:
        # `brew` not installed/on PATH (e.g. Linux/Docker, where
        # FFMPEG_BINARY is set explicitly anyway) -- not an error.
        return None


def _resolve_ffmpeg_binary(env_var: str, preexisting_system_binary, binary_name: str, fallback: str) -> str:
    # Explicit env var always wins -- assume the operator verified it.
    # Otherwise, try candidates in order of "most likely to already be a
    # good, already-installed ffmpeg" and take the first one that
    # actually supports burning in subtitles at all; a candidate without
    # libass compiled in would make every burn-in job fail outright,
    # which is worse than static-ffmpeg's known-imperfect-but-functional
    # libass 0.15.2. Falls back to the bundled static-ffmpeg build if
    # nothing better checks out.
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value

    for candidate in (
        preexisting_system_binary,
        _homebrew_ffmpeg_full_candidate(binary_name),
    ):
        if candidate and _ffmpeg_has_subtitles_filter(candidate):
            return candidate

    return fallback


FFMPEG_BINARY = _resolve_ffmpeg_binary("FFMPEG_BINARY", _preexisting_system_ffmpeg, "ffmpeg", "ffmpeg")
FFPROBE_BINARY = _resolve_ffmpeg_binary("FFPROBE_BINARY", _preexisting_system_ffprobe, "ffprobe", "ffprobe")
if FFMPEG_BINARY == "ffmpeg":
    print("[config] Using the auto-provisioned static-ffmpeg binary -- its bundled "
          "libass (0.15.2) mis-renders complex scripts (Devanagari/Thai/Arabic) in "
          "burned-in subtitles (matras may be misplaced). No system ffmpeg with a "
          "working subtitles filter was found on PATH to prefer instead. Install an "
          "ffmpeg build with libass 0.17+ (e.g. `brew install ffmpeg` / "
          "`apt install ffmpeg`, confirming `ffmpeg -filters | grep subtitles` is "
          "non-empty) and it will be preferred automatically, or set "
          "FFMPEG_BINARY/FFPROBE_BINARY explicitly.")
else:
    print(f"[config] Using ffmpeg with a working subtitles filter (verified via "
          f"`-filters`) for burned-in subtitles: {FFMPEG_BINARY} -- not the "
          f"static-ffmpeg build known to mis-shape Devanagari/Thai/Arabic. Its "
          f"libass version was not independently confirmed here; verify with "
          f"`{FFMPEG_BINARY} -version` if unsure.")

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

# ---------- Stage 2 — ASR (whisper.cpp) ----------
# Previously used faster-whisper's "small" model (ctranslate2, int8, CPU).
# Verified directly against real job audio that this was the dominant cause
# of "nonsensical translation" complaints: on a real segment, faster-whisper
# "small" misdetected the language as Sinhala (35% confidence) and produced
# pure garbage, while whisper.cpp's quantized medium model transcribed the
# identical audio as a clean, correct Marathi sentence. faster-whisper
# "small" also used vad_filter=True (Silero VAD), which was verified to
# silently drop multi-minute stretches of real, audible speech from the
# transcript entirely -- confirmed by re-transcribing a "gap" span directly
# from source audio and finding coherent speech the pipeline had produced
# zero output for.
#
# Plain faster-whisper "medium" (ctranslate2, int8) was tried as a fix and
# reliably got OOM-killed on 8GB RAM. whisper.cpp's quantized medium model
# (ggml-medium-q5_0, ~514MB on disk) runs the same accuracy tier in a much
# smaller memory footprint -- verified stable on the same 8GB machine -- and
# has no separate VAD gate by default, so it doesn't drop real speech the
# way vad_filter=True did.
WHISPER_CPP_BINARY = os.environ.get("WHISPER_CPP_BINARY", "whisper-cli")
WHISPER_CPP_MODEL_DIR = os.path.join(MODELS_DIR, "whisper_cpp")
WHISPER_CPP_MODEL_FILENAME = "ggml-medium-q5_0.bin"
WHISPER_CPP_MODEL_PATH = os.path.join(WHISPER_CPP_MODEL_DIR, WHISPER_CPP_MODEL_FILENAME)
WHISPER_CPP_MODEL_URL = (
    f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{WHISPER_CPP_MODEL_FILENAME}"
)

# whisper.cpp's own `-l` flag takes its language codes, which are mostly
# ISO-639-1 (2-letter) -- e.g. "mr", not our internal 3-letter "mar". Every
# other caller of a 3-letter code in this codebase (source_lang_hint from
# the UI/API, INDIC_LANGS, etc.) needs translating through this map before
# it reaches whisper.cpp, or whisper-cli silently treats the value as
# unrecognised, prints its own --help text, and exits 0 with zero
# transcribed segments -- no error, just an empty transcript.
#
# Verified empirically against whisper-cli directly (each candidate code
# passed via `-l <code>` on real audio): only 14 of our 22 INDIC_LANGS have
# a whisper.cpp checkpoint at all. The other 8 (Bodo, Dogri, Kashmiri,
# Konkani, Maithili, Manipuri, Odia, Santali) have no whisper.cpp language
# support whatsoever, mapped or not -- those fall back to auto-detect with
# a logged warning rather than being passed through as an invalid code.
WHISPER_LANG_MAP = {
    "asm": "as", "ben": "bn", "guj": "gu", "hin": "hi", "kan": "kn",
    "mal": "ml", "mar": "mr", "nep": "ne", "pan": "pa", "san": "sa",
    "snd": "sd", "tam": "ta", "tel": "te", "urd": "ur", "eng": "en",
}

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
# the ASR stage, which runs whisper.cpp as a separate quantized-model
# subprocess, and not the optional IndicConformer ASR model, and not
# Indic-TTS which runs in its own separate venv/process entirely). Default
# on; flip off to isolate a quality regression if one is ever suspected.
QUANTIZE_TRANSLATION_MODELS = True

# ---------- CPU speedup: translation decoding width ----------
# generate()'s num_beams for both IndicTrans2 and NLLB. Same trade-off
# already made for whisper.cpp in stage2_asr.py ("greedy decoding gives a
# much better speed/quality trade-off" for offline field use) -- applied
# here too. Measured directly on this machine (CPU, real cached NLLB
# checkpoint, 8-segment batch of representative field-style sentences):
#
#     num_beams=5 (previous default): 23.1s
#     num_beams=1 (greedy):            7.8s   -- ~3x faster, identical output
#
# Override per-run without touching code if a specific job's quality looks
# worse and you want to compare against wider beam search:
#
#     TRANSLATION_NUM_BEAMS=5 python app.py
#
# Included in model_version_tag() below so the translation-memory cache
# key changes when this changes -- a cached translation made at one beam
# width is never silently served as if made at another.
TRANSLATION_NUM_BEAMS = int(
    os.environ.get(
        "TRANSLATION_NUM_BEAMS",
        "1",
    )
)

# ---------- Memory bound: resident IndicTrans2 checkpoints ----------
# pipeline/translate.py caches each loaded IndicTrans2 checkpoint
# (en_indic / indic_en / indic_indic -- up to 3 distinct models, each
# several hundred MB to ~1GB+ depending on INDICTRANS2_MODEL_SIZE) in a
# module-level dict, keyed by model name, with no eviction. A long-running
# deployment (this process staying up across many jobs -- the actual
# field-use pattern, not a one-shot script) that happens to serve more
# than one direction over its lifetime would accumulate all of them
# resident at once, on top of whatever NLLB/whisper.cpp/Indic-TTS are
# using at the time. This bound exists so that ceiling is a config value
# to tune, not a silent accumulation to discover via an OOM kill in the
# field. 1 means only the checkpoint the most recent job actually needed
# stays loaded; raise it if a deployment's RAM headroom and job mix (e.g.
# frequently alternating target languages that hit different
# checkpoints) make reload cost a worse trade-off than the extra memory.
INDICTRANS2_MAX_RESIDENT_MODELS = int(
    os.environ.get(
        "INDICTRANS2_MAX_RESIDENT_MODELS",
        "1",
    )
)

# ---------- Memory bound: sharing one budget across IndicTrans2 + NLLB ----------
# The bound above only governs IndicTrans2-vs-IndicTrans2 (e.g. an
# en_indic checkpoint evicting to make room for indic_indic). It does
# nothing about IndicTrans2 vs. NLLB, which are two entirely separate
# caches in pipeline/translate.py -- a session that runs an Indic-pair
# job (loading IndicTrans2) and then a non-Indic-pair job (loading NLLB)
# would keep BOTH resident, even though no single translation call ever
# needs both at once (routing picks exactly one engine per language
# pair). Verified directly under a real 8GB memory limit (Docker,
# --memory=8g): with INDICTRANS2_MAX_RESIDENT_MODELS=1 already active
# and correctly bounding the IndicTrans2 side, loading NLLB on top of an
# already-resident IndicTrans2 checkpoint (~1.8GB RSS at that point)
# still triggered an OOM kill (exit code 137) during NLLB's load/
# quantization step. Default on: evicts whichever engine's cache isn't
# the one about to be used. Turn off only for a deployment with enough
# headroom that avoiding reload cost across a mixed workload is worth
# more than the memory -- untested above 8GB, no specific number to
# recommend yet.
TRANSLATION_SHARE_MEMORY_ACROSS_ENGINES = (
    os.environ.get(
        "TRANSLATION_SHARE_MEMORY_ACROSS_ENGINES",
        "true",
    ).lower()
    not in ("false", "0", "")
)

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
# ffmpeg's atempo filter before stitching.
VOICEOVER_TIME_STRETCH = True
# Segment slot durations come from ASR/VAD timing, which is often noisy (a
# handful of words can get a 20+ second slot right next to a full sentence
# squeezed into 1-2 seconds). Fitting every segment's TTS audio exactly to
# its slot -- the old default was 2.5, i.e. anywhere from 0.4x to 2.5x speed
# -- produces wildly inconsistent pacing from one segment to the next
# ("randomly slow and randomly quick" playback). This is the *soft* bound
# used for normal segments: +/-20%, close to natural human pacing variance.
VOICEOVER_MAX_TEMPO_RATIO = 1.2   # clamp; within native atempo range (0.5-2.0), no chaining needed

# Letting every segment drift from its slot without limit (the previous
# approach: place strictly sequentially, one after another, never checking
# against original timing) was verified against a real job to cause several
# seconds of audio/video desync within any run of tightly-packed segments --
# reported as "the screen moves onto another frame but the output audio is
# still of the previous frame". VOICEOVER_MAX_DRIFT_SEC bounds this: once a
# segment's earliest possible start (after the previous segment) has
# drifted more than this far from its own original subtitle start, that
# segment gets compressed harder -- up to VOICEOVER_CATCHUP_MAX_TEMPO_RATIO
# -- specifically to pull the timeline back within budget, instead of
# letting drift keep compounding.
#
# VOICEOVER_CATCHUP_MAX_TEMPO_RATIO was briefly narrowed from 1.6 to 1.3 to
# make catch-up bursts sound less jarring. That was a mistake: this ratio
# isn't just a naturalness knob, it's what makes the VOICEOVER_MAX_DRIFT_SEC
# guarantee actually hold -- when a segment needs more compression than the
# ceiling allows to stay in budget, the code intentionally lets it exceed
# the drift cap rather than distort the audio further (see the "physical
# intelligibility floor" comment in delivery.py). Weakening the ceiling to
# 1.3 therefore weakened the sync guarantee itself, not just pacing --
# verified on a real job: max drift went from ~1.5s to 13.7s. Reverted to
# 1.6 (the value actually validated to keep drift bounded). The zero-
# duration-segment bug in stage2_asr.py that was injecting most of the
# drift bursts in the first place is now fixed at its source, which should
# also mean the 1.6 ceiling gets invoked less often than before -- i.e.
# smoother pacing AND bounded sync, rather than trading one for the other.
#
# That held for the systemic (12-occurrence) zero-duration case, but a real
# job still showed 6.35s of drift afterward -- traced to a different,
# rarer failure mode: whisper.cpp assigning a segment a slot wildly too
# short for its actual content (one case: a 128-char sentence, ~10.8s of
# real TTS speech, given only a 0.84s window) with almost no room before
# the next segment's own start -- not enough runway for any tempo
# compression, however aggressive, to fully absorb. stage2_asr.py now
# extends such segments' timing where there's room to (capped at the next
# segment's start, never creating an overlap), which helps the common
# case; for the genuinely pathological case above there wasn't enough room
# regardless. Raised the ceiling to 2.0 -- the native atempo range's own
# limit (see _build_atempo_chain), so this still never triggers filter
# chaining/its extra quality loss -- and tightened the drift budget to 1.0s,
# which together brought that same job's worst-case drift down from 6.35s
# to ~4.55s. Not a full elimination of every possible case (that would need
# smarter multi-segment lookahead scheduling, a bigger change), but a large,
# measured improvement, and the common/systemic causes are now gone.
VOICEOVER_MAX_DRIFT_SEC = 1.0
VOICEOVER_CATCHUP_MAX_TEMPO_RATIO = 2.0   # only used while actively catching up drift; native atempo ceiling

# ---------- Stage 4 — Subtitle Generation ----------
MAX_CHARS_PER_LINE = 42
MAX_LINES_PER_SUBTITLE = 2
MIN_GAP_BETWEEN_SUBTITLES_SEC = 0.08

# ---------- CPU speedup: burned-in video re-encode ----------
# burn_in_subtitles() (delivery.py) has to re-encode video (subtitle
# burn-in can't be a stream copy) -- ffmpeg's libx264 default preset is
# "medium". Measured on this machine: "veryfast" cut encode time by ~35%
# for a comparable output size/bitrate, with no visible quality difference
# at default CRF. Only affects the burned-in-subtitles output; voiceover
# muxing already uses "-c:v copy" (no re-encode) and is unaffected.
BURN_IN_ENCODE_PRESET = os.environ.get("BURN_IN_ENCODE_PRESET", "veryfast")

# ---------- Stage 5 — Archive & Reuse ----------
ARCHIVE_DB_PATH = os.path.join(DATA_DIR, "translation_memory.db")  # same DB, different tables

# ---------- Server ----------
# Defaults to loopback-only, matching HANDOVER.md's "no auth layer, don't
# expose this" stance for a normal local install. Override via env var --
# needed inside Docker specifically: 127.0.0.1 *inside* a container is the
# container's own loopback, unreachable from the host even with `-p`
# published, so the container image sets HOST=0.0.0.0 explicitly (see
# Dockerfile) rather than changing this default for everyone.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))