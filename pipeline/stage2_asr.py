"""
Stage 2 — ASR (whisper.cpp)

Pipeline:

    Audio -> whisper.cpp -> timestamped transcript

Features:
    - Uses whisper.cpp through subprocess
    - Supports confirmed language hints such as "mr"
    - Uses GPU acceleration when the whisper.cpp binary was built with it
      (e.g. Metal on macOS); CPU-only otherwise -- see WHISPER_NO_GPU
    - Optional VAD
    - JSON + full JSON output
    - Per-token confidence calculation
    - Hallucination/repetition filtering
    - Timestamp correction for badly undertimed segments
    - Optional IndicConformer refinement

The project intentionally uses whisper.cpp rather than faster-whisper
because whisper.cpp's quantized models are substantially easier to run
within the available memory constraints.

whisper.cpp 1.9.2, ggml-medium-q5_0.bin. Verified working on both macOS
(Apple Silicon, Metal-accelerated) for local dev and Debian-based Linux
(CPU-only build, no GPU passthrough) via the Docker image -- see
DOCKER.md.
"""

import json
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import time
import urllib.request
import wave

from dataclasses import dataclass, field
from typing import List, Optional


from config import (
    WHISPER_CPP_BINARY,
    WHISPER_CPP_MODEL_DIR,
    WHISPER_CPP_MODEL_PATH,
    WHISPER_CPP_MODEL_URL,
    WHISPER_LANG_MAP,
    INDIC_CONFORMER_MODEL,
    INDIC_CONFORMER_DECODING,
    INDIC_CONFORMER_LANG_MAP,
)

from pipeline.cancellation import check_cancelled, run_cancellable


# ============================================================
# Caches
# ============================================================

_conformer_cache = {}

_model_ensured = False


# ============================================================
# Configuration
# ============================================================

# Thread count for whisper.cpp's CPU-side work (always used; on a build
# with GPU acceleration compiled in, e.g. Metal, the neural-network work
# itself mostly runs on the GPU instead, but the Docker image's build is
# CPU-only -- see WHISPER_NO_GPU below and DOCKER.md).
#
# 8 is a reasonable default for a machine with around that many cores.
# This can be overridden through the environment without modifying code:
#
#     WHISPER_THREADS=6
#
# or:
#
#     WHISPER_THREADS=10
#
WHISPER_THREADS = int(
    os.environ.get(
        "WHISPER_THREADS",
        "8",
    )
)


# Number of whisper.cpp processors.
#
# Keep this at 1. Increasing this can duplicate work and increase
# memory pressure rather than improving latency.
WHISPER_PROCESSORS = int(
    os.environ.get(
        "WHISPER_PROCESSORS",
        "1",
    )
)


# Greedy decoding (best-of=1, beam=1) is considerably cheaper than beam
# search, but was found to directly cause catastrophic hallucination on
# real field audio -- not just lower-quality output.
#
# Verified directly: a genuinely clear, well-recorded Marathi clip
# (BAIF field video "401.6", ~7 min) decoded as 100% repeated-token
# garbage end-to-end at best-of=1/beam=1, with the correct language
# forced explicitly (so this wasn't a language-detection problem). The
# *identical* audio, same model, same thresholds, transcribed cleanly
# throughout at best-of=5/beam=5. A worse video-wide hallucination rate
# at beam=1 was also measured across a batch of 8 real field videos
# (21-100% of each video's audio dropped as hallucinated repetition).
#
# whisper.cpp's own CLI default is 5/5; we now match that rather than
# overriding it down to 1, since the speed win isn't worth silent total
# job failures. Still overridable via env if a deployment needs the
# older speed/quality trade-off and has verified its own audio is clean
# enough to tolerate it:
#
#     WHISPER_BEST_OF=1 WHISPER_BEAM_SIZE=1 python app.py
#
WHISPER_BEST_OF = int(
    os.environ.get(
        "WHISPER_BEST_OF",
        "5",
    )
)

WHISPER_BEAM_SIZE = int(
    os.environ.get(
        "WHISPER_BEAM_SIZE",
        "5",
    )
)


# Conservative hallucination thresholds.
#
# The previous implementation used:
#
#     no-speech = 0.20
#     entropy   = 3.0
#
# Those settings make whisper.cpp more willing to decode ambiguous
# audio, which can contribute to long repetition loops.
WHISPER_NO_SPEECH_THRESHOLD = float(
    os.environ.get(
        "WHISPER_NO_SPEECH_THRESHOLD",
        "0.60",
    )
)

WHISPER_ENTROPY_THRESHOLD = float(
    os.environ.get(
        "WHISPER_ENTROPY_THRESHOLD",
        "2.40",
    )
)

WHISPER_LOGPROB_THRESHOLD = float(
    os.environ.get(
        "WHISPER_LOGPROB_THRESHOLD",
        "-1.0",
    )
)


# Temperature fallback.
#
# Default is False for speed.
#
# If quality drops on difficult speech, set:
#
#     WHISPER_NO_FALLBACK=0
#
# in the environment.
WHISPER_NO_FALLBACK = os.environ.get(
    "WHISPER_NO_FALLBACK",
    "1",
).lower() not in {
    "0",
    "false",
    "no",
}


# Flash attention is available in the user's whisper.cpp 1.9.2
# installation and is enabled by default there.
WHISPER_FLASH_ATTENTION = os.environ.get(
    "WHISPER_FLASH_ATTENTION",
    "1",
).lower() not in {
    "0",
    "false",
    "no",
}


# Suppress non-speech tokens.
WHISPER_SUPPRESS_NON_SPEECH = os.environ.get(
    "WHISPER_SUPPRESS_NON_SPEECH",
    "1",
).lower() not in {
    "0",
    "false",
    "no",
}


# Optional VAD.
#
# Disabled by default because your project previously experienced
# missing multi-minute speech when VAD was used incorrectly.
#
# We keep the feature available but do NOT enable it automatically.
WHISPER_ENABLE_VAD = os.environ.get(
    "WHISPER_ENABLE_VAD",
    "0",
).lower() in {
    "1",
    "true",
    "yes",
}


# Force whisper.cpp onto CPU only (disables Apple Metal).
#
# Useful for isolating how much of a run's time is GPU-bound, e.g.
# to A/B against a normal (Metal) run on the same clip:
#
#     WHISPER_NO_GPU=1 python app.py
#
# Off by default -- Metal is used automatically when available.
WHISPER_NO_GPU = os.environ.get(
    "WHISPER_NO_GPU",
    "0",
).lower() in {
    "1",
    "true",
    "yes",
}


# ============================================================
# Data classes
# ============================================================

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float  # 0..1


@dataclass
class ASRResult:
    detected_language: str
    language_probability: float
    segments: List[TranscriptSegment] = field(
        default_factory=list
    )
    # (start, end) ranges whisper.cpp produced but which were dropped as
    # hallucinated repetition -- surfaced so a job with real coverage
    # gaps says so, instead of silently having no subtitles/translation
    # for that stretch with no indication why (previously this only ever
    # reached a print() statement, never the caller).
    dropped_ranges: List[tuple] = field(
        default_factory=list
    )


# ============================================================
# Model download
# ============================================================

def _download_whispercpp_model(
    url: str,
    dest_path: str,
    max_retries: int = 5,
):
    """
    Download whisper.cpp model with retries.

    The file is first written to .part and atomically renamed after
    successful completion so a partially downloaded model is never
    mistaken for a valid model.
    """

    try:
        import certifi

        ctx = ssl.create_default_context(
            cafile=certifi.where()
        )

    except ImportError:

        ctx = ssl.create_default_context()

    tmp_path = dest_path + ".part"

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            print(
                "[stage2_asr] Downloading whisper.cpp model "
                f"(attempt {attempt}/{max_retries}): {url}"
            )

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent":
                        "Offline-Field-Translator/1.0"
                },
            )

            with urllib.request.urlopen(
                request,
                context=ctx,
                timeout=120,
            ) as resp:

                with open(
                    tmp_path,
                    "wb",
                ) as out:

                    while True:

                        chunk = resp.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        out.write(chunk)

            if (
                not os.path.exists(tmp_path)
                or os.path.getsize(tmp_path) == 0
            ):
                raise RuntimeError(
                    "Downloaded file is empty."
                )

            os.replace(
                tmp_path,
                dest_path,
            )

            print(
                "[stage2_asr]   -> download complete"
            )

            return

        except Exception as exc:

            print(
                "[stage2_asr] Download attempt "
                f"{attempt} failed: {exc}"
            )

            if os.path.exists(tmp_path):

                try:
                    os.remove(tmp_path)

                except OSError:
                    pass

            if attempt == max_retries:

                raise RuntimeError(
                    "Failed to download whisper.cpp model "
                    f"after {max_retries} attempts: {url}"
                ) from exc

            time.sleep(
                min(
                    2 ** attempt,
                    30,
                )
            )


def _ensure_whispercpp_model() -> str:
    """
    Ensure the configured whisper.cpp model exists locally.
    """

    global _model_ensured

    if (
        _model_ensured
        and os.path.exists(
            WHISPER_CPP_MODEL_PATH
        )
    ):
        return WHISPER_CPP_MODEL_PATH

    if (
        os.path.exists(
            WHISPER_CPP_MODEL_PATH
        )
        and os.path.getsize(
            WHISPER_CPP_MODEL_PATH
        ) > 0
    ):

        _model_ensured = True

        return WHISPER_CPP_MODEL_PATH

    if shutil.which(
        WHISPER_CPP_BINARY
    ) is None:

        raise RuntimeError(
            f"whisper.cpp binary "
            f"'{WHISPER_CPP_BINARY}' not found on PATH. "
            "Build/install whisper.cpp (e.g. https://github.com/ggml-org/whisper.cpp) "
            "or set WHISPER_CPP_BINARY to its full path."
        )

    os.makedirs(
        WHISPER_CPP_MODEL_DIR,
        exist_ok=True,
    )

    _download_whispercpp_model(
        WHISPER_CPP_MODEL_URL,
        WHISPER_CPP_MODEL_PATH,
    )

    _model_ensured = True

    return WHISPER_CPP_MODEL_PATH


# ============================================================
# Hallucination / repetition detection
# ============================================================

# Unicode block ranges for the scripts our supported languages use.
_SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F),  # Hindi, Marathi, Nepali, Sanskrit, Bodo, Dogri, Konkani, Maithili
    "Bengali": (0x0980, 0x09FF),     # Bengali, Assamese
    "Gurmukhi": (0x0A00, 0x0A7F),    # Punjabi
    "Gujarati": (0x0A80, 0x0AFF),
    "Oriya": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F),
}

# Maps every language code that could show up as whisper.cpp's own
# detected_language (its native 2-letter codes) or as our internal
# 3-letter code (the language_hint fallback in run_stage2) to the script
# its speech should be written in.
_LANG_EXPECTED_SCRIPT = {
    "hi": "Devanagari", "hin": "Devanagari",
    "mr": "Devanagari", "mar": "Devanagari",
    "ne": "Devanagari", "nep": "Devanagari",
    "sa": "Devanagari", "san": "Devanagari",
    "bn": "Bengali", "ben": "Bengali",
    "as": "Bengali", "asm": "Bengali",
    "pa": "Gurmukhi", "pan": "Gurmukhi",
    "gu": "Gujarati", "guj": "Gujarati",
    "or": "Oriya", "ori": "Oriya",
    "ta": "Tamil", "tam": "Tamil",
    "te": "Telugu", "tel": "Telugu",
    "kn": "Kannada", "kan": "Kannada",
    "ml": "Malayalam", "mal": "Malayalam",
}

# A segment needs at least this many script-block characters before its
# script mix is trusted as evidence either way -- a two-word segment is
# too short to judge reliably and risks false positives.
_MIN_SCRIPT_CHARS_TO_JUDGE = 8


def _is_wrong_script(text: str, expected_language: str) -> bool:
    """
    Detect decoded text written in an Indic script that doesn't match the
    audio's own (detected or hinted) language -- a hallucination pattern
    distinct from _is_degenerate_repetition's mechanical token/character
    loops.

    Verified directly on real field audio ("401.1.mp4", Marathi/Devanagari
    speech): whisper.cpp's beam=5 decode confidently produced fluent,
    non-repeating text in entirely unrelated scripts for large stretches
    -- "ব ব ব ব..." (Bengali) in one run, "਷ਿਲਿਪਾਲਨ ਷ਿਲਿਪਾਲਨ..." (Gurmukhi)
    in another, same audio, same settings, just different runs (whisper.cpp's
    decode has run-to-run variance). Because this text doesn't loop the
    same short token/phrase, _is_degenerate_repetition doesn't catch it --
    it silently passed through as "real" output, vanishing from the final
    subtitles without ever being counted in dropped_ranges. A job with
    ~80% of its audio affected this way still reported "finished ok" with
    no indication anything was wrong -- reported as "not producing any
    output at all". This doesn't recover the correct transcription for
    that stretch (whisper.cpp simply couldn't decode it right), but it
    makes the gap visible through the same dropped-ranges warning as any
    other unrecoverable stretch, instead of a silently near-empty job.
    """

    expected_script = _LANG_EXPECTED_SCRIPT.get(expected_language)

    if expected_script is None:
        # Unknown/unmapped language (e.g. non-Indic, or a script this
        # check doesn't cover) -- nothing to compare against.
        return False

    expected_range = _SCRIPT_RANGES[expected_script]

    matching_chars = 0
    other_script_chars = 0

    for ch in text:

        codepoint = ord(ch)

        for script_name, (lo, hi) in _SCRIPT_RANGES.items():

            if lo <= codepoint <= hi:

                if script_name == expected_script:
                    matching_chars += 1
                else:
                    other_script_chars += 1

                break

    total = matching_chars + other_script_chars

    if total < _MIN_SCRIPT_CHARS_TO_JUDGE:
        return False

    return (other_script_chars / total) > 0.5


def _is_degenerate_repetition(text: str) -> bool:
    """
    Detect obvious Whisper repetition loops.

    Examples of problematic output:

        बबबबबबबबबबबबब...
        abc abc abc abc abc...
        काही शब्द काही शब्द काही शब्द...

    The test intentionally focuses on strong repetition rather than
    ordinary repeated words in legitimate speech.
    """

    text = text.strip()

    if len(text) < 12:
        return False

    # --------------------------------------------------------
    # Character-level repetition
    # --------------------------------------------------------

    collapsed = re.sub(
        r"(.{1,12}?)\1{2,}",
        r"\1",
        text,
    )

    if len(collapsed) < len(text) * 0.50:
        return True

    # --------------------------------------------------------
    # Word-level repetition
    # --------------------------------------------------------

    words = text.split()

    if len(words) >= 6:

        # Check last 2/3/4 word patterns.
        for pattern_size in (
            1,
            2,
            3,
            4,
        ):

            if len(words) < pattern_size * 3:
                continue

            pattern = words[
                -pattern_size:
            ]

            repetitions = 0

            pos = len(words)

            while (
                pos >= pattern_size
                and words[
                    pos - pattern_size:pos
                ] == pattern
            ):

                repetitions += 1
                pos -= pattern_size

            if repetitions >= 3:

                repeated_words = (
                    repetitions
                    * pattern_size
                )

                if (
                    repeated_words
                    >= len(words) * 0.50
                ):
                    return True

    return False


def _drop_consecutive_duplicate_segments(
    segments: list,
    dropped_ranges: list,
) -> int:
    """
    Drop segments whose text exactly repeats the immediately preceding
    segment's text -- a distinct hallucination pattern from
    _is_degenerate_repetition (which only catches repetition *within* one
    segment's own text, e.g. one word or short phrase looping).

    Verified directly against a real job: whisper.cpp transcribed the
    same ~24s stretch of coherent-looking text identically across 3
    consecutive decode windows in a row (158-182s, 182-205s, 205-230s).
    Each segment's own text doesn't internally repeat, so the existing
    filter doesn't flag any of them -- they survive as several
    individually-plausible segments. Left alone, a later display-layer
    dedup pass (stage4_subtitle.py's duplicate-caption merge, meant for
    genuine short split-utterance artifacts) would then merge all of
    them into one nonsensical 72-second caption. Handled the same way as
    any other hallucination instead: all but the first occurrence are
    dropped and counted in dropped_ranges, so it's surfaced through the
    same "N seconds of audio could not be transcribed reliably" warning
    as any other unrecoverable stretch, rather than silently producing a
    misleading giant caption.

    Mutates `segments` in place (removes items) and appends to
    `dropped_ranges`. Returns the number of segments dropped.
    """

    dropped = 0
    i = 1

    while i < len(segments):

        prev_text = segments[i - 1].text.strip()
        cur = segments[i]

        if cur.text.strip() and cur.text.strip() == prev_text:

            dropped_ranges.append(
                (cur.start, cur.end)
            )

            print(
                "[stage2_asr] Dropping segment-level "
                f"repeated transcription [{cur.start:.2f}-"
                f"{cur.end:.2f}]: {cur.text[:120]!r}"
            )

            del segments[i]

            dropped += 1

        else:

            i += 1

    return dropped


# ============================================================
# Confidence
# ============================================================

def _segment_confidence(
    tokens: list,
) -> float:
    """
    Calculate average per-token probability.

    Special tokens such as:

        [_BEG_]
        [_TT_123]

    are excluded.
    """

    probs = []

    for token in tokens:

        token_text = str(
            token.get(
                "text",
                "",
            )
        ).strip()

        if token_text.startswith("["):
            continue

        probability = token.get(
            "p",
            None,
        )

        if probability is None:
            continue

        try:
            probability = float(
                probability
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if 0.0 <= probability <= 1.0:

            probs.append(
                probability
            )

    if not probs:
        return 0.5

    return round(
        sum(probs) / len(probs),
        3,
    )


# Minimum gap between two consecutive tokens' own timestamps, in seconds,
# treated as a natural pause worth splitting a caption on. whisper.cpp has
# no VAD and decodes in ~30s windows regardless of pauses (that's
# deliberate -- see WHISPER_BEST_OF/WHISPER_BEAM_SIZE's comment on why
# VAD-based filtering was tried and reverted here), so a single raw
# segment can span a full window as one caption even when the speaker
# paused for several seconds partway through it.
SEGMENT_SPLIT_GAP_SEC = 0.7


def _split_on_internal_pauses(
    tokens: list,
    start: float,
    end: float,
    text: str,
    confidence: float,
):
    """
    Split one already-decoded, already-accepted segment into several
    smaller ones at internal pauses, using each token's own offsets.

    This is purely a re-chunking of text/timing that whisper.cpp already
    successfully produced -- nothing gets dropped or re-decoded, unlike
    VAD-based filtering (which runs *before* decoding and can misjudge
    real speech as silence). If a segment has no usable per-token
    offsets, or no internal gap large enough to split on, it's returned
    unchanged as a single piece -- this only ever produces equal or finer
    granularity, never coarser.

    Returns a list of (start, end, text, confidence) tuples.
    """

    fallback = [(start, end, text, confidence)]

    # A segment whose own declared end <= start is already the "badly
    # undertimed" case _extend_undertimed_segments (below) exists to fix
    # -- verified directly that such segments can carry token-level
    # offsets inconsistent with the segment's own boundaries (e.g. token
    # offsets referencing an earlier decode window entirely), which would
    # make gap-based splitting here produce nonsense (negative
    # durations, single-character fragments). Leave these to the
    # existing correction pass instead of guessing from unreliable
    # per-token data.
    if end <= start:
        return fallback

    words = [
        token
        for token in tokens
        if not str(token.get("text", "")).strip().startswith("[")
        and token.get("offsets", {}).get("from") is not None
        and token.get("offsets", {}).get("to") is not None
    ]

    if not words:
        return fallback

    groups = [[words[0]]]

    for prev_token, cur_token in zip(words, words[1:]):

        gap_sec = (
            cur_token["offsets"]["from"]
            - prev_token["offsets"]["to"]
        ) / 1000.0

        if gap_sec >= SEGMENT_SPLIT_GAP_SEC:
            groups.append([])

        groups[-1].append(cur_token)

    if len(groups) <= 1:
        return fallback

    pieces = []
    # Small slack for rounding between the segment's own declared bounds
    # and its tokens' individual offsets -- not zero tolerance, since
    # whisper.cpp's segment- and token-level timestamps don't always
    # agree to the millisecond even in the normal case.
    tolerance_sec = 1.0

    for group in groups:

        piece_text = "".join(
            str(t.get("text", "")) for t in group
        ).strip()

        if not piece_text:
            continue

        piece_start = group[0]["offsets"]["from"] / 1000.0
        piece_end = group[-1]["offsets"]["to"] / 1000.0

        # Sanity-check against the segment's own declared bounds -- if
        # the tokens' offsets disagree with them by more than a rounding
        # slack, the token-level data for this entry isn't trustworthy;
        # bail out to the single unsplit segment entirely rather than
        # split on bad data.
        #
        # Also bail out if a piece's own duration is too short for its
        # text length by the same 12-chars/sec heuristic
        # _extend_undertimed_segments (below) uses to detect "obviously
        # too short" timestamps. Verified directly against a real job:
        # whisper.cpp's per-token offsets can degrade partway through a
        # long segment, with every remaining token pinned to the exact
        # same timestamp (the segment's own end) instead of real
        # individual timing -- a piece built from a run of those tokens
        # gets a near-zero duration for real text, which
        # _extend_undertimed_segments would then "correct" by stretching
        # it forward to the start of whatever the next real segment
        # happens to be, however far away that is (one real case: a
        # ~5s phrase stretched into a 72-second caption). Splitting on
        # unreliable per-token data is worse than not splitting at all.
        piece_duration = piece_end - piece_start
        piece_estimated_min_duration = max(0.4, len(piece_text) / 12.0) * 0.5

        if (
            piece_end <= piece_start
            or piece_start < start - tolerance_sec
            or piece_end > end + tolerance_sec
            or piece_duration < piece_estimated_min_duration
        ):
            return fallback

        pieces.append((
            piece_start,
            piece_end,
            piece_text,
            _segment_confidence(group),
        ))

    return pieces if pieces else fallback


# ============================================================
# whisper.cpp command construction
# ============================================================

def _build_whisper_command(
    model_path: str,
    wav_path: str,
    out_prefix: str,
    language_hint: Optional[str],
) -> list:
    """
    Build the whisper.cpp command.

    Tuned for medium-q5_0 on Marathi / Indic speech; the exact options are
    compatible with whisper.cpp 1.9.2. GPU usage (e.g. Metal) depends
    entirely on whether the whisper-cli binary on this machine was built
    with it -- the Docker image's build has none compiled in, so it's
    always CPU-only there regardless of these flags.
    """

    cmd = [
        WHISPER_CPP_BINARY,

        "-m",
        model_path,

        "-f",
        wav_path,

        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        "-l",
        language_hint or "auto",

        # ----------------------------------------------------
        # Threading / compute
        # ----------------------------------------------------

        "-t",
        str(WHISPER_THREADS),

        "-p",
        str(WHISPER_PROCESSORS),

        # ----------------------------------------------------
        # Fast deterministic decoding
        # ----------------------------------------------------

        "-bo",
        str(WHISPER_BEST_OF),

        "-bs",
        str(WHISPER_BEAM_SIZE),

        # ----------------------------------------------------
        # Hallucination / fallback handling
        # ----------------------------------------------------

        "-nth",
        str(
            WHISPER_NO_SPEECH_THRESHOLD
        ),

        "-et",
        str(
            WHISPER_ENTROPY_THRESHOLD
        ),

        "-lpt",
        str(
            WHISPER_LOGPROB_THRESHOLD
        ),
    ]

    # --------------------------------------------------------
    # Flash attention
    # --------------------------------------------------------

    if WHISPER_FLASH_ATTENTION:
        cmd.append("-fa")

    # --------------------------------------------------------
    # CPU-only (disable Metal), for A/B timing tests.
    # --------------------------------------------------------

    if WHISPER_NO_GPU:
        cmd.append("-ng")

    # --------------------------------------------------------
    # Suppress non-speech tokens
    # --------------------------------------------------------

    if WHISPER_SUPPRESS_NON_SPEECH:
        cmd.append("-sns")

    # --------------------------------------------------------
    # Disable temperature fallback when requested.
    # --------------------------------------------------------

    if WHISPER_NO_FALLBACK:
        cmd.append("-nf")

    # --------------------------------------------------------
    # Optional VAD.
    #
    # Disabled by default.
    # --------------------------------------------------------

    if WHISPER_ENABLE_VAD:

        cmd.extend([
            "--vad",

            # Conservative VAD threshold.
            "--vad-threshold",
            "0.50",

            # Don't discard short speech.
            "--vad-min-speech-duration-ms",
            "250",

            # Small silence split.
            "--vad-min-silence-duration-ms",
            "500",

            # Allow reasonably long speech.
            "--vad-max-speech-duration-s",
            "30",

            # Preserve a little context around boundaries.
            "--vad-speech-pad-ms",
            "100",
        ])

    # --------------------------------------------------------
    # JSON output
    # --------------------------------------------------------

    cmd.extend([
        "-oj",
        "-ojf",

        "-of",
        out_prefix,

        "-np",
    ])

    return cmd


# ============================================================
# Main Stage 2
# ============================================================

def run_stage2(
    wav_path: str,
    language_hint: Optional[str] = None,
    cancel_event=None,
) -> ASRResult:
    """
    Run whisper.cpp on a 16kHz mono WAV file.

    language_hint:
        Whisper language code such as:

            mr
            hi
            en
            bn

        None / empty:
            auto-detect.
    """

    model_path = _ensure_whispercpp_model()

    # --------------------------------------------------------
    # Validate audio path
    # --------------------------------------------------------

    if not os.path.exists(wav_path):

        raise RuntimeError(
            f"Stage 2 audio file does not exist: "
            f"{wav_path}"
        )

    # --------------------------------------------------------
    # Validate whisper binary
    # --------------------------------------------------------

    if shutil.which(
        WHISPER_CPP_BINARY
    ) is None:

        raise RuntimeError(
            f"whisper.cpp binary "
            f"'{WHISPER_CPP_BINARY}' not found."
        )

    # --------------------------------------------------------
    # Normalise language_hint to whisper.cpp's own code
    # --------------------------------------------------------
    #
    # Callers elsewhere in this codebase (the web UI, run_job) pass our
    # internal 3-letter codes (e.g. "mar"), but whisper.cpp's `-l` flag
    # only understands its own codes (mostly 2-letter, e.g. "mr") -- an
    # unrecognised value doesn't error, it makes whisper-cli print its
    # own --help text and exit 0 with zero transcribed segments. Convert
    # here so every caller is protected, not just the ones that remember
    # to convert first.
    whisper_language_hint = language_hint

    if language_hint:

        mapped = WHISPER_LANG_MAP.get(language_hint)

        if mapped:

            whisper_language_hint = mapped

        elif language_hint not in WHISPER_LANG_MAP.values():

            # Neither a known 3-letter code we can map, nor already one
            # of whisper.cpp's own 2-letter codes -- most likely one of
            # the 8 INDIC_LANGS whisper.cpp has no checkpoint for at all
            # (Bodo, Dogri, Kashmiri, Konkani, Maithili, Manipuri, Odia,
            # Santali). Fall back to auto-detect rather than silently
            # producing zero segments.
            print(
                "[stage2_asr] Warning: whisper.cpp has no language "
                f"support for {language_hint!r} -- falling back to "
                "auto-detect."
            )

            whisper_language_hint = None

    # --------------------------------------------------------
    # Temporary output directory
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as tmp_dir:

        out_prefix = os.path.join(
            tmp_dir,
            "out",
        )

        cmd = _build_whisper_command(
            model_path=model_path,
            wav_path=wav_path,
            out_prefix=out_prefix,
            language_hint=whisper_language_hint,
        )

        print(
            "[stage2_asr] whisper.cpp configuration:"
        )

        print(
            f"[stage2_asr]   model: "
            f"{model_path}"
        )

        print(
            f"[stage2_asr]   language: "
            f"{whisper_language_hint or 'auto'}"
        )

        print(
            f"[stage2_asr]   threads: "
            f"{WHISPER_THREADS}"
        )

        print(
            f"[stage2_asr]   processors: "
            f"{WHISPER_PROCESSORS}"
        )

        print(
            f"[stage2_asr]   best-of: "
            f"{WHISPER_BEST_OF}"
        )

        print(
            f"[stage2_asr]   beam-size: "
            f"{WHISPER_BEAM_SIZE}"
        )

        print(
            f"[stage2_asr]   flash-attention: "
            f"{WHISPER_FLASH_ATTENTION}"
        )

        print(
            f"[stage2_asr]   no-speech threshold: "
            f"{WHISPER_NO_SPEECH_THRESHOLD}"
        )

        print(
            f"[stage2_asr]   entropy threshold: "
            f"{WHISPER_ENTROPY_THRESHOLD}"
        )

        print(
            f"[stage2_asr]   no-fallback: "
            f"{WHISPER_NO_FALLBACK}"
        )

        print(
            f"[stage2_asr]   VAD: "
            f"{WHISPER_ENABLE_VAD}"
        )

        print(
            "[stage2_asr] Running whisper.cpp..."
        )

        started = time.perf_counter()

        proc = run_cancellable(
            cmd,
            cancel_event,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f"[stage2_asr] whisper.cpp finished "
            f"in {elapsed:.2f}s"
        )

        if proc.returncode != 0:

            stderr_text = (
                proc.stderr
                .decode(
                    errors="ignore"
                )
            )

            stdout_text = (
                proc.stdout
                .decode(
                    errors="ignore"
                )
            )

            raise RuntimeError(
                "whisper.cpp transcription failed:\n\n"
                f"STDOUT:\n{stdout_text}\n\n"
                f"STDERR:\n{stderr_text}"
            )

        json_path = (
            out_prefix
            + ".json"
        )

        if not os.path.exists(
            json_path
        ):

            raise RuntimeError(
                "whisper.cpp completed successfully "
                "but did not produce the expected JSON file:\n"
                f"{json_path}"
            )

        # ----------------------------------------------------
        # Read JSON defensively.
        # ----------------------------------------------------

        with open(
            json_path,
            "rb",
        ) as f:

            raw_json = f.read()

        try:

            data = json.loads(
                raw_json.decode(
                    "utf-8"
                )
            )

        except UnicodeDecodeError:

            # A pathological repetition can occasionally result
            # in malformed UTF-8 at the end of the output.
            print(
                "[stage2_asr] Warning: "
                "JSON contains invalid UTF-8; "
                "decoding with replacement."
            )

            data = json.loads(
                raw_json.decode(
                    "utf-8",
                    errors="replace",
                )
            )

        except json.JSONDecodeError as exc:

            # Give the caller useful diagnostics rather than an
            # opaque JSON exception.
            preview = raw_json[
                :2000
            ].decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                "whisper.cpp produced invalid JSON.\n"
                f"JSON error: {exc}\n"
                f"Output preview:\n{preview}"
            ) from exc

    # ========================================================
    # Parse language
    # ========================================================

    detected_language = (
        data
        .get(
            "result",
            {}
        )
        .get(
            "language"
        )
        or language_hint
        or "unknown"
    )

    result = ASRResult(
        detected_language=(
            detected_language
        ),
        language_probability=1.0,
    )

    # ========================================================
    # Parse segments
    # ========================================================

    transcription = data.get(
        "transcription",
        [],
    )

    print(
        f"[stage2_asr] whisper.cpp returned "
        f"{len(transcription)} raw segment(s)."
    )

    dropped_repetition = 0
    dropped_wrong_script = 0
    dropped_empty = 0

    for entry in transcription:

        offsets = entry.get(
            "offsets",
            {},
        )

        start_ms = offsets.get(
            "from"
        )

        end_ms = offsets.get(
            "to"
        )

        if (
            start_ms is None
            or end_ms is None
        ):
            continue

        try:

            start = (
                float(start_ms)
                / 1000.0
            )

            end = (
                float(end_ms)
                / 1000.0
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        text = (
            entry.get(
                "text",
                "",
            )
            .strip()
        )

        if not text:

            dropped_empty += 1

            continue

        # ----------------------------------------------------
        # Strong repetition filter
        # ----------------------------------------------------

        if _is_degenerate_repetition(
            text
        ):

            dropped_repetition += 1

            result.dropped_ranges.append(
                (start, end)
            )

            print(
                "[stage2_asr] Dropping "
                f"hallucinated repetition "
                f"[{start:.2f}-{end:.2f}]: "
                f"{text[:120]!r}"
            )

            continue

        # ----------------------------------------------------
        # Wrong-script filter
        # ----------------------------------------------------

        if _is_wrong_script(
            text,
            detected_language,
        ):

            dropped_wrong_script += 1

            result.dropped_ranges.append(
                (start, end)
            )

            print(
                "[stage2_asr] Dropping "
                f"wrong-script hallucination "
                f"[{start:.2f}-{end:.2f}]: "
                f"{text[:120]!r}"
            )

            continue

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = (
            _segment_confidence(
                entry.get(
                    "tokens",
                    [],
                )
            )
        )

        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        if end < start:

            print(
                "[stage2_asr] Ignoring segment "
                "with invalid timestamp: "
                f"{start:.3f} -> {end:.3f}"
            )

            continue

        if end == start:

            # Preserve the segment for the timestamp correction
            # pass below.
            end = start

        for (
            piece_start,
            piece_end,
            piece_text,
            piece_confidence,
        ) in _split_on_internal_pauses(
            entry.get("tokens", []),
            start,
            end,
            text,
            confidence,
        ):

            result.segments.append(
                TranscriptSegment(
                    start=round(
                        piece_start,
                        3,
                    ),
                    end=round(
                        piece_end,
                        3,
                    ),
                    text=piece_text,
                    confidence=piece_confidence,
                )
            )

    dropped_segment_repeats = _drop_consecutive_duplicate_segments(
        result.segments,
        result.dropped_ranges,
    )

    print(
        "[stage2_asr] Parsed "
        f"{len(result.segments)} usable segment(s)."
    )

    print(
        "[stage2_asr] Dropped "
        f"{dropped_empty} empty segment(s), "
        f"{dropped_repetition} repetition segment(s), "
        f"{dropped_wrong_script} wrong-script segment(s), "
        f"{dropped_segment_repeats} segment-level repeat(s)."
    )

    # ========================================================
    # Timestamp correction
    # ========================================================

    _extend_undertimed_segments(
        result.segments
    )

    _clamp_segments_to_audio_duration(
        result.segments,
        wav_path,
    )

    return result


# ============================================================
# Timestamp correction
# ============================================================

def _clamp_segments_to_audio_duration(
    segments: List[TranscriptSegment],
    wav_path: str,
):
    """
    Clamp the final segment's end time to the actual audio file's own
    duration.

    whisper.cpp's own timestamps (or _extend_undertimed_segments'
    correction above, which has no upper bound) can occasionally run
    slightly past the real end of the audio -- verified directly against
    a real job: a final segment timestamped to end 9.4s after the source
    video's actual (ffprobe-confirmed) duration. Downstream, that
    over-length timestamp reaches the SRT/VTT (a caption "ending" after
    the video already stopped) and voiceover placement (dub audio timed
    to a slot that doesn't fully exist). Only the last segment can run
    past the file's real end (earlier segments are already bounded by
    the segment after them), so only it needs checking.
    """

    if not segments:
        return

    try:
        with wave.open(wav_path, "rb") as wf:
            audio_duration = wf.getnframes() / float(wf.getframerate())
    except Exception:
        # Don't fail a whole job over a best-effort sanity clamp.
        return

    last = segments[-1]

    if last.end > audio_duration:
        last.end = max(
            audio_duration,
            last.start,
        )


def _extend_undertimed_segments(
    segments: List[TranscriptSegment],
):
    """
    Correct whisper.cpp timestamps that are obviously too short.

    whisper.cpp can occasionally assign a large amount of spoken text
    to an unrealistically short timestamp.

    Example:

        text length:       ~128 characters
        reported duration: 0.84 sec
        expected duration: ~10+ sec

    Such timestamps cause major problems later when delivery.py tries
    to fit TTS audio into the segment.

    The correction:

        estimated_duration = characters / 12

    is intentionally conservative.

    Corrections are capped before the next segment so segments cannot
    overlap.
    """

    if not segments:
        return

    corrections = 0

    for i, seg in enumerate(
        segments
    ):

        actual_duration = (
            seg.end
            - seg.start
        )

        # ----------------------------------------------------
        # Rough speaking-rate estimate.
        #
        # 12 characters/second is intentionally conservative.
        # ----------------------------------------------------

        estimated_duration = max(
            0.4,
            len(seg.text) / 12.0,
        )

        # If the current duration is at least half the estimated
        # duration, leave it alone.
        if (
            actual_duration
            >= estimated_duration * 0.5
        ):
            continue

        desired_end = (
            seg.start
            + estimated_duration
        )

        # ----------------------------------------------------
        # Don't overlap the next segment.
        # ----------------------------------------------------

        if (
            i + 1
            < len(segments)
        ):

            next_start = (
                segments[
                    i + 1
                ].start
            )

            desired_end = min(
                desired_end,
                next_start - 0.05,
            )

        if desired_end > seg.end:

            old_end = seg.end

            seg.end = round(
                desired_end,
                3,
            )

            corrections += 1

            print(
                "[stage2_asr] Extended "
                f"undertimed segment "
                f"[{seg.start:.2f}-{old_end:.2f}] "
                f"-> [{seg.start:.2f}-{seg.end:.2f}] "
                f"({len(seg.text)} chars)"
            )

    if corrections:

        print(
            "[stage2_asr] Corrected "
            f"{corrections} undertimed segment(s)."
        )


# ============================================================
# IndicConformer
# ============================================================

def _get_indic_conformer_model():
    """
    Lazy-load IndicConformer once per process.

    IndicConformer remains optional and is only loaded when:

        asr_engine == "indic_conformer"
    """

    if "model" not in _conformer_cache:

        try:

            import torch
            from transformers import AutoModel

        except ImportError as e:

            raise RuntimeError(
                "IndicConformer requires extra packages "
                "(torchaudio, onnxruntime) not present in this "
                "environment.\n"
                "Install them with: pip install -r requirements.txt"
            ) from e

        print(
            "[stage2_asr] Loading IndicConformer..."
        )

        _conformer_cache[
            "model"
        ] = AutoModel.from_pretrained(
            INDIC_CONFORMER_MODEL,
            trust_remote_code=True,
        )

        _conformer_cache[
            "torch"
        ] = torch

        print(
            "[stage2_asr] IndicConformer loaded."
        )

    return (
        _conformer_cache["model"],
        _conformer_cache["torch"],
    )


# ============================================================
# WAV loader for IndicConformer
# ============================================================

def _load_wav_mono_16k_tensor(
    wav_path: str,
    torch_mod,
):
    """
    Load WAV once.

    Deliberately reads raw PCM via the stdlib `wave` module rather than
    torchaudio.load() -- torchaudio >= 2.9 dropped its sox/soundfile
    decoding backends and now hard-requires the separate `torchcodec`
    package for load(), which itself needs an ffmpeg build matching the
    installed torch/torchaudio versions. Pulling in that whole dependency
    chain just to read the plain 16kHz mono PCM WAV that stage 1 already
    guarantees is unnecessary fragility. `torchaudio.functional.resample`
    below still works fine without torchcodec -- it operates purely on
    tensors and never touches the I/O backends.

    Stage 1 normally produces:

        16 kHz
        mono
        PCM WAV

    so this should normally require no resampling.
    """

    import numpy as np

    with wave.open(wav_path, "rb") as w:

        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    dtype = {
        1: np.uint8,
        2: np.int16,
        4: np.int32,
    }.get(sample_width)

    if dtype is None:

        raise RuntimeError(
            f"Unsupported WAV sample width "
            f"({sample_width} bytes) in {wav_path!r}."
        )

    samples = np.frombuffer(raw, dtype=dtype)

    if sample_width == 1:

        # 8-bit PCM WAV is unsigned with a 128 midpoint, unlike 16/32-bit.
        samples = (
            samples.astype(np.float32) - 128.0
        ) / 128.0

    else:

        samples = samples.astype(np.float32) / float(
            2 ** (8 * sample_width - 1)
        )

    if n_channels > 1:

        samples = samples.reshape(-1, n_channels).T

    else:

        samples = samples.reshape(1, -1)

    wav = torch_mod.from_numpy(
        samples.copy()
    )

    # --------------------------------------------------------
    # Mono
    # --------------------------------------------------------

    if wav.shape[0] > 1:

        wav = torch_mod.mean(
            wav,
            dim=0,
            keepdim=True,
        )

    # --------------------------------------------------------
    # Resample if necessary
    # --------------------------------------------------------

    if sr != 16000:

        import torchaudio

        wav = (
            torchaudio.functional.resample(
                wav,
                sr,
                16000,
            )
        )

        sr = 16000

    return wav, sr


# ============================================================
# IndicConformer refinement
# ============================================================

def refine_segments_with_indic_conformer(
    segments: List[TranscriptSegment],
    wav_path: str,
    language: str,
    decoding: Optional[str] = None,
    cancel_event=None,
) -> List[TranscriptSegment]:
    """
    Optional post-pass using IndicConformer.

    Important:

    IndicConformer does not provide the same segmentation/timestamp
    behavior used by whisper.cpp here.

    Therefore:

        whisper.cpp = timestamps
        IndicConformer = text refinement

    Existing start/end/confidence values remain unchanged.

    The WAV is loaded once and individual segments are sliced from
    that tensor.
    """

    lang_code = (
        INDIC_CONFORMER_LANG_MAP.get(
            language
        )
    )

    if (
        lang_code is None
        or not segments
    ):

        return segments

    decoding = (
        decoding
        or INDIC_CONFORMER_DECODING
    )

    model, torch = (
        _get_indic_conformer_model()
    )

    wav, sr = (
        _load_wav_mono_16k_tensor(
            wav_path,
            torch,
        )
    )

    print(
        "[stage2_asr] IndicConformer "
        f"refinement for {len(segments)} "
        "segment(s)..."
    )

    for index, seg in enumerate(
        segments,
        start=1,
    ):

        check_cancelled(cancel_event)

        start_sample = max(
            0,
            int(
                seg.start * sr
            ),
        )

        end_sample = min(
            wav.shape[1],
            int(
                seg.end * sr
            ),
        )

        if (
            end_sample
            <= start_sample
        ):
            continue

        chunk = wav[
            :,
            start_sample:end_sample,
        ]

        with torch.no_grad():

            text = model(
                chunk,
                lang_code,
                decoding,
            )

        text = (
            text
            if isinstance(
                text,
                str,
            )
            else str(text)
        ).strip()

        if not text:
            continue

        # Don't replace good text with a newly introduced
        # repetition hallucination.
        if _is_degenerate_repetition(
            text
        ):

            print(
                "[stage2_asr] IndicConformer "
                f"produced repetition on segment "
                f"{index}; keeping whisper.cpp text."
            )

            continue

        seg.text = text

    print(
        "[stage2_asr] IndicConformer "
        "refinement complete."
    )

    return segments