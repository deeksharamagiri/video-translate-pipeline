"""
Stage 2 — ASR (whisper.cpp)
  - Audio -> timestamped transcript
  - Language auto-detected
  - User confirms before run (enforced by the Flask route / UI, not here)

Only invoked on the "NO EXTRACTABLE SUBTITLE" branch of the diagram.

Runs whisper.cpp (a separate binary, via subprocess) with a quantized
medium-size model rather than the Python faster-whisper package. See
config.py's Stage 2 comment for why: faster-whisper's "small" model was
verified to misdetect language and produce garbage on real content, its
vad_filter=True was verified to silently drop multi-minute stretches of
real speech, and plain faster-whisper "medium" reliably OOM-killed on
constrained (8GB) hardware where whisper.cpp's quantized medium model
runs stably.
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
from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    WHISPER_CPP_BINARY, WHISPER_CPP_MODEL_DIR, WHISPER_CPP_MODEL_PATH,
    WHISPER_CPP_MODEL_URL,
    INDIC_CONFORMER_MODEL, INDIC_CONFORMER_DECODING, INDIC_CONFORMER_LANG_MAP,
)

_conformer_cache = {}
_model_ensured = False


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float  # 0..1 (higher = more confident)


@dataclass
class ASRResult:
    detected_language: str
    language_probability: float
    segments: List[TranscriptSegment] = field(default_factory=list)


def _download_whispercpp_model(url: str, dest_path: str, max_retries: int = 5):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    tmp_path = dest_path + ".part"

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[stage2_asr] Downloading whisper.cpp model "
                  f"(attempt {attempt}/{max_retries}): {url}")

            with urllib.request.urlopen(
                urllib.request.Request(
                    url, headers={"User-Agent": "Offline-Field-Translator/1.0"}
                ),
                context=ctx,
                timeout=120,
            ) as resp:
                with open(tmp_path, "wb") as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)

            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                raise RuntimeError("Downloaded file is empty.")

            os.replace(tmp_path, dest_path)
            print("[stage2_asr]   -> download complete")
            return

        except Exception as exc:
            print(f"[stage2_asr] Download attempt {attempt} failed: {exc}")

            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if attempt == max_retries:
                raise RuntimeError(
                    "Failed to download whisper.cpp model after "
                    f"{max_retries} attempts: {url}"
                ) from exc

            time.sleep(min(2 ** attempt, 30))


def _ensure_whispercpp_model() -> str:
    """Download the quantized whisper.cpp model once, cache under MODELS_DIR."""
    global _model_ensured

    if _model_ensured and os.path.exists(WHISPER_CPP_MODEL_PATH):
        return WHISPER_CPP_MODEL_PATH

    if os.path.exists(WHISPER_CPP_MODEL_PATH) and os.path.getsize(WHISPER_CPP_MODEL_PATH) > 0:
        _model_ensured = True
        return WHISPER_CPP_MODEL_PATH

    if shutil.which(WHISPER_CPP_BINARY) is None:
        raise RuntimeError(
            f"whisper.cpp binary '{WHISPER_CPP_BINARY}' not found on PATH. "
            "Install it (macOS: `brew install whisper-cpp`) or set "
            "WHISPER_CPP_BINARY to its full path."
        )

    os.makedirs(WHISPER_CPP_MODEL_DIR, exist_ok=True)
    _download_whispercpp_model(WHISPER_CPP_MODEL_URL, WHISPER_CPP_MODEL_PATH)

    _model_ensured = True
    return WHISPER_CPP_MODEL_PATH


def _is_degenerate_repetition(text: str) -> bool:
    """
    Whisper-family models can get stuck in a repetition loop on hard/
    ambiguous audio (music, noise, silence right at the edge of the
    no-speech threshold) instead of admitting low confidence -- e.g. a
    single character or short phrase repeated for the entire segment
    ("অববববব...", "अब तक, अब तक, अब तक..."). -nth/-et tuning on the
    whisper.cpp invocation reduces how often this happens but doesn't
    eliminate it, so this is a last-resort filter: collapse any short
    (<=12 char) unit that repeats 3+ times in a row down to one instance,
    and if that shrinks the text by more than half, treat the whole
    segment as a hallucinated loop rather than real speech.
    """
    text = text.strip()
    if len(text) < 12:
        return False
    collapsed = re.sub(r"(.{1,12}?)\1{2,}", r"\1", text)
    return len(collapsed) < len(text) * 0.5


def _segment_confidence(tokens: list) -> float:
    """
    Average per-token probability, skipping special tokens (e.g. "[_BEG_]",
    "[_TT_123]") which aren't real spoken content and would drag the score
    down misleadingly.
    """
    probs = [
        t.get("p", 0.0)
        for t in tokens
        if not str(t.get("text", "")).strip().startswith("[")
    ]

    if not probs:
        return 0.5

    return round(sum(probs) / len(probs), 3)


def run_stage2(wav_path: str, language_hint: Optional[str] = None) -> ASRResult:
    """
    Run whisper.cpp on a 16kHz mono WAV file.

    language_hint: Whisper-style language code (e.g. "mr") if the user
                   already confirmed a language; None/"" = auto-detect.
    """
    model_path = _ensure_whispercpp_model()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_prefix = os.path.join(tmp_dir, "out")

        cmd = [
            WHISPER_CPP_BINARY,
            "-m", model_path,
            "-f", wav_path,
            "-l", language_hint or "auto",
            # Raising the no-speech threshold and lowering the entropy
            # threshold makes the decoder back off instead of forcing a
            # confident transcription (and often a repetition loop) onto
            # ambiguous/low-SNR audio -- verified directly against real
            # job audio to fix exactly this failure mode.
            "-nth", "0.2",
            "-et", "3.0",
            "-oj", "-ojf",
            "-of", out_prefix,
            "-np",
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if proc.returncode != 0:
            raise RuntimeError(
                "whisper.cpp transcription failed:\n"
                + proc.stderr.decode(errors="ignore")
            )

        json_path = out_prefix + ".json"

        # A degenerate repetition loop can repeat a multi-byte character
        # so many times it gets cut off mid-sequence at whisper.cpp's
        # internal length cap, corrupting the JSON's UTF-8 -- decode
        # leniently so one bad segment can't crash the whole transcript;
        # _is_degenerate_repetition() below drops that segment anyway.
        with open(json_path, "rb") as f:
            data = json.loads(f.read().decode("utf-8", errors="replace"))

    detected_language = data.get("result", {}).get("language") or language_hint or "unknown"

    result = ASRResult(
        detected_language=detected_language,
        language_probability=1.0,
    )

    for entry in data.get("transcription", []):
        start = entry["offsets"]["from"] / 1000.0
        end = entry["offsets"]["to"] / 1000.0
        text = entry.get("text", "").strip()

        if not text:
            continue

        if _is_degenerate_repetition(text):
            print(
                f"[stage2_asr] Dropping segment [{start:.2f}-{end:.2f}]: "
                f"looks like a hallucinated repetition loop, not real speech."
            )
            continue

        confidence = _segment_confidence(entry.get("tokens", []))

        result.segments.append(
            TranscriptSegment(
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                confidence=confidence,
            )
        )

    _extend_undertimed_segments(result.segments)

    return result


def _extend_undertimed_segments(segments: List[TranscriptSegment]):
    """
    whisper.cpp sometimes assigns a segment far too little time for how
    much text it actually contains -- not only the exact-zero case
    (offsets.from == offsets.to), but also just-plain-too-short windows
    (verified against a real job: a 128-character sentence, clearly ~10+
    seconds of speech given its TTS output actually took 10.79s, assigned a
    0.84s slot -- a 12x mismatch). Passed straight through, a slot this
    wrong makes it physically impossible for delivery.py's voiceover timing
    to fit that segment's audio within any reasonable tempo bound, which
    was the dominant source of multi-second audio/video desync even after
    the exact-zero case was fixed.

    Extend `end` toward a rough speaking-rate estimate whenever the
    reported timing is less than half of what that estimate implies --
    generous enough to leave normal (if brisk) speech untouched, tight
    enough to catch genuine mistimings like this one. Capped at the next
    segment's start (minus a small gap) so this can never create an
    overlapping cue -- when the next segment starts almost immediately
    after, there simply isn't room to fully correct, and that's a real
    whisper.cpp segmentation problem no amount of end-time nudging can fix;
    delivery.py's tempo compression is left to do what it can with the
    (still improved, but not perfect) result.
    """
    for i, seg in enumerate(segments):
        estimated_duration = max(0.4, len(seg.text) / 12.0)
        if (seg.end - seg.start) >= estimated_duration * 0.5:
            continue
        desired_end = seg.start + estimated_duration
        if i + 1 < len(segments):
            desired_end = min(desired_end, segments[i + 1].start - 0.05)
        if desired_end > seg.end:
            seg.end = round(desired_end, 3)


# ------------------------------------------------------- IndicConformer (optional)
def _get_indic_conformer_model():
    """Lazy-load IndicConformer once per process. Off by default -- only
    invoked when asr_engine='indic_conformer' is explicitly requested."""
    if "model" not in _conformer_cache:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:
            raise RuntimeError(
                "IndicConformer requires extra packages (torchaudio, onnxruntime) "
                "that aren't installed by default. Install with: "
                "pip install -r requirements-optional.txt"
            ) from e
        _conformer_cache["model"] = AutoModel.from_pretrained(
            INDIC_CONFORMER_MODEL, trust_remote_code=True
        )
        _conformer_cache["torch"] = torch
    return _conformer_cache["model"], _conformer_cache["torch"]


def _load_wav_mono_16k_tensor(wav_path: str, torch_mod):
    """Load once; Stage 1 already writes 16kHz mono PCM WAV so this is
    normally a no-op resample, but guard it anyway (defensive, cheap)."""
    import torchaudio
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = torch_mod.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav, sr


def refine_segments_with_indic_conformer(segments: List[TranscriptSegment], wav_path: str,
                                          language: str, decoding: Optional[str] = None) -> List[TranscriptSegment]:
    """
    Optional post-pass: re-transcribes each segment's audio slice with
    IndicConformer, replacing segment.text in place (keeps start/end/
    confidence from whichever base engine produced them -- whisper.cpp or
    the embedded-SRT parser). IndicConformer has no built-in
    segmentation/timestamps of its own, so whisper.cpp's segmentation is
    still what drives timing; this only swaps the transcribed text. Loads +
    resamples the full wav ONCE, then slices by sample offset per segment --
    no re-invocation of ffmpeg or re-loading the file per segment. No-op
    (returns segments unchanged) if `language` isn't in
    INDIC_CONFORMER_LANG_MAP.
    """
    lang_code = INDIC_CONFORMER_LANG_MAP.get(language)
    if lang_code is None or not segments:
        return segments

    decoding = decoding or INDIC_CONFORMER_DECODING
    model, torch = _get_indic_conformer_model()
    wav, sr = _load_wav_mono_16k_tensor(wav_path, torch)

    for seg in segments:
        start_sample = max(0, int(seg.start * sr))
        end_sample = min(wav.shape[1], int(seg.end * sr))
        if end_sample <= start_sample:
            continue
        chunk = wav[:, start_sample:end_sample]
        with torch.no_grad():
            text = model(chunk, lang_code, decoding)
        text = (text if isinstance(text, str) else str(text)).strip()
        if text:
            seg.text = text
    return segments
