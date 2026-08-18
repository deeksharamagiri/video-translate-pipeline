"""
Stage 2 — ASR (faster-whisper)
  - Audio -> timestamped transcript
  - Language auto-detected
  - User confirms before run (enforced by the Flask route / UI, not here)

Only invoked on the "NO EXTRACTABLE SUBTITLE" branch of the diagram.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    INDIC_CONFORMER_MODEL, INDIC_CONFORMER_DECODING, INDIC_CONFORMER_LANG_MAP,
)

_model_cache = {}
_conformer_cache = {}


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float  # avg_logprob-derived, 0..1 (higher = more confident)


@dataclass
class ASRResult:
    detected_language: str
    language_probability: float
    segments: List[TranscriptSegment] = field(default_factory=list)


def _get_model():
    """Lazy-load faster-whisper model once per process."""
    key = (WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
    if key not in _model_cache:
        from faster_whisper import WhisperModel
        _model_cache[key] = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _model_cache[key]


def _logprob_to_confidence(avg_logprob: Optional[float]) -> float:
    """
    faster-whisper gives avg_logprob typically in range [-1, 0] for decent
    transcriptions. Map it to an intuitive 0..1 confidence score for the
    QC-flag column in the job report.
    """
    if avg_logprob is None:
        return 0.5
    # clamp to [-1.5, 0] then rescale to [0,1]
    clamped = max(-1.5, min(0.0, avg_logprob))
    return round((clamped + 1.5) / 1.5, 3)


def run_stage2(wav_path: str, language_hint: Optional[str] = None) -> ASRResult:
    """
    Run faster-whisper on a 16kHz mono WAV file.
    language_hint: ISO 639-1 code if the user already confirmed a language
                   (skips auto-detect uncertainty); None = auto-detect.
    """
    model = _get_model()

    segments_iter, info = model.transcribe(
        wav_path,
        language=language_hint,
        vad_filter=True,                 # skip silence, improves timestamp accuracy
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=False,
    )

    result = ASRResult(
        detected_language=info.language,
        language_probability=round(float(info.language_probability), 3),
    )

    for seg in segments_iter:
        result.segments.append(
            TranscriptSegment(
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
                confidence=_logprob_to_confidence(seg.avg_logprob),
            )
        )

    return result


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
    confidence from whichever base engine produced them -- faster-whisper or
    the embedded-SRT parser). IndicConformer has no built-in
    segmentation/timestamps of its own, so faster-whisper's VAD-based
    segmentation is still what drives timing; this only swaps the
    transcribed text. Loads + resamples the full wav ONCE, then slices by
    sample offset per segment -- no re-invocation of ffmpeg or re-loading
    the file per segment. No-op (returns segments unchanged) if `language`
    isn't in INDIC_CONFORMER_LANG_MAP.
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
