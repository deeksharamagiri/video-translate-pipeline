"""
Stage 2 — ASR (faster-whisper)
  - Audio -> timestamped transcript
  - Language auto-detected
  - User confirms before run (enforced by the Flask route / UI, not here)

Only invoked on the "NO EXTRACTABLE SUBTITLE" branch of the diagram.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

_model_cache = {}


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
