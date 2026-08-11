"""
pipeline/delivery.py

Delivery stage.

Outputs:
    1. Burned-in MP4
       - Hard-codes the generated SRT subtitles into the video.

    2. Voiceover MP4
       - Generates one WAV per translated segment using
         AI4Bharat Indic Parler-TTS.
       - Places each generated segment on the original timeline.
       - Mixes the generated segments into one WAV.
       - Muxes that WAV onto the original video.

Parler-TTS is loaded lazily and cached for the lifetime of this Python
process. This is important because the Indic Parler-TTS model is large
and should NOT be loaded once per subtitle segment.
"""

import gc
import os
import subprocess
from typing import List, Optional, Tuple

import soundfile as sf

from config import (
    FFMPEG_BIN,
    INDIC_PARLER_TTS_DEVICE,
    INDIC_PARLER_TTS_MODEL,
    INDIC_PARLER_VOICE_DESCRIPTIONS,
)

from pipeline.stage3_segment_tm import Segment


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str]) -> bytes:
    """
    Run a subprocess and raise a useful error if it fails.
    """
    print("[delivery] Running:", " ".join(str(x) for x in cmd))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="ignore")
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}:\n"
            f"{' '.join(str(x) for x in cmd)}\n\n"
            f"{stderr}"
        )

    return proc.stdout


def _escape_ffmpeg_filter_path(path: str) -> str:
    """
    Escape a filesystem path for FFmpeg's filtergraph syntax.

    In particular:
        /path/to/file -> /path/to/file
        C:\\path      -> C:/path
        C:/path      -> C\\:/path
    """
    normalized = os.path.abspath(path).replace("\\", "/")
    return normalized.replace(":", "\\:")


# ---------------------------------------------------------------------------
# Subtitle delivery
# ---------------------------------------------------------------------------

def burn_in_subtitles(
    video_path: str,
    srt_path: str,
    out_mp4_path: str,
) -> str:
    """
    Hard-code subtitles onto the video using FFmpeg.
    """

    escaped_srt = _escape_ffmpeg_filter_path(srt_path)

    _run(
        [
            FFMPEG_BIN,
            "-y",
            "-i",
            video_path,
            "-vf",
            f"subtitles=filename='{escaped_srt}'",
            "-c:v",
            "libx264",
            "-c:a",
            "copy",
            out_mp4_path,
        ]
    )

    return out_mp4_path


# ---------------------------------------------------------------------------
# Parler-TTS language / configuration helpers
# ---------------------------------------------------------------------------

def voiceover_available(lang: str) -> bool:
    """
    Return True if the requested language has a configured Parler-TTS
    voice description.

    The actual model detects the language from the translated text.
    The description controls speaker/style characteristics.
    """
    return (
        isinstance(lang, str)
        and lang.strip() in INDIC_PARLER_VOICE_DESCRIPTIONS
    )


def _normalize_language(lang: str) -> str:
    """
    Normalize language values coming from the translation pipeline.

    We deliberately do not perform aggressive alias conversion here,
    because config.py is the authoritative source for the application's
    language names.
    """
    if not isinstance(lang, str):
        raise TypeError(f"Language must be a string, got {type(lang)!r}")

    normalized = lang.strip()

    if not normalized:
        raise ValueError("Target language is empty.")

    return normalized


# ---------------------------------------------------------------------------
# Lazy Parler-TTS model
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_description_tokenizer = None
_device = None


def _get_model():
    """
    Lazily load Indic Parler-TTS.

    Returns:
        model,
        prompt tokenizer,
        description tokenizer,
        device

    The model is loaded exactly once per Python process.
    """

    global _model
    global _tokenizer
    global _description_tokenizer
    global _device

    if (
        _model is not None
        and _tokenizer is not None
        and _description_tokenizer is not None
    ):
        return (
            _model,
            _tokenizer,
            _description_tokenizer,
            _device,
        )

    import torch

    try:
        from parler_tts import ParlerTTSForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Parler-TTS could not be imported.\n\n"
            "Install it with:\n"
            "python -m pip install --no-cache-dir "
            "\"git+https://github.com/huggingface/parler-tts.git\"\n\n"
            f"Original import error:\n{exc}"
        ) from exc

    from transformers import AutoTokenizer

    requested_device = str(INDIC_PARLER_TTS_DEVICE).strip().lower()

    # ---------------------------------------------------------------
    # Device selection
    # ---------------------------------------------------------------

    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            device = requested_device
        else:
            print(
                "[delivery] CUDA was requested, but CUDA is unavailable. "
                "Using CPU."
            )
            device = "cpu"

    elif requested_device in {"mps", "mps:0"}:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            print(
                "[delivery] MPS was requested, but MPS is unavailable. "
                "Using CPU."
            )
            device = "cpu"

    else:
        device = "cpu"

    print(
        f"[delivery] Loading Indic Parler-TTS "
        f"({INDIC_PARLER_TTS_MODEL}) on {device}."
    )
    print(
        "[delivery] The first load may download the Hugging Face model."
    )

    try:
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            INDIC_PARLER_TTS_MODEL
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load Parler-TTS model "
            f"'{INDIC_PARLER_TTS_MODEL}'.\n\n"
            f"Original error:\n{exc}"
        ) from exc

    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------------
    # Two-tokenizer setup used by Indic Parler-TTS
    # ---------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        INDIC_PARLER_TTS_MODEL
    )

    text_encoder_name = getattr(
        model.config.text_encoder,
        "_name_or_path",
        None,
    )

    if not text_encoder_name:
        raise RuntimeError(
            "Could not determine the Parler-TTS text encoder tokenizer "
            "from model.config.text_encoder._name_or_path."
        )

    description_tokenizer = AutoTokenizer.from_pretrained(
        text_encoder_name
    )

    _model = model
    _tokenizer = tokenizer
    _description_tokenizer = description_tokenizer
    _device = device

    print("[delivery] Indic Parler-TTS ready.")

    return (
        _model,
        _tokenizer,
        _description_tokenizer,
        _device,
    )


# ---------------------------------------------------------------------------
# Parler-TTS generation
# ---------------------------------------------------------------------------

def _synthesize_segment_wav(
    text: str,
    lang: str,
    out_wav_path: str,
) -> str:
    """
    Generate speech for one translated segment.

    `text` is the translated text.

    `lang` is used to select the configured speaker/style description.

    The Parler-TTS model detects the spoken language from the prompt text.
    """

    lang = _normalize_language(lang)

    if not text or not text.strip():
        raise ValueError(
            "Cannot synthesize an empty TTS segment."
        )

    description = INDIC_PARLER_VOICE_DESCRIPTIONS.get(lang)

    if not description:
        available = ", ".join(
            sorted(INDIC_PARLER_VOICE_DESCRIPTIONS.keys())
        )

        raise RuntimeError(
            f"No Parler-TTS voice description is configured for "
            f"language '{lang}'.\n\n"
            f"Languages currently configured in config.py:\n"
            f"{available}"
        )

    model, tokenizer, description_tokenizer, device = _get_model()

    import torch

    print(
        f"[delivery] TTS: language={lang}, "
        f"characters={len(text)}, "
        f"output={out_wav_path}"
    )

    # ---------------------------------------------------------------
    # Tokenize description and translated text
    # ---------------------------------------------------------------

    description_inputs = description_tokenizer(
        description,
        return_tensors="pt",
    )

    prompt_inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
    )

    description_input_ids = description_inputs.input_ids.to(device)
    description_attention_mask = (
        description_inputs.attention_mask.to(device)
    )

    prompt_input_ids = prompt_inputs.input_ids.to(device)
    prompt_attention_mask = prompt_inputs.attention_mask.to(device)

    # ---------------------------------------------------------------
    # Generate
    # ---------------------------------------------------------------

    try:
        with torch.inference_mode():
            generation = model.generate(
                input_ids=description_input_ids,
                attention_mask=description_attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
            )

        audio_arr = generation.detach().cpu().numpy().squeeze()

    finally:
        # Release segment-level tensors immediately.
        del description_inputs
        del prompt_inputs
        del description_input_ids
        del description_attention_mask
        del prompt_input_ids
        del prompt_attention_mask

        gc.collect()

    # ---------------------------------------------------------------
    # Validate generated audio
    # ---------------------------------------------------------------

    if audio_arr.size == 0:
        raise RuntimeError(
            f"Parler-TTS generated empty audio for language '{lang}'."
        )

    sampling_rate = int(model.config.sampling_rate)

    if sampling_rate <= 0:
        raise RuntimeError(
            f"Invalid Parler-TTS sampling rate: {sampling_rate}"
        )

    os.makedirs(
        os.path.dirname(os.path.abspath(out_wav_path)),
        exist_ok=True,
    )

    # PCM_16 gives a conventional WAV suitable for FFmpeg and downstream
    # processing.
    sf.write(
        out_wav_path,
        audio_arr,
        sampling_rate,
        subtype="PCM_16",
    )

    return out_wav_path


# ---------------------------------------------------------------------------
# WAV utilities
# ---------------------------------------------------------------------------

def _get_wav_duration(wav_path: str) -> float:
    """
    Return WAV duration in seconds.
    """

    info = sf.info(wav_path)

    if info.samplerate <= 0:
        raise RuntimeError(
            f"Invalid sample rate in WAV file: {wav_path}"
        )

    return info.frames / float(info.samplerate)


def _get_sequential_timeline(
    segment_starts: List[float],
    segment_durations: List[float],
    min_gap_sec: float = 0.05,
) -> List[float]:
    """
    Preserve original segment timestamps while preventing generated
    TTS clips from overlapping.

    If generated TTS takes longer than the original subtitle interval,
    the following clip is moved forward rather than overlapping it.
    """

    if not segment_starts:
        return []

    if len(segment_starts) != len(segment_durations):
        raise ValueError(
            "segment_starts and segment_durations must have "
            "the same length."
        )

    adjusted_starts = []

    cursor_end = 0.0

    for index, (start, duration) in enumerate(
        zip(segment_starts, segment_durations)
    ):
        start = max(0.0, float(start))
        duration = max(0.01, float(duration))

        if index == 0:
            desired_start = start
        else:
            desired_start = max(
                start,
                cursor_end + min_gap_sec,
            )

        adjusted_starts.append(desired_start)

        cursor_end = desired_start + duration

    return adjusted_starts


# ---------------------------------------------------------------------------
# Voiceover track construction
# ---------------------------------------------------------------------------

def build_voiceover_track(
    segments: List[Segment],
    target_lang: str,
    work_dir: str,
) -> str:
    """
    Generate one WAV per translated segment and combine them onto a
    single timeline.

    Returns:
        Path to voiceover_mixed.wav
    """

    target_lang = _normalize_language(target_lang)

    if not voiceover_available(target_lang):
        raise RuntimeError(
            f"Voiceover is not configured for language '{target_lang}'."
        )

    tts_dir = os.path.join(
        work_dir,
        "tts_segments",
    )

    os.makedirs(
        tts_dir,
        exist_ok=True,
    )

    segment_wavs: List[
        Tuple[float, str, float]
    ] = []

    # ---------------------------------------------------------------
    # Generate TTS segments
    # ---------------------------------------------------------------

    for seg in segments:
        text = (
            seg.translated_text
            if seg.translated_text is not None
            else seg.text
        )

        if text is None or not text.strip():
            print(
                f"[delivery] Skipping empty segment {seg.index}."
            )
            continue

        seg_wav = os.path.join(
            tts_dir,
            f"seg_{seg.index:05d}.wav",
        )

        _synthesize_segment_wav(
            text=text,
            lang=target_lang,
            out_wav_path=seg_wav,
        )

        duration = _get_wav_duration(seg_wav)

        segment_wavs.append(
            (
                float(seg.start),
                seg_wav,
                duration,
            )
        )

    if not segment_wavs:
        raise RuntimeError(
            "No translated segments produced TTS audio."
        )

    # ---------------------------------------------------------------
    # Calculate timeline
    # ---------------------------------------------------------------

    starts = [
        item[0]
        for item in segment_wavs
    ]

    durations = [
        item[2]
        for item in segment_wavs
    ]

    adjusted_starts = _get_sequential_timeline(
        starts,
        durations,
    )

    # ---------------------------------------------------------------
    # FFmpeg filter graph
    # ---------------------------------------------------------------

    inputs: List[str] = []
    filter_parts: List[str] = []
    mix_labels: List[str] = []

    for i, (_, wav_path, _) in enumerate(segment_wavs):
        inputs.extend(
            [
                "-i",
                wav_path,
            ]
        )

        delay_ms = max(
            0,
            int(round(adjusted_starts[i] * 1000)),
        )

        label = f"a{i}"

        filter_parts.append(
            f"[{i}:a]"
            f"adelay={delay_ms}:all=1"
            f"[{label}]"
        )

        mix_labels.append(
            f"[{label}]"
        )

    filter_complex = (
        ";".join(filter_parts)
        + ";"
        + "".join(mix_labels)
        + f"amix="
          f"inputs={len(mix_labels)}:"
          f"duration=longest:"
          f"dropout_transition=0:"
          f"normalize=0"
          f"[out]"
    )

    out_wav = os.path.join(
        work_dir,
        "voiceover_mixed.wav",
    )

    cmd = (
        [
            FFMPEG_BIN,
            "-y",
        ]
        + inputs
        + [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            out_wav,
        ]
    )

    _run(cmd)

    return out_wav


# ---------------------------------------------------------------------------
# Video + voiceover muxing
# ---------------------------------------------------------------------------

def mux_voiceover_onto_video(
    video_path: str,
    voiceover_wav: str,
    out_mp4_path: str,
    mix_original_audio: bool = False,
    original_audio_gain_db: float = -18.0,
) -> str:
    """
    Put the generated voiceover onto the video.

    Default:
        Original audio is replaced.

    Optional:
        Original audio can be mixed underneath the generated voiceover.
    """

    if mix_original_audio:
        _run(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                video_path,
                "-i",
                voiceover_wav,
                "-filter_complex",
                (
                    f"[0:a]"
                    f"volume={original_audio_gain_db}dB"
                    f"[orig];"
                    f"[orig][1:a]"
                    f"amix=inputs=2:"
                    f"duration=first:"
                    f"dropout_transition=0"
                    f"[aout]"
                ),
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                out_mp4_path,
            ]
        )

    else:
        _run(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                video_path,
                "-i",
                voiceover_wav,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                out_mp4_path,
            ]
        )

    return out_mp4_path
