"""
pipeline/delivery.py

Delivery stage.

Outputs:
    1. Burned-in MP4
       - Hard-codes the generated SRT subtitles into the video.

    2. Voiceover MP4
       - Generates one WAV per translated segment using MMS-TTS.
       - Places each generated segment on the original timeline.
       - Mixes the generated segments into one WAV.
       - Muxes that WAV onto the original video.

MMS-TTS is loaded lazily and cached (one checkpoint per language, since
unlike a single multilingual model, MMS-TTS ships a separate Hugging Face
repo per language) for the lifetime of this Python process.
"""

import gc
import os
import subprocess
from typing import List, Tuple

import soundfile as sf

from config import (
    FFMPEG_BIN,
    MMS_TTS_DEVICE,
    MMS_TTS_LANGS,
    resolve_torch_device,
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
# MMS-TTS language helpers
# ---------------------------------------------------------------------------

def voiceover_available(lang: str) -> bool:
    """Return True if MMS-TTS has a checkpoint for this language."""
    return isinstance(lang, str) and lang.strip() in MMS_TTS_LANGS


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
# Lazy MMS-TTS models (one checkpoint per language)
# ---------------------------------------------------------------------------

_mms_tts_cache = {}  # lang -> {"model", "tokenizer", "device"}


def _get_mms_tts_model(lang: str):
    """
    Lazily load the MMS-TTS checkpoint for one language. Each language is a
    separate Hugging Face repo, so this caches per language.

    Returns: model, tokenizer, device.
    """
    if lang in _mms_tts_cache:
        c = _mms_tts_cache[lang]
        return c["model"], c["tokenizer"], c["device"]

    from transformers import VitsModel, AutoTokenizer

    device = resolve_torch_device(MMS_TTS_DEVICE)
    model_name = f"facebook/mms-tts-{lang}"

    print(f"[delivery] Loading MMS-TTS ({model_name}) on {device}.")
    print("[delivery] The first load may download the Hugging Face model.")

    try:
        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load MMS-TTS ('{model_name}') for language '{lang}'. "
            f"Original error: {exc}"
        ) from exc

    model = model.to(device)
    model.eval()

    _mms_tts_cache[lang] = {"model": model, "tokenizer": tokenizer, "device": device}

    print(f"[delivery] MMS-TTS ({model_name}) ready.")

    return model, tokenizer, device


# ---------------------------------------------------------------------------
# MMS-TTS generation
# ---------------------------------------------------------------------------

def _synthesize_segment_wav(
    text: str,
    lang: str,
    out_wav_path: str,
) -> str:
    """
    Generate speech for one translated segment via MMS-TTS. One fixed voice
    per language; the whole waveform is generated in one non-autoregressive
    forward pass.
    """
    lang = _normalize_language(lang)

    if not text or not text.strip():
        raise ValueError("Cannot synthesize an empty TTS segment.")

    if lang not in MMS_TTS_LANGS:
        available = ", ".join(sorted(MMS_TTS_LANGS))
        raise RuntimeError(
            f"No MMS-TTS checkpoint is configured for language '{lang}'.\n\n"
            f"Languages currently configured in config.py:\n{available}"
        )

    model, tokenizer, device = _get_mms_tts_model(lang)

    import torch

    print(
        f"[delivery] TTS (MMS-TTS): language={lang}, "
        f"characters={len(text)}, "
        f"output={out_wav_path}"
    )

    inputs = tokenizer(text.strip(), return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    try:
        with torch.no_grad():
            outputs = model(input_ids)
        audio_arr = outputs.waveform.detach().cpu().numpy().squeeze()
    finally:
        del inputs
        del input_ids
        gc.collect()

    if audio_arr.size == 0:
        raise RuntimeError(f"MMS-TTS generated empty audio for language '{lang}'.")

    sampling_rate = int(model.config.sampling_rate)
    if sampling_rate <= 0:
        raise RuntimeError(f"Invalid MMS-TTS sampling rate: {sampling_rate}")

    os.makedirs(os.path.dirname(os.path.abspath(out_wav_path)), exist_ok=True)
    sf.write(out_wav_path, audio_arr, sampling_rate, subtype="PCM_16")

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
    max_gap_sec: float = 0.4,
) -> List[float]:
    """
    Places generated TTS clips on a timeline derived from, but not strictly
    bound to, the original segment timestamps.

    If generated TTS takes longer than the original subtitle interval, the
    following clip is moved forward rather than overlapping it (min_gap_sec
    apart). If TTS finishes well before the next segment's original
    timestamp — the common case, since translated/synthesized speech
    duration essentially never matches the original audio's pacing — the
    next clip is pulled forward too, capping dead air between segments at
    max_gap_sec instead of leaving it at whatever gap existed in the
    original video. Without this cap, voiceover narration has an audible
    silent "hole" before nearly every segment.
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
            desired_start = min(
                desired_start,
                cursor_end + max_gap_sec,
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
            f"Voiceover is not configured for language '{target_lang}'. "
            f"MMS-TTS covers: {', '.join(sorted(MMS_TTS_LANGS))}."
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
