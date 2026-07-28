"""
Stage 1 — Pre-Processing (FFmpeg)
  - Extract audio
  - Normalise to 16 kHz mono WAV
  - Detect embedded subtitle track
  - Auto-denoise if SNR too low (warns user)

Matches the "Stage 1 - Pre-Processing" box in the architecture diagram exactly.
"""
import json
import math
import os
import subprocess
import wave

import numpy as np

from config import TARGET_SAMPLE_RATE, SNR_DENOISE_THRESHOLD_DB


class PreprocessResult:
    def __init__(self):
        self.audio_wav_path = None
        self.subtitle_track_found = False
        self.subtitle_srt_path = None
        self.denoise_applied = False
        self.snr_db = None
        self.warnings = []
        self.input_kind = None  # "video" or "audio"

    def to_dict(self):
        return {
            "audio_wav_path": self.audio_wav_path,
            "subtitle_track_found": self.subtitle_track_found,
            "subtitle_srt_path": self.subtitle_srt_path,
            "denoise_applied": self.denoise_applied,
            "snr_db": self.snr_db,
            "warnings": self.warnings,
            "input_kind": self.input_kind,
        }


def _run(cmd):
    """Run a subprocess command, raise with stderr on failure."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\n{proc.stderr.decode(errors='ignore')}"
        )
    return proc.stdout.decode(errors="ignore")


def probe_streams(input_path):
    """Return ffprobe stream info as a list of dicts."""
    out = _run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", input_path
    ])
    return json.loads(out)


def detect_subtitle_stream(streams_info):
    """Return the ffmpeg stream index of the first subtitle stream, or None."""
    for s in streams_info.get("streams", []):
        if s.get("codec_type") == "subtitle":
            return s.get("index")
    return None


def extract_embedded_subtitles(input_path, stream_index, out_srt_path):
    """Pull an embedded subtitle track out to .srt using its own timestamps."""
    _run([
        "ffmpeg", "-y", "-i", input_path,
        "-map", f"0:{stream_index}", out_srt_path
    ])
    return out_srt_path


def extract_audio_to_wav(input_path, out_wav_path, sample_rate=TARGET_SAMPLE_RATE):
    """Extract + normalise to mono WAV at target sample rate."""
    _run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn",                      # no video
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # sample rate
        "-acodec", "pcm_s16le",
        out_wav_path
    ])
    return out_wav_path


def estimate_snr_db(wav_path):
    """
    Lightweight SNR estimate: treat the bottom 10th percentile of frame
    energy as the noise floor and the top 10th percentile as signal peak.
    This is a heuristic (not lab-grade) but is exactly the kind of quick
    check that's appropriate for an offline field tool with no reference signal.
    """
    with wave.open(wav_path, "rb") as w:
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        sampwidth = w.getsampwidth()

    if sampwidth != 2:
        # Only 16-bit PCM supported by this quick estimator; skip check.
        return None

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return None

    frame_len = 1024
    n_full_frames = samples.size // frame_len
    if n_full_frames < 2:
        return None

    frames = samples[: n_full_frames * frame_len].reshape(n_full_frames, frame_len)
    energies = np.mean(frames ** 2, axis=1)
    energies = np.clip(energies, 1e-9, None)

    noise_floor = np.percentile(energies, 10)
    signal_peak = np.percentile(energies, 90)

    snr_db = 10 * math.log10(signal_peak / noise_floor)
    return round(snr_db, 2)


def apply_denoise(in_wav_path, out_wav_path):
    """FFmpeg's afftdn (FFT-based denoiser) — safe default for speech."""
    _run([
        "ffmpeg", "-y", "-i", in_wav_path,
        "-af", "afftdn=nf=-25",
        out_wav_path
    ])
    return out_wav_path


def get_duration_seconds(input_path):
    info = probe_streams(input_path)
    return float(info.get("format", {}).get("duration", 0.0))


def run_stage1(input_path, work_dir, input_kind):
    """
    Full Stage 1 pipeline.
    input_kind: "video" or "audio" (which top box this came from)
    Returns a PreprocessResult.
    """
    os.makedirs(work_dir, exist_ok=True)
    result = PreprocessResult()
    result.input_kind = input_kind

    streams_info = probe_streams(input_path)

    # 1. Detect embedded subtitle track (video only — audio files never carry subs)
    sub_stream_idx = None
    if input_kind == "video":
        sub_stream_idx = detect_subtitle_stream(streams_info)

    if sub_stream_idx is not None:
        srt_path = os.path.join(work_dir, "embedded_subs.srt")
        try:
            extract_embedded_subtitles(input_path, sub_stream_idx, srt_path)
            result.subtitle_track_found = True
            result.subtitle_srt_path = srt_path
        except RuntimeError as e:
            result.warnings.append(f"Subtitle track detected but extraction failed: {e}")

    # 2. Extract audio -> normalise to 16kHz mono WAV
    raw_wav = os.path.join(work_dir, "audio_raw.wav")
    extract_audio_to_wav(input_path, raw_wav)

    # 3. SNR check -> auto-denoise if needed
    snr = estimate_snr_db(raw_wav)
    result.snr_db = snr

    final_wav = raw_wav
    if snr is not None and snr < SNR_DENOISE_THRESHOLD_DB:
        denoised_wav = os.path.join(work_dir, "audio_denoised.wav")
        apply_denoise(raw_wav, denoised_wav)
        final_wav = denoised_wav
        result.denoise_applied = True
        result.warnings.append(
            f"Low signal-to-noise ratio detected (~{snr} dB, threshold {SNR_DENOISE_THRESHOLD_DB} dB). "
            f"Auto-denoise applied. Transcript quality may still be reduced — consider a cleaner source recording."
        )

    result.audio_wav_path = final_wav
    return result
