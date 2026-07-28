"""
Delivery — "USER OPT-IN" branch of the diagram:
  Burned-in MP4 (subtitles hard-coded onto the video via FFmpeg)
  Voiceover MP4 (Piper TTS per segment -> stitched WAV -> muxed over video)
"""
import os
import subprocess
import wave
from typing import List

from config import PIPER_VOICES_DIR, PIPER_VOICE_MAP, PIPER_VOICE_URLS
from pipeline.stage3_segment_tm import Segment


def _run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\n{proc.stderr.decode(errors='ignore')}")
    return proc.stdout


def _escape_ffmpeg_filter_path(path: str) -> str:
    """Normalize to forward slashes and escape ':' — the combo ffmpeg's
    filtergraph parser needs regardless of platform (Windows paths with
    backslashes + a drive-letter colon are the usual failure case)."""
    p = os.path.abspath(path).replace("\\", "/")
    return p.replace(":", "\\:")


def burn_in_subtitles(video_path: str, srt_path: str, out_mp4_path: str) -> str:
    """Hard-code subtitles onto the video using ffmpeg's subtitles filter."""
    escaped_srt = _escape_ffmpeg_filter_path(srt_path)
    _run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles='{escaped_srt}'",
        "-c:a", "copy",
        out_mp4_path
    ])
    return out_mp4_path


def voiceover_available(lang: str) -> bool:
    """Whether a Piper voice is actually configured for this language —
    check this BEFORE attempting voiceover generation so an unsupported
    language degrades gracefully (skip voiceover, keep subtitles/burned-in)
    instead of raising partway through and failing the whole job."""
    return lang in PIPER_VOICE_MAP


def _ensure_piper_voice(voice_file: str) -> str:
    """
    Return the local path to a Piper voice model, downloading it (and its
    matching .json config) on first use if it isn't present yet. This is what
    lets the app work from a plain `pip install -r requirements.txt` with no
    separate setup script — the voice is fetched the first time it's needed.
    """
    os.makedirs(PIPER_VOICES_DIR, exist_ok=True)
    voice_path = os.path.join(PIPER_VOICES_DIR, voice_file)
    json_file = voice_file + ".json"
    json_path = os.path.join(PIPER_VOICES_DIR, json_file)

    for fname, fpath in [(voice_file, voice_path), (json_file, json_path)]:
        if os.path.exists(fpath):
            continue
        url = PIPER_VOICE_URLS.get(fname)
        if not url:
            raise RuntimeError(
                f"No download URL configured for Piper voice file '{fname}'. "
                f"Add one to config.PIPER_VOICE_URLS or place the file manually "
                f"in {PIPER_VOICES_DIR}."
            )
        print(f"[delivery] Downloading Piper voice file '{fname}' (first use only)...")
        import urllib.request
        urllib.request.urlretrieve(url, fpath)
        print(f"[delivery]   -> saved to {fpath}")

    return voice_path


def _synthesize_segment_wav(text: str, lang: str, out_wav_path: str):
    """Call Piper TTS CLI (installed via `pip install piper-tts`) for one segment."""
    voice_file = PIPER_VOICE_MAP.get(lang)
    if voice_file is None:
        raise RuntimeError(f"No Piper voice configured for language '{lang}'. "
                            f"Add one in config.PIPER_VOICE_MAP (and a download URL in "
                            f"config.PIPER_VOICE_URLS, or place the .onnx file manually).")
    voice_path = _ensure_piper_voice(voice_file)

    proc = subprocess.run(
        ["piper", "--model", voice_path, "--output_file", out_wav_path],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Piper TTS failed: {proc.stderr.decode(errors='ignore')}")
    return out_wav_path


def _get_wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def build_voiceover_track(segments: List[Segment], target_lang: str, work_dir: str) -> str:
    """
    Synthesize one WAV per segment with Piper, then stitch them onto a
    single timeline (silence-padded to match original segment timestamps)
    using ffmpeg's concat + adelay filters. Returns path to the mixed WAV.
    """
    tts_dir = os.path.join(work_dir, "tts_segments")
    os.makedirs(tts_dir, exist_ok=True)

    segment_wavs = []
    for seg in segments:
        text = seg.translated_text if seg.translated_text is not None else seg.text
        if not text.strip():
            continue
        seg_wav = os.path.join(tts_dir, f"seg_{seg.index:05d}.wav")
        _synthesize_segment_wav(text, target_lang, seg_wav)
        segment_wavs.append((seg.start, seg_wav))

    if not segment_wavs:
        raise RuntimeError("No segments produced TTS audio.")

    # Build an ffmpeg filter_complex that delays each clip to its start time and mixes.
    inputs = []
    filter_parts = []
    mix_labels = []
    for i, (start, wav_path) in enumerate(segment_wavs):
        inputs += ["-i", wav_path]
        delay_ms = max(0, int(start * 1000))
        filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
        mix_labels.append(f"[a{i}]")

    filter_complex = ";".join(filter_parts) + ";" + "".join(mix_labels) + \
        f"amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=0[out]"

    out_wav = os.path.join(work_dir, "voiceover_mixed.wav")
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[out]", out_wav
    ]
    _run(cmd)
    return out_wav


def mux_voiceover_onto_video(video_path: str, voiceover_wav: str, out_mp4_path: str,
                              mix_original_audio: bool = False,
                              original_audio_gain_db: float = -18.0) -> str:
    """
    Puts the translated voiceover onto the video.

    mix_original_audio=False (default): fully replaces the original audio
    track with the dub — this is what "voiceover" means for most dubbing
    use cases, and avoids the original speaker bleeding through under the
    translation.

    mix_original_audio=True: keeps original audio ducked underneath the dub
    (at original_audio_gain_db) instead of removing it — useful if the
    source has music/ambience you want preserved, but not a good default
    for spoken narration since it sounds like two people talking at once.
    """
    if mix_original_audio:
        _run([
            "ffmpeg", "-y", "-i", video_path, "-i", voiceover_wav,
            "-filter_complex",
            f"[0:a]volume={original_audio_gain_db}dB[orig];[orig][1:a]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy",
            out_mp4_path
        ])
    else:
        _run([
            "ffmpeg", "-y", "-i", video_path, "-i", voiceover_wav,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-shortest",
            out_mp4_path
        ])
    return out_mp4_path