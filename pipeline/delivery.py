"""
Delivery — "USER OPT-IN" branch of the diagram:
  Burned-in MP4 (subtitles hard-coded onto the video via FFmpeg)
  Voiceover MP4 (Indic-TTS FastPitch+HiFi-GAN per segment -> stitched WAV -> muxed over video)
"""
import json
import os
import subprocess
import wave
import zipfile
from typing import List, Optional

from config import (
    FFMPEG_BINARY,
    INDIC_TTS_VENV_DIR, INDIC_TTS_WORKER_SCRIPT, INDIC_TTS_CHECKPOINTS_DIR,
    INDIC_TTS_RELEASE_BASE_URL, INDIC_TTS_LANG_ZIP_MAP, INDIC_TTS_DEFAULT_SPEAKER,
    VOICEOVER_TIME_STRETCH, VOICEOVER_MAX_TEMPO_RATIO,
)
from pipeline.stage3_segment_tm import Segment


def _run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\n{proc.stderr.decode(errors='ignore')}")
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
        FFMPEG_BINARY, "-y", "-i", video_path,
        "-vf", f"subtitles='{escaped_srt}'",
        "-c:a", "copy",
        out_mp4_path
    ])
    return out_mp4_path


def voiceover_available(lang: str) -> bool:
    """Whether voiceover can run for this language -- check this BEFORE
    attempting voiceover generation so an unsupported language degrades
    gracefully (skip voiceover, keep subtitles/burned-in) instead of
    raising partway through and failing the whole job."""
    return lang in INDIC_TTS_LANG_ZIP_MAP


def _get_wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ------------------------------------------------- duration-aware alignment
def _build_atempo_chain(ratio: float) -> str:
    """atempo only accepts [0.5, 2.0] per filter instance; chain multiple
    stages for more extreme ratios. Clamps to VOICEOVER_MAX_TEMPO_RATIO,
    beyond which the segment is left as-is and allowed to overflow into the
    next gap (existing _get_sequential_timeline collision avoidance already
    handles that) rather than distorting the audio further."""
    ratio = max(1.0 / VOICEOVER_MAX_TEMPO_RATIO, min(VOICEOVER_MAX_TEMPO_RATIO, ratio))
    stages = []
    remaining = ratio
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={s:.4f}" for s in stages)


def _time_stretch_to_fit(in_wav_path: str, out_wav_path: str, target_duration: float) -> str:
    """Speed up/slow down in_wav_path so its duration matches target_duration
    (the segment's original time slot). Returns in_wav_path unchanged (no
    re-encode) if already close enough or if target_duration is degenerate."""
    current = _get_wav_duration(in_wav_path)
    if target_duration <= 0 or current <= 0:
        return in_wav_path
    ratio = current / target_duration  # >1 => too long => speed up
    if abs(ratio - 1.0) < 0.03:
        return in_wav_path
    filt = _build_atempo_chain(ratio)
    _run([FFMPEG_BINARY, "-y", "-i", in_wav_path, "-filter:a", filt, out_wav_path])
    return out_wav_path


# ---------------------------------------------------------------- Indic-TTS
def _get_tts_venv_python() -> str:
    """
    Path to the python interpreter in the isolated TTS venv (.venv-tts),
    kept separate from the main project's venv: coqui-tts (needed to load
    Indic-TTS's FastPitch+HiFi-GAN checkpoints) requires a much newer
    `transformers` than transformers==4.44.2, which this project pins for
    IndicTrans2/NLLB -- installing both in one environment risks breaking
    translation.
    """
    venv_python = os.path.join(INDIC_TTS_VENV_DIR, "bin", "python3")
    if os.name == "nt":
        venv_python = os.path.join(INDIC_TTS_VENV_DIR, "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        raise RuntimeError(
            "Voiceover TTS venv not set up. From the project root, run:\n"
            f"  python3 -m venv {INDIC_TTS_VENV_DIR}\n"
            f"  {venv_python} -m pip install -r tts_worker/requirements.txt\n"
            "This is a separate venv (deliberately) so coqui-tts's newer "
            "transformers requirement doesn't conflict with this project's "
            "pinned transformers==4.44.2 (needed for IndicTrans2/NLLB)."
        )
    return venv_python


def _find_checkpoint_subdir(root: str, name: str) -> Optional[str]:
    """
    Recursively search for a directory named `name` under `root`. The
    release zip's exact top-level wrapping wasn't confirmed by direct
    inspection (a ~1.5GB download per language made that impractical to
    verify byte-for-byte) -- this is defensive against it not being exactly
    `root/name`, based on the layout AI4Bharat's own reference sample.py
    script expects (`checkpoints/<lang>/fastpitch/`, `.../hifigan/`).
    """
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return os.path.join(dirpath, name)
    return None


def _download_file_streamed(
    url: str,
    dest_path: str,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 5,
):
    """Download a large file safely with retries and ZIP validation."""
    import os
    import time
    import ssl
    import urllib.request
    import urllib.error
    import zipfile

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()

    tmp_path = dest_path + ".part"

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[delivery] Download attempt {attempt}/{max_retries}: "
                f"{url}"
            )

            downloaded = 0

            with urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    headers={"User-Agent": "Offline-Field-Translator/1.0"},
                ),
                context=ctx,
                timeout=120,
            ) as resp:

                total = resp.headers.get("Content-Length")
                total = int(total) if total else None

                with open(tmp_path, "wb") as out:
                    while True:
                        chunk = resp.read(chunk_size)

                        if not chunk:
                            break

                        out.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            percent = downloaded * 100 / total
                            print(
                                f"\r[delivery]   "
                                f"{downloaded / (1024**3):.2f} / "
                                f"{total / (1024**3):.2f} GB "
                                f"({percent:.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r[delivery]   "
                                f"{downloaded / (1024**3):.2f} GB",
                                end="",
                                flush=True,
                            )

            print()

            # Make sure the download isn't empty/truncated.
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                raise RuntimeError("Downloaded file is empty.")

            # Validate the ZIP before accepting it.
            print("[delivery]   -> validating ZIP...")

            with zipfile.ZipFile(tmp_path, "r") as zf:
                bad_file = zf.testzip()

            if bad_file is not None:
                raise RuntimeError(
                    f"ZIP validation failed; corrupted file: {bad_file}"
                )

            # Only replace the destination after validation succeeds.
            os.replace(tmp_path, dest_path)

            print("[delivery]   -> download verified successfully")
            return

        except Exception as exc:
            print(
                f"\n[delivery] Download attempt {attempt} failed: {exc}"
            )

            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download valid checkpoint after "
                    f"{max_retries} attempts: {url}"
                ) from exc

            wait = min(2 ** attempt, 30)
            print(f"[delivery] Retrying in {wait} seconds...")
            time.sleep(wait)


def _resolve_checkpoint_dir(lang_dir: str) -> Optional[str]:
    """
    Find the directory that directly contains `fastpitch/` and `hifigan/`
    subfolders somewhere under lang_dir. The release zips wrap their
    contents in an extra top-level folder (e.g. hi.zip extracts to
    <lang_dir>/hi/{fastpitch,hifigan}/, confirmed by inspection), so this
    can't just be lang_dir itself -- it's whatever directory turns out to
    be the immediate parent of both subfolders.
    """
    fastpitch_dir = _find_checkpoint_subdir(lang_dir, "fastpitch")
    hifigan_dir = _find_checkpoint_subdir(lang_dir, "hifigan")
    if not (fastpitch_dir and hifigan_dir):
        return None
    parent = os.path.dirname(fastpitch_dir)
    if parent != os.path.dirname(hifigan_dir):
        return None
    return parent


def _find_file_by_name(root: str, filename: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def _rewrite_config_file_paths(obj, checkpoint_dir: str) -> bool:
    """
    Recursively rewrite any string value ending in .pth/.npy/.npz to the
    file's actual location under checkpoint_dir (matched by basename).
    Returns True if anything changed.
    """
    changed = False
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and value.lower().endswith((".pth", ".npy", ".npz")):
                real_path = _find_file_by_name(checkpoint_dir, os.path.basename(value))
                if real_path and os.path.abspath(real_path) != os.path.abspath(value):
                    obj[key] = real_path
                    changed = True
            elif isinstance(value, (dict, list)):
                if _rewrite_config_file_paths(value, checkpoint_dir):
                    changed = True
    elif isinstance(obj, list):
        for item in obj:
            if _rewrite_config_file_paths(item, checkpoint_dir):
                changed = True
    return changed


def _patch_checkpoint_config_paths(checkpoint_dir: str):
    """
    AI4Bharat's published configs bake in file paths from their own
    training environment (e.g. `speakers_file: "models/v1/hi/fastpitch/
    speakers.pth"`), which don't exist here and silently override whatever
    path we pass to Synthesizer's constructor. Rewrite any such reference
    (matched by filename) to point at wherever that file actually landed
    after extraction. Idempotent -- safe to call on every resolution,
    including checkpoints already extracted before this fix existed.
    """
    for sub in ("fastpitch", "hifigan"):
        config_path = os.path.join(checkpoint_dir, sub, "config.json")
        if not os.path.exists(config_path):
            continue
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if _rewrite_config_file_paths(config, checkpoint_dir):
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)


def _ensure_indic_tts_checkpoint(lang: str) -> str:
    """
    Download + extract this language's FastPitch+HiFi-GAN checkpoint bundle
    from AI4Bharat/Indic-TTS's GitHub Release (~1.5GB, MIT licensed, first
    use only) if not already present. Returns a directory containing
    `fastpitch/` and `hifigan/` subfolders directly.
    """
    zip_lang = INDIC_TTS_LANG_ZIP_MAP.get(lang)
    if zip_lang is None:
        raise RuntimeError(f"No Indic-TTS checkpoint mapping for language '{lang}'.")

    lang_dir = os.path.join(INDIC_TTS_CHECKPOINTS_DIR, lang)
    if os.path.isdir(lang_dir):
        checkpoint_dir = _resolve_checkpoint_dir(lang_dir)
        if checkpoint_dir:
            _patch_checkpoint_config_paths(checkpoint_dir)
            return checkpoint_dir

    os.makedirs(lang_dir, exist_ok=True)
    zip_path = os.path.join(INDIC_TTS_CHECKPOINTS_DIR, f"{zip_lang}.zip")
    url = f"{INDIC_TTS_RELEASE_BASE_URL}/{zip_lang}.zip"
    print(f"[delivery] Downloading Indic-TTS checkpoint for '{lang}' "
          f"(~1.5GB, first use only)...")
    _download_file_streamed(url, zip_path)
    print("[delivery]   -> extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(lang_dir)
    os.remove(zip_path)

    checkpoint_dir = _resolve_checkpoint_dir(lang_dir)
    if not checkpoint_dir:
        raise RuntimeError(
            f"Downloaded Indic-TTS checkpoint for '{lang}' but couldn't find "
            f"expected fastpitch/hifigan subfolders under {lang_dir}. The "
            "release zip's internal layout may not match what this code "
            "expects -- inspect it manually."
        )
    _patch_checkpoint_config_paths(checkpoint_dir)
    return checkpoint_dir


def synthesize_with_indic_tts(text: str, lang: str, out_wav_path: str,
                               speaker: Optional[str] = None) -> str:
    """
    AI4Bharat/Indic-TTS: language-specific FastPitch (acoustic) + HiFi-GAN
    (vocoder) synthesis, run via subprocess in the isolated .venv-tts (see
    _get_tts_venv_python). speaker: "male" or "female"; defaults to
    INDIC_TTS_DEFAULT_SPEAKER (brx has no male speaker upstream).
    """
    checkpoint_dir = _ensure_indic_tts_checkpoint(lang)
    venv_python = _get_tts_venv_python()
    speaker = speaker or INDIC_TTS_DEFAULT_SPEAKER
    if lang == "brx" and speaker == "male":
        speaker = "female"

    proc = subprocess.run(
        [venv_python, INDIC_TTS_WORKER_SCRIPT,
         "--checkpoint-dir", checkpoint_dir,
         "--text", text,
         "--speaker", speaker,
         "--out", out_wav_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Indic-TTS synthesis failed: {proc.stderr.decode(errors='ignore')}")
    return out_wav_path


def _get_sequential_timeline(segment_starts: List[float], segment_durations: List[float], min_gap_sec: float = 0.05) -> List[float]:
    """Return start times that preserve natural gaps but prevent one clip from overlapping the next."""
    if not segment_starts:
        return []

    adjusted_starts = []
    cursor_end = 0.0
    for index, (start, duration) in enumerate(zip(segment_starts, segment_durations)):
        if duration <= 0:
            duration = 0.01
        if index == 0:
            desired_start = start
        else:
            desired_start = max(start, cursor_end + min_gap_sec)
        adjusted_starts.append(desired_start)
        cursor_end = desired_start + duration
    return adjusted_starts


def build_voiceover_track(segments: List[Segment], target_lang: str, work_dir: str,
                           speaker: Optional[str] = None) -> str:
    """
    Synthesize one WAV per segment with Indic-TTS, time-stretch each to fit
    its original segment slot, then stitch them onto a single timeline
    (silence-padded to match original segment timestamps) using ffmpeg's
    concat + adelay filters. Returns path to the mixed WAV.

    speaker: optional "male"/"female" override (see synthesize_with_indic_tts).
    """
    tts_dir = os.path.join(work_dir, "tts_segments")
    os.makedirs(tts_dir, exist_ok=True)

    segment_wavs = []
    for seg in segments:
        text = seg.translated_text if seg.translated_text is not None else seg.text
        if not text.strip():
            continue
        seg_wav = os.path.join(tts_dir, f"seg_{seg.index:05d}.wav")
        synthesize_with_indic_tts(text, target_lang, seg_wav, speaker)

        final_wav = seg_wav
        if VOICEOVER_TIME_STRETCH:
            slot = seg.end - seg.start
            fitted = os.path.join(tts_dir, f"seg_{seg.index:05d}_fit.wav")
            final_wav = _time_stretch_to_fit(seg_wav, fitted, slot)

        segment_wavs.append((seg.start, final_wav, _get_wav_duration(final_wav)))

    if not segment_wavs:
        raise RuntimeError("No segments produced TTS audio.")

    starts = [item[0] for item in segment_wavs]
    durations = [item[2] for item in segment_wavs]
    adjusted_starts = _get_sequential_timeline(starts, durations)

    # Build an ffmpeg filter_complex that delays each clip to its start time and mixes.
    inputs = []
    filter_parts = []
    mix_labels = []
    for i, (original_start, wav_path, _) in enumerate(segment_wavs):
        inputs += ["-i", wav_path]
        delay_ms = max(0, int(adjusted_starts[i] * 1000))
        filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
        mix_labels.append(f"[a{i}]")

    filter_complex = ";".join(filter_parts) + ";" + "".join(mix_labels) + \
        f"amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=0[out]"

    out_wav = os.path.join(work_dir, "voiceover_mixed.wav")
    cmd = [FFMPEG_BINARY, "-y"] + inputs + [
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
            FFMPEG_BINARY, "-y", "-i", video_path, "-i", voiceover_wav,
            "-filter_complex",
            f"[0:a]volume={original_audio_gain_db}dB[orig];[orig][1:a]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy",
            out_mp4_path
        ])
    else:
        _run([
            FFMPEG_BINARY, "-y", "-i", video_path, "-i", voiceover_wav,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-shortest",
            out_mp4_path
        ])
    return out_mp4_path
