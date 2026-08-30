"""
Delivery — final media generation.

Produces:

1. Burned-in MP4
2. Voiceover MP4
3. Voiceover + burned-in subtitles
4. Voiceover quality metrics
5. Automatic translation quality metrics

Indic-TTS runs in the isolated .venv-tts environment.
"""

import json
import os
import re
import subprocess
import wave
import zipfile
from typing import Dict, List, Optional

from config import (
    FFMPEG_BINARY,
    INDIC_TTS_VENV_DIR,
    INDIC_TTS_WORKER_SCRIPT,
    INDIC_TTS_CHECKPOINTS_DIR,
    INDIC_TTS_RELEASE_BASE_URL,
    INDIC_TTS_LANG_ZIP_MAP,
    INDIC_TTS_DEFAULT_SPEAKER,
    VOICEOVER_TIME_STRETCH,
    VOICEOVER_MAX_TEMPO_RATIO,
    VOICEOVER_MAX_DRIFT_SEC,
    VOICEOVER_CATCHUP_MAX_TEMPO_RATIO,
    BURN_IN_ENCODE_PRESET,
)

from pipeline.stage3_segment_tm import Segment


class NoVoiceoverContentError(RuntimeError):
    """
    Raised by build_voiceover_track() when every segment's translated
    text turned out to be empty or synthesizable-content-free (e.g. a
    video where ASR hallucination consumed most of the runtime, leaving
    only punctuation-only fragments in what survived). A distinct type
    from a generic RuntimeError so callers can degrade gracefully
    (skip voiceover, still deliver subtitles) instead of failing the
    whole job outright -- see orchestrator.py's handling.
    """


def _run(cmd):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\n"
            f"{proc.stderr.decode(errors='ignore')}"
        )

    return proc.stdout


def _escape_ffmpeg_filter_path(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    return p.replace(":", "\\:")


def burn_in_subtitles(
    video_path: str,
    srt_path: str,
    out_mp4_path: str,
) -> str:

    escaped_srt = _escape_ffmpeg_filter_path(
        srt_path
    )

    _run([
        FFMPEG_BINARY,
        "-y",
        "-i",
        video_path,
        "-vf",
        f"subtitles='{escaped_srt}'",
        "-preset",
        BURN_IN_ENCODE_PRESET,
        "-c:a",
        "copy",
        out_mp4_path,
    ])

    return out_mp4_path


def voiceover_available(lang: str) -> bool:
    return lang in INDIC_TTS_LANG_ZIP_MAP


def _get_wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as w:
        return w.getnframes() / float(
            w.getframerate()
        )


# =========================================================
# Quality helpers
# =========================================================

def _numeric_count(text: str) -> int:
    """
    Count integer/decimal expressions in text.

    This is used only for QC metrics.
    """

    if not text:
        return 0

    return len(
        re.findall(
            r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])",
            text,
        )
    )


def _extract_numeric_values(
    text: str,
) -> List[str]:
    """
    Extract numeric expressions from text.
    """

    if not text:
        return []

    return re.findall(
        r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])",
        text,
    )


def _calculate_alignment_score(
    segment_slot: float,
    generated_duration: float,
) -> float:

    if segment_slot <= 0:
        return 0.0

    difference = abs(
        generated_duration - segment_slot
    )

    relative_error = (
        difference / segment_slot
    )

    score = max(
        0.0,
        100.0 - relative_error * 100.0,
    )

    return round(
        min(score, 100.0),
        2,
    )


def _translation_completeness_score(
    source_text: str,
    translated_text: str,
) -> float:
    """
    Estimate whether the translation contains a
    reasonable amount of content compared with the
    source.

    This is NOT semantic translation accuracy.

    It detects:
    - missing translations
    - extremely short translations
    - extremely long/suspicious translations
    """

    source = (
        source_text or ""
    ).strip()

    translated = (
        translated_text or ""
    ).strip()

    if not source:
        return 100.0

    if not translated:
        return 0.0

    source_words = len(
        source.split()
    )

    translated_words = len(
        translated.split()
    )

    if source_words == 0:
        return 100.0

    ratio = (
        translated_words
        / source_words
    )

    # Translation length can vary significantly
    # between languages. Only penalize extreme cases.
    if 0.50 <= ratio <= 2.00:
        return 100.0

    if ratio < 0.50:
        return round(
            max(
                0.0,
                ratio / 0.50 * 100.0,
            ),
            2,
        )

    # ratio > 2.0
    return round(
        max(
            0.0,
            100.0 - (
                (ratio - 2.0)
                * 25.0
            ),
        ),
        2,
    )


def _numeric_preservation_score(
    source_text: str,
    translated_text: str,
) -> float:
    """
    Estimate whether numeric expressions in the
    source are preserved in the translation.

    This does NOT verify semantic correctness.

    Example:

        Source:      Apply 25 kg.
        Translation: Apply 25 kg.

    -> 100

    If the source contains 25 and 50 but the
    translation contains only 25:

    -> 50
    """

    source_numbers = (
        _extract_numeric_values(
            source_text
        )
    )

    if not source_numbers:
        return 100.0

    translated_numbers = (
        _extract_numeric_values(
            translated_text
        )
    )

    if not translated_numbers:
        return 0.0

    # Use counts rather than sets so repeated numbers
    # are also considered.
    remaining = list(
        translated_numbers
    )

    preserved = 0

    for number in source_numbers:

        if number in remaining:

            preserved += 1

            remaining.remove(
                number
            )

    return round(
        preserved
        / len(source_numbers)
        * 100.0,
        2,
    )


def _calculate_translation_quality(
    segments: List[Segment],
) -> Dict:
    """
    Calculate an automatic translation-quality proxy.

    Components:

        40% ASR confidence
        35% translation completeness
        25% numeric preservation

    IMPORTANT:

    This is NOT a human-verified semantic translation
    accuracy score.

    Without a reference translation, the system cannot
    reliably determine whether the translated meaning
    is correct.
    """

    if not segments:

        return {
            "overall_translation_quality_score": 0.0,
            "average_asr_confidence": 0.0,
            "translation_completeness_score": 0.0,
            "numeric_preservation_score": 0.0,
            "translation_segments": 0,
            "translation_warnings": [],
        }

    confidence_scores = []
    completeness_scores = []
    numeric_scores = []

    warnings = []

    for seg in segments:

        source = (
            seg.text or ""
        ).strip()

        translated = (
            seg.translated_text
            or ""
        ).strip()

        confidence = float(
            getattr(
                seg,
                "confidence",
                0.0,
            )
        )

        # ASR confidence is normally 0–1.
        asr_score = max(
            0.0,
            min(
                100.0,
                confidence * 100.0,
            ),
        )

        confidence_scores.append(
            asr_score
        )

        completeness = (
            _translation_completeness_score(
                source,
                translated,
            )
        )

        completeness_scores.append(
            completeness
        )

        numeric_score = (
            _numeric_preservation_score(
                source,
                translated,
            )
        )

        numeric_scores.append(
            numeric_score
        )

        if confidence < 0.55:

            warnings.append(
                f"Segment {seg.index}: "
                f"low ASR confidence "
                f"({confidence:.2f})."
            )

        if source and not translated:

            warnings.append(
                f"Segment {seg.index}: "
                "missing translation."
            )

        if numeric_score < 100:

            warnings.append(
                f"Segment {seg.index}: "
                "one or more numeric expressions "
                "may not have been preserved."
            )

    avg_asr = (
        sum(confidence_scores)
        / len(confidence_scores)
    )

    avg_completeness = (
        sum(completeness_scores)
        / len(completeness_scores)
    )

    avg_numeric = (
        sum(numeric_scores)
        / len(numeric_scores)
    )

    overall = (
        avg_asr * 0.40
        + avg_completeness * 0.35
        + avg_numeric * 0.25
    )

    return {
        "overall_translation_quality_score":
            round(
                max(
                    0.0,
                    min(
                        100.0,
                        overall,
                    ),
                ),
                2,
            ),

        "average_asr_confidence":
            round(
                avg_asr,
                2,
            ),

        "translation_completeness_score":
            round(
                avg_completeness,
                2,
            ),

        "numeric_preservation_score":
            round(
                avg_numeric,
                2,
            ),

        "translation_segments":
            len(segments),

        "translation_warnings":
            warnings,
    }


def calculate_translation_quality(
    segments: List[Segment],
) -> Dict:
    """
    Public wrapper used by orchestrator.py.
    """

    return _calculate_translation_quality(
        segments
    )


# =========================================================
# TTS timing
# =========================================================

_MIN_VOICEOVER_GAP_SEC = 0.05


def _build_atempo_chain(
    ratio: float,
    max_ratio: float = VOICEOVER_MAX_TEMPO_RATIO,
) -> str:

    ratio = max(
        1.0 / max_ratio,
        min(
            max_ratio,
            ratio,
        ),
    )

    stages = []

    remaining = ratio

    while remaining > 2.0:

        stages.append(
            2.0
        )

        remaining /= 2.0

    while remaining < 0.5:

        stages.append(
            0.5
        )

        remaining /= 0.5

    stages.append(
        remaining
    )

    return ",".join(
        f"atempo={s:.4f}"
        for s in stages
    )


def _time_stretch_to_fit(
    in_wav_path: str,
    out_wav_path: str,
    target_duration: float,
    max_ratio: float = VOICEOVER_MAX_TEMPO_RATIO,
) -> str:

    current = _get_wav_duration(
        in_wav_path
    )

    if (
        target_duration <= 0
        or current <= 0
    ):
        return in_wav_path

    ratio = (
        current
        / target_duration
    )

    if abs(
        ratio - 1.0
    ) < 0.03:

        return in_wav_path

    filt = _build_atempo_chain(
        ratio,
        max_ratio,
    )

    _run([
        FFMPEG_BINARY,
        "-y",
        "-i",
        in_wav_path,
        "-filter:a",
        filt,
        out_wav_path,
    ])

    return out_wav_path


# =========================================================
# Indic-TTS
# =========================================================

def _get_tts_venv_python() -> str:

    venv_python = os.path.join(
        INDIC_TTS_VENV_DIR,
        "bin",
        "python3",
    )

    if os.name == "nt":

        venv_python = os.path.join(
            INDIC_TTS_VENV_DIR,
            "Scripts",
            "python.exe",
        )

    if not os.path.exists(
        venv_python
    ):

        raise RuntimeError(
            "Voiceover TTS venv not set up. "
            "From the project root, run:\n"
            f"python3 -m venv "
            f"{INDIC_TTS_VENV_DIR}\n"
            f"{venv_python} -m pip install "
            "-r tts_worker/requirements.txt"
        )

    return venv_python


def _find_checkpoint_subdir(
    root: str,
    name: str,
) -> Optional[str]:

    for dirpath, dirnames, _ in os.walk(
        root
    ):

        if name in dirnames:

            return os.path.join(
                dirpath,
                name,
            )

    return None


def _download_file_streamed(
    url: str,
    dest_path: str,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 5,
):

    import time
    import ssl
    import urllib.request

    try:

        import certifi

        ctx = (
            ssl.create_default_context(
                cafile=certifi.where()
            )
        )

    except ImportError:

        ctx = (
            ssl.create_default_context()
        )

    tmp_path = (
        dest_path
        + ".part"
    )

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            print(
                f"[delivery] Download attempt "
                f"{attempt}/{max_retries}: {url}"
            )

            downloaded = 0

            with urllib.request.urlopen(
                urllib.request.Request(
                    url,
                    headers={
                        "User-Agent":
                        "Offline-Field-Translator/1.0"
                    },
                ),
                context=ctx,
                timeout=120,
            ) as resp:

                total = resp.headers.get(
                    "Content-Length"
                )

                total = (
                    int(total)
                    if total
                    else None
                )

                with open(
                    tmp_path,
                    "wb",
                ) as out:

                    while True:

                        chunk = resp.read(
                            chunk_size
                        )

                        if not chunk:
                            break

                        out.write(
                            chunk
                        )

                        downloaded += (
                            len(chunk)
                        )

                        if total:

                            percent = (
                                downloaded
                                * 100
                                / total
                            )

                            print(
                                f"\r[delivery] "
                                f"{downloaded/(1024**3):.2f}"
                                f" / "
                                f"{total/(1024**3):.2f} GB "
                                f"({percent:.1f}%)",
                                end="",
                                flush=True,
                            )

            print()

            if (
                not os.path.exists(
                    tmp_path
                )
                or os.path.getsize(
                    tmp_path
                ) == 0
            ):

                raise RuntimeError(
                    "Downloaded file is empty."
                )

            print(
                "[delivery]   -> validating ZIP..."
            )

            with zipfile.ZipFile(
                tmp_path,
                "r",
            ) as zf:

                bad_file = (
                    zf.testzip()
                )

            if bad_file is not None:

                raise RuntimeError(
                    "ZIP validation failed: "
                    f"{bad_file}"
                )

            os.replace(
                tmp_path,
                dest_path,
            )

            print(
                "[delivery]   -> download "
                "verified successfully"
            )

            return

        except Exception as exc:

            print(
                f"\n[delivery] Download attempt "
                f"{attempt} failed: {exc}"
            )

            try:

                if os.path.exists(
                    tmp_path
                ):

                    os.remove(
                        tmp_path
                    )

            except OSError:
                pass

            if attempt == max_retries:

                raise RuntimeError(
                    "Failed to download valid "
                    f"checkpoint after "
                    f"{max_retries} attempts: "
                    f"{url}"
                ) from exc

            wait = min(
                2 ** attempt,
                30,
            )

            print(
                f"[delivery] Retrying in "
                f"{wait} seconds..."
            )

            time.sleep(
                wait
            )


def _resolve_checkpoint_dir(
    lang_dir: str,
) -> Optional[str]:

    fastpitch_dir = (
        _find_checkpoint_subdir(
            lang_dir,
            "fastpitch",
        )
    )

    hifigan_dir = (
        _find_checkpoint_subdir(
            lang_dir,
            "hifigan",
        )
    )

    if not (
        fastpitch_dir
        and hifigan_dir
    ):

        return None

    parent = os.path.dirname(
        fastpitch_dir
    )

    if parent != os.path.dirname(
        hifigan_dir
    ):

        return None

    # An interrupted/truncated download can leave both folders present
    # with just their (tiny) config.json -- without this check that
    # satisfies the check above and gets cached as "already downloaded"
    # forever, permanently failing synthesis instead of ever retrying.
    if not (
        os.path.exists(
            os.path.join(fastpitch_dir, "best_model.pth")
        )
        and os.path.exists(
            os.path.join(hifigan_dir, "best_model.pth")
        )
    ):

        return None

    return parent


def _find_file_by_name(
    root: str,
    filename: str,
) -> Optional[str]:

    for dirpath, _, filenames in os.walk(
        root
    ):

        if filename in filenames:

            return os.path.join(
                dirpath,
                filename,
            )

    return None


def _rewrite_config_file_paths(
    obj,
    checkpoint_dir: str,
) -> bool:

    changed = False

    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            if (
                isinstance(
                    value,
                    str,
                )
                and value.lower().endswith(
                    (
                        ".pth",
                        ".npy",
                        ".npz",
                    )
                )
            ):

                real_path = (
                    _find_file_by_name(
                        checkpoint_dir,
                        os.path.basename(
                            value
                        ),
                    )
                )

                if (
                    real_path
                    and os.path.abspath(
                        real_path
                    )
                    != os.path.abspath(
                        value
                    )
                ):

                    obj[key] = real_path
                    changed = True

            elif isinstance(
                value,
                (
                    dict,
                    list,
                ),
            ):

                if _rewrite_config_file_paths(
                    value,
                    checkpoint_dir,
                ):

                    changed = True

    elif isinstance(
        obj,
        list,
    ):

        for item in obj:

            if _rewrite_config_file_paths(
                item,
                checkpoint_dir,
            ):

                changed = True

    return changed


def _patch_checkpoint_config_paths(
    checkpoint_dir: str,
):

    for sub in (
        "fastpitch",
        "hifigan",
    ):

        config_path = os.path.join(
            checkpoint_dir,
            sub,
            "config.json",
        )

        if not os.path.exists(
            config_path
        ):

            continue

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:

            config = json.load(f)

        if _rewrite_config_file_paths(
            config,
            checkpoint_dir,
        ):

            with open(
                config_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    config,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )


def _ensure_indic_tts_checkpoint(
    lang: str,
) -> str:

    zip_lang = (
        INDIC_TTS_LANG_ZIP_MAP.get(
            lang
        )
    )

    if zip_lang is None:

        raise RuntimeError(
            f"No Indic-TTS checkpoint mapping "
            f"for language '{lang}'."
        )

    lang_dir = os.path.join(
        INDIC_TTS_CHECKPOINTS_DIR,
        lang,
    )

    if os.path.isdir(
        lang_dir
    ):

        checkpoint_dir = (
            _resolve_checkpoint_dir(
                lang_dir
            )
        )

        if checkpoint_dir:

            _patch_checkpoint_config_paths(
                checkpoint_dir
            )

            return checkpoint_dir

    os.makedirs(
        lang_dir,
        exist_ok=True,
    )

    zip_path = os.path.join(
        INDIC_TTS_CHECKPOINTS_DIR,
        f"{zip_lang}.zip",
    )

    url = (
        f"{INDIC_TTS_RELEASE_BASE_URL}/"
        f"{zip_lang}.zip"
    )

    print(
        f"[delivery] Downloading Indic-TTS "
        f"checkpoint for '{lang}'..."
    )

    _download_file_streamed(
        url,
        zip_path,
    )

    print(
        "[delivery]   -> extracting..."
    )

    with zipfile.ZipFile(
        zip_path
    ) as zf:

        zf.extractall(
            lang_dir
        )

    os.remove(
        zip_path
    )

    checkpoint_dir = (
        _resolve_checkpoint_dir(
            lang_dir
        )
    )

    if not checkpoint_dir:

        raise RuntimeError(
            f"Downloaded Indic-TTS checkpoint "
            f"for '{lang}' but couldn't find "
            "fastpitch/hifigan."
        )

    _patch_checkpoint_config_paths(
        checkpoint_dir
    )

    return checkpoint_dir


def synthesize_with_indic_tts(
    text: str,
    lang: str,
    out_wav_path: str,
    speaker: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> Dict:
    """
    Single-shot synthesis: spawns a fresh TTS worker subprocess, which
    loads the checkpoint, synthesizes one piece of text, and exits.

    Kept for one-off/manual use. For anything with more than one segment
    (i.e. build_voiceover_track below), use IndicTTSWorker instead --
    calling this function per segment means reloading the ~1.5GB
    FastPitch+HiFiGAN checkpoint pair from disk for every single segment.
    """

    checkpoint_dir = (
        _ensure_indic_tts_checkpoint(
            lang
        )
    )

    venv_python = (
        _get_tts_venv_python()
    )

    speaker = (
        speaker
        or INDIC_TTS_DEFAULT_SPEAKER
    )

    if (
        lang == "brx"
        and speaker == "male"
    ):

        speaker = "female"

    cmd = [
        venv_python,
        INDIC_TTS_WORKER_SCRIPT,

        "--checkpoint-dir",
        checkpoint_dir,

        "--text",
        text,

        "--lang",
        lang,

        "--speaker",
        speaker,

        "--out",
        out_wav_path,
    ]

    if metadata_path:

        cmd.extend([
            "--metadata-out",
            metadata_path,
        ])

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stderr = proc.stderr.decode(
        errors="ignore"
    )

    if proc.returncode != 0:

        raise RuntimeError(
            "Indic-TTS synthesis failed:\n"
            + stderr
        )

    metadata = {
        "success": True,
        "input_text": text,
        "output_wav": out_wav_path,
        "stderr": stderr,
    }

    if (
        metadata_path
        and os.path.exists(
            metadata_path
        )
    ):

        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as f:

                worker_metadata = (
                    json.load(f)
                )

            metadata.update(
                worker_metadata
            )

        except Exception:
            pass

    return metadata


class IndicTTSWorker:
    """
    Persistent Indic-TTS worker process.

    Spawns tts_worker/synthesize.py in --serve mode ONCE per
    (lang, speaker), which loads the FastPitch+HiFiGAN checkpoint a
    single time and then keeps the process alive to synthesize many
    segments over stdin/stdout. This replaces spawning a fresh
    subprocess (and reloading the checkpoint) for every segment.

    Usage:

        worker = IndicTTSWorker(checkpoint_dir, lang, speaker,
                                 venv_python, worker_script)
        try:
            for seg in segments:
                metadata = worker.synthesize(text, seg_wav, metadata_path)
        finally:
            worker.close()
    """

    def __init__(
        self,
        checkpoint_dir: str,
        lang: str,
        speaker: str,
        venv_python: str,
        worker_script: str,
        startup_timeout_sec: float = 300.0,
    ):
        self._proc = subprocess.Popen(
            [
                venv_python,
                worker_script,
                "--checkpoint-dir",
                checkpoint_dir,
                "--lang",
                lang,
                "--speaker",
                speaker,
                "--serve",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Block until the worker reports it has finished loading the
        # checkpoint (or dies trying). This is the one place per job we
        # pay the checkpoint-load cost.
        ready_line = self._proc.stdout.readline()

        if not ready_line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(
                "Indic-TTS worker exited before becoming ready:\n"
                + stderr
            )

        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError:
            stderr = self._proc.stderr.read()
            raise RuntimeError(
                "Indic-TTS worker sent unexpected startup output "
                f"({ready_line!r}):\n{stderr}"
            )

        if not ready.get("ready"):
            stderr = self._proc.stderr.read()
            raise RuntimeError(
                "Indic-TTS worker failed to start:\n" + stderr
            )

    def synthesize(
        self,
        text: str,
        out_wav_path: str,
        metadata_path: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> Dict:

        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read()
            raise RuntimeError(
                "Indic-TTS worker process is no longer running:\n"
                + stderr
            )

        request = {
            "text": text,
            "out": out_wav_path,
        }

        if metadata_path:
            request["metadata_out"] = metadata_path

        if speaker:
            request["speaker"] = speaker

        self._proc.stdin.write(
            json.dumps(request, ensure_ascii=False) + "\n"
        )
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()

        if not line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(
                "Indic-TTS worker died mid-synthesis:\n" + stderr
            )

        result = json.loads(line)

        if not result.get("ok"):
            raise RuntimeError(
                "Indic-TTS synthesis failed: "
                + str(result.get("error"))
            )

        return result

    def close(self):

        try:
            if self._proc.poll() is None:
                self._proc.stdin.write(
                    json.dumps({"cmd": "shutdown"}) + "\n"
                )
                self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.terminate()


# =========================================================
# Timeline
# =========================================================

def _get_sequential_timeline(
    segment_starts: List[float],
    segment_durations: List[float],
    min_gap_sec: float = 0.05,
) -> List[float]:

    if not segment_starts:
        return []

    adjusted_starts = []

    cursor_end = 0.0

    for index, (
        start,
        duration,
    ) in enumerate(
        zip(
            segment_starts,
            segment_durations,
        )
    ):

        if duration <= 0:
            duration = 0.01

        if index == 0:

            desired_start = start

        else:

            desired_start = max(
                start,
                cursor_end
                + min_gap_sec,
            )

        adjusted_starts.append(
            desired_start
        )

        cursor_end = (
            desired_start
            + duration
        )

    return adjusted_starts


# =========================================================
# Voiceover generation + quality metrics
# =========================================================

def build_voiceover_track(
    segments: List[Segment],
    target_lang: str,
    work_dir: str,
    speaker: Optional[str] = None,
):

    tts_dir = os.path.join(
        work_dir,
        "tts_segments",
    )

    os.makedirs(
        tts_dir,
        exist_ok=True,
    )

    segment_wavs = []

    quality = {
        "voiceover_requested": True,
        "target_language": target_lang,
        "total_segments": len(
            segments
        ),

        "tts_success_segments": 0,
        "tts_failed_segments": 0,

        "tts_coverage_percent": 0.0,

        "numeric_segments": 0,
        "numeric_expressions": 0,
        "numeric_normalization_success": 0,
        "numeric_normalization_failures": 0,
        "numeric_normalization_percent": 100.0,
        "decimal_expressions": 0,

        "alignment_scores": [],
        "average_alignment_score": 0.0,

        "segment_details": [],
        "warnings": [],
    }

    # -----------------------------------------------------
    # Resolve the checkpoint + venv ONCE, and start a single
    # persistent TTS worker process for the whole job instead
    # of spawning (and reloading the checkpoint for) one
    # subprocess per segment.
    # -----------------------------------------------------

    checkpoint_dir = _ensure_indic_tts_checkpoint(target_lang)
    venv_python = _get_tts_venv_python()

    effective_speaker = speaker or INDIC_TTS_DEFAULT_SPEAKER

    if target_lang == "brx" and effective_speaker == "male":
        effective_speaker = "female"

    worker = IndicTTSWorker(
        checkpoint_dir,
        target_lang,
        effective_speaker,
        venv_python,
        INDIC_TTS_WORKER_SCRIPT,
    )

    cursor_end = 0.0

    try:
        for seg in segments:

            text = (
                seg.translated_text
                if seg.translated_text is not None
                else seg.text
            )

            # Punctuation-only text (e.g. a lone "," left over from a
            # translation/segmentation edge case) is not just "empty" by
            # .strip(), but crashes Indic-TTS's FastPitch model the same
            # way real emptiness would -- verified directly against a
            # real job: translated_text=="," fed to synthesize() raised
            # "Given groups=1, weight of size [256, 512, 3], expected
            # input[1, 1, 512] to have 512 channels, but got 1 channels
            # instead" (a degenerate phoneme sequence breaking FastPitch's
            # conv1d shapes), so treat "no actual word characters" the
            # same as empty rather than attempting synthesis.
            if not re.search(r"\w", text, flags=re.UNICODE):

                quality[
                    "warnings"
                ].append(
                    f"Segment {seg.index}: "
                    "empty or punctuation-only translation skipped."
                )

                quality[
                    "segment_details"
                ].append({
                    "segment": seg.index,
                    "source_text": seg.text,
                    "translated_text": text,
                    "numeric_count": 0,
                    "status": "SKIPPED",
                })

                continue

            seg_wav = os.path.join(
                tts_dir,
                f"seg_{seg.index:05d}.wav",
            )

            metadata_path = os.path.join(
                tts_dir,
                f"seg_{seg.index:05d}.json",
            )

            numeric_count = _numeric_count(
                text
            )

            if numeric_count > 0:

                quality[
                    "numeric_segments"
                ] += 1

            try:

                metadata = worker.synthesize(
                    text,
                    seg_wav,
                    metadata_path,
                )

                quality[
                    "tts_success_segments"
                ] += 1

                numeric_expressions = (
                    metadata.get(
                        "numeric_expressions",
                        [],
                    )
                )

                for item in numeric_expressions:

                    quality[
                        "numeric_expressions"
                    ] += 1

                    if item.get(
                        "success"
                    ):

                        quality[
                            "numeric_normalization_success"
                        ] += 1

                    else:

                        quality[
                            "numeric_normalization_failures"
                        ] += 1

                    if (
                        item.get("type")
                        == "decimal"
                    ):

                        quality[
                            "decimal_expressions"
                        ] += 1

                final_wav = seg_wav

                slot = (
                    seg.end
                    - seg.start
                )

                placed_start = seg.start

                if VOICEOVER_TIME_STRETCH:

                    raw_duration = (
                        _get_wav_duration(
                            seg_wav
                        )
                    )

                    earliest_start = max(
                        seg.start,
                        cursor_end + _MIN_VOICEOVER_GAP_SEC,
                    )

                    # Natural, gentle fit to this segment's own slot -- what
                    # we'd use if timing weren't a concern at all.
                    natural_target = min(
                        raw_duration * VOICEOVER_MAX_TEMPO_RATIO,
                        max(
                            raw_duration / VOICEOVER_MAX_TEMPO_RATIO,
                            slot,
                        ),
                    )

                    # Cap on how long THIS segment's audio may run so it
                    # doesn't add more than VOICEOVER_MAX_DRIFT_SEC of new
                    # desync by the time it ends -- applied to every segment,
                    # not only ones already in backlog, so a single segment
                    # whose translated text is much longer than its slot can't
                    # single-handedly saddle every later segment with an
                    # unbounded backlog the way naive sequential placement did.
                    drift_cap_target = (
                        (seg.end + VOICEOVER_MAX_DRIFT_SEC)
                        - earliest_start
                    )

                    target_duration = min(
                        natural_target,
                        drift_cap_target,
                    )

                    # Physical intelligibility floor: never compress past this,
                    # even if that means this one segment still overruns its
                    # drift budget (unavoidable when translated content is
                    # simply much longer than its assigned slot -- the
                    # important part is that this doesn't compound into every
                    # later segment too, which the cap above prevents).
                    target_duration = max(
                        target_duration,
                        raw_duration / VOICEOVER_CATCHUP_MAX_TEMPO_RATIO,
                    )

                    # target_duration is already bounded (by construction above)
                    # to raw_duration * [1/VOICEOVER_CATCHUP_MAX_TEMPO_RATIO,
                    # VOICEOVER_MAX_TEMPO_RATIO], so the implied ratio can never
                    # exceed the catch-up ratio -- pass it as the outer safety
                    # bound rather than recomputing a tighter one here.
                    fitted = os.path.join(
                        tts_dir,
                        f"seg_{seg.index:05d}_fit.wav",
                    )

                    final_wav = (
                        _time_stretch_to_fit(
                            seg_wav,
                            fitted,
                            target_duration,
                            VOICEOVER_CATCHUP_MAX_TEMPO_RATIO,
                        )
                    )

                    placed_start = earliest_start

                duration = (
                    _get_wav_duration(
                        final_wav
                    )
                )

                cursor_end = (
                    placed_start
                    + duration
                )

                alignment_score = (
                    _calculate_alignment_score(
                        slot,
                        duration,
                    )
                )

                quality[
                    "alignment_scores"
                ].append(
                    alignment_score
                )

                quality[
                    "segment_details"
                ].append({
                    "segment": seg.index,
                    "start": seg.start,
                    "end": seg.end,
                    # Where the dubbed audio actually landed, which can
                    # differ from seg.start/seg.end above once drift
                    # catch-up shifts a segment's placement -- used to
                    # resync burned-in captions to the audible dub rather
                    # than the original (pre-drift) ASR timing. See
                    # orchestrator.py's burned-in-on-voiceover step.
                    "placed_start": placed_start,
                    "placed_end": placed_start + duration,
                    "source_text": seg.text,
                    "translated_text": text,
                    "tts_duration": duration,
                    "slot_duration": slot,
                    "alignment_score":
                        alignment_score,
                    "numeric_count":
                        numeric_count,
                    "numeric_normalization":
                        metadata.get(
                            "numeric_expressions",
                            [],
                        ),
                    "status": "OK",
                })

                segment_wavs.append(
                    (
                        placed_start,
                        final_wav,
                        duration,
                    )
                )

            except Exception as exc:

                quality[
                    "tts_failed_segments"
                ] += 1

                quality[
                    "warnings"
                ].append(
                    f"Segment {seg.index}: "
                    f"TTS failed — {exc}"
                )

                quality[
                    "segment_details"
                ].append({
                    "segment": seg.index,
                    "source_text": seg.text,
                    "translated_text": text,
                    "numeric_count":
                        numeric_count,
                    "status": "FAILED",
                    "error": str(exc),
                })

    finally:
        worker.close()

    if not segment_wavs:

        raise NoVoiceoverContentError(
            "No segments produced TTS audio -- every segment's "
            "translated text was empty or had no synthesizable content."
        )

    # -----------------------------------------------------
    # Timeline
    # -----------------------------------------------------

    starts = [
        item[0]
        for item in segment_wavs
    ]

    if VOICEOVER_TIME_STRETCH:

        # Placement was already resolved per-segment above (drift-bounded
        # against each segment's original slot), so `starts` already holds
        # final positions -- re-running plain sequential chaining here
        # would throw that bounding away.
        adjusted_starts = starts

    else:

        durations = [
            item[2]
            for item in segment_wavs
        ]

        adjusted_starts = (
            _get_sequential_timeline(
                starts,
                durations,
            )
        )

    # -----------------------------------------------------
    # FFmpeg mixing
    # -----------------------------------------------------

    inputs = []
    filter_parts = []
    mix_labels = []

    for i, (
        original_start,
        wav_path,
        _,
    ) in enumerate(
        segment_wavs
    ):

        inputs += [
            "-i",
            wav_path,
        ]

        delay_ms = max(
            0,
            int(
                adjusted_starts[i]
                * 1000
            ),
        )

        filter_parts.append(
            f"[{i}:a]"
            f"adelay={delay_ms}|{delay_ms}"
            f"[a{i}]"
        )

        mix_labels.append(
            f"[a{i}]"
        )

    filter_complex = (
        ";".join(
            filter_parts
        )
        + ";"
        + "".join(
            mix_labels
        )
        + f"amix=inputs="
        f"{len(mix_labels)}:"
        "duration=longest:"
        "dropout_transition=0:"
        # amix defaults to normalize=1, which divides the WHOLE mixed
        # output by the input count -- correct for genuinely simultaneous
        # sources, but wrong here: each segment is adelay-padded into its
        # own mostly-non-overlapping slot, so normalizing by segment count
        # (60-120+ per job) silently attenuated the entire voiceover by
        # 35-40+ dB, audible only at max system volume. Segments only
        # actually overlap in rare edge cases, where a hard sample-sum
        # clip is far preferable to uniformly crushing the whole track.
        "normalize=0"
        "[out]"
    )

    out_wav = os.path.join(
        work_dir,
        "voiceover_mixed.wav",
    )

    cmd = [
        FFMPEG_BINARY,
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        out_wav,
    ]

    _run(cmd)

    # -----------------------------------------------------
    # Final quality calculations
    # -----------------------------------------------------

    successful = quality[
        "tts_success_segments"
    ]

    total = quality[
        "total_segments"
    ]

    quality[
        "tts_coverage_percent"
    ] = round(
        (
            successful
            / total
            * 100
            if total
            else 0
        ),
        2,
    )

    numeric_total = quality[
        "numeric_expressions"
    ]

    numeric_success = quality[
        "numeric_normalization_success"
    ]

    quality[
        "numeric_normalization_percent"
    ] = round(
        (
            numeric_success
            / numeric_total
            * 100
            if numeric_total
            else 100
        ),
        2,
    )

    alignment_scores = quality[
        "alignment_scores"
    ]

    quality[
        "average_alignment_score"
    ] = round(
        (
            sum(
                alignment_scores
            )
            / len(
                alignment_scores
            )
            if alignment_scores
            else 0
        ),
        2,
    )

    # -----------------------------------------------------
    # Overall Voiceover Quality
    #
    # 35% TTS coverage
    # 30% number handling
    # 25% timing alignment
    # 10% base score
    # minus failure penalty
    # -----------------------------------------------------

    coverage_score = quality[
        "tts_coverage_percent"
    ]

    number_score = quality[
        "numeric_normalization_percent"
    ]

    alignment_score = quality[
        "average_alignment_score"
    ]

    failure_penalty = min(
        quality[
            "tts_failed_segments"
        ] * 5,
        20,
    )

    overall = (
        coverage_score * 0.35
        + number_score * 0.30
        + alignment_score * 0.25
        + 100 * 0.10
        - failure_penalty
    )

    quality[
        "overall_quality_score"
    ] = round(
        max(
            0,
            min(
                100,
                overall,
            ),
        ),
        2,
    )

    # -----------------------------------------------------
    # Human-readable warnings
    # -----------------------------------------------------

    if coverage_score < 100:

        quality[
            "warnings"
        ].append(
            "Some segments did not produce "
            "voiceover audio."
        )

    if (
        numeric_total > 0
        and quality[
            "numeric_normalization_percent"
        ] < 100
    ):

        quality[
            "warnings"
        ].append(
            "Some numeric expressions could "
            "not be normalized for TTS."
        )

    if (
        quality[
            "average_alignment_score"
        ] < 85
    ):

        quality[
            "warnings"
        ].append(
            "Voiceover timing differs "
            "noticeably from the original "
            "segment timing."
        )

    return out_wav, quality


# =========================================================
# Mux
# =========================================================

def mux_voiceover_onto_video(
    video_path: str,
    voiceover_wav: str,
    out_mp4_path: str,
    mix_original_audio: bool = False,
    original_audio_gain_db: float = -18.0,
) -> str:

    if mix_original_audio:

        _run([
            FFMPEG_BINARY,
            "-y",
            "-i",
            video_path,
            "-i",
            voiceover_wav,
            "-filter_complex",
            (
                f"[0:a]volume="
                f"{original_audio_gain_db}dB"
                "[orig];"
                "[orig][1:a]"
                "amix=inputs=2:"
                "duration=first:"
                "normalize=0"  # see build_voiceover_track's amix -- same normalize=1 pitfall
                "[aout]"
            ),
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            out_mp4_path,
        ])

    else:

        # The synthesized voiceover track is stitched from individual TTS
        # segments and is NOT guaranteed to reach the source video's exact
        # duration (e.g. a trailing outro/credits stretch after the last
        # subtitle segment has no corresponding dubbed audio). "-shortest"
        # alone resolves to min(video, audio) -- if the voiceover track is
        # the shorter stream, that silently truncates the OUTPUT VIDEO
        # ITSELF to match it, dropping real trailing video frames (verified
        # directly: up to 40s / 6.9% of a real field video's runtime cut
        # this way). "-af apad" pads the audio with silence for as long as
        # needed first, so the audio stream is never the shorter one --
        # "-shortest" then only ever trims padding silence off the tail of
        # the (now-longer) audio to match the video, never the video itself.
        _run([
            FFMPEG_BINARY,
            "-y",
            "-i",
            video_path,
            "-i",
            voiceover_wav,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-af",
            "apad",
            "-shortest",
            out_mp4_path,
        ])

    return out_mp4_path