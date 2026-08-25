"""
Standalone Indic-TTS (FastPitch + HiFi-GAN) synthesis worker.

Runs inside its own venv (.venv-tts).

Responsibilities:
1. Normalize numbers before TTS.
2. Handle decimal numbers explicitly.
3. Preserve zeros after decimal points.
4. Generate WAV audio.
5. Optionally write metadata used by the pipeline quality report.

The original subtitle/translation text is NEVER modified.
Only the text sent to TTS is normalized.

Two modes:

  --serve       Persistent mode. Loads the FastPitch + HiFi-GAN checkpoint
                ONCE, then reads one JSON request per line from stdin and
                writes one JSON response per line to stdout until it
                receives {"cmd": "shutdown"} or stdin closes. This is what
                the pipeline should use for any job with more than one
                segment -- loading a ~1.5GB checkpoint pair per segment was
                previously the dominant cost of voiceover generation.

  (default)     Single-shot mode, unchanged from before: loads the
                checkpoint, synthesizes --text once, exits. Kept around for
                manual testing of a single segment from the command line.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from indic_numtowords import num2words
from scipy.io.wavfile import write as scipy_wav_write
from TTS.utils.synthesizer import Synthesizer


LANG_MAP = {
    "asm": "as",
    "ben": "bn",
    "brx": "brx",
    "eng": "en",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "mal": "ml",
    "mni": "mni",
    "mar": "mr",
    "ori": "or",
    "pan": "pa",
    "tam": "ta",
    "tel": "te",
}


SUPPORTED_NUM_LANGS = {
    "as",
    "bn",
    "brx",
    "en",
    "gu",
    "hi",
    "kn",
    "ml",
    "mni",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
}


DECIMAL_WORDS = {
    "hi": "दशमलव",
    "bn": "দশমিক",
    "gu": "દશાંશ",
    "mr": "दशांश",
    "pa": "ਦਸ਼ਮਲਵ",
    "ta": "புள்ளி",
    "te": "దశాంశం",
    "kn": "ದಶಮಾಂಶ",
    "ml": "ദശാംശം",
    "or": "ଦଶମିକ",
    "as": "দশমিক",
    "en": "point",
    "brx": "दशमलव",
    "mni": "ꯄꯣꯏꯟꯠ",
}


def normalize_numbers_for_tts(text: str, lang: str):
    """
    Convert numeric expressions into spoken words.

    Examples:

        25
        -> पच्चीस

        25.05
        -> पच्चीस दशमलव शून्य पाँच

        3.14
        -> तीन दशमलव एक चार

    Decimal digits are deliberately converted one-by-one so that
    zeros are not lost.

    Returns:
        normalized_text, metadata
    """

    num_lang = LANG_MAP.get(lang, lang)

    metadata = {
        "input_text": text,
        "normalized_text": text,
        "language": lang,
        "numeric_expressions": [],
        "numeric_expression_count": 0,
        "decimal_count": 0,
        "integer_count": 0,
        "normalization_failures": 0,
    }

    if num_lang not in SUPPORTED_NUM_LANGS:
        return text, metadata

    # Decimal expressions first.
    decimal_pattern = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")

    # Integer expressions.
    integer_pattern = re.compile(r"(?<![\w.])\d+(?![\w.])")

    def convert_decimal(match):
        raw = match.group(0)

        try:
            whole, fraction = raw.split(".", 1)

            whole_words = num2words(
                int(whole),
                lang=num_lang,
            )

            fraction_words = []

            for digit in fraction:
                digit_words = num2words(
                    int(digit),
                    lang=num_lang,
                )
                fraction_words.append(digit_words)

            decimal_word = DECIMAL_WORDS.get(
                num_lang,
                "point",
            )

            normalized = (
                f"{whole_words} "
                f"{decimal_word} "
                f"{' '.join(fraction_words)}"
            )

            metadata["numeric_expressions"].append({
                "input": raw,
                "output": normalized,
                "type": "decimal",
                "success": True,
            })

            metadata["numeric_expression_count"] += 1
            metadata["decimal_count"] += 1

            return normalized

        except Exception as exc:
            metadata["normalization_failures"] += 1

            metadata["numeric_expressions"].append({
                "input": raw,
                "output": raw,
                "type": "decimal",
                "success": False,
                "error": str(exc),
            })

            print(
                f"[tts] Decimal normalization failed for "
                f"{raw!r}: {exc}",
                file=sys.stderr,
            )

            return raw

    def convert_integer(match):
        raw = match.group(0)

        try:
            normalized = num2words(
                int(raw),
                lang=num_lang,
            )

            metadata["numeric_expressions"].append({
                "input": raw,
                "output": normalized,
                "type": "integer",
                "success": True,
            })

            metadata["numeric_expression_count"] += 1
            metadata["integer_count"] += 1

            return normalized

        except Exception as exc:
            metadata["normalization_failures"] += 1

            metadata["numeric_expressions"].append({
                "input": raw,
                "output": raw,
                "type": "integer",
                "success": False,
                "error": str(exc),
            })

            print(
                f"[tts] Integer normalization failed for "
                f"{raw!r}: {exc}",
                file=sys.stderr,
            )

            return raw

    normalized = decimal_pattern.sub(
        convert_decimal,
        text,
    )

    normalized = integer_pattern.sub(
        convert_integer,
        normalized,
    )

    metadata["normalized_text"] = normalized

    return normalized, metadata


def _validate_checkpoint_layout(checkpoint_dir: str):
    fastpitch_dir = Path(checkpoint_dir) / "fastpitch"
    hifigan_dir = Path(checkpoint_dir) / "hifigan"

    required_files = [
        fastpitch_dir / "best_model.pth",
        fastpitch_dir / "config.json",
        fastpitch_dir / "speakers.pth",
        hifigan_dir / "best_model.pth",
        hifigan_dir / "config.json",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Indic-TTS checkpoint is incomplete. Missing:\n"
            + "\n".join(missing)
        )

    return fastpitch_dir, hifigan_dir


def build_synthesizer(checkpoint_dir: str) -> Synthesizer:
    """
    Load the FastPitch + HiFi-GAN checkpoint pair and construct a
    Synthesizer. This is the expensive part (disk I/O + model init) --
    callers should do this ONCE and reuse the returned object across many
    synthesize_one() calls rather than rebuilding it per segment.
    """

    fastpitch_dir, hifigan_dir = _validate_checkpoint_layout(checkpoint_dir)

    return Synthesizer(
        tts_checkpoint=str(
            fastpitch_dir / "best_model.pth"
        ),
        tts_config_path=str(
            fastpitch_dir / "config.json"
        ),
        tts_speakers_file=str(
            fastpitch_dir / "speakers.pth"
        ),
        tts_languages_file=None,
        vocoder_checkpoint=str(
            hifigan_dir / "best_model.pth"
        ),
        vocoder_config=str(
            hifigan_dir / "config.json"
        ),
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=False,
    )


def synthesize_one(
    synth: Synthesizer,
    text: str,
    lang: str,
    speaker: str,
    out_path: str,
    metadata_path: str = None,
) -> dict:
    """
    Synthesize a single piece of text with an already-loaded Synthesizer.
    Returns the same metadata shape the old single-shot main() wrote out,
    so callers (delivery.py) don't need to change how they read the result.
    """

    tts_text, number_metadata = normalize_numbers_for_tts(
        text,
        lang,
    )

    print(
        f"[tts] Input: {text!r} "
        f"-> normalized: {tts_text!r}",
        file=sys.stderr,
    )

    wav = synth.tts(
        tts_text,
        speaker_name=speaker,
    )

    wav = np.asarray(
        wav,
        dtype=np.float32,
    )

    wav = np.clip(
        wav,
        -1.0,
        1.0,
    )

    pcm16 = (
        wav * 32767.0
    ).astype(np.int16)

    sample_rate = getattr(
        synth,
        "output_sample_rate",
        22050,
    )

    scipy_wav_write(
        out_path,
        sample_rate,
        pcm16,
    )

    metadata = {
        **number_metadata,
        "output_wav": str(out_path),
        "sample_rate": sample_rate,
        "sample_count": int(len(pcm16)),
        "duration_seconds": (
            float(len(pcm16)) / sample_rate
            if sample_rate
            else 0.0
        ),
        "speaker": speaker,
        "success": True,
    }

    if metadata_path:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )

    return metadata


def serve(checkpoint_dir: str, lang: str, default_speaker: str):
    """
    Persistent worker loop. Loads the checkpoint once, then services one
    synthesis request per line of stdin until told to shut down.

    Request  (one JSON object per line on stdin):
        {"text": "...", "out": "/path/seg.wav",
         "metadata_out": "/path/seg.json",   # optional
         "speaker": "female"}                # optional, defaults to default_speaker
        {"cmd": "shutdown"}                  # tells the loop to exit

    Response (one JSON object per line on stdout):
        {"ready": true}                                  # emitted once, at startup
        {"ok": true, ...same fields as synthesize_one...} # per successful request
        {"ok": false, "error": "..."}                     # per failed request
    """

    synth = build_synthesizer(checkpoint_dir)

    # Emitted only after the slow checkpoint load finishes, so the parent
    # process knows exactly when it's safe to start sending segments.
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"ok": False, "error": f"bad request JSON: {exc}"}), flush=True)
            continue

        if req.get("cmd") == "shutdown":
            break

        try:
            speaker = req.get("speaker") or default_speaker

            metadata = synthesize_one(
                synth,
                req["text"],
                lang,
                speaker,
                req["out"],
                req.get("metadata_out"),
            )

            print(json.dumps({"ok": True, **metadata}, ensure_ascii=False), flush=True)

        except Exception as exc:
            print(
                f"[tts] serve() request failed: {exc}",
                file=sys.stderr,
            )
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing fastpitch/ and hifigan/",
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Persistent mode: load the checkpoint once and service JSON "
            "requests from stdin until shutdown. Use this for any job with "
            "more than one segment."
        ),
    )

    parser.add_argument(
        "--text",
        required=False,
        help="Required in single-shot mode (i.e. when --serve is not set).",
    )

    parser.add_argument(
        "--lang",
        required=True,
        help="Target language code, e.g. hin, tel, mar",
    )

    parser.add_argument(
        "--speaker",
        default="female",
        choices=["male", "female"],
    )

    parser.add_argument(
        "--out",
        required=False,
        help="Required in single-shot mode (i.e. when --serve is not set).",
    )

    parser.add_argument(
        "--metadata-out",
        default=None,
        help="Optional JSON file containing TTS normalization metadata.",
    )

    args = parser.parse_args()

    if args.serve:
        serve(args.checkpoint_dir, args.lang, args.speaker)
        return

    if not args.text or not args.out:
        parser.error("--text and --out are required unless --serve is set")

    # ---------------------------------------------------------
    # Single-shot mode (unchanged behavior from before, just
    # reorganized to share build_synthesizer()/synthesize_one()
    # with --serve mode).
    # ---------------------------------------------------------

    synth = build_synthesizer(args.checkpoint_dir)

    synthesize_one(
        synth,
        args.text,
        args.lang,
        args.speaker,
        args.out,
        args.metadata_out,
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print(
            f"synthesize.py failed: {e}",
            file=sys.stderr,
        )

        sys.exit(1)