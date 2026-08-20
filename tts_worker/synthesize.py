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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing fastpitch/ and hifigan/",
    )

    parser.add_argument(
        "--text",
        required=True,
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
        required=True,
    )

    parser.add_argument(
        "--metadata-out",
        default=None,
        help="Optional JSON file containing TTS normalization metadata.",
    )

    args = parser.parse_args()

    fastpitch_dir = Path(args.checkpoint_dir) / "fastpitch"
    hifigan_dir = Path(args.checkpoint_dir) / "hifigan"

    # ---------------------------------------------------------
    # Validate checkpoint layout before initializing TTS.
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Number normalization
    # ---------------------------------------------------------

    tts_text, number_metadata = normalize_numbers_for_tts(
        args.text,
        args.lang,
    )

    print(
        f"[tts] Input: {args.text!r} "
        f"-> normalized: {tts_text!r}",
        file=sys.stderr,
    )

    # ---------------------------------------------------------
    # Initialize Indic-TTS
    # ---------------------------------------------------------

    synth = Synthesizer(
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

    # ---------------------------------------------------------
    # Synthesis
    # ---------------------------------------------------------

    wav = synth.tts(
        tts_text,
        speaker_name=args.speaker,
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
        args.out,
        sample_rate,
        pcm16,
    )

    # ---------------------------------------------------------
    # Metadata for quality report
    # ---------------------------------------------------------

    metadata = {
        **number_metadata,
        "output_wav": str(args.out),
        "sample_rate": sample_rate,
        "sample_count": int(len(pcm16)),
        "duration_seconds": (
            float(len(pcm16)) / sample_rate
            if sample_rate
            else 0.0
        ),
        "speaker": args.speaker,
        "success": True,
    }

    if args.metadata_out:
        metadata_path = Path(args.metadata_out)
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


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print(
            f"synthesize.py failed: {e}",
            file=sys.stderr,
        )

        sys.exit(1)