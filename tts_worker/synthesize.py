"""
Standalone Indic-TTS (FastPitch + HiFi-GAN) synthesis worker.

Runs inside its own venv (.venv-tts) — deliberately isolated from the main
project's venv because coqui-tts requires a much newer `transformers` than
this project pins for IndicTrans2/NLLB (transformers==4.44.2). Invoked via
subprocess from pipeline/delivery.py, never imported directly.

Do NOT import anything from this project here — this interpreter doesn't
have it installed.
"""
import argparse
import sys

import numpy as np
from scipy.io.wavfile import write as scipy_wav_write
from TTS.utils.synthesizer import Synthesizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True,
                         help="Directory containing fastpitch/ and hifigan/ subfolders")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="female", choices=["male", "female"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    fastpitch_dir = f"{args.checkpoint_dir}/fastpitch"
    hifigan_dir = f"{args.checkpoint_dir}/hifigan"

    synth = Synthesizer(
        tts_checkpoint=f"{fastpitch_dir}/best_model.pth",
        tts_config_path=f"{fastpitch_dir}/config.json",
        tts_speakers_file=f"{fastpitch_dir}/speakers.pth",
        tts_languages_file=None,
        vocoder_checkpoint=f"{hifigan_dir}/best_model.pth",
        vocoder_config=f"{hifigan_dir}/config.json",
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=False,
    )

    wav = synth.tts(args.text, speaker_name=args.speaker)
    wav = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)
    sample_rate = getattr(synth, "output_sample_rate", 22050)
    scipy_wav_write(args.out, sample_rate, pcm16)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"synthesize.py failed: {e}", file=sys.stderr)
        sys.exit(1)
