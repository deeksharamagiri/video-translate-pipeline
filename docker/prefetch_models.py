"""
Runs during `docker build` -- while the build host has internet -- to bake
model weights into the image so the container itself needs zero internet
at runtime. This is what makes "docker save the image, load it on an
offline machine" actually work.

Scoped to Hindi as the target language (see DOCKER.md for how to add more
languages later -- it's a re-run of this script with LANG additions, not a
code change):

  - whisper.cpp ASR model: multilingual, source-language-agnostic, always
    needed regardless of target language.
  - IndicConformer (ai4bharat/indic-conformer-600m-multilingual): the
    default ASR engine (see static/app.js's hardcoded asr_engine field and
    pipeline/orchestrator.py's refine_segments_with_indic_conformer call)
    -- every job from the web UI requests this, not just an opt-in
    refinement, so it has to be baked in the same as whisper.cpp itself.
    Ungated, no HF_TOKEN needed. Without this prefetched, the container's
    HF_HUB_OFFLINE=1 makes every job's load attempt fail fast at runtime;
    orchestrator.py catches that and silently falls back to the plain
    whisper.cpp transcript, so jobs still complete but with quietly worse
    transcript quality than a build that has this baked in.
  - NLLB-200-distilled-600M: universal fallback for any source language
    that isn't itself Indic (see pipeline/translate.py routing).
  - IndicTrans2 en-indic + indic-en + indic-indic (dist checkpoints):
    covers English<->Hindi and Hindi<->Hindi/other-Indic-source. Requires
    HF_TOKEN (gated model -- see README "The one manual step"; the token's
    account must have already clicked "Agree and access" on each repo).
  - Indic-TTS Hindi voice checkpoint (voiceover).

Reuses the exact same download/cache functions the app itself calls at
runtime (not reimplemented here), so "prefetched during build" and "the
app's normal first-use download" are guaranteed to write to the same
place with the same logic.

Each _get_*_model()/_ensure_*() call below loads the full model into
memory as a side effect of confirming the download, and caches it in a
module-level dict that's never cleared -- exactly what you want in the
app's own long-running process (reuse across jobs instead of reloading
every time), but fatal here if all five ran in this one process: memory
accumulates step over step with nothing ever freed, and five multi-GB
models loaded back to back overran a real build's container memory and
got OOM-killed partway through (verified directly -- IndicConformer
loaded fine alone, then NLLB loading on top of it still-resident got
"Killed"/"cannot allocate memory" from the kernel). Each step therefore
runs in its own subprocess here: same download logic, but the OS fully
reclaims that step's memory the moment its process exits, so peak memory
stays bounded by whichever single model is largest instead of growing
with every step.
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _run_step(code: str):
    """Run `code` in a fresh Python process so its memory is released on exit."""
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True)


def main():
    print("=== [1/5] whisper.cpp ASR model (ggml-medium-q5_0) ===", flush=True)
    _run_step(
        "from pipeline.stage2_asr import _ensure_whispercpp_model; "
        "_ensure_whispercpp_model()"
    )

    print("=== [2/5] IndicConformer (ai4bharat/indic-conformer-600m-multilingual) ===", flush=True)
    _run_step(
        "from pipeline.stage2_asr import _get_indic_conformer_model; "
        "_get_indic_conformer_model()"
    )

    print("=== [3/5] NLLB-200-distilled-600M ===", flush=True)
    _run_step(
        "from pipeline.translate import _get_nllb_model; "
        "_get_nllb_model()"
    )

    print("=== [4/5] IndicTrans2 (English<->Hindi, Hindi<->Hindi/Indic) ===", flush=True)
    if not os.environ.get("HF_TOKEN"):
        print(
            "ERROR: HF_TOKEN is not set. IndicTrans2 is a free but gated "
            "Hugging Face model -- see README.md 'The one manual step' for "
            "how to accept the terms and create a token, then re-run the "
            "build with:\n"
            "  docker build --secret id=hf_token,env=HF_TOKEN ...\n",
            file=sys.stderr,
        )
        sys.exit(1)

    from config import INDICTRANS2_MODELS, INDICTRANS2_MODEL_SIZE
    for direction in ("en_indic", "indic_en", "indic_indic"):
        model_name = INDICTRANS2_MODELS[direction][INDICTRANS2_MODEL_SIZE]
        print(f"    -> {model_name}", flush=True)
        _run_step(
            "from pipeline.translate import _get_indictrans_model; "
            f"_get_indictrans_model({model_name!r})"
        )

    print("=== [5/5] Indic-TTS Hindi voice checkpoint ===", flush=True)
    _run_step(
        "from pipeline.delivery import _ensure_indic_tts_checkpoint; "
        "_ensure_indic_tts_checkpoint('hin')"
    )

    print("All models prefetched.", flush=True)


if __name__ == "__main__":
    main()
