# Docker: build once online, run fully offline elsewhere

This packages the app as a self-contained image for a **different
machine that may have no internet at all**. Everything the app needs at
runtime -- the whisper.cpp ASR model, translation models, the Hindi
Indic-TTS voice, ffmpeg -- is baked into the image at build time. Building
the image still needs internet (to download those weights once); running
the built image does not.

The image also forces `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` at
runtime (see Dockerfile). This isn't just belt-and-suspenders: verified
directly that without it, loading a *cached* model still makes
`huggingface_hub` try to reach huggingface.co first to check for updates
-- fine with internet, but on a machine with genuinely no route out, that
attempt hangs rather than failing fast, which would otherwise make every
job stall on first translation/TTS use instead of running immediately
from the baked-in cache.

Current scope, chosen to keep the image and build time reasonable:

- **Target machine**: x86_64 (Intel/AMD) Linux or Windows/Mac with Docker.
  Not built for Apple Silicon targets -- see "Other architectures" below.
- **Fully-offline languages**: Hindi as the target language (source
  language is auto-detected by whisper.cpp's multilingual model, no extra
  download needed per source language). Covers English→Hindi, Hindi→
  English, Hindi→Hindi/other-Indic-source, and any other source language
  via the NLLB fallback. See "Adding more languages" to extend this.
- **Voiceover**: included (Indic-TTS Hindi voice baked in).
- **GPU**: none. Metal has no passthrough into Linux containers at all,
  and this build has no CUDA toolchain either, so ASR always runs
  CPU-only inside Docker regardless of what the host machine has. See
  README "Speeding it up on CPU" -- `TRANSLATION_NUM_BEAMS` and
  `WHISPER_THREADS` matter more here than on bare metal with a GPU.

## 1. Build (on a machine with internet)

IndicTrans2 is gated on Hugging Face (free, but requires accepting terms
once) -- see README.md "The one manual step" if you haven't done that yet
for the account whose token you use here.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

docker build \
  --platform linux/amd64 \
  --secret id=hf_token,env=HF_TOKEN \
  -t field-translator:latest \
  .
```

`--platform linux/amd64` matters even if you're building on Apple
Silicon (Docker will cross-build via emulation, just slower) -- it's what
makes the resulting image actually runnable on an x86_64 target machine.

The `--secret` flag (not a plain `--build-arg`) is deliberate: build args
get baked into image history/layers permanently, which is fine for model
weights but not for an API token. The secret is only available to the one
`RUN` step that needs it and never lands in a layer.

Expect this to take a while the first time -- it compiles whisper.cpp
from source and downloads several GB of model weights (NLLB ~2.4GB,
IndicTrans2 dist checkpoints ~1GB combined, Indic-TTS Hindi voice
~1.5GB, whisper.cpp's ggml-medium-q5_0 ~514MB, IndicConformer ~2.4GB --
baked in because it's the ASR engine the web UI always requests by
default, not an opt-in extra).

**Verify it built correctly** before moving on:

```bash
docker run --rm --platform linux/amd64 field-translator:latest \
  whisper-cli --help
```

## 2. Transfer to the offline machine

```bash
# On the machine with internet:
docker save field-translator:latest -o field-translator.tar
# Copy field-translator.tar to a USB drive / external disk.

# On the OFFLINE target machine:
docker load -i field-translator.tar
```

`docker load` needs no internet -- it's reading the tarball, not pulling
from a registry.

## 3. Run

```bash
docker run -d \
  -p 5000:5000 \
  -v "$(pwd)/jobs:/app/jobs" \
  -v "$(pwd)/data:/app/data" \
  --name field-translator \
  field-translator:latest
```

Or with the included `docker-compose.yml` (build step needs `HF_TOKEN`
exported the same way as above; skip `build` and just `docker compose up
-d` on the offline machine once the image is loaded):

```bash
docker compose up -d
```

Open `http://localhost:5000` (or the target machine's address, if you're
reaching it from another device on the same network).

**The `-v .../data:/app/data` mount will shadow the baked-in
`glossary.json`** if your host's local `data/` directory is empty --
Docker mounts replace the container's directory contents entirely, they
don't merge. Either seed `./data/glossary.json` on the host before first
run (copy it out of the image: `docker cp field-translator:/app/data/glossary.json ./data/`),
or drop the `data` volume mount if you don't need glossary edits to
persist across container recreates.

## 4. Adding more languages later

Re-run the build with `docker/prefetch_models.py` extended for the
languages you need, then rebuild:

- **Translation**: add the target language's NLLB code to
  `NLLB_LANG_CODE_MAP` (config.py) if it's not already there (most are).
  For Indic↔Indic routing quality, no code change is needed --
  `indic_indic`/`en_indic`/`indic_en` dist checkpoints already cover all
  22 `INDIC_LANGS`; the prefetch script just needs to run
  `_get_indictrans_model(...)` for those checkpoints if you skipped them
  initially (this default build already includes all three directions,
  so most additional *Indic* target languages need no prefetch changes at
  all -- only additional Indic-TTS voices below do).
- **Voiceover**: add a call to
  `_ensure_indic_tts_checkpoint("<lang>")` in
  `docker/prefetch_models.py` for each additional language (must be one
  of the 13 in `config.INDIC_TTS_LANG_ZIP_MAP` -- see README "Adding more
  Indic-TTS voices").

Then rebuild and re-save/transfer per steps 1-2.

## Other architectures

- **Apple Silicon (arm64) target machine**: change `--platform` to
  `linux/arm64` in the build command and rebuild. whisper.cpp will
  compile for arm64 instead -- CPU-only either way (Docker containers
  still can't reach the host GPU/Metal, even on an M-series host).
- **Both**: `docker buildx build --platform linux/amd64,linux/arm64 ...`
  with `--push` to a registry produces a multi-arch manifest, but that
  needs a registry (not a plain `docker save` tarball) -- out of scope
  for a pure offline-transfer workflow.

## Troubleshooting

- **Build fails at the prefetch step with a gated-model / 401 error** --
  `HF_TOKEN` wasn't picked up, or the token's account hasn't accepted
  IndicTrans2's terms yet. See README.md "The one manual step".
- **Container starts but `http://localhost:5000` doesn't respond** --
  check `docker logs field-translator`. Also confirm you didn't override
  `HOST` back to `127.0.0.1` -- the image sets `HOST=0.0.0.0` by design
  (see `config.py`'s comment on why 127.0.0.1 wouldn't be reachable from
  outside the container).
- **Voiceover jobs fail inside the container** -- check
  `jobs/pipeline.log` (mounted to the host via the `jobs` volume) for the
  actual traceback from the `.venv-tts` subprocess.
- **A job seems to hang on first translation/TTS use** -- confirm
  `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` weren't unset (e.g. by a custom
  `docker run -e` override) -- see the offline-mode note above.
