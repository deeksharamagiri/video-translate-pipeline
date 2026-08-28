# syntax=docker/dockerfile:1
#
# Offline field translator -- see DOCKER.md for build/save/load/run
# instructions. Two things this image cannot give you that the native
# macOS setup can:
#
#   1. GPU acceleration. Metal has no passthrough into Linux containers at
#      all, and this build targets plain x86_64 (no CUDA toolchain here
#      either) -- ASR always runs CPU-only inside Docker. See README
#      "Speeding it up on CPU" for the knobs that matter most as a result
#      (TRANSLATION_NUM_BEAMS, WHISPER_THREADS).
#   2. Every language. Models are prefetched at build time for a specific
#      scope (default: Hindi) so the container needs zero internet at
#      runtime -- see docker/prefetch_models.py to extend that scope.

# =====================================================================
# Stage 1: build whisper.cpp from source (not pip-installable).
# CPU-only build -- no Metal (Linux has none), no CUDA (no GPU assumed
# for the target machine).
# =====================================================================
FROM debian:bookworm-slim AS whispercpp-build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Pinned to the version pipeline/stage2_asr.py documents as tested against.
RUN git clone --branch v1.9.2 --depth 1 \
    https://github.com/ggerganov/whisper.cpp.git .

RUN cmake -B build -DGGML_METAL=OFF -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j --config Release --target whisper-cli

# =====================================================================
# Stage 2: application image
# =====================================================================
FROM python:3.12-slim-bookworm AS app

# Real system ffmpeg, not the bundled static-ffmpeg binary config.py falls
# back to. Verified directly: Debian bookworm ships ffmpeg 5.1.9 linked
# against libass 1:0.17.1, which fixes the complex-script (Devanagari/
# Thai/Arabic) subtitle-shaping bug present in static-ffmpeg's bundled
# libass 0.15.2 (see config.py's FFMPEG_BINARY comments). Picked up
# automatically via the FFMPEG_BINARY/FFPROBE_BINARY env vars below --
# no code changes needed, that override already existed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the whole whisper.cpp build output dir (binary + any shared libs
# ggml/whisper produce alongside it) rather than cherry-picking one file,
# so this doesn't silently break if a future whisper.cpp version changes
# its output layout.
COPY --from=whispercpp-build /build/build/bin/ /opt/whispercpp/
ENV PATH="/opt/whispercpp:${PATH}"
ENV LD_LIBRARY_PATH="/opt/whispercpp"
ENV WHISPER_CPP_BINARY=/opt/whispercpp/whisper-cli

WORKDIR /app

# ---- Main application dependencies (installed directly -- the container
# itself is the isolation boundary, no venv needed here) ----
#
# torch installed from the CPU-only wheel index FIRST, deliberately: plain
# `pip install torch` on Linux defaults to the CUDA build, which drags in
# ~1.5GB+ of nvidia-*/triton packages that are dead weight here -- this
# image has no GPU access at all (no Metal passthrough into containers,
# and no CUDA toolchain in this build; see the top-of-file note).
# Confirmed by actually building this image once: switching to the CPU
# wheel took torch from 526MB (+ nvidia-cudnn 366MB + nvidia-cusparselt
# 170MB + nvidia-nccl 206MB + more) down to 192MB total. Installed before
# `-r requirements.txt` so that file's unpinned `torch>=2.2` sees the
# constraint already satisfied and doesn't re-resolve it from PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ---- Indic-TTS voiceover: a SEPARATE venv is a genuine requirement, not
# just local-dev convenience -- coqui-tts needs a much newer transformers
# than IndicTrans2/NLLB are pinned to (see tts_worker/requirements.txt).
# Same CPU-only-wheel reasoning as above applies here too. ----
COPY tts_worker/requirements.txt tts_worker/requirements.txt
RUN python -m venv /app/.venv-tts \
    && /app/.venv-tts/bin/pip install --no-cache-dir torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
    && /app/.venv-tts/bin/pip install --no-cache-dir -r tts_worker/requirements.txt

# ---- App code ----
COPY . .

# ---- Bake in models for the scoped language set (default: Hindi) so the
# container needs zero internet at runtime. Requires HF_TOKEN as a build
# secret for the gated IndicTrans2 checkpoints -- see DOCKER.md.
#
# This step's very first `import config` (transitively, via pipeline.*)
# also triggers static-ffmpeg's own auto-provisioning as a side effect --
# confirmed by an actual build -- so its bundled binary gets cached into
# the image here too, even though FFMPEG_BINARY/FFPROBE_BINARY below make
# the app use the real system ffmpeg instead at runtime. No separate
# pre-warm step needed. ----
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python docker/prefetch_models.py

# 127.0.0.1 (config.py's default) is unreachable from outside the
# container even with -p published -- see config.py's HOST comment.
ENV HOST=0.0.0.0
ENV PORT=5000
EXPOSE 5000

VOLUME ["/app/jobs", "/app/data"]

CMD ["python", "app.py"]
