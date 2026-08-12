#!/bin/bash

set -e

echo ""
echo "=========================================="
echo " Video Translate Pipeline - Mac Setup"
echo "=========================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# --------------------------------------------------
# 1. Check Python
# --------------------------------------------------

echo "[1/8] Checking Python..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python 3 is not installed."
    echo "Install Python 3.12 first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

echo "Python: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.12" ]; then
    echo ""
    echo "WARNING: This project is tested with Python 3.12."
    echo "Detected: $PYTHON_VERSION"
    echo ""
fi

# --------------------------------------------------
# 2. Check Homebrew
# --------------------------------------------------

echo ""
echo "[2/8] Checking Homebrew..."

if ! command -v brew >/dev/null 2>&1; then
    echo "ERROR: Homebrew is not installed."
    echo "Install Homebrew first."
    exit 1
fi

echo "Homebrew: OK"

# --------------------------------------------------
# 3. Check FFmpeg
# --------------------------------------------------

echo ""
echo "[3/8] Checking FFmpeg..."

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg not found."
    echo "Installing FFmpeg + libass..."

    brew install ffmpeg libass
else
    echo "FFmpeg: $(ffmpeg -version | head -1)"
fi

# --------------------------------------------------
# 4. Check subtitle support
# --------------------------------------------------

echo ""
echo "[4/8] Checking FFmpeg subtitle support..."

if ffmpeg -filters 2>/dev/null | grep -qE '(^| )subtitles( |$)'; then
    echo "FFmpeg subtitles filter: OK"
else
    echo ""
    echo "ERROR: Your FFmpeg does NOT contain the subtitles filter."
    echo ""
    echo "The application requires FFmpeg with libass support."
    echo ""
    echo "Current FFmpeg:"
    ffmpeg -version | head -1
    echo ""
    echo "Please install an FFmpeg build containing libass support."
    echo "After installing it, verify with:"
    echo ""
    echo "    ffmpeg -filters | grep subtitles"
    echo ""
    exit 1
fi

# --------------------------------------------------
# 5. Create virtual environment
# --------------------------------------------------

echo ""
echo "[5/8] Creating virtual environment..."

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists."
fi

source .venv/bin/activate

echo "Python inside venv:"
python --version

# --------------------------------------------------
# 6. Upgrade pip
# --------------------------------------------------

echo ""
echo "[6/8] Updating pip..."

python -m pip install --upgrade pip setuptools wheel

# --------------------------------------------------
# 7. Install Python dependencies
# --------------------------------------------------

echo ""
echo "[7/8] Installing Python dependencies..."

if [ -f "requirements-mac.txt" ]; then
    python -m pip install -r requirements-mac.txt
else
    echo "ERROR: requirements-mac.txt not found."
    exit 1
fi

# Fix the protobuf conflict caused by descript-audiotools.
#
# ONNX Runtime needs protobuf >= 4.25.8 while old
# descript-audiotools declares protobuf < 3.20.
#
# We deliberately keep protobuf 4.25.9 because the
# pipeline's actual imports work with it.

python -m pip install --upgrade "protobuf==4.25.9"

# --------------------------------------------------
# 8. Verification
# --------------------------------------------------

echo ""
echo "[8/8] Running verification..."
echo ""

echo "Checking Torch..."
python -c "import torch; print('  torch:', torch.__version__)"

echo "Checking Transformers..."
python -c "import transformers; print('  transformers:', transformers.__version__)"

echo "Checking faster-whisper..."
python -c "from faster_whisper import WhisperModel; print('  faster-whisper: OK')"

echo "Checking IndicTransToolkit..."
python -c "from IndicTransToolkit import IndicProcessor; print('  IndicTransToolkit: OK')"

echo "Checking Parler-TTS..."
python -c "from parler_tts import ParlerTTSForConditionalGeneration; print('  Parler-TTS: OK')"

echo "Checking DAC..."
python -c "import dac; print('  DAC: OK')" || true

echo "Checking AudioTools..."
python -c "import audiotools; print('  AudioTools: OK')"

echo "Checking FFmpeg..."
ffmpeg -version | head -1

echo ""
echo "Checking Python dependencies..."
python -m pip check || true

echo ""
echo "=========================================="
echo " SETUP COMPLETE"
echo "=========================================="
echo ""
echo "To run the application:"
echo ""
echo "    source .venv/bin/activate"
echo "    python app.py"
echo ""
echo "Or simply:"
echo ""
echo "    ./run_mac.sh"
echo ""