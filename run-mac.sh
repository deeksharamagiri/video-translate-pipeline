#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "=========================================="
echo " Starting Video Translate Pipeline"
echo "=========================================="
echo ""

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found."
    echo "Running setup first..."
    echo ""

    ./setup_mac.sh
fi

source .venv/bin/activate

echo "Python:"
python --version

echo ""
echo "FFmpeg:"
which ffmpeg

echo ""
echo "Starting Flask..."
echo ""

python app.py