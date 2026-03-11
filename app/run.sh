#!/bin/bash
# ──────────────────────────────────────────────────────────────
# LTX-2.3 Video Studio — Startup Script
# ──────────────────────────────────────────────────────────────

set -e

# Resolve directories
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT_DIR="$(cd "$APP_DIR/.." && pwd)"

# CUDA environment (adjust version if your system differs)
export PATH="/usr/local/cuda/bin:/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Add LTX-2 packages to Python path
export PYTHONPATH="$KIT_DIR/packages/ltx-core/src:$KIT_DIR/packages/ltx-pipelines/src:${PYTHONPATH}"

# Activate virtualenv (check multiple possible locations)
VENV=""
if [ -d "$KIT_DIR/venv" ]; then
    VENV="$KIT_DIR/venv"
elif [ -d "$HOME/Genaiprotos/Edge_AI/ltx-video-env" ]; then
    VENV="$HOME/Genaiprotos/Edge_AI/ltx-video-env"
fi

if [ -n "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "Activated venv: $VENV"
else
    echo "WARNING: No virtual environment found."
    echo "  Expected at: $KIT_DIR/venv/"
    echo "  Run: bash $KIT_DIR/scripts/setup.sh  to create one."
fi

# Ensure directories exist
mkdir -p "$APP_DIR/outputs" "$APP_DIR/uploads"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  LTX-2.3 Video Studio"
echo "  Kit:    $KIT_DIR"
echo "  Models: $KIT_DIR/models/"
echo "  URL:    http://0.0.0.0:5000"
echo "════════════════════════════════════════════════════════════"
echo ""

# Run Flask app
cd "$APP_DIR"
exec python app.py
