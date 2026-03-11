#!/bin/bash
# ──────────────────────────────────────────────────────────────
# LTX-2.3 Video Studio — Full Environment Setup
# ──────────────────────────────────────────────────────────────
#
# This script sets up everything needed to run LTX-2.3:
#   1. Creates a Python virtual environment
#   2. Installs PyTorch with CUDA support
#   3. Installs the LTX-2 packages (ltx-core, ltx-pipelines)
#   4. Installs all remaining dependencies
#
# Prerequisites:
#   - Python 3.10+ with venv module
#   - NVIDIA GPU with CUDA toolkit installed (CUDA 12+ or 13)
#   - ~10 GB disk space for packages
#   - ffmpeg (for video encoding)
#
# Usage:
#   cd LTX-Video-Kit
#   bash scripts/setup.sh
#
# ──────────────────────────────────────────────────────────────

set -e

# ── Resolve paths ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$KIT_DIR/venv"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  LTX-2.3 Video Studio — Setup"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Kit directory: $KIT_DIR"
echo "Venv target:   $VENV_DIR"
echo ""

# ── Step 1: Check prerequisites ──────────────────────────────
echo "[1/6] Checking prerequisites..."

# Python
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ is required but not found."
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "  Python:  $PY_VERSION ($PYTHON)"

# CUDA
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
    echo "  CUDA:    $CUDA_VERSION"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | sed 's/,//' || echo "detected")
    echo "  CUDA:    $CUDA_VERSION (at /usr/local/cuda)"
else
    echo "  WARNING: CUDA not found. GPU acceleration may not work."
fi

# ffmpeg
if command -v ffmpeg &>/dev/null; then
    echo "  ffmpeg:  $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
    echo "  WARNING: ffmpeg not found. Install with: sudo apt install ffmpeg"
fi

# GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU:     $GPU_NAME ($GPU_MEM)"
fi

echo ""

# ── Step 2: Create virtual environment ──────────────────────
echo "[2/6] Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "  Venv already exists at $VENV_DIR, skipping creation."
else
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
echo "  Activated venv. pip $(pip --version | awk '{print $2}')"
echo ""

# ── Step 3: Install PyTorch ─────────────────────────────────
echo "[3/6] Installing PyTorch with CUDA support..."
echo "  This may take a few minutes..."

# Detect CUDA major version for PyTorch index URL
# CUDA 13.x → cu130, CUDA 12.x → cu124 or cu121
CUDA_MAJOR=""
if [ -f "/usr/local/cuda/version.txt" ] || command -v nvcc &>/dev/null; then
    NVCC_PATH=$(command -v nvcc || echo "/usr/local/cuda/bin/nvcc")
    if [ -x "$NVCC_PATH" ]; then
        CUDA_MAJOR=$($NVCC_PATH --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f1)
    fi
fi

if [ "$CUDA_MAJOR" = "13" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    echo "  Detected CUDA 13 → using cu130 wheels"
elif [ "$CUDA_MAJOR" = "12" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo "  Detected CUDA 12 → using cu124 wheels"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo "  CUDA version unknown → defaulting to cu124 wheels"
fi

pip install torch torchaudio --index-url "$TORCH_INDEX" --quiet
echo "  PyTorch installed: $(python -c 'import torch; print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")')"
echo ""

# ── Step 4: Install LTX-2 packages ──────────────────────────
echo "[4/6] Installing LTX-2 packages (ltx-core, ltx-pipelines)..."
pip install uv-build --quiet
pip install -e "$KIT_DIR/packages/ltx-core" --quiet
pip install -e "$KIT_DIR/packages/ltx-pipelines" --quiet
echo "  ltx-core and ltx-pipelines installed (editable mode)"
echo ""

# ── Step 5: Install remaining dependencies ──────────────────
echo "[5/6] Installing remaining dependencies..."
pip install -r "$KIT_DIR/app/requirements.txt" --quiet
echo "  All dependencies installed"
echo ""

# ── Step 6: Verify ──────────────────────────────────────────
echo "[6/6] Verifying installation..."
python -c "
import torch
print(f'  PyTorch:        {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:            {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'  GPU Memory:     {free/(1024**3):.1f} GB free / {total/(1024**3):.1f} GB total')
from ltx_pipelines.distilled import DistilledPipeline
print(f'  LTX Pipeline:   OK')
from flask import Flask
print(f'  Flask:          OK')
" 2>&1 | grep -v "UserWarning"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Download models (if not already present):"
echo "       bash $KIT_DIR/scripts/download_models.sh"
echo ""
echo "    2. Start the web UI:"
echo "       cd $KIT_DIR/app && bash run.sh"
echo "════════════════════════════════════════════════════════════"
