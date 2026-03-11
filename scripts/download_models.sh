#!/bin/bash
# ──────────────────────────────────────────────────────────────
# LTX-2.3 Video Studio — Model Downloader
# ──────────────────────────────────────────────────────────────
#
# Downloads all required model weights from HuggingFace:
#   1. LTX-2.3 22B Distilled checkpoint     (~43 GB)
#   2. LTX-2.3 Spatial Upscaler 2x          (~950 MB)
#   3. Gemma-3 12B Text Encoder (quantized)  (~23 GB)
#
# Total download: ~67 GB
# Requires: huggingface-cli (pip install huggingface_hub)
#
# Usage:
#   bash scripts/download_models.sh
#
# If you need to log in to HuggingFace first:
#   huggingface-cli login
#
# ──────────────────────────────────────────────────────────────

set -e

# ── Resolve paths ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$KIT_DIR/models"

mkdir -p "$MODEL_DIR"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  LTX-2.3 — Model Downloader"
echo "  Target: $MODEL_DIR"
echo "════════════════════════════════════════════════════════════"
echo ""

# Activate venv if it exists (for huggingface-cli)
if [ -d "$KIT_DIR/venv" ]; then
    source "$KIT_DIR/venv/bin/activate"
fi

# Check huggingface-cli is available
if ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface-cli not found."
    echo "  Install with: pip install huggingface_hub"
    echo "  Then log in:  huggingface-cli login"
    exit 1
fi

# ── Download 1: LTX-2.3 Distilled Checkpoint ────────────────
FILE1="$MODEL_DIR/ltx-2.3-22b-distilled.safetensors"
if [ -f "$FILE1" ]; then
    SIZE=$(du -h "$FILE1" | cut -f1)
    echo "[1/3] Distilled checkpoint already exists ($SIZE) — skipping"
else
    echo "[1/3] Downloading LTX-2.3 22B Distilled checkpoint (~43 GB)..."
    echo "      This will take a while depending on your connection."
    huggingface-cli download Lightricks/LTX-2.3 \
        ltx-2.3-22b-distilled.safetensors \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    echo "      Done!"
fi
echo ""

# ── Download 2: Spatial Upscaler ─────────────────────────────
FILE2="$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
if [ -f "$FILE2" ]; then
    SIZE=$(du -h "$FILE2" | cut -f1)
    echo "[2/3] Spatial upscaler already exists ($SIZE) — skipping"
else
    echo "[2/3] Downloading Spatial Upscaler 2x (~950 MB)..."
    huggingface-cli download Lightricks/LTX-2.3 \
        ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    echo "      Done!"
fi
echo ""

# ── Download 3: Gemma-3 Text Encoder ────────────────────────
GEMMA_DIR="$MODEL_DIR/gemma-3-12b-it-qat-q4_0-unquantized"
if [ -d "$GEMMA_DIR" ] && [ "$(ls -1 "$GEMMA_DIR"/*.safetensors 2>/dev/null | wc -l)" -ge 5 ]; then
    SIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
    echo "[3/3] Gemma-3 text encoder already exists ($SIZE) — skipping"
else
    echo "[3/3] Downloading Gemma-3 12B Text Encoder (~23 GB)..."
    echo "      NOTE: This model requires accepting Google's license on HuggingFace."
    echo "      Visit: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
    huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
        --local-dir "$GEMMA_DIR" \
        --local-dir-use-symlinks False
    echo "      Done!"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  All models downloaded!"
echo ""
echo "  Files:"
ls -lhS "$MODEL_DIR"/*.safetensors 2>/dev/null | awk '{printf "    %-55s %s\n", $NF, $5}'
if [ -d "$GEMMA_DIR" ]; then
    GSIZE=$(du -sh "$GEMMA_DIR" | cut -f1)
    echo "    gemma-3-12b-it-qat-q4_0-unquantized/             $GSIZE"
fi
echo ""
echo "  Total: $(du -sh "$MODEL_DIR" | cut -f1)"
echo ""
echo "  Next: cd $KIT_DIR/app && bash run.sh"
echo "════════════════════════════════════════════════════════════"
