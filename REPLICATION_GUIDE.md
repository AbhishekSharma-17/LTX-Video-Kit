# LTX-2.3 Video Studio — Step-by-Step Replication Guide

This guide walks you through setting up LTX-2.3 video generation from a **fresh Ubuntu machine** with an NVIDIA GPU. Every single step is explained. Follow them in order.

> **Tested on**: NVIDIA DGX Spark (GB10, 128GB unified memory, CUDA 13.0, aarch64/ARM, Ubuntu)
> **Time to complete**: ~1 hour (mostly waiting for downloads)

---

## Table of Contents

1. [Check Your System](#step-1-check-your-system)
2. [Install System Packages](#step-2-install-system-packages)
3. [Get the Kit](#step-3-get-the-kit)
4. [Create Python Environment](#step-4-create-python-environment)
5. [Install PyTorch](#step-5-install-pytorch)
6. [Install LTX-2 Packages](#step-6-install-ltx-2-packages)
7. [Install Dependencies](#step-7-install-dependencies)
8. [Download Model Weights](#step-8-download-model-weights)
9. [Test: Generate Your First Video (CLI)](#step-9-test-generate-your-first-video-cli)
10. [Start the Web UI](#step-10-start-the-web-ui)
11. [Verification Checklist](#step-11-verification-checklist)
12. [Known Issues and Fixes](#step-12-known-issues-and-fixes)

---

## Step 1: Check Your System

Before starting, make sure your system meets the requirements. Run these commands:

### Check your GPU
```bash
nvidia-smi
```
You should see your NVIDIA GPU listed. You need **at least 40GB of GPU memory** (80GB+ recommended). Example output:
```
NVIDIA GB10  |  128GB  |  CUDA Version: 13.0
```

### Check CUDA
```bash
nvcc --version
```
If this fails, CUDA might be installed but not on your PATH. Try:
```bash
ls /usr/local/cuda*/bin/nvcc
```
If you find it (e.g., `/usr/local/cuda-13.0/bin/nvcc`), add it to your PATH:
```bash
echo 'export PATH="/usr/local/cuda-13.0/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```
> **What is CUDA?** It's NVIDIA's toolkit that lets programs use the GPU. PyTorch needs it.

### Check Python
```bash
python3 --version
```
You need **Python 3.10 or newer**. If you don't have it:
```bash
sudo apt update && sudo apt install python3 python3-venv python3-pip
```

### Check disk space
```bash
df -h .
```
You need at least **80 GB of free space** (67 GB for models + ~10 GB for packages).

### Check architecture
```bash
uname -m
```
This tells you if your CPU is `x86_64` (Intel/AMD) or `aarch64` (ARM). The PyTorch install step differs based on this.

---

## Step 2: Install System Packages

These are system-level packages needed before Python packages.

### Install ffmpeg
ffmpeg is needed to encode the generated video into MP4 format.
```bash
sudo apt update
sudo apt install -y ffmpeg
```
Verify: `ffmpeg -version` should print version info.

### Install cuDNN (if not already installed)
cuDNN is NVIDIA's deep learning library. Check if it's installed:
```bash
dpkg -l | grep cudnn
```
If not installed:
```bash
# For CUDA 13:
sudo apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13

# For CUDA 12:
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
```
> **What is cuDNN?** It has optimized functions for neural networks (convolutions, attention, etc.). It makes inference faster.

---

## Step 3: Get the Kit

### Option A: Copy from an existing setup
If someone gave you the kit folder (with models already included):
```bash
# Just copy the entire folder
cp -r /path/to/LTX-Video-Kit ~/LTX-Video-Kit
cd ~/LTX-Video-Kit
```

### Option B: Clone from GitHub + download models
```bash
# Clone the LTX-2 repository
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2

# The kit structure should be set up like this:
mkdir -p ~/LTX-Video-Kit/{app,packages,models,scripts,sample_outputs}

# Copy the packages
cp -r packages/ltx-core ~/LTX-Video-Kit/packages/
cp -r packages/ltx-pipelines ~/LTX-Video-Kit/packages/
cp -r packages/ltx-trainer ~/LTX-Video-Kit/packages/
cp pyproject.toml uv.lock LICENSE ~/LTX-Video-Kit/packages/
cp LICENSE ~/LTX-Video-Kit/

# Copy the app files (from this kit)
# The app/ directory contains: app.py, pipeline_worker.py, generate.py, etc.
# These files are included in the kit distribution
```

### Option C: You already have this kit
If you're reading this file inside the kit, you're already set:
```bash
cd ~/LTX-Video-Kit   # or wherever you placed it
```

---

## Step 4: Create Python Environment

A virtual environment keeps LTX-2.3's packages separate from your system Python. This prevents conflicts.

```bash
cd ~/LTX-Video-Kit

# Create the virtual environment
python3 -m venv venv

# Activate it (you'll need to do this every time you open a new terminal)
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

> **What is a venv?** It's an isolated Python installation. Packages installed here won't affect your system Python or other projects.

When activated, your terminal prompt will show `(venv)` at the beginning.

---

## Step 5: Install PyTorch

PyTorch is the deep learning framework that runs the model. You need the version that matches your CUDA version.

### Find your CUDA version
```bash
nvcc --version | grep release
```
Look for something like `release 13.0` or `release 12.4`.

### Install PyTorch

**For CUDA 13.x (DGX Spark, Blackwell GPUs):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**For CUDA 12.x (most other NVIDIA GPUs):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

This downloads ~2-3 GB and takes a few minutes.

### Verify PyTorch works with your GPU
```bash
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name:        {torch.cuda.get_device_name(0)}')
    print(f'GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

You should see `CUDA available: True` and your GPU name. If not, your CUDA installation or PyTorch version doesn't match — go back and check.

> **Note**: You may see a warning about `sm_121` not being supported. This is safe to ignore — it still works.

---

## Step 6: Install LTX-2 Packages

LTX-2 has two main packages:
- **ltx-core**: The model architecture (transformer, VAE, quantization)
- **ltx-pipelines**: The inference pipelines (distilled, HQ, etc.)

```bash
# Install the build system first
pip install "uv-build>=0.9.8,<0.11"

# Install ltx-core (editable mode so you can modify if needed)
pip install -e packages/ltx-core

# Install ltx-pipelines
pip install -e packages/ltx-pipelines
```

> **What does `-e` mean?** Editable mode. Python uses the source code directly from the `packages/` folder instead of copying it. This means if you update the code, you don't need to reinstall.

---

## Step 7: Install Dependencies

### Critical: Pin transformers version
The `transformers` library (by HuggingFace) must be version **4.57.6**. Newer versions have a breaking change with the Gemma-3 text encoder.

```bash
pip install "transformers==4.57.6"
```

### Install everything else
```bash
pip install -r app/requirements.txt
```

This installs Flask (web server), scipy, accelerate, and other packages.

### Verify
```bash
python3 -c "
from ltx_pipelines.distilled import DistilledPipeline
print('LTX Pipeline: OK')
from flask import Flask
print('Flask: OK')
import transformers
print(f'Transformers: {transformers.__version__}')
"
```

All three should print without errors. Transformers should show `4.57.6`.

---

## Step 8: Download Model Weights

The model weights are the AI's "brain" — the trained parameters. Total: **~67 GB**.

### Automated download
```bash
bash scripts/download_models.sh
```

### Manual download (if the script doesn't work)

First, log in to HuggingFace:
```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

Then download each model:

**1. Main model (43 GB)**
```bash
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled.safetensors \
    --local-dir models/ \
    --local-dir-use-symlinks False
```

**2. Spatial upscaler (950 MB)**
```bash
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --local-dir models/ \
    --local-dir-use-symlinks False
```

**3. Gemma-3 text encoder (23 GB)**
> You must first visit [the model page](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) and accept Google's license.

```bash
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir models/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir-use-symlinks False
```

### Verify models are downloaded
```bash
ls -lh models/
# Should show:
#   ltx-2.3-22b-distilled.safetensors           43G
#   ltx-2.3-spatial-upscaler-x2-1.0.safetensors  950M
#   gemma-3-12b-it-qat-q4_0-unquantized/         (directory, ~23G total)
```

---

## Step 9: Test: Generate Your First Video (CLI)

Let's generate a quick test video at low resolution to make sure everything works.

```bash
cd app

# Set the Python path
export PYTHONPATH="../packages/ltx-core/src:../packages/ltx-pipelines/src:$PYTHONPATH"

# Generate a 1-second video at 512x768 (fast, ~10 seconds)
python generate.py \
    --prompt "A golden retriever running through a sunny meadow, slow motion, cinematic" \
    --height 512 --width 768 \
    --num-frames 25 \
    --seed 42
```

### What to expect
1. "Loading models..." — First time takes 30-60 seconds (loading 67GB into GPU memory)
2. "Generating video..." — About 10-30 seconds for this small resolution
3. "Saved to: outputs/ltx23_XXXXXXXX_XXXXXX.mp4" — Your video!

### Play the video
```bash
# If you have a display:
xdg-open outputs/ltx23_*.mp4

# Or check the file details:
ffprobe -v quiet -print_format json -show_format -show_streams outputs/ltx23_*.mp4 | head -30
```

You should see an MP4 file with H.264 video and AAC audio.

> **If this step fails**, see [Known Issues and Fixes](#step-12-known-issues-and-fixes) below.

---

## Step 10: Start the Web UI

```bash
cd ~/LTX-Video-Kit/app
bash run.sh
```

You should see:
```
════════════════════════════════════════════════════════════
  LTX-2.3 Video Studio
  URL:    http://0.0.0.0:5000
════════════════════════════════════════════════════════════
```

Open your browser and go to:
- **Local**: `http://localhost:5000`
- **Network**: `http://<your-ip>:5000` (so others on the network can access it)

### Using the Web UI

1. **Type a prompt** in the text box on the left
2. **Choose resolution** — start with "512 × 768 (Fast)" for quick tests
3. **Choose frames** — "25 frames (1.0s)" is fastest
4. **Click "Generate Video"**
5. Watch the **progress bar** fill up with step-by-step updates
6. When done, the video plays automatically
7. Past videos appear in the **Gallery** at the bottom

### Tips
- The **first generation** is slow because models need to load (~30-60s). After that, it's much faster.
- Use **FP8 Quantization** toggle if you're running low on GPU memory
- The **"Enhance Prompt"** toggle uses Gemma to add cinematic details to your prompt
- For **image-to-video**, click the "Image → Video" tab, upload an image, and add a prompt

---

## Step 11: Verification Checklist

After setup, verify each of these works:

- [ ] `nvidia-smi` shows your GPU
- [ ] `python3 -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [ ] `python3 -c "from ltx_pipelines.distilled import DistilledPipeline; print('OK')"` prints `OK`
- [ ] Models exist: `ls models/*.safetensors` shows 2 files
- [ ] Gemma encoder exists: `ls models/gemma-3-12b-it-qat-q4_0-unquantized/*.safetensors | wc -l` shows `5`
- [ ] CLI test: `cd app && python generate.py --prompt "test" --height 512 --width 768 --num-frames 25` creates a video
- [ ] Web UI: `cd app && bash run.sh` → open browser → generate a video → gallery shows it

---

## Step 12: Known Issues and Fixes

These are real errors we encountered during setup and their solutions.

### Error: `'Gemma3TextConfig' object has no attribute 'rope_local_base_freq'`
**Cause**: Wrong version of `transformers`. Version 5.x changed the Gemma config format.
**Fix**:
```bash
pip install "transformers==4.57.6"
```
This is the version pinned in the LTX-2 lockfile and is guaranteed to work.

---

### Error: `CUDA out of memory`
**Cause**: Not enough GPU memory for the chosen resolution/frame count.
**Fix**: Use smaller settings:
- Resolution: 512×768 instead of 1024×1536
- Frames: 25 instead of 121
- Enable FP8: `--quantization fp8-cast`

---

### Error: `nvcc not found` or `CUDA_HOME not set`
**Cause**: CUDA is installed but not on your PATH.
**Fix**: Add to `~/.bashrc`:
```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```
Then run `source ~/.bashrc`.

---

### Error: `ModuleNotFoundError: No module named 'ltx_pipelines'`
**Cause**: The LTX packages aren't installed or aren't on the Python path.
**Fix** (option A — set PYTHONPATH):
```bash
export PYTHONPATH="/path/to/LTX-Video-Kit/packages/ltx-core/src:/path/to/LTX-Video-Kit/packages/ltx-pipelines/src:$PYTHONPATH"
```
**Fix** (option B — reinstall):
```bash
pip install -e packages/ltx-core packages/ltx-pipelines
```

---

### Error: `ImportError: cannot import name 'calculate_weight_float8' from partially initialized module`
**Cause**: Circular import. Happens when running Python from inside the `packages/` directory.
**Fix**: Always run from the `app/` directory:
```bash
cd ~/LTX-Video-Kit/app
python generate.py --prompt "test" ...
```

---

### Error: `Permission denied` when downloading models (xet log error)
**Cause**: HuggingFace cache directory is owned by root.
**Fix**:
```bash
sudo chown -R $(whoami) ~/.cache/huggingface/
```

---

### Error: `cuDNN version mismatch` or `libcudnn not found`
**Cause**: cuDNN isn't installed or is the wrong version for your CUDA.
**Fix**:
```bash
# For CUDA 13:
sudo apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13

# For CUDA 12:
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
```

---

### Warning: `Found GPU which is of cuda capability 12.1. Minimum supported is (8.0) - (12.0)`
**Cause**: PyTorch doesn't officially list compute capability 12.1 (Blackwell/GB10) yet.
**Impact**: None — it works fine. This is just a warning.
**Fix**: Ignore it. Everything runs correctly.

---

### Web UI: First generation takes very long
**Cause**: Normal! The models (67GB) need to be loaded into GPU memory the first time.
**Impact**: First generation takes 30-60 seconds extra. Subsequent ones are fast.
**Fix**: Just wait. The progress bar will show "Loading models..." during this phase.

---

## Summary

| Step | Time | What it does |
|------|------|-------------|
| System check | 2 min | Verify GPU, CUDA, Python |
| System packages | 2 min | Install ffmpeg, cuDNN |
| Create venv | 1 min | Isolated Python environment |
| Install PyTorch | 3 min | GPU-accelerated ML framework |
| Install LTX packages | 2 min | Model code + pipelines |
| Install deps | 2 min | Flask, transformers, etc. |
| Download models | 30-60 min | 67 GB of model weights |
| Test CLI | 2 min | Verify generation works |
| Start Web UI | 1 min | Launch the video studio |
| **Total** | **~45-75 min** | **Mostly download time** |

Once set up, generating a 1-second video at 512×768 takes about **10 seconds** (after the initial model load).

---

*This guide was written based on a successful setup on NVIDIA DGX Spark (March 2026).*
