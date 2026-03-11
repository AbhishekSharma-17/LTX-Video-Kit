# LTX-2.3 Video Studio

A self-contained kit for generating AI videos using **LTX-2.3** — a 22-billion-parameter open-source video generation model by [Lightricks](https://github.com/Lightricks/LTX-2). This kit includes a polished web UI, CLI tool, and everything needed to generate high-quality videos from text prompts (or images) on an NVIDIA GPU.

> **Built and tested on NVIDIA DGX Spark (GB10 GPU, 128GB unified memory, ARM/aarch64)**

---

## Clone & Run (3 commands)

```bash
# 1. Clone the repo
git clone <your-repo-url> LTX-Video-Kit
cd LTX-Video-Kit

# 2. Setup environment + install dependencies + download models (~67 GB)
bash scripts/setup.sh
bash scripts/download_models.sh

# 3. Start the web UI
cd app && bash run.sh
# Open http://localhost:5000 in your browser
```

> **First time?** Read the full walkthrough in [REPLICATION_GUIDE.md](REPLICATION_GUIDE.md).

### What gets cloned vs downloaded separately

| Included in Git (~4 MB) | Downloaded by scripts (~72 GB) |
|--------------------------|-------------------------------|
| All application code | `models/` — 67 GB model weights |
| LTX-2 source packages | `venv/` — ~5 GB Python environment |
| Web UI (HTML/CSS/JS) | |
| Setup & download scripts | |
| Documentation | |

The `models/` and `venv/` directories are in `.gitignore` — they are created locally by `setup.sh` and `download_models.sh`.

---

## What is LTX-2.3?

LTX-2.3 is a **Diffusion Transformer (DiT)** model that generates videos with synchronized audio. Key facts:

| Property | Value |
|----------|-------|
| Parameters | 22 billion |
| Architecture | DiT (Diffusion Transformer) |
| Released | March 2026 by Lightricks |
| Input | Text prompt (and/or image) |
| Output | MP4 video with audio |
| Max resolution | 1024 × 1536 |
| Max duration | 5 seconds (121 frames @ 24fps) |
| Inference steps | 8 (Stage 1) + 3 (Stage 2) = 11 total |
| Model format | SafeTensors (bf16) |
| Text encoder | Gemma-3 12B (quantized to 4-bit) |

The **distilled** variant used here generates video in just **11 denoising steps** through a two-stage process:
1. **Stage 1**: Generate video at half resolution (8 denoising steps)
2. **Spatial upsampling**: 2x upscale with a dedicated upsampler model
3. **Stage 2**: Refine at full resolution (3 denoising steps)

---

## Features

### Web UI (Video Studio)
- **Text-to-Video**: Type a prompt, get a video
- **Image-to-Video**: Upload an image, animate it with a prompt
- **Resolution presets**: 512x768 (fast), 768x1280 (medium), 1024x1536 (full quality)
- **Frame presets**: 25 (1s), 49 (2s), 73 (3s), 97 (4s), 121 (5s)
- **Real-time progress**: Per-step progress bar with ETA, elapsed time, step timing
- **Gallery**: Browse, replay, and delete past generations
- **FP8 quantization toggle**: Use less memory at the cost of slight quality reduction
- **Prompt enhancement**: Let Gemma improve your prompt before generation
- **Pipeline status**: See GPU memory usage and model loading state in the header
- **Metrics dashboard**: Generation time, load time, resolution, seed, file size

### CLI Tool
- Same features via command-line arguments
- Good for scripting and batch generation

### Architecture
```
  Browser (Tailwind CSS dark UI)
     |
     v
  Flask (app.py) --- SSE (Server-Sent Events) --> Browser updates
     |
     v
  PipelineManager (pipeline_worker.py)
     |  - Singleton pipeline (loaded once, reused)
     |  - Background worker thread + job queue
     |  - Monkey-patched denoising loop for progress
     |
     v
  DistilledPipeline (ltx-pipelines)
     |  - Stage 1: 8-step denoising at half res
     |  - Spatial upsampling 2x
     |  - Stage 2: 3-step refinement at full res
     |  - Video VAE decode + Audio VAE decode
     |
     v
  MP4 output (H.264 video + AAC audio)
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| GPU | NVIDIA with 40GB+ VRAM | 80GB+ or unified memory |
| CUDA | 12.0+ | 13.0 |
| Python | 3.10+ | 3.12 |
| RAM | 32 GB | 64 GB+ |
| Storage | 80 GB free | 100 GB free |
| OS | Linux (Ubuntu 22.04+) | Ubuntu 24.04 |
| ffmpeg | Required | Required |

**Tested on**: DGX Spark — NVIDIA GB10 (Blackwell), 128GB unified memory, CUDA 13.0, aarch64/ARM, Ubuntu.

---

## CLI Usage

```bash
# Activate the environment
source venv/bin/activate

# Basic text-to-video
cd app
python generate.py --prompt "A golden retriever running through a meadow"

# Lower resolution for faster generation (~10 seconds)
python generate.py --prompt "A sunset over the ocean" \
    --height 512 --width 768 --num-frames 25

# With FP8 quantization (uses less memory)
python generate.py --prompt "A forest scene" --quantization fp8-cast

# Image-to-video
python generate.py --prompt "The scene comes alive" \
    --image path/to/photo.jpg 0 1.0
```

---

## Web UI API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI page |
| `POST` | `/api/generate` | Submit a generation job. Body: `{prompt, height, width, num_frames, ...}`. Returns `{job_id}` |
| `GET` | `/api/status/<job_id>` | SSE stream. Events: `stage`, `progress`, `complete`, `error` |
| `GET` | `/api/gallery` | List of past generations (JSON array) |
| `DELETE` | `/api/gallery/<job_id>` | Delete a generation and its video file |
| `POST` | `/api/upload` | Upload an image for image-to-video. Returns `{path, filename}` |
| `GET` | `/api/pipeline/status` | Pipeline state, GPU info. Returns `{loaded, loading, gpu: {...}}` |
| `GET` | `/outputs/<filename>` | Serve a generated video file |

---

## Model Files

Models are **not included in the git repo** (67 GB total). Download them with:

```bash
bash scripts/download_models.sh
```

This downloads into the `models/` directory:

| File | Size | Source |
|------|------|--------|
| `ltx-2.3-22b-distilled.safetensors` | 43 GB | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | 950 MB | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | 23 GB | [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) |

The Gemma-3 model requires accepting Google's license on HuggingFace before downloading.

---

## Folder Structure

```
LTX-Video-Kit/
├── README.md                    <- You are here
├── REPLICATION_GUIDE.md         <- Step-by-step setup from scratch
├── LICENSE                      <- Apache 2.0 (from Lightricks)
├── .gitignore                   <- Excludes models/, venv/, outputs/
|
├── app/                         <- Web application
│   ├── app.py                   <- Flask routes + SSE streaming
│   ├── pipeline_worker.py       <- Pipeline engine + progress tracking
│   ├── generate.py              <- CLI generation tool
│   ├── run.sh                   <- Start the web server
│   ├── requirements.txt         <- Python dependencies
│   ├── templates/index.html     <- Web UI (Tailwind CSS dark theme)
│   └── static/css/custom.css    <- Animations + styling
|
├── packages/                    <- LTX-2 source code (in git)
│   ├── ltx-core/                <- Core ML library
│   ├── ltx-pipelines/           <- Inference pipelines
│   └── ltx-trainer/             <- Training framework (optional)
|
├── models/                      <- Model weights - NOT in git (~67 GB)
├── venv/                        <- Python venv - NOT in git (~5 GB)
|
└── scripts/
    ├── setup.sh                 <- Creates venv + installs everything
    └── download_models.sh       <- Downloads models from HuggingFace
```

---

## How It Works (Simple Explanation)

### What happens when you click "Generate"?

1. **You type a prompt** like "A cat playing piano" and click Generate
2. **The prompt goes to the server** (Flask receives it as a JSON POST)
3. **Gemma-3 encodes the text** — the 12B text encoder converts your words into numbers (embeddings) that the video model understands
4. **Stage 1 — Low-res generation** — The 22B transformer starts with random noise and removes it step by step (8 steps), producing a blurry low-resolution video
5. **Spatial upsampling** — A specialized model doubles the resolution (e.g., 384x256 -> 768x512)
6. **Stage 2 — High-res refinement** — 3 more denoising steps clean up the upscaled video
7. **VAE decoding** — The model converts internal representations back to actual pixel colors (video) and sound waves (audio)
8. **MP4 encoding** — The raw pixels are compressed into an H.264 video with AAC audio
9. **Done!** — The video URL is sent to your browser, which plays it automatically

### Why does the progress bar work?

The system intercepts the denoising loop (the heart of the AI) by replacing it with a version that reports every step. Each step takes ~1-3 seconds. The browser receives these updates in real-time via **Server-Sent Events** (a simple streaming protocol built into every browser).

### Why is the first generation slow?

The first time you generate, the system loads 67GB of model weights into GPU memory. This takes 30-60 seconds. After that, the models stay loaded and subsequent generations are much faster (10-120 seconds depending on resolution and frame count).

---

## Troubleshooting

### `'Gemma3TextConfig' object has no attribute 'rope_local_base_freq'`
**Fix**: You have the wrong version of `transformers`. Pin it:
```bash
pip install "transformers==4.57.6"
```

### `CUDA out of memory`
**Fix**: Use a lower resolution or enable FP8 quantization:
- In the web UI: toggle "FP8 Quantization" on
- CLI: add `--quantization fp8-cast`
- Or use 512x768 resolution with 25 frames

### `nvcc not found` or CUDA errors
**Fix**: Add CUDA to your PATH:
```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

### `ModuleNotFoundError: No module named 'ltx_pipelines'`
**Fix**: Make sure the packages are on your Python path:
```bash
export PYTHONPATH="packages/ltx-core/src:packages/ltx-pipelines/src:$PYTHONPATH"
```
Or reinstall them in editable mode:
```bash
pip install -e packages/ltx-core packages/ltx-pipelines
```

### `ImportError: cannot import name 'calculate_weight_float8'`
**Fix**: This is a circular import issue. Always run from the `app/` directory, not from inside the `packages/` directory.

### Port 5000 already in use
**Fix**: Kill the existing process or use a different port:
```bash
lsof -ti:5000 | xargs kill
```

### HuggingFace download fails
**Fix**: Make sure you're logged in and have accepted model licenses:
```bash
huggingface-cli login
# Then visit the model page and accept the license
```

---

## Credits

- **LTX-2.3 model**: [Lightricks](https://github.com/Lightricks/LTX-2) (Apache 2.0)
- **Gemma-3 text encoder**: [Google](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
- **Web UI**: Built with Flask, Tailwind CSS, and vanilla JavaScript
- **Tested on**: NVIDIA DGX Spark
