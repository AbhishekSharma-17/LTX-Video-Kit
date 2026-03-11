#!/usr/bin/env python3
"""
LTX-2.3 Video Generation Wrapper for DGX Spark

Usage:
    # Text-to-video (defaults: 1024x1536, 121 frames, 24fps)
    python generate.py --prompt "A golden retriever running through a meadow"

    # Lower resolution for faster/safer generation
    python generate.py --prompt "A sunset over the ocean" --height 512 --width 768 --num-frames 25

    # Image-to-video
    python generate.py --prompt "The scene comes alive" --image input.jpg 0 1.0

    # With FP8 quantization (less memory)
    python generate.py --prompt "A forest scene" --quantization fp8-cast
"""

import argparse
import os
import sys
import time

# Memory optimization for unified memory architecture
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

# Ensure CUDA is on PATH
if "/usr/local/cuda-13.0/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"/usr/local/cuda-13.0/bin:{os.environ.get('PATH', '')}"
if "/usr/local/cuda-13.0/lib64" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda-13.0/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent       # app/
_KIT_DIR = _SCRIPT_DIR.parent                       # LTX-Video-Kit/
MODEL_DIR = str(_KIT_DIR / "models")
OUTPUT_DIR = str(_SCRIPT_DIR / "outputs")

DISTILLED_CKPT = os.path.join(MODEL_DIR, "ltx-2.3-22b-distilled.safetensors")
SPATIAL_UPSAMPLER = os.path.join(MODEL_DIR, "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
GEMMA_ROOT = os.path.join(MODEL_DIR, "gemma-3-12b-it-qat-q4_0-unquantized")


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 Video Generator for DGX Spark")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--height", type=int, default=1024,
                        help="Output height in pixels, divisible by 64 (default: 1024)")
    parser.add_argument("--width", type=int, default=1536,
                        help="Output width in pixels, divisible by 64 (default: 1536)")
    parser.add_argument("--num-frames", type=int, default=121,
                        help="Number of frames (must be 8n+1, default: 121)")
    parser.add_argument("--frame-rate", type=float, default=24.0,
                        help="Frame rate (default: 24.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--image", nargs="+", action="append", default=[],
                        help="Image conditioning: PATH FRAME_IDX STRENGTH [CRF]. Can repeat.")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["fp8-cast", "fp8-scaled-mm"],
                        help="Enable FP8 quantization for lower memory")
    parser.add_argument("--enhance-prompt", action="store_true",
                        help="Use Gemma to enhance the prompt before generation")
    parser.add_argument("--distilled-checkpoint", type=str, default=DISTILLED_CKPT,
                        help="Path to distilled model checkpoint")
    parser.add_argument("--spatial-upsampler", type=str, default=SPATIAL_UPSAMPLER,
                        help="Path to spatial upsampler")
    parser.add_argument("--gemma-root", type=str, default=GEMMA_ROOT,
                        help="Path to Gemma text encoder directory")
    args = parser.parse_args()

    # Validate dimensions
    assert args.height % 64 == 0, f"Height must be divisible by 64, got {args.height}"
    assert args.width % 64 == 0, f"Width must be divisible by 64, got {args.width}"
    assert (args.num_frames - 1) % 8 == 0, (
        f"num-frames must be 8n+1 (e.g., 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121), "
        f"got {args.num_frames}"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(OUTPUT_DIR, f"ltx23_{timestamp}.mp4")

    print("=" * 60)
    print("LTX-2.3 Video Generation (DistilledPipeline)")
    print("=" * 60)
    print(f"Prompt:     {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames:     {args.num_frames} @ {args.frame_rate}fps "
          f"({args.num_frames / args.frame_rate:.1f}s)")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {args.output}")
    if args.quantization:
        print(f"Quantization: {args.quantization}")
    print("=" * 60)

    import torch
    from ltx_core.quantization import QuantizationPolicy
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput

    # Blackwell (GB10) optimization: enable cuDNN auto-tuning and TF32 math
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from ltx_pipelines.utils.media_io import encode_video

    # Parse image conditioning args
    images = []
    for img_args in args.image:
        if len(img_args) < 3:
            parser.error("--image requires at least PATH FRAME_IDX STRENGTH")
        path = img_args[0]
        frame_idx = int(img_args[1])
        strength = float(img_args[2])
        crf = int(img_args[3]) if len(img_args) > 3 else 23
        images.append(ImageConditioningInput(path, frame_idx, strength, crf))

    # Set up quantization
    quantization = None
    if args.quantization == "fp8-cast":
        quantization = QuantizationPolicy.fp8_cast()
    elif args.quantization == "fp8-scaled-mm":
        quantization = QuantizationPolicy.fp8_scaled_mm()

    print("\nLoading models...")
    load_start = time.time()

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint,
        spatial_upsampler_path=args.spatial_upsampler,
        gemma_root=args.gemma_root,
        loras=[],
        quantization=quantization,
    )

    load_time = time.time() - load_start
    print(f"Models loaded in {load_time:.1f}s")

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    print("\nGenerating video...")
    gen_start = time.time()

    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    gen_time = time.time() - gen_start
    print(f"Generation complete in {gen_time:.1f}s")

    print("\nEncoding output video...")
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output,
        video_chunks_number=video_chunks_number,
    )

    total_time = time.time() - load_start
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    import torch
    # NOTE: Use torch.no_grad() instead of torch.inference_mode().
    # The pipeline returns a lazy video iterator — VAE decode runs inside
    # encode_video().  inference_mode() creates special tensors that break
    # when the VAE's conv3d layers try to save them for backward.
    with torch.no_grad():
        main()
