"""
Pipeline Worker — backend engine for the LTX-2.3 Web UI.

Manages:
- Pipeline singleton (load once, reuse across jobs)
- Job queue + background worker thread
- Monkey-patched euler_denoising_loop for step-level progress
- SSE event broadcasting per job
- History file (JSON) management
"""

import gc
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── Environment ──────────────────────────────────────────────
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

if "/usr/local/cuda-13.0/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"/usr/local/cuda-13.0/bin:{os.environ.get('PATH', '')}"
if "/usr/local/cuda-13.0/lib64" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = (
        f"/usr/local/cuda-13.0/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

# ── Paths ────────────────────────────────────────────────────
# BASE_DIR = app/    KIT_DIR = LTX-Video-Kit/
BASE_DIR = Path(__file__).resolve().parent
KIT_DIR = BASE_DIR.parent
MODEL_DIR = KIT_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
HISTORY_FILE = BASE_DIR / "history.json"

DISTILLED_CKPT = str(MODEL_DIR / "ltx-2.3-22b-distilled.safetensors")
SPATIAL_UPSAMPLER = str(MODEL_DIR / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
GEMMA_ROOT = str(MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized")


# ── Smart Tiling ────────────────────────────────────────────

def compute_tiling_config(
    height: int,
    width: int,
    num_frames: int,
    spatial_tile_override: int = 0,
    temporal_tile_override: int = 0,
):
    """
    Compute optimal tiling configuration based on resolution and frame count.

    Tiling constraints (from ltx_core tiling.py):
    - spatial tile_size >= 64, divisible by 32
    - spatial overlap divisible by 32, < tile_size
    - temporal tile_size >= 16, divisible by 8
    - temporal overlap divisible by 8, < tile_size
    """
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig

    max_dim = max(height, width)

    # ── Spatial tiling ──
    if spatial_tile_override > 0:
        spatial_tile = spatial_tile_override
        spatial_overlap = min(128, spatial_tile // 4)
        spatial_overlap = max(32, (spatial_overlap // 32) * 32)
    elif max_dim <= 1024:
        spatial_tile = 512
        spatial_overlap = 64
    elif max_dim <= 1920:
        spatial_tile = 512
        spatial_overlap = 64
    elif max_dim <= 2560:
        spatial_tile = 640
        spatial_overlap = 96
    else:
        # 4K: larger tiles with more overlap for seam quality
        spatial_tile = 768
        spatial_overlap = 128

    # ── Temporal tiling ──
    if temporal_tile_override > 0:
        temporal_tile = temporal_tile_override
        temporal_overlap = min(32, temporal_tile // 3)
        temporal_overlap = max(8, (temporal_overlap // 8) * 8)
    elif num_frames <= 64:
        temporal_tile = max(16, ((num_frames - 1) // 8) * 8)
        temporal_overlap = min(24, max(8, ((temporal_tile // 3) // 8) * 8))
    elif num_frames <= 200:
        temporal_tile = 64
        temporal_overlap = 24
    else:
        # Long videos: same tile size, larger overlap for temporal coherence
        temporal_tile = 64
        temporal_overlap = 32

    return TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=spatial_tile,
            tile_overlap_in_pixels=spatial_overlap,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=temporal_tile,
            tile_overlap_in_frames=temporal_overlap,
        ),
    )


# ── Job / Event types ───────────────────────────────────────
@dataclass
class JobRequest:
    job_id: str
    prompt: str
    height: int = 1024
    width: int = 1536
    num_frames: int = 121
    frame_rate: float = 24.0
    seed: int = 42
    quantization: Optional[str] = None
    enhance_prompt: bool = False
    images: list = field(default_factory=list)   # list of dicts
    # Optional tiling overrides (0 = auto)
    tiling_spatial_tile: int = 0
    tiling_temporal_tile: int = 0


@dataclass
class JobEvent:
    event: str           # "progress" | "complete" | "error" | "stage"
    data: dict = field(default_factory=dict)


# ── Pipeline Manager ────────────────────────────────────────
class PipelineManager:
    """Thread-safe singleton that owns the DistilledPipeline and processes jobs."""

    def __init__(self):
        self._pipeline = None
        self._current_quantization: Optional[str] = None
        self._loading = False
        self._loaded = False

        # Job infrastructure
        self._job_queue: queue.Queue[JobRequest] = queue.Queue()
        self._event_queues: dict[str, queue.Queue[JobEvent]] = {}
        self._active_job_id: Optional[str] = None
        self._lock = threading.Lock()

        # Progress state (written by monkey-patch, read by SSE)
        self._denoising_call_count = 0

        # Start background worker
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # Pre-load pipeline via the worker thread to avoid threading issues.
        # The worker thread loads models in the same context as real jobs.
        self._preload_quantization = os.environ.get("LTX_DEFAULT_QUANTIZATION", "fp8-cast")
        if os.environ.get("LTX_SKIP_PRELOAD", "").lower() not in ("1", "true", "yes"):
            preload_req = JobRequest(
                job_id="__preload__",
                prompt="__preload__",
                quantization=self._preload_quantization if self._preload_quantization != "none" else None,
            )
            self._job_queue.put(preload_req)

    # ── public helpers ───────────────────────────────────────

    def submit_job(self, req: JobRequest) -> str:
        eq: queue.Queue[JobEvent] = queue.Queue()
        with self._lock:
            self._event_queues[req.job_id] = eq
        self._job_queue.put(req)
        return req.job_id

    def get_event_queue(self, job_id: str) -> Optional[queue.Queue]:
        with self._lock:
            return self._event_queues.get(job_id)

    def cleanup_job(self, job_id: str):
        with self._lock:
            self._event_queues.pop(job_id, None)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def active_job(self) -> Optional[str]:
        return self._active_job_id

    @property
    def current_quantization(self) -> Optional[str]:
        return self._current_quantization

    def gpu_info(self) -> dict:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                total_gb = total / (1024 ** 3)
                free_gb = free / (1024 ** 3)
                used_gb = total_gb - free_gb
                return {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "total_gb": round(total_gb, 1),
                    "used_gb": round(used_gb, 1),
                    "free_gb": round(free_gb, 1),
                }
        except Exception:
            pass
        return {}

    # ── estimation ────────────────────────────────────────────

    def estimate_generation(
        self,
        height: int,
        width: int,
        num_frames: int,
        quantization: Optional[str] = None,
    ) -> dict:
        """Estimate VRAM usage and generation time for given parameters."""

        # Latent dimensions (VAE: 8x temporal, ~32x spatial compression)
        f_latent = (num_frames - 1) // 8 + 1

        # Stage 2 (full-res) latent is the peak — larger than stage 1
        s2_h_lat = height // 16
        s2_w_lat = width // 16
        s2_latent_elements = 128 * f_latent * s2_h_lat * s2_w_lat
        s2_latent_bytes = s2_latent_elements * 2  # bfloat16

        # Model memory (approximate)
        transformer_gb = 22.0 if quantization else 43.0
        upsampler_gb = 0.95
        gemma_gb = 12.0  # Q4 variant loaded transiently

        # Peak VRAM: model + latent + ~3x activations overhead
        peak_latent_gb = s2_latent_bytes / (1024 ** 3)
        activation_overhead = peak_latent_gb * 3
        peak_vram_gb = transformer_gb + peak_latent_gb + activation_overhead + upsampler_gb

        # Time estimation (calibrated against DGX Spark reference)
        # Reference: ~5s per step at 1024x1536, 121 frames, BF16
        ref_pixels = 1024 * 1536 * 121
        pixel_ratio = (height * width * num_frames) / ref_pixels
        base_time_per_step = 5.0
        time_per_step = base_time_per_step * max(pixel_ratio, 0.1)
        if quantization:
            time_per_step *= 0.7  # FP8 ~30% faster

        total_steps = 11  # 8 + 3 distilled
        gen_time_est = time_per_step * total_steps
        load_time_est = 0 if self._loaded else 90

        # GPU info
        total_gb = 128.0
        free_gb = 128.0
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                free_gb = free / (1024 ** 3)
                total_gb = total / (1024 ** 3)
        except Exception:
            pass

        feasible = peak_vram_gb < free_gb * 0.9

        return {
            "estimated_vram_gb": round(peak_vram_gb, 1),
            "available_vram_gb": round(free_gb, 1),
            "estimated_gen_time_s": round(gen_time_est),
            "estimated_load_time_s": round(load_time_est),
            "estimated_total_time_s": round(gen_time_est + load_time_est),
            "feasible": feasible,
            "warning": (
                None
                if feasible
                else (
                    f"Estimated VRAM ({peak_vram_gb:.1f}GB) may exceed available "
                    f"({free_gb:.1f}GB). Enable FP8 or reduce resolution."
                )
            ),
        }

    # ── event emitter ────────────────────────────────────────

    def _emit(self, job_id: str, event: str, data: dict | None = None):
        eq = self.get_event_queue(job_id)
        if eq:
            eq.put(JobEvent(event=event, data=data or {}))

    # ── worker loop ──────────────────────────────────────────

    def _worker_loop(self):
        while True:
            req = self._job_queue.get()
            # Handle pre-load sentinel (just load models, skip generation)
            if req.job_id == "__preload__":
                try:
                    print(f"[PipelineManager] Pre-loading models (quantization={req.quantization})...")
                    self._ensure_pipeline(req.quantization)
                    print("[PipelineManager] Models pre-loaded successfully.")
                except Exception as e:
                    print(f"[PipelineManager] Pre-load failed (will retry on first request): {e}")
                finally:
                    self._job_queue.task_done()
                continue

            self._active_job_id = req.job_id
            try:
                self._run_job(req)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._emit(req.job_id, "error", {"message": str(e)})
            finally:
                self._active_job_id = None
                self._job_queue.task_done()

    # ── model loading ────────────────────────────────────────

    def _ensure_pipeline(self, quantization: Optional[str] = None):
        if self._pipeline is not None and self._current_quantization == quantization:
            return

        self._loading = True  # Set early so UI shows "Loading…" during imports

        import torch
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        # Blackwell (GB10) optimization: enable cuDNN auto-tuning and TF32 math
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            gc.collect()
            torch.cuda.empty_cache()

        quant_policy = None
        if quantization == "fp8-cast":
            quant_policy = QuantizationPolicy.fp8_cast()
        elif quantization == "fp8-scaled-mm":
            quant_policy = QuantizationPolicy.fp8_scaled_mm()

        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=DISTILLED_CKPT,
            spatial_upsampler_path=SPATIAL_UPSAMPLER,
            gemma_root=GEMMA_ROOT,
            loras=[],
            quantization=quant_policy,
        )

        # Apply torch.compile() to the transformer for 20-40% speedup.
        # Uses max-autotune-no-cudagraphs to work with varying input sizes.
        # Falls back gracefully if compilation is not supported.
        try:
            original_transformer_fn = self._pipeline.model_ledger.transformer
            _compile_cache = {}

            def compiled_transformer():
                if "model" not in _compile_cache:
                    model = original_transformer_fn()
                    try:
                        _compile_cache["model"] = torch.compile(
                            model, mode="max-autotune-no-cudagraphs"
                        )
                        print("[PipelineManager] torch.compile() applied to transformer")
                    except Exception as e:
                        print(f"[PipelineManager] torch.compile() failed, using eager: {e}")
                        _compile_cache["model"] = model
                return _compile_cache["model"]

            self._pipeline.model_ledger.transformer = compiled_transformer
        except Exception as e:
            print(f"[PipelineManager] Could not set up torch.compile: {e}")

        # Cache all non-text-encoder models in memory across requests.
        # With 128GB unified memory, keeping ~6GB of extra models resident
        # eliminates ~10-20s of model rebuild time on every generation
        # after the first.
        #
        # We do NOT cache text_encoder or gemma_embeddings_processor because
        # encode_prompts() manages their lifecycle: create → use → delete
        # to free ~12GB before the transformer loads.
        _model_cache = {}

        def _make_cached_factory(name, original_fn):
            def cached_factory():
                if name not in _model_cache:
                    print(f"[PipelineManager] Building and caching {name}")
                    _model_cache[name] = original_fn()
                else:
                    print(f"[PipelineManager] Reusing cached {name}")
                return _model_cache[name]
            return cached_factory

        for model_name in [
            "video_encoder", "video_decoder", "spatial_upsampler",
            "audio_decoder", "vocoder",
        ]:
            try:
                original = getattr(self._pipeline.model_ledger, model_name)
                setattr(
                    self._pipeline.model_ledger,
                    model_name,
                    _make_cached_factory(model_name, original),
                )
            except Exception as e:
                print(f"[PipelineManager] Could not cache {model_name}: {e}")

        self._current_quantization = quantization
        self._loaded = True
        self._loading = False

    # ── monkey-patch for progress ────────────────────────────

    def _install_progress_hook(self):
        """
        Replace euler_denoising_loop in the distilled module namespace
        with a version that emits per-step SSE events.

        The DistilledPipeline.__call__ calls euler_denoising_loop twice:
          1st call → Stage 1 (8 steps, 9 sigma values)
          2nd call → Stage 2 (3 steps, 4 sigma values)
        We use a call counter to distinguish them.
        """
        import ltx_pipelines.distilled as distilled_mod
        import ltx_pipelines.utils.samplers as samplers_mod

        original_fn = samplers_mod.euler_denoising_loop
        self._denoising_call_count = 0
        manager = self

        def patched_euler_denoising_loop(
            sigmas, video_state, audio_state, stepper, denoise_fn
        ):
            from dataclasses import replace as dc_replace
            from ltx_pipelines.utils.helpers import post_process_latent

            manager._denoising_call_count += 1
            call_num = manager._denoising_call_count
            total_steps = len(sigmas) - 1

            # Determine stage name
            if call_num == 1:
                stage = "stage1"
                stage_label = f"Stage 1: Denoising at half resolution"
            else:
                stage = "stage2"
                stage_label = f"Stage 2: Refining at full resolution"

            job_id = manager._active_job_id
            if job_id:
                manager._emit(job_id, "stage", {
                    "stage": stage,
                    "message": stage_label,
                    "total_steps": total_steps,
                })

            loop_start = time.time()

            for step_idx in range(total_steps):
                step_start = time.time()

                denoised_video, denoised_audio = denoise_fn(
                    video_state, audio_state, sigmas, step_idx
                )
                denoised_video = post_process_latent(
                    denoised_video, video_state.denoise_mask, video_state.clean_latent
                )
                denoised_audio = post_process_latent(
                    denoised_audio, audio_state.denoise_mask, audio_state.clean_latent
                )

                video_state = dc_replace(
                    video_state,
                    latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx),
                )
                audio_state = dc_replace(
                    audio_state,
                    latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx),
                )

                step_time = time.time() - step_start
                elapsed = time.time() - loop_start
                avg = elapsed / (step_idx + 1)
                eta = avg * (total_steps - step_idx - 1)

                if job_id:
                    manager._emit(job_id, "progress", {
                        "stage": stage,
                        "step": step_idx + 1,
                        "total_steps": total_steps,
                        "step_time": round(step_time, 2),
                        "elapsed": round(elapsed, 2),
                        "eta": round(eta, 2),
                    })

            return (video_state, audio_state)

        # Patch both namespaces so the inner function picks it up
        distilled_mod.euler_denoising_loop = patched_euler_denoising_loop
        samplers_mod.euler_denoising_loop = patched_euler_denoising_loop
        return original_fn

    def _restore_hook(self, original_fn):
        import ltx_pipelines.distilled as distilled_mod
        import ltx_pipelines.utils.samplers as samplers_mod

        distilled_mod.euler_denoising_loop = original_fn
        samplers_mod.euler_denoising_loop = original_fn

    # ── job execution ────────────────────────────────────────

    def _run_job(self, req: JobRequest):
        import torch
        from ltx_core.model.video_vae import get_video_chunks_number
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.media_io import encode_video

        overall_start = time.time()

        # ── Load models ──────────────────────────────────────
        self._emit(req.job_id, "stage", {
            "stage": "loading",
            "message": "Loading models…",
        })
        load_start = time.time()
        self._ensure_pipeline(req.quantization)
        load_time = time.time() - load_start
        self._emit(req.job_id, "stage", {
            "stage": "loaded",
            "message": f"Models ready ({load_time:.1f}s)",
            "load_time": round(load_time, 1),
        })

        # ── Prepare image conditioning ───────────────────────
        images = []
        for img in req.images:
            images.append(ImageConditioningInput(
                path=img["path"],
                frame_idx=int(img["frame_idx"]),
                strength=float(img["strength"]),
                crf=int(img.get("crf", 23)),
            ))

        # Smart tiling: adapt to resolution and duration
        tiling_config = compute_tiling_config(
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            spatial_tile_override=req.tiling_spatial_tile,
            temporal_tile_override=req.tiling_temporal_tile,
        )
        video_chunks_number = get_video_chunks_number(req.num_frames, tiling_config)

        # ── Install progress hook ────────────────────────────
        original_fn = self._install_progress_hook()

        try:
            self._emit(req.job_id, "stage", {
                "stage": "encoding_prompt",
                "message": "Encoding prompt with Gemma…",
            })
            gen_start = time.time()

            # NOTE: Use torch.no_grad() instead of torch.inference_mode().
            # The pipeline returns a lazy video iterator — the VAE decode
            # runs inside encode_video() below.  inference_mode() creates
            # special tensors that break when the VAE's conv3d layers try
            # to save them for backward.  no_grad() avoids that.
            with torch.no_grad():
                video, audio = self._pipeline(
                    prompt=req.prompt,
                    seed=req.seed,
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    frame_rate=req.frame_rate,
                    images=images,
                    tiling_config=tiling_config,
                    enhance_prompt=req.enhance_prompt,
                )

                gen_time = time.time() - gen_start

                # ── Encode output ────────────────────────────
                # Must stay inside no_grad() because the video iterator
                # triggers lazy VAE decoding when iterated.
                self._emit(req.job_id, "stage", {
                    "stage": "encoding_video",
                    "message": "Encoding output video…",
                })

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"ltx23_{timestamp}_{req.job_id[:8]}.mp4"
                output_path = str(OUTPUT_DIR / output_filename)

                encode_video(
                    video=video,
                    fps=req.frame_rate,
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
                )

            total_time = time.time() - overall_start

            result = {
                "job_id": req.job_id,
                "filename": output_filename,
                "video_url": f"/outputs/{output_filename}",
                "prompt": req.prompt,
                "height": req.height,
                "width": req.width,
                "num_frames": req.num_frames,
                "frame_rate": req.frame_rate,
                "duration": round(req.num_frames / req.frame_rate, 2),
                "seed": req.seed,
                "quantization": req.quantization,
                "enhance_prompt": req.enhance_prompt,
                "load_time": round(load_time, 1),
                "gen_time": round(gen_time, 1),
                "total_time": round(total_time, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_size_mb": round(
                    os.path.getsize(output_path) / (1024 * 1024), 2
                ),
            }

            self._save_to_history(result)
            self._emit(req.job_id, "complete", result)

        finally:
            self._restore_hook(original_fn)

    # ── history ──────────────────────────────────────────────

    def _save_to_history(self, entry: dict):
        history = self.load_history()
        history.insert(0, entry)
        history = history[:100]
        self._write_history(history)

    def load_history(self) -> list:
        if HISTORY_FILE.exists():
            try:
                return json.loads(HISTORY_FILE.read_text())
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _write_history(self, history: list):
        tmp = str(HISTORY_FILE) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(tmp, str(HISTORY_FILE))

    def delete_history_entry(self, job_id: str) -> bool:
        history = self.load_history()
        new_history = []
        deleted = False
        for entry in history:
            if entry.get("job_id") == job_id:
                vp = OUTPUT_DIR / entry.get("filename", "")
                if vp.exists():
                    vp.unlink()
                deleted = True
            else:
                new_history.append(entry)
        if deleted:
            self._write_history(new_history)
        return deleted


# ── Module-level singleton ───────────────────────────────────
pipeline_manager = PipelineManager()
