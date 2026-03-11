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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

    # ── event emitter ────────────────────────────────────────

    def _emit(self, job_id: str, event: str, data: dict | None = None):
        eq = self.get_event_queue(job_id)
        if eq:
            eq.put(JobEvent(event=event, data=data or {}))

    # ── worker loop ──────────────────────────────────────────

    def _worker_loop(self):
        while True:
            req = self._job_queue.get()
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

        import torch
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        self._loading = True

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
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
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

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(req.num_frames, tiling_config)

        # ── Install progress hook ────────────────────────────
        original_fn = self._install_progress_hook()

        try:
            self._emit(req.job_id, "stage", {
                "stage": "encoding_prompt",
                "message": "Encoding prompt with Gemma…",
            })
            gen_start = time.time()

            with torch.inference_mode():
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

            # ── Encode output ────────────────────────────────
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
