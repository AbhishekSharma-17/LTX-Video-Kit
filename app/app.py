#!/usr/bin/env python3
"""
LTX-2.3 Video Studio — Flask Web Application

Routes:
  GET  /                        Serve UI
  POST /api/generate            Submit generation job → { job_id }
  GET  /api/status/<job_id>     SSE stream (progress/stage/complete/error)
  GET  /api/gallery             Return history JSON
  DELETE /api/gallery/<job_id>  Delete entry + video file
  POST /api/upload              Upload image for image-to-video
  GET  /api/pipeline/status     Pipeline state + GPU info
  POST /api/estimate            Estimate VRAM + generation time
  GET  /outputs/<filename>      Serve generated videos
"""

import os
import json
import uuid
import queue
import time

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from pipeline_worker import (
    BASE_DIR,
    OUTPUT_DIR,
    UPLOAD_DIR,
    JobRequest,
    pipeline_manager,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


def _allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


# ── Pages ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API: Generate ────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    height = int(data.get("height", 1024))
    width = int(data.get("width", 1536))
    num_frames = int(data.get("num_frames", 121))
    frame_rate = float(data.get("frame_rate", 24.0))
    seed = int(data.get("seed", 42))
    quantization = data.get("quantization", None)
    enhance_prompt = bool(data.get("enhance_prompt", False))
    images = data.get("images", [])

    # Optional tiling overrides from advanced settings
    tiling_spatial_tile = int(data.get("tiling_spatial_tile", 0))
    tiling_temporal_tile = int(data.get("tiling_temporal_tile", 0))

    # Validation
    if height % 64 != 0:
        return jsonify({"error": f"Height must be divisible by 64, got {height}"}), 400
    if width % 64 != 0:
        return jsonify({"error": f"Width must be divisible by 64, got {width}"}), 400
    if (num_frames - 1) % 8 != 0:
        return jsonify({
            "error": f"num_frames must be 8n+1 (e.g. 9,17,25,33,…,121), got {num_frames}"
        }), 400

    # Long video constraint: >10s only at ≤1080p
    duration_s = num_frames / frame_rate
    if duration_s > 10.5 and (height > 1088 or width > 1920):
        return jsonify({
            "error": (
                f"Videos longer than 10s ({duration_s:.1f}s) are only supported at "
                f"1080p (1088×1920) or lower. Please reduce resolution or duration."
            )
        }), 400

    if pipeline_manager.active_job:
        return jsonify({"error": "A generation is already in progress"}), 429

    job_id = uuid.uuid4().hex[:16]
    req = JobRequest(
        job_id=job_id,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        seed=seed,
        quantization=quantization,
        enhance_prompt=enhance_prompt,
        images=images,
        tiling_spatial_tile=tiling_spatial_tile,
        tiling_temporal_tile=tiling_temporal_tile,
    )
    pipeline_manager.submit_job(req)
    return jsonify({"job_id": job_id})


# ── API: Estimate ────────────────────────────────────────────

@app.route("/api/estimate", methods=["POST"])
def api_estimate():
    """Estimate VRAM usage and generation time for given parameters."""
    data = request.get_json(force=True)
    height = int(data.get("height", 1024))
    width = int(data.get("width", 1536))
    num_frames = int(data.get("num_frames", 121))
    quantization = data.get("quantization", None)

    result = pipeline_manager.estimate_generation(height, width, num_frames, quantization)
    return jsonify(result)


# ── API: SSE Status Stream ──────────────────────────────────

@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    def event_stream():
        eq = pipeline_manager.get_event_queue(job_id)
        if eq is None:
            yield f"event: error\ndata: {json.dumps({'message': 'Unknown job'})}\n\n"
            return

        while True:
            try:
                evt = eq.get(timeout=30)
            except queue.Empty:
                # Send keepalive
                yield ": keepalive\n\n"
                continue

            payload = json.dumps(evt.data)
            yield f"event: {evt.event}\ndata: {payload}\n\n"

            if evt.event in ("complete", "error"):
                # Give client a moment then clean up
                time.sleep(0.5)
                pipeline_manager.cleanup_job(job_id)
                break

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── API: Gallery ─────────────────────────────────────────────

@app.route("/api/gallery")
def api_gallery():
    return jsonify(pipeline_manager.load_history())


@app.route("/api/gallery/<job_id>", methods=["DELETE"])
def api_gallery_delete(job_id: str):
    ok = pipeline_manager.delete_history_entry(job_id)
    if ok:
        return jsonify({"deleted": True})
    return jsonify({"error": "Not found"}), 404


# ── API: Upload Image ────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not _allowed_image(f.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_IMAGE_EXTENSIONS}"}), 400

    os.makedirs(str(UPLOAD_DIR), exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename)}"
    save_path = str(UPLOAD_DIR / safe_name)
    f.save(save_path)
    return jsonify({"path": save_path, "filename": safe_name})


# ── API: Pipeline Status ────────────────────────────────────

@app.route("/api/pipeline/status")
def api_pipeline_status():
    return jsonify({
        "loaded": pipeline_manager.is_loaded,
        "loading": pipeline_manager.is_loading,
        "quantization": pipeline_manager.current_quantization,
        "active_job": pipeline_manager.active_job,
        "gpu": pipeline_manager.gpu_info(),
    })


# ── Static: Serve videos ────────────────────────────────────

@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(str(OUTPUT_DIR), filename)


# ── Main ─────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    os.makedirs(str(UPLOAD_DIR), exist_ok=True)
    print("\n" + "=" * 60)
    print("  LTX-2.3 Video Studio  (Enhanced: 4K + 20s + Compile)")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
