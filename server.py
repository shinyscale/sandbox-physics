#!/usr/bin/env python3
"""
Flask server for bodypipe SAM2 — serves static files, handles video uploads,
kicks off SAM2 processing, and reports progress.

Usage:
    conda run -n gvhmr python sandbox-physics/server.py
"""

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, redirect

BASE_DIR = Path(__file__).resolve().parent.parent  # /mnt/f
VIDEOS_DIR = BASE_DIR / "research" / "videos"
CONDA_PREFIX = Path(os.environ.get(
    "CONDA_PREFIX",
    os.path.expanduser("~/miniconda3/envs/gvhmr"),
))
VENV_PYTHON = CONDA_PREFIX / "bin" / "python"
PROCESS_SCRIPT = Path(__file__).resolve().parent / "process_video_sam2.py"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

# ── Job state ──
# Only one job at a time (single GPU). Simple lock + shared state.
job_lock = threading.Lock()
current_job = {
    "name": None,
    "status": "idle",  # idle | extracting | running_sam2 | done | error
    "progress": 0.0,
    "error": None,
}


def sanitize_name(name):
    """Turn a filename into a safe directory name."""
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^\w\s-]", "", name.lower())
    name = re.sub(r"[\s_]+", "-", name).strip("-")
    # Dedupe if exists
    base = name
    suffix = 2
    while (VIDEOS_DIR / name).exists():
        name = f"{base}-{suffix}"
        suffix += 1
    return name


def process_video_thread(video_path, name, max_frames=0):
    """Run SAM2 processing in background, parsing stdout for progress."""
    global current_job

    status_file = VIDEOS_DIR / name / "_status.json"
    os.makedirs(VIDEOS_DIR / name, exist_ok=True)

    def update(status, progress=0.0, error=None):
        current_job["status"] = status
        current_job["progress"] = progress
        current_job["error"] = error
        with open(status_file, "w") as f:
            json.dump({"status": status, "progress": progress, "error": error}, f)

    try:
        update("extracting")

        cmd = [
            str(VENV_PYTHON), str(PROCESS_SCRIPT),
            str(video_path),
            "--name", name,
            "--max-frames", str(max_frames),
        ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            # Parse progress from "  frame 50/300"
            m = re.search(r"frame\s+(\d+)/(\d+)", line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                update("running_sam2", done / total)
                continue

            if "Running SAM2" in line or "Propagating" in line:
                update("running_sam2", 0.0)
            elif "Extracting frames" in line:
                update("extracting")
            elif "=== Done ===" in line:
                update("done", 1.0)

        proc.wait()

        if proc.returncode != 0:
            update("error", error=f"Process exited with code {proc.returncode}")
        else:
            update("done", 1.0)
            # Clean up source video to save disk
            try:
                os.remove(video_path)
            except OSError:
                pass

    except Exception as e:
        update("error", error=str(e))
    finally:
        with job_lock:
            current_job["name"] = None
            current_job["status"] = "idle"


# ── Routes ──

@app.route("/")
def index():
    return redirect("/sandbox-physics/sandpipe-body-webgpu-v2.html")


@app.route("/api/videos")
def list_videos():
    """Return list of all videos with metadata."""
    videos = []
    if not VIDEOS_DIR.exists():
        return jsonify(videos)

    for d in sorted(VIDEOS_DIR.iterdir()):
        if not d.is_dir():
            continue

        entry = {"name": d.name}

        # Check for config (means processing completed)
        config_path = d / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                entry.update(cfg)
                entry["status"] = "ready"
                entry["thumbnail"] = f"/research/videos/{d.name}/frames/00000.jpg"
            except (json.JSONDecodeError, KeyError):
                entry["status"] = "error"
        else:
            # Check for in-progress status
            status_path = d / "_status.json"
            if status_path.exists():
                try:
                    with open(status_path) as f:
                        st = json.load(f)
                    entry["status"] = st.get("status", "unknown")
                    entry["progress"] = st.get("progress", 0)
                    entry["error"] = st.get("error")
                except (json.JSONDecodeError, KeyError):
                    entry["status"] = "unknown"
            elif (d / "frames").exists():
                entry["status"] = "incomplete"
            else:
                continue  # empty dir, skip

        videos.append(entry)

    return jsonify(videos)


@app.route("/api/videos/<name>/status")
def video_status(name):
    """Return processing status for a specific video."""
    # Check live job first
    if current_job["name"] == name:
        return jsonify({
            "status": current_job["status"],
            "progress": current_job["progress"],
            "error": current_job["error"],
        })

    # Check status file
    status_path = VIDEOS_DIR / name / "_status.json"
    if status_path.exists():
        with open(status_path) as f:
            return jsonify(json.load(f))

    # Check if it's already done
    config_path = VIDEOS_DIR / name / "config.json"
    if config_path.exists():
        return jsonify({"status": "done", "progress": 1.0})

    return jsonify({"status": "unknown"}), 404


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Accept a video upload and start processing."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Check if a job is already running
    with job_lock:
        if current_job["status"] not in ("idle", "done", "error"):
            return jsonify({
                "error": f"Already processing '{current_job['name']}'. Try again later.",
                "busy": True,
            }), 409

    custom_name = request.form.get("name", "").strip()
    name = sanitize_name(custom_name or file.filename)

    # Save uploaded file
    video_dir = VIDEOS_DIR / name
    os.makedirs(video_dir, exist_ok=True)
    video_path = video_dir / "source.mp4"
    file.save(str(video_path))

    max_frames = int(request.form.get("max_frames", 0))

    # Start processing
    with job_lock:
        current_job["name"] = name
        current_job["status"] = "queued"
        current_job["progress"] = 0.0
        current_job["error"] = None

    thread = threading.Thread(
        target=process_video_thread,
        args=(str(video_path), name, max_frames),
        daemon=True,
    )
    thread.start()

    return jsonify({"name": name, "status": "queued"})


@app.route("/api/reprocess/<name>", methods=["POST"])
def reprocess_video(name):
    """Re-run SAM2 on an existing video that has frames but no/bad masks."""
    video_dir = VIDEOS_DIR / name
    if not video_dir.exists():
        return jsonify({"error": "Video not found"}), 404

    with job_lock:
        if current_job["status"] not in ("idle", "done", "error"):
            return jsonify({"error": "Already processing another video"}), 409

    # Look for source video or just re-run on existing frames
    source = video_dir / "source.mp4"
    if not source.exists():
        return jsonify({"error": "No source video found to reprocess"}), 400

    max_frames = int(request.form.get("max_frames", 0))

    with job_lock:
        current_job["name"] = name
        current_job["status"] = "queued"
        current_job["progress"] = 0.0
        current_job["error"] = None

    thread = threading.Thread(
        target=process_video_thread,
        args=(str(source), name, max_frames),
        daemon=True,
    )
    thread.start()

    return jsonify({"name": name, "status": "queued"})


@app.route("/api/videos/<name>/physics", methods=["POST"])
def save_physics(name):
    """Save sandpipe physics extraction JSON alongside the video data."""
    video_dir = VIDEOS_DIR / name
    if not video_dir.exists():
        return jsonify({"error": "Video not found"}), 404
    data = request.get_json(force=True)
    out_path = video_dir / "sandpipe-physics.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    n = len(data.get("frames", []))
    print(f"Saved sandpipe physics: {out_path} ({n} frames)")
    return jsonify({"ok": True, "path": str(out_path), "frames": n})


# ── Static file serving (replaces python -m http.server) ──

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(str(BASE_DIR), path)


if __name__ == "__main__":
    print(f"Serving from {BASE_DIR}")
    print(f"Videos dir: {VIDEOS_DIR}")
    print(f"Open: http://localhost:8887/sandbox-physics/sandpipe-body-webgpu-v2.html")
    app.run(host="0.0.0.0", port=8887, debug=False, threaded=True)
