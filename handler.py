#!/usr/bin/env python3
"""
RunPod Serverless worker entrypoint.

Expected job ``input`` (JSON), for example::

    {"video_url": "https://example.com/clip.mp4"}

Optional fields: ``device`` (default ``cuda``), ``transparent_gif`` (bool),
``timeout_sec`` (int, default 3600).

Start the worker (set container command to)::

    python -u handler.py

Requires ``rvm_resnet50.pth`` in the app directory (bake into image or mount).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import runpod

APP_ROOT = Path(__file__).resolve().parent
CHECKPOINT = APP_ROOT / "rvm_resnet50.pth"
OUTPUT_ROOT = Path(os.environ.get("RVM_OUTPUTS_DIR", str(APP_ROOT / "api_outputs"))).resolve()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "formloop-runpod/1.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def handler(job: dict) -> dict:
    job_input = job.get("input") or {}
    video_url = job_input.get("video_url")
    if not video_url or not isinstance(video_url, str):
        return {"error": "Missing or invalid input.video_url (string)."}

    if not CHECKPOINT.is_file():
        return {
            "error": (
                f"Checkpoint not found: {CHECKPOINT}. "
                "Add rvm_resnet50.pth to the image or mount it at this path."
            )
        }

    job_id = str(job.get("id") or "anonymous")
    work = OUTPUT_ROOT / "runpod" / job_id
    work.mkdir(parents=True, exist_ok=True)
    src = work / "input.mp4"
    out_gif = work / "output.gif"

    try:
        _download(video_url, src)
    except Exception as e:
        return {"error": f"Download failed: {e}"}

    cmd = [
        os.environ.get("PYTHON", "python3"),
        str(APP_ROOT / "process_video.py"),
        "--input",
        str(src),
        "--gif",
        str(out_gif),
        "--device",
        str(job_input.get("device") or "cuda"),
    ]
    if job_input.get("transparent_gif"):
        cmd.append("--transparent-gif")

    timeout = int(job_input.get("timeout_sec") or 3600)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(APP_ROOT))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Job timed out after {timeout}s"}

    if proc.returncode != 0:
        return {
            "error": "process_video.py failed",
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-8000:],
            "stdout_tail": (proc.stdout or "")[-4000:],
        }

    if not out_gif.is_file():
        return {"error": "process_video.py exited 0 but output.gif was not created."}

    return {
        "output_gif": str(out_gif),
        "size_bytes": out_gif.stat().st_size,
    }


runpod.serverless.start({"handler": handler})
