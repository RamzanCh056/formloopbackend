#!/usr/bin/env python3
"""
HTTP API for RobustVideoMatting (demo): upload MP4 → JSON with **URLs** to outputs.

Run (from this directory):
  ../venv/bin/python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8765

- **Demo page:** GET /  or  GET /demo  — human-readable spec + try-it form.
- **Process:** POST /api/v1/matte  (multipart field ``file``) → JSON with download links.

**Note:** H.264 MP4 cannot store alpha. Use ``matte_transparent.webm`` (VP9+alpha) or composite
``foreground.mp4`` + ``alpha.mp4`` on the client.

**model=pro** runs ``process_video_pro.py`` (BiRefNet + YOLOv8x); requires the same Python deps as
that script (``transformers``, ``einops``, ``kornia``, ``timm``, etc.). No ``rvm_resnet50.pth`` needed for pro.

Optional env ``RVM_PUBLIC_BASE_URL`` (e.g. https://abc.ngrok.io) — used in JSON links when your
server is behind a tunnel or reverse proxy. Otherwise links use the request Host.

``RVM_PRO_FAST_MODE`` (default ``1``): BiRefNet-only ``process_video_pro`` path for speed (~2–3 min SLA).
Set ``0`` to enable full YOLO pose/segmentation again (higher quality, slower). ``RVM_PRO_NO_YOLO=1`` always skips YOLO.

``RVM_USE_RUNPOD`` (default ``1`` when unset): set ``0`` for local processing even if ``RUNPOD_*`` keys exist (dev/testing).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

APP_ROOT = Path(__file__).resolve().parent

# macOS + Homebrew Python can crash on fork/exec from multithreaded servers.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# Resolve ffmpeg/ffprobe at import time — probe multiple Nix/system paths so
# Railway (Nix package manager, not apt) finds the binaries reliably.
import shutil as _shutil

def _find_binary(name: str) -> str:
    import os
    # static-ffmpeg is pip-installed and works on any build system (Railway Railpack, etc.)
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except ImportError:
        pass
    p = _shutil.which(name)
    if p:
        return p
    for path in [
        f"/usr/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/nix/var/nix/profiles/default/bin/{name}",
        f"/run/current-system/sw/bin/{name}",
    ]:
        if os.path.isfile(path):
            return path
    return name

_FFMPEG  = _find_binary("ffmpeg")
_FFPROBE = _find_binary("ffprobe")


def _probe_video_fps(data: bytes, suffix: str = ".mp4") -> int:
    """Detect source video FPS via ffprobe. Caps at 24. Fallback: 12."""
    if not suffix.startswith("."):
        suffix = "." + suffix
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(data)
            tmp = Path(tf.name)
        try:
            result = subprocess.run(
                [_FFPROBE, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(tmp)],
                capture_output=True, text=True, timeout=15,
            )
            fps_str = result.stdout.strip().split("\n")[0].strip()
            if "/" in fps_str:
                num, den = fps_str.split("/", 1)
                fps = float(num) / max(float(den), 1e-6)
            else:
                fps = float(fps_str)
            return max(1, min(24, round(fps)))
        finally:
            tmp.unlink(missing_ok=True)
    except Exception:
        return 12


def _apply_video_rotation(src_path: Path, rotation: int) -> Path:
    """Return path to a rotated copy (new file) if rotation != 0, else src_path unchanged."""
    rotation = int(rotation or 0)
    if rotation not in (90, 180, 270):
        return src_path
    if rotation == 90:
        vf = "transpose=1"
    elif rotation == 180:
        vf = "vflip,hflip"
    else:  # 270
        vf = "transpose=2"
    out_path = src_path.with_name(src_path.stem + f"_rot{rotation}{src_path.suffix}")
    subprocess.run(
        [_FFMPEG, "-y", "-i", str(src_path), "-vf", vf,
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
         "-c:a", "copy", str(out_path)],
        check=True, capture_output=True,
    )
    return out_path


def _rotate_export_file(path: Path, rotation: int) -> None:
    """Rotate an already-rendered export (GIF/WebM) in place via ffmpeg transpose."""
    rotation = int(rotation or 0)
    if rotation not in (90, 180, 270) or not path.is_file():
        return
    if rotation == 90:
        transpose = "transpose=1"
    elif rotation == 180:
        transpose = "transpose=1,transpose=1"
    else:  # 270
        transpose = "transpose=2"

    suffix = path.suffix.lower()
    tmp_out = path.with_name(path.stem + f"_rot{rotation}_tmp{path.suffix}")

    if suffix == ".gif":
        # The default GIF encoder rebuilds the palette without reserving a
        # transparent index, so a plain transpose silently turns alpha into
        # white. Rebuild the palette explicitly with a reserved transparent
        # entry so rotated GIFs keep their transparency.
        vf = (
            f"{transpose},split[s0][s1];"
            "[s0]palettegen=reserve_transparent=1:transparency_color=ffffff[p];"
            "[s1][p]paletteuse=alpha_threshold=128"
        )
        cmd = [_FFMPEG, "-y", "-i", str(path), "-vf", vf, str(tmp_out)]
    else:
        cmd = [_FFMPEG, "-y", "-i", str(path), "-vf", transpose]
        if suffix == ".webm":
            cmd += ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p"]
        cmd += [str(tmp_out)]

    subprocess.run(cmd, check=True, capture_output=True)
    tmp_out.replace(path)


def _apply_reverse_loop(gif_path: Path) -> bool:
    """Append reversed frames to gif_path in-place (boomerang / seamless loop).

    Single-pass: split → reverse → concat, so no intermediate GIF encoding that
    would lose transparency before the palette is rebuilt.
    """
    if not gif_path.is_file():
        return False
    tmp_out = gif_path.with_name(gif_path.stem + "_revout.gif")
    try:
        # Read the GIF once, split into two streams, reverse the second,
        # concat forward+reverse, then rebuild a transparent palette.
        fc = (
            "[0:v]split[fwd][src];"
            "[src]reverse[rev];"
            "[fwd][rev]concat=n=2:v=1:a=0[cv];"
            "[cv]split[s0][s1];"
            "[s0]palettegen=reserve_transparent=1:transparency_color=000000[p];"
            "[s1][p]paletteuse=dither=none:diff_mode=rectangle"
        )
        r = subprocess.run(
            [_FFMPEG, "-y", "-i", str(gif_path),
             "-filter_complex", fc, "-loop", "0", str(tmp_out)],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not tmp_out.is_file():
            return False
        tmp_out.replace(gif_path)
        return True
    except Exception:
        return False
    finally:
        try:
            tmp_out.unlink(missing_ok=True)
        except OSError:
            pass


def _load_env_file(path: Path) -> None:
    """Populate os.environ from a .env file (first '=' splits key / value)."""
    if not path.is_file():
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip().strip("'").strip('"')
        os.environ[key] = val


try:
    from dotenv import load_dotenv

    load_dotenv(APP_ROOT / ".env")
except ImportError:
    _load_env_file(APP_ROOT / ".env")

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from billing_plans import exports_watermark_for_tier, gif_limit_for_tier
from gif_watermark import apply_png_watermark_to_gif, resolve_watermark_png
from output_job_store import (
    increment_quota_usage,
    mark_job_saved,
    read_job_owner,
    read_quota_usage,
    write_job_owner,
)
from user_billing import billing_period_key_for_uid, effective_plan_tier


def _uid_from_bearer(request: Request) -> str | None:
    """Extract Firebase uid from Authorization: Bearer <token> (for mobile clients)."""
    auth = (request.headers.get("authorization") or "").strip()
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    if not token:
        return None
    try:
        from firebase_auth import verify_firebase_id_token
        decoded = verify_firebase_id_token(token)
        if isinstance(decoded, dict):
            uid = (decoded.get("uid") or decoded.get("user_id") or "").strip()
            return uid or None
    except Exception:
        pass
    return None
from process_video import _assert_input_has_video, _pick_device, _python_for_inference
from web_ui_routes import SESSION_SECRET, router as ui_router

_log = logging.getLogger(__name__)
PROCESS_SCRIPT = APP_ROOT / "process_video.py"
PROCESS_REMBG_SCRIPT = APP_ROOT / "process_video_rembg.py"
PROCESS_PRO_SCRIPT = APP_ROOT / "process_video_pro.py"
CHECKPOINT = APP_ROOT / "rvm_resnet50.pth"
OUTPUTS_DIR = Path(os.environ.get("RVM_OUTPUTS_DIR", str(APP_ROOT / "api_outputs"))).resolve()

MAX_UPLOAD_BYTES = int(os.environ.get("RVM_MAX_UPLOAD_MB", "300")) * 1024 * 1024
PROCESS_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("RVM_MAX_CONCURRENT_JOBS", "1")))
OUTPUT_RETENTION_SEC = int(os.environ.get("RVM_OUTPUT_RETENTION_SEC", str(86400)))
RUNPOD_STATUS_POLL_SEC = float(os.environ.get("RUNPOD_STATUS_POLL_SEC", "2.0"))
# RunPod "pro" jobs can exceed 15m on busy queues / long clips.
RUNPOD_JOB_TIMEOUT_SEC = int(os.environ.get("RUNPOD_JOB_TIMEOUT_SEC", "1800"))
RUNPOD_MAX_RAW_VIDEO_BYTES = int(os.environ.get("RUNPOD_MAX_RAW_VIDEO_BYTES", "7200000"))
# Cost/runtime guardrails for RunPod serverless path.
RUNPOD_FORCE_FAST_MODE = (os.environ.get("RUNPOD_FORCE_FAST_MODE", "0").strip().lower() not in {"0", "false", "no"})
RUNPOD_MAX_GIF_WIDTH = max(320, min(1280, int(os.environ.get("RUNPOD_MAX_GIF_WIDTH", "960"))))
RUNPOD_MAX_GIF_FPS = max(1, min(30, int(os.environ.get("RUNPOD_MAX_GIF_FPS", "10"))))
RUNPOD_ALLOW_LOCAL_FALLBACK = (
    os.environ.get("RUNPOD_ALLOW_LOCAL_FALLBACK", "0").strip().lower() not in {"0", "false", "no"}
)


def _pro_skip_yolo() -> bool:
    """True = fast BiRefNet-only (no YOLO). False = full pipeline with YOLO."""
    if (os.environ.get("RVM_PRO_NO_YOLO") or "").strip().lower() in {"1", "true", "yes"}:
        return True
    fast = (os.environ.get("RVM_PRO_FAST_MODE", "1") or "1").strip().lower()
    return fast not in {"0", "false", "no"}

# Only these names can be fetched under /api/v1/matte/files/{job_id}/...
ALLOWED_RESULT_FILES = frozenset(
    {
        "foreground.mp4",
        "alpha.mp4",
        "matte.gif",
        "matte.webp",
        "matte.apng",
        "matte_transparent.webm",
        "preview_white_backdrop.mp4",
    }
)
ALLOWED_REMOTE_DOWNLOAD_HOSTS = frozenset(
    {
        "firebasestorage.googleapis.com",
        "storage.googleapis.com",
    }
)

_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_EXPORT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,120}$")
SAVE_FLOW_VERSION = "firebase-first-ca154cd-plus"
_STORAGE_URLS_FILE = ".storage_urls.json"
_TQDM_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
_RVM_PROGRESS_RE = re.compile(r"RVM_PROGRESS:(\d+)/(\d+):(\d{1,3})")
_JOB_STATE_LOCK = threading.Lock()
_JOB_STATES: dict[str, dict] = {}


def _quota_enforced() -> bool:
    return True


def _public_base_url(request: Request) -> str:
    fixed = os.environ.get("RVM_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if fixed:
        return fixed
    return str(request.base_url).rstrip("/")


def _file_url(request: Request, job_id: str, name: str) -> str:
    base = _public_base_url(request)
    return f"{base}/api/v1/matte/files/{job_id}/{name}"


def _prune_old_outputs() -> None:
    if not OUTPUTS_DIR.is_dir():
        return
    now = time.time()
    for p in OUTPUTS_DIR.iterdir():
        if not p.is_dir():
            continue
        try:
            if now - p.stat().st_mtime > OUTPUT_RETENTION_SEC:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def _mux_webm_alpha(fg_mp4: Path, alpha_mp4: Path, out_webm: Path) -> None:
    fc = (
        "[0:v]format=rgb24[rgb];"
        "[1:v]format=gray,extractplanes=y[am];"
        "[rgb][am]alphamerge,format=yuva420p[v]"
    )
    cmd = [
        _FFMPEG,
        "-y",
        "-i",
        str(fg_mp4),
        "-i",
        str(alpha_mp4),
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-auto-alt-ref",
        "0",
        "-crf",
        "32",
        "-b:v",
        "0",
        "-deadline",
        "good",
        "-cpu-used",
        "4",
        str(out_webm),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _run_matte_job(
    src_video: Path,
    work: Path,
    *,
    device: str,
    gif_width: int,
    gif_fps: int,
    transparent_formats: str,
    no_premium: bool = False,
) -> None:
    """Runs process_video; writes full stdout/stderr to work/_process_video.log (avoids exit 120 pipe/TQDM issues)."""
    fg = work / "foreground.mp4"
    alpha = work / "alpha.mp4"
    temp_preview = work / "preview_white_backdrop.mp4"
    gif_out = work / "matte.gif"

    py = _python_for_inference(APP_ROOT.parent)
    cmd = [
        str(py),
        str(PROCESS_SCRIPT),
        "--input",
        str(src_video),
        "--fg",
        str(fg),
        "--alpha",
        str(alpha),
        "--temp",
        str(temp_preview),
        "--gif",
        str(gif_out),
        "--background",
        "white",
        "--gif-width",
        str(gif_width),
        "--gif-fps",
        str(max(1, min(30, int(gif_fps)))),
        "--device",
        device,
    ]
    if no_premium:
        cmd.append("--no-premium")
    else:
        cmd.append("--gear-alpha-boost")

    log_path = work / "_process_video.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_DISABLE"] = "1"

    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            check=True,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )


def _run_rembg_job(
    src_video: Path,
    work: Path,
    *,
    gif_width: int,
) -> None:
    """Runs process_video_rembg for better held-object retention (dumbbells, bars)."""
    fg = work / "foreground.mp4"
    alpha = work / "alpha.mp4"
    temp_preview = work / "preview_white_backdrop.mp4"
    gif_out = work / "matte.gif"
    py = _python_for_inference(APP_ROOT.parent)
    cmd = [
        str(py),
        str(PROCESS_REMBG_SCRIPT),
        "--input",
        str(src_video),
        "--fg",
        str(fg),
        "--alpha",
        str(alpha),
        "--temp",
        str(temp_preview),
        "--gif",
        str(gif_out),
        "--gif-width",
        str(gif_width),
        "--workers",
        str(max(1, int(os.environ.get("RVM_REMBG_WORKERS", "2")))),
        "--max-side",
        str(max(0, int(os.environ.get("RVM_REMBG_MAX_SIDE", "0")))),
    ]
    if os.environ.get("RVM_REMBG_HQ", "1").strip() not in {"0", "false", "False"}:
        cmd.append("--hq")
    tmr = max(0, int(os.environ.get("RVM_REMBG_TEMPORAL_MAX_RADIUS", "1")))
    if tmr > 0:
        cmd += ["--temporal-max-radius", str(tmr)]
    if os.environ.get("RVM_REMBG_NO_DEFRINGE", "1").strip() not in {"0", "false", "False"}:
        cmd.append("--no-defringe")
    if os.environ.get("RVM_REMBG_NO_GYM_REFINE", "0").strip() in {"1", "true", "True"}:
        cmd.append("--no-gym-refine")
    log_path = work / "_process_video.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            check=True,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )


def _run_pro_job(
    src_video: Path,
    work: Path,
    *,
    gif_width: int,
    gif_fps: int = 0,
    gif_white_bg: bool = False,
) -> None:
    """Runs process_video_pro (BiRefNet; optional YOLO via ``RVM_PRO_FAST_MODE``)."""
    fg = work / "foreground.mp4"
    alpha = work / "alpha.mp4"
    gif_out = work / "matte.gif"
    py = _python_for_inference(APP_ROOT.parent)
    effective_fps = gif_fps if gif_fps > 0 else max(1, int(os.environ.get("RVM_PRO_GIF_FPS", "15")))
    print(f"[GIF FPS] _run_pro_job: received gif_fps={gif_fps} → effective_fps={effective_fps} → --gif-fps={max(1, min(24, effective_fps))}", flush=True)
    cmd = [
        str(py),
        str(PROCESS_PRO_SCRIPT),
        "--input",
        str(src_video),
        "--gif",
        str(gif_out),
        "--fg",
        str(fg),
        "--alpha",
        str(alpha),
        "--gif-width",
        str(gif_width),
        "--gif-fps",
        str(max(1, min(24, effective_fps))),
        "--device",
        os.environ.get("RVM_DEVICE", _pick_device("auto")),
        "--dilation",
        str(max(0, int(os.environ.get("RVM_PRO_DILATION", "12")))),
        "--conf",
        str(float(os.environ.get("RVM_PRO_CONF", "0.20"))),
        "--rvm-downsample",
        str(float(os.environ.get("RVM_PRO_RVM_DOWNSAMPLE", "0.4"))),
    ]
    if os.environ.get("RVM_PRO_NO_RVM", "").strip().lower() in {"1", "true", "yes"}:
        cmd.append("--no-rvm")
    if _pro_skip_yolo():
        cmd.append("--no-yolo")
    log_path = work / "_process_video.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Never pause at frame 100 in API mode.
    env["RVM_PRO_HALT_AFTER_FRAME"] = "0"
    env["RVM_PRO_GIF_WHITE_BG"] = "1" if gif_white_bg else "0"
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            check=True,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )


def _runpod_base_url() -> str:
    endpoint_url = (os.environ.get("RUNPOD_ENDPOINT_URL") or "").strip().rstrip("/")
    if endpoint_url:
        return endpoint_url
    endpoint_id = (os.environ.get("RUNPOD_ENDPOINT_ID") or "").strip()
    if not endpoint_id:
        return ""
    return f"https://api.runpod.ai/v2/{endpoint_id}"


def _runpod_enabled() -> bool:
    use = (os.environ.get("RVM_USE_RUNPOD") or "1").strip().lower()
    if use in {"0", "false", "no"}:
        return False
    return bool((os.environ.get("RUNPOD_API_KEY") or "").strip() and _runpod_base_url())


def _runpod_only_mode() -> bool:
    """Production switch: require RunPod path and disable local processing path."""
    v = (os.environ.get("RVM_RUNPOD_ONLY") or "1").strip().lower()
    return v not in {"0", "false", "no"}


def _runpod_headers() -> dict[str, str]:
    api_key = (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is not configured")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _runpod_submit(
    raw: bytes,
    *,
    job_id: str,
    filename: str | None,
    gif_width: int,
    gif_fps: int,
    public_base: str,
    pro_fast_mode: bool | None = None,
    gif_white_bg: bool = False,
    start_time: float | None = None,
    end_time: float | None = None,
    rotation: int = 0,
    loop_style: str = "normal",
    use_sam2: bool = False,
) -> str:
    base = _runpod_base_url()
    if not base:
        raise RuntimeError("RUNPOD_ENDPOINT_ID or RUNPOD_ENDPOINT_URL is not configured")
    exercise_name = Path(filename or f"clip_{uuid.uuid4().hex[:8]}.mp4").stem[:120]
    suffix = Path(filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        suffix = ".mp4"
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    src_name = f"{exercise_name}_{job_id[:8]}{suffix}"
    src_path = job_dir / src_name
    src_path.write_bytes(raw)
    _original_size = len(raw)
    if (start_time and float(start_time) > 0) or end_time is not None:
        trimmed_path = str(src_path).replace(suffix, f"_trimmed{suffix}")
        _probe = subprocess.run(
            [_FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src_path)],
            capture_output=True, text=True,
        )
        duration = _probe.stdout.strip() or "?"
        print(f"[Trim] Trimming video: {start_time}s to {end_time}s, duration={duration}s", flush=True)
        st = float(start_time) if start_time else 0.0
        et = float(end_time) if end_time is not None else None
        trim_cmd = [_FFMPEG, "-y", "-ss", str(st), "-i", str(src_path)]
        if et is not None:
            trim_cmd += ["-t", str(et - st)]
        trim_cmd += ["-c", "copy", trimmed_path]
        subprocess.run(trim_cmd, check=True, capture_output=True)
        src_path = Path(trimmed_path)
        with open(src_path, "rb") as f:
            raw = f.read()
        print(f"[Trim] Video trimmed: {start_time}s to {end_time}s", flush=True)
        trimmed_size = os.path.getsize(src_path)
        print(f"[Trim] Original: {_original_size/1024:.0f}KB → Trimmed: {trimmed_size/1024:.0f}KB", flush=True)
    if rotation:
        try:
            rotated = _apply_video_rotation(src_path, rotation)
            if rotated != src_path:
                src_path = rotated
                with open(src_path, "rb") as f:
                    raw = f.read()
                print(f"[Rotation] Applied {rotation}° rotation", flush=True)
        except Exception as exc:
            _log.warning("Rotation failed (rotation=%s): %s — continuing without rotation", rotation, exc)
    try:
        from firebase_storage_admin import upload_runpod_input_video

        input_url = upload_runpod_input_video(job_id=job_id, filename=src_name, local_path=src_path)
    except Exception as exc:
        raise RuntimeError(
            "RunPod input upload failed. Configure Firebase Storage so backend can create a public input URL. "
            f"Reason: {exc}"
        ) from exc
    fast = _pro_skip_yolo() if pro_fast_mode is None else bool(pro_fast_mode)
    payload = {
        "input": {
            "video_url": input_url,
            "exercise_name": exercise_name,
            "gif_width": int(gif_width),
            "gif_fps": int(max(1, min(30, gif_fps))),
            "pro_fast_mode": fast,
            "gif_white_bg": bool(gif_white_bg),
            "start_time": float(start_time) if start_time else 0,
            "end_time": float(end_time) if end_time else None,
            "loop_style": loop_style,
            "use_sam2": bool(use_sam2),
        }
    }
    print(f"[RUNPOD] submitting: gif_fps={gif_fps} loop_style={loop_style} rotation={rotation}", flush=True)
    # Backward compatibility: older deployed RunPod handlers only read `input.video`.
    if len(raw) <= RUNPOD_MAX_RAW_VIDEO_BYTES:
        payload["input"]["video"] = base64.b64encode(raw).decode("utf-8")
    resp = requests.post(f"{base}/run", headers=_runpod_headers(), json=payload, timeout=120)
    if resp.status_code >= 400:
        detail = (resp.text or "").strip()
        if len(detail) > 500:
            detail = detail[:500] + "...[truncated]"
        raise RuntimeError(f"RunPod /run failed ({resp.status_code}): {detail or 'no response body'}")
    data = resp.json()
    run_id = str(data.get("id") or "").strip()
    if not run_id:
        raise RuntimeError("RunPod response missing job id")
    return run_id


def _runpod_extract_output_dict(payload: dict) -> dict:
    """RunPod status payload may nest the handler return under ``output`` as dict or JSON string."""
    out: dict = {}
    raw = payload.get("output")
    if isinstance(raw, dict):
        out.update(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            o = json.loads(raw)
            if isinstance(o, dict):
                out.update(o)
        except json.JSONDecodeError:
            pass
    for k in ("gif_url", "gif_b64", "webm_url"):
        v = payload.get(k)
        if v is not None and k not in out:
            out[k] = v
    return out


def _persist_runpod_job_artifacts(job_id: str, payload: dict) -> None:
    """Optional: mirror RunPod GIF/WebM into ``api_outputs`` for ``GET /api/v1/matte/files/...`` — library Save uses Firebase URLs directly."""
    d = _runpod_extract_output_dict(payload)
    if not d:
        return
    work = OUTPUTS_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)
    if isinstance(d.get("gif_b64"), str) and d["gif_b64"].strip():
        try:
            (work / "matte.gif").write_bytes(base64.b64decode(d["gif_b64"]))
        except Exception as exc:
            _log.warning("RunPod: could not decode/persist GIF b64 for job_id=%s: %s", job_id, exc)
    gu = d.get("gif_url") if isinstance(d.get("gif_url"), str) else None
    wu = d.get("webm_url") if isinstance(d.get("webm_url"), str) else None
    _ensure_job_assets_from_client_urls(work, gu, wu)


def _fetch_remote_asset_if_missing(url: str | None, dest: Path, *, min_size: int, timeout: int) -> bool:
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        return dest.is_file() and dest.stat().st_size >= min_size
    if dest.is_file() and dest.stat().st_size >= min_size:
        return True
    hdrs = {
        "User-Agent": "FormLoop-Server/1.0",
        "Accept": "*/*",
    }
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(u, timeout=timeout, headers=hdrs, allow_redirects=True)
            r.raise_for_status()
            data = r.content
            if len(data) >= min_size:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                _log.info("Fetched remote asset %s -> %s (%s bytes)", u[:80], dest.name, len(data))
                return True
            _log.warning("Remote fetch too small (%s bytes): %s", len(data), u[:96])
            break
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))
            else:
                _log.warning("Remote fetch failed after 3 tries %s -> %s: %s", u[:96], dest.name, exc)
    return dest.is_file() and dest.stat().st_size >= min_size


def _ensure_job_assets_from_client_urls(job_dir: Path, gif_url: str | None, webm_url: str | None) -> None:
    """Browser passes RunPod/Firebase URLs when ``matte.gif`` was never written under ``api_outputs``."""
    _fetch_remote_asset_if_missing(gif_url, job_dir / "matte.gif", min_size=64, timeout=180)
    _fetch_remote_asset_if_missing(webm_url, job_dir / "matte_transparent.webm", min_size=256, timeout=300)


def _runpod_cancel(run_id: str) -> bool:
    """Cancel a RunPod job. Returns True if the cancel request was accepted."""
    try:
        base = _runpod_base_url()
        if not base or not run_id:
            return False
        resp = requests.post(f"{base}/cancel/{run_id}", headers=_runpod_headers(), timeout=30)
        return resp.status_code < 300
    except Exception as exc:
        print(f"[RUNPOD] cancel({run_id}) failed: {exc}", flush=True)
        return False


def _runpod_wait(run_id: str, max_wait_sec: float | None = None, job_id: str | None = None) -> dict:
    base = _runpod_base_url()
    if not base:
        raise RuntimeError("RUNPOD endpoint is not configured")
    started = time.time()
    limit_sec = float(RUNPOD_JOB_TIMEOUT_SEC if max_wait_sec is None else max(1.0, min(float(max_wait_sec), float(RUNPOD_JOB_TIMEOUT_SEC))))
    last: dict = {}
    run_started: float | None = None
    while True:
        if time.time() - started > limit_sec:
            raise TimeoutError("RunPod job timed out")
        # Check if the local job was cancelled by the user.
        if job_id:
            st = _get_job_state(job_id)
            if st and str(st.get("status") or "") == "cancelled":
                _runpod_cancel(run_id)
                return {"status": "CANCELLED", "id": run_id}
        resp = requests.get(f"{base}/status/{run_id}", headers=_runpod_headers(), timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        last = payload if isinstance(payload, dict) else {}
        status = str(last.get("status") or "").upper()
        if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return last
        if job_id:
            if status == "IN_QUEUE":
                _set_job_state(job_id, status="queued", progress=1, message="Waiting in queue...")
            elif status == "IN_PROGRESS":
                if run_started is None:
                    run_started = time.time()
                elapsed_run = time.time() - run_started
                if elapsed_run < 8:
                    progress, message = 10, "Initializing model..."
                else:
                    progress = min(90, 20 + int(elapsed_run))
                    message = "Almost done..." if progress >= 80 else "Processing..."
                _set_job_state(job_id, status="running", progress=progress, message=message)
        time.sleep(max(0.2, RUNPOD_STATUS_POLL_SEC))


def _runpod_output_to_body(job_id: str, payload: dict) -> dict:
    output = _runpod_extract_output_dict(payload)
    gif_url = output.get("gif_url")
    gif_b64 = output.get("gif_b64")
    webm_url = output.get("webm_url")
    body: dict = {
        "success": True,
        "job_id": job_id,
        "message": "Processed by RunPod.",
        "background_removed": {
            "transparent_video_webm": webm_url if isinstance(webm_url, str) else None,
            "foreground_mp4": None,
            "alpha_mp4": None,
            "preview_white_backdrop_mp4": None,
        },
        "gif_and_animation": {
            "transparent_gif": gif_url if isinstance(gif_url, str) else None,
            "transparent_webp": None,
            "transparent_apng": None,
        },
        "runpod": {
            "id": str(payload.get("id") or ""),
            "status": str(payload.get("status") or ""),
        },
    }
    if isinstance(gif_b64, str) and gif_b64:
        body["gif_base64"] = gif_b64
    return body


def _collect_matte_outputs(work: Path) -> dict[str, Path]:
    fg = work / "foreground.mp4"
    alpha = work / "alpha.mp4"
    gif_out = work / "matte.gif"
    out: dict[str, Path] = {
        "foreground": fg,
        "alpha": alpha,
        "gif": gif_out,
    }
    webm = work / "matte_transparent.webm"
    try:
        _mux_webm_alpha(fg, alpha, webm)
        if webm.is_file() and webm.stat().st_size > 0:
            out["webm"] = webm
    except (subprocess.CalledProcessError, OSError):
        pass

    for extra in (work / "matte.webp", work / "matte.apng"):
        if extra.is_file() and extra.stat().st_size > 0:
            key = "webp" if extra.suffix == ".webp" else "apng"
            out[key] = extra

    return out


def _ensure_webm_for_job(job_dir: Path) -> bool:
    webm = job_dir / "matte_transparent.webm"
    if webm.is_file() and webm.stat().st_size > 0:
        return True
    fg = job_dir / "foreground.mp4"
    alpha = job_dir / "alpha.mp4"
    if not fg.is_file() or not alpha.is_file():
        return False
    try:
        _mux_webm_alpha(fg, alpha, webm)
    except (subprocess.CalledProcessError, OSError):
        return False
    return webm.is_file() and webm.stat().st_size > 0


def _gif_frame_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(getattr(im, "n_frames", 1))
    except Exception:
        return 1


def _rebuild_transparent_gif_from_fg_alpha(work: Path, gif_width: int) -> bool:
    fg = work / "foreground.mp4"
    alpha = work / "alpha.mp4"
    gif_out = work / "matte.gif"
    if not fg.is_file() or not alpha.is_file():
        return False
    gw = max(320, min(1280, int(gif_width)))
    gf = max(8, int(os.environ.get("RVM_GIF_FPS", "12")))
    fc = (
        f"[0:v]format=rgb24[rgb];"
        f"[1:v]format=gray,extractplanes=y[am];"
        f"[rgb][am]alphamerge,format=rgba,"
        f"fps={gf},scale={gw}:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen=stats_mode=full:max_colors=255:reserve_transparent=1[p];"
        f"[s1][p]paletteuse=alpha_threshold=64:diff_mode=rectangle:dither=sierra2_4a"
    )
    cmd = [
        _FFMPEG,
        "-y",
        "-threads",
        "0",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(fg),
        "-i",
        str(alpha),
        "-filter_complex",
        fc,
        "-loop",
        "0",
        str(gif_out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError):
        return False
    return _gif_frame_count(gif_out) > 1


def _log_tail(work: Path, max_chars: int = 12000) -> str:
    log_path = work / "_process_video.log"
    if not log_path.is_file():
        return ""
    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    raw = raw.strip()
    if len(raw) <= max_chars:
        return raw
    return "...[truncated]\n" + raw[-max_chars:]


def _set_job_state(job_id: str, **fields) -> None:
    with _JOB_STATE_LOCK:
        st = _JOB_STATES.setdefault(job_id, {})
        st.update(fields)
        st["updated_at"] = time.time()


def _get_job_state(job_id: str) -> dict | None:
    with _JOB_STATE_LOCK:
        st = _JOB_STATES.get(job_id)
        return dict(st) if st else None


def _build_result_body(base_url: str, job_id: str, *, gif_width: int, effective_formats: str, premium: bool, model: str) -> dict:
    def u(name: str) -> str | None:
        p = OUTPUTS_DIR / job_id / name
        return f"{base_url}/api/v1/matte/files/{job_id}/{name}" if p.is_file() else None

    return {
        "success": True,
        "job_id": job_id,
        "message": "Background removal complete. Use the URLs below (valid until cleaned up; see retention).",
        "background_removed": {
            "transparent_video_webm": u("matte_transparent.webm"),
            "foreground_mp4": u("foreground.mp4"),
            "alpha_mp4": u("alpha.mp4"),
            "preview_white_backdrop_mp4": u("preview_white_backdrop.mp4"),
        },
        "gif_and_animation": {
            "transparent_gif": u("matte.gif"),
            "transparent_webp": u("matte.webp"),
            "transparent_apng": u("matte.apng"),
        },
        "phone_implementation": {
            "single_file_with_alpha": "Prefer transparent_video_webm in WebView/ExoPlayer where VP9 alpha is supported.",
            "if_webm_unavailable": "Download foreground_mp4 + alpha_mp4 and composite in your player or shader.",
            "gif": "matte.gif is 256-color; for smoother edges use transparent_webp when present.",
        },
        "query_params_used": {
            "gif_width": gif_width,
            "transparent_formats": effective_formats,
            "premium": premium,
            "model": model,
        },
    }


def _estimate_job_progress(job_id: str, *, job_status: str | None = None) -> tuple[int, str]:
    work = OUTPUTS_DIR / job_id
    if not work.exists():
        return 0, "Queued"
    st = (job_status or "").strip().lower()
    has_gif = (work / "matte.gif").is_file()
    has_webm = (work / "matte_transparent.webm").is_file()
    # GIF/WebM exist on disk before the API finishes (FFmpeg watermark on GIF can take many minutes).
    if has_gif or has_webm:
        if st == "completed":
            return 100, "Completed"
        if st in {"running", "queued"}:
            return 97, "Finishing export (watermark / packaging)"
    if (work / "foreground.mp4").is_file() and (work / "alpha.mp4").is_file():
        pct = 82
        msg = "Encoding transparent outputs"
    else:
        pct = 3
        msg = "Initializing model"

    log_path = work / "_process_video.log"
    if log_path.is_file():
        try:
            raw = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw = ""
        if raw:
            matches = _TQDM_PERCENT_RE.findall(raw)
            marker = _RVM_PROGRESS_RE.findall(raw)
            if marker:
                try:
                    infer_pct = max(0, min(100, int(marker[-1][2])))
                    # Inference phase mapped to early-majority of whole pipeline.
                    pct = max(pct, min(82, 5 + int(infer_pct * 0.77)))
                    msg = "Removing background"
                except ValueError:
                    pass
            if matches:
                try:
                    infer_pct = max(0, min(100, int(matches[-1])))
                    # Inference is the heavy portion of total pipeline.
                    pct = max(pct, min(80, 4 + int(infer_pct * 0.76)))
                    msg = "Removing background"
                except ValueError:
                    pass
            if "Foreground and alpha videos created" in raw:
                pct = max(pct, 85)
                msg = "Building cutout outputs"
            if "Solid backdrop video created" in raw:
                pct = max(pct, 92)
                msg = "Encoding GIF/WebM"
            if "GIF created" in raw or "Transparent animation" in raw:
                pct = max(pct, 98)
                msg = "Finalizing files"
            # process_video_pro.py: MP4s exist early; long stall is usually PIL GIF quantize
            if "[GIF] encoding" in raw and "[GIF] Saved" not in raw:
                pct = max(pct, 92)
                msg = "Encoding GIF"
            if "[GIF] quantize" in raw:
                pct = max(pct, 94)
                msg = "Encoding GIF"
    # No matte.gif on disk yet — do not show 98–99% (ffmpeg GIF can run 10–30+ min with high fps).
    if not has_gif and not has_webm:
        pct = min(pct, 94)
    return max(0, min(99, pct)), msg


def _process_to_job_dir(
    job_id: str,
    data: bytes,
    filename: str | None,
    *,
    gif_width: int,
    gif_fps: int,
    transparent_formats: str,
    no_premium: bool = False,
    model: str = "rvm",
    gif_white_bg: bool = False,
    rotation: int = 0,
) -> dict[str, Path]:
    suffix = Path(filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        suffix = ".mp4"

    work = OUTPUTS_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"input{suffix}"
    src.write_bytes(data)
    _assert_input_has_video(src)

    if rotation:
        try:
            src = _apply_video_rotation(src, rotation)
        except Exception as exc:
            _log.warning("Rotation failed (rotation=%s): %s — continuing with original", rotation, exc)

    if model == "rembg":
        _run_rembg_job(src, work, gif_width=gif_width)
    elif model == "pro":
        _run_pro_job(src, work, gif_width=gif_width, gif_fps=gif_fps, gif_white_bg=gif_white_bg)
    else:
        device = os.environ.get("RVM_DEVICE", _pick_device("auto"))
        _run_matte_job(
            src,
            work,
            device=device,
            gif_width=gif_width,
            gif_fps=gif_fps,
            transparent_formats=transparent_formats,
            no_premium=no_premium,
        )
    try:
        src.unlink(missing_ok=True)
    except OSError:
        pass
    return _collect_matte_outputs(work)


app = FastAPI(
    title="RobustVideoMatting API (demo)",
    description="Upload MP4 → JSON with URLs to background-removed video and transparent GIF.",
    version="1.1.1",
    docs_url="/docs",
    redoc_url="/redoc",
)
_origins = [o.strip() for o in os.environ.get("RVM_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=False,
)
_static_dir = APP_ROOT / "static"
_static_dir.mkdir(parents=True, exist_ok=True)


def _discover_logo_source() -> Path | None:
    """Find logo on disk: env overrides, then RobustVideoMatting/static/, then repo-root static/."""
    env_path = os.environ.get("RVM_FORMLOOP_LOGO", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_file():
            return p
    static_dir_env = os.environ.get("RVM_STATIC_DIR", "").strip()
    if static_dir_env:
        p = Path(static_dir_env).expanduser().resolve() / "formloop-logo.png"
        if p.is_file():
            return p
    for base in (_static_dir, APP_ROOT.parent / "static"):
        p = base / "formloop-logo.png"
        if p.is_file():
            return p
    return None


def _sync_logo_into_package_static() -> Path:
    """Copy discovered logo into package static/ so StaticFiles always has the file (mount cannot 404)."""
    dst = _static_dir / "formloop-logo.png"
    src = _discover_logo_source()
    if src is not None and src.resolve() != dst.resolve():
        try:
            shutil.copy2(src, dst)
        except OSError:
            pass
    return dst


_FORMLOOP_LOGO = _sync_logo_into_package_static()


def _sniff_image_media_type(path: Path) -> str:
    try:
        h = path.read_bytes()[:12]
    except OSError:
        return "application/octet-stream"
    if len(h) >= 2 and h[:2] == b"\xff\xd8":
        return "image/jpeg"
    if h.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(h) >= 6 and h[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(h) >= 12 and h[:4] == b"RIFF" and h[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


_FORMLOOP_LOGO_MEDIA = (
    _sniff_image_media_type(_FORMLOOP_LOGO) if _FORMLOOP_LOGO.is_file() else "image/png"
)


@app.get("/brand/formloop-logo.png", include_in_schema=False)
async def formloop_brand_logo() -> FileResponse:
    """Logo URL outside /static so it is never shadowed by StaticFiles mount ordering."""
    if not _FORMLOOP_LOGO.is_file():
        raise HTTPException(status_code=404, detail="Logo file missing on server")
    return FileResponse(
        _FORMLOOP_LOGO,
        filename="formloop-logo.png",
        media_type=_FORMLOOP_LOGO_MEDIA,
    )


app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
app.include_router(ui_router)


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap() -> FileResponse:
    p = _static_dir / "sitemap.xml"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="sitemap.xml not found")
    return FileResponse(p, media_type="application/xml")


@app.get("/robots.txt", include_in_schema=False)
async def robots() -> FileResponse:
    p = _static_dir / "robots.txt"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="robots.txt not found")
    return FileResponse(p, media_type="text/plain")


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request) -> JSONResponse:
    """Stripe sends subscription lifecycle events here. Set STRIPE_WEBHOOK_SECRET in Dashboard → webhook."""
    import stripe

    from stripe_integration import ensure_stripe_env, handle_webhook_event

    ensure_stripe_env()
    secret = (os.environ.get("STRIPE_WEBHOOK_SECRET") or "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="STRIPE_WEBHOOK_SECRET not configured")
    payload = await request.body()
    sig = request.headers.get("stripe-signature") or ""
    try:
        event = stripe.Webhook.construct_event(payload, sig, secret)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid webhook payload") from exc
    except Exception as exc:
        if "Signature" in type(exc).__name__ or "signature" in str(exc).lower():
            raise HTTPException(status_code=400, detail="Invalid webhook signature") from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await asyncio.to_thread(handle_webhook_event, event)
    return JSONResponse({"received": True})


@app.on_event("startup")
def _startup() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    from firebase_storage_admin import backfill_quota_counters
    asyncio.create_task(asyncio.to_thread(backfill_quota_counters))
    if (os.environ.get("STRIPE_SECRET_KEY") or "").strip():
        _log.info("Stripe: STRIPE_SECRET_KEY loaded — checkout enabled.")
        if not (os.environ.get("STRIPE_WEBHOOK_SECRET") or "").strip():
            _log.warning(
                "Stripe: STRIPE_WEBHOOK_SECRET is unset — POST /api/stripe/webhook returns 503 until "
                "you add the signing secret from Stripe Dashboard → Webhooks (same .env as the API)."
            )
    else:
        _log.info("Stripe: STRIPE_SECRET_KEY unset — paid checkout disabled (set in RobustVideoMatting/.env).")


@app.get("/health")
async def health() -> JSONResponse:
    rvm_ready = CHECKPOINT.is_file() and PROCESS_SCRIPT.is_file()
    pro_ready = PROCESS_PRO_SCRIPT.is_file()
    return JSONResponse(
        {
            "ok": rvm_ready or pro_ready,
            "has_checkpoint": CHECKPOINT.is_file(),
            "rvm_ready": rvm_ready,
            "pro_ready": pro_ready,
            "outputs_dir": str(OUTPUTS_DIR),
            "save_flow_version": SAVE_FLOW_VERSION,
        }
    )


@app.get("/api/v1/matte/files/{job_id}/{filename}")
async def download_result(job_id: str, filename: str) -> FileResponse:
    if not _JOB_ID_RE.match(job_id) or filename not in ALLOWED_RESULT_FILES:
        raise HTTPException(status_code=404, detail="Not found")
    path = OUTPUTS_DIR / job_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    media = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".apng": "image/apng",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, filename=filename, media_type=media)


@app.get("/api/v1/matte/download")
async def download_remote_asset(url: str, filename: str = "download.bin") -> Response:
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in ALLOWED_REMOTE_DOWNLOAD_HOSTS:
        raise HTTPException(status_code=400, detail="Unsupported download URL")

    safe_name = Path(filename).name or "download.bin"
    try:
        upstream = requests.get(url, timeout=120, allow_redirects=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Download failed: {exc}") from exc

    if upstream.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Download failed: upstream status {upstream.status_code}")

    media = upstream.headers.get("content-type", "application/octet-stream")
    return Response(
        content=upstream.content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


def _sanitize_export_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "", s)[:120]
    return cleaned if _EXPORT_ID_RE.match(cleaned) else ""


@app.get("/api/v1/quota")
async def get_quota(request: Request) -> JSONResponse:
    """Return current quota usage for the signed-in user.
    Accepts session cookie (web) or Authorization: Bearer <token> (mobile).
    Response: {"uid": "...", "used": N, "cap": N|null, "tier": "free"|"pro", "exceeded": bool}

    ``used`` comes from the permanent users/{uid}.quotaUsed Firestore counter
    (incremented once per completed job, never decremented). It must NOT be
    derived from the current count of export documents — deleting a saved
    GIF removes its export doc but must not free up quota, otherwise users
    could delete+re-export to bypass their plan cap indefinitely. Falls back
    to the local on-disk counter only when Firebase isn't configured at all
    (e.g. pure local dev without Firestore).
    """
    uid = (request.session.get("user_id") or "").strip() or None
    if not uid:
        uid = _uid_from_bearer(request)
        if uid:
            request.session["user_id"] = uid
    if not uid:
        raise HTTPException(status_code=401, detail="Sign in required.")
    tier = effective_plan_tier(request) if uid else "free"
    cap = gif_limit_for_tier(tier)
    from firebase_storage_admin import get_quota_counter_from_firestore, firebase_storage_ready
    if firebase_storage_ready():
        used = await asyncio.to_thread(get_quota_counter_from_firestore, uid)
    else:
        used = read_quota_usage(uid, billing_period_key_for_uid(uid))
    exceeded = (cap is not None and used >= cap) if _quota_enforced() else False
    return JSONResponse({
        "uid": uid,
        "used": used,
        "cap": cap,
        "tier": tier,
        "exceeded": exceeded,
    })


@app.post("/api/v1/matte/save/{job_id}")
async def save_export_to_library(job_id: str, request: Request) -> JSONResponse:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")
    uid = (request.session.get("user_id") or "").strip()
    if not uid:
        uid = (_uid_from_bearer(request) or "")
        if uid:
            request.session["user_id"] = uid  # persist Bearer-derived uid for subsequent requests
    if not uid:
        raise HTTPException(status_code=401, detail="Sign in required to save this export.")
    payload: dict = {}
    ct = (request.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try:
            body = await request.json()
            if isinstance(body, dict):
                payload = body
        except Exception:
            payload = {}
    export_id = _sanitize_export_id(str(payload.get("export_id") or ""))
    gif_remote = str(payload.get("gif_url") or "").strip() or None
    webm_remote = str(payload.get("webm_url") or "").strip() or None
    platform = str(payload.get("platform") or "").strip() or None
    title = str(payload.get("title") or "").strip() or None
    try:
        output_rotation = int(payload.get("output_rotation") or 0)
    except (TypeError, ValueError):
        output_rotation = 0
    if output_rotation not in (90, 180, 270):
        output_rotation = 0

    job_dir = OUTPUTS_DIR / job_id
    # Tiny on-disk job record (.owner, .saved); GIF/WebM for your library go to Firebase via URLs below.
    if not job_dir.is_dir():
        if gif_remote or webm_remote:
            job_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise HTTPException(
                status_code=404,
                detail="Unknown job — process the clip again from this app, or Save must include gif_url from the result.",
            )
    owner = read_job_owner(job_dir)
    if owner and owner != uid:
        raise HTTPException(status_code=403, detail="not your export")
    # Optional cache for preview URLs only; Firebase library upload does not depend on this.
    if gif_remote or webm_remote:
        await asyncio.to_thread(_ensure_job_assets_from_client_urls, job_dir, gif_remote, webm_remote)
    try:
        _ensure_webm_for_job(job_dir)
        if not owner:
            write_job_owner(job_id, uid)
        mark_job_saved(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid job_id") from None
    out: dict = {"ok": True, "job_id": job_id}
    if export_id:
        try:
            from firebase_storage_admin import (
                firebase_storage_ready,
                upload_user_export_media,
                upload_user_export_media_from_urls,
            )

            if firebase_storage_ready():
                gif_path = job_dir / "matte.gif"
                webm_path = job_dir / "matte_transparent.webm"
                if output_rotation:
                    try:
                        if gif_path.is_file():
                            await asyncio.to_thread(_rotate_export_file, gif_path, output_rotation)
                        if webm_path.is_file():
                            await asyncio.to_thread(_rotate_export_file, webm_path, output_rotation)
                        print(f"[Rotation] Applied {output_rotation}° to export job_id={job_id}", flush=True)
                    except Exception as exc:
                        _log.warning(
                            "Output rotation failed (rotation=%s job_id=%s): %s — saving unrotated",
                            output_rotation, job_id, exc,
                        )
                urls = None
                # Prefer local file (watermarked) over remote URL to avoid self-fetch issues.
                # Fall back to remote URL only when no local file is present (e.g. Firebase-only job).
                if gif_path.is_file():
                    urls = await asyncio.to_thread(
                        upload_user_export_media,
                        uid=uid,
                        export_id=export_id,
                        gif_path=gif_path,
                        webm_path=webm_path if webm_path.is_file() else None,
                    )
                elif gif_remote and gif_remote.startswith(("http://", "https://")):
                    urls = await asyncio.to_thread(
                        upload_user_export_media_from_urls,
                        uid=uid,
                        export_id=export_id,
                        gif_url=gif_remote,
                        webm_url=webm_remote if webm_remote and webm_remote.startswith(("http://", "https://")) else None,
                    )
                else:
                    out["storageError"] = (
                        "Could not upload to Firebase: no gif_url in this save request. "
                        "Go back to the result screen and tap Save export again."
                    )
                if urls:
                    (job_dir / _STORAGE_URLS_FILE).write_text(
                        json.dumps(urls, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    out["storageGifUrl"] = urls["gifUrl"]
                    out["storageWebmUrl"] = urls.get("webmUrl")
                    # Write to Firestore so the library always shows this export
                    try:
                        from firebase_storage_admin import write_export_to_firestore
                        write_export_to_firestore(
                            uid=uid,
                            export_id=export_id,
                            job_id=job_id,
                            gif_url=urls["gifUrl"],
                            webm_url=urls.get("webmUrl"),
                            platform=platform,
                            title=title,
                        )
                    except Exception:
                        _log.debug(
                            "Firestore export write skipped job_id=%s export_id=%s",
                            job_id, export_id, exc_info=True,
                        )
        except Exception as exc:
            _log.exception("Firebase Storage upload failed job_id=%s export_id=%s", job_id, export_id)
            out["storageError"] = str(exc)
    return JSONResponse(out, headers={"X-FormLoop-Save-Flow": SAVE_FLOW_VERSION})


async def _run_job_async(
    *,
    public_base: str,
    job_id: str,
    raw: bytes,
    filename: str | None,
    gif_width: int,
    gif_fps: int,
    effective_formats: str,
    premium: bool,
    model: str,
    no_premium: bool,
    uid: str | None,
    tier: str,
    gif_white_bg: bool = False,
    rotation: int = 0,
    loop_style: str = "normal",
) -> None:
    _set_job_state(job_id, status="queued", progress=0, message="Queued")
    async with PROCESS_SEMAPHORE:
        _set_job_state(job_id, status="running", progress=1, message="Starting model")
        try:
            skip_wm = (os.environ.get("RVM_SKIP_GIF_WATERMARK") or "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            await asyncio.to_thread(
                _process_to_job_dir,
                job_id,
                raw,
                filename,
                gif_width=gif_width,
                gif_fps=gif_fps,
                transparent_formats=effective_formats,
                no_premium=no_premium,
                model=model,
                gif_white_bg=gif_white_bg,
                rotation=rotation,
            )
            gif_path = OUTPUTS_DIR / job_id / "matte.gif"
            if loop_style == "reverse" and gif_path.is_file():
                ok = await asyncio.to_thread(_apply_reverse_loop, gif_path)
                print(f"[REVERSE] _apply_reverse_loop result={ok} gif_exists={gif_path.is_file()}", flush=True)
            if uid:
                try:
                    write_job_owner(job_id, uid)
                except ValueError:
                    pass
                increment_quota_usage(uid, billing_period_key_for_uid(uid))
                from firebase_storage_admin import increment_quota_counter_in_firestore
                await asyncio.to_thread(increment_quota_counter_in_firestore, uid)
            # Watermark applies to ALL free-tier users, including mobile (uid from Bearer token)
            if exports_watermark_for_tier(tier) and not skip_wm:
                wm = resolve_watermark_png(APP_ROOT)
                if wm and gif_path.is_file():
                    _set_job_state(
                        job_id,
                        progress=96,
                        message="Applying watermark…",
                        status="running",
                    )
                    await asyncio.to_thread(apply_png_watermark_to_gif, gif_path, wm)
            if gif_path.is_file() and _gif_frame_count(gif_path) <= 1:
                rebuilt = await asyncio.to_thread(_rebuild_transparent_gif_from_fg_alpha, OUTPUTS_DIR / job_id, gif_width)
                if rebuilt and exports_watermark_for_tier(tier) and not skip_wm:
                    wm = resolve_watermark_png(APP_ROOT)
                    if wm:
                        await asyncio.to_thread(apply_png_watermark_to_gif, gif_path, wm)
            body = _build_result_body(
                public_base,
                job_id,
                gif_width=gif_width,
                effective_formats=effective_formats,
                premium=premium,
                model=model,
            )
            _set_job_state(job_id, status="completed", progress=100, message="Completed", result=body)
        except subprocess.CalledProcessError as exc:
            work = OUTPUTS_DIR / job_id
            tail = await asyncio.to_thread(_log_tail, work)
            shutil.rmtree(work, ignore_errors=True)
            msg = (
                f"Processing failed (subprocess exit {exc.returncode}). "
                "If this persists: try a shorter clip, premium=false (default), transparent_formats=gif, "
                "or check _process_video.log on server for FFmpeg/torch errors."
            )
            if tail:
                msg = f"{msg}\n\n--- process log (tail) ---\n{tail}"
            _set_job_state(job_id, status="failed", progress=100, message=msg, error=msg)
        except Exception as exc:
            work = OUTPUTS_DIR / job_id
            tail = await asyncio.to_thread(_log_tail, work)
            shutil.rmtree(work, ignore_errors=True)
            detail = str(exc)
            if tail:
                detail = f"{detail}\n\n--- process log (tail) ---\n{tail}"
            _set_job_state(job_id, status="failed", progress=100, message=detail, error=detail)


async def _run_runpod_job_async(
    *,
    job_id: str,
    raw: bytes,
    filename: str | None,
    gif_width: int,
    gif_fps: int,
    public_base: str,
    uid: str | None = None,
    tier: str = "free",
    pro_fast_mode: bool | None = None,
    gif_white_bg: bool = False,
    start_time: float | None = None,
    end_time: float | None = None,
    rotation: int = 0,
    loop_style: str = "normal",
    use_sam2: bool = False,
) -> None:
    _set_job_state(job_id, status="queued", progress=0, message="Queued for RunPod")
    try:
        run_id = await asyncio.to_thread(
            _runpod_submit,
            raw,
            job_id=job_id,
            filename=filename,
            gif_width=gif_width,
            gif_fps=gif_fps,
            public_base=public_base,
            pro_fast_mode=pro_fast_mode,
            gif_white_bg=gif_white_bg,
            start_time=start_time,
            end_time=end_time,
            rotation=rotation,
            loop_style=loop_style,
            use_sam2=use_sam2,
        )
        _set_job_state(job_id, status="queued", progress=1, message="Waiting in queue...", runpod_job_id=run_id)
        payload = await asyncio.to_thread(_runpod_wait, run_id, None, job_id)
        status = str(payload.get("status") or "").upper()
        if status == "CANCELLED":
            return
        if status != "COMPLETED":
            msg = str(payload.get("error") or payload.get("status") or "RunPod failed")
            if RUNPOD_ALLOW_LOCAL_FALLBACK:
                _set_job_state(
                    job_id,
                    status="running",
                    progress=35,
                    message="RunPod slow/unavailable. Finishing locally for faster delivery",
                )
                await asyncio.to_thread(
                    _process_to_job_dir,
                    job_id,
                    raw,
                    filename,
                    gif_width=gif_width,
                    gif_fps=gif_fps,
                    transparent_formats="gif",
                    no_premium=True,
                    model="pro",
                    gif_white_bg=gif_white_bg,
                    rotation=rotation,
                )
                body = _build_result_body(
                    public_base,
                    job_id,
                    gif_width=gif_width,
                    effective_formats="gif",
                    premium=False,
                    model="pro",
                )
                _fb1_gif = OUTPUTS_DIR / job_id / "matte.gif"
                if loop_style == "reverse" and _fb1_gif.is_file():
                    await asyncio.to_thread(_apply_reverse_loop, _fb1_gif)
                _fb1_skip_wm = (os.environ.get("RVM_SKIP_GIF_WATERMARK") or "").strip().lower() in {"1", "true", "yes"}
                if uid and exports_watermark_for_tier(tier) and not _fb1_skip_wm:
                    wm = resolve_watermark_png(APP_ROOT)
                    if wm and _fb1_gif.is_file():
                        await asyncio.to_thread(apply_png_watermark_to_gif, _fb1_gif, wm)
                body["message"] = "Completed locally after RunPod delay."
                _set_job_state(
                    job_id,
                    status="completed",
                    progress=100,
                    message="Completed (local fallback)",
                    result=body,
                )
                return
            _set_job_state(job_id, status="failed", progress=100, message=msg, error=msg)
            return
        await asyncio.to_thread(_persist_runpod_job_artifacts, job_id, payload)

        # Register owner and increment lifetime quota counter (bug fix: RunPod path was missing this)
        if uid:
            try:
                write_job_owner(job_id, uid)
            except ValueError:
                pass
            increment_quota_usage(uid, billing_period_key_for_uid(uid))
            from firebase_storage_admin import increment_quota_counter_in_firestore
            await asyncio.to_thread(increment_quota_counter_in_firestore, uid)

        gif_path = OUTPUTS_DIR / job_id / "matte.gif"

        # Apply reverse loop before watermark so watermark lands on the final frames
        print(f"[REVERSE] loop_style={loop_style}", flush=True)
        if loop_style == "reverse":
            print(f"[REVERSE] gif_path={gif_path} exists={gif_path.is_file()}", flush=True)
            if gif_path.is_file():
                await asyncio.to_thread(_apply_reverse_loop, gif_path)
            else:
                print("[REVERSE] WARNING: matte.gif not found locally — reverse loop skipped", flush=True)

        # Apply watermark for all free-tier users (uid no longer required — mobile uses Bearer)
        skip_wm = (os.environ.get("RVM_SKIP_GIF_WATERMARK") or "").strip().lower() in {"1", "true", "yes"}
        if exports_watermark_for_tier(tier) and not skip_wm and gif_path.is_file():
            wm = resolve_watermark_png(APP_ROOT)
            if wm:
                print(f"[WATERMARK] Applying watermark tier={tier}", flush=True)
                await asyncio.to_thread(apply_png_watermark_to_gif, gif_path, wm)

        # Serve local URL when GIF is present (watermarked/reversed), Firebase URL otherwise
        if gif_path.is_file():
            result_body = _build_result_body(
                public_base, job_id, gif_width=gif_width, effective_formats="gif", premium=False, model="pro"
            )
        else:
            result_body = _runpod_output_to_body(job_id, payload)

        _set_job_state(
            job_id,
            status="completed",
            progress=100,
            message="Completed",
            result=result_body,
        )
    except Exception as exc:
        msg = str(exc)
        if RUNPOD_ALLOW_LOCAL_FALLBACK:
            try:
                _set_job_state(
                    job_id,
                    status="running",
                    progress=35,
                    message="RunPod error. Finishing locally for faster delivery",
                )
                await asyncio.to_thread(
                    _process_to_job_dir,
                    job_id,
                    raw,
                    filename,
                    gif_width=gif_width,
                    gif_fps=gif_fps,
                    transparent_formats="gif",
                    no_premium=True,
                    model="pro",
                    gif_white_bg=gif_white_bg,
                    rotation=rotation,
                )
                _fb2_gif = OUTPUTS_DIR / job_id / "matte.gif"
                if loop_style == "reverse" and _fb2_gif.is_file():
                    await asyncio.to_thread(_apply_reverse_loop, _fb2_gif)
                _fb2_skip_wm = (os.environ.get("RVM_SKIP_GIF_WATERMARK") or "").strip().lower() in {"1", "true", "yes"}
                if uid and exports_watermark_for_tier(tier) and not _fb2_skip_wm:
                    wm = resolve_watermark_png(APP_ROOT)
                    if wm and _fb2_gif.is_file():
                        await asyncio.to_thread(apply_png_watermark_to_gif, _fb2_gif, wm)
                body = _build_result_body(
                    public_base,
                    job_id,
                    gif_width=gif_width,
                    effective_formats="gif",
                    premium=False,
                    model="pro",
                )
                body["message"] = "Completed locally after RunPod error."
                _set_job_state(
                    job_id,
                    status="completed",
                    progress=100,
                    message="Completed (local fallback)",
                    result=body,
                )
                return
            except Exception as local_exc:
                msg = f"{msg}; local fallback failed: {local_exc}"
        _set_job_state(job_id, status="failed", progress=100, message=msg, error=msg)


@app.post("/api/v1/matte/start")
async def matte_video_start(
    request: Request,
    file: UploadFile = File(..., description="Video file (MP4, MOV, etc.)"),
    gif_width: int = Query(960, ge=320, le=1280, description="GIF max width in px."),
    gif_fps: int = Query(0, ge=0, le=30, description="GIF FPS (0 = auto-detect from source, capped at 24)."),
    fast_mode: bool = Query(True, description="True = BiRefNet fast mode on RunPod (no YOLO)."),
    transparent_formats: str = Query("gif", description="Comma-separated: gif, webp, apng."),
    premium: bool = Query(False, description="True = heavy premium pass."),
    model: str = Query("rvm", pattern="^(rvm|rembg|pro)$"),
    gif_white_bg: bool = Query(
        False,
        description="If true, GIF is opaque on white. Default false = transparent GIF (clean on white via edge whitening).",
    ),
    start_time: float | None = Query(None, description="Trim start in seconds."),
    end_time: float | None = Query(None, description="Trim end in seconds."),
    rotation: int = Query(0, ge=0, le=270, description="Clockwise rotation degrees (0, 90, 180, 270)."),
    loop_style: str = Query("normal", description="Loop style: 'normal' or 'reverse' (boomerang)."),
    use_sam2: bool = Query(False, description="True = Ultra Quality (SAM2 + BiRefNet on RunPod). Slower, sharper edges."),
) -> JSONResponse:
    if _runpod_only_mode() and not _runpod_enabled():
        raise HTTPException(
            status_code=503,
            detail="RunPod is required in this environment but not configured.",
        )
    if _runpod_enabled() and model != "pro":
        model = "pro"
    if model == "rvm" and not CHECKPOINT.is_file():
        raise HTTPException(status_code=503, detail="Model checkpoint missing (rvm_resnet50.pth)")
    if model == "pro" and not PROCESS_PRO_SCRIPT.is_file():
        raise HTTPException(status_code=503, detail="process_video_pro.py missing")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )
    if len(raw) < 256:
        raise HTTPException(status_code=400, detail="Empty or too small file")

    # Auto-detect source FPS and cap at 24; fall back to 12 if probe fails
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        suffix = ".mp4"
    if gif_fps <= 0:
        gif_fps = await asyncio.to_thread(_probe_video_fps, raw, suffix)
    print(f"[GIF FPS] matte_video_start: probed/requested gif_fps={gif_fps}", flush=True)

    # Clamp rotation to valid values
    rotation = rotation if rotation in (0, 90, 180, 270) else 0
    loop_style = loop_style if loop_style in ("normal", "reverse") else "normal"

    await asyncio.to_thread(_prune_old_outputs)

    job_id = uuid.uuid4().hex
    uid = (request.session.get("user_id") or "").strip() or None
    if not uid:
        uid = _uid_from_bearer(request)
        if uid:
            request.session["user_id"] = uid  # persist Bearer-derived uid for subsequent requests
    tier = "free"
    if uid:
        request.session.setdefault("plan_tier", "free")
        tier = effective_plan_tier(request)
        cap = gif_limit_for_tier(tier)
        if _quota_enforced() and cap is not None:
            from firebase_storage_admin import get_quota_counter_from_firestore, firebase_storage_ready
            if firebase_storage_ready():
                used = await asyncio.to_thread(get_quota_counter_from_firestore, uid)
            else:
                used = read_quota_usage(uid, billing_period_key_for_uid(uid))
            if used >= cap:
                raise HTTPException(
                    status_code=403,
                    detail=f"You've used all {cap} GIFs on your plan. Please upgrade to continue.",
                )

    no_premium = not premium
    effective_formats = "gif" if no_premium else transparent_formats
    if _runpod_enabled():
        # RunPod serverless guardrails: force single fast pass to reduce timeout risk/cost.
        if RUNPOD_FORCE_FAST_MODE:
            fast_mode = True
            if model == "pro":
                gif_width = min(gif_width, RUNPOD_MAX_GIF_WIDTH)
                # gif_fps is already probed from the source and capped at 24 — don't
                # squash it further with RUNPOD_MAX_GIF_FPS (was 10, caused "too fast" GIFs).
        print(f"[GIF FPS] sending to RunPod: gif_fps={gif_fps} loop_style={loop_style}", flush=True)
        public_base = _public_base_url(request)
        _set_job_state(job_id, status="queued", progress=0, message="Queued for RunPod")
        asyncio.create_task(
            _run_runpod_job_async(
                job_id=job_id,
                raw=raw,
                filename=file.filename,
                gif_width=gif_width,
                gif_fps=gif_fps,
                public_base=public_base,
                uid=uid,
                tier=tier,
                pro_fast_mode=bool(fast_mode),
                gif_white_bg=bool(gif_white_bg),
                start_time=start_time,
                end_time=end_time,
                rotation=rotation,
                loop_style=loop_style,
                use_sam2=bool(use_sam2),
            )
        )
        return JSONResponse(
            {
                "success": True,
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "RunPod processing started",
                "progress_url": f"{public_base}/api/v1/matte/progress/{job_id}",
            }
        )
    _set_job_state(job_id, status="queued", progress=0, message="Queued")
    asyncio.create_task(
        _run_job_async(
            public_base=_public_base_url(request),
            job_id=job_id,
            raw=raw,
            filename=file.filename,
            gif_width=gif_width,
            gif_fps=gif_fps,
            effective_formats=effective_formats,
            premium=premium,
            model=model,
            no_premium=no_premium,
            uid=uid,
            tier=tier,
            gif_white_bg=bool(gif_white_bg),
            rotation=rotation,
            loop_style=loop_style,
        )
    )
    return JSONResponse(
        {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Processing started",
            "progress_url": f"{_public_base_url(request)}/api/v1/matte/progress/{job_id}",
        }
    )


@app.get("/api/v1/matte/progress/{job_id}")
async def matte_video_progress(job_id: str) -> JSONResponse:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")
    state = _get_job_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    status = str(state.get("status") or "queued")
    progress = int(state.get("progress") or 0)
    message = str(state.get("message") or "Queued")
    if status in {"queued", "running"}:
        est_progress, est_message = _estimate_job_progress(job_id, job_status=status)
        # Only let the disk-based estimate take over once it actually knows more
        # than the state we already have (e.g. RunPod queue/init messages set by
        # _runpod_wait) — otherwise it clobbers accurate messages with "Queued".
        if est_progress > progress:
            progress = est_progress
            message = est_message
            _set_job_state(job_id, progress=progress, message=est_message, status="running")
    body = {
        "job_id": job_id,
        "status": status,
        "progress": max(0, min(100, progress)),
        "message": message,
        "done": status in {"completed", "failed"},
    }
    if status == "completed" and state.get("result"):
        body["result"] = state["result"]
    if status == "failed":
        body["error"] = str(state.get("error") or message)
    return JSONResponse(body)


@app.post("/api/v1/matte/cancel/{job_id}")
async def matte_cancel(job_id: str) -> JSONResponse:
    """Cancel an in-progress matte job and terminate the RunPod job if active."""
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")
    state = _get_job_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    current_status = str(state.get("status") or "")
    if current_status in {"completed", "failed", "cancelled"}:
        return JSONResponse({"ok": True, "job_id": job_id, "status": current_status, "note": "already terminal"})
    _set_job_state(job_id, status="cancelled", progress=0, message="Cancelled by user")
    runpod_job_id = str(state.get("runpod_job_id") or "")
    cancelled_runpod = False
    if runpod_job_id and _runpod_enabled():
        cancelled_runpod = await asyncio.to_thread(_runpod_cancel, runpod_job_id)
    return JSONResponse({"ok": True, "job_id": job_id, "status": "cancelled", "runpod_cancelled": cancelled_runpod})


@app.post("/api/v1/matte")
async def matte_video(
    request: Request,
    file: UploadFile = File(..., description="Video file (MP4, MOV, etc.)"),
    gif_width: int = Query(
        960,
        ge=320,
        le=1280,
        description="GIF max width in px — higher = sharper (default 960 for demo quality).",
    ),
    gif_fps: int = Query(12, ge=1, le=30, description="GIF FPS (1-30; lower is faster)."),
    fast_mode: bool = Query(True, description="True = BiRefNet fast mode on RunPod (no YOLO)."),
    transparent_formats: str = Query(
        "gif",
        description="Comma-separated: gif, webp, apng. WebP/APNG need premium=true.",
    ),
    premium: bool = Query(
        False,
        description="True = full edge refine + YOLO subject gate + WebP/APNG capable (heavy). False = fast transparent GIF only (best for phones).",
    ),
    model: str = Query(
        "rvm",
        pattern="^(rvm|rembg|pro)$",
        description="rvm (default), rembg, or pro (BiRefNet+YOLO; mobile app uses model=pro).",
    ),
    gif_white_bg: bool = Query(
        False,
        description="If true, pro GIF is opaque on white. Default false = transparent GIF.",
    ),
    start_time: float | None = Query(None, description="Trim start in seconds."),
    end_time: float | None = Query(None, description="Trim end in seconds."),
    use_sam2: bool = Query(False, description="True = Ultra Quality (SAM2 + BiRefNet on RunPod). Slower, sharper edges."),
) -> JSONResponse:
    """
    Returns JSON with **absolute URLs** for each generated asset (easy to use from mobile).

    Multipart field name: **file**.
    """
    if _runpod_only_mode() and not _runpod_enabled():
        raise HTTPException(
            status_code=503,
            detail="RunPod is required in this environment but not configured.",
        )
    if _runpod_enabled() and model != "pro":
        model = "pro"
    if model == "rvm" and not CHECKPOINT.is_file():
        raise HTTPException(status_code=503, detail="Model checkpoint missing (rvm_resnet50.pth)")
    if model == "pro" and not PROCESS_PRO_SCRIPT.is_file():
        raise HTTPException(status_code=503, detail="process_video_pro.py missing")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )
    if len(raw) < 256:
        raise HTTPException(status_code=400, detail="Empty or too small file")

    await asyncio.to_thread(_prune_old_outputs)

    job_id = uuid.uuid4().hex
    uid = (request.session.get("user_id") or "").strip() or None
    if not uid:
        uid = _uid_from_bearer(request)
        if uid:
            request.session["user_id"] = uid  # persist Bearer-derived uid for subsequent requests
    tier = "free"
    if uid:
        request.session.setdefault("plan_tier", "free")
        tier = effective_plan_tier(request)
        cap = gif_limit_for_tier(tier)
        if _quota_enforced() and cap is not None:
            from firebase_storage_admin import get_quota_counter_from_firestore, firebase_storage_ready
            if firebase_storage_ready():
                used = await asyncio.to_thread(get_quota_counter_from_firestore, uid)
            else:
                used = read_quota_usage(uid, billing_period_key_for_uid(uid))
            if used >= cap:
                raise HTTPException(
                    status_code=403,
                    detail=f"You've used all {cap} GIFs on your plan. Please upgrade to continue.",
                )

    no_premium = not premium
    effective_formats = "gif" if no_premium else transparent_formats
    if _runpod_enabled():
        if RUNPOD_FORCE_FAST_MODE:
            fast_mode = True
            if model == "pro":
                gif_width = min(gif_width, RUNPOD_MAX_GIF_WIDTH)
                gif_fps = min(gif_fps, RUNPOD_MAX_GIF_FPS)
        public_base = _public_base_url(request)
        try:
            run_id = await asyncio.to_thread(
                _runpod_submit,
                raw,
                job_id=job_id,
                filename=file.filename,
                gif_width=gif_width,
                gif_fps=gif_fps,
                public_base=public_base,
                pro_fast_mode=bool(fast_mode),
                gif_white_bg=bool(gif_white_bg),
                start_time=start_time,
                end_time=end_time,
                use_sam2=bool(use_sam2),
            )
            payload = await asyncio.to_thread(_runpod_wait, run_id, None, job_id)
            status = str(payload.get("status") or "").upper()
            if status == "CANCELLED":
                raise HTTPException(status_code=499, detail="Job cancelled by client")
            if status != "COMPLETED":
                raise HTTPException(
                    status_code=500,
                    detail=str(payload.get("error") or payload.get("status") or "RunPod failed"),
                )
            await asyncio.to_thread(_persist_runpod_job_artifacts, job_id, payload)
            return JSONResponse(_runpod_output_to_body(job_id, payload))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"RunPod processing failed: {exc}") from exc

    async with PROCESS_SEMAPHORE:
        try:
            paths = await asyncio.to_thread(
                _process_to_job_dir,
                job_id,
                raw,
                file.filename,
                gif_width=gif_width,
                gif_fps=gif_fps,
                transparent_formats=effective_formats,
                no_premium=no_premium,
                model=model,
                gif_white_bg=bool(gif_white_bg),
            )
        except subprocess.CalledProcessError as exc:
            work = OUTPUTS_DIR / job_id
            tail = await asyncio.to_thread(_log_tail, work)
            shutil.rmtree(work, ignore_errors=True)
            msg = (
                f"Processing failed (subprocess exit {exc.returncode}). "
                "If this persists: try a shorter clip, premium=false (default), transparent_formats=gif, "
                "or check _process_video.log on server for FFmpeg/torch errors."
            )
            if tail:
                msg = f"{msg}\n\n--- process log (tail) ---\n{tail}"
            raise HTTPException(status_code=500, detail=msg) from exc
        except HTTPException:
            raise
        except Exception as exc:
            work = OUTPUTS_DIR / job_id
            tail = await asyncio.to_thread(_log_tail, work)
            shutil.rmtree(work, ignore_errors=True)
            detail = str(exc)
            if tail:
                detail = f"{detail}\n\n--- process log (tail) ---\n{tail}"
            raise HTTPException(status_code=500, detail=detail) from exc

    skip_wm = (os.environ.get("RVM_SKIP_GIF_WATERMARK") or "").strip().lower() in {"1", "true", "yes"}
    if uid:
        try:
            write_job_owner(job_id, uid)
        except ValueError:
            pass
        increment_quota_usage(uid, billing_period_key_for_uid(uid))
        from firebase_storage_admin import increment_quota_counter_in_firestore
        await asyncio.to_thread(increment_quota_counter_in_firestore, uid)
    # Watermark applies to all free-tier users (uid not required — mobile uses Bearer token)
    gif_path = OUTPUTS_DIR / job_id / "matte.gif"
    if exports_watermark_for_tier(tier) and not skip_wm:
        wm = resolve_watermark_png(APP_ROOT)
        if wm and gif_path.is_file():
            await asyncio.to_thread(apply_png_watermark_to_gif, gif_path, wm)
    if gif_path.is_file() and _gif_frame_count(gif_path) <= 1:
        rebuilt = await asyncio.to_thread(_rebuild_transparent_gif_from_fg_alpha, OUTPUTS_DIR / job_id, gif_width)
        if rebuilt and exports_watermark_for_tier(tier) and not skip_wm:
            wm = resolve_watermark_png(APP_ROOT)
            if wm:
                await asyncio.to_thread(apply_png_watermark_to_gif, gif_path, wm)

    body: dict = _build_result_body(
        _public_base_url(request),
        job_id,
        gif_width=gif_width,
        effective_formats=effective_formats,
        premium=premium,
        model=model,
    )
    return JSONResponse(body)


@app.get("/api/v1/matte/input/{job_id}/{filename}")
async def download_input_for_runpod(job_id: str, filename: str, token: str = "") -> FileResponse:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="Not found")
    st = _get_job_state(job_id) or {}
    want = str(st.get("runpod_input_token") or "")
    if not want or token != want:
        raise HTTPException(status_code=403, detail="Forbidden")
    safe = Path(filename).name
    path = OUTPUTS_DIR / job_id / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    media = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".m4v": "video/mp4",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, filename=safe, media_type=media)


_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Formloop API — spec & try</title>
  <style>
    :root { font-family: system-ui, sans-serif; line-height: 1.5; color: #1a1a1a; }
    body { max-width: 52rem; margin: 0 auto; padding: 1.25rem; }
    h1 { font-size: 1.35rem; margin-top: 0; }
    h2 { font-size: 1.05rem; margin-top: 1.75rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }
    code, pre { font-family: ui-monospace, monospace; font-size: 0.82rem; }
    pre { background: #f4f4f5; padding: 0.75rem 1rem; overflow-x: auto; border-radius: 6px; }
    .muted { color: #555; font-size: 0.9rem; }
    label { display: block; margin: 0.5rem 0 0.25rem; font-weight: 600; }
    input[type=file] { margin: 0.25rem 0 1rem; }
    button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; border-radius: 6px; border: 1px solid #333; background: #111; color: #fff; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #out { margin-top: 1rem; }
    #err { color: #b00020; white-space: pre-wrap; }
    a { color: #0645ad; word-break: break-all; }
    ul.compact { margin: 0.25rem 0; padding-left: 1.25rem; }
  </style>
</head>
<body>
  <h1>Formloop — video API</h1>
  <p class="muted">Upload an MP4 (or similar). Response is <strong>JSON with download URLs</strong> — no API key. Open this page from the same host/port you expose to the client (or set <code>RVM_PUBLIC_BASE_URL</code> on the server for correct links behind ngrok).</p>

  <h2>Endpoints</h2>
  <ul class="compact">
    <li><code>GET /health</code> — server + model check</li>
    <li><code>POST /api/v1/matte</code> — multipart <code>file</code>; query: <code>gif_width</code> (320–1280, default 960), <code>premium</code> (default <code>false</code> = fast GIF-only), <code>transparent_formats</code>, <code>model</code> (<code>rvm</code> | <code>rembg</code> | <code>pro</code> — mobile demo uses <code>pro</code>)</li>
    <li><code>GET /api/v1/matte/files/{job_id}/{filename}</code> — each asset from the JSON</li>
    <li><code>GET /docs</code> — OpenAPI (Swagger)</li>
  </ul>

  <h2>Example response (shape)</h2>
  <pre>{
  "success": true,
  "job_id": "…32 hex chars…",
  "background_removed": {
    "transparent_video_webm": "https://YOUR_HOST/api/v1/matte/files/…/matte_transparent.webm",
    "foreground_mp4": "…",
    "alpha_mp4": "…",
    "preview_white_backdrop_mp4": "…"
  },
  "gif_and_animation": {
    "transparent_gif": "…/matte.gif",
    "transparent_webp": "…/matte.webp",
    "transparent_apng": null
  },
  "phone_implementation": { … }
}</pre>

  <h2>Try it (browser)</h2>
  <form id="f">
    <label for="vid">Video file</label>
    <input id="vid" name="file" type="file" accept="video/mp4,video/quicktime,video/*" required />
    <button type="submit" id="go">Upload &amp; process</button>
  </form>
  <p id="err"></p>
  <div id="out"></div>

  <script>
    const form = document.getElementById('f');
    const out = document.getElementById('out');
    const err = document.getElementById('err');
    const go = document.getElementById('go');
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      err.textContent = '';
      out.innerHTML = '';
      const fd = new FormData(form);
      go.disabled = true;
      try {
        const r = await fetch('/api/v1/matte', { method: 'POST', body: fd });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) {
          err.textContent = (j.detail && (typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail))) || r.statusText;
          return;
        }
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(j, null, 2);
        out.appendChild(pre);
        const links = document.createElement('div');
        links.style.marginTop = '1rem';
        links.innerHTML = '<strong>Quick links</strong><br/>';
        function add(title, url) {
          if (!url) return;
          const a = document.createElement('a');
          a.href = url; a.textContent = title + ' → ' + url; a.target = '_blank';
          links.appendChild(a);
          links.appendChild(document.createElement('br'));
        }
        const br = j.background_removed || {};
        const ga = j.gif_and_animation || {};
        add('WebM (alpha)', br.transparent_video_webm);
        add('GIF', ga.transparent_gif);
        add('WebP', ga.transparent_webp);
        add('Foreground MP4', br.foreground_mp4);
        add('Alpha MP4', br.alpha_mp4);
        out.appendChild(links);
      } catch (x) {
        err.textContent = String(x);
      } finally {
        go.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/")
async def demo_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/demo", response_class=HTMLResponse)
async def demo_page() -> str:
    return _DEMO_HTML
