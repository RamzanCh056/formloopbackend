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
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from billing_plans import exports_watermark_for_tier, gif_limit_for_tier, plan_tier_from_session
from gif_watermark import apply_png_watermark_to_gif, resolve_watermark_png
from output_job_store import (
    increment_owner_usage,
    mark_job_saved,
    read_job_owner,
    read_owner_usage_count,
    write_job_owner,
)
from process_video import _assert_input_has_video, _pick_device, _python_for_inference
from web_ui_routes import SESSION_SECRET, router as ui_router

_log = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parent
PROCESS_SCRIPT = APP_ROOT / "process_video.py"
PROCESS_REMBG_SCRIPT = APP_ROOT / "process_video_rembg.py"
PROCESS_PRO_SCRIPT = APP_ROOT / "process_video_pro.py"
CHECKPOINT = APP_ROOT / "rvm_resnet50.pth"
OUTPUTS_DIR = Path(os.environ.get("RVM_OUTPUTS_DIR", str(APP_ROOT / "api_outputs"))).resolve()

MAX_UPLOAD_BYTES = int(os.environ.get("RVM_MAX_UPLOAD_MB", "300")) * 1024 * 1024
PROCESS_SEMAPHORE = asyncio.Semaphore(int(os.environ.get("RVM_MAX_CONCURRENT_JOBS", "1")))
OUTPUT_RETENTION_SEC = int(os.environ.get("RVM_OUTPUT_RETENTION_SEC", str(86400)))

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

_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_EXPORT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,120}$")
_STORAGE_URLS_FILE = ".storage_urls.json"
_TQDM_PERCENT_RE = re.compile(r"(\d{1,3})%\|")
_RVM_PROGRESS_RE = re.compile(r"RVM_PROGRESS:(\d+)/(\d+):(\d{1,3})")
_JOB_STATE_LOCK = threading.Lock()
_JOB_STATES: dict[str, dict] = {}


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
        "ffmpeg",
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
        "--transparent-gif",
        "--transparent-formats",
        transparent_formats,
        "--gif-width",
        str(gif_width),
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
) -> None:
    """Runs process_video_pro (BiRefNet + YOLOv8x-pose + YOLOv8x-seg; same CLI as local GIF gen)."""
    fg = work / "foreground.mp4"
    alpha = work / "alpha.mp4"
    gif_out = work / "matte.gif"
    py = _python_for_inference(APP_ROOT.parent)
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
        str(max(1, int(os.environ.get("RVM_PRO_GIF_FPS", "15")))),
        "--device",
        os.environ.get("RVM_DEVICE", _pick_device("auto")),
        "--dilation",
        str(max(0, int(os.environ.get("RVM_PRO_DILATION", "18")))),
        "--conf",
        str(float(os.environ.get("RVM_PRO_CONF", "0.20"))),
        "--rvm-downsample",
        str(float(os.environ.get("RVM_PRO_RVM_DOWNSAMPLE", "0.4"))),
    ]
    if os.environ.get("RVM_PRO_NO_RVM", "").strip().lower() in {"1", "true", "yes"}:
        cmd.append("--no-rvm")
    if os.environ.get("RVM_PRO_NO_YOLO", "").strip().lower() in {"1", "true", "yes"}:
        cmd.append("--no-yolo")
    log_path = work / "_process_video.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Never pause at frame 100 in API mode.
    env["RVM_PRO_HALT_AFTER_FRAME"] = "0"
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            check=True,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )


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
    gf = max(8, int(os.environ.get("RVM_GIF_FPS", "20")))
    fc = (
        f"[0:v]format=rgb24[rgb];"
        f"[1:v]format=gray,extractplanes=y[am];"
        f"[rgb][am]alphamerge,format=rgba,"
        f"fps={gf},scale={gw}:-1:flags=lanczos,split[s0][s1];"
        f"[s0]palettegen=stats_mode=full:max_colors=255:reserve_transparent=1[p];"
        f"[s1][p]paletteuse=alpha_threshold=64:diff_mode=rectangle:dither=sierra2_4a"
    )
    cmd = [
        "ffmpeg",
        "-y",
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


def _estimate_job_progress(job_id: str) -> tuple[int, str]:
    work = OUTPUTS_DIR / job_id
    if not work.exists():
        return 0, "Queued"
    if (work / "matte_transparent.webm").is_file() or (work / "matte.gif").is_file():
        return 100, "Finalizing output"
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
    return max(0, min(100, pct)), msg


def _process_to_job_dir(
    job_id: str,
    data: bytes,
    filename: str | None,
    *,
    gif_width: int,
    transparent_formats: str,
    no_premium: bool = False,
    model: str = "rvm",
) -> dict[str, Path]:
    suffix = Path(filename or "upload.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        suffix = ".mp4"

    work = OUTPUTS_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)
    src = work / f"input{suffix}"
    src.write_bytes(data)
    _assert_input_has_video(src)

    if model == "rembg":
        _run_rembg_job(src, work, gif_width=gif_width)
    elif model == "pro":
        _run_pro_job(src, work, gif_width=gif_width)
    else:
        device = os.environ.get("RVM_DEVICE", _pick_device("auto"))
        _run_matte_job(
            src,
            work,
            device=device,
            gif_width=gif_width,
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


@app.on_event("startup")
def _startup() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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


def _sanitize_export_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "", s)[:120]
    return cleaned if _EXPORT_ID_RE.match(cleaned) else ""


@app.post("/api/v1/matte/save/{job_id}")
async def save_export_to_library(job_id: str, request: Request) -> JSONResponse:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")
    uid = (request.session.get("user_id") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="Sign in required to save this export.")
    job_dir = OUTPUTS_DIR / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="job not found")
    owner = read_job_owner(job_dir)
    if owner and owner != uid:
        raise HTTPException(status_code=403, detail="not your export")
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
            from firebase_storage_admin import firebase_storage_ready, upload_user_export_media

            if firebase_storage_ready():
                gif_path = job_dir / "matte.gif"
                webm_path = job_dir / "matte_transparent.webm"
                urls = await asyncio.to_thread(
                    upload_user_export_media,
                    uid=uid,
                    export_id=export_id,
                    gif_path=gif_path,
                    webm_path=webm_path if webm_path.is_file() else None,
                )
                (job_dir / _STORAGE_URLS_FILE).write_text(
                    json.dumps(urls, separators=(",", ":")),
                    encoding="utf-8",
                )
                out["storageGifUrl"] = urls["gifUrl"]
                out["storageWebmUrl"] = urls.get("webmUrl")
        except Exception as exc:
            _log.exception("Firebase Storage upload failed job_id=%s export_id=%s", job_id, export_id)
            out["storageError"] = str(exc)
    return JSONResponse(out)


async def _run_job_async(
    *,
    public_base: str,
    job_id: str,
    raw: bytes,
    filename: str | None,
    gif_width: int,
    effective_formats: str,
    premium: bool,
    model: str,
    no_premium: bool,
    uid: str | None,
    tier: str,
) -> None:
    _set_job_state(job_id, status="queued", progress=0, message="Queued")
    async with PROCESS_SEMAPHORE:
        _set_job_state(job_id, status="running", progress=1, message="Starting model")
        try:
            await asyncio.to_thread(
                _process_to_job_dir,
                job_id,
                raw,
                filename,
                gif_width=gif_width,
                transparent_formats=effective_formats,
                no_premium=no_premium,
                model=model,
            )
            if uid:
                try:
                    write_job_owner(job_id, uid)
                except ValueError:
                    pass
                increment_owner_usage(uid)
                if exports_watermark_for_tier(tier):
                    gif_path = OUTPUTS_DIR / job_id / "matte.gif"
                    wm = resolve_watermark_png(APP_ROOT)
                    if wm and gif_path.is_file():
                        await asyncio.to_thread(apply_png_watermark_to_gif, gif_path, wm)
            gif_path = OUTPUTS_DIR / job_id / "matte.gif"
            if gif_path.is_file() and _gif_frame_count(gif_path) <= 1:
                rebuilt = await asyncio.to_thread(_rebuild_transparent_gif_from_fg_alpha, OUTPUTS_DIR / job_id, gif_width)
                if rebuilt and uid and exports_watermark_for_tier(tier):
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


@app.post("/api/v1/matte/start")
async def matte_video_start(
    request: Request,
    file: UploadFile = File(..., description="Video file (MP4, MOV, etc.)"),
    gif_width: int = Query(960, ge=320, le=1280, description="GIF max width in px."),
    transparent_formats: str = Query("gif", description="Comma-separated: gif, webp, apng."),
    premium: bool = Query(False, description="True = heavy premium pass."),
    model: str = Query("rvm", pattern="^(rvm|rembg|pro)$"),
) -> JSONResponse:
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
    tier = "free"
    if uid:
        request.session.setdefault("plan_tier", "free")
        tier = plan_tier_from_session(dict(request.session))
        cap = gif_limit_for_tier(tier)
        if cap is not None:
            used = read_owner_usage_count(uid)
            if used >= cap:
                raise HTTPException(
                    status_code=403,
                    detail=f"GIF quota reached ({cap} for your plan). Upgrade at /subscription",
                )

    no_premium = not premium
    effective_formats = "gif" if no_premium else transparent_formats
    _set_job_state(job_id, status="queued", progress=0, message="Queued")
    asyncio.create_task(
        _run_job_async(
            public_base=_public_base_url(request),
            job_id=job_id,
            raw=raw,
            filename=file.filename,
            gif_width=gif_width,
            effective_formats=effective_formats,
            premium=premium,
            model=model,
            no_premium=no_premium,
            uid=uid,
            tier=tier,
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
        est_progress, est_message = _estimate_job_progress(job_id)
        progress = max(progress, est_progress)
        if progress > int(state.get("progress") or 0):
            _set_job_state(job_id, progress=progress, message=est_message, status="running")
        message = est_message
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
) -> JSONResponse:
    """
    Returns JSON with **absolute URLs** for each generated asset (easy to use from mobile).

    Multipart field name: **file**.
    """
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
    tier = "free"
    if uid:
        request.session.setdefault("plan_tier", "free")
        tier = plan_tier_from_session(dict(request.session))
        cap = gif_limit_for_tier(tier)
        if cap is not None:
            used = read_owner_usage_count(uid)
            if used >= cap:
                raise HTTPException(
                    status_code=403,
                    detail=f"GIF quota reached ({cap} for your plan). Upgrade at /subscription",
                )

    no_premium = not premium
    effective_formats = "gif" if no_premium else transparent_formats

    async with PROCESS_SEMAPHORE:
        try:
            paths = await asyncio.to_thread(
                _process_to_job_dir,
                job_id,
                raw,
                file.filename,
                gif_width=gif_width,
                transparent_formats=effective_formats,
                no_premium=no_premium,
                model=model,
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

    if uid:
        try:
            write_job_owner(job_id, uid)
        except ValueError:
            pass
        increment_owner_usage(uid)
        if exports_watermark_for_tier(tier):
            gif_path = OUTPUTS_DIR / job_id / "matte.gif"
            wm = resolve_watermark_png(APP_ROOT)
            if wm and gif_path.is_file():
                await asyncio.to_thread(apply_png_watermark_to_gif, gif_path, wm)
    gif_path = OUTPUTS_DIR / job_id / "matte.gif"
    if gif_path.is_file() and _gif_frame_count(gif_path) <= 1:
        rebuilt = await asyncio.to_thread(_rebuild_transparent_gif_from_fg_alpha, OUTPUTS_DIR / job_id, gif_width)
        if rebuilt and uid and exports_watermark_for_tier(tier):
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


@app.get("/", response_class=HTMLResponse)
async def demo_root() -> str:
    return _DEMO_HTML


@app.get("/demo", response_class=HTMLResponse)
async def demo_page() -> str:
    return _DEMO_HTML
