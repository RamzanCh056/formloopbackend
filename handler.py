import runpod
import os
import tempfile
import base64
import json
import shutil
import subprocess
import requests
from types import SimpleNamespace
import firebase_admin
from firebase_admin import credentials, storage

try:
    import torch
except Exception as e:
    pass

os.environ["TRANSFORMERS_CACHE"] = "/app/model_cache"
os.environ["HF_HOME"] = "/app/model_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/model_cache"
os.environ["YOLO_CONFIG_DIR"] = "/app/yolo_cache"

# Initialize Firebase
firebase_config_str = os.environ.get("FIREBASE_CONFIG", "{}")
firebase_bucket = os.environ.get("FIREBASE_BUCKET", "")

if firebase_config_str and firebase_config_str != "{}" and not firebase_admin._apps:
    try:
        firebase_config = json.loads(firebase_config_str)
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {
            'storageBucket': firebase_bucket
        })
    except Exception as e:
        pass

# Pre-warm models at worker startup so first job is fast
import process_video_pro as _pvp
import torch as _torch

_PREWARM_DEVICE = os.environ.get("RVM_DEVICE", "auto")
try:
    print("[Worker] Pre-warming BiRefNet model...", flush=True)
    _dev = _pvp.pick_device(_PREWARM_DEVICE)
    _pvp._get_birefnet(_dev)
    print("[Worker] BiRefNet ready.", flush=True)
except Exception as _e:
    print(f"[Worker] Pre-warm failed (will load on first job): {_e}", flush=True)

try:
    print("[Worker] Pre-warming YOLO model...", flush=True)
    from ultralytics import YOLO as _YOLO
    _yolo_model = _YOLO("yolov8n-seg.pt")
    _yolo_model.overrides["conf"] = 0.15
    print("[Worker] YOLO ready.", flush=True)
except Exception as _e:
    print(f"[Worker] YOLO pre-warm failed: {_e}", flush=True)

try:
    if _pvp.SAM2_AVAILABLE:
        print("[Worker] Pre-warming SAM2...", flush=True)
        _sam2_dev = _dev if '_dev' in dir() else _pvp.pick_device(_PREWARM_DEVICE)
        _pvp._load_sam2(_sam2_dev)
        print("[Worker] SAM2 ready.", flush=True)
except Exception as _e:
    print(f"[Worker] SAM2 pre-warm failed: {_e}", flush=True)

def upload_to_firebase(local_path, dest_path, content_type):
    """Upload a file to Firebase Storage and return a public https URL."""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(dest_path)
        blob.upload_from_filename(local_path, content_type=content_type)
        blob.make_public()
        url = blob.public_url
        return url
    except Exception as e:
        return None


def _mux_webm_alpha(fg_mp4, alpha_mp4, out_webm, fps=12):
    """VP9 + alpha WebM with realtime settings. `fps` must match the fps the
    fg/alpha mp4s were actually written at (process_video_pro.run_pipeline
    sets this on ns.out_fps) so the muxed WebM's duration matches the source
    clip instead of a stale hardcoded rate."""
    fps_str = str(fps)
    fc = (
        "[0:v]format=rgb24[rgb];"
        "[1:v]format=gray,extractplanes=y[am];"
        "[rgb][am]alphamerge,format=yuva420p[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-r",
        fps_str,
        "-i",
        str(fg_mp4),
        "-r",
        fps_str,
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
        "24",
        "-b:v",
        "0",
        "-deadline",
        "realtime",
        "-cpu-used",
        "8",
        str(out_webm),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("[WebM] mux timed out after 120s — skipping WebM", flush=True)
        return
    if r.returncode == 0 and os.path.isfile(out_webm) and os.path.getsize(out_webm) > 32:
        return
    last_err = (r.stderr or r.stdout or "")[-800:]
    raise RuntimeError(f"WebM mux failed: {last_err}")

def _apply_reverse_loop(gif_path: str) -> bool:
    """In-place: appends reversed GIF to create a boomerang (forward+reverse) loop."""
    try:
        out_path = gif_path + ".boom.gif"
        r = subprocess.run([
            "ffmpeg", "-y", "-i", gif_path,
            "-filter_complex",
            "[0:v]split[fwd][bwd];[bwd]reverse[rev];[fwd][rev]concat=n=2:v=1:a=0[cat];"
            "[cat]split[sp][sv];[sp]palettegen=reserve_transparent=1:transparency_color=000000[pal];"
            "[sv][pal]paletteuse=dither=none:diff_mode=rectangle",
            "-loop", "0", out_path,
        ], capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 32:
            os.replace(out_path, gif_path)
            return True
        print(f"[Reverse] failed: {(r.stderr or '')[-300:]}", flush=True)
        return False
    except Exception as e:
        print(f"[Reverse] failed: {e}", flush=True)
        return False


def handler(job):
    job_input = job["input"]

    video_b64     = job_input.get("video")
    video_url     = job_input.get("video_url")
    exercise_name = job_input.get("exercise_name", "exercise")
    gif_width     = max(320, min(1280, int(job_input.get("gif_width", 640))))
    gif_fps       = max(1, min(30, int(job_input.get("gif_fps", 12))))
    rotation      = int(job_input.get("rotation", 0))
    loop_style    = str(job_input.get("loop_style", "normal"))
    dilation      = job_input.get("dilation", 12)
    conf          = job_input.get("conf", 0.20)
    use_sam2      = bool(job_input.get("use_sam2", False))
    # True = BiRefNet-only (fast). False = full pipeline with YOLO pose/seg (slower, better props/hands).
    # Stability-first default: keep YOLO disabled unless explicitly allowed.
    force_fast = os.environ.get("RVM_FORCE_FAST_MODE", "0").strip().lower() not in {"0", "false", "no"}
    pro_fast_raw = job_input.get("pro_fast_mode")
    if pro_fast_raw is None:
        # Compatibility alias used by some clients.
        pro_fast_raw = job_input.get("fast_mode")
    if force_fast:
        pro_fast_mode = True
    elif pro_fast_raw is None:
        pro_fast_mode = os.environ.get("RVM_PRO_FAST_MODE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
    else:
        pro_fast_mode = bool(pro_fast_raw)

    if not video_b64 and not video_url:
        return {"error": "No video provided"}

    gif_white_raw = job_input.get("gif_white_bg")
    if gif_white_raw is None:
        gif_white_bg = os.environ.get("RVM_PRO_GIF_WHITE_BG", "0").strip().lower() not in {"0", "false", "no"}
    else:
        gif_white_bg = bool(gif_white_raw)

    tmp_dir  = tempfile.mkdtemp(dir="/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    vid_path = os.path.join(tmp_dir, f"{exercise_name}.mp4")
    gif_path = os.path.join(tmp_dir, f"{exercise_name}.gif")
    fg_path = os.path.join(tmp_dir, f"{exercise_name}_foreground.mp4")
    alpha_path = os.path.join(tmp_dir, f"{exercise_name}_alpha.mp4")
    webm_path = os.path.join(tmp_dir, f"{exercise_name}_transparent.webm")

    try:
        # Save video from either base64 or URL.
        if video_url:
            resp = requests.get(str(video_url), timeout=300)
            resp.raise_for_status()
            with open(vid_path, "wb") as f:
                f.write(resp.content)
        else:
            with open(vid_path, "wb") as f:
                f.write(base64.b64decode(video_b64))

        # Apply rotation before processing if requested
        if rotation in (90, 180, 270):
            vf_map = {90: "transpose=1", 180: "vflip,hflip", 270: "transpose=2"}
            rotated = vid_path + ".rot.mp4"
            rr = subprocess.run([
                "ffmpeg", "-y", "-i", vid_path, "-vf", vf_map[rotation],
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", rotated,
            ], capture_output=True, text=True, timeout=120)
            if rr.returncode == 0 and os.path.isfile(rotated) and os.path.getsize(rotated) > 32:
                os.replace(rotated, vid_path)
                print(f"[Job] rotation={rotation}° applied", flush=True)
            else:
                print(f"[Job] rotation ffmpeg error: {(rr.stderr or '')[-200:]}", flush=True)

        # Run pipeline in-process so the worker can reuse loaded models across jobs.
        mode = "SAM2 + BiRefNet" if use_sam2 else ("BiRefNet-only (fast)" if pro_fast_mode else "BiRefNet + YOLO")
        print(f"[Job] start processing: {mode}")
        ns = SimpleNamespace(
            input=vid_path,
            gif=gif_path,
            fg=fg_path,
            alpha=alpha_path,
            gif_width=gif_width,
            gif_fps=gif_fps,
            device=os.environ.get("RVM_DEVICE", "cuda"),
            dilation=int(dilation),
            conf=float(conf),
            rvm_downsample=0.4,
            no_rvm=False,
            no_yolo=bool(pro_fast_mode),
            use_sam2=use_sam2,
        )
        prev_wb = os.environ.get("RVM_PRO_GIF_WHITE_BG")
        try:
            os.environ["RVM_PRO_GIF_WHITE_BG"] = "1" if gif_white_bg else "0"
            _pvp.run_pipeline(ns)
        finally:
            if prev_wb is None:
                os.environ.pop("RVM_PRO_GIF_WHITE_BG", None)
            else:
                os.environ["RVM_PRO_GIF_WHITE_BG"] = prev_wb
        if not os.path.exists(gif_path):
            return {"error": "GIF not created by pipeline"}

        if loop_style == "reverse":
            print("[Job] applying reverse loop...", flush=True)
            _apply_reverse_loop(gif_path)

        webm_url = None
        if os.path.isfile(fg_path) and os.path.isfile(alpha_path):
            try:
                _mux_webm_alpha(fg_path, alpha_path, webm_path, fps=getattr(ns, "out_fps", 12))
                if os.path.isfile(webm_path) and os.path.getsize(webm_path) > 0:
                    webm_url = upload_to_firebase(
                        webm_path,
                        f"webms/{exercise_name}.webm",
                        "video/webm",
                    )
            except Exception:
                pass

        # Upload to Firebase
        gif_public_url = upload_to_firebase(
            gif_path,
            f"gifs/{exercise_name}.gif",
            "image/gif",
        )

        if gif_public_url:
            print("[Job] save success", flush=True)
            out = {
                "status": "success",
                "gif_url": gif_public_url,
                "exercise": exercise_name,
            }
            if webm_url:
                out["webm_url"] = webm_url
            return out
        # Fallback: return base64 GIF only
        with open(gif_path, "rb") as f:
            gif_b64 = base64.b64encode(f.read()).decode("utf-8")
        out = {
            "status": "success",
            "gif_b64": gif_b64,
            "warning": "Firebase failed, returning base64",
        }
        print("[Job] save failure: Firebase failed, returned base64", flush=True)
        if webm_url:
            out["webm_url"] = webm_url
        return out

    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
