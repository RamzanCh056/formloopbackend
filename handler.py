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
import process_video_pro

try:
    import torch
    print(f"[Env] torch={torch.__version__}")
except Exception as e:
    print(f"[Env] torch import warning: {e}")

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
        print("[Firebase] Initialized successfully")
    except Exception as e:
        print(f"[Firebase] Init error: {e}")

def upload_to_firebase(local_path, dest_path, content_type):
    """Upload a file to Firebase Storage and return a public https URL."""
    try:
        bucket = storage.bucket()
        blob = bucket.blob(dest_path)
        blob.upload_from_filename(local_path, content_type=content_type)
        blob.make_public()
        url = blob.public_url
        print(f"[Firebase] Uploaded {dest_path}: {url}")
        return url
    except Exception as e:
        print(f"[Firebase] Upload error ({dest_path}): {e}")
        return None


def _mux_webm_alpha(fg_mp4, alpha_mp4, out_webm):
    """VP9 + alpha WebM; fall back to VP8 if VP9 encoder missing (some minimal FFmpeg builds)."""
    fc = (
        "[0:v]format=rgb24[rgb];"
        "[1:v]format=gray,extractplanes=y[am];"
        "[rgb][am]alphamerge,format=yuva420p[v]"
    )
    last_err = ""
    for enc_args in (
        [
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
        ],
        [
            "-c:v",
            "libvpx",
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
        ],
    ):
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(fg_mp4),
            "-i",
            str(alpha_mp4),
            "-filter_complex",
            fc,
            "-map",
            "[v]",
            *enc_args,
            str(out_webm),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.isfile(out_webm) and os.path.getsize(out_webm) > 32:
            return
        last_err = (r.stderr or r.stdout or "")[-800:]
    raise RuntimeError(f"WebM mux failed (vp9+vp8): {last_err}")

def handler(job):
    print(f"[Job] Starting: {job['id']}")
    job_input = job["input"]

    video_b64     = job_input.get("video")
    video_url     = job_input.get("video_url")
    exercise_name = job_input.get("exercise_name", "exercise")
    gif_width     = max(320, min(1280, int(job_input.get("gif_width", 640))))
    gif_fps       = max(1, min(30, int(job_input.get("gif_fps", 12))))
    dilation      = job_input.get("dilation", 12)
    conf          = job_input.get("conf", 0.20)
    # True = BiRefNet-only (fast). False = full pipeline with YOLO pose/seg (slower, better props/hands).
    # Stability-first default: keep YOLO disabled unless explicitly allowed.
    force_fast = os.environ.get("RVM_FORCE_FAST_MODE", "1").strip().lower() not in {"0", "false", "no"}
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

    tmp_dir  = tempfile.mkdtemp()
    vid_path = os.path.join(tmp_dir, f"{exercise_name}.mp4")
    gif_path = os.path.join(tmp_dir, f"{exercise_name}.gif")
    fg_path = os.path.join(tmp_dir, f"{exercise_name}_foreground.mp4")
    alpha_path = os.path.join(tmp_dir, f"{exercise_name}_alpha.mp4")
    webm_path = os.path.join(tmp_dir, f"{exercise_name}_transparent.webm")

    try:
        # Save video from either base64 or URL.
        if video_url:
            print("[Job] Downloading video URL...")
            resp = requests.get(str(video_url), timeout=300)
            resp.raise_for_status()
            with open(vid_path, "wb") as f:
                f.write(resp.content)
        else:
            print("[Job] Decoding video...")
            with open(vid_path, "wb") as f:
                f.write(base64.b64decode(video_b64))

        # Run pipeline in-process so the worker can reuse loaded models across jobs.
        mode = "BiRefNet-only (fast)" if pro_fast_mode else "BiRefNet + YOLO"
        print(f"[Job] Running pipeline: {mode}")
        ns = SimpleNamespace(
            input=vid_path,
            gif=gif_path,
            fg=fg_path,
            alpha=alpha_path,
            gif_width=gif_width,
            gif_fps=gif_fps,
            device=os.environ.get("RVM_DEVICE", "auto"),
            dilation=int(dilation),
            conf=float(conf),
            rvm_downsample=0.4,
            no_rvm=False,
            no_yolo=bool(pro_fast_mode),
        )
        prev_wb = os.environ.get("RVM_PRO_GIF_WHITE_BG")
        try:
            os.environ["RVM_PRO_GIF_WHITE_BG"] = "1" if gif_white_bg else "0"
            process_video_pro.run_pipeline(ns)
        finally:
            if prev_wb is None:
                os.environ.pop("RVM_PRO_GIF_WHITE_BG", None)
            else:
                os.environ["RVM_PRO_GIF_WHITE_BG"] = prev_wb

        webm_url = None
        if os.path.isfile(fg_path) and os.path.isfile(alpha_path):
            try:
                print("[Job] Muxing transparent WebM (VP9/VP8 + alpha)...")
                _mux_webm_alpha(fg_path, alpha_path, webm_path)
                if os.path.isfile(webm_path) and os.path.getsize(webm_path) > 0:
                    webm_url = upload_to_firebase(
                        webm_path,
                        f"webms/{exercise_name}.webm",
                        "video/webm",
                    )
            except Exception as mux_exc:
                print(f"[Job] WebM mux/upload skipped: {mux_exc}")

        # Upload to Firebase
        print("[Job] Uploading GIF to Firebase...")
        gif_public_url = upload_to_firebase(
            gif_path,
            f"gifs/{exercise_name}.gif",
            "image/gif",
        )

        if gif_public_url:
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
        if webm_url:
            out["webm_url"] = webm_url
        return out

    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
