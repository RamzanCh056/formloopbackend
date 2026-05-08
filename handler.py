import runpod
import subprocess
import os
import tempfile
import glob
import base64
import json
import shutil
import requests
import firebase_admin
from firebase_admin import credentials, storage

RVM_RUN_TIMEOUT_SEC = int(os.environ.get("RVM_RUN_TIMEOUT_SEC", "1800"))

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
    """Upload file to Firebase Storage; return public URL or None."""
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
    """VP9 + alpha WebM; fall back to VP8 if VP9 encoder missing."""
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
    gif_width     = job_input.get("gif_width", 640)
    gif_fps       = job_input.get("gif_fps", 12)
    dilation      = job_input.get("dilation", 12)
    conf          = job_input.get("conf", 0.20)
    # True = BiRefNet-only (fast). False = full pipeline with YOLO pose/seg (slower, better props/hands).
    pro_fast_raw = job_input.get("pro_fast_mode")
    if pro_fast_raw is None:
        pro_fast_mode = os.environ.get("RVM_PRO_FAST_MODE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
    else:
        pro_fast_mode = bool(pro_fast_raw)

    if not video_b64 and not video_url:
        return {"error": "No video provided"}

    tmp_dir = tempfile.mkdtemp(dir="/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
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

        # Run pipeline
        mode = "BiRefNet-only (fast)" if pro_fast_mode else "BiRefNet + YOLO"
        print(f"[Job] Running pipeline: {mode}")
        print(f"[DEBUG] Expected gif_path: {gif_path}")
        print(f"[DEBUG] tmp_dir: {tmp_dir}")
        cmd = [
            "python", "process_video_pro.py",
            "--input",     vid_path,
            "--gif",       gif_path,
            "--fg",        fg_path,
            "--alpha",     alpha_path,
            "--webm",      webm_path,
            "--gif-width", str(gif_width),
            "--gif-fps",   str(gif_fps),
            "--dilation",  str(dilation),
            "--conf",      str(conf),
            "--device",    os.environ.get("RVM_DEVICE", "auto"),
        ]
        if pro_fast_mode:
            cmd.append("--no-yolo")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=RVM_RUN_TIMEOUT_SEC)

        print(f"[Job] stdout: {result.stdout[-500:]}")
        print(f"[DEBUG] gif_path = {gif_path}")
        print(f"[DEBUG] gif exists = {os.path.exists(gif_path)}")
        print(f"[DEBUG] tmp_dir contents = {os.listdir(tmp_dir)}")
        all_files = glob.glob(f"{tmp_dir}/**/*", recursive=True)
        print(f"[DEBUG] All files after pipeline: {all_files}")

        if result.returncode != 0:
            print(f"[Job] Error: {result.stderr[-500:]}")
            return {"error": result.stderr[-500:]}

        if not os.path.exists(gif_path):
            gifs = glob.glob(f"{tmp_dir}/*.gif")
            print(f"[DEBUG] GIFs found in tmp_dir: {gifs}")
            if gifs:
                gif_path = gifs[0]
                print(f"[DEBUG] Using found gif: {gif_path}")
            else:
                return {"error": f"GIF not created. Pipeline stdout: {result.stdout[-1000:]}"}

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

        print("[Job] Uploading GIF to Firebase...")
        gif_public = upload_to_firebase(
            gif_path,
            f"gifs/{exercise_name}.gif",
            "image/gif",
        )

        if gif_public:
            out = {"status": "success", "gif_url": gif_public, "exercise": exercise_name}
            if webm_url:
                out["webm_url"] = webm_url
            return out
        with open(gif_path, "rb") as f:
            gif_b64 = base64.b64encode(f.read()).decode("utf-8")
        out = {"status": "success", "gif_b64": gif_b64, "warning": "Firebase failed, returning base64"}
        if webm_url:
            out["webm_url"] = webm_url
        return out

    except subprocess.TimeoutExpired:
        return {"error": f"Timeout after {RVM_RUN_TIMEOUT_SEC}s - video too long"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
