import runpod
import subprocess
import os
import tempfile
import base64
import json
import requests
import firebase_admin
from firebase_admin import credentials, storage

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

def upload_to_firebase(gif_path, exercise_name):
    try:
        bucket = storage.bucket()
        blob = bucket.blob(f"gifs/{exercise_name}.gif")
        blob.upload_from_filename(gif_path, content_type="image/gif")
        blob.make_public()
        url = blob.public_url
        print(f"[Firebase] Uploaded: {url}")
        return url
    except Exception as e:
        print(f"[Firebase] Upload error: {e}")
        return None

def handler(job):
    print(f"[Job] Starting: {job['id']}")
    job_input = job["input"]

    video_b64     = job_input.get("video")
    video_url     = job_input.get("video_url")
    exercise_name = job_input.get("exercise_name", "exercise")
    gif_width     = job_input.get("gif_width", 640)
    gif_fps       = job_input.get("gif_fps", 12)
    dilation      = job_input.get("dilation", 18)
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

    tmp_dir  = tempfile.mkdtemp()
    vid_path = os.path.join(tmp_dir, f"{exercise_name}.mp4")
    gif_path = os.path.join(tmp_dir, f"{exercise_name}.gif")

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
        cmd = [
            "python", "process_video_pro.py",
            "--input",     vid_path,
            "--gif",       gif_path,
            "--gif-width", str(gif_width),
            "--gif-fps",   str(gif_fps),
            "--dilation",  str(dilation),
            "--conf",      str(conf),
            "--device",    "cuda",
        ]
        if pro_fast_mode:
            cmd.append("--no-yolo")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

        print(f"[Job] stdout: {result.stdout[-500:]}")

        if result.returncode != 0:
            print(f"[Job] Error: {result.stderr[-500:]}")
            return {"error": result.stderr[-500:]}

        # Upload to Firebase
        print("[Job] Uploading to Firebase...")
        public_url = upload_to_firebase(gif_path, exercise_name)

        if public_url:
            return {
                "status":   "success",
                "gif_url":  public_url,
                "exercise": exercise_name
            }
        else:
            # Fallback: return base64
            with open(gif_path, "rb") as f:
                gif_b64 = base64.b64encode(f.read()).decode("utf-8")
            return {
                "status":  "success",
                "gif_b64": gif_b64,
                "warning": "Firebase failed, returning base64"
            }

    except subprocess.TimeoutExpired:
        return {"error": "Timeout - video too long"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(vid_path): os.unlink(vid_path)
        if os.path.exists(gif_path): os.unlink(gif_path)
        try: os.rmdir(tmp_dir)
        except: pass

runpod.serverless.start({"handler": handler})
