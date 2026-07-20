"""
Parallel Modal deployment of the existing formloopbackend video pipeline.

This is a NEW file — it does not modify handler.py, process_video_pro.py,
Dockerfile, or api_server.py. Production stays on RunPod/Railway; this is a
side-by-side test target that pins a fixed GPU (never Blackwell) instead of
relying on RunPod's auto-assignment.

It reuses the existing, proven Dockerfile via modal.Image.from_dockerfile() —
same torch==2.5.1 / CUDA 12.1 / SAM2 / BiRefNet / YOLO setup, weights already
baked into the image at build time (see Dockerfile's "Bake ... into image"
steps), so no Modal Volume is needed for weights.

The actual pipeline (process_video_pro.run_pipeline) is imported and called
unmodified. Three small glue helpers that live inside handler.py
(upload_to_firebase, _mux_webm_alpha, _apply_reverse_loop) are duplicated here
verbatim, because handler.py cannot be imported directly — its last line
(`runpod.serverless.start(...)`) executes unconditionally at import time and
would try to start RunPod's polling loop inside this Modal container.
"""

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace

import modal

app = modal.App("formloop-modal-test")

image = modal.Image.from_dockerfile(
    "Dockerfile",
    context_dir=".",
)

firebase_secret = modal.Secret.from_name("formloop-firebase")

# Configurable so a slow VP9+alpha encode on a long/high-frame-count clip never
# gets silently truncated at a hardcoded 120s (handler.py's original value).
WEBM_MUX_TIMEOUT_SEC = int(os.environ.get("WEBM_MUX_TIMEOUT_SEC", "300"))


# ---------------------------------------------------------------------------
# Glue helpers — copied from handler.py verbatim (see module docstring for why
# this file can't just `import handler`). Keep these in sync with handler.py
# by hand if that file's helpers ever change; they are not re-exported from
# handler.py to avoid triggering its module-level runpod.serverless.start().
# ---------------------------------------------------------------------------

def upload_to_firebase(local_path, dest_path, content_type):
    """Upload a file to Firebase Storage and return a public https URL."""
    from firebase_admin import storage
    try:
        bucket = storage.bucket()
        blob = bucket.blob(dest_path)
        blob.upload_from_filename(local_path, content_type=content_type)
        blob.make_public()
        return blob.public_url
    except Exception:
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
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=WEBM_MUX_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        print(f"[WebM] mux timed out after {WEBM_MUX_TIMEOUT_SEC}s — skipping WebM", flush=True)
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


def _process(job_input: dict) -> dict:
    """Same contract as handler.py's handler(job): takes job["input"]-shaped
    dict, returns the same {"status"/"error", "gif_url"/"gif_b64", ...} shape.
    """
    # Extracted first, before any heavy import, so we can report "container is
    # warm and running your job" as early as physically possible — Modal's own
    # cold-start (pulling/booting the container) happens before this function
    # is even called and is invisible to us, but everything from here on is not.
    progress_callback_url = job_input.get("progress_callback_url")
    progress_token = job_input.get("progress_token")
    if progress_callback_url:
        # TEMPORARY TRACE (remove after diagnosing the smooth-progress-bar
        # regression): print the exact URL once so we can see its scheme —
        # root cause was an http:// callback URL getting 301-redirected to
        # https by Railway's edge, and `requests` downgrading POST->GET on
        # that redirect, which the (correctly POST-only) endpoint then 405'd.
        print(f"[Progress:url] {progress_callback_url}", flush=True)

    def _report_progress(stage: str, pct: int) -> None:
        """Best-effort, fire-and-forget: dispatched on a daemon thread so the much
        higher call frequency used for smooth sub-progress (every 1-2% of frames
        during SAM2 propagation) can never add network latency to the pipeline's
        hot loop. Delivery order isn't guaranteed once threaded — api_server's
        _bump_progress clamps upward so an out-of-order/late callback can only
        ever be a no-op, never a visible regression."""
        if not progress_callback_url:
            return
        # TEMPORARY TRACE (remove after diagnosing the smooth-progress-bar
        # regression): point #1 — call time, before the thread is even spawned.
        print(f"[Progress:call] stage={stage} pct={pct} t={time.time():.3f}", flush=True)

        def _send() -> None:
            try:
                # TEMPORARY TRACE: point #2 — inside the thread, right before POST.
                print(f"[Progress:send] stage={stage} pct={pct} t={time.time():.3f}", flush=True)
                import requests as _requests
                # allow_redirects=False: a POST must NEVER silently become a GET
                # via a followed 301/302 again. If the callback URL is ever
                # somehow non-canonical (e.g. wrong scheme) this now surfaces as
                # a loud logged 3xx instead of a quietly-broken callback.
                resp = _requests.post(
                    progress_callback_url,
                    json={"token": progress_token, "stage": stage, "pct": int(pct)},
                    timeout=3,
                    allow_redirects=False,
                )
                print(
                    f"[Progress:resp] stage={stage} pct={pct} status={resp.status_code} "
                    f"url={resp.url} history={resp.history}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[Progress:send:FAILED] stage={stage} pct={pct} error={exc!r}", flush=True)

        import threading
        threading.Thread(target=_send, daemon=True).start()

    # Container is warm and our code is running — this is the earliest possible
    # signal, closing the "stuck at a stale estimate" gap during Modal cold start.
    _report_progress("starting", 2)

    import firebase_admin
    from firebase_admin import credentials

    os.environ["TRANSFORMERS_CACHE"] = "/app/model_cache"
    os.environ["HF_HOME"] = "/app/model_cache"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/model_cache"
    os.environ["YOLO_CONFIG_DIR"] = "/app/yolo_cache"

    firebase_config_str = os.environ.get("FIREBASE_CONFIG", "{}")
    firebase_bucket = os.environ.get("FIREBASE_BUCKET", "")
    if firebase_config_str and firebase_config_str != "{}" and not firebase_admin._apps:
        try:
            firebase_config = json.loads(firebase_config_str)
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred, {"storageBucket": firebase_bucket})
        except Exception:
            pass

    import process_video_pro as _pvp

    # Heaviest imports (torch/transformers/sam2/ultralytics, all pulled in by
    # process_video_pro) are done — models are ready to load onto the GPU next.
    _report_progress("starting", 3)

    video_b64 = job_input.get("video")
    video_url = job_input.get("video_url")
    exercise_name = job_input.get("exercise_name", "exercise")
    gif_width = max(320, min(1280, int(job_input.get("gif_width", 640))))
    gif_fps = max(1, min(30, int(job_input.get("gif_fps", 12))))
    rotation = int(job_input.get("rotation", 0))
    loop_style = str(job_input.get("loop_style", "normal"))
    dilation = job_input.get("dilation", 12)
    conf = job_input.get("conf", 0.20)
    use_sam2 = bool(job_input.get("use_sam2", False))

    force_fast = os.environ.get("RVM_FORCE_FAST_MODE", "0").strip().lower() not in {"0", "false", "no"}
    pro_fast_raw = job_input.get("pro_fast_mode")
    if pro_fast_raw is None:
        pro_fast_raw = job_input.get("fast_mode")
    if force_fast:
        pro_fast_mode = True
    elif pro_fast_raw is None:
        pro_fast_mode = os.environ.get("RVM_PRO_FAST_MODE", "1").strip().lower() not in {"0", "false", "no"}
    else:
        pro_fast_mode = bool(pro_fast_raw)

    if not video_b64 and not video_url:
        return {"error": "No video provided"}

    gif_white_raw = job_input.get("gif_white_bg")
    if gif_white_raw is None:
        gif_white_bg = os.environ.get("RVM_PRO_GIF_WHITE_BG", "0").strip().lower() not in {"0", "false", "no"}
    else:
        gif_white_bg = bool(gif_white_raw)

    tmp_dir = tempfile.mkdtemp(dir="/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    vid_path = os.path.join(tmp_dir, f"{exercise_name}.mp4")
    gif_path = os.path.join(tmp_dir, f"{exercise_name}.gif")
    fg_path = os.path.join(tmp_dir, f"{exercise_name}_foreground.mp4")
    alpha_path = os.path.join(tmp_dir, f"{exercise_name}_alpha.mp4")
    webm_path = os.path.join(tmp_dir, f"{exercise_name}_transparent.webm")

    try:
        if video_url:
            import requests
            resp = requests.get(str(video_url), timeout=300)
            resp.raise_for_status()
            with open(vid_path, "wb") as f:
                f.write(resp.content)
        else:
            with open(vid_path, "wb") as f:
                f.write(base64.b64decode(video_b64))

        _report_progress("upload", 4)

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

        mode = "SAM2 + BiRefNet" if use_sam2 else ("BiRefNet-only (fast)" if pro_fast_mode else "BiRefNet + YOLO")
        print(f"[Job] start processing: {mode}", flush=True)
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
            progress_cb=_report_progress if progress_callback_url else None,
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

        # process_video_pro's own tail already reported "encoding" up to 92 —
        # no separate call needed here, it would just be clamped as a no-op.

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
                    webm_url = upload_to_firebase(webm_path, f"webms/{exercise_name}.webm", "video/webm")
                    if webm_url:
                        print(f"[WebM] mux+upload OK: {webm_url}", flush=True)
                    else:
                        print("[WebM] mux OK but Firebase upload failed (upload_to_firebase returned None)", flush=True)
            except Exception as exc:
                print(f"[WebM] mux/upload failed: {exc}", flush=True)
        else:
            print(
                f"[WebM] skipped — fg exists={os.path.isfile(fg_path)} alpha exists={os.path.isfile(alpha_path)}",
                flush=True,
            )
        _report_progress("finalizing", 94)

        gif_public_url = upload_to_firebase(gif_path, f"gifs/{exercise_name}.gif", "image/gif")
        # Capped below 99 — api_server's own post-return tail (99 "Downloading
        # results" -> 99 "Almost ready" -> 100 "Completed") owns the final
        # stretch, so this avoids a visible progress dip when control returns.
        _report_progress("finalizing", 96)

        if gif_public_url:
            print("[Job] save success", flush=True)
            out = {"status": "success", "gif_url": gif_public_url, "exercise": exercise_name}
            if webm_url:
                out["webm_url"] = webm_url
            return out

        with open(gif_path, "rb") as f:
            gif_b64 = base64.b64encode(f.read()).decode("utf-8")
        out = {"status": "success", "gif_b64": gif_b64, "warning": "Firebase failed, returning base64"}
        print("[Job] save failure: Firebase failed, returned base64", flush=True)
        if webm_url:
            out["webm_url"] = webm_url
        return out

    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# GPU pinned to A10 — A100 is the documented future upgrade (Ampere, 40GB,
# higher memory bandwidth, speeds up SAM2's memory-bound video-mask propagation
# without changing model/frames/fps) but stays blocked until a Modal payment
# method is on file. Once a card is added, change this to "A100" and redeploy.
# Never Blackwell; there is no "auto" here.
@app.function(image=image, gpu="A10", secrets=[firebase_secret], timeout=900)
@modal.fastapi_endpoint(method="POST")
def process_video(job_input: dict) -> dict:
    """HTTP endpoint. POST body is the same shape as RunPod's job["input"]."""
    return _process(job_input)
