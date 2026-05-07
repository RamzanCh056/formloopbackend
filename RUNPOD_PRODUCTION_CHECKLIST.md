# RunPod Production Checklist

This project is now configured for RunPod-first processing.

## 1) Backend env (web API)

Set these in the API host environment:

- `RVM_USE_RUNPOD=1`
- `RVM_RUNPOD_ONLY=1`
- `RUNPOD_API_KEY=<your_runpod_api_key>`
- `RUNPOD_ENDPOINT_ID=<your_endpoint_id>` (or `RUNPOD_ENDPOINT_URL`)
- `RVM_PUBLIC_BASE_URL=https://formloop.app`

Recommended:

- `RVM_PRO_FAST_MODE=1` (default fast mode; dashboard can override per request)
- `RUNPOD_JOB_TIMEOUT_SEC=900`
- `RUNPOD_STATUS_POLL_SEC=2.0`

## 2) RunPod worker env

Set these in the RunPod endpoint/worker:

- `FIREBASE_CONFIG=<service_account_json>`
- `FIREBASE_BUCKET=<bucket_name>`
- Optional: `RVM_PRO_FAST_MODE=1`

Worker files expected:

- `handler.py`
- `process_video_pro.py`
- `Dockerfile`

## 3) Dashboard request mode

Dashboard now sends:

- `model=pro`
- `fast_mode=true|false`
- `gif_width` and `gif_fps` based on mode toggle

Modes:

- Fast preview: width 640, fps 10
- High quality: width 960, fps 12

## 4) Verification steps

1. Open `/dashboard`
2. Upload a clip
3. Verify API response contains `job_id` and RunPod progress updates
4. Verify completed payload includes `gif_and_animation.transparent_gif`
5. Verify Save Export works and appears in `/dashboard/gifs`

## 5) Mobile integration notes

Use the async flow:

1. `POST /api/v1/matte/start` (multipart `file`)
2. Poll `GET /api/v1/matte/progress/{job_id}`
3. On `status=completed`, consume `result.gif_and_animation.transparent_gif`

If `status=failed`, show `error` from progress payload.
