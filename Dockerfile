# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ARG CACHE_BUST=20260512_local_birefnet
RUN echo "Cache bust: $CACHE_BUST"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    RVM_FORCE_FAST_MODE=0 \
    RVM_PRO_GIF_WHITE_BG=0 \
    RVM_PRO_MATTE_SHRINK_PX=0 \
    RVM_PRO_GIF_WHITEN_FOR_WHITE=0 \
    TRANSFORMERS_CACHE=/app/model_cache \
    HF_HOME=/app/model_cache \
    YOLO_CONFIG_DIR=/app/yolo_cache

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    "transformers==4.45.0" \
    accelerate \
    ultralytics==8.3.0 \
    requests \
    opencv-python-headless \
    pillow \
    imageio \
    firebase-admin \
    runpod \
    huggingface_hub \
    scipy \
    einops \
    kornia \
    timm \
    torch \
    torchaudio

# Bake BiRefNet model into image — eliminates cold start download
RUN python3 -c "import os; os.environ['HF_HOME']='/app/model_cache'; from huggingface_hub import snapshot_download; snapshot_download(repo_id='ZhengPeng7/BiRefNet', local_dir='/app/model_cache/birefnet_local', ignore_patterns=['*.msgpack','flax_model*','tf_model*','rust_model*']); print('BiRefNet snapshot downloaded OK')"

# Bake YOLO model into image
RUN python3 -c "import os; os.makedirs('/app/yolo_cache', exist_ok=True); os.environ['YOLO_CONFIG_DIR']='/app/yolo_cache'; from ultralytics import YOLO; m=YOLO('yolov8n-seg.pt'); print('YOLO baked OK')"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY process_video_pro.py .
COPY handler.py .

CMD ["python", "-u", "handler.py"]
