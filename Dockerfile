# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ARG CACHE_BUST=20260511_bake_models
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

RUN pip install --upgrade --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 \
    torchaudio==2.4.1

RUN pip uninstall -y torchvision || true

RUN pip install \
    transformers \
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
    timm

# Bake BiRefNet model into image — eliminates cold start download
RUN python3 -c "
import os
os.makedirs('/app/model_cache', exist_ok=True)
from transformers import AutoModelForImageSegmentation
model = AutoModelForImageSegmentation.from_pretrained(
    'ZhengPeng7/BiRefNet',
    trust_remote_code=True,
    device_map='cpu',
)
print('BiRefNet baked into image OK')
del model
"

# Bake YOLO model into image
RUN python3 -c "
import os
os.makedirs('/app/yolo_cache', exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = '/app/yolo_cache'
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
import shutil
for pt in ['yolov8n-seg.pt']:
    if os.path.exists(pt):
        shutil.copy(pt, '/app/' + pt)
print('YOLO baked into image OK')
"

COPY process_video_pro.py .
COPY handler.py .

CMD ["python", "-u", "handler.py"]
