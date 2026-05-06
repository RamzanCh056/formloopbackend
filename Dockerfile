# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
ARG CACHE_BUST=20260506_1743
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    RVM_FORCE_FAST_MODE=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 \
    torchvision==0.19.1

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
    timm && \
    python3 -c "print('cache bust:', '${CACHE_BUST}')"

COPY process_video_pro.py .
COPY handler.py .

# Keep build lightweight/stable for RunPod builder.
# Models download on first worker run and are cached per worker instance.

CMD ["python", "-u", "handler.py"]
