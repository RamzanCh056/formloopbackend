# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
ARG CACHE_BUST=20260506_1919
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    RVM_FORCE_FAST_MODE=1 \
    RVM_PRO_BIREFNET_FRAME_STRIDE=6 \
    RVM_PRO_BIREFNET_SIZE=768 \
    RVM_PRO_INFER_WIDTH=448 \
    RVM_PRO_GIF_MAX_FRAMES=280 \
    RVM_PRO_GIF_WHITE_BG=0 \
    RVM_PRO_GIF_WHITEN_FOR_WHITE=1 \
    RVM_PRO_MATTE_SHRINK_PX=1 \
    RVM_PRO_MASK_OPEN_PX=2 \
    RVM_PRO_ALPHA_FRINGE_GAMMA=0.88 \
    RVM_PRO_GIF_COLORS=192 \
    RVM_PRO_GIF_QUANTIZE=fast

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
    timm && \
    python3 -c "print('cache bust:', '${CACHE_BUST}')"

COPY process_video_pro.py .
COPY handler.py .

# Keep build lightweight/stable for RunPod builder.
# Models download on first worker run and are cached per worker instance.

CMD ["python", "-u", "handler.py"]
