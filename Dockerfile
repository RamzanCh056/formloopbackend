# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ARG CACHE_BUST=20260512_local_birefnet
RUN echo "Cache bust: $CACHE_BUST"

# Pin torch/torchvision/torchaudio to exactly what the base image already ships
# (conda-installed, matched to this image's cuDNN9) via a pip constraint file —
# NOT a reinstall. This blocks any downstream `pip install` (ultralytics, sam2,
# transformers, etc.) from silently upgrading/replacing them, without touching
# the base image's already-correct torch+cuDNN install. A prior attempt at this
# fix reinstalled torch from a pip wheel (download.pytorch.org/whl/cu121), which
# bundles its own separate cuDNN via nvidia-cudnn-cu12 and conflicted with the
# base image's conda cuDNN, causing "cuDNN error: CUDNN_STATUS_NOT_INITIALIZED".
RUN printf "torch==2.4.1\ntorchvision==0.19.1\ntorchaudio==2.4.1\n" > /tmp/torch-constraints.txt
ENV PIP_CONSTRAINT=/tmp/torch-constraints.txt

# PIP_CONSTRAINT only pins the version IF something triggers an install — it
# doesn't control which index a fresh install comes from. torch/torchaudio are
# already present via the base image's conda install, so nothing pulls them
# fresh. torchvision is NOT bundled in this base image, so ultralytics' install
# below would otherwise fetch it fresh from default PyPI — a build that can be
# ABI-incompatible with the base's conda torch ("operator torchvision::nms
# does not exist"). Install it explicitly, from the CUDA-matched index, first.
# Unlike torch's pip wheel, torchvision's wheel does not bundle its own
# separate cuDNN, so this does not reintroduce the CUDNN_STATUS_NOT_INITIALIZED
# conflict — only torch's pip wheel does that.
RUN pip install torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

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
    git \
    && rm -rf /var/lib/apt/lists/*

# Non-torch Python deps first. Do NOT list torch/torchaudio/torchvision here —
# they are pinned explicitly below, LAST, so nothing in this layer (or the sam2
# install below) can silently drag in a newer, incompatible CUDA build.
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
    hydra-core \
    iopath

# Install SAM2 last — its setup.py may declare its own torch version
# constraint, but PIP_CONSTRAINT (set above) blocks pip from acting on it by
# installing/upgrading torch; it can only proceed if the base image's existing
# torch==2.4.1 already satisfies whatever SAM2 requires.
RUN pip install git+https://github.com/facebookresearch/sam2.git

# Fail the build loudly if the constraint ever failed to hold (i.e. torch got
# upgraded anyway) or if the base image's version ever drifts, instead of
# silently shipping an image that falls back to CPU inference on
# older-driver hosts.
RUN python -c "import torch, torchvision; \
    assert torch.__version__.startswith('2.4.1'), torch.__version__; \
    assert torchvision.__version__.startswith('0.19.1'), torchvision.__version__; \
    print('torch OK:', torch.__version__, 'cuda tag:', torch.version.cuda)"

# Bake BiRefNet-matting model into image — eliminates cold start download
RUN python3 -c "import os; os.environ['HF_HOME']='/app/model_cache'; from huggingface_hub import snapshot_download; snapshot_download(repo_id='ZhengPeng7/BiRefNet-matting', local_dir='/app/model_cache/birefnet_local', ignore_patterns=['*.msgpack','flax_model*','tf_model*','rust_model*']); print('BiRefNet-matting snapshot downloaded OK')"

# Bake YOLO model into image
RUN python3 -c "import os; os.makedirs('/app/yolo_cache', exist_ok=True); os.environ['YOLO_CONFIG_DIR']='/app/yolo_cache'; from ultralytics import YOLO; m=YOLO('yolov8n-seg.pt'); print('YOLO baked OK')"

# Bake SAM2 checkpoint into image
RUN python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='facebook/sam2.1-hiera-large', \
local_dir='/app/model_cache/sam2_local'); print('SAM2 baked OK')"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY process_video_pro.py .
COPY handler.py .

CMD ["python", "-u", "handler.py"]
