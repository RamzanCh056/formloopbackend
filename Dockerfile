# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

ARG CACHE_BUST=20260512_local_birefnet
RUN echo "Cache bust: $CACHE_BUST"

# This base image's pytorch/torchvision/torchaudio/cudnn are conda packages
# living under /opt/conda (verified against PyTorch's own official Dockerfile
# source for this release: conda install -c pytorch-nightly -c nvidia pytorch
# torchvision torchaudio pytorch-cuda=12.1 — cudnn9 comes in transitively via
# pytorch-cuda). Conda's cuDNN .so files live under /opt/conda/lib/. Our pip
# install below pulls its own separate cuDNN via nvidia-cudnn-cu12, installed
# under site-packages/nvidia/cudnn/lib/ — a different path. With both present
# on disk, the dynamic linker can resolve either one first, which is a
# concrete, verified (not speculative) route to the
# "cuDNN error: CUDNN_STATUS_NOT_INITIALIZED" failure we hit earlier. Remove
# conda's copies first so there is exactly one torch+cuDNN stack on disk.
RUN conda remove -y pytorch torchvision torchaudio pytorch-cuda cudnn --force 2>/dev/null || true

# SAM2 (pinned to a specific commit below) requires torch>=2.5.1 at build time —
# newer than the base image's torch==2.4.1, so we can't just "protect" what's
# already there this time; we need a real upgrade. torch==2.5.1 still publishes
# a cu121-tagged wheel, so we stay on the same CUDA-12.1 target (and therefore
# the same older-driver RunPod host compatibility) as before — no need to move
# to a newer CUDA/driver floor.
#
# Install torch+torchvision+torchaudio together in ONE pip command so the
# resolver picks one mutually-consistent set of transitive nvidia-*-cu12
# (cuDNN/cuBLAS/etc.) dependencies for all three at once. A prior attempt
# installed torchvision separately from torch/torchaudio, in different layers,
# and reinstalling torch from a pip wheel here pulls its own separate cuDNN via
# nvidia-cudnn-cu12 — which conflicted with the base image's conda-installed
# cuDNN9, causing "cuDNN error: CUDNN_STATUS_NOT_INITIALIZED". Installing all
# three together removes one plausible source of that mismatch, but is not a
# guaranteed fix — if CUDNN_STATUS_NOT_INITIALIZED recurs, the next step is to
# check whether the base image's conda cudnn lib dir is shadowing the
# pip-installed one via LD_LIBRARY_PATH.
RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Pin the constraint file to the same versions so nothing downstream
# (ultralytics, sam2, transformers) can silently drift them further.
RUN printf "torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\n" > /tmp/torch-constraints.txt
ENV PIP_CONSTRAINT=/tmp/torch-constraints.txt

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

# Install SAM2 last, pinned to the exact commit that requires torch>=2.5.1 —
# pinning the commit (not just the branch) means a future upstream change to
# SAM2's own torch requirement can't silently break this build again the same
# way. PIP_CONSTRAINT (set above) still blocks pip from acting on SAM2's
# declared torch constraint by installing/upgrading torch further; it can only
# proceed if the pinned torch==2.5.1 already satisfies it (confirmed it does).
RUN pip install git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4

# Fail the build loudly if the constraint ever failed to hold (i.e. torch got
# upgraded anyway) or if the pinned version ever drifts, instead of silently
# shipping an image that falls back to CPU inference on older-driver hosts.
RUN python -c "import torch, torchvision; \
    assert torch.__version__.startswith('2.5.1'), torch.__version__; \
    assert torchvision.__version__.startswith('0.20.1'), torchvision.__version__; \
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
