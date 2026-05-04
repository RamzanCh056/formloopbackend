# GPU image for FormLoop / RVM: HTTP API (default) or RunPod Serverless (see handler.py).
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_api.txt requirements_inference.txt ./

# Base image supplies torch/torchvision; install API + inference stack (skip pinned torch from inference file).
RUN pip install --no-cache-dir -r requirements_api.txt && \
    pip install --no-cache-dir av tqdm pims opencv-python-headless ultralytics runpod

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Model weights: place rvm_resnet50.pth in build context (or mount at runtime).
# .dockerignore excludes large local caches; copy checkpoint in CI or: docker build --secret, etc.

EXPOSE 8000

# RunPod Pod / generic HTTP
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

# RunPod Serverless: override start command to:
#   python -u handler.py
