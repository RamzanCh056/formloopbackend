# syntax=docker/dockerfile:1.4
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8-devel

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    ultralytics \
    mediapipe \
    opencv-python-headless \
    pillow \
    imageio \
    firebase-admin \
    runpod \
    huggingface_hub \
    scipy

COPY process_video_pro.py .
COPY handler.py .

RUN python3 -c "from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True); print('BiRefNet OK')"

RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x-seg.pt'); YOLO('yolov8x-pose.pt'); print('YOLO OK')"

CMD ["python", "-u", "handler.py"]
