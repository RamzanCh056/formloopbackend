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

RUN python3 -c "from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained(chr(39)+chr(90)+chr(104)+chr(101)+chr(110)+chr(103)+chr(80)+chr(101)+chr(110)+chr(103)+chr(55)+chr(47)+chr(66)+chr(105)+chr(82)+chr(101)+chr(102)+chr(78)+chr(101)+chr(116)+chr(39), trust_remote_code=True); print(chr(39)+chr(66)+chr(105)+chr(82)+chr(101)+chr(102)+chr(78)+chr(101)+chr(116)+chr(32)+chr(79)+chr(75)+chr(39))"

RUN python3 -c "from ultralytics import YOLO; YOLO(chr(39)+chr(121)+chr(111)+chr(108)+chr(111)+chr(118)+chr(56)+chr(120)+chr(45)+chr(115)+chr(101)+chr(103)+chr(46)+chr(112)+chr(116)+chr(39)); YOLO(chr(39)+chr(121)+chr(111)+chr(108)+chr(111)+chr(118)+chr(56)+chr(120)+chr(45)+chr(112)+chr(111)+chr(115)+chr(101)+chr(46)+chr(112)+chr(116)+chr(39)); print(chr(39)+chr(89)+chr(79)+chr(76)+chr(79)+chr(32)+chr(79)+chr(75)+chr(39))"

CMD ["python", "-u", "handler.py"]
