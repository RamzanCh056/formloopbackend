#!/usr/bin/env python3
"""
RunPod pro pipeline rewrite:
- BiRefNet alpha + YOLOv8l segmentation union only.
- No connected components / cleanup / exclusions / erosion.
- YOLO every 3 frames with mask reuse for speed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from ultralytics import YOLO

INFER_MAX_DIM = 640
_BIREFNET_CACHE: dict[str, tuple[torch.nn.Module, torch.device]] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input video path")
    p.add_argument("--gif", required=True, help="output GIF path")
    p.add_argument("--fg", default=None, help="output foreground MP4 (optional)")
    p.add_argument("--alpha", default=None, help="output alpha MP4 (optional)")
    p.add_argument("--gif-width", type=int, default=640, help="GIF width in pixels")
    p.add_argument("--gif-fps", type=int, default=15, help="GIF fps")
    p.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    p.add_argument("--dilation", type=int, default=0, help="kept for compatibility; not used")
    p.add_argument("--conf", type=float, default=0.15, help="kept for compatibility; not used")
    p.add_argument("--rvm-downsample", type=float, default=0.4, help="kept for compatibility")
    p.add_argument("--no-rvm", action="store_true", help="kept for compatibility")
    p.add_argument("--no-yolo", action="store_true", help="kept for compatibility")
    return p.parse_args()


def _cuda_runtime_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(256, 256, device="cuda")
        x = x * 1.001
        torch.cuda.synchronize()
        del x
        return True
    except Exception:
        return False


def pick_device(pref: str) -> torch.device:
    if pref in ("cuda", "mps", "cpu"):
        if pref == "cuda" and not _cuda_runtime_usable():
            return torch.device("cpu")
        return torch.device(pref)
    if pref != "auto":
        return torch.device(pref)
    if _cuda_runtime_usable():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize_for_inference(frame_bgr: np.ndarray, max_dim: int = INFER_MAX_DIM) -> tuple[np.ndarray, float]:
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return frame_bgr, 1.0
    scale = min(float(max_dim) / float(max(h, w)), 1.0)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return resized, scale


def build_transform_img(side: int = 1024):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _transform(pil_img: Image.Image) -> torch.Tensor:
        img = pil_img.resize((side, side), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    return _transform


def get_rvm_alpha(frame_bgr: np.ndarray, model: torch.nn.Module, device: torch.device, transform_img) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inp = transform_img(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        pred = pred.sigmoid().cpu()
    alpha = (pred.squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if alpha.ndim != 2:
        alpha = alpha.reshape(alpha.shape[-2], alpha.shape[-1])
    return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)


def get_yolo_mask(frame_bgr_small: np.ndarray, yolo_model):
    height, width = frame_bgr_small.shape[:2]

    yolo_mask = np.zeros((height, width), dtype=np.float32)

    results = yolo_model(frame_bgr_small, verbose=False)

    for r in results:
        if r.masks is not None:
            for m in r.masks.data:
                mask = m.cpu().numpy()
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                yolo_mask = np.maximum(yolo_mask, mask * 255.0)

    return np.clip(yolo_mask, 0, 255)


def _rgba_from_bgr_and_alpha(frame_bgr: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return np.dstack([rgb, alpha_u8])


def _decimate_frames_for_gif(
    frames: list[np.ndarray],
    *,
    video_fps: float,
    gif_fps: int,
    max_frames: int,
) -> list[np.ndarray]:
    if len(frames) <= 2:
        return frames
    vf = max(0.001, float(video_fps))
    duration = len(frames) / vf
    target = min(max_frames, max(2, int(round(duration * max(1, gif_fps)))))
    n = len(frames)
    if n <= target:
        return frames
    idxs = np.linspace(0, n - 1, num=target, dtype=int)
    return [frames[int(i)] for i in idxs]


def frames_to_gif(rgba_frames: list[np.ndarray], path: str | Path, fps: int, width: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames: list[Image.Image] = []
    duration_ms = max(1, int(round(1000.0 / float(max(1, fps)))))
    resize_f = getattr(Image, "LANCZOS", Image.Resampling.LANCZOS)
    q_colors = max(32, min(255, int(os.environ.get("RVM_PRO_GIF_COLORS", "192"))))
    dither = Image.Dither.NONE
    white_bg = os.environ.get("RVM_PRO_GIF_WHITE_BG", "0").strip().lower() not in {"0", "false", "no"}
    for rgba in rgba_frames:
        img = Image.fromarray(rgba, "RGBA")
        ow, oh = img.size
        nh = max(1, int(round(oh * width / max(1, ow))))
        img = img.resize((width, nh), resize_f)
        if white_bg:
            base = Image.new("RGBA", img.size, (255, 255, 255, 255))
            flat = Image.alpha_composite(base, img).convert("RGB")
            pil_frames.append(flat.quantize(colors=q_colors, method=Image.Quantize.FASTOCTREE, dither=dither))
        else:
            r, g, b, a = img.split()
            rgb_p = (
                Image.merge("RGB", (r, g, b))
                .quantize(colors=q_colors, method=Image.Quantize.FASTOCTREE, dither=dither)
                .convert("P")
            )
            arr = np.array(rgb_p)
            arr[np.array(a) < 128] = 255
            frame_p = Image.fromarray(arr, "P")
            frame_p.putpalette(rgb_p.getpalette())
            pil_frames.append(frame_p)
    if not pil_frames:
        return
    first = pil_frames[0]
    save_kw: dict = {
        "format": "GIF",
        "save_all": True,
        "append_images": pil_frames[1:],
        "loop": 0,
        "duration": duration_ms,
        "optimize": False,
    }
    if white_bg:
        first.save(str(path), **save_kw)
    else:
        first.info["transparency"] = 255
        first.save(str(path), disposal=2, transparency=255, **save_kw)
    print(f"[GIF] Saved {path}", flush=True)


def _writer(path: Path, fps: float, w: int, h: int, gray: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (w, h), isColor=(not gray))


def _get_birefnet(preferred_device: torch.device) -> tuple[torch.nn.Module, torch.device]:
    key = str(preferred_device)
    cached = _BIREFNET_CACHE.get(key)
    if cached is not None:
        return cached
    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True,
        device_map="cpu",
    )
    model = model.float()
    model.eval()
    actual_device = preferred_device
    try:
        model = model.to(preferred_device)
    except Exception:
        actual_device = torch.device("cpu")
        model = model.to(actual_device)
    _BIREFNET_CACHE[key] = (model, actual_device)
    return model, actual_device


def run_pipeline(args: argparse.Namespace) -> None:
    start_time = time.time()
    print("[Pipeline] start processing", flush=True)
    inp = Path(args.input).resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    requested_device = pick_device(args.device)
    birefnet, device = _get_birefnet(requested_device)
    transform_img = build_transform_img(1024)

    # Use lighter YOLO model for serverless performance.
    yolo_model = YOLO("yolov8n-seg.pt")
    yolo_model.overrides["conf"] = 0.15

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {inp}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = 12.0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fg_writer = _writer(Path(args.fg).resolve(), out_fps, ow, oh, gray=False) if args.fg else None
    a_writer = _writer(Path(args.alpha).resolve(), out_fps, ow, oh, gray=False) if args.alpha else None

    gif_frames: list[np.ndarray] = []
    last_yolo_mask_small: np.ndarray | None = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame_small, _scale = resize_for_inference(frame)

            # RVM alpha on resized frame, then scale back.
            rvm_alpha_small = get_rvm_alpha(frame_small, birefnet, device, transform_img)
            rvm_alpha = cv2.resize(rvm_alpha_small, (ow, oh), interpolation=cv2.INTER_LINEAR)

            # YOLO every 5 frames, reuse previous mask.
            if frame_idx % 5 == 0 or last_yolo_mask_small is None:
                yolo_mask_small = get_yolo_mask(frame_small, yolo_model)
                last_yolo_mask_small = yolo_mask_small
            else:
                yolo_mask_small = last_yolo_mask_small
            yolo_mask = cv2.resize(yolo_mask_small, (ow, oh), interpolation=cv2.INTER_LINEAR)

            # STEP 1 — union only (no cleanup / erosion / connected components).
            final_mask = np.maximum(rvm_alpha.astype(np.float32), yolo_mask)
            final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
            final_mask = final_mask.astype(np.uint8)

            rgba = _rgba_from_bgr_and_alpha(frame, final_mask)

            if fg_writer is not None:
                m3 = (final_mask.astype(np.float32) / 255.0)[..., None]
                fg = np.clip(frame.astype(np.float32) * m3, 0, 255).astype(np.uint8)
                fg_writer.write(fg)
            if a_writer is not None:
                a_writer.write(cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))

            # STEP 2.5 — skip GIF frames.
            if frame_idx % 2 == 0:
                gif_frames.append(rgba)
    finally:
        cap.release()
        if fg_writer is not None:
            fg_writer.release()
        if a_writer is not None:
            a_writer.release()

    actual_gif_fps = 12
    max_gif = max(24, int(os.environ.get("RVM_PRO_GIF_MAX_FRAMES", "280")))
    gif_src = _decimate_frames_for_gif(
        gif_frames,
        video_fps=float(source_fps),
        gif_fps=actual_gif_fps,
        max_frames=max_gif,
    )
    frames_to_gif(gif_src, Path(args.gif).resolve(), actual_gif_fps, int(args.gif_width))
    print(f"[Pipeline] completed processing: {Path(args.gif).resolve().parent}", flush=True)
    print(f"TOTAL PROCESS TIME: {time.time()-start_time:.2f}s", flush=True)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
