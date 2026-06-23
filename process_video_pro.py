#!/usr/bin/env python3
"""
RunPod pro pipeline - BiRefNet + YOLO union mask + YOLO bbox crop for white bg removal.
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

INFER_MAX_DIM = 1024
_BIREFNET_CACHE: dict[str, tuple[torch.nn.Module, torch.device]] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--gif", required=True)
    p.add_argument("--fg", default=None)
    p.add_argument("--alpha", default=None)
    p.add_argument("--gif-width", type=int, default=640)
    p.add_argument("--gif-fps", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dilation", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--rvm-downsample", type=float, default=0.4)
    p.add_argument("--no-rvm", action="store_true")
    p.add_argument("--no-yolo", action="store_true")
    return p.parse_args()


def _cuda_runtime_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    print("[CUDA] testing runtime...", flush=True)
    try:
        x = torch.zeros(256, 256, device="cuda")
        x = x * 1.001
        torch.cuda.synchronize()
        del x
        print("[CUDA] runtime OK", flush=True)
        return True
    except Exception as e:
        print(f"[CUDA] runtime FAILED: {e}", flush=True)
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

    def _transform(pil_img: Image.Image):
        # Pad-to-square (letterbox) instead of stretch, so non-square frames
        # (e.g. 9:16 portrait) aren't distorted before BiRefNet inference.
        w0, h0 = pil_img.size
        scale = side / max(w0, h0)
        nw, nh = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        resized = pil_img.resize((nw, nh), Image.Resampling.BILINEAR)
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        pad_x = (side - nw) // 2
        pad_y = (side - nh) // 2
        padded.paste(resized, (pad_x, pad_y))
        arr = np.asarray(padded, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr), (pad_x, pad_y, nw, nh)

    return _transform


def get_rvm_alpha(frame_bgr: np.ndarray, model: torch.nn.Module, device: torch.device, transform_img) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor, (pad_x, pad_y, nw, nh) = transform_img(pil)
    inp = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        pred = pred.sigmoid().cpu()
    alpha = (pred.squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if alpha.ndim != 2:
        alpha = alpha.reshape(alpha.shape[-2], alpha.shape[-1])
    # Crop out the letterbox padding before resizing back to the original aspect ratio.
    alpha = alpha[pad_y:pad_y + nh, pad_x:pad_x + nw]
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
    # Zero out RGB where alpha == 0 to prevent white matte in GIF
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    mask = alpha_u8 > 0
    rgba[..., 0][mask] = rgb[..., 0][mask]
    rgba[..., 1][mask] = rgb[..., 1][mask]
    rgba[..., 2][mask] = rgb[..., 2][mask]
    rgba[..., 3] = alpha_u8
    return np.ascontiguousarray(rgba)


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
    import tempfile
    import shutil
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rgba_frames:
        return
    fps_str = str(max(1, int(fps)))
    resize_f = getattr(Image, "LANCZOS", Image.Resampling.LANCZOS)
    tmp_dir = tempfile.mkdtemp(prefix="gif_frames_")
    try:
        oh, ow = rgba_frames[0].shape[:2]
        # Autocrop: find the bounding box of all non-transparent pixels
        # across all frames, then crop to that region with padding.
        _all_alpha = np.zeros((oh, ow), dtype=np.uint8)
        for _f in rgba_frames:
            _all_alpha = np.maximum(_all_alpha, _f[..., 3])
        _rows = np.any(_all_alpha > 10, axis=1)
        _cols = np.any(_all_alpha > 10, axis=0)
        if _rows.any() and _cols.any():
            _rmin, _rmax = np.where(_rows)[0][[0, -1]]
            _cmin, _cmax = np.where(_cols)[0][[0, -1]]
            _pad = 20
            _rmin = max(0, _rmin - _pad)
            _rmax = min(oh, _rmax + _pad)
            _cmin = max(0, _cmin - _pad)
            _cmax = min(ow, _cmax + _pad)
            rgba_frames = [_f[_rmin:_rmax, _cmin:_cmax] for _f in rgba_frames]
            oh, ow = rgba_frames[0].shape[:2]
        nh = max(1, int(round(oh * width / max(1, ow))))
        for i, rgba in enumerate(rgba_frames):
            img = Image.fromarray(rgba, "RGBA")
            img = img.resize((width, nh), resize_f)
            img.save(os.path.join(tmp_dir, f"frame_{i:05d}.png"), format="PNG")
        pattern = os.path.join(tmp_dir, "frame_%05d.png")
        palette = os.path.join(tmp_dir, "palette.png")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", fps_str, "-i", pattern,
            "-vf", "palettegen=max_colors=255:reserve_transparent=1:stats_mode=diff",
            palette,
        ], check=True, capture_output=True)
        subprocess.run([
            "ffmpeg", "-y", "-framerate", fps_str,
            "-i", pattern, "-i", palette,
            "-lavfi", "paletteuse=dither=none:diff_mode=rectangle",
            "-loop", "0",
            str(path),
        ], check=True, capture_output=True)
        print(f"[GIF] Saved {path} ({path.stat().st_size // 1024}KB, {len(rgba_frames)} frames)", flush=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        print(f"[GIF] ffmpeg error: {err[-400:]}", flush=True)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _writer(path: Path, fps: float, w: int, h: int, gray: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (w, h), isColor=(not gray))


def _get_birefnet(preferred_device: torch.device) -> tuple[torch.nn.Module, torch.device]:
    os.environ["TRANSFORMERS_CACHE"] = "/app/model_cache"
    os.environ["HF_HOME"] = "/app/model_cache"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/model_cache"
    key = str(preferred_device)
    cached = _BIREFNET_CACHE.get(key)
    if cached is not None:
        return cached
    _local = '/app/model_cache/birefnet_local'
    _model_path = _local if os.path.isdir(_local) else 'ZhengPeng7/BiRefNet-matting'
    print(f'[BiRefNet] loading from {_model_path}', flush=True)
    model = AutoModelForImageSegmentation.from_pretrained(
        _model_path, trust_remote_code=True,
    )
    model = model.float()
    model.eval()
    import torch
    if torch.cuda.is_available():
        actual_device = torch.device("cuda")
        model = model.cuda()
        print(f"[BiRefNet] FORCED to CUDA, dtype=float32", flush=True)
    else:
        actual_device = torch.device("cpu")
        model = model.to(actual_device)
        print(f"[BiRefNet] CUDA not available, using CPU", flush=True)
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

    # Reuse cached YOLO model if already loaded
    if not hasattr(run_pipeline, "_yolo_model"):
        _yolo_pt = "/app/yolov8n-seg.pt" if os.path.exists("/app/yolov8n-seg.pt") else "yolov8n-seg.pt"
        run_pipeline._yolo_model = YOLO(_yolo_pt)
        run_pipeline._yolo_model.overrides["conf"] = 0.15
    yolo_model = run_pipeline._yolo_model

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
    last_rvm_alpha: np.ndarray | None = None
    last_yolo_mask_small: np.ndarray | None = None
    last_yolo_results = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            frame_small, _scale = resize_for_inference(frame)

            # BiRefNet alpha
            if frame_idx % 3 == 1 or last_rvm_alpha is None:
                rvm_alpha_small = get_rvm_alpha(frame_small, birefnet, device, transform_img)
                rvm_alpha = cv2.resize(rvm_alpha_small, (ow, oh), interpolation=cv2.INTER_LINEAR)
                if last_rvm_alpha is not None:
                    # EMA blend with the previous mask to reduce flicker between
                    # the every-3rd-frame BiRefNet re-inference.
                    rvm_alpha = cv2.addWeighted(
                        rvm_alpha.astype(np.float32), 0.7,
                        last_rvm_alpha.astype(np.float32), 0.3, 0,
                    ).astype(np.uint8)
                last_rvm_alpha = rvm_alpha.copy()
            else:
                rvm_alpha = last_rvm_alpha

            # YOLO every 5 frames
            if frame_idx % 5 == 0 or last_yolo_mask_small is None:
                yolo_results = yolo_model(frame_small, verbose=False)
                last_yolo_results = yolo_results
                yolo_mask_small = np.zeros((frame_small.shape[0], frame_small.shape[1]), dtype=np.float32)
                for r in yolo_results:
                    if r.masks is not None:
                        for m in r.masks.data:
                            mask = m.cpu().numpy()
                            mask = cv2.resize(mask, (frame_small.shape[1], frame_small.shape[0]), interpolation=cv2.INTER_LINEAR)
                            yolo_mask_small = np.maximum(yolo_mask_small, mask * 255.0)
                yolo_mask_small = np.clip(yolo_mask_small, 0, 255)
                last_yolo_mask_small = yolo_mask_small
            else:
                yolo_mask_small = last_yolo_mask_small
                yolo_results = last_yolo_results

            yolo_mask = cv2.resize(yolo_mask_small, (ow, oh), interpolation=cv2.INTER_LINEAR)

            # Union of BiRefNet + YOLO
            final_mask = np.maximum(rvm_alpha.astype(np.float32), yolo_mask)
            final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

            # Light unsharp pass to recover edge definition lost to the blur above
            _sharpened = cv2.addWeighted(final_mask, 1.5, cv2.GaussianBlur(final_mask, (3, 3), 0), -0.5, 0)
            final_mask = np.clip(_sharpened, 0, 255).astype(np.uint8)

            # Levels stretch — push semi-transparent edge pixels toward a hard cutout
            final_mask = np.clip((final_mask.astype(np.float32) - 40) * 1.5, 0, 255).astype(np.uint8)

            # Hard threshold — kill remaining soft fringe pixels (fully transparent, not white)
            final_mask = np.where(final_mask > 20, final_mask, 0).astype(np.uint8)

            # Use YOLO segmentation mask dilated as the crop boundary.
            # This traces the exact person shape instead of a rectangle.
            if yolo_mask.max() > 10:
                _seg = (yolo_mask > 10).astype(np.uint8) * 255
                _dk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
                _seg_dilated = cv2.dilate(_seg, _dk2)
                final_mask[_seg_dilated == 0] = 0

            rgba = _rgba_from_bgr_and_alpha(frame, final_mask)

            if fg_writer is not None:
                m3 = (final_mask.astype(np.float32) / 255.0)[..., None]
                fg = np.clip(frame.astype(np.float32) * m3, 0, 255).astype(np.uint8)
                fg_writer.write(fg)
            if a_writer is not None:
                a_writer.write(cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))

            if frame_idx % 2 == 0:
                gif_frames.append(rgba)
    finally:
        cap.release()
        if fg_writer is not None:
            fg_writer.release()
        if a_writer is not None:
            a_writer.release()

    actual_gif_fps = max(1, min(24, int(getattr(args, 'gif_fps', 12))))
    max_gif = max(24, int(os.environ.get("RVM_PRO_GIF_MAX_FRAMES", "280")))
    gif_src = _decimate_frames_for_gif(
        gif_frames,
        video_fps=float(source_fps),
        gif_fps=actual_gif_fps,
        max_frames=max_gif,
    )
    frames_to_gif(gif_src, Path(args.gif).resolve(), actual_gif_fps, int(args.gif_width))
    print(f"[Pipeline] done in {time.time() - start_time:.2f}s", flush=True)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()