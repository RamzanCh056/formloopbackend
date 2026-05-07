#!/usr/bin/env python3
"""
Video matting: BiRefNet foreground alpha + YOLOv8x-pose wrists + YOLOv8x-seg held objects.

INSTALL:
    pip install transformers accelerate torch torchvision
    pip install ultralytics mediapipe opencv-python pillow imageio
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation

# ---------------------------------------------------------------------------
EXCLUDE = {
    "tv",
    "monitor",
    "laptop",
    "chair",
    "couch",
    "bed",
    "dining table",
    "refrigerator",
    "microwave",
    "potted plant",
    "person",
    "clock",
    "vase",
}

INFER_WIDTH = 512
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
    p.add_argument("--dilation", type=int, default=18, help="extra final mask dilation px")
    p.add_argument("--conf", type=float, default=0.20, help="YOLO default confidence (seg uses 0.08 in crops)")
    p.add_argument(
        "--rvm-downsample",
        type=float,
        default=0.4,
        help="kept for API/subprocess compatibility (RVM builds); ignored when using BiRefNet",
    )
    p.add_argument(
        "--no-rvm",
        action="store_true",
        help="kept for CLI compatibility; this pipeline uses BiRefNet (flag ignored)",
    )
    p.add_argument(
        "--no-yolo",
        action="store_true",
        help="fast path: skip YOLO (BiRefNet only). Omit for full quality (props/hands).",
    )
    return p.parse_args()


def _cuda_runtime_usable() -> bool:
    """True only if CUDA alloc + sync works (driver matches PyTorch CUDA build)."""
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
            print(
                "[WARN] CUDA unusable (driver / PyTorch CUDA mismatch); using CPU",
                flush=True,
            )
            return torch.device("cpu")
        return torch.device(pref)
    if pref != "auto":
        return torch.device(pref)
    if _cuda_runtime_usable():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize_infer_bgr(frame_bgr: np.ndarray, target_w: int = INFER_WIDTH) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w == target_w:
        return frame_bgr
    nh = max(1, int(round(h * (target_w / float(max(1, w))))))
    return cv2.resize(frame_bgr, (target_w, nh), interpolation=cv2.INTER_AREA)


def build_transform_img():
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _transform(pil_img: Image.Image) -> torch.Tensor:
        img = pil_img.resize((1024, 1024), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    return _transform


def get_alpha(frame_bgr: np.ndarray, model: torch.nn.Module, device: torch.device, transform_img) -> np.ndarray:
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


def get_wrists(frame_bgr: np.ndarray, pose_model) -> list[tuple[int, int]]:
    results = pose_model(frame_bgr, verbose=False)
    wrists: list[tuple[int, int]] = []
    if not results or results[0].keypoints is None:
        return wrists
    kpts = results[0].keypoints.xy.cpu().numpy()
    for person_kpts in kpts:
        for idx in (9, 10):
            if idx < len(person_kpts):
                x, y = float(person_kpts[idx, 0]), float(person_kpts[idx, 1])
                if x > 0 and y > 0:
                    wrists.append((int(round(x)), int(round(y))))
    return wrists


def get_held_mask(
    frame_bgr: np.ndarray,
    wrists: list[tuple[int, int]],
    frame_h: int,
    frame_w: int,
    seg_model,
) -> np.ndarray | None:
    if not wrists:
        return None
    combined = np.zeros((frame_h, frame_w), dtype=np.uint8)
    fh, fw = frame_h, frame_w
    for wx, wy in wrists:
        if wy < fh * 0.35:
            x1, y1 = max(0, wx - 150), max(0, wy - 250)
            x2, y2 = min(fw, wx + 150), min(fh, wy + 100)
        elif wy < fh * 0.65:
            x1, y1 = max(0, wx - 160), max(0, wy - 160)
            x2, y2 = min(fw, wx + 160), min(fh, wy + 160)
        else:
            x1, y1 = max(0, wx - 130), max(0, wy - 80)
            x2, y2 = min(fw, wx + 130), min(fh, wy + 180)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        res = seg_model(crop, conf=0.05, verbose=False)

        if not res or res[0].masks is None:
            continue

        names = seg_model.names
        ch, cw = crop.shape[:2]
        cls_tensor = res[0].boxes.cls.cpu().numpy().astype(int)
        for i, cls_id in enumerate(cls_tensor):
            nm_raw = names[cls_id] if isinstance(names, dict) else names[int(cls_id)]
            nm = str(nm_raw).lower().strip()
            if nm in EXCLUDE:
                continue
            m = cv2.resize(
                res[0].masks.data[i].cpu().numpy(),
                (cw, ch),
                interpolation=cv2.INTER_LINEAR,
            )
            full = np.zeros((frame_h, frame_w), dtype=np.float32)
            full[y1:y2, x1:x2] = m
            obj = (full > 0.3).astype(np.uint8) * 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (70, 70))
            obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k)
            combined = np.maximum(combined, obj)

    return combined if combined.any() else None


def combine(birefnet_alpha: np.ndarray, held_mask: np.ndarray | None) -> np.ndarray:
    hard = (birefnet_alpha > 127).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    total_px = hard.shape[0] * hard.shape[1]
    clean = np.zeros_like(hard)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > total_px * 0.008:
            clean[labels == i] = 255

    if held_mask is not None:
        hm = held_mask.astype(np.uint8)
        if hm.shape != clean.shape:
            hm = cv2.resize(hm, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_NEAREST)
        k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        person_zone = cv2.dilate(clean, k_big)

        held_in_zone = np.where(person_zone > 0, hm, 0).astype(np.uint8)

        overlap = cv2.bitwise_and(held_in_zone, clean)
        if int(overlap.sum()) > 20 * 255:
            clean = np.maximum(clean, held_in_zone)

    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    clean = cv2.erode(clean, k_erode, iterations=1)
    return cv2.GaussianBlur(clean, (7, 7), 0)


class ObjectMemory:
    def __init__(self, ttl: int = 8) -> None:
        self.ttl = ttl
        self.slots: list[list] = []

    def update(self, mask: np.ndarray | None) -> None:
        self.slots = [[m, t - 1] for m, t in self.slots if t > 1]
        if mask is not None:
            self.slots.append([mask, self.ttl])

    def get(self) -> np.ndarray | None:
        if not self.slots:
            return None
        out = np.zeros_like(self.slots[0][0])
        for m, t in self.slots:
            out = np.maximum(out, (m.astype(np.float32) * (t / float(self.ttl))).clip(0, 255).astype(np.uint8))
        return out


class EMA:
    def __init__(self, a: float = 0.65) -> None:
        self.a = float(a)
        self.prev: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f = x.astype(np.float32)
        if self.prev is None:
            self.prev = f
            return x
        s = self.a * f + (1.0 - self.a) * self.prev
        s = np.maximum(s, f * 0.85)
        self.prev = s
        return np.clip(s, 0, 255).astype(np.uint8)


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
    """Keep ~duration×gif_fps frames so GIF time matches video without encoding every source frame."""
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
    q_method = getattr(Image.Quantize, "FASTOCTREE", Image.Quantize.MEDIANCUT)
    q_colors = max(32, min(255, int(os.environ.get("RVM_PRO_GIF_COLORS", "128"))))
    use_dither = (os.environ.get("RVM_PRO_GIF_DITHER", "0").strip().lower() not in {"0", "false", "no"})
    dither = Image.Dither.FLOYDSTEINBERG if use_dither else Image.Dither.NONE
    total = len(rgba_frames)
    for fi, rgba in enumerate(rgba_frames):
        if fi > 0 and fi % 80 == 0:
            print(f"[GIF] quantize {fi}/{total}", flush=True)
        img = Image.fromarray(rgba, "RGBA")
        ow, oh = img.size
        nh = max(1, int(round(oh * width / max(1, ow))))
        img = img.resize((width, nh), resize_f)
        r, g, b, a = img.split()
        rgb_p = (
            Image.merge("RGB", (r, g, b))
            .quantize(colors=q_colors, method=q_method, dither=dither)
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
    first.info["transparency"] = 255
    first.save(
        str(path),
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        disposal=2,
        transparency=255,
        optimize=False,
    )
    sz = path.stat().st_size / 1e6
    print(f"[GIF] Saved {path} ({sz:.1f}MB)", flush=True)


def _writer(path: Path, fps: float, w: int, h: int, gray: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (w, h), isColor=(not gray))


def _get_birefnet(preferred_device: torch.device) -> tuple[torch.nn.Module, torch.device]:
    key = str(preferred_device)
    cached = _BIREFNET_CACHE.get(key)
    if cached is not None:
        return cached
    print("[BiRefNet] loading ZhengPeng7/BiRefNet …", flush=True)
    try:
        # Keep checkpoint load on CPU; HF / remote code can otherwise touch CUDA during load
        # (progress hits 100% then fails) before our .to(device) fallback runs.
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                trust_remote_code=True,
                device_map="cpu",
            )
        except Exception as load_exc:
            if "device_map" not in str(load_exc).lower():
                raise
            print(
                "[WARN] BiRefNet device_map=cpu unavailable; retrying default load …",
                flush=True,
            )
            model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                trust_remote_code=True,
            )
        model = model.float()
        model.eval()
        actual_device = preferred_device
        try:
            model = model.to(preferred_device)
        except Exception as move_exc:
            if preferred_device.type != "cuda":
                raise SystemExit(f"BiRefNet load failed: {move_exc}") from move_exc
            print(
                f"[WARN] BiRefNet could not use CUDA ({move_exc}); using CPU",
                flush=True,
            )
            actual_device = torch.device("cpu")
            model = model.to(actual_device)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"BiRefNet load failed: {exc}") from exc
    _BIREFNET_CACHE[key] = (model, actual_device)
    return model, actual_device


def run_pipeline(args: argparse.Namespace) -> None:
    inp = Path(args.input).resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    requested_device = pick_device(args.device)
    birefnet, device = _get_birefnet(requested_device)
    print(f"[INFO] device={device}", flush=True)

    transform_img = build_transform_img()

    pose_model = None
    seg_model = None
    if not args.no_yolo:
        try:
            from ultralytics import YOLO

            print("[YOLO] loading yolov8n-pose.pt …", flush=True)
            pose_model = YOLO("yolov8n-pose.pt")
            pose_model.overrides["conf"] = float(args.conf)
            print("[YOLO] loading yolov8n-seg.pt …", flush=True)
            seg_model = YOLO("yolov8n-seg.pt")
            seg_model.overrides["conf"] = float(args.conf)
        except Exception as exc:
            print(f"[YOLO] disabled (load failed): {exc}", flush=True)
            args.no_yolo = True

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {inp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {ow}x{oh} @ {fps:.2f}fps, frames={n}", flush=True)

    fg_writer = _writer(Path(args.fg).resolve(), fps, ow, oh, gray=False) if args.fg else None
    a_writer = _writer(Path(args.alpha).resolve(), fps, ow, oh, gray=True) if args.alpha else None

    obj_mem = ObjectMemory(ttl=6)
    ema = EMA(a=0.65)
    gif_rgba: list[np.ndarray] = []

    last_biref_alpha: np.ndarray | None = None
    last_wrists: list[tuple[int, int]] | None = None

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % 10 == 0:
                print(f"[INFO] frame {idx}/{n}", flush=True)

            inf = resize_infer_bgr(frame, INFER_WIDTH)
            ih, iw = inf.shape[:2]

            if last_biref_alpha is None or idx % 2 == 1:
                last_biref_alpha = get_alpha(inf, birefnet, device, transform_img)
            biref_inf = last_biref_alpha

            held_inf: np.ndarray | None = None
            if not args.no_yolo and pose_model is not None and seg_model is not None:
                if last_wrists is None or idx % 2 == 0:
                    last_wrists = get_wrists(inf, pose_model)
                if idx % 2 == 0:
                    held_inf = get_held_mask(inf, last_wrists or [], ih, iw, seg_model)
                obj_mem.update(held_inf)
                held_fused = obj_mem.get()
            else:
                held_fused = None

            clean_inf = combine(biref_inf, held_fused)
            alpha_full = cv2.resize(clean_inf, (ow, oh), interpolation=cv2.INTER_LINEAR)

            alpha_f = alpha_full.astype(np.float32)
            if int(args.dilation) > 0:
                dk = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (int(args.dilation), int(args.dilation)),
                )
                alpha_f = cv2.dilate(alpha_f, dk, iterations=1)

            alpha_u8 = np.clip(alpha_f, 0, 255).astype(np.uint8)
            alpha_u8 = ema(alpha_u8)
            rgba = _rgba_from_bgr_and_alpha(frame, alpha_u8)
            gif_rgba.append(rgba)

            if fg_writer is not None:
                fg = frame.astype(np.float32)
                m3 = (alpha_u8.astype(np.float32) / 255.0)[..., None]
                fg = np.clip(fg * m3, 0, 255).astype(np.uint8)
                fg_writer.write(fg)
        if a_writer is not None:
            a_writer.write(alpha_u8)
    finally:
        cap.release()
        if fg_writer is not None:
            fg_writer.release()
        if a_writer is not None:
            a_writer.release()

    max_gif = max(24, int(os.environ.get("RVM_PRO_GIF_MAX_FRAMES", "480")))
    gif_src = _decimate_frames_for_gif(
        gif_rgba, video_fps=float(fps), gif_fps=int(args.gif_fps), max_frames=max_gif
    )
    print(
        f"[GIF] encoding {len(gif_src)} frames (video frames={len(gif_rgba)}, target_fps={args.gif_fps})",
        flush=True,
    )
    frames_to_gif(gif_src, Path(args.gif).resolve(), int(args.gif_fps), int(args.gif_width))
    print(f"[DONE] outputs under: {Path(args.gif).resolve().parent}", flush=True)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
