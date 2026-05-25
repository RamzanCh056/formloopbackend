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
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import shutil as _shutil

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from ultralytics import YOLO

def _find_binary(name: str) -> str:
    import os
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    except ImportError:
        pass
    p = _shutil.which(name)
    if p:
        return p
    for path in [
        f"/usr/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/nix/var/nix/profiles/default/bin/{name}",
        f"/run/current-system/sw/bin/{name}",
    ]:
        if os.path.isfile(path):
            return path
    return name

_FFMPEG = _find_binary("ffmpeg")

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


def _biref_stride(no_yolo: bool) -> int:
    raw = (os.environ.get("RVM_PRO_BIREFNET_FRAME_STRIDE") or "").strip()
    if raw:
        return max(1, int(raw))
    return 6 if no_yolo else 2


def _infer_resize_w(no_yolo: bool) -> int:
    raw = (os.environ.get("RVM_PRO_INFER_WIDTH") or "").strip()
    if raw:
        return max(256, min(768, int(raw)))
    return 448 if no_yolo else INFER_WIDTH


def _birefnet_input_side(no_yolo: bool) -> int:
    raw = (os.environ.get("RVM_PRO_BIREFNET_SIZE") or "").strip()
    if raw:
        return max(512, min(1024, int(raw)))
    return 768 if no_yolo else 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input video path")
    p.add_argument("--gif", required=True, help="output GIF path")
    p.add_argument("--fg", default=None, help="output foreground MP4 (optional)")
    p.add_argument("--alpha", default=None, help="output alpha MP4 (optional)")
    p.add_argument("--webm", default=None, help="output transparent WebM (optional)")
    p.add_argument("--gif-width", type=int, default=640, help="GIF width in pixels")
    p.add_argument("--gif-fps", type=int, default=15, help="GIF fps")
    p.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    p.add_argument("--dilation", type=int, default=12, help="extra final mask dilation px (lower = tighter edge, less bg bleed)")
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


def build_transform_img(side: int = 1024) -> transforms.Compose:
    side = max(512, min(1024, int(side)))
    return transforms.Compose(
        [
            transforms.Resize((side, side)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


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


def get_wrists(frame_bgr: np.ndarray, pose_model: YOLO) -> list[tuple[int, int]]:
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
    seg_model: YOLO,
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
    opx = max(0, int(os.environ.get("RVM_PRO_MASK_OPEN_PX", "0")))
    if opx > 0:
        k0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * opx + 1, 2 * opx + 1))
        hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, k0)

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


def _alpha_fringe_gamma(alpha_u8: np.ndarray) -> np.ndarray:
    """Gamma < 1 slightly crushes semi-transparent fringe (cheap halo reduction)."""
    raw = (os.environ.get("RVM_PRO_ALPHA_FRINGE_GAMMA") or "").strip()
    if not raw:
        return alpha_u8
    g = float(raw)
    if abs(g - 1.0) < 1e-6:
        return alpha_u8
    g = max(0.55, min(1.05, g))
    a = np.clip((alpha_u8.astype(np.float32) / 255.0) ** g * 255.0, 0, 255).astype(np.uint8)
    return a


def _rgba_whiten_fringe_for_white_backdrop(rgba: np.ndarray) -> np.ndarray:
    """Keep full alpha; pull RGB toward white in semi-transparent edges so halos look neutral on white."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return rgba
    if os.environ.get("RVM_PRO_GIF_WHITEN_FOR_WHITE", "1").strip().lower() in {"0", "false", "no"}:
        return rgba
    strength = float(os.environ.get("RVM_PRO_GIF_WHITEN_STRENGTH", "0.9"))
    strength = max(0.0, min(1.0, strength))
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    t = strength * (1.0 - a)
    rgb = rgb + (255.0 - rgb) * t
    return np.concatenate(
        [np.clip(rgb, 0, 255).astype(np.uint8), rgba[..., 3:4]],
        axis=-1,
    )


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
    if not rgba_frames:
        return
    resize_f = getattr(Image, "LANCZOS", Image.Resampling.LANCZOS)
    with tempfile.TemporaryDirectory(prefix="gif_rgba_") as tmp:
        tmp_dir = Path(tmp)
        if rgba_frames:
            sample = rgba_frames[0]
            print(sample.shape, flush=True)
            print(sample.dtype, flush=True)
            print(np.unique(sample[..., 3])[:20], flush=True)
        for idx, rgba in enumerate(rgba_frames):
            img = Image.fromarray(rgba, "RGBA")
            print(img.mode, flush=True)
            ow, oh = img.size
            nh = max(1, int(round(oh * width / max(1, ow))))
            img = img.resize((width, nh), resize_f)
            img.save(tmp_dir / f"frame_{idx:05d}.png", format="PNG")

        pattern = str(tmp_dir / "frame_%05d.png")
        palette = str(tmp_dir / "palette.png")
        fps_str = str(max(1, int(fps)))

        cmd_palette = [
            _FFMPEG,
            "-y",
            "-framerate",
            fps_str,
            "-i",
            pattern,
            "-vf",
            "palettegen=reserve_transparent=1",
            palette,
        ]
        rp = subprocess.run(cmd_palette, capture_output=True, text=True)
        if rp.returncode != 0:
            raise RuntimeError((rp.stderr or rp.stdout or "palettegen failed")[-1200:])

        print(f"[GIF FPS] frames_to_gif: fps={fps} fps_str={fps_str}, n_frames={len(rgba_frames)}", flush=True)
        cmd_gif = [
            _FFMPEG,
            "-y",
            "-framerate",
            fps_str,
            "-i",
            pattern,
            "-i",
            palette,
            "-filter_complex",
            "paletteuse=alpha_threshold=128",
            "-loop",
            "0",
            str(path),
        ]
        rg = subprocess.run(cmd_gif, capture_output=True, text=True)
        if rg.returncode != 0:
            raise RuntimeError((rg.stderr or rg.stdout or "paletteuse failed")[-1200:])

    print(f"[GIF] Saved {path}", flush=True)


def _writer(path: Path, fps: float, w: int, h: int, gray: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), (w, h), isColor=(not gray))


def _mux_webm_alpha(fg_mp4: Path, alpha_mp4: Path, out_webm: Path) -> None:
    """Write transparent WebM using fg+alpha tracks; fallback VP8 if VP9 unavailable."""
    out_webm.parent.mkdir(parents=True, exist_ok=True)
    fc = (
        "[0:v]format=rgb24[rgb];"
        "[1:v]format=gray,extractplanes=y[am];"
        "[rgb][am]alphamerge,format=yuva420p[v]"
    )
    last_err = ""
    for enc_args in (
        [
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-auto-alt-ref",
            "0",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
        ],
        [
            "-c:v",
            "libvpx",
            "-pix_fmt",
            "yuva420p",
            "-auto-alt-ref",
            "0",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
        ],
    ):
        cmd = [
            _FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(fg_mp4),
            "-i",
            str(alpha_mp4),
            "-filter_complex",
            fc,
            "-map",
            "[v]",
            *enc_args,
            str(out_webm),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and out_webm.is_file() and out_webm.stat().st_size > 32:
            return
        last_err = (r.stderr or r.stdout or "")[-800:]
    raise RuntimeError(f"WebM mux failed (vp9+vp8): {last_err}")


def main() -> None:
    args = parse_args()
    inp = Path(args.input).resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    print(f"[GIF FPS] process_video_pro: using gif_fps={args.gif_fps}, gif_width={args.gif_width}", flush=True)
    device = pick_device(args.device)
    print(f"[INFO] device={device}", flush=True)

    print("[BiRefNet] loading ZhengPeng7/BiRefNet …", flush=True)
    try:
        # Keep checkpoint load on CPU; HF / remote code can otherwise touch CUDA during load
        # (progress hits 100% then fails) before our .to(device) fallback runs.
        try:
            birefnet = AutoModelForImageSegmentation.from_pretrained(
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
            birefnet = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                trust_remote_code=True,
            )
        # MPS/CUDA half weights vs float32 inputs → keep matting in float32 for stability
        birefnet = birefnet.float()
        birefnet.eval()
        try:
            birefnet = birefnet.to(device)
        except Exception as move_exc:
            if device.type != "cuda":
                raise SystemExit(f"BiRefNet load failed: {move_exc}") from move_exc
            print(
                f"[WARN] BiRefNet could not use CUDA ({move_exc}); using CPU",
                flush=True,
            )
            device = torch.device("cpu")
            birefnet = birefnet.to(device)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(f"BiRefNet load failed: {exc}") from exc

    biref_stride = _biref_stride(bool(args.no_yolo))
    infer_w = _infer_resize_w(bool(args.no_yolo))
    biref_side = _birefnet_input_side(bool(args.no_yolo))
    print(
        f"[SLA] BiRefNet stride={biref_stride} infer_w={infer_w} square={biref_side} no_yolo={args.no_yolo}",
        flush=True,
    )
    transform_img = build_transform_img(biref_side)

    _gif_white = os.environ.get("RVM_PRO_GIF_WHITE_BG", "0").strip().lower() not in {"0", "false", "no"}
    _whiten_edges = os.environ.get("RVM_PRO_GIF_WHITEN_FOR_WHITE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    _matte_shrink_px = max(
        0,
        int(
            os.environ.get(
                "RVM_PRO_MATTE_SHRINK_PX",
                "1" if (_gif_white or _whiten_edges) else "0",
            )
        ),
    )

    pose_model: YOLO | None = None
    seg_model: YOLO | None = None
    if not args.no_yolo:
        print("[YOLO] loading yolov8n-pose.pt …", flush=True)
        pose_model = YOLO("yolov8n-pose.pt")
        pose_model.overrides["conf"] = float(args.conf)
        print("[YOLO] loading yolov8n-seg.pt …", flush=True)
        seg_model = YOLO("yolov8n-seg.pt")
        seg_model.overrides["conf"] = float(args.conf)

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {inp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {ow}x{oh} @ {fps:.2f}fps, frames={n}", flush=True)

    fg_writer = _writer(Path(args.fg).resolve(), fps, ow, oh, gray=False) if args.fg else None
    a_writer = _writer(Path(args.alpha).resolve(), fps, ow, oh, gray=False) if args.alpha else None

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

            inf = resize_infer_bgr(frame, infer_w)
            ih, iw = inf.shape[:2]

            if last_biref_alpha is None or idx % biref_stride == 1:
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
            alpha_full = cv2.resize(
                clean_inf,
                (ow, oh),
                interpolation=getattr(cv2, "INTER_LANCZOS4", cv2.INTER_LINEAR),
            )

            alpha_f = alpha_full.astype(np.float32)
            if int(args.dilation) > 0:
                dk = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (int(args.dilation), int(args.dilation)),
                )
                alpha_f = cv2.dilate(alpha_f, dk, iterations=1)

            alpha_u8 = np.clip(alpha_f, 0, 255).astype(np.uint8)
            alpha_u8 = ema(alpha_u8)
            if _matte_shrink_px > 0:
                ks = max(3, _matte_shrink_px * 2 + 1)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
                alpha_u8 = cv2.erode(alpha_u8, k, iterations=1)
            alpha_u8 = _alpha_fringe_gamma(alpha_u8)
            rgba = _rgba_from_bgr_and_alpha(frame, alpha_u8)
            rgba = np.ascontiguousarray(rgba.astype(np.uint8))
            gif_rgba.append(rgba)

            if fg_writer is not None:
                wrgb = rgba[:, :, :3].astype(np.float32)
                a = rgba[:, :, 3:4].astype(np.float32) / 255.0
                premul = np.clip(wrgb * a, 0, 255).astype(np.uint8)
                fg = cv2.cvtColor(premul, cv2.COLOR_RGB2BGR)
                fg_writer.write(fg)
            if a_writer is not None:
                a_writer.write(cv2.cvtColor(alpha_u8, cv2.COLOR_GRAY2BGR))
    finally:
        cap.release()
        if fg_writer is not None:
            fg_writer.release()
        if a_writer is not None:
            a_writer.release()

    if args.webm:
        fg_mp4 = Path(args.fg).resolve() if args.fg else None
        alpha_mp4 = Path(args.alpha).resolve() if args.alpha else None
        webm_out = Path(args.webm).resolve()
        if fg_mp4 and alpha_mp4 and fg_mp4.is_file() and alpha_mp4.is_file():
            try:
                _mux_webm_alpha(fg_mp4, alpha_mp4, webm_out)
                print(f"[WEBM] Saved {webm_out}", flush=True)
            except Exception as exc:
                print(f"[WEBM] skip ({exc})", flush=True)

    _gif_max_default = "280" if args.no_yolo else "480"
    max_gif = max(24, int(os.environ.get("RVM_PRO_GIF_MAX_FRAMES", _gif_max_default)))
    gif_src = _decimate_frames_for_gif(
        gif_rgba, video_fps=float(fps), gif_fps=int(args.gif_fps), max_frames=max_gif
    )
    print(
        f"[GIF] encoding {len(gif_src)} frames (video frames={len(gif_rgba)}, target_fps={args.gif_fps})",
        flush=True,
    )
    print(f"[DEBUG] Total rgba_frames collected: {len(gif_rgba)}", flush=True)
    print(f"[DEBUG] GIF output path: {args.gif}", flush=True)
    print(f"[DEBUG] GIF dir exists: {os.path.exists(os.path.dirname(args.gif))}", flush=True)
    os.makedirs(os.path.dirname(args.gif), exist_ok=True)
    print(f"[GIF FPS] calling frames_to_gif: fps={args.gif_fps}, source_video_fps={fps:.2f}, total_gif_frames={len(gif_src)}", flush=True)
    frames_to_gif(gif_src, Path(args.gif).resolve(), int(args.gif_fps), int(args.gif_width))
    print(f"[DONE] outputs under: {Path(args.gif).resolve().parent}", flush=True)


if __name__ == "__main__":
    main()
