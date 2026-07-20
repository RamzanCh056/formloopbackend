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

try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

INFER_MAX_DIM = 1024
_BIREFNET_CACHE: dict[str, tuple[torch.nn.Module, torch.device]] = {}
SAM2_CACHE: dict = {}


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
    p.add_argument("--use-sam2", action="store_true")
    return p.parse_args()


def _cuda_runtime_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    print("[CUDA] testing runtime...", flush=True)
    try:
        # torch.cuda.is_available() only confirms a driver + device exist — it
        # does NOT confirm this torch build has compiled kernels for the
        # device's compute capability. Newer GPU generations (e.g. Blackwell,
        # sm_120) on an older/pinned torch build pass is_available() but then
        # hard-crash on the first real kernel launch ("no kernel image is
        # available for execution on the device"). Check the capability
        # against what this build actually supports first, so we route
        # through the same CPU-fallback path as the other two cases instead
        # of letting torch attempt — and crash — a real kernel launch.
        cap_major, cap_minor = torch.cuda.get_device_capability()
        cap_str = f"sm_{cap_major}{cap_minor}"
        supported = torch.cuda.get_arch_list()
        if supported and cap_str not in supported:
            print(
                f"[CUDA] GPU compute capability {cap_str} not supported by this "
                f"torch build (supports: {', '.join(supported)}) — falling back "
                f"to BiRefNet+YOLO on CPU",
                flush=True,
            )
            return False
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
    duration: float,
    gif_fps: int,
    max_frames: int,
) -> list[np.ndarray]:
    if len(frames) <= 2:
        return frames
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
    # Use the same capability-aware check as pick_device()/_cuda_runtime_usable(),
    # not a raw torch.cuda.is_available() — that only confirms a driver+device
    # exist, not that this torch build has kernels for it (see _cuda_runtime_usable
    # for why: an unsupported architecture like Blackwell/sm_120 passes
    # is_available() but crashes on the first real kernel launch).
    if _cuda_runtime_usable():
        actual_device = torch.device("cuda")
        model = model.cuda()
        print(f"[BiRefNet] FORCED to CUDA, dtype=float32", flush=True)
    else:
        actual_device = torch.device("cpu")
        model = model.to(actual_device)
        print(f"[BiRefNet] CUDA not available, using CPU", flush=True)
    _BIREFNET_CACHE[key] = (model, actual_device)
    return model, actual_device


def _load_sam2(device: torch.device):
    if 'model' not in SAM2_CACHE:
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "/app/model_cache/sam2_local/sam2.1_hiera_large.pt",
            device=device,
        )
        SAM2_CACHE['model'] = predictor
    return SAM2_CACHE['model']


def get_sam2_mask(
    frames_bgr: list[np.ndarray],
    device: torch.device,
    yolo_box: tuple[float, float, float, float] | None = None,
    progress_cb=None,
) -> list[np.ndarray]:
    """Use the SAM2 video predictor to get a temporally consistent person mask
    across all frames. Prompts frame 0 with a YOLO box if available, otherwise
    a center point, then propagates through the rest of the clip.

    Returns a list of per-frame binary masks (0/255 uint8), one per input frame.
    """
    import tempfile
    import shutil

    predictor = _load_sam2(device)
    h, w = frames_bgr[0].shape[:2]
    tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    try:
        for i, frame in enumerate(frames_bgr):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(os.path.join(tmp_dir, f"{i:05d}.jpg"), quality=90)

        inference_state = predictor.init_state(video_path=tmp_dir)
        predictor.reset_state(inference_state)
        ann_frame_idx = 0
        ann_obj_id = 1

        if yolo_box is not None:
            box = np.array(yolo_box, dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=ann_frame_idx,
                obj_id=ann_obj_id, box=box,
            )
        else:
            points = np.array([[w / 2.0, h / 2.0]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=ann_frame_idx,
                obj_id=ann_obj_id, points=points, labels=labels,
            )

        masks: list[np.ndarray | None] = [None] * len(frames_bgr)
        total = max(1, len(frames_bgr))
        # Sub-progress within the "matting" stage (5-80%) — this loop is ~75% of
        # total pipeline time in SAM2 mode, so it owns the biggest slice of the
        # bar. Emit on every integer-percent change rather than a fixed frame
        # stride: self-adjusting to clip length, gives ~1 update per 1-2% of
        # frames for typical clips, and guarantees at least one update even for
        # very short clips (a handful of frames still crosses several percent
        # points).
        _last_pct = [4]

        def _emit(step: int) -> None:
            if progress_cb is None:
                return
            pct = 5 + int(75 * min(1.0, step / total))
            if pct > _last_pct[0]:
                _last_pct[0] = pct
                try:
                    progress_cb("matting", pct)
                except Exception:
                    pass

        for step, (out_frame_idx, _out_obj_ids, out_mask_logits) in enumerate(
            predictor.propagate_in_video(inference_state)
        ):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            if mask.ndim != 2:
                mask = mask.reshape(mask.shape[-2], mask.shape[-1])
            if 0 <= out_frame_idx < len(masks):
                masks[out_frame_idx] = (mask.astype(np.uint8) * 255)
            _emit(step)

        # Fill any frame SAM2 didn't propagate to with the nearest available mask.
        last = None
        for i in range(len(masks)):
            if masks[i] is None:
                masks[i] = last if last is not None else np.zeros((h, w), dtype=np.uint8)
            else:
                last = masks[i]
        return masks
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_birefnet_yolo_pass(
    frame_source,
    *,
    birefnet: torch.nn.Module,
    device: torch.device,
    transform_img,
    yolo_model,
    ow: int,
    oh: int,
    fg_writer,
    a_writer,
    progress_cb=None,
    total_frames: int | None = None,
) -> tuple[list[np.ndarray], int]:
    """BiRefNet alpha + YOLO union mask pipeline (the original/default pipeline,
    used when SAM2 is not requested or as a fallback if SAM2 fails).

    Returns (gif_frames, frames_processed) — frames_processed is the actual
    count of source frames consumed, used by the caller to compute the true
    source-clip duration (frame count from container metadata can be wrong)."""
    gif_frames: list[np.ndarray] = []
    last_rvm_alpha: np.ndarray | None = None
    last_yolo_mask_small: np.ndarray | None = None
    last_yolo_results = None
    frame_idx = 0
    # No separate compositing stage in this pipeline (matting and compositing
    # are the same loop), so this one loop spans the full 5-88% "matting" band —
    # same integer-percent-change technique as the SAM2 loops.
    _byp_total = max(1, int(total_frames) if total_frames else 0)
    _byp_last_pct = [4]

    def _emit_byp(step: int) -> None:
        if progress_cb is None or _byp_total <= 0:
            return
        pct = 5 + int(83 * min(1.0, step / _byp_total))
        if pct > _byp_last_pct[0]:
            _byp_last_pct[0] = pct
            try:
                progress_cb("matting", pct)
            except Exception:
                pass

    for frame in frame_source:
        _emit_byp(frame_idx)
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

    return gif_frames, frame_idx


def _run_sam2_pipeline(
    frames_buffer: list[np.ndarray],
    frames_small_buffer: list[np.ndarray],
    *,
    birefnet: torch.nn.Module,
    device: torch.device,
    transform_img,
    yolo_model,
    ow: int,
    oh: int,
    fg_writer,
    a_writer,
    progress_cb=None,
) -> tuple[list[np.ndarray], int]:
    """SAM2 + BiRefNet pipeline. SAM2 supplies a temporally-consistent person
    silhouette (used as the crop boundary in place of YOLO's union mask);
    BiRefNet still supplies the soft alpha edge detail. final = BiRefNet_alpha
    AND SAM2_mask (intersection), then the same edge-quality chain as the
    default pipeline (blur/unsharp/levels/threshold).

    Returns (gif_frames, frames_processed) — see _run_birefnet_yolo_pass.
    """
    if not frames_small_buffer:
        return [], 0

    # Run YOLO once on frame 0 only, to get a box prompt for SAM2.
    yolo_box: tuple[float, float, float, float] | None = None
    try:
        first_results = yolo_model(frames_small_buffer[0], verbose=False)
        for r in first_results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                yolo_box = tuple(boxes[int(np.argmax(areas))].tolist())
                break
    except Exception as e:
        print(f"[SAM2] YOLO first-frame box detection failed: {e}", flush=True)

    sam2_masks_small = get_sam2_mask(frames_small_buffer, device, yolo_box=yolo_box, progress_cb=progress_cb)

    gif_frames: list[np.ndarray] = []
    last_rvm_alpha: np.ndarray | None = None
    frame_idx = 0
    _comp_total = max(1, len(frames_buffer))
    _comp_last_pct = [79]

    def _emit_compositing(step: int) -> None:
        if progress_cb is None:
            return
        pct = 80 + int(8 * min(1.0, step / _comp_total))
        if pct > _comp_last_pct[0]:
            _comp_last_pct[0] = pct
            try:
                progress_cb("compositing", pct)
            except Exception:
                pass

    for i, frame in enumerate(frames_buffer):
        frame_idx += 1
        frame_small = frames_small_buffer[i]
        _emit_compositing(i)

        if frame_idx % 3 == 1 or last_rvm_alpha is None:
            rvm_alpha_small = get_rvm_alpha(frame_small, birefnet, device, transform_img)
            rvm_alpha = cv2.resize(rvm_alpha_small, (ow, oh), interpolation=cv2.INTER_LINEAR)
            if last_rvm_alpha is not None:
                rvm_alpha = cv2.addWeighted(
                    rvm_alpha.astype(np.float32), 0.7,
                    last_rvm_alpha.astype(np.float32), 0.3, 0,
                ).astype(np.uint8)
            last_rvm_alpha = rvm_alpha.copy()
        else:
            rvm_alpha = last_rvm_alpha

        sam2_mask_small = sam2_masks_small[i] if i < len(sam2_masks_small) else sam2_masks_small[-1]
        sam2_mask = cv2.resize(sam2_mask_small, (ow, oh), interpolation=cv2.INTER_NEAREST)

        # Blend instead of intersect: SAM2 confident regions use full BiRefNet
        # alpha (stable person silhouette); BiRefNet-only regions (e.g. held
        # equipment SAM2's person-only prompt missed) are kept at 80% so
        # dumbbells/objects don't get cut out just because SAM2 didn't track them.
        sam2_confident = sam2_mask > 0
        birefnet_confident = rvm_alpha > 100
        final_mask = np.where(
            sam2_confident,
            rvm_alpha,
            np.where(birefnet_confident, rvm_alpha * 0.8, 0),
        ).astype(np.float32)
        final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

        _sharpened = cv2.addWeighted(final_mask, 1.5, cv2.GaussianBlur(final_mask, (3, 3), 0), -0.5, 0)
        final_mask = np.clip(_sharpened, 0, 255).astype(np.uint8)

        # Lighter levels stretch than the default pipeline — SAM2 already gives
        # cleaner masks, so less aggressive stretching is needed to avoid
        # eating into edge detail.
        final_mask = np.clip((final_mask.astype(np.float32) - 30) * 1.4, 0, 255).astype(np.uint8)
        final_mask = np.where(final_mask > 20, final_mask, 0).astype(np.uint8)

        rgba = _rgba_from_bgr_and_alpha(frame, final_mask)

        if fg_writer is not None:
            m3 = (final_mask.astype(np.float32) / 255.0)[..., None]
            fg = np.clip(frame.astype(np.float32) * m3, 0, 255).astype(np.uint8)
            fg_writer.write(fg)
        if a_writer is not None:
            a_writer.write(cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR))

        if frame_idx % 2 == 0:
            gif_frames.append(rgba)

    return gif_frames, frame_idx


def run_pipeline(args: argparse.Namespace) -> None:
    start_time = time.time()
    print("[Pipeline] start processing", flush=True)
    # Additive, optional milestone-progress hook — only set by the Modal path
    # (modal_app.py). RunPod's handler.py never sets this, so getattr(..., None)
    # keeps RunPod behavior byte-for-byte unchanged.
    progress_cb = getattr(args, "progress_cb", None)

    def _report(stage: str, pct: int) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(stage, pct)
        except Exception:
            pass

    inp = Path(args.input).resolve()
    if not inp.is_file():
        raise SystemExit(f"input not found: {inp}")

    requested_device = pick_device(args.device)
    birefnet, device = _get_birefnet(requested_device)

    # Reuse cached YOLO model if already loaded
    if not hasattr(run_pipeline, "_yolo_model"):
        _yolo_pt = "/app/yolov8n-seg.pt" if os.path.exists("/app/yolov8n-seg.pt") else "yolov8n-seg.pt"
        run_pipeline._yolo_model = YOLO(_yolo_pt)
        run_pipeline._yolo_model.overrides["conf"] = 0.15
    yolo_model = run_pipeline._yolo_model

    use_sam2 = bool(getattr(args, "use_sam2", False))
    if use_sam2 and not SAM2_AVAILABLE:
        print("[SAM2] use_sam2 requested but sam2 package not installed — falling back to BiRefNet + YOLO", flush=True)
        use_sam2 = False
    if use_sam2 and device.type != "cuda":
        print("[SAM2] CUDA not available on this worker — falling back to BiRefNet+YOLO to avoid a CPU stall", flush=True)
        use_sam2 = False

    if use_sam2:
        infer_dim = 1280
        if int(getattr(args, "gif_width", 0) or 0) < 960:
            args.gif_width = 960
            print("[SAM2] Auto-upgrading gif_width to 960 for Ultra mode", flush=True)
    else:
        infer_dim = INFER_MAX_DIM
    transform_img = build_transform_img(infer_dim)

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {inp}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # fg/alpha writers keep every source frame 1:1 (no decimation below), so
    # writing them at the real source fps makes the resulting foreground/alpha
    # mp4s (and therefore the muxed WebM) play back at the true clip duration.
    # Previously hardcoded to 12.0, which desynced WebM duration from the
    # source whenever source_fps != 12 (see modal_app.py/handler.py's
    # _mux_webm_alpha, which mirrors this value via ns.out_fps).
    out_fps = float(source_fps)
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fg_writer = _writer(Path(args.fg).resolve(), out_fps, ow, oh, gray=False) if args.fg else None
    a_writer = _writer(Path(args.alpha).resolve(), out_fps, ow, oh, gray=False) if args.alpha else None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Read back by handler.py/modal_app.py after run_pipeline() returns, so the
    # WebM mux's ffmpeg -r matches what the fg/alpha mp4s were actually written at.
    args.out_fps = out_fps

    gif_frames: list[np.ndarray] = []
    try:
        if use_sam2:
            print("[Pipeline] mode: SAM2 + BiRefNet", flush=True)
            _report("matting", 5)
            # SAM2 needs the whole clip up front for video mask propagation.
            # These clips are a few seconds, so buffering them in memory is cheap.
            frames_buffer: list[np.ndarray] = []
            frames_small_buffer: list[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_buffer.append(frame)
                frame_small, _scale = resize_for_inference(frame, max_dim=infer_dim)
                frames_small_buffer.append(frame_small)
            cap.release()

            try:
                gif_frames, frames_processed = _run_sam2_pipeline(
                    frames_buffer, frames_small_buffer,
                    birefnet=birefnet, device=device, transform_img=transform_img,
                    yolo_model=yolo_model, ow=ow, oh=oh,
                    fg_writer=fg_writer, a_writer=a_writer,
                    progress_cb=progress_cb,
                )
            except Exception as e:
                print(f"[SAM2] pipeline failed, falling back to BiRefNet + YOLO: {e}", flush=True)
                gif_frames, frames_processed = _run_birefnet_yolo_pass(
                    iter(frames_buffer),
                    birefnet=birefnet, device=device, transform_img=transform_img,
                    yolo_model=yolo_model, ow=ow, oh=oh,
                    fg_writer=fg_writer, a_writer=a_writer,
                    progress_cb=progress_cb, total_frames=len(frames_buffer),
                )
        else:
            print("[Pipeline] mode: BiRefNet + YOLO", flush=True)
            _report("matting", 5)

            def _cap_frames():
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        return
                    yield frame

            gif_frames, frames_processed = _run_birefnet_yolo_pass(
                _cap_frames(),
                birefnet=birefnet, device=device, transform_img=transform_img,
                yolo_model=yolo_model, ow=ow, oh=oh,
                fg_writer=fg_writer, a_writer=a_writer,
                progress_cb=progress_cb, total_frames=total_frames,
            )
    finally:
        cap.release()
        if fg_writer is not None:
            fg_writer.release()
        if a_writer is not None:
            a_writer.release()

    # Safety-net floor — harmless no-op if the loop-driven sub-progress above
    # already reached 88 (the common case); guarantees this milestone even if a
    # very short/edge-case clip never crossed it, since _bump_progress on the
    # api_server side clamps upward.
    _report("compositing", 88)

    actual_gif_fps = max(1, min(24, int(getattr(args, 'gif_fps', 12))))
    max_gif = max(24, int(os.environ.get("RVM_PRO_GIF_MAX_FRAMES", "280")))
    # True source-clip duration, from the actual number of frames the matting
    # pass consumed (not container frame-count metadata, which can be wrong)
    # divided by real fps. gif_frames already reflects a 1-in-2 pre-thinning
    # (see the `frame_idx % 2 == 0` keep above) — passing the real duration
    # directly here (rather than re-deriving it from len(gif_frames) and a raw
    # fps) is what keeps that pre-thinning from silently halving playback time.
    true_duration = frames_processed / max(0.001, float(source_fps))
    gif_src = _decimate_frames_for_gif(
        gif_frames,
        duration=true_duration,
        gif_fps=actual_gif_fps,
        max_frames=max_gif,
    )
    _report("encoding", 89)
    frames_to_gif(gif_src, Path(args.gif).resolve(), actual_gif_fps, int(args.gif_width))
    _report("encoding", 92)
    print(f"[Pipeline] done in {time.time() - start_time:.2f}s", flush=True)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()