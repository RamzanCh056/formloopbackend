"""Optional FormLoop watermark on animated GIF (FFmpeg overlay)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _png_has_transparency(path: Path) -> bool:
    """Return True only if the PNG has at least some pixels with alpha < 255."""
    try:
        from PIL import Image
        im = Image.open(str(path))
        if im.mode == "RGBA":
            import numpy as np
            arr = __import__("numpy").array(im)
            return bool(arr[:, :, 3].min() < 255)
        if im.mode == "P" and "transparency" in im.info:
            return True
        if im.mode == "LA":
            return True
        return False
    except Exception:
        return False


def resolve_watermark_png(app_root: Path) -> Path | None:
    """
    PNG with real alpha transparency for FFmpeg overlay.
    Order: RVM_WATERMARK_PNG env → static/formloop-watermark.png (if transparent)
           → generated static asset (always transparent).
    Static PNGs are skipped if they have no actual transparency (all alpha=255).
    """
    env = (os.environ.get("RVM_WATERMARK_PNG") or "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    static_dir = app_root / "static"
    for name in ("formloop-watermark.png", "formloop-logo-transparent.png", "formloop-logo.png"):
        cand = static_dir / name
        if cand.is_file() and _png_has_transparency(cand):
            return cand
    static_dir.mkdir(parents=True, exist_ok=True)
    gen = static_dir / ".generated-formloop-watermark.png"
    if gen.is_file():
        return gen
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    w, h = 480, 112
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 56)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
        except OSError:
            font = ImageFont.load_default()
    text = "FormLoop"
    bbox = dr.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (w - tw) // 2, (h - th) // 2
    # Semi-transparent dark pill behind text for legibility on any background
    pad = 14
    dr.rounded_rectangle([x - pad, y - pad, x + tw + pad, y + th + pad],
                          radius=10, fill=(0, 0, 0, 110))
    dr.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 180))
    dr.text((x, y), text, font=font, fill=(255, 255, 255, 240))
    im.save(gen, format="PNG")
    return gen if gen.is_file() else None


def apply_png_watermark_to_gif(gif_path: Path, watermark_png: Path) -> bool:
    """
    Overlay a PNG (with alpha) on each frame of an animated GIF.
    Returns True if output replaced input; False if FFmpeg failed (caller keeps original).
    """
    if not gif_path.is_file() or not watermark_png.is_file():
        return False
    tmp = gif_path.with_suffix(".wm.gif")

    # Scale watermark to 28% of GIF width (minimum 240px) so it's clearly visible
    # on both landscape and tall portrait GIFs.
    gif_w = 960
    try:
        from PIL import Image as _PILImg
        with _PILImg.open(str(gif_path)) as _im:
            gif_w = _im.width
    except Exception:
        pass
    wm_w = max(240, int(gif_w * 0.28))

    # palettegen reserve_transparent=1 is required to keep GIF transparency intact.
    flt = (
        f"[1:v]format=rgba,scale={wm_w}:-1[wm];"
        "[0:v][wm]overlay=W-w-14:14[overlaid];"
        "[overlaid]split[a][b];"
        "[a]palettegen=reserve_transparent=1:transparency_color=000000[pal];"
        "[b][pal]paletteuse=dither=none:diff_mode=rectangle"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(gif_path),
        "-i",
        str(watermark_png),
        "-filter_complex",
        flt,
        "-loop",
        "0",
        str(tmp),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        _ = result  # suppress unused-variable warning
    except subprocess.CalledProcessError as exc:
        print(f"[WATERMARK] FFmpeg failed rc={exc.returncode}\nSTDERR: {exc.stderr[-2000:]}", flush=True)
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return False
    except OSError as exc:
        print(f"[WATERMARK] OSError: {exc}", flush=True)
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return False
    if not tmp.is_file() or tmp.stat().st_size < 32:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(gif_path)
    return True
