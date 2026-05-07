"""Optional FormLoop watermark on animated GIF (FFmpeg overlay)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def resolve_watermark_png(app_root: Path) -> Path | None:
    """
    PNG with alpha for FFmpeg overlay.
    Order: RVM_WATERMARK_PNG env → static/formloop-watermark.png → generated static asset.
    """
    env = (os.environ.get("RVM_WATERMARK_PNG") or "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    static_dir = app_root / "static"
    for name in ("formloop-watermark.png", "formloop-logo.png"):
        cand = static_dir / name
        if cand.is_file():
            return cand
    static_dir.mkdir(parents=True, exist_ok=True)
    gen = static_dir / ".generated-formloop-watermark.png"
    if gen.is_file():
        return gen
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    w, h = 280, 72
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        except OSError:
            font = ImageFont.load_default()
    text = "FormLoop"
    bbox = dr.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (w - tw) // 2, (h - th) // 2
    dr.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 160))
    dr.text((x, y), text, font=font, fill=(255, 255, 255, 220))
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
    # Keep full GIF animation. Do NOT use shortest=1 here; watermark input is a single PNG frame.
    flt = "[1:v]format=rgba,scale=140:-1[wm];[0:v][wm]overlay=(W-w)/2:H-h-14:format=auto"
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
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError):
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        return False
    if not tmp.is_file() or tmp.stat().st_size < 32:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(gif_path)
    return True
