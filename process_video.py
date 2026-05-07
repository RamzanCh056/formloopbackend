#!/usr/bin/env python3
"""
RVM foreground + alpha → composite on solid white (no chromakey) → palette GIF,
or **transparent** GIF (`--transparent-gif`) from straight fg + alpha (best for overlays / signage).

Chroma-keying green-screen H.264 is unreliable (YUV compression shifts colors), which can
erase the subject. Using the model's alpha matte avoids that entirely.

Compositing: raw alpha (no blur) so thin objects (e.g. barbell shafts) are not eroded.
Inference uses full-res matting by default (--downsample-ratio 1). Encoder CRF 18.

**Transparent GIF:** premium pass uses **YOLO-seg** (default ``yolov8s-seg.pt``) at **1280px**
inference, morphological fill for holes (e.g. between legs), other people subtracted, then
edge despill / fringe kill. **Gear alpha boost is off by default** (pass ``--gear-alpha-boost`` if
bars look thin). RVM cannot separate background people from the athlete—keep the subject gate on.
``--no-yolo-seg-gate`` uses bbox gating; ``--yolo-weights yolov8m-seg.pt`` for harder footage.
**On-screen text or UI** in the source MP4 (e.g. workout labels) is part of the pixels—remove in
edit or re-export clean video. Optional ``--logo``.

**Transparent animations:** with ``--transparent-gif``, use ``--transparent-formats`` to also write
**animated WebP** and **APNG** (full RGBA, no 256-color GIF banding)—better for apps and digital
signage. Default is ``gif,webp,apng`` (same basename as ``--gif``). Requires ``ffmpeg`` with
``libwebp_anim`` / apng support (typical Homebrew build).
"""


import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _python_for_inference(project_root: Path) -> Path:
    """Prefer repo venv so inference avoids system/Homebrew Python mismatches."""
    search_dirs = [
        project_root / "venv" / "bin",
        project_root.parent / "venv" / "bin",
    ]
    for venv_dir in search_dirs:
        for name in ("python3", "python"):
            candidate = venv_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return Path(sys.executable)


def _video_fps(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    s = r.stdout.strip()
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def _cuda_runtime_usable() -> bool:
    """True only if CUDA alloc + sync works (driver matches PyTorch CUDA build)."""
    import torch

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


def _pick_device(preference: str) -> str:
    import torch

    if preference != "auto":
        if preference == "cuda" and not _cuda_runtime_usable():
            print(
                "[WARN] CUDA unusable (driver / PyTorch CUDA mismatch); using CPU",
                flush=True,
            )
            return "cpu"
        return preference

    if _cuda_runtime_usable():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_transparent_formats(s: str) -> set[str]:
    allowed = frozenset({"gif", "webp", "apng"})
    parts = {p.strip().lower() for p in s.split(",") if p.strip()}
    out = parts & allowed
    return out if out else {"gif"}


def _animation_output_paths(base_gif_path: Path, formats: set[str]) -> dict[str, Path]:
    """Sibling paths: e.g. output_transparent.gif → output_transparent.webp / .apng."""
    stem = base_gif_path.with_suffix("")
    m: dict[str, Path] = {}
    if "gif" in formats:
        m["gif"] = stem.with_suffix(".gif")
    if "webp" in formats:
        m["webp"] = stem.with_suffix(".webp")
    if "apng" in formats:
        m["apng"] = stem.with_suffix(".apng")
    return m


def _ffmpeg_rgba_sequence_to_gif(
    frames_dir: Path,
    *,
    fps_src: str,
    gif_fps: int,
    scale_vf: str,
    dest: Path,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fc = (
        f"[0:v]format=rgba,fps={gif_fps},{scale_vf},split[s0][s1];"
        f"[s0]palettegen=stats_mode=full:max_colors=255:reserve_transparent=1[p];"
        f"[s1][p]paletteuse=dither=sierra2_4a:alpha_threshold=64:diff_mode=rectangle"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-framerate",
            fps_src,
            "-i",
            str(frames_dir / "%06d.png"),
            "-filter_complex",
            fc,
            "-loop",
            "0",
            str(dest),
        ],
        cwd=str(frames_dir.parent),
        check=True,
    )


def _pillow_rgba_sequence_to_webp(
    frames_dir: Path,
    *,
    gif_fps: int,
    target_w: int,
    dest: Path,
    lossless: bool,
    quality: int,
) -> None:
    """Animated WebP with alpha (works without ffmpeg libwebp)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Animated WebP needs Pillow when ffmpeg has no libwebp_anim. "
            "pip install pillow"
        ) from exc

    paths = sorted(frames_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    if not paths:
        raise RuntimeError(f"no PNG frames in {frames_dir}")
    frames: list[Image.Image] = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        w, h = im.size
        nh = max(1, int(round(h * float(target_w) / float(max(1, w)))))
        if (w, h) != (target_w, nh):
            im = im.resize((target_w, nh), Image.Resampling.LANCZOS)
        frames.append(im)
    first, *rest = frames
    dur = max(1, int(round(1000.0 / float(max(1, gif_fps)))))
    kw: dict = {
        "save_all": True,
        "append_images": rest,
        "duration": dur,
        "loop": 0,
    }
    if lossless:
        kw["lossless"] = True
    else:
        kw["quality"] = max(1, min(100, int(quality)))
        kw["method"] = 6
    dest.parent.mkdir(parents=True, exist_ok=True)
    first.save(dest, format="WEBP", **kw)
    for im in frames:
        im.close()


def _ffmpeg_rgba_sequence_to_webp(
    frames_dir: Path,
    *,
    fps_src: str,
    gif_fps: int,
    scale_vf: str,
    dest: Path,
    lossless: bool,
    quality: int,
    target_w: int,
) -> None:
    vf = f"format=rgba,fps={gif_fps},{scale_vf}"
    cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-framerate",
        fps_src,
        "-i",
        str(frames_dir / "%06d.png"),
        "-vf",
        vf,
        "-an",
        "-loop",
        "0",
    ]
    if lossless:
        cmd += ["-c:v", "libwebp_anim", "-lossless", "1", "-preset", "default"]
    else:
        q = max(1, min(100, int(quality)))
        cmd += [
            "-c:v",
            "libwebp_anim",
            "-lossless",
            "0",
            "-q:v",
            str(q),
            "-compression_level",
            "6",
            "-preset",
            "default",
        ]
    cmd.append(str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, cwd=str(frames_dir.parent), check=True)
    except subprocess.CalledProcessError:
        print(
            "ffmpeg libwebp_anim unavailable or failed; using Pillow for animated WebP.",
            flush=True,
        )
        try:
            _pillow_rgba_sequence_to_webp(
                frames_dir,
                gif_fps=gif_fps,
                target_w=target_w,
                dest=dest,
                lossless=lossless,
                quality=quality,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not encode animated WebP (ffmpeg + Pillow both failed)."
            ) from exc


def _ffmpeg_rgba_sequence_to_apng(
    frames_dir: Path,
    *,
    fps_src: str,
    gif_fps: int,
    scale_vf: str,
    dest: Path,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    vf = f"format=rgba,fps={gif_fps},{scale_vf}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-framerate",
            fps_src,
            "-i",
            str(frames_dir / "%06d.png"),
            "-vf",
            vf,
            "-plays",
            "0",
            "-f",
            "apng",
            str(dest),
        ],
        cwd=str(frames_dir.parent),
        check=True,
    )


def _assert_input_has_video(path: Path) -> None:
    """Fail fast on audio-only files (e.g. .mp3) or containers with no video track."""
    audio_only = {
        ".mp3",
        ".m4a",
        ".aac",
        ".wav",
        ".flac",
        ".ogg",
        ".opus",
    }
    suf = path.suffix.lower()
    if suf in audio_only:
        raise ValueError(
            f"{path.name} is audio-only. RobustVideoMatting needs a **video** file "
            f"(e.g. person.mp4), not {suf}."
        )
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    line = (r.stdout or "").strip().lower()
    if r.returncode != 0 or "video" not in line:
        msg = f"No video stream in {path}. Use a file with a video track."
        if (r.stderr or "").strip():
            msg = f"{msg}\n{r.stderr.strip()}"
        raise ValueError(msg)


def _video_wh(path: Path) -> tuple[int, int]:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="RVM fg+alpha → white (or gray) backdrop → high-quality GIF (output.gif)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Source video (default: input.mp4 if present, else gym.mp4).",
    )
    parser.add_argument("--fg", type=Path, default=None, help="Foreground video (default: fg.mp4).")
    parser.add_argument("--alpha", type=Path, default=None, help="Alpha video (default: alpha.mp4).")
    parser.add_argument(
        "--temp",
        type=Path,
        default=None,
        help="Intermediate backdrop video (default: temp.mp4).",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Final GIF (default: output.gif).",
    )
    parser.add_argument(
        "--palette",
        type=Path,
        default=None,
        help="Palette image (default: palette.png).",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="white",
        choices=("white", "lightgray"),
        help="Solid backdrop behind the subject (default: white).",
    )
    parser.add_argument(
        "--downsample-ratio",
        type=float,
        default=1.0,
        help="RVM backbone scale (default 1.0 = full resolution; best for thin bars / hair). Lower = faster, less detail.",
    )
    parser.add_argument("--no-gif", action="store_true", help="Stop after temp.mp4 (no GIF).")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device for RVM: "auto", "cuda", "mps", or "cpu" (default: auto).',
    )
    parser.add_argument(
        "--transparent-gif",
        action="store_true",
        help="Encode background-free animation(s) from fg+alpha (see --transparent-formats).",
    )
    parser.add_argument(
        "--transparent-formats",
        type=str,
        default="gif,webp,apng",
        help="Comma-separated: gif, webp, apng. WebP/APNG keep full RGBA. Default all three (slower); use 'gif' alone for fastest export.",
    )
    parser.add_argument(
        "--webp-lossy",
        action="store_true",
        help="Animated WebP: lossy encode (smaller files; default is lossless).",
    )
    parser.add_argument(
        "--webp-quality",
        type=int,
        default=92,
        help="With --webp-lossy: quality 1–100 (default 92).",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=720,
        help="Max GIF width in px (320–1280; default 720; larger = sharper but bigger file).",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=12,
        help="Frames per second in GIF (default 12; API uses RVM_GIF_FPS; lower = faster encode, smaller files).",
    )
    parser.add_argument(
        "--no-premium",
        action="store_true",
        help="Skip premium matte (edge smooth + despill) and simpler GIF dither; transparent GIF only.",
    )
    parser.add_argument(
        "--logo",
        type=Path,
        default=None,
        help="Optional RGBA PNG to composite near bottom (with soft shadow). Use a sharp 2× export.",
    )
    parser.add_argument(
        "--logo-width-frac",
        type=float,
        default=0.34,
        help="Logo width as fraction of frame width (default 0.34).",
    )
    parser.add_argument(
        "--logo-margin-bottom",
        type=int,
        default=36,
        help="Pixels from bottom to logo baseline (default 36).",
    )
    parser.add_argument(
        "--matte-yuv420",
        action="store_true",
        help="Encode RVM fg/alpha as H.264 yuv420p (smaller). Default uses yuv444p + higher bitrate so barbells stay visible.",
    )
    parser.add_argument(
        "--matte-bitrate-mbps",
        type=int,
        default=0,
        help="Override fg/alpha H.264 bitrate (0 = use defaults: 18 hi-fi, 12 with --matte-yuv420).",
    )
    parser.add_argument(
        "--gear-alpha-boost",
        action="store_true",
        help="Lift alpha on metal/dark equipment (optional; can re-introduce background in limb gaps).",
    )
    parser.add_argument(
        "--no-gear-alpha-boost",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-yolo-subject-gate",
        action="store_true",
        help="Do not crop alpha to the largest YOLO person box (may keep background people).",
    )
    parser.add_argument(
        "--yolo-subject-pad",
        type=float,
        default=0.34,
        help="Expand primary-person box by this × max(w,h) so arms/barbells stay inside (default 0.34; bbox fallback only).",
    )
    parser.add_argument(
        "--no-yolo-seg-gate",
        action="store_true",
        help="Use padded YOLO boxes instead of segmentation masks for the subject gate (transparent GIF).",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolov8s-seg.pt",
        help="Ultralytics seg weights (default yolov8s-seg.pt; try yolov8m-seg.pt for harder clips).",
    )
    parser.add_argument(
        "--yolo-inference-imgsz",
        type=int,
        default=1280,
        help="YOLO inference size (larger = sharper masks, slower; default 1280).",
    )
    args = parser.parse_args()

    src = args.input
    if src is None:
        for candidate in (root / "input.mp4", root / "gym.mp4"):
            if candidate.is_file():
                src = candidate
                break
        if src is None:
            print(
                "Error: no default video found. Place input.mp4 or gym.mp4 in the project folder, or pass --input.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        src = (root / src).resolve() if not src.is_absolute() else src
    if not src.is_file():
        print(f"Error: input video not found: {src}", file=sys.stderr)
        sys.exit(1)
    _assert_input_has_video(src)

    fg_mp4 = (root / args.fg) if args.fg else root / "fg.mp4"
    alpha_mp4 = (root / args.alpha) if args.alpha else root / "alpha.mp4"
    temp_mp4 = (root / args.temp) if args.temp else root / "temp.mp4"
    if args.gif is not None:
        output_gif = (root / args.gif).resolve() if not args.gif.is_absolute() else args.gif.resolve()
    elif args.transparent_gif:
        output_gif = root / "output_transparent.gif"
    else:
        output_gif = root / "output.gif"
    palette_png = (root / args.palette) if args.palette else root / "palette.png"

    bg_color = "white" if args.background == "white" else "0xF0F0F0"

    checkpoint = root / "rvm_resnet50.pth"
    inference_script = root / "inference.py"

    if not checkpoint.is_file():
        print(f"Error: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not inference_script.is_file():
        print(f"Error: inference script not found: {inference_script}", file=sys.stderr)
        sys.exit(1)

    device = _pick_device(args.device)
    py = _python_for_inference(root)
    matte_pix = "yuv420p" if args.matte_yuv420 else "yuv444p"
    if args.matte_bitrate_mbps > 0:
        matte_mbps = int(args.matte_bitrate_mbps)
    else:
        matte_mbps = 12 if args.matte_yuv420 else 18
    inference_cmd = [
        str(py),
        str(inference_script),
        "--variant",
        "resnet50",
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--input-source",
        str(src),
        "--output-type",
        "video",
        "--output-alpha",
        str(alpha_mp4),
        "--output-foreground",
        str(fg_mp4),
        "--downsample-ratio",
        str(args.downsample_ratio),
        "--output-video-mbps",
        str(matte_mbps),
        "--output-video-pix-fmt",
        matte_pix,
        "--progress-interval",
        str(max(1, int(os.environ.get("RVM_PROGRESS_INTERVAL", "8")))),
    ]
    subprocess.run(inference_cmd, cwd=str(root), check=True)
    print(
        f"Foreground and alpha videos created ({matte_pix}, {matte_mbps} Mbps)",
        flush=True,
    )

    w, h = _video_wh(fg_mp4)
    fps_src = _video_fps(fg_mp4)
    fps_encode = f"{fps_src:.6f}".rstrip("0").rstrip(".")
    # Straight RGB + separate alpha → rgba (RVM fgr is straight; compositing matches inference).
    filter_complex = (
        f"[0:v]format=rgb24[rgb];"
        f"[1:v]format=gray,extractplanes=y[am];"
        f"[rgb][am]alphamerge,format=rgba[ck];"
        f"color=c={bg_color}:s={w}x{h}:d=3600:r={fps_encode}[bg];"
        f"[bg][ck]overlay=shortest=1:format=auto[out]"
    )
    white_cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-i",
        str(fg_mp4),
        "-i",
        str(alpha_mp4),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(temp_mp4),
    ]
    subprocess.run(white_cmd, cwd=str(root), check=True)
    print("Solid backdrop video created", flush=True)

    if args.no_gif:
        return

    gw = max(320, min(1280, int(args.gif_width)))
    gf = max(1, min(60, int(args.gif_fps)))
    scale = f"scale={gw}:-1:flags=lanczos"

    if args.transparent_gif:
        logo_path = args.logo
        if logo_path is not None:
            logo_path = (root / logo_path).resolve() if not logo_path.is_absolute() else logo_path
            if not logo_path.is_file():
                print(f"Error: --logo not found: {logo_path}", file=sys.stderr)
                sys.exit(1)
        tformats = _parse_transparent_formats(args.transparent_formats)
        anim_paths = _animation_output_paths(output_gif, tformats)

        do_refine = not args.no_premium
        gear_boost = bool(args.gear_alpha_boost) and not bool(
            getattr(args, "no_gear_alpha_boost", False)
        )
        use_rgba_pipeline = (
            do_refine
            or (logo_path is not None)
            or gear_boost
            or ("webp" in tformats)
            or ("apng" in tformats)
        )

        if use_rgba_pipeline:
            from premium_matte import write_premium_rgba_sequence

            yolo_model = None
            if not args.no_yolo_subject_gate:
                try:
                    from ultralytics import YOLO

                    wpath = str(args.yolo_weights)
                    yolo_model = YOLO(wpath)
                    if args.no_yolo_seg_gate:
                        print(
                            f"YOLO bbox subject gate on ({wpath}; box fallback mode).",
                            flush=True,
                        )
                    else:
                        print(
                            f"YOLO-seg subject gate on ({wpath}, imgsz={int(args.yolo_inference_imgsz)}; "
                            "morph-filled person mask, others subtracted).",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"Warning: YOLO gate skipped ({exc}). "
                        "pip install ultralytics to remove background people.",
                        flush=True,
                    )

            work_rgba = Path(tempfile.mkdtemp(prefix="rgba_pre_", dir=str(root)))
            try:
                write_premium_rgba_sequence(
                    fg_mp4,
                    alpha_mp4,
                    work_rgba,
                    refine=do_refine,
                    gear_alpha_boost=gear_boost,
                    yolo_model=yolo_model,
                    yolo_use_seg_gate=not args.no_yolo_seg_gate,
                    yolo_pad_frac=float(args.yolo_subject_pad),
                    yolo_inference_imgsz=int(args.yolo_inference_imgsz),
                    logo_path=logo_path,
                    logo_width_frac=float(args.logo_width_frac),
                    logo_margin_bottom=int(args.logo_margin_bottom),
                )
                lossless_webp = not bool(args.webp_lossy)
                wq = int(args.webp_quality)
                encode_order = ("gif", "webp", "apng")
                written: list[str] = []
                for fmt in encode_order:
                    if fmt not in tformats:
                        continue
                    dest = anim_paths[fmt]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if fmt == "gif":
                        _ffmpeg_rgba_sequence_to_gif(
                            work_rgba,
                            fps_src=fps_encode,
                            gif_fps=gf,
                            scale_vf=scale,
                            dest=dest,
                        )
                        written.append(f"GIF → {dest}")
                    elif fmt == "webp":
                        _ffmpeg_rgba_sequence_to_webp(
                            work_rgba,
                            fps_src=fps_encode,
                            gif_fps=gf,
                            scale_vf=scale,
                            dest=dest,
                            lossless=lossless_webp,
                            quality=wq,
                            target_w=gw,
                        )
                        mode = "lossless" if lossless_webp else f"lossy q={wq}"
                        written.append(f"WebP ({mode}) → {dest}")
                    else:
                        try:
                            _ffmpeg_rgba_sequence_to_apng(
                                work_rgba,
                                fps_src=fps_encode,
                                gif_fps=gf,
                                scale_vf=scale,
                                dest=dest,
                            )
                            written.append(f"APNG → {dest}")
                        except subprocess.CalledProcessError as exc:
                            print(
                                f"Warning: APNG encode failed ({exc}). "
                                "Try `brew reinstall ffmpeg` with apng support.",
                                file=sys.stderr,
                            )
            finally:
                shutil.rmtree(work_rgba, ignore_errors=True)
            parts = []
            if "gif" in tformats:
                parts.append("GIF palette (sierra2_4a)")
            if "webp" in tformats:
                parts.append("animated WebP (full alpha)")
            if "apng" in tformats:
                parts.append("APNG (full alpha)")
            if do_refine:
                parts.insert(0, "edge smooth + despill")
            if yolo_model is not None:
                parts.insert(
                    0,
                    "YOLO-seg primary-person gate"
                    if not args.no_yolo_seg_gate
                    else "YOLO bbox primary-person gate",
                )
            if gear_boost:
                parts.insert(0, "gear alpha recovery")
            if logo_path is not None:
                parts.append(f"logo {logo_path.name}")
            print(
                f"Transparent animation(s), width≤{gw}px, {gf} fps ({'; '.join(parts)}):",
                flush=True,
            )
            for line in written:
                print(f"  {line}", flush=True)
            return

        if tformats != {"gif"}:
            print(
                "Error: WebP/APNG require the RGBA matte pass. "
                "Use default (premium on), or add --logo / --gear-alpha-boost, "
                "or set --transparent-formats gif only.",
                file=sys.stderr,
            )
            sys.exit(1)

        fc = (
            f"[0:v]format=rgb24[rgb];"
            f"[1:v]format=gray,extractplanes=y[am];"
            f"[rgb][am]alphamerge,format=rgba,"
            f"fps={gf},{scale},split[s0][s1];"
            f"[s0]palettegen=stats_mode=full:max_colors=255:reserve_transparent=1[p];"
            f"[s1][p]paletteuse=alpha_threshold=64:diff_mode=rectangle:"
            f"dither=bayer:bayer_scale=2"
        )
        gif_dest = anim_paths["gif"]
        gif_dest.parent.mkdir(parents=True, exist_ok=True)
        gif_cmd = [
            "ffmpeg",
            "-y",
            "-threads",
            "0",
            "-i",
            str(fg_mp4),
            "-i",
            str(alpha_mp4),
            "-filter_complex",
            fc,
            "-loop",
            "0",
            str(gif_dest),
        ]
        subprocess.run(gif_cmd, cwd=str(root), check=True)
        print(f"GIF created (transparent), width≤{gw}px, {gf} fps → {gif_dest}", flush=True)
        return

    palette_cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-i",
        str(temp_mp4),
        "-vf",
        f"fps={gf},{scale},palettegen=stats_mode=full",
        "-frames:v",
        "1",
        str(palette_png),
    ]
    subprocess.run(palette_cmd, cwd=str(root), check=True)

    gif_cmd = [
        "ffmpeg",
        "-y",
        "-threads",
        "0",
        "-i",
        str(temp_mp4),
        "-i",
        str(palette_png),
        "-filter_complex",
        f"fps={gf},{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
        str(output_gif),
    ]
    subprocess.run(gif_cmd, cwd=str(root), check=True)
    print(f"GIF created (opaque backdrop), width≤{gw}px → {output_gif}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(
            f"Error: command failed (exit {exc.returncode}): {' '.join(str(x) for x in exc.cmd)}",
            file=sys.stderr,
        )
        sys.exit(exc.returncode if exc.returncode is not None else 1)
    except FileNotFoundError as exc:
        print(
            f"Error: executable not found (install ffmpeg/ffprobe and use the same Python as for RVM): {exc}",
            file=sys.stderr,
        )
        sys.exit(127)
