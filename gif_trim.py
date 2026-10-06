"""Lossless GIF trimming for the "over 50 MB" flow (Google Slides / Docs limit).

Trimming here only ever *drops whole frames* via gifsicle frame selection
(``gifsicle in.gif "#a-b" -o out.gif``). No -O / --lossy / --colors / --resize
flags are ever passed, so every kept frame keeps its exact pixels and palette.
The original GIF is never modified; results are written to a separate file.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

# Google's "50 MB" may be decimal or binary; use the stricter decimal reading
# so "fits" is true under both interpretations.
SLIDES_SIZE_LIMIT_BYTES = 50_000_000
# Auto-fit aims well under that line (~44.8 MiB) for headroom.
SLIDES_SAFE_TARGET_BYTES = 47_000_000

_IMAGE_LINE_RE = re.compile(r"^\s*\+ image #(\d+)")
_DISPOSAL_RE = re.compile(r"disposal (\w+)")
_DELAY_RE = re.compile(r"delay ([\d.]+)s")


def gifsicle_available() -> bool:
    return shutil.which("gifsicle") is not None


def oversize_for_slides(size_bytes: int | None) -> bool:
    return bool(size_bytes) and int(size_bytes) > SLIDES_SIZE_LIMIT_BYTES


def gif_info(path: Path) -> dict:
    """Frame count, per-frame delays (seconds) and disposals, read via ``gifsicle --info``."""
    out = subprocess.run(
        ["gifsicle", "--info", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    delays: list[float] = []
    disposals: list[str] = []
    for line in out.splitlines():
        if _IMAGE_LINE_RE.match(line):
            delays.append(0.0)
            disposals.append("none")
            continue
        if not delays:
            continue
        m = _DELAY_RE.search(line)
        if m:
            delays[-1] = float(m.group(1))
        m = _DISPOSAL_RE.search(line)
        if m:
            disposals[-1] = m.group(1)
    size = path.stat().st_size
    return {
        "size_bytes": size,
        "frame_count": len(delays),
        "frame_delays": delays,
        "disposals": disposals,
        "oversize_for_slides": oversize_for_slides(size),
        "limit_bytes": SLIDES_SIZE_LIMIT_BYTES,
        "target_bytes": SLIDES_SAFE_TARGET_BYTES,
    }


def _select_frames(src: Path, dest: Path, start: int, end: int) -> int:
    tmp = dest.with_name(dest.name + ".tmp")
    subprocess.run(
        ["gifsicle", str(src), f"#{start}-{end}", "-o", str(tmp)],
        check=True, capture_output=True, timeout=300,
    )
    tmp.replace(dest)
    return dest.stat().st_size


def trim_range(src: Path, dest: Path, start: int, end: int, info: dict | None = None) -> dict:
    """Extract frames ``start..end`` (inclusive) of ``src`` into ``dest``."""
    info = info or gif_info(src)
    total = info["frame_count"]
    if total < 1:
        raise ValueError("GIF has no frames")
    if not (0 <= start <= end < total):
        raise ValueError(f"frame range must satisfy 0 <= start <= end < {total}")
    if start > 0 and info["disposals"][start - 1] not in ("background", "previous"):
        # Frame `start` would be drawn on top of what frame start-1 left on the
        # canvas; cutting it off would change the first kept frame's pixels.
        # Our matte GIFs always use disposal=background, so this is defensive.
        raise ValueError("this GIF can only be trimmed from the end (frames depend on earlier frames)")
    size = _select_frames(src, dest, start, end)
    return {
        "start": start,
        "end": end,
        "frames_kept": end - start + 1,
        "frame_count": total,
        "original_size_bytes": info["size_bytes"],
        "size_bytes": size,
        "oversize_for_slides": oversize_for_slides(size),
    }


def auto_fit(src: Path, dest: Path, target_bytes: int = SLIDES_SAFE_TARGET_BYTES) -> dict:
    """Keep frames #0..N for the largest N whose output fits under ``target_bytes``."""
    info = gif_info(src)
    total = info["frame_count"]
    orig = info["size_bytes"]
    if total < 1:
        raise ValueError("GIF has no frames")
    if orig <= target_bytes:
        return trim_range(src, dest, 0, total - 1, info)
    # Binary search on the real gifsicle output size, seeded by the linear estimate.
    lo, hi = 0, total - 1  # lo: known/assumed fitting end index, hi: known too big
    guess = max(0, min(total - 2, int(total * target_bytes / orig) - 1))
    best: int | None = None
    probe = dest.with_name(dest.name + ".probe")
    try:
        while lo <= hi:
            mid = guess if guess is not None else (lo + hi) // 2
            guess = None
            size = _select_frames(src, probe, 0, mid)
            if size <= target_bytes:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
    finally:
        probe.unlink(missing_ok=True)
    if best is None:
        raise ValueError("even a single frame exceeds the size target")
    return trim_range(src, dest, 0, best, info)


# Trimmed outputs, newest wins when several exist. Each one's frame selection
# is recorded in a sidecar so it can be re-cut from a re-rendered matte.gif
# (e.g. after save-time rotation) instead of serving a stale copy.
TRIM_OUTPUT_NAMES = {"auto": "matte_fit.gif", "range": "matte_trim.gif"}
_TRIM_META_FILE = ".trim_meta.json"


def _read_trim_meta(job_dir: Path) -> dict:
    try:
        data = json.loads((job_dir / _TRIM_META_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def record_trim(job_dir: Path, name: str, mode: str, start: int, end: int) -> None:
    meta = _read_trim_meta(job_dir)
    meta[name] = {"mode": mode, "start": start, "end": end}
    (job_dir / _TRIM_META_FILE).write_text(json.dumps(meta), encoding="utf-8")


def trimmed_outputs(job_dir: Path) -> list[Path]:
    """Existing trimmed GIFs for a job, oldest first (so the last one is the newest)."""
    found = [job_dir / n for n in TRIM_OUTPUT_NAMES.values() if (job_dir / n).is_file()]
    return sorted(found, key=lambda p: p.stat().st_mtime)


def recut_trimmed_outputs(job_dir: Path) -> None:
    """Re-derive existing trimmed GIFs from the current matte.gif, keeping their order.

    Auto-fit is re-run (sizes can change after re-rendering); manual trims reuse
    their recorded frame range. Still frame selection only.
    """
    src = job_dir / "matte.gif"
    meta = _read_trim_meta(job_dir)
    for out in trimmed_outputs(job_dir):
        m = meta.get(out.name) or {}
        if m.get("mode") == "range":
            res = trim_range(src, out, int(m["start"]), int(m["end"]))
        else:
            res = auto_fit(src, out)
        record_trim(job_dir, out.name, m.get("mode") or "auto", res["start"], res["end"])

