"""Per-user job ownership and listing under api_outputs (GIF matte exports)."""

from __future__ import annotations

import os
import re
from pathlib import Path

OUTPUTS_DIR = Path(os.environ.get("RVM_OUTPUTS_DIR", str(Path(__file__).resolve().parent / "api_outputs"))).resolve()
_JOB_HEX = re.compile(r"^[0-9a-f]{32}$")
USAGE_DIR = OUTPUTS_DIR / "_usage"


def owner_file(job_dir: Path) -> Path:
    return job_dir / ".owner"


def saved_file(job_dir: Path) -> Path:
    return job_dir / ".saved"


def write_job_owner(job_id: str, uid: str) -> None:
    if not _JOB_HEX.match(job_id):
        raise ValueError("invalid job_id")
    d = OUTPUTS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    owner_file(d).write_text(uid.strip(), encoding="utf-8")


def read_job_owner(job_dir: Path) -> str | None:
    p = owner_file(job_dir)
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def mark_job_saved(job_id: str) -> None:
    if not _JOB_HEX.match(job_id):
        raise ValueError("invalid job_id")
    d = OUTPUTS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    saved_file(d).write_text("1", encoding="utf-8")


def is_job_saved(job_dir: Path) -> bool:
    return saved_file(job_dir).is_file()


def list_matte_gifs_for_owner(owner_uid: str | None) -> list[dict]:
    """
    Same shape as web_ui _scan_gifs entries, filtered by .owner uid.
    If owner_uid is None/empty, returns [] (no cross-user listing).
    """
    if not owner_uid:
        return []
    out: list[dict] = []
    if not OUTPUTS_DIR.is_dir():
        return out
    uid = owner_uid.strip()
    for job_dir in sorted(OUTPUTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not job_dir.is_dir() or not _JOB_HEX.match(job_dir.name):
            continue
        if read_job_owner(job_dir) != uid:
            continue
        if not is_job_saved(job_dir):
            continue
        gif = job_dir / "matte.gif"
        if not gif.is_file():
            continue
        try:
            st = gif.stat()
        except OSError:
            continue
        out.append(
            {
                "job_id": job_dir.name,
                "path": str(gif),
                "mtime": st.st_mtime,
                "size": st.st_size,
            }
        )
    return out


def count_matte_gifs_for_owner(owner_uid: str | None) -> int:
    return len(list_matte_gifs_for_owner(owner_uid))


def _usage_file(owner_uid: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.@-]", "_", owner_uid.strip())
    return USAGE_DIR / f"{safe}.count"


def read_owner_usage_count(owner_uid: str | None) -> int:
    """All-time created exports count (does not decrease on delete)."""
    if not owner_uid:
        return 0
    uid = owner_uid.strip()
    p = _usage_file(uid)
    try:
        if p.is_file():
            raw = p.read_text(encoding="utf-8").strip()
            n = int(raw)
            # Migration guard: never report lower than currently visible saved items.
            return max(n, count_matte_gifs_for_owner(uid))
    except (OSError, ValueError):
        pass
    return count_matte_gifs_for_owner(uid)


def increment_owner_usage(owner_uid: str | None) -> int:
    """Increment persistent usage counter for quota/billing."""
    if not owner_uid:
        return 0
    uid = owner_uid.strip()
    USAGE_DIR.mkdir(parents=True, exist_ok=True)
    current = read_owner_usage_count(uid)
    nxt = current + 1
    _usage_file(uid).write_text(str(nxt), encoding="utf-8")
    return nxt


def _usage_period_file(owner_uid: str, period_key: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.@-]", "_", owner_uid.strip())
    pk = re.sub(r"[^0-9]", "", period_key)[:16] or "0"
    return USAGE_DIR / f"{safe}.p{pk}.count"


def read_period_usage(owner_uid: str, period_key: str) -> int:
    if not owner_uid or not period_key:
        return 0
    uid = owner_uid.strip()
    p = _usage_period_file(uid, period_key)
    try:
        if p.is_file():
            return max(0, int(p.read_text(encoding="utf-8").strip() or "0"))
    except (OSError, ValueError):
        pass
    return 0


def increment_period_usage(owner_uid: str, period_key: str) -> int:
    if not owner_uid or not period_key:
        return 0
    uid = owner_uid.strip()
    USAGE_DIR.mkdir(parents=True, exist_ok=True)
    nxt = read_period_usage(uid, period_key) + 1
    _usage_period_file(uid, period_key).write_text(str(nxt), encoding="utf-8")
    return nxt


def read_quota_usage(owner_uid: str | None, billing_period_key: str | None) -> int:
    """Free: lifetime counter. Paid: usage in current Stripe billing period."""
    if not owner_uid:
        return 0
    if billing_period_key:
        return read_period_usage(owner_uid, billing_period_key)
    return read_owner_usage_count(owner_uid)


def increment_quota_usage(owner_uid: str | None, billing_period_key: str | None) -> int:
    if not owner_uid:
        return 0
    if billing_period_key:
        return increment_period_usage(owner_uid, billing_period_key)
    return increment_owner_usage(owner_uid)
