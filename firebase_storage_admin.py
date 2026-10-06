"""Optional Firebase Storage uploads using the Admin SDK (service account).

Set ``GOOGLE_APPLICATION_CREDENTIALS`` to a JSON key path, or set
``FIREBASE_SERVICE_ACCOUNT_JSON`` / ``FIREBASE_CONFIG`` to the service-account JSON.
Multiline ``FIREBASE_CONFIG`` in ``RobustVideoMatting/.env`` (value starts on the line after ``FIREBASE_CONFIG=``)
is parsed via ``JSONDecoder.raw_decode``. Bucket from ``FIREBASE_BUCKET`` or web config.

If credentials are missing, upload helpers are no-ops and callers fall back to
local API URLs.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import tempfile
import time
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

import requests

from firebase_auth import get_firebase_web_config

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_ready: bool | None = None

_APP_DIR = Path(__file__).resolve().parent


def _credential_dict_from_env_firebase_config() -> dict | None:
    fc = (os.environ.get("FIREBASE_CONFIG") or "").strip()
    if not fc.startswith("{"):
        return None
    try:
        obj = json.loads(fc)
        # Only treat as service-account credentials when the key is present;
        # FIREBASE_CONFIG may also hold the web config (apiKey, authDomain…)
        # which must NOT be passed to credentials.Certificate().
        if isinstance(obj, dict) and obj.get("type") == "service_account":
            return obj
        return None
    except json.JSONDecodeError:
        return None


def _credential_dict_from_dotenv_multiline() -> dict | None:
    """When .env has ``FIREBASE_CONFIG=`` then JSON on following lines (dotenv cannot load that into env)."""
    path = _APP_DIR / ".env"
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    key = "FIREBASE_CONFIG="
    i = raw.find(key)
    if i < 0:
        return None
    tail = raw[i + len(key) :].lstrip(" \t\r\n")
    if not tail.startswith("{"):
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError as exc:
        _log.warning("FIREBASE_CONFIG in .env is not valid JSON: %s", exc)
        return None
    if isinstance(obj, dict) and obj.get("type") == "service_account":
        return obj
    return None


def firebase_storage_ready() -> bool:
    """True when Admin SDK is initialized and Storage is available."""
    global _ready
    with _lock:
        if _ready is not None:
            return _ready
        try:
            _ready = _init_locked()
        except Exception as exc:
            _log.warning("Firebase Admin init failed: %s", exc)
            _ready = False
        return bool(_ready)


def _init_locked() -> bool:
    json_str = (os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON") or "").strip()
    cred_path = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()

    cred_dict: dict | None = None
    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                cred_dict = parsed
        except json.JSONDecodeError:
            pass
    if cred_dict is None:
        cred_dict = _credential_dict_from_env_firebase_config()
    if cred_dict is None:
        cred_dict = _credential_dict_from_dotenv_multiline()

    if cred_dict is None and not cred_path:
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError:
        _log.warning("firebase-admin not installed; Storage uploads disabled.")
        return False
    cfg = get_firebase_web_config()
    bucket_name = (os.environ.get("FIREBASE_BUCKET") or cfg.get("storageBucket") or "").strip()
    if not bucket_name:
        return False
    try:
        firebase_admin.get_app()
    except ValueError:
        if cred_dict is not None:
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
    return True


def _download_url(bucket_name: str, object_path: str, token: str) -> str:
    enc = quote(object_path, safe="")
    return f"https://firebasestorage.googleapis.com/v0/b/{bucket_name}/o/{enc}?alt=media&token={token}"


def _upload_file(local_path: Path, object_path: str, content_type: str) -> str:
    from firebase_admin import storage

    bucket = storage.bucket()
    blob = bucket.blob(object_path)
    blob.upload_from_filename(str(local_path), content_type=content_type)
    dl_token = str(uuid.uuid4())
    meta = dict(blob.metadata or {})
    meta["firebaseStorageDownloadTokens"] = dl_token
    blob.metadata = meta
    blob.patch()
    return _download_url(bucket.name, object_path, dl_token)


def upload_user_export_media(
    *,
    uid: str,
    export_id: str,
    gif_path: Path,
    webm_path: Path | None,
) -> dict[str, str | None]:
    """Upload GIF (and WebM if present) under users/{uid}/exports/{export_id}/.

    Returns ``{"gifUrl": str, "webmUrl": str | None}``.
    """
    if not firebase_storage_ready():
        raise RuntimeError("Firebase Storage is not configured")
    if not gif_path.is_file():
        raise FileNotFoundError("matte.gif missing")
    base = f"users/{uid}/exports/{export_id}"
    gif_url = _upload_file(gif_path, f"{base}/matte.gif", "image/gif")
    webm_url: str | None = None
    if webm_path is not None and webm_path.is_file():
        webm_url = _upload_file(webm_path, f"{base}/matte_transparent.webm", "video/webm")
    return {"gifUrl": gif_url, "webmUrl": webm_url}


def upload_user_export_media_from_urls(
    *,
    uid: str,
    export_id: str,
    gif_url: str,
    webm_url: str | None,
) -> dict[str, str | None]:
    """Copy GIF/WebM bytes from HTTPS (RunPod / `gifs/` / `webms/` bucket URLs) into **your** Firebase path ``users/{uid}/exports/{export_id}/``.

    The Admin SDK needs bytes once; we fetch from URL then upload — nothing is stored as the user's library on the API machine.
    """
    if not firebase_storage_ready():
        raise RuntimeError("Firebase Storage is not configured")
    gu = (gif_url or "").strip()
    if not gu.startswith(("http://", "https://")):
        raise ValueError("gif_url must be an http(s) URL")
    base = f"users/{uid}/exports/{export_id}"
    hdrs = {"User-Agent": "FormLoop-Server/1.0", "Accept": "*/*"}

    def _get(url: str, timeout: int) -> bytes:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=timeout, headers=hdrs, allow_redirects=True)
                r.raise_for_status()
                return r.content
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.4 * (attempt + 1))
        if last_exc:
            raise last_exc
        raise RuntimeError("download failed")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        gif_local = td_path / "matte.gif"
        gif_local.write_bytes(_get(gu, 180))
        if not gif_local.is_file() or gif_local.stat().st_size < 64:
            raise ValueError("downloaded GIF too small or missing")
        out_gif = _upload_file(gif_local, f"{base}/matte.gif", "image/gif")

        out_webm: str | None = None
        wu = (webm_url or "").strip()
        if wu.startswith(("http://", "https://")):
            wl = td_path / "matte_transparent.webm"
            wl.write_bytes(_get(wu, 300))
            if wl.is_file() and wl.stat().st_size > 64:
                out_webm = _upload_file(wl, f"{base}/matte_transparent.webm", "video/webm")

    return {"gifUrl": out_gif, "webmUrl": out_webm}


import re as _re
_JOB_HEX_RE = _re.compile(r"^[0-9a-f]{32}$")
# Local-dev URLs stored in Firestore by old sessions — never reachable on Railway
_LOCAL_URL_RE = _re.compile(
    r'^https?://(localhost|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)(:\d+)?/'
)


def list_user_exports_from_firestore(uid: str, limit: int | None = None) -> list[dict]:
    """Return the user's saved exports from Firestore users/{uid}/exports.

    Each entry: {job_id, source_filename, gif_url, webm_url}.
    Returns [] if Firebase is not configured or Firestore is unreachable.
    """
    if not firebase_storage_ready():
        return []
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        snaps = list(db.collection("users").document(uid).collection("exports").stream())
        rows: list[dict] = []
        for snap in snaps:
            data = snap.to_dict() or {}
            gif_url = (data.get("gifUrl") or "").strip()
            if not gif_url:
                continue
            if _LOCAL_URL_RE.match(gif_url):
                continue  # localhost URL from an old dev session — dead on Railway
            job_id = (data.get("jobId") or "").strip()
            if not _JOB_HEX_RE.match(job_id):
                continue  # skip exports missing a valid hex job_id
            raw_tags = data.get("customTags") or []
            tags = [str(t).strip() for t in raw_tags if str(t).strip()] if isinstance(raw_tags, list) else []
            rows.append({
                "job_id": job_id,
                "export_doc_id": snap.id,
                "source_filename": (data.get("title") or "").strip() or None,
                "gif_url": gif_url,
                "webm_url": (data.get("webmUrl") or "").strip() or None,
                "created_at": (data.get("createdAt") or ""),
                "tags": tags,
                "platform": (data.get("platform") or "").strip() or None,
                # Always present, default None -- absent/null means
                # Uncategorized. Never used to drop a GIF from the listing.
                "region_id": (data.get("regionId") or None),
                "equipment_id": (data.get("equipmentId") or None),
            })
        def _sort_ts(r) -> float:
            v = r["created_at"]
            if hasattr(v, "timestamp"):
                return float(v.timestamp())
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        rows.sort(key=_sort_ts, reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return rows
    except Exception as exc:
        _log.warning("Firestore list_user_exports uid=%s: %s", uid, exc)
        return []


def write_export_to_firestore(
    *,
    uid: str,
    export_id: str,
    job_id: str,
    gif_url: str,
    webm_url: str | None = None,
    title: str | None = None,
    platform: str | None = None,
    region_id: str | None = None,
    equipment_id: str | None = None,
) -> bool:
    """Write (or overwrite) a user's export document in Firestore users/{uid}/exports/{export_id}.

    Returns True on success, False if Firebase is not configured or the write fails.
    """
    if not firebase_storage_ready():
        return False
    try:
        from datetime import datetime, timezone
        from firebase_admin import firestore as _fs
        db = _fs.client()
        doc_data: dict = {
            "jobId": job_id,
            "gifUrl": gif_url,
            "createdAt": datetime.now(timezone.utc),
            "customTags": [],
            # Always present, default None -- absent/null means Uncategorized.
            "regionId": region_id,
            "equipmentId": equipment_id,
        }
        if webm_url:
            doc_data["webmUrl"] = webm_url
        if title:
            doc_data["title"] = title
        if platform:
            doc_data["platform"] = platform
        db.collection("users").document(uid).collection("exports").document(export_id).set(doc_data)
        return True
    except Exception as exc:
        _log.warning("write_export_to_firestore uid=%s export_id=%s: %s", uid, export_id, exc)
        return False


def backfill_quota_counters() -> None:
    """One-time: set quota_used for any user who has export docs but no
    quota_used field yet, so existing users don't get a free reset to 0 when
    the permanent-counter quota fix ships. Safe to call more than once —
    only touches users missing the field.
    """
    if not firebase_storage_ready():
        return
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        users = db.collection("users").stream()
        for user_doc in users:
            uid = user_doc.id
            data = user_doc.to_dict() or {}
            if "quota_used" not in data:
                exports = db.collection("users").document(uid).collection("exports").stream()
                count = sum(1 for _ in exports)
                if count > 0:
                    db.collection("users").document(uid).set(
                        {"quota_used": count}, merge=True,
                    )
                    print(f"Backfilled {uid}: {count} exports", flush=True)
    except Exception as e:
        print(f"Backfill error: {e}", flush=True)


def delete_user_export_from_firestore(uid: str, job_id: str) -> bool:
    """Delete the Firestore export doc(s) for this user/job_id. Returns True if any were deleted."""
    if not firebase_storage_ready():
        return False
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        col = db.collection("users").document(uid).collection("exports")
        # The doc key is exportId, but jobId is a field — query for it.
        hits = col.where("jobId", "==", job_id).stream()
        deleted = False
        for snap in hits:
            snap.reference.delete()
            deleted = True
        if deleted:
            # Deleting from the library also kills its public /g/{job_id} link.
            delete_public_link(uid, job_id)
        return deleted
    except Exception as exc:
        _log.warning("Firestore delete_user_export uid=%s job_id=%s: %s", uid, job_id, exc)
        return False


# ---------------------------------------------------------------------------
# Public share links: /g/{job_id}.gif streams users/{uid}/exports/{exportId}/.
# Top-level publicLinks/{job_id} -> {uid, exportId} lets the unauthenticated
# route resolve a job_id (uuid4 hex, unguessable) to its storage path without
# a collection-group query. exportId itself is exp_<ms timestamp> (guessable),
# so it is never used as the public key.
# ---------------------------------------------------------------------------
_PUBLIC_LINKS = "publicLinks"
_EXPORT_OBJECT_NAMES = {"gif": "matte.gif", "webm": "matte_transparent.webm"}


def find_export_id_for_job(uid: str, job_id: str) -> str | None:
    if not firebase_storage_ready():
        return None
    try:
        from firebase_admin import firestore as _fs
        col = _fs.client().collection("users").document(uid).collection("exports")
        for snap in col.where("jobId", "==", job_id).limit(1).stream():
            return snap.id
    except Exception as exc:
        _log.warning("find_export_id_for_job uid=%s job_id=%s: %s", uid, job_id, exc)
    return None


def write_public_link(uid: str, job_id: str, export_id: str) -> bool:
    if not firebase_storage_ready():
        return False
    try:
        from firebase_admin import firestore as _fs
        _fs.client().collection(_PUBLIC_LINKS).document(job_id).set(
            {"uid": uid, "exportId": export_id}
        )
        return True
    except Exception as exc:
        _log.warning("write_public_link job_id=%s: %s", job_id, exc)
        return False


def delete_public_link(uid: str, job_id: str) -> None:
    try:
        from firebase_admin import firestore as _fs
        ref = _fs.client().collection(_PUBLIC_LINKS).document(job_id)
        snap = ref.get()
        if snap.exists and (snap.to_dict() or {}).get("uid") == uid:
            ref.delete()
    except Exception as exc:
        _log.warning("delete_public_link job_id=%s: %s", job_id, exc)


def get_public_link(job_id: str) -> tuple[str, str] | None:
    """Return (uid, exportId) for a registered public link, or None."""
    if not firebase_storage_ready():
        return None
    try:
        from firebase_admin import firestore as _fs
        snap = _fs.client().collection(_PUBLIC_LINKS).document(job_id).get()
        data = (snap.to_dict() or {}) if snap.exists else {}
        uid, export_id = data.get("uid"), data.get("exportId")
        return (uid, export_id) if uid and export_id else None
    except Exception as exc:
        _log.warning("get_public_link job_id=%s: %s", job_id, exc)
        return None


# Frame-trimmed copies (see gif_trim.TRIM_OUTPUT_NAMES), uploaded next to the
# original. The public GIF link serves the newest one; matte.gif stays the
# untouched full-quality original.
TRIMMED_GIF_NAMES = ("matte_fit.gif", "matte_trim.gif")


def upload_trimmed_gifs(uid: str, export_id: str, paths: list[Path]) -> None:
    """Upload trimmed GIFs in the given order (oldest first, so 'updated' order matches)."""
    if not firebase_storage_ready():
        return
    base = f"users/{uid}/exports/{export_id}"
    for p in paths:
        if p.name in TRIMMED_GIF_NAMES and p.is_file():
            _upload_file(p, f"{base}/{p.name}", "image/gif")


def get_public_export_blob(job_id: str, kind: str):
    """Return the storage Blob (with metadata loaded) for a public link, or None.

    For GIFs, the most recently uploaded trimmed copy wins over the original.
    """
    name = _EXPORT_OBJECT_NAMES.get(kind)
    link = get_public_link(job_id) if name else None
    if not link:
        return None
    uid, export_id = link
    try:
        from firebase_admin import storage
        bucket = storage.bucket()
        base = f"users/{uid}/exports/{export_id}"
        if kind == "gif":
            trimmed = [b for b in (bucket.get_blob(f"{base}/{n}") for n in TRIMMED_GIF_NAMES) if b is not None]
            if trimmed:
                return max(trimmed, key=lambda b: b.updated)
        return bucket.get_blob(f"{base}/{name}")
    except Exception as exc:
        _log.warning("get_public_export_blob job_id=%s kind=%s: %s", job_id, kind, exc)
        return None


_UNSET = object()  # sentinel: "field not provided" vs "explicitly set to null"

# Seeded once per user, on first library access, only if they have zero
# regions yet. Equipment names repeat per-region on purpose (locked design:
# equipment is scoped PER region, e.g. "Upper Body > Bands" and
# "Lower Body > Bands" are different subcategory docs) -- renaming/deleting
# one never touches the other.
_DEFAULT_LIBRARY_REGIONS = ["Upper Body", "Lower Body", "Full Body", "Core"]
_DEFAULT_LIBRARY_EQUIPMENT = ["Kettlebells", "Bands", "Barbells", "Dumbbells", "Bodyweight"]


def update_export_fields(
    uid: str,
    job_id: str,
    *,
    region_id: object = _UNSET,
    equipment_id: object = _UNSET,
    title: object = _UNSET,
    tags: object = _UNSET,
) -> bool:
    """Update fields on an existing export doc, found by jobId (edit-export,
    net new -- lets a user categorize/re-title/re-tag a GIF they already
    saved). Only fields actually passed (not _UNSET) are written, so passing
    region_id=None explicitly clears it (moves the GIF to Uncategorized)
    without touching title/tags/equipment_id. Returns True if any doc was
    updated, False if not found or Firebase isn't configured.
    """
    if not firebase_storage_ready():
        return False
    updates: dict = {}
    if region_id is not _UNSET:
        updates["regionId"] = region_id
    if equipment_id is not _UNSET:
        updates["equipmentId"] = equipment_id
    if title is not _UNSET:
        updates["title"] = (str(title).strip() or None) if title is not None else None
    if tags is not _UNSET:
        updates["customTags"] = [str(t).strip() for t in (tags or []) if str(t).strip()]
    if not updates:
        return False
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        col = db.collection("users").document(uid).collection("exports")
        hits = list(col.where("jobId", "==", job_id).stream())
        if not hits:
            return False
        updates["updatedAtServer"] = _fs.SERVER_TIMESTAMP
        for snap in hits:
            snap.reference.update(updates)
        return True
    except Exception as exc:
        _log.warning("update_export_fields uid=%s job_id=%s: %s", uid, job_id, exc)
        return False


def list_library_categories(uid: str) -> list[dict]:
    """Return this user's regions (each with its nested equipment
    subcategories), sorted by `order`. Seeds the default region/equipment
    set on first-ever access (i.e. only when the user has zero regions) --
    never re-seeds afterward, even if they later delete everything, so an
    intentional "delete all my categories" sticks.
    """
    if not firebase_storage_ready():
        return []
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        regions_col = db.collection("users").document(uid).collection("libraryCategories")
        region_snaps = list(regions_col.stream())
        if not region_snaps:
            _seed_default_library_categories(uid)
            region_snaps = list(regions_col.stream())
        regions = []
        for rsnap in region_snaps:
            rdata = rsnap.to_dict() or {}
            sub_snaps = list(regions_col.document(rsnap.id).collection("subcategories").stream())
            subs = sorted(
                (
                    {"id": s.id, "name": (s.to_dict() or {}).get("name") or "", "order": (s.to_dict() or {}).get("order") or 0}
                    for s in sub_snaps
                ),
                key=lambda x: x["order"],
            )
            regions.append({
                "id": rsnap.id,
                "name": rdata.get("name") or "",
                "order": rdata.get("order") or 0,
                "equipment": subs,
            })
        regions.sort(key=lambda x: x["order"])
        return regions
    except Exception as exc:
        _log.warning("list_library_categories uid=%s: %s", uid, exc)
        return []


def _seed_default_library_categories(uid: str) -> None:
    from firebase_admin import firestore as _fs
    from datetime import datetime, timezone
    db = _fs.client()
    regions_col = db.collection("users").document(uid).collection("libraryCategories")
    now = datetime.now(timezone.utc)
    for r_order, region_name in enumerate(_DEFAULT_LIBRARY_REGIONS):
        region_ref = regions_col.document()
        region_ref.set({"name": region_name, "order": r_order, "createdAt": now})
        sub_col = region_ref.collection("subcategories")
        for s_order, equip_name in enumerate(_DEFAULT_LIBRARY_EQUIPMENT):
            sub_col.document().set({"name": equip_name, "order": s_order, "createdAt": now})


def create_library_region(uid: str, name: str) -> dict | None:
    if not firebase_storage_ready():
        return None
    try:
        from firebase_admin import firestore as _fs
        from datetime import datetime, timezone
        db = _fs.client()
        regions_col = db.collection("users").document(uid).collection("libraryCategories")
        existing = list(regions_col.stream())
        order = len(existing)
        ref = regions_col.document()
        ref.set({"name": (name or "").strip() or "Untitled", "order": order, "createdAt": datetime.now(timezone.utc)})
        return {"id": ref.id, "name": (name or "").strip() or "Untitled", "order": order, "equipment": []}
    except Exception as exc:
        _log.warning("create_library_region uid=%s: %s", uid, exc)
        return None


def rename_library_region(uid: str, region_id: str, name: str) -> bool:
    if not firebase_storage_ready():
        return False
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        ref = db.collection("users").document(uid).collection("libraryCategories").document(region_id)
        if not ref.get().exists:
            return False
        ref.update({"name": (name or "").strip() or "Untitled"})
        return True
    except Exception as exc:
        _log.warning("rename_library_region uid=%s region_id=%s: %s", uid, region_id, exc)
        return False


def reorder_library_regions(uid: str, ordered_ids: list[str]) -> bool:
    if not firebase_storage_ready():
        return False
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        regions_col = db.collection("users").document(uid).collection("libraryCategories")
        for order, region_id in enumerate(ordered_ids):
            regions_col.document(region_id).update({"order": order})
        return True
    except Exception as exc:
        _log.warning("reorder_library_regions uid=%s: %s", uid, exc)
        return False


def delete_library_region(uid: str, region_id: str) -> dict:
    """Delete a region (and its subcategory docs). Any export referencing
    this region (or an equipment subcategory scoped under it) is reassigned
    to Uncategorized (regionId=None, equipmentId=None) -- GIFs are NEVER
    deleted. Returns {"deleted": bool, "reassigned_count": int}.
    """
    result = {"deleted": False, "reassigned_count": 0}
    if not firebase_storage_ready():
        return result
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        region_ref = db.collection("users").document(uid).collection("libraryCategories").document(region_id)
        if not region_ref.get().exists:
            return result
        # Reassign every export pointing at this region (equipment is scoped
        # per-region, so clearing regionId must also clear equipmentId --
        # an equipment id from a deleted region is meaningless without it).
        exports_col = db.collection("users").document(uid).collection("exports")
        hits = list(exports_col.where("regionId", "==", region_id).stream())
        for snap in hits:
            snap.reference.update({"regionId": None, "equipmentId": None})
        result["reassigned_count"] = len(hits)
        # Delete subcategory docs, then the region doc itself.
        for sub in region_ref.collection("subcategories").stream():
            sub.reference.delete()
        region_ref.delete()
        result["deleted"] = True
        return result
    except Exception as exc:
        _log.warning("delete_library_region uid=%s region_id=%s: %s", uid, region_id, exc)
        return result


def create_library_equipment(uid: str, region_id: str, name: str) -> dict | None:
    if not firebase_storage_ready():
        return None
    try:
        from firebase_admin import firestore as _fs
        from datetime import datetime, timezone
        db = _fs.client()
        region_ref = db.collection("users").document(uid).collection("libraryCategories").document(region_id)
        if not region_ref.get().exists:
            return None
        sub_col = region_ref.collection("subcategories")
        existing = list(sub_col.stream())
        order = len(existing)
        ref = sub_col.document()
        ref.set({"name": (name or "").strip() or "Untitled", "order": order, "createdAt": datetime.now(timezone.utc)})
        return {"id": ref.id, "name": (name or "").strip() or "Untitled", "order": order}
    except Exception as exc:
        _log.warning("create_library_equipment uid=%s region_id=%s: %s", uid, region_id, exc)
        return None


def rename_library_equipment(uid: str, region_id: str, sub_id: str, name: str) -> bool:
    if not firebase_storage_ready():
        return False
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        ref = (
            db.collection("users").document(uid).collection("libraryCategories")
            .document(region_id).collection("subcategories").document(sub_id)
        )
        if not ref.get().exists:
            return False
        ref.update({"name": (name or "").strip() or "Untitled"})
        return True
    except Exception as exc:
        _log.warning("rename_library_equipment uid=%s region_id=%s sub_id=%s: %s", uid, region_id, sub_id, exc)
        return False


def reorder_library_equipment(uid: str, region_id: str, ordered_ids: list[str]) -> bool:
    if not firebase_storage_ready():
        return False
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        sub_col = (
            db.collection("users").document(uid).collection("libraryCategories")
            .document(region_id).collection("subcategories")
        )
        for order, sub_id in enumerate(ordered_ids):
            sub_col.document(sub_id).update({"order": order})
        return True
    except Exception as exc:
        _log.warning("reorder_library_equipment uid=%s region_id=%s: %s", uid, region_id, exc)
        return False


def delete_library_equipment(uid: str, region_id: str, sub_id: str) -> dict:
    """Delete an equipment subcategory. Any export referencing it is
    reassigned to Uncategorized-within-region (equipmentId=None; regionId is
    left alone -- the region itself is still valid). GIFs are NEVER deleted.
    Returns {"deleted": bool, "reassigned_count": int}.
    """
    result = {"deleted": False, "reassigned_count": 0}
    if not firebase_storage_ready():
        return result
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        sub_ref = (
            db.collection("users").document(uid).collection("libraryCategories")
            .document(region_id).collection("subcategories").document(sub_id)
        )
        if not sub_ref.get().exists:
            return result
        exports_col = db.collection("users").document(uid).collection("exports")
        hits = list(exports_col.where("equipmentId", "==", sub_id).stream())
        for snap in hits:
            snap.reference.update({"equipmentId": None})
        result["reassigned_count"] = len(hits)
        sub_ref.delete()
        result["deleted"] = True
        return result
    except Exception as exc:
        _log.warning("delete_library_equipment uid=%s region_id=%s sub_id=%s: %s", uid, region_id, sub_id, exc)
        return result


def upload_runpod_input_video(*, job_id: str, filename: str, local_path: Path) -> str:
    """Upload a job input video and return a signed Firebase media URL."""
    if not firebase_storage_ready():
        raise RuntimeError("Firebase Storage is not configured")
    if not local_path.is_file():
        raise FileNotFoundError("RunPod input video missing")
    safe_name = (filename or "input.mp4").strip().replace("/", "_").replace("\\", "_")
    if not safe_name:
        safe_name = "input.mp4"
    guessed, _ = mimetypes.guess_type(safe_name)
    content_type = guessed or "video/mp4"
    object_path = f"runpod-inputs/{job_id}/{safe_name}"
    return _upload_file(local_path, object_path, content_type)


def get_quota_counter_from_firestore(uid: str) -> int:
    """Return the quota_used counter stored on the user's Firestore document."""
    if not firebase_storage_ready():
        return 0
    try:
        from firebase_admin import firestore as _fs
        db = _fs.client()
        doc = db.collection("users").document(uid).get()
        if doc.exists:
            return int(doc.to_dict().get("quota_used", 0))
        return 0
    except Exception:
        return 0


def increment_quota_counter_in_firestore(uid: str) -> None:
    """Atomically increment the quota_used counter on the user's Firestore document."""
    if not firebase_storage_ready():
        return
    try:
        from firebase_admin import firestore as _fs
        from google.cloud.firestore_v1 import Increment
        db = _fs.client()
        db.collection("users").document(uid).set(
            {"quota_used": Increment(1)},
            merge=True,
        )
    except Exception:
        pass
