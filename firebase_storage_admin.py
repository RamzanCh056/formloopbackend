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
            rows.append({
                "job_id": job_id,
                "export_doc_id": snap.id,
                "source_filename": (data.get("title") or "").strip() or None,
                "gif_url": gif_url,
                "webm_url": (data.get("webmUrl") or "").strip() or None,
                "created_at": (data.get("createdAt") or ""),
            })
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return rows
    except Exception as exc:
        _log.warning("Firestore list_user_exports uid=%s: %s", uid, exc)
        return []


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
        return deleted
    except Exception as exc:
        _log.warning("Firestore delete_user_export uid=%s job_id=%s: %s", uid, job_id, exc)
        return False


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
