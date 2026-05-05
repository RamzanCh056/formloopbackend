"""Optional Firebase Storage uploads using the Admin SDK (service account).

Set ``GOOGLE_APPLICATION_CREDENTIALS`` to a JSON key path, or set
``FIREBASE_SERVICE_ACCOUNT_JSON`` to the full JSON string. Bucket comes from
``get_firebase_web_config()`` / ``FIREBASE_STORAGE_BUCKET``.

If credentials are missing, upload helpers are no-ops and callers fall back to
local API URLs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

from firebase_auth import get_firebase_web_config

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_ready: bool | None = None


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
    if not json_str and not cred_path:
        return False
    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError:
        _log.warning("firebase-admin not installed; Storage uploads disabled.")
        return False
    cfg = get_firebase_web_config()
    bucket_name = (cfg.get("storageBucket") or "").strip()
    if not bucket_name:
        return False
    try:
        firebase_admin.get_app()
    except ValueError:
        if json_str:
            cred = credentials.Certificate(json.loads(json_str))
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
