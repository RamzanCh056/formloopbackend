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
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

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
        return json.loads(fc)
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
    return obj if isinstance(obj, dict) else None


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
