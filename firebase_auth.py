"""Public Firebase Web config + server-side ID token verification (no service account file)."""

from __future__ import annotations

import os
from typing import Any

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

# Defaults match project `gifremovel`. Override any key with FIREBASE_* env vars in production.
_DEFAULT_WEB: dict[str, str] = {
    "apiKey": "AIzaSyBX4DUzGqIOSClzoFOHVaBsOb9Yd6Njyy0",
    "authDomain": "gifremovel.firebaseapp.com",
    "projectId": "gifremovel",
    "storageBucket": "gifremovel.firebasestorage.app",
    "messagingSenderId": "716190162020",
    "appId": "1:716190162020:web:b8b8cfd96fced49b9f1371",
    "measurementId": "G-FZKSC6J7ET",
}


def get_firebase_web_config() -> dict[str, str]:
    """Config safe to expose to the browser (Firebase Web SDK)."""
    return {
        "apiKey": os.environ.get("FIREBASE_API_KEY", _DEFAULT_WEB["apiKey"]),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", _DEFAULT_WEB["authDomain"]),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", _DEFAULT_WEB["projectId"]),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", _DEFAULT_WEB["storageBucket"]),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", _DEFAULT_WEB["messagingSenderId"]),
        "appId": os.environ.get("FIREBASE_APP_ID", _DEFAULT_WEB["appId"]),
        "measurementId": os.environ.get("FIREBASE_MEASUREMENT_ID", _DEFAULT_WEB["measurementId"]),
    }


def firebase_project_id() -> str:
    return get_firebase_web_config()["projectId"]


def get_firebase_functions_region() -> str:
    """Region for callable Cloud Functions (signup OTP). Default us-central1."""
    r = (os.environ.get("FIREBASE_FUNCTIONS_REGION") or "us-central1").strip()
    return r or "us-central1"


def verify_firebase_id_token(raw_token: str) -> dict[str, Any]:
    raw = (raw_token or "").strip()
    if not raw:
        raise ValueError("missing id token")
    request = google_requests.Request()
    audience = firebase_project_id()
    return id_token.verify_firebase_token(raw, request, audience=audience)
