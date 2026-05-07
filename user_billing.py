"""Persisted Stripe subscription → plan tier (survives new sessions)."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request

from billing_plans import plan_tier_from_session, tier_is_paid

_log = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parent
USER_BILLING_COLLECTION = (os.environ.get("RVM_FIRESTORE_BILLING_COLLECTION") or "userBilling").strip() or "userBilling"
_BILLING_PATH = Path(os.environ.get("RVM_BILLING_STORE", str(APP_ROOT / "data" / "user_billing.json"))).resolve()
_lock = threading.Lock()


@dataclass
class BillingState:
    plan_tier: str = "free"
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None
    subscription_status: str | None = None
    current_period_start: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_tier": self.plan_tier,
            "stripe_customer_id": self.stripe_customer_id,
            "stripe_subscription_id": self.stripe_subscription_id,
            "subscription_status": self.subscription_status,
            "current_period_start": self.current_period_start,
            "updated_at": int(time.time()),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BillingState:
        return cls(
            plan_tier=str(d.get("plan_tier") or "free").strip().lower() or "free",
            stripe_customer_id=(str(d["stripe_customer_id"]).strip() if d.get("stripe_customer_id") else None),
            stripe_subscription_id=(str(d["stripe_subscription_id"]).strip() if d.get("stripe_subscription_id") else None),
            subscription_status=(str(d["subscription_status"]).strip() if d.get("subscription_status") else None),
            current_period_start=int(d["current_period_start"]) if d.get("current_period_start") is not None else None,
        )


def _read_all() -> dict[str, dict[str, Any]]:
    if not _BILLING_PATH.is_file():
        return {}
    try:
        raw = json.loads(_BILLING_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = v
    return out


def _write_all(data: dict[str, dict[str, Any]]) -> None:
    _BILLING_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _BILLING_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=0, sort_keys=True), encoding="utf-8")
    tmp.replace(_BILLING_PATH)


def _firestore_billing_ready() -> bool:
    try:
        from firebase_storage_admin import firebase_storage_ready

        return firebase_storage_ready()
    except Exception:
        return False


def _coerce_fs_billing_dict(data: dict[str, Any]) -> dict[str, Any]:
    cps = data.get("current_period_start")
    if cps is not None and hasattr(cps, "timestamp"):
        cps = int(cps.timestamp())
    elif cps is not None:
        try:
            cps = int(cps)
        except (TypeError, ValueError):
            cps = None
    return {
        "plan_tier": data.get("plan_tier") or "free",
        "stripe_customer_id": data.get("stripe_customer_id"),
        "stripe_subscription_id": data.get("stripe_subscription_id"),
        "subscription_status": data.get("subscription_status"),
        "current_period_start": cps,
    }


def _read_billing_firestore(uid: str) -> BillingState | None:
    if not _firestore_billing_ready():
        return None
    try:
        from firebase_admin import firestore

        snap = firestore.client().collection(USER_BILLING_COLLECTION).document(uid).get()
        if not snap.exists:
            return None
        raw = snap.to_dict()
        if not isinstance(raw, dict):
            return None
        return BillingState.from_dict(_coerce_fs_billing_dict(raw))
    except Exception as exc:
        _log.warning("Firestore read billing failed for %s: %s", uid, exc)
        return None


def _write_billing_firestore(uid: str, state: BillingState) -> None:
    if not _firestore_billing_ready():
        return
    try:
        from firebase_admin import firestore

        firestore.client().collection(USER_BILLING_COLLECTION).document(uid).set(state.to_dict(), merge=True)
    except Exception as exc:
        _log.warning("Firestore write billing failed for %s: %s", uid, exc)


def read_billing(uid: str | None) -> BillingState | None:
    if not uid or not str(uid).strip():
        return None
    uid = str(uid).strip()
    fs_row = _read_billing_firestore(uid)
    if fs_row is not None:
        return fs_row
    with _lock:
        all_d = _read_all()
        row = all_d.get(uid)
    if not row:
        return None
    try:
        return BillingState.from_dict(row)
    except (TypeError, ValueError):
        return None


def write_billing(uid: str, state: BillingState) -> None:
    uid = str(uid).strip()
    if not uid:
        return
    with _lock:
        all_d = _read_all()
        all_d[uid] = state.to_dict()
        _write_all(all_d)
    _write_billing_firestore(uid, state)


def merge_billing(uid: str, **updates: Any) -> BillingState:
    prev = read_billing(uid) or BillingState()
    d = prev.to_dict()
    for k, v in updates.items():
        if k in d:
            d[k] = v
    st = BillingState.from_dict(d)
    write_billing(uid, st)
    return st


_ACTIVE_SUB = frozenset({"active", "trialing"})


def effective_plan_tier(request: Request) -> str:
    """Paid tier from persisted Stripe state wins over session."""
    uid = (request.session.get("user_id") or "").strip()
    if uid:
        st = read_billing(uid)
        if st and st.subscription_status in _ACTIVE_SUB and tier_is_paid(st.plan_tier):
            return plan_tier_from_session({"plan_tier": st.plan_tier})
    return plan_tier_from_session(dict(request.session))


def billing_period_key_for_uid(uid: str | None) -> str | None:
    """Stripe subscription period start (unix); None = use lifetime counter (free)."""
    if not uid:
        return None
    st = read_billing(str(uid).strip())
    if not st or not tier_is_paid(st.plan_tier):
        return None
    if st.subscription_status not in _ACTIVE_SUB:
        return None
    if st.current_period_start is None:
        return None
    return str(int(st.current_period_start))
