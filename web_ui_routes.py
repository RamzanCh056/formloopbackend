"""SSR pages for FormLoop dashboard (Jinja). Wired from api_server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from typing import Any
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from billing_plans import (
    FREE_TIER_GIF_LIMIT,
    YEARLY_DISCOUNT,
    exports_watermark_for_tier,
    free_tier_bullets,
    gif_limit_for_tier,
    plan_dict_for_template,
    plan_spec_by_key,
    tier_is_paid,
    PLANS_ORDERED,
    plan_display_name,
)
from firebase_auth import get_firebase_functions_region, get_firebase_web_config, verify_firebase_id_token
from user_billing import billing_period_key_for_uid, effective_plan_tier, read_billing
from output_job_store import list_matte_gifs_for_owner, read_job_owner, read_quota_usage

_log = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parent


def _bootstrap_env_from_dotenv() -> None:
    """Load ``RobustVideoMatting/.env`` so Stripe keys exist when this module is imported (same folder as api_server)."""
    p = APP_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(p, override=False)
    except ImportError:
        if not p.is_file():
            return
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError:
            return
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, _, val = s.partition("=")
            key = key.strip()
            if not key or key in os.environ:
                continue
            val = val.strip().strip("'").strip('"')
            os.environ[key] = val


_bootstrap_env_from_dotenv()

OUTPUTS_DIR = Path(os.environ.get("RVM_OUTPUTS_DIR", str(APP_ROOT / "api_outputs"))).resolve()
TEMPLATES_DIR = APP_ROOT / "templates"
SESSION_SECRET = os.environ.get("RVM_SESSION_SECRET", "dev-change-me-use-long-random-string").encode()


def _quota_enforced() -> bool:
    return True


def _establish_firebase_session(request: Request, claims: dict[str, Any], name: str, next_path: str) -> JSONResponse:
    uid = claims.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid token claims")
    email = (claims.get("email") or "").strip()
    request.session["user_id"] = str(uid)
    request.session["login_label"] = email or "signed-in@formloop.app"
    request.session.setdefault("member_since_ts", time.time())
    request.session.setdefault("plan_tier", "free")
    nm = (name or "").strip()[:120]
    if nm:
        request.session["user_name"] = nm
    return JSONResponse({"ok": True, "redirect": _safe_next_path(next_path)})

_JOB_HEX = re.compile(r"^[0-9a-f]{32}$")
router = APIRouter(tags=["ui"])
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
def _public_base(request: Request) -> str:
    fixed = os.environ.get("RVM_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if fixed:
        return fixed
    return str(request.base_url).rstrip("/")


def _storage_urls(job_id: str) -> dict[str, str] | None:
    path = OUTPUTS_DIR / job_id / ".storage_urls.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    out: dict[str, str] = {}
    gu = raw.get("gifUrl")
    if isinstance(gu, str) and gu.strip():
        out["gifUrl"] = gu.strip()
    wu = raw.get("webmUrl")
    if isinstance(wu, str) and wu.strip():
        out["webmUrl"] = wu.strip()
    return out or None


def _gif_url(request: Request, job_id: str) -> str:
    side = _storage_urls(job_id)
    if side and side.get("gifUrl"):
        return side["gifUrl"]
    base = f"{_public_base(request)}/api/v1/matte/files/{job_id}/matte.gif"
    gif_path = OUTPUTS_DIR / job_id / "matte.gif"
    try:
        v = int(gif_path.stat().st_mtime)
    except OSError:
        v = int(time.time())
    return f"{base}?v={v}"


def _webm_url(request: Request, job_id: str) -> str | None:
    side = _storage_urls(job_id)
    if side and side.get("webmUrl"):
        return side["webmUrl"]
    webm_path = OUTPUTS_DIR / job_id / "matte_transparent.webm"
    if not webm_path.is_file():
        return None
    return f"{_public_base(request)}/api/v1/matte/files/{job_id}/matte_transparent.webm"


def _redirect_after_gif_delete(request: Request) -> str:
    ref = (request.headers.get("referer") or "").strip()
    default = "/dashboard/gifs"
    if not ref:
        return default
    try:
        u = urlparse(ref)
        b = urlparse(str(request.base_url))
        if u.netloc != b.netloc:
            return default
        path = (u.path or "").rstrip("/") or "/"
        if path == "/dashboard":
            return "/dashboard"
        if path == "/dashboard/gifs":
            return "/dashboard/gifs"
    except Exception:
        pass
    return default


def _scan_gifs(request: Request, limit: int | None = None) -> list[dict]:
    """Legacy: all jobs with a GIF (no owner filter). Prefer `_user_gif_entries` for signed-in UI."""
    items: list[dict] = []
    if not OUTPUTS_DIR.is_dir():
        return items
    dirs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and _JOB_HEX.match(p.name)]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in dirs:
        if not (p / "matte.gif").is_file():
            continue
        jid = p.name
        items.append(
            {
                "job_id": jid,
                "source_filename": None,
                "gif_url": _gif_url(request, jid),
                "webm_url": _webm_url(request, jid),
            }
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def _user_gif_entries(request: Request, limit: int | None = None) -> list[dict]:
    uid = request.session.get("user_id")
    if not uid:
        return []

    # Primary: Firestore — survives Railway redeployments (ephemeral filesystem)
    try:
        from firebase_storage_admin import list_user_exports_from_firestore
        fs_rows = list_user_exports_from_firestore(str(uid), limit=limit)
        if fs_rows:
            return fs_rows
    except Exception:
        _log.debug("Firestore gif fetch skipped, falling back to local scan", exc_info=True)

    # Fallback: local api_outputs/ scan (local dev or non-Firebase envs)
    rows = list_matte_gifs_for_owner(str(uid))
    items: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        jid = row["job_id"]
        seen.add(jid)
        items.append(
            {
                "job_id": jid,
                "source_filename": None,
                "gif_url": _gif_url(request, jid),
                "webm_url": _webm_url(request, jid),
            }
        )
    # Backward compatibility: older/manual jobs may miss `.owner`.
    if limit is None or len(items) < limit:
        if OUTPUTS_DIR.is_dir():
            dirs = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and _JOB_HEX.match(p.name)]
            dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for p in dirs:
                jid = p.name
                if jid in seen:
                    continue
                if not (p / "matte.gif").is_file():
                    continue
                if not (p / ".saved").is_file():
                    continue
                if read_job_owner(p) is not None:
                    continue
                items.append(
                    {
                        "job_id": jid,
                        "source_filename": None,
                        "gif_url": _gif_url(request, jid),
                        "webm_url": _webm_url(request, jid),
                    }
                )
                seen.add(jid)
                if limit is not None and len(items) >= limit:
                    break
    if limit is not None:
        items = items[:limit]
    return items


def _guest_user():
    return SimpleNamespace(
        id="",
        email="Guest",
        is_premium=False,
        plan_tier="guest",
        plan_label="Guest",
        gif_limit=None,
        payment_status="none",
        stripe_customer_id=None,
        created_at=0.0,
    )


def _safe_next_path(nxt: str, default: str = "/dashboard") -> str:
    nxt = (nxt or "").strip()
    if nxt.startswith("/") and not nxt.startswith("//"):
        return nxt
    return default


def _session_user(request: Request):
    uid = request.session.get("user_id")
    if not uid:
        return None
    email = (request.session.get("login_label") or "you@formloop.app").strip()
    request.session.setdefault("plan_tier", "free")
    tier = effective_plan_tier(request)
    bill = read_billing(str(uid))
    cust = bill.stripe_customer_id if bill else None
    return SimpleNamespace(
        id=str(uid),
        email=email,
        is_premium=tier_is_paid(tier),
        plan_tier=tier,
        plan_label=plan_display_name(tier),
        gif_limit=gif_limit_for_tier(tier),
        payment_status="none",
        stripe_customer_id=cust,
        created_at=float(request.session.get("member_since_ts") or time.time()),
    )


@router.get("/auth/login", response_class=HTMLResponse)
async def login_get(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "error": None,
            "next": request.query_params.get("next") or "",
            "firebase_config": get_firebase_web_config(),
        },
    )


@router.get("/auth/signup", response_class=HTMLResponse)
async def signup_get(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse(
        request,
        "signup.html",
        {
            "next": request.query_params.get("next") or "",
            "firebase_config": get_firebase_web_config(),
            "firebase_functions_region": get_firebase_functions_region(),
            "free_tier_gif_limit": FREE_TIER_GIF_LIMIT,
        },
    )


class FirebaseSessionBody(BaseModel):
    id_token: str = Field(..., min_length=10)
    next: str = ""
    name: str = Field("", max_length=120)


@router.post("/auth/session")
async def auth_session_from_firebase(request: Request, body: FirebaseSessionBody):
    try:
        claims = verify_firebase_id_token(body.id_token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Missing or invalid token") from None
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired sign-in. Try again or re-open the sign-in page.",
        ) from None
    return _establish_firebase_session(request, claims, body.name, body.next)


@router.post("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/dashboard", status_code=302)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_home(request: Request):
    user = _session_user(request)
    guest_mode = user is None
    if guest_mode:
        user = _guest_user()
        recent: list[dict] = []
        n = 0
        used_count = 0
    else:
        all_gifs = _user_gif_entries(request, limit=None)
        n = len(all_gifs)
        used_count = read_quota_usage(str(user.id), billing_period_key_for_uid(str(user.id)))
        recent = _user_gif_entries(request, limit=12)
    account_email = ""
    account_name = ""
    if not guest_mode and user:
        account_email = (request.session.get("login_label") or getattr(user, "email", "") or "").strip()
        account_name = (request.session.get("user_name") or "").strip()
    tier = getattr(user, "plan_tier", "free") or "free"
    gif_limit = getattr(user, "gif_limit", None)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "gif_count": n,
            "used_count": used_count,
            "gif_limit": gif_limit,
            "show_watermark_note": exports_watermark_for_tier(tier),
            "guest_mode": guest_mode,
            "recent_gifs": recent,
            "account_email": account_email,
            "account_name": account_name,
            "firebase_config": get_firebase_web_config(),
            "firebase_functions_region": get_firebase_functions_region(),
            "session_uid": str(request.session.get("user_id") or ""),
            "session_email": account_email,
            "dev_test_asset_video_url": (os.environ.get("RVM_TEST_ASSET_VIDEO_URL", "") or "").strip(),
        },
    )


@router.get("/dashboard/gifs", response_class=HTMLResponse)
def dashboard_gifs(request: Request):
    user = _session_user(request)
    guest_mode = user is None
    if guest_mode:
        user = _guest_user()
        items: list[dict] = []
    else:
        items = _user_gif_entries(request, limit=None)
    tier = getattr(user, "plan_tier", "free") or "free"
    gif_limit = getattr(user, "gif_limit", None)
    n = len(items)
    used_count = 0 if guest_mode else read_quota_usage(str(user.id), billing_period_key_for_uid(str(user.id)))
    at_quota = _quota_enforced() and (not guest_mode and gif_limit is not None and used_count >= gif_limit)
    return templates.TemplateResponse(
        request,
        "gifs.html",
        {
            "user": user,
            "items": items,
            "guest_mode": guest_mode,
            "gif_limit": gif_limit,
            "at_quota": at_quota,
            "show_watermark_note": not guest_mode and exports_watermark_for_tier(tier),
            "used_count": used_count,
        },
    )


@router.post("/dashboard/gifs/{job_id}/delete")
async def delete_gif_job(request: Request, job_id: str):
    if _session_user(request) is None:
        return RedirectResponse("/auth/login?next=/dashboard/gifs", status_code=302)
    if not _JOB_HEX.match(job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")
    uid = str(request.session.get("user_id") or "")
    # Remove local job dir if present (local dev / non-ephemeral envs)
    target = (OUTPUTS_DIR / job_id).resolve()
    parent = OUTPUTS_DIR.resolve()
    if target.is_dir() and target.parent == parent:
        owner = read_job_owner(target)
        if owner != uid:
            raise HTTPException(status_code=403, detail="not your clip")
        shutil.rmtree(target, ignore_errors=True)
    # Remove from Firestore (Railway / Firebase envs)
    try:
        from firebase_storage_admin import delete_user_export_from_firestore
        delete_user_export_from_firestore(uid, job_id)
    except Exception:
        _log.debug("Firestore delete skipped for job_id=%s", job_id, exc_info=True)
    return RedirectResponse(_redirect_after_gif_delete(request), status_code=303)


@router.get("/profile", response_class=HTMLResponse)
@router.get("/dashboard/profile", response_class=HTMLResponse)
def profile_page(request: Request):
    user = _session_user(request)
    if not user:
        return RedirectResponse("/auth/login?next=/profile", status_code=302)
    items = _user_gif_entries(request, limit=None)
    n = len(items)
    used_count = read_quota_usage(str(user.id), billing_period_key_for_uid(str(user.id)))
    webm_ready = 0
    for it in items:
        try:
            jid = str(it.get("job_id") or "")
            if jid and (OUTPUTS_DIR / jid / "matte_transparent.webm").is_file():
                webm_ready += 1
        except Exception:
            pass
    quality_metric = 0 if n == 0 else int(round((webm_ready / n) * 100))
    member_since = datetime.fromtimestamp(float(getattr(user, "created_at", 0) or time.time()), tz=timezone.utc).strftime(
        "%B %d, %Y"
    )
    display_email = (request.session.get("login_label") or user.email or "").strip()
    display_name = (request.session.get("user_name") or "").strip()
    tier = getattr(user, "plan_tier", "free") or "free"
    gif_limit = getattr(user, "gif_limit", None)
    plan_name = getattr(user, "plan_label", "Free") or "Free"
    if user.is_premium:
        lim_note = "unlimited GIFs" if gif_limit is None else f"{gif_limit} GIFs per billing month"
        billing_summary = f"You’re on {plan_name} ({lim_note}, no FormLoop watermark)."
    else:
        billing_summary = (
            f"Free tier: up to {FREE_TIER_GIF_LIMIT} GIFs saved to your library, with a small FormLoop watermark on exports. "
            f"You’ve used {used_count} of {FREE_TIER_GIF_LIMIT}."
        )
    return templates.TemplateResponse(
        request,
        "profile.html",
        {
            "user": user,
            "gif_count": n,
            "gif_limit": gif_limit,
            "member_since": member_since,
            "display_email": display_email,
            "display_name": display_name,
            "plan_name": plan_name,
            "billing_summary": billing_summary,
            "stripe_customer_masked": None,
            "free_features": [
                "Transparent GIF exports",
                "Library on the dashboard",
            ],
            "premium_features": [
                "Priority processing",
                "Higher-quality pipeline",
            ],
            "exports_count": n,
            "used_count": used_count,
            "quality_metric": quality_metric,
            "share_link_url": (os.environ.get("RVM_SHARE_LINK_URL", "") or "").strip(),
        },
    )


@router.get("/subscription", response_class=HTMLResponse)
async def subscription_page(request: Request):
    from stripe_integration import ensure_stripe_env

    ensure_stripe_env()
    user = _session_user(request)
    guest_mode = user is None
    if guest_mode:
        user = _guest_user()
    e = (request.query_params.get("e") or "").strip().lower()
    if e == "no_stripe" and (os.environ.get("STRIPE_SECRET_KEY") or "").strip():
        return RedirectResponse("/subscription", status_code=302)
    paid_plans: list[dict[str, Any]] = []
    for p in PLANS_ORDERED:
        row = plan_dict_for_template(p)
        try:
            from stripe_integration import fetch_plan_display_prices

            stripe_row = await asyncio.to_thread(fetch_plan_display_prices, p.key)
            if stripe_row:
                row.update(stripe_row)
        except Exception:
            _log.debug("Stripe display prices unavailable for %s", p.key, exc_info=True)
        paid_plans.append(row)
    header_discount_pct = (
        max(int(pl.get("yearly_discount_pct") or 0) for pl in paid_plans)
        if paid_plans
        else int(YEARLY_DISCOUNT * 100)
    )
    notice = None
    if e == "no_stripe":
        notice = (
            "STRIPE_SECRET_KEY is still missing after reading RobustVideoMatting/.env. "
            "Put the key in that file on this machine or set STRIPE_SECRET_KEY in the host/container environment."
        )
    elif e == "no_price":
        notice = "That billing option isn’t available yet — try the other period or check back later."
    elif e == "bad_plan":
        notice = "Unknown plan. Pick Starter, Pro, or Unlimited."
    elif e == "checkout_pending":
        notice = "Checkout is almost ready — payment will complete here once the server finishes the billing step."
    elif e == "canceled":
        notice = "Checkout was canceled. No charge was made."
    elif e == "bad_session":
        notice = "Missing checkout session. Open Subscription and try again."
    elif e == "verify_failed":
        notice = "We could not verify that payment session. Contact support if you were charged."
    elif e == "wrong_user":
        notice = "That checkout session does not match your signed-in account."
    elif e == "checkout_error":
        notice = "Checkout could not be started. Check Stripe keys and that each Product has monthly/yearly Prices."
    elif e == "success":
        notice = "Welcome to your paid plan — your higher GIF quota and watermark-free exports are active."
    return templates.TemplateResponse(
        request,
        "subscription.html",
        {
            "user": user,
            "guest_mode": guest_mode,
            "paid_plans": paid_plans,
            "free_tier_bullets": free_tier_bullets(),
            "header_discount_pct": header_discount_pct,
            "upgrade_notice": notice,
        },
    )


@router.post("/subscription/checkout")
async def subscription_checkout(
    request: Request,
    plan: str = Form("starter"),
    billing: str = Form("monthly"),
):
    user = _session_user(request)
    if not user:
        return RedirectResponse("/auth/login?next=/subscription", status_code=302)
    base = str(request.base_url).rstrip("/")
    from stripe_integration import ensure_stripe_env

    ensure_stripe_env()
    if not os.environ.get("STRIPE_SECRET_KEY", "").strip():
        return RedirectResponse(f"{base}/subscription?e=no_stripe", status_code=302)
    spec = plan_spec_by_key(plan)
    if not spec:
        return RedirectResponse(f"{base}/subscription?e=bad_plan", status_code=302)
    yearly = (billing or "").strip().lower() in ("yearly", "annual", "year")
    bill = read_billing(str(user.id))
    existing = bill.stripe_customer_id if bill else None
    try:
        from stripe_integration import create_subscription_checkout_session

        sess = await asyncio.to_thread(
            create_subscription_checkout_session,
            uid=str(user.id),
            email=str(getattr(user, "email", "") or ""),
            plan_key=str(spec.key),
            yearly=yearly,
            public_base=base,
            existing_customer_id=existing,
        )
    except ValueError as exc:
        _log.warning("Stripe checkout: bad price/plan — %s", exc)
        return RedirectResponse(f"{base}/subscription?e=no_price", status_code=302)
    except Exception:
        _log.exception("Stripe checkout failed (see server log)")
        return RedirectResponse(f"{base}/subscription?e=checkout_error", status_code=302)
    url = getattr(sess, "url", None) or (sess.get("url") if isinstance(sess, dict) else None)
    if not url:
        return RedirectResponse(f"{base}/subscription?e=checkout_error", status_code=302)
    return RedirectResponse(str(url), status_code=303)


@router.get("/subscription/success", response_class=HTMLResponse)
async def subscription_success(request: Request, session_id: str = ""):
    user = _session_user(request)
    if not user or not str(getattr(user, "id", "") or "").strip():
        return RedirectResponse("/auth/login?next=/subscription", status_code=302)
    base = str(request.base_url).rstrip("/")
    sid = (session_id or request.query_params.get("session_id") or "").strip()
    if not sid:
        return RedirectResponse(f"{base}/subscription?e=bad_session", status_code=302)
    from stripe_integration import ensure_stripe_env

    ensure_stripe_env()
    if not os.environ.get("STRIPE_SECRET_KEY", "").strip():
        return RedirectResponse(f"{base}/subscription?e=no_stripe", status_code=302)
    try:
        from stripe_integration import fulfill_checkout_session

        uid_f, plan_key = await asyncio.to_thread(fulfill_checkout_session, sid)
    except Exception:
        return RedirectResponse(f"{base}/subscription?e=verify_failed", status_code=302)
    if not uid_f or uid_f != str(user.id):
        return RedirectResponse(f"{base}/subscription?e=wrong_user", status_code=302)
    if plan_key:
        request.session["plan_tier"] = plan_key
    return RedirectResponse(f"{base}/subscription?e=success", status_code=302)


@router.get("/upgrade", response_class=HTMLResponse)
async def upgrade_redirect():
    return RedirectResponse("/subscription", status_code=302)
