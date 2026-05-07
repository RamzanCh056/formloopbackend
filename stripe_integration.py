"""Stripe Checkout + webhooks. Env Price IDs override; else resolve from Product IDs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import stripe

_APP = Path(__file__).resolve().parent


def ensure_stripe_env() -> None:
    """Copy ``STRIPE_SECRET_KEY`` / ``STRIPE_WEBHOOK_SECRET`` from ``RobustVideoMatting/.env`` if missing from ``os.environ``.

    Production and Docker often run without dotenv; the file may still sit next to ``api_server.py``.
    """
    path = _APP / ".env"
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        if key == "STRIPE_SECRET_KEY" and not (os.environ.get("STRIPE_SECRET_KEY") or "").strip():
            os.environ["STRIPE_SECRET_KEY"] = val
        elif key == "STRIPE_WEBHOOK_SECRET" and not (os.environ.get("STRIPE_WEBHOOK_SECRET") or "").strip():
            os.environ["STRIPE_WEBHOOK_SECRET"] = val

from billing_plans import PLANS_ORDERED, plan_spec_by_key, stripe_price_id
from user_billing import BillingState, merge_billing, write_billing

# Default Stripe Product IDs — each Product must have an active recurring Price (month / year).
# Override any via env (optional): STRIPE_PRODUCT_STARTER_MONTHLY, _YEARLY, STRIPE_PRODUCT_PRO_*, etc.
_DEFAULT_PRODUCT_IDS: dict[tuple[str, bool], str] = {
    ("starter", False): "prod_USF1P7ttcc5apr",
    ("starter", True): "prod_USF38Lk88mbzdu",
    ("pro", False): "prod_USF4aqysH9AIs4",
    ("pro", True): "prod_USF5W48av0iAnV",
    ("unlimited", False): "prod_USF62DWwJb7Xtu",
    ("unlimited", True): "prod_USF7dSrWUkDpQk",
}

_PRODUCT_ENV: dict[tuple[str, bool], str] = {
    ("starter", False): "STRIPE_PRODUCT_STARTER_MONTHLY",
    ("starter", True): "STRIPE_PRODUCT_STARTER_YEARLY",
    ("pro", False): "STRIPE_PRODUCT_PRO_MONTHLY",
    ("pro", True): "STRIPE_PRODUCT_PRO_YEARLY",
    ("unlimited", False): "STRIPE_PRODUCT_UNLIMITED_MONTHLY",
    ("unlimited", True): "STRIPE_PRODUCT_UNLIMITED_YEARLY",
}


def product_id_for_plan(plan_key: str, yearly: bool) -> str | None:
    k = (plan_key.lower(), yearly)
    ev = _PRODUCT_ENV.get(k)
    if ev:
        over = (os.environ.get(ev) or "").strip()
        if over:
            return over
    return _DEFAULT_PRODUCT_IDS.get(k)


def configure_stripe() -> None:
    ensure_stripe_env()
    key = (os.environ.get("STRIPE_SECRET_KEY") or "").strip()
    if not key:
        raise RuntimeError("STRIPE_SECRET_KEY is not set")
    stripe.api_key = key


def _checkout_payment_ok(payment_status: str | None) -> bool:
    """Subscription checkouts can be `paid` or `no_payment_required` (e.g. trial)."""
    ps = (payment_status or "").strip()
    return ps in ("paid", "no_payment_required")


def price_id_from_product(product_id: str, *, yearly: bool) -> str | None:
    configure_stripe()
    want = "year" if yearly else "month"
    prices = stripe.Price.list(product=product_id, active=True, limit=30)
    for pr in prices.data:
        rec = pr.recurring
        if rec and getattr(rec, "interval", None) == want:
            return pr.id
    return None


def resolve_checkout_price_id(plan_key: str, yearly: bool) -> str | None:
    spec = plan_spec_by_key(plan_key)
    if not spec:
        return None
    pid = stripe_price_id(spec, yearly=yearly)
    if pid:
        return pid
    prod = product_id_for_plan(plan_key, yearly)
    if not prod:
        return None
    try:
        return price_id_from_product(prod, yearly=yearly)
    except Exception:
        return None


def fetch_plan_display_prices(plan_key: str) -> dict[str, Any] | None:
    """Read monthly/yearly unit amounts from Stripe for the subscription page (no hardcoded dollars).

    Uses the same Price resolution as checkout. Returns None if the key is missing or Stripe errors.
    """
    ensure_stripe_env()
    if not (os.environ.get("STRIPE_SECRET_KEY") or "").strip():
        return None
    try:
        mid = resolve_checkout_price_id(plan_key, yearly=False)
        yid = resolve_checkout_price_id(plan_key, yearly=True)
        if not mid or not yid:
            return None
        configure_stripe()
        pm = stripe.Price.retrieve(mid)
        py = stripe.Price.retrieve(yid)
    except Exception:
        return None
    u_m = getattr(pm, "unit_amount", None)
    u_y = getattr(py, "unit_amount", None)
    if u_m is None or u_y is None:
        return None
    monthly_usd = round(u_m / 100.0, 2)
    yearly_total = round(u_y / 100.0, 2)
    yearly_monthly_equiv = round(yearly_total / 12.0, 2)
    annual_at_monthly_rate = monthly_usd * 12.0
    yearly_discount_pct = 0
    if annual_at_monthly_rate > 0:
        yearly_discount_pct = max(
            0, min(100, int(round((1.0 - yearly_total / annual_at_monthly_rate) * 100.0)))
        )
    return {
        "monthly_usd": monthly_usd,
        "yearly_total": yearly_total,
        "yearly_monthly_equiv": yearly_monthly_equiv,
        "yearly_discount_pct": yearly_discount_pct,
        "prices_from_stripe": True,
    }


def create_subscription_checkout_session(
    *,
    uid: str,
    email: str,
    plan_key: str,
    yearly: bool,
    public_base: str,
    existing_customer_id: str | None,
) -> stripe.checkout.Session:
    configure_stripe()
    price_id = resolve_checkout_price_id(plan_key, yearly=yearly)
    if not price_id:
        raise ValueError("No Stripe price for this plan — add recurring Prices to each Product or set STRIPE_PRICE_* env vars.")

    success = f"{public_base.rstrip('/')}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}"
    cancel = f"{public_base.rstrip('/')}/subscription?e=canceled"

    # Recurring Price → Stripe invoices each period and charges the saved payment method automatically.
    params: dict[str, Any] = {
        "mode": "subscription",
        "client_reference_id": uid,
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": success,
        "cancel_url": cancel,
        "metadata": {"uid": uid, "plan_key": plan_key.lower(), "billing": "yearly" if yearly else "monthly"},
        "subscription_data": {"metadata": {"uid": uid, "plan_key": plan_key.lower()}},
        "allow_promotion_codes": True,
    }
    if existing_customer_id and existing_customer_id.startswith("cus_"):
        params["customer"] = existing_customer_id
    elif email:
        params["customer_email"] = email

    return stripe.checkout.Session.create(**params)


def _meta_dict(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return dict(meta)
    try:
        return dict(meta)
    except Exception:
        return {}


def _subscription_tier_from_price(price_id: str | None) -> str | None:
    if not price_id:
        return None
    for (key, yr), _ in _PRODUCT_ENV.items():
        prod = product_id_for_plan(key, yr)
        if not prod:
            continue
        try:
            configure_stripe()
            prices = stripe.Price.list(product=prod, active=True, limit=40)
            for pr in prices.data:
                if pr.id == price_id:
                    return key
        except Exception:
            continue
    for spec in PLANS_ORDERED:
        if stripe_price_id(spec, yearly=False) == price_id or stripe_price_id(spec, yearly=True) == price_id:
            return spec.key
    return None


def _first_price_id(sub: Any) -> str | None:
    try:
        items = getattr(sub, "items", None)
        data = getattr(items, "data", None) if items else None
        if not data:
            return None
        li0 = data[0]
        pr = getattr(li0, "price", None)
        if pr and getattr(pr, "id", None):
            return str(pr.id)
    except Exception:
        pass
    return None


def _apply_subscription_object(sub: Any) -> None:
    meta = _meta_dict(getattr(sub, "metadata", None))
    uid = (meta.get("uid") or "").strip()
    if not uid:
        return

    status = str(getattr(sub, "status", "") or "")
    price_id = _first_price_id(sub)
    tier = (meta.get("plan_key") or _subscription_tier_from_price(price_id) or "free").strip().lower()
    cps = getattr(sub, "current_period_start", None)
    cus = getattr(sub, "customer", None)
    cus_id = cus if isinstance(cus, str) else getattr(cus, "id", None)
    sub_id = getattr(sub, "id", None)

    if status in ("active", "trialing"):
        merge_billing(
            uid,
            plan_tier=tier,
            stripe_customer_id=cus_id,
            stripe_subscription_id=sub_id,
            subscription_status=status,
            current_period_start=int(cps) if cps else None,
        )
    elif status in ("canceled", "unpaid", "incomplete_expired"):
        write_billing(
            uid,
            BillingState(
                plan_tier="free",
                stripe_customer_id=cus_id,
                stripe_subscription_id=None,
                subscription_status=status,
                current_period_start=None,
            ),
        )
    else:
        merge_billing(
            uid,
            plan_tier=tier,
            stripe_customer_id=cus_id,
            stripe_subscription_id=sub_id,
            subscription_status=status,
            current_period_start=int(cps) if cps else None,
        )


def fulfill_checkout_session(session_id: str) -> tuple[str | None, str | None]:
    configure_stripe()
    sess = stripe.checkout.Session.retrieve(session_id, expand=["subscription"])
    if not _checkout_payment_ok(getattr(sess, "payment_status", None)):
        return None, None

    meta = _meta_dict(getattr(sess, "metadata", None))
    uid = (getattr(sess, "client_reference_id", None) or meta.get("uid") or "").strip()
    plan_key = (meta.get("plan_key") or "starter").strip().lower()
    if not uid:
        return None, None

    sub = getattr(sess, "subscription", None)
    if isinstance(sub, str):
        sub = stripe.Subscription.retrieve(sub, expand=["items.data.price"])
    if sub:
        _apply_subscription_object(sub)
    return uid, plan_key


def handle_webhook_event(event: Any) -> None:
    et = getattr(event, "type", None) or event.get("type")
    data = getattr(event, "data", None) or event.get("data")
    obj = getattr(data, "object", None) if data is not None else None
    if obj is None and isinstance(data, dict):
        obj = data.get("object")

    if et == "checkout.session.completed":
        mode = getattr(obj, "mode", None) if obj is not None else None
        ps = getattr(obj, "payment_status", None) if obj is not None else None
        if mode is None and isinstance(obj, dict):
            mode = obj.get("mode")
            ps = obj.get("payment_status")
        if mode == "subscription" and _checkout_payment_ok(str(ps) if ps is not None else None):
            sid = getattr(obj, "id", None) or (obj.get("id") if isinstance(obj, dict) else None)
            if sid:
                fulfill_checkout_session(str(sid))
        return

    if et in ("customer.subscription.updated", "customer.subscription.deleted"):
        oid = getattr(obj, "id", None) if obj is not None else None
        if oid is None and isinstance(obj, dict):
            oid = obj.get("id")
        if not oid:
            return
        configure_stripe()
        sub = stripe.Subscription.retrieve(str(oid), expand=["items.data.price"])
        meta = _meta_dict(getattr(sub, "metadata", None))
        uid = (meta.get("uid") or "").strip()
        if not uid:
            return
        st = getattr(sub, "status", None) or ""
        if et == "customer.subscription.deleted" or st == "canceled":
            cus = getattr(sub, "customer", None)
            cus_id = cus if isinstance(cus, str) else getattr(cus, "id", None)
            write_billing(
                uid,
                BillingState(
                    plan_tier="free",
                    stripe_customer_id=cus_id,
                    stripe_subscription_id=None,
                    subscription_status="canceled",
                    current_period_start=None,
                ),
            )
        else:
            _apply_subscription_object(sub)
        return

    if et == "invoice.payment_succeeded":
        # Each renewal (and initial invoice): refresh subscription + current_period_start for billing-period GIF quota.
        inv = obj
        if inv is None:
            return
        sub_raw = getattr(inv, "subscription", None)
        if sub_raw is None and isinstance(inv, dict):
            sub_raw = inv.get("subscription")
        if not sub_raw:
            return
        sub_id = sub_raw if isinstance(sub_raw, str) else getattr(sub_raw, "id", None)
        if not sub_id:
            return
        configure_stripe()
        sub = stripe.Subscription.retrieve(str(sub_id), expand=["items.data.price"])
        _apply_subscription_object(sub)
