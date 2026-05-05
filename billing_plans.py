"""FormLoop plan tiers: GIF quotas, pricing display, Stripe env key hints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

YEARLY_DISCOUNT = 0.20

FREE_TIER_GIF_LIMIT = 5


@dataclass(frozen=True)
class PlanSpec:
    key: str
    name: str
    monthly_usd: int
    gif_limit: int | None  # None = unlimited
    tagline: str
    features: tuple[str, ...]
    featured: bool
    stripe_price_monthly_env: str
    stripe_price_yearly_env: str
    batch_processing: bool = False


def _discounted_yearly_monthly_equiv(monthly: int) -> float:
    return round(monthly * 12 * (1 - YEARLY_DISCOUNT) / 12.0, 2)


PLANS_ORDERED: tuple[PlanSpec, ...] = (
    PlanSpec(
        key="starter",
        name="Starter",
        monthly_usd=29,
        gif_limit=15,
        tagline="15 transparent GIFs / month · no watermark",
        features=(
            "15 GIFs per billing month",
            "No FormLoop watermark",
            "Standard processing queue",
            "Email support",
        ),
        featured=False,
        stripe_price_monthly_env="STRIPE_PRICE_STARTER_MONTHLY",
        stripe_price_yearly_env="STRIPE_PRICE_STARTER_YEARLY",
    ),
    PlanSpec(
        key="pro",
        name="Pro",
        monthly_usd=79,
        gif_limit=50,
        tagline="50 GIFs / month · batch processing",
        features=(
            "50 GIFs per billing month",
            "Batch processing",
            "Priority queue",
            "Higher-quality pipeline options",
        ),
        featured=True,
        stripe_price_monthly_env="STRIPE_PRICE_PRO_MONTHLY",
        stripe_price_yearly_env="STRIPE_PRICE_PRO_YEARLY",
        batch_processing=True,
    ),
    PlanSpec(
        key="unlimited",
        name="Unlimited",
        monthly_usd=149,
        gif_limit=None,
        tagline="Unlimited GIFs · best for teams",
        features=(
            "Unlimited transparent GIFs",
            "Batch processing",
            "Priority queue",
            "Dedicated throughput",
        ),
        featured=False,
        stripe_price_monthly_env="STRIPE_PRICE_UNLIMITED_MONTHLY",
        stripe_price_yearly_env="STRIPE_PRICE_UNLIMITED_YEARLY",
        batch_processing=True,
    ),
)


def plan_tier_from_session(session: dict[str, Any]) -> str:
    raw = (session.get("plan_tier") or "free").strip().lower()
    allowed = {"free", "starter", "pro", "unlimited"}
    return raw if raw in allowed else "free"


def gif_limit_for_tier(tier: str) -> int | None:
    if tier == "free":
        return FREE_TIER_GIF_LIMIT
    for p in PLANS_ORDERED:
        if p.key == tier:
            return p.gif_limit
    return FREE_TIER_GIF_LIMIT


def tier_is_paid(tier: str) -> bool:
    return tier in {p.key for p in PLANS_ORDERED}


def plan_spec_by_key(key: str) -> PlanSpec | None:
    k = (key or "").strip().lower()
    for p in PLANS_ORDERED:
        if p.key == k:
            return p
    return None


def plan_display_name(tier: str) -> str:
    if tier == "free":
        return "Free"
    for p in PLANS_ORDERED:
        if p.key == tier:
            return p.name
    return "Free"


def exports_watermark_for_tier(tier: str) -> bool:
    """Free (and unknown) accounts get the FormLoop GIF watermark."""
    return plan_tier_from_session({"plan_tier": tier}) == "free"


def stripe_price_id(plan: PlanSpec, yearly: bool) -> str:
    env = plan.stripe_price_yearly_env if yearly else plan.stripe_price_monthly_env
    return (os.environ.get(env) or "").strip()


def plan_dict_for_template(p: PlanSpec) -> dict[str, Any]:
    y_eq = _discounted_yearly_monthly_equiv(p.monthly_usd)
    return {
        "key": p.key,
        "name": p.name,
        "monthly_usd": p.monthly_usd,
        "yearly_monthly_equiv": y_eq,
        "yearly_total": round(p.monthly_usd * 12 * (1 - YEARLY_DISCOUNT), 2),
        "tagline": p.tagline,
        "features": list(p.features),
        "featured": p.featured,
        "stripe_monthly": stripe_price_id(p, yearly=False),
        "stripe_yearly": stripe_price_id(p, yearly=True),
        "has_monthly": bool(stripe_price_id(p, yearly=False)),
        "has_yearly": bool(stripe_price_id(p, yearly=True)),
    }


def free_tier_bullets() -> list[str]:
    return [
        f"{FREE_TIER_GIF_LIMIT} transparent GIFs per account (lifetime)",
        "Exports include a small FormLoop watermark",
        "Upgrade anytime for more GIFs and no watermark",
    ]
