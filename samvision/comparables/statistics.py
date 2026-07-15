"""Decimal-safe price statistics for the comparable engine.

Weighted median / quantiles are deterministic: values sorted ascending, weights
accumulated, the quantile is the first value whose cumulative weight reaches the
threshold. Money is Decimal end-to-end to avoid float rounding drift.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence


class StatisticsError(ValueError):
    """Raised on invalid inputs (zero sqft, non-positive price, empty series)."""


def price_per_sqft(sold_price: Decimal, living_area_sqft: int) -> Decimal:
    if sold_price is None or sold_price <= 0:
        raise StatisticsError("sold_price must be positive")
    if not living_area_sqft or living_area_sqft <= 0:
        raise StatisticsError("living_area_sqft must be positive (zero sqft rejected)")
    return Decimal(sold_price) / Decimal(living_area_sqft)


def size_normalized_price(comp_price: Decimal, subject_sqft: int, comp_sqft: int) -> Decimal:
    """comp_price scaled to the subject's size: price * subject_sqft / comp_sqft."""
    if comp_price is None or comp_price <= 0:
        raise StatisticsError("comp_price must be positive")
    if not subject_sqft or subject_sqft <= 0:
        raise StatisticsError("subject_sqft must be positive")
    if not comp_sqft or comp_sqft <= 0:
        raise StatisticsError("comp_sqft must be positive (zero sqft rejected)")
    return Decimal(comp_price) * Decimal(subject_sqft) / Decimal(comp_sqft)


def weighted_quantile(pairs: Sequence[tuple[Decimal, Decimal]], q: float) -> Decimal:
    """Weighted quantile q in [0,1]. ``pairs`` = (value, weight); weight >= 0.

    Deterministic: sort by value asc (ties broken by original order via a stable
    sort), accumulate weight, return the first value whose cumulative weight
    reaches q * total. Falls back to equal weights when all weights are zero.
    """
    if not pairs:
        raise StatisticsError("cannot compute a quantile of an empty series")
    if not (0.0 <= q <= 1.0):
        raise StatisticsError("q must be in [0, 1]")
    ordered = sorted(pairs, key=lambda p: p[0])
    total = sum((w for _, w in ordered), Decimal(0))
    if total <= 0:
        # all weights zero -> treat as equal weights
        ordered = [(v, Decimal(1)) for v, _ in ordered]
        total = Decimal(len(ordered))
    threshold = Decimal(str(q)) * total
    cumulative = Decimal(0)
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def weighted_median(pairs: Sequence[tuple[Decimal, Decimal]]) -> Decimal:
    return weighted_quantile(pairs, 0.5)


def median(values: Sequence[float]) -> float:
    if not values:
        raise StatisticsError("cannot take the median of an empty series")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


def round_to_nearest(value: Decimal, nearest: int) -> Decimal:
    """Round a Decimal to the nearest multiple of ``nearest`` (no false precision)."""
    if nearest <= 0:
        raise StatisticsError("nearest must be positive")
    steps = (value / Decimal(nearest)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return steps * Decimal(nearest)


def relative_spread(low: Decimal, high: Decimal, mid: Decimal) -> float:
    """(high - low) / mid, as a float. 0 when mid is non-positive."""
    if mid is None or mid <= 0:
        return 0.0
    return float((Decimal(high) - Decimal(low)) / Decimal(mid))
