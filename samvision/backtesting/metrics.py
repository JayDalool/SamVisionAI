"""Deterministic backtest metrics. Money is ``Decimal``; no randomness.

Percentage error is a fraction (``abs(indicated - actual) / actual``). Callers
reject ``actual <= 0`` before ratios. Aggregate helpers operate on already-valued
cases and never touch a database or a model.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Iterable, Optional, Sequence

from . import models
from .models import BacktestCase, GroupMetrics

_CENTS = Decimal("0.01")


def _money(x: Decimal) -> Decimal:
    return x.quantize(_CENTS, rounding=ROUND_HALF_UP)


# --- per-case primitives ---------------------------------------------------
def absolute_error(indicated: Decimal, actual: Decimal) -> Decimal:
    return _money(abs(Decimal(indicated) - Decimal(actual)))


def percentage_error(indicated: Decimal, actual: Decimal) -> Decimal:
    """Absolute percentage error as a fraction. Rejects non-positive actual."""
    actual = Decimal(actual)
    if actual <= 0:
        raise ValueError("actual_sold_price must be positive for percentage error")
    return (abs(Decimal(indicated) - actual) / actual).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )


def squared_error(indicated: Decimal, actual: Decimal) -> Decimal:
    d = Decimal(indicated) - Decimal(actual)
    return d * d


# --- distribution helpers --------------------------------------------------
def _sorted_decimals(values: Iterable[Decimal]) -> list[Decimal]:
    return sorted(Decimal(v) for v in values)


def median(values: Sequence[Decimal]) -> Optional[Decimal]:
    return percentile(values, Decimal("0.5"))


def percentile(values: Sequence[Decimal], q) -> Optional[Decimal]:
    """Linear-interpolation percentile (numpy 'linear'), Decimal-safe.

    ``q`` in [0, 1]. Returns None for an empty input.
    """
    xs = _sorted_decimals(values)
    n = len(xs)
    if n == 0:
        return None
    if n == 1:
        return _money(xs[0])
    q = Decimal(str(q))
    pos = q * Decimal(n - 1)
    lo = int(pos)  # floor for non-negative
    frac = pos - Decimal(lo)
    if lo + 1 >= n:
        return _money(xs[-1])
    interp = xs[lo] + (xs[lo + 1] - xs[lo]) * frac
    return _money(interp)


def mean(values: Sequence[Decimal]) -> Optional[Decimal]:
    xs = list(values)
    if not xs:
        return None
    return _money(sum((Decimal(v) for v in xs), Decimal(0)) / Decimal(len(xs)))


def rmse(squared_errors: Sequence[Decimal]) -> Optional[Decimal]:
    xs = list(squared_errors)
    if not xs:
        return None
    ms = sum((Decimal(v) for v in xs), Decimal(0)) / Decimal(len(xs))
    return _money(ms.sqrt())


def mape_pct(percentage_errors: Sequence[Decimal]) -> Optional[float]:
    """Mean absolute percentage error as a percentage (e.g. 7.31)."""
    xs = list(percentage_errors)
    if not xs:
        return None
    m = sum((Decimal(v) for v in xs), Decimal(0)) / Decimal(len(xs))
    return float((m * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def median_ape_pct(percentage_errors: Sequence[Decimal]) -> Optional[float]:
    xs = _sorted_decimals(percentage_errors)
    if not xs:
        return None
    med = percentile([x * 100 for x in xs], Decimal("0.5"))
    return float(med) if med is not None else None


def within_threshold_pct(percentage_errors: Sequence[Decimal], threshold: float) -> Optional[float]:
    """Share (percentage) of cases with APE <= threshold (fraction)."""
    xs = list(percentage_errors)
    if not xs:
        return None
    t = Decimal(str(threshold))
    hits = sum(1 for v in xs if Decimal(v) <= t)
    return float((Decimal(hits) / Decimal(len(xs)) * 100).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP))


# --- aggregate over valued cases ------------------------------------------
def overall_metrics(cases: Sequence[BacktestCase]) -> dict:
    abs_errs = [c.absolute_error for c in cases]
    pct_errs = [c.percentage_error for c in cases]
    sq_errs = [squared_error(c.indicated_value, c.actual_sold_price) for c in cases]
    mae = mean(abs_errs)
    med = median(abs_errs)
    return {
        "n": len(cases),
        "mae": str(mae) if mae is not None else None,
        "median_absolute_error": str(med) if med is not None else None,
        "rmse": str(rmse(sq_errs)) if cases else None,
        "mape_pct": mape_pct(pct_errs),
        "median_ape_pct": median_ape_pct(pct_errs),
    }


def threshold_table(cases: Sequence[BacktestCase]) -> dict[str, float]:
    pct = [c.percentage_error for c in cases]
    return {
        "within_5pct": within_threshold_pct(pct, 0.05),
        "within_10pct": within_threshold_pct(pct, 0.10),
        "within_15pct": within_threshold_pct(pct, 0.15),
        "within_20pct": within_threshold_pct(pct, 0.20),
    }


def distribution_table(cases: Sequence[BacktestCase]) -> dict[str, str]:
    errs = [c.absolute_error for c in cases]
    if not errs:
        return {}
    def s(q):
        v = percentile(errs, Decimal(str(q)))
        return str(v) if v is not None else None
    return {
        "min": str(min(errs)),
        "p25": s(0.25),
        "median": s(0.50),
        "p75": s(0.75),
        "p90": s(0.90),
        "p95": s(0.95),
        "max": str(max(errs)),
    }


# --- grouped metrics -------------------------------------------------------
def grouped_metrics(
    cases: Sequence[BacktestCase],
    key_fn: Callable[[BacktestCase], Optional[str]],
    *,
    min_group: int = 3,
) -> list[GroupMetrics]:
    """Group valued cases by ``key_fn`` and compute per-group metrics.

    Groups smaller than ``min_group`` are still returned but flagged
    ``stable=False``. Output is ordered by descending n, then group key.
    """
    buckets: dict[str, list[BacktestCase]] = {}
    for c in cases:
        k = key_fn(c)
        buckets.setdefault("<none>" if k is None else str(k), []).append(c)

    out: list[GroupMetrics] = []
    for key, group in buckets.items():
        abs_errs = [c.absolute_error for c in group]
        pct_errs = [c.percentage_error for c in group]
        out.append(GroupMetrics(
            group_key=key,
            n=len(group),
            stable=len(group) >= min_group,
            mae=mean(abs_errs),
            median_ae=median(abs_errs),
            mape_pct=mape_pct(pct_errs),
            median_ape_pct=median_ape_pct(pct_errs),
            within_10_pct=within_threshold_pct(pct_errs, 0.10),
        ))
    out.sort(key=lambda g: (-g.n, g.group_key))
    return out


# --- bucket helpers --------------------------------------------------------
def comp_count_bucket(n: int) -> str:
    if n <= 3:
        return "3"
    if n <= 5:
        return "4-5"
    if n <= 7:
        return "6-7"
    return "8-10"


def similarity_bucket(sim: Optional[float]) -> str:
    if sim is None:
        return "<none>"
    if sim < 50:
        return "<50"
    if sim < 60:
        return "50-59.99"
    if sim < 70:
        return "60-69.99"
    if sim < 80:
        return "70-79.99"
    return "80+"


def coverage_bucket(cov: float) -> str:
    if cov < 0.25:
        return "<0.25"
    if cov < 0.50:
        return "0.25-0.49"
    if cov < 0.75:
        return "0.50-0.74"
    return "0.75+"


def sold_month(c: BacktestCase) -> str:
    return c.sold_date.strftime("%Y-%m")


# --- heuristic failure classification -------------------------------------
def classify_failure(
    *,
    status: str,
    tier_widened: bool,
    median_similarity: Optional[float],
    mean_coverage: float,
    price_spread: Optional[float],
    percentage_error: Optional[Decimal],
    size_mismatch: bool,
) -> str:
    """Best-effort single failure category, ordered by likely dominant cause."""
    if status == "insufficient_data":
        return models.FC_INSUFFICIENT_COMPS
    if price_spread is not None and price_spread > 0.35:
        return models.FC_WIDE_PRICE_SPREAD
    if size_mismatch:
        return models.FC_SIZE_MISMATCH
    if tier_widened:
        return models.FC_GEOGRAPHIC_WIDENING
    if median_similarity is not None and median_similarity < 60:
        return models.FC_LOW_SIMILARITY
    if mean_coverage < 0.50:
        return models.FC_LOW_DATA_COVERAGE
    if percentage_error is not None and percentage_error > Decimal("0.25"):
        # large error despite decent comps -> the sale itself is atypical
        return models.FC_PROPERTY_OUTLIER
    return models.FC_UNKNOWN
