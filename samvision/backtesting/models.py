"""Typed containers for the comparable-sales backtest.

Money is ``Decimal``. Nothing here writes to a database. Reports are sanitized:
MLS is masked, and private fields are never carried into serializable output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Optional

# --- skip / outcome reason codes ------------------------------------------
MISSING_MLS = "MISSING_MLS"
MISSING_SOLD_DATE = "MISSING_SOLD_DATE"
INVALID_SOLD_PRICE = "INVALID_SOLD_PRICE"
INVALID_LIVING_AREA = "INVALID_LIVING_AREA"
MISSING_PROPERTY_TYPE = "MISSING_PROPERTY_TYPE"
MISSING_GEOGRAPHY = "MISSING_GEOGRAPHY"
INSUFFICIENT_COMPARABLES = "INSUFFICIENT_COMPARABLES"
ENGINE_ERROR = "ENGINE_ERROR"

# Pre-run eligibility skips (subject never reaches the engine).
ELIGIBILITY_SKIP_CODES = (
    MISSING_MLS,
    MISSING_SOLD_DATE,
    INVALID_SOLD_PRICE,
    INVALID_LIVING_AREA,
    MISSING_PROPERTY_TYPE,
    MISSING_GEOGRAPHY,
)

# --- failure categories (heuristic, for the largest-error report) ---------
FC_INSUFFICIENT_COMPS = "INSUFFICIENT_COMPS"
FC_WIDE_PRICE_SPREAD = "WIDE_PRICE_SPREAD"
FC_GEOGRAPHIC_WIDENING = "GEOGRAPHIC_WIDENING"
FC_LOW_SIMILARITY = "LOW_SIMILARITY"
FC_LOW_DATA_COVERAGE = "LOW_DATA_COVERAGE"
FC_PROPERTY_OUTLIER = "PROPERTY_OUTLIER"
FC_MARKET_TIMING = "MARKET_TIMING"
FC_SIZE_MISMATCH = "SIZE_MISMATCH"
FC_UNKNOWN = "UNKNOWN"

# Fields that must never appear in any written report.
REPORT_BANNED_FIELDS = frozenset({
    "address", "linc_number", "normalized_property_id", "postal_code",
    "agent", "agent_name", "agent_phone", "agent_email", "brokerage",
    "office", "phone", "email", "remarks", "public_remarks", "private_remarks",
    "showing_instructions", "showing", "lockbox", "appointment", "contact",
})


def mask_mls(mls: Optional[str]) -> str:
    """Mask an MLS number to its last 3 characters (privacy)."""
    if not mls:
        return "<none>"
    s = str(mls)
    return ("*" * max(0, len(s) - 3)) + s[-3:] if len(s) > 3 else "*" * len(s)


@dataclass
class HeldOutSubject:
    """A canonical row treated as the held-out subject for one LOO trial."""
    mls_number: str
    sold_date: date
    actual_sold_price: Decimal
    row: dict[str, Any]  # raw canonical mapping used to build SubjectProperty


@dataclass
class BacktestCase:
    """A successfully valued held-out trial (status == ok)."""
    mls_number: str
    area_code: Optional[str]
    neighbourhood: Optional[str]
    property_type: Optional[str]
    sold_date: date
    actual_sold_price: Decimal
    indicated_value: Decimal
    absolute_error: Decimal
    percentage_error: Decimal  # fraction, e.g. 0.0731 == 7.31%
    confidence: str
    tier: Optional[str]
    tier_widened: bool
    selected_comparable_count: int
    median_similarity: Optional[float]
    price_spread: Optional[float]
    mean_coverage: float
    warnings: list[str] = field(default_factory=list)
    missing_subject_fields: list[str] = field(default_factory=list)
    failure_category: str = FC_UNKNOWN

    def sanitized_dict(self) -> dict[str, Any]:
        """Serializable, privacy-safe view (MLS masked, no private fields)."""
        return {
            "mls_masked": mask_mls(self.mls_number),
            "area_code": self.area_code,
            "neighbourhood": self.neighbourhood,
            "property_type": self.property_type,
            "sold_date": self.sold_date.isoformat(),
            "actual_sold_price": str(self.actual_sold_price),
            "indicated_value": str(self.indicated_value),
            "absolute_error": str(self.absolute_error),
            "percentage_error": f"{self.percentage_error:.4f}",
            "confidence": self.confidence,
            "tier": self.tier,
            "tier_widened": self.tier_widened,
            "selected_comparable_count": self.selected_comparable_count,
            "median_similarity": self.median_similarity,
            "price_spread": self.price_spread,
            "mean_coverage": round(self.mean_coverage, 4),
            "warnings": ";".join(self.warnings),
            "missing_subject_fields": ";".join(self.missing_subject_fields),
            "failure_category": self.failure_category,
        }


@dataclass
class SkippedCase:
    """A held-out row that was not valued (ineligible, insufficient, or error)."""
    mls_number: Optional[str]
    reason_code: str

    def sanitized_dict(self) -> dict[str, Any]:
        return {"mls_masked": mask_mls(self.mls_number), "reason_code": self.reason_code}


@dataclass
class GroupMetrics:
    """Metrics for one segment (e.g. one neighbourhood)."""
    group_key: str
    n: int
    stable: bool
    mae: Optional[Decimal]
    median_ae: Optional[Decimal]
    mape_pct: Optional[float]
    median_ape_pct: Optional[float]
    within_10_pct: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group_key,
            "n": self.n,
            "stable": self.stable,
            "mae": str(self.mae) if self.mae is not None else None,
            "median_ae": str(self.median_ae) if self.median_ae is not None else None,
            "mape_pct": self.mape_pct,
            "median_ape_pct": self.median_ape_pct,
            "within_10_pct": self.within_10_pct,
        }


@dataclass
class AggregateReport:
    """Full backtest outcome: counts, overall metrics, and grouped metrics."""
    total_records: int
    eligible: int
    valued: int
    insufficient: int
    skipped_ineligible: int
    engine_errors: int
    coverage_rate: float
    overall: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    distribution: dict[str, str] = field(default_factory=dict)
    grouped: dict[str, list[GroupMetrics]] = field(default_factory=dict)
    skip_reason_counts: dict[str, int] = field(default_factory=dict)
    cases: list[BacktestCase] = field(default_factory=list)
    skips: list[SkippedCase] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        """Aggregate JSON view — no per-case private data."""
        return {
            "counts": {
                "total_records": self.total_records,
                "eligible": self.eligible,
                "valued": self.valued,
                "insufficient_comparables": self.insufficient,
                "skipped_ineligible": self.skipped_ineligible,
                "engine_errors": self.engine_errors,
            },
            "coverage_rate": round(self.coverage_rate, 4),
            "overall_metrics": self.overall,
            "within_threshold_pct": self.thresholds,
            "absolute_error_distribution": self.distribution,
            "skip_reason_counts": self.skip_reason_counts,
            "grouped_metrics": {
                dim: [g.to_dict() for g in groups] for dim, groups in self.grouped.items()
            },
        }
