"""Leave-one-out backtest runner.

For each approved canonical row: treat it as the held-out subject, set
``valuation_date = sold_date``, exclude the subject MLS, run the **unmodified**
comparable service, and compare the indicated value with the actual sold price.

Read-only: this module holds no database handle and no write-authorization flag.
It depends only on a ``ComparableService`` built over any ``ComparableDataSource``.
The service enforces the ``sold_date <= valuation_date`` upper bound and the
subject-MLS exclusion, so no future sale can enter.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from decimal import Decimal
from typing import Any, Iterable, Optional

from samvision.comparables.models import SubjectProperty
from samvision.comparables.service import ComparableService

from . import metrics, models
from .models import AggregateReport, BacktestCase, SkippedCase

# Optional subject dimensions surfaced in the failure report.
_OPTIONAL_SUBJECT_FIELDS = (
    "bedrooms_total", "full_bathrooms", "half_bathrooms",
    "year_built", "style", "garage_type", "basement_type",
)
_SIZE_MISMATCH_PCT = 25.0  # median |size diff| above this flags SIZE_MISMATCH


def _eligibility_skip(row: dict[str, Any]) -> Optional[str]:
    """Return a skip reason code if the row is not an eligible subject."""
    if not row.get("mls_number"):
        return models.MISSING_MLS
    if not row.get("sold_date"):
        return models.MISSING_SOLD_DATE
    price = row.get("sold_price")
    if price is None or Decimal(str(price)) <= 0:
        return models.INVALID_SOLD_PRICE
    area = row.get("living_area_sqft")
    if area is None or int(area) <= 0:
        return models.INVALID_LIVING_AREA
    if not row.get("property_type"):
        return models.MISSING_PROPERTY_TYPE
    if not row.get("area_code") and not row.get("neighbourhood"):
        return models.MISSING_GEOGRAPHY
    return None


def _missing_subject_fields(row: dict[str, Any]) -> list[str]:
    return [f for f in _OPTIONAL_SUBJECT_FIELDS if row.get(f) in (None, "")]


def _size_mismatch(result) -> bool:
    diffs = [
        abs(c.differences.get("size_diff_pct", 0.0))
        for c in result.comparables
        if c.differences
    ]
    if not diffs:
        return False
    diffs.sort()
    mid = diffs[len(diffs) // 2]
    return mid > _SIZE_MISMATCH_PCT


class BacktestRunner:
    def __init__(self, service: ComparableService):
        self._service = service

    def _value_one(self, row: dict[str, Any]) -> tuple[Optional[BacktestCase], Optional[SkippedCase]]:
        skip = _eligibility_skip(row)
        if skip is not None:
            return None, SkippedCase(mls_number=row.get("mls_number"), reason_code=skip)

        actual = Decimal(str(row["sold_price"]))
        subject_map = dict(row)
        subject_map["valuation_date"] = row["sold_date"]  # LOO: value at the sale date

        try:
            subject = SubjectProperty.from_mapping(subject_map)
            result = self._service.find_comparables(subject)
        except Exception:  # pragma: no cover - defensive; captured, never fatal
            return None, SkippedCase(row.get("mls_number"), models.ENGINE_ERROR)

        if result.status != "ok" or result.indicated_value is None:
            return None, SkippedCase(row.get("mls_number"), models.INSUFFICIENT_COMPARABLES)

        indicated = Decimal(result.indicated_value)
        abs_err = metrics.absolute_error(indicated, actual)
        pct_err = metrics.percentage_error(indicated, actual)
        mean_cov = (
            sum(c.data_coverage for c in result.comparables) / len(result.comparables)
            if result.comparables else 0.0
        )
        failure = metrics.classify_failure(
            status=result.status,
            tier_widened=result.tier_widened,
            median_similarity=result.median_similarity,
            mean_coverage=mean_cov,
            price_spread=result.price_spread,
            percentage_error=pct_err,
            size_mismatch=_size_mismatch(result),
        )
        case = BacktestCase(
            mls_number=str(row["mls_number"]),
            area_code=row.get("area_code"),
            neighbourhood=row.get("neighbourhood"),
            property_type=row.get("property_type"),
            sold_date=result.subject.valuation_date,
            actual_sold_price=actual,
            indicated_value=indicated,
            absolute_error=abs_err,
            percentage_error=pct_err,
            confidence=result.confidence,
            tier=result.tier,
            tier_widened=result.tier_widened,
            selected_comparable_count=result.selected_comparable_count,
            median_similarity=result.median_similarity,
            price_spread=result.price_spread,
            mean_coverage=mean_cov,
            warnings=list(result.warnings),
            missing_subject_fields=_missing_subject_fields(row),
            failure_category=failure,
        )
        return case, None

    def run(self, subject_rows: Iterable[dict[str, Any]], *, min_group: int = 3) -> AggregateReport:
        rows = list(subject_rows)
        cases: list[BacktestCase] = []
        skips: list[SkippedCase] = []
        for row in rows:
            case, skip = self._value_one(row)
            if case is not None:
                cases.append(case)
            if skip is not None:
                skips.append(skip)

        reason_counts = Counter(s.reason_code for s in skips)
        skipped_ineligible = sum(reason_counts[c] for c in models.ELIGIBILITY_SKIP_CODES)
        insufficient = reason_counts.get(models.INSUFFICIENT_COMPARABLES, 0)
        engine_errors = reason_counts.get(models.ENGINE_ERROR, 0)
        eligible = len(rows) - skipped_ineligible
        coverage = (len(cases) / eligible) if eligible else 0.0

        grouped = {
            "neighbourhood": metrics.grouped_metrics(cases, lambda c: c.neighbourhood, min_group=min_group),
            "area_code": metrics.grouped_metrics(cases, lambda c: c.area_code, min_group=min_group),
            "property_type": metrics.grouped_metrics(cases, lambda c: c.property_type, min_group=min_group),
            "confidence": metrics.grouped_metrics(cases, lambda c: c.confidence, min_group=min_group),
            "tier": metrics.grouped_metrics(cases, lambda c: c.tier, min_group=min_group),
            "comp_count_bucket": metrics.grouped_metrics(
                cases, lambda c: metrics.comp_count_bucket(c.selected_comparable_count), min_group=min_group),
            "similarity_bucket": metrics.grouped_metrics(
                cases, lambda c: metrics.similarity_bucket(c.median_similarity), min_group=min_group),
            "coverage_bucket": metrics.grouped_metrics(
                cases, lambda c: metrics.coverage_bucket(c.mean_coverage), min_group=min_group),
            "sold_month": metrics.grouped_metrics(cases, metrics.sold_month, min_group=min_group),
        }

        return AggregateReport(
            total_records=len(rows),
            eligible=eligible,
            valued=len(cases),
            insufficient=insufficient,
            skipped_ineligible=skipped_ineligible,
            engine_errors=engine_errors,
            coverage_rate=coverage,
            overall=metrics.overall_metrics(cases),
            thresholds=metrics.threshold_table(cases),
            distribution=metrics.distribution_table(cases),
            grouped=grouped,
            skip_reason_counts=dict(reason_counts),
            cases=cases,
            skips=skips,
        )
