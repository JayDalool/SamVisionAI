"""Deterministic eligibility + similarity scoring.

No learning, no randomness, no current-date arithmetic. A missing optional
dimension scores 0 (never an implicit match), so a sparse record cannot outrank a
complete one purely by having missing dimensions ignored.
"""
from __future__ import annotations

from typing import Any, Optional

from . import constants
from .models import CandidateSale, ScoredComparable, SubjectProperty


def _geo_match(cand: CandidateSale, subject: SubjectProperty) -> bool:
    if subject.neighbourhood and cand.neighbourhood == subject.neighbourhood:
        return True
    if subject.area_code and cand.area_code == subject.area_code:
        return True
    return False


def eligibility(cand: CandidateSale, subject: SubjectProperty,
                max_lookback_days: int = constants.MAX_LOOKBACK_DAYS) -> tuple[bool, str]:
    """Return (eligible, reason_code). Chronology uses sold_date only."""
    if cand.data_quality_status != "approved":
        return False, "not_approved"
    if not cand.mls_number:
        return False, "missing_mls"
    if subject.mls_number and cand.mls_number == subject.mls_number:
        return False, "subject_self"
    if cand.property_type != subject.property_type:
        return False, "property_type_mismatch"
    if cand.sold_price is None or cand.sold_price <= 0:
        return False, "non_positive_price"
    if cand.sold_date is None:
        return False, "missing_sold_date"
    if cand.sold_date > subject.valuation_date:
        return False, "future_sale"
    if (subject.valuation_date - cand.sold_date).days > max_lookback_days:
        return False, "outside_lookback"
    if not cand.living_area_sqft or cand.living_area_sqft <= 0:
        return False, "missing_size"
    if not _geo_match(cand, subject):
        return False, "no_geographic_match"
    return True, "eligible"


# --- component subscores (each 0..1) --------------------------------------
def geography_subscore(cand: CandidateSale, subject: SubjectProperty) -> float:
    if subject.neighbourhood and cand.neighbourhood and cand.neighbourhood == subject.neighbourhood:
        return 1.0
    if subject.area_code and cand.area_code == subject.area_code:
        return constants.GEO_SAME_AREA_SUBSCORE
    return 0.0


def living_area_subscore(cand: CandidateSale, subject: SubjectProperty) -> float:
    pct = abs(cand.living_area_sqft - subject.living_area_sqft) / subject.living_area_sqft
    return max(0.0, 1.0 - pct / constants.LIVING_AREA_ZERO_AT_PCT)


def recency_subscore(age_days: int) -> float:
    return max(0.0, 1.0 - age_days / constants.MAX_LOOKBACK_DAYS)


def bedroom_subscore(cand: CandidateSale, subject: SubjectProperty) -> float:
    if cand.bedrooms_total is None or subject.bedrooms_total is None:
        return 0.0
    diff = abs(cand.bedrooms_total - subject.bedrooms_total)
    return max(0.0, 1.0 - diff / constants.BEDROOM_ZERO_AT_DIFF)


def _baths(full: Optional[int], half: Optional[int]) -> Optional[float]:
    if full is None and half is None:
        return None
    return (full or 0) + 0.5 * (half or 0)


def bathroom_subscore(cand: CandidateSale, subject: SubjectProperty) -> float:
    sb = _baths(subject.full_bathrooms, subject.half_bathrooms)
    cb = _baths(cand.full_bathrooms, cand.half_bathrooms)
    if sb is None or cb is None:
        return 0.0
    return max(0.0, 1.0 - abs(cb - sb) / constants.BATHROOM_ZERO_AT_DIFF)


def year_built_subscore(cand: CandidateSale, subject: SubjectProperty) -> float:
    """Age difference computed at historical dates, never the current year."""
    if cand.year_built is None or subject.year_built is None or cand.sold_date is None:
        return 0.0
    subject_age = subject.valuation_date.year - subject.year_built
    comp_age = cand.sold_date.year - cand.year_built
    return max(0.0, 1.0 - abs(subject_age - comp_age) / constants.YEAR_BUILT_ZERO_AT_DIFF)


def _exact_match_subscore(a: Optional[str], b: Optional[str]) -> float:
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0


def data_coverage(cand: CandidateSale) -> float:
    """Fraction of optional scoring dimensions present on the candidate."""
    present = 0
    if cand.bedrooms_total is not None:
        present += 1
    if cand.full_bathrooms is not None or cand.half_bathrooms is not None:
        present += 1
    if cand.year_built is not None:
        present += 1
    if cand.style is not None:
        present += 1
    if cand.garage_type is not None:
        present += 1
    if cand.basement_type is not None:
        present += 1
    return present / len(constants.COVERAGE_DIMENSIONS)


def _differences(cand: CandidateSale, subject: SubjectProperty, age_days: int) -> dict[str, Any]:
    size_diff_pct = round(
        (cand.living_area_sqft - subject.living_area_sqft) / subject.living_area_sqft * 100, 1
    )
    diffs: dict[str, Any] = {
        "size_diff_pct": size_diff_pct,
        "same_neighbourhood": bool(
            subject.neighbourhood and cand.neighbourhood == subject.neighbourhood
        ),
        "age_days": age_days,
    }
    if cand.bedrooms_total is not None and subject.bedrooms_total is not None:
        diffs["bedrooms_diff"] = cand.bedrooms_total - subject.bedrooms_total
    sb, cb = _baths(subject.full_bathrooms, subject.half_bathrooms), _baths(
        cand.full_bathrooms, cand.half_bathrooms)
    if sb is not None and cb is not None:
        diffs["bathrooms_diff"] = round(cb - sb, 1)
    if cand.year_built is not None and subject.year_built is not None:
        diffs["year_built_diff"] = cand.year_built - subject.year_built
    return diffs


def score(cand: CandidateSale, subject: SubjectProperty, tier: str) -> ScoredComparable:
    """Score an *eligible* candidate. Total = Σ weight × subscore (0..100)."""
    age_days = (subject.valuation_date - cand.sold_date).days
    subs = {
        "geography": geography_subscore(cand, subject),
        "living_area": living_area_subscore(cand, subject),
        "recency": recency_subscore(age_days),
        "bedrooms": bedroom_subscore(cand, subject),
        "bathrooms": bathroom_subscore(cand, subject),
        "year_built": year_built_subscore(cand, subject),
        "style": _exact_match_subscore(cand.style, subject.style),
        "garage": _exact_match_subscore(cand.garage_type, subject.garage_type),
        "basement": _exact_match_subscore(cand.basement_type, subject.basement_type),
    }
    weighted = {k: constants.WEIGHTS[k] * v for k, v in subs.items()}
    total = sum(weighted.values())
    return ScoredComparable(
        candidate=cand,
        similarity=round(total, 2),
        components={k: round(v, 4) for k, v in subs.items()},
        weighted_components={k: round(v, 2) for k, v in weighted.items()},
        data_coverage=round(data_coverage(cand), 4),
        tier=tier,
        age_days=age_days,
        differences=_differences(cand, subject, age_days),
    )
