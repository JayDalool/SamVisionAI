"""Human-readable reasons and warnings — concise, realtor-facing.

Pure formatting; no data access. Never emits private operational fields.
"""
from __future__ import annotations

from .models import ScoredComparable, SubjectProperty


def comparable_reasons(sc: ScoredComparable, subject: SubjectProperty) -> list[str]:
    d = sc.differences
    reasons: list[str] = []
    if d.get("same_neighbourhood"):
        reasons.append("same neighbourhood")
    elif sc.candidate.area_code and sc.candidate.area_code == subject.area_code:
        reasons.append("same area code (different/unknown neighbourhood)")
    size = d.get("size_diff_pct")
    if size is not None:
        reasons.append(f"living area {size:+.1f}% vs subject")
    reasons.append(f"sold {sc.age_days} days before valuation date")
    if "bedrooms_diff" in d:
        reasons.append(f"bedrooms {d['bedrooms_diff']:+d}")
    if "bathrooms_diff" in d:
        reasons.append(f"bathrooms {d['bathrooms_diff']:+.1f}")
    return reasons


def comparable_warnings(sc: ScoredComparable) -> list[str]:
    warnings: list[str] = []
    if sc.data_coverage < 0.5:
        warnings.append("low_data_coverage")
    if abs(sc.differences.get("size_diff_pct", 0)) >= 30:
        warnings.append("large_size_difference")
    if sc.age_days > 365:
        warnings.append("older_than_one_year")
    return warnings
