"""Typed models for the comparable-sales engine.

Money is ``Decimal``. Nothing here reaches a database. Optional fields left as
``None`` are never treated as a match by scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Mapping, Optional

from . import constants


class SubjectValidationError(ValueError):
    """Raised when a subject property fails validation."""


def _as_date(value: Any) -> Optional[date]:
    if value is None or isinstance(value, date):
        return value if isinstance(value, date) else None
    return date.fromisoformat(str(value))


def _as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _as_decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    return Decimal(str(value))


@dataclass
class SubjectProperty:
    """The property being valued. valuation_date is always caller-supplied."""
    valuation_date: date
    property_type: str
    living_area_sqft: int
    area_code: Optional[str] = None
    neighbourhood: Optional[str] = None
    mls_number: Optional[str] = None
    linc_number: Optional[str] = None
    address: Optional[str] = None
    postal_code: Optional[str] = None
    style: Optional[str] = None
    year_built: Optional[int] = None
    bedrooms_total: Optional[int] = None
    full_bathrooms: Optional[int] = None
    half_bathrooms: Optional[int] = None
    basement_type: Optional[str] = None
    basement_development: Optional[str] = None
    lot_area_sqft: Optional[int] = None
    garage_type: Optional[str] = None
    parking_spaces: Optional[int] = None

    def validate(self) -> "SubjectProperty":
        if not isinstance(self.valuation_date, date):
            raise SubjectValidationError("valuation_date must be a real date")
        if not self.property_type:
            raise SubjectValidationError("property_type is required")
        if self.living_area_sqft is None or self.living_area_sqft <= 0:
            raise SubjectValidationError("living_area_sqft must be positive")
        if not (self.area_code or self.neighbourhood):
            raise SubjectValidationError("at least one of area_code / neighbourhood is required")
        if self.year_built is not None:
            if not (constants.MIN_PLAUSIBLE_YEAR_BUILT <= self.year_built <= self.valuation_date.year):
                raise SubjectValidationError(
                    "year_built must be plausible and not later than the valuation year"
                )
        for name in ("bedrooms_total", "full_bathrooms", "half_bathrooms", "parking_spaces"):
            v = getattr(self, name)
            if v is not None and v < 0:
                raise SubjectValidationError(f"{name} must be non-negative")
        return self

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "SubjectProperty":
        """Build from a JSON-like mapping. valuation_date must be present (ISO)."""
        if "valuation_date" not in m or not m["valuation_date"]:
            raise SubjectValidationError("valuation_date is required (never defaults to today)")
        return cls(
            valuation_date=_as_date(m["valuation_date"]),
            property_type=m.get("property_type"),
            living_area_sqft=_as_int(m.get("living_area_sqft")),
            area_code=m.get("area_code") or None,
            neighbourhood=m.get("neighbourhood") or None,
            mls_number=m.get("mls_number") or None,
            linc_number=m.get("linc_number") or None,
            address=m.get("address") or None,
            postal_code=m.get("postal_code") or None,
            style=m.get("style") or None,
            year_built=_as_int(m.get("year_built")),
            bedrooms_total=_as_int(m.get("bedrooms_total")),
            full_bathrooms=_as_int(m.get("full_bathrooms")),
            half_bathrooms=_as_int(m.get("half_bathrooms")),
            basement_type=m.get("basement_type") or None,
            basement_development=m.get("basement_development") or None,
            lot_area_sqft=_as_int(m.get("lot_area_sqft")),
            garage_type=m.get("garage_type") or None,
            parking_spaces=_as_int(m.get("parking_spaces")),
        )


@dataclass
class CandidateSale:
    """A canonical sale considered as a comparable (scoring-relevant fields)."""
    mls_number: Optional[str]
    sold_date: Optional[date]
    sold_price: Optional[Decimal]
    living_area_sqft: Optional[int]
    area_code: Optional[str] = None
    neighbourhood: Optional[str] = None
    property_type: Optional[str] = None
    style: Optional[str] = None
    year_built: Optional[int] = None
    bedrooms_total: Optional[int] = None
    full_bathrooms: Optional[int] = None
    half_bathrooms: Optional[int] = None
    basement_type: Optional[str] = None
    garage_type: Optional[str] = None
    data_quality_status: Optional[str] = None
    address: Optional[str] = None

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "CandidateSale":
        return cls(
            mls_number=m.get("mls_number"),
            sold_date=_as_date(m.get("sold_date")),
            sold_price=_as_decimal(m.get("sold_price")),
            living_area_sqft=_as_int(m.get("living_area_sqft")),
            area_code=m.get("area_code"),
            neighbourhood=m.get("neighbourhood"),
            property_type=m.get("property_type"),
            style=m.get("style"),
            year_built=_as_int(m.get("year_built")),
            bedrooms_total=_as_int(m.get("bedrooms_total")),
            full_bathrooms=_as_int(m.get("full_bathrooms")),
            half_bathrooms=_as_int(m.get("half_bathrooms")),
            basement_type=m.get("basement_type"),
            garage_type=m.get("garage_type"),
            data_quality_status=m.get("data_quality_status"),
            address=m.get("address"),
        )


@dataclass
class ScoredComparable:
    candidate: CandidateSale
    similarity: float                       # 0..100
    components: dict[str, float]            # subscores 0..1
    weighted_components: dict[str, float]   # points contributed
    data_coverage: float                    # 0..1
    tier: str
    age_days: int
    ppsf: Optional[Decimal] = None
    size_normalized_price: Optional[Decimal] = None
    differences: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, *, include_address: bool = False) -> dict[str, Any]:
        c = self.candidate
        out: dict[str, Any] = {
            "mls_number": c.mls_number,
            "sold_date": c.sold_date.isoformat() if c.sold_date else None,
            "sold_price": str(c.sold_price) if c.sold_price is not None else None,
            "living_area_sqft": c.living_area_sqft,
            "area_code": c.area_code,
            "neighbourhood": c.neighbourhood,
            "property_type": c.property_type,
            "similarity": self.similarity,
            "components": self.components,
            "weighted_components": self.weighted_components,
            "data_coverage": self.data_coverage,
            "tier": self.tier,
            "age_days": self.age_days,
            "ppsf": str(self.ppsf) if self.ppsf is not None else None,
            "size_normalized_price": (
                str(self.size_normalized_price) if self.size_normalized_price is not None else None
            ),
            "differences": self.differences,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }
        if include_address:
            out["address"] = c.address
        return out


@dataclass
class ComparableResult:
    subject: SubjectProperty
    status: str                             # "ok" | "insufficient_data"
    tier: Optional[str]
    tier_widened: bool
    candidate_count: int
    comparables: list[ScoredComparable]
    indicated_value: Optional[Decimal] = None
    indicated_range_low: Optional[Decimal] = None
    indicated_range_high: Optional[Decimal] = None
    direct_weighted_median_price: Optional[Decimal] = None
    weighted_median_ppsf: Optional[Decimal] = None
    median_similarity: Optional[float] = None
    price_spread: Optional[float] = None
    confidence: str = "insufficient"
    confidence_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sold_date_range: Optional[tuple[str, str]] = None

    @property
    def selected_comparable_count(self) -> int:
        return len(self.comparables)

    def to_dict(self, *, include_address: bool = False) -> dict[str, Any]:
        def money(x: Optional[Decimal]) -> Optional[str]:
            return str(x) if x is not None else None

        return {
            "status": self.status,
            "valuation_date": self.subject.valuation_date.isoformat(),
            "subject": {
                "area_code": self.subject.area_code,
                "neighbourhood": self.subject.neighbourhood,
                "property_type": self.subject.property_type,
                "living_area_sqft": self.subject.living_area_sqft,
            },
            "tier": self.tier,
            "tier_widened": self.tier_widened,
            "candidate_count": self.candidate_count,
            "selected_comparable_count": self.selected_comparable_count,
            "indicated_value": money(self.indicated_value),
            "indicated_range_low": money(self.indicated_range_low),
            "indicated_range_high": money(self.indicated_range_high),
            "direct_weighted_median_price": money(self.direct_weighted_median_price),
            "weighted_median_ppsf": money(self.weighted_median_ppsf),
            "median_similarity": self.median_similarity,
            "price_spread": self.price_spread,
            "confidence": self.confidence,
            "confidence_reasons": self.confidence_reasons,
            "warnings": self.warnings,
            "sold_date_range": list(self.sold_date_range) if self.sold_date_range else None,
            "comparables": [c.to_dict(include_address=include_address) for c in self.comparables],
        }
