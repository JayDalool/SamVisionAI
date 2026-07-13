"""Field validation for canonical sales.

A record is accepted only when every required field is present and valid and no
critical conflict is unresolved. Missing optional fields become warnings, never
silent zero/"none"/today substitutions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .models import CanonicalSale, Issue, ReasonCode

# critical conflict codes that block acceptance
_BLOCKING = {ReasonCode.SOLD_DATE_CONFLICT, ReasonCode.SOLD_PRICE_CONFLICT,
             ReasonCode.DUPLICATE_MLS}


@dataclass
class ValidationLimits:
    min_price: int = 1
    max_price: int = 50_000_000
    min_living_area: int = 100
    max_living_area: int = 30_000
    min_year_built: int = 1850
    max_year_built: int = 2100
    max_bedrooms: int = 20
    max_bathrooms: int = 20


REQUIRED_FIELDS = ("mls_number", "sold_price", "sold_date", "property_type_code")


def validate(sales: list[CanonicalSale], limits: ValidationLimits | None = None,
             today: date | None = None) -> list[CanonicalSale]:
    limits = limits or ValidationLimits()
    today = today or date.today()

    for cs in sales:
        iss = cs.issues

        for f in REQUIRED_FIELDS:
            if getattr(cs, f) in (None, ""):
                iss.append(Issue(ReasonCode.MISSING_REQUIRED_FIELD, f, mls_number=cs.mls_number or ""))
        if not (cs.address or cs.linc_number):
            iss.append(Issue(ReasonCode.MISSING_REQUIRED_FIELD, "address_or_linc",
                             mls_number=cs.mls_number or ""))

        if cs.sold_date and cs.sold_date > today:
            iss.append(Issue(ReasonCode.FUTURE_SOLD_DATE, "sold_date", mls_number=cs.mls_number or "",
                             detail=str(cs.sold_date)))

        for f in ("sold_price", "list_price"):
            v = getattr(cs, f)
            if v is not None and not (limits.min_price <= v <= limits.max_price):
                iss.append(Issue(ReasonCode.INVALID_PRICE, f, mls_number=cs.mls_number or "", detail=str(v)))

        if cs.living_area_sqft is not None and not (
                limits.min_living_area <= cs.living_area_sqft <= limits.max_living_area):
            iss.append(Issue(ReasonCode.OUT_OF_RANGE, "living_area_sqft",
                             mls_number=cs.mls_number or "", detail=str(cs.living_area_sqft)))
        if cs.year_built is not None and not (
                limits.min_year_built <= cs.year_built <= limits.max_year_built):
            iss.append(Issue(ReasonCode.OUT_OF_RANGE, "year_built",
                             mls_number=cs.mls_number or "", detail=str(cs.year_built)))
        if cs.bedrooms_total is not None and cs.bedrooms_total > limits.max_bedrooms:
            iss.append(Issue(ReasonCode.OUT_OF_RANGE, "bedrooms_total", mls_number=cs.mls_number or ""))

        blocking = {i.code for i in iss} & (
            _BLOCKING | {ReasonCode.MISSING_REQUIRED_FIELD, ReasonCode.INVALID_PRICE,
                         ReasonCode.INVALID_DATE, ReasonCode.FUTURE_SOLD_DATE})
        cs.data_quality_status = "rejected" if blocking else "accepted"

    return sales
