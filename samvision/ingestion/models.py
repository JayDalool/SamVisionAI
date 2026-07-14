"""Canonical data models and reason codes for the WRREB ingestion pipeline.

Everything here is deliberately dependency-light (stdlib + dataclasses) so the
parsers, reconciler and validator can be unit-tested without a database, pandas
or a live model. Nothing in this package writes to the production database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Reason codes — every rejection / warning / conflict carries a stable code so
# downstream review is filterable, never a free-text guess.
# ---------------------------------------------------------------------------
class ReasonCode:
    MISSING_CLIENT_FULL_RECORD = "MISSING_CLIENT_FULL_RECORD"
    MISSING_SINGLE_LINE_RECORD = "MISSING_SINGLE_LINE_RECORD"
    SOLD_DATE_CONFLICT = "SOLD_DATE_CONFLICT"
    SOLD_PRICE_CONFLICT = "SOLD_PRICE_CONFLICT"
    LIST_PRICE_CONFLICT = "LIST_PRICE_CONFLICT"
    ADDRESS_CONFLICT = "ADDRESS_CONFLICT"
    DUPLICATE_MLS = "DUPLICATE_MLS"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_DATE = "INVALID_DATE"
    FUTURE_SOLD_DATE = "FUTURE_SOLD_DATE"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    OCR_UNVERIFIED_FIELD = "OCR_UNVERIFIED_FIELD"
    NEEDS_OCR = "NEEDS_OCR"
    OUT_OF_RANGE = "OUT_OF_RANGE"


@dataclass
class Issue:
    """A single machine-readable finding attached to an MLS / page."""
    code: str
    field: str = ""
    detail: str = ""
    mls_number: str = ""

    def as_row(self) -> dict[str, Any]:
        return {"mls_number": self.mls_number, "code": self.code,
                "field": self.field, "detail": self.detail}


def normalize_mls(raw: Optional[str]) -> Optional[str]:
    """Canonical MLS key: strip ®, spaces, punctuation; keep the digits.

    WRREB MLS numbers are 9 digits prefixed with the listing year (202611485).
    Returns None when no such number is present so callers never key on junk.
    """
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw))
    m = re.search(r"20\d{7}", digits)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Per-source parsed records (pre-reconciliation)
# ---------------------------------------------------------------------------
@dataclass
class SingleLineRecord:
    """One row of the Agent Single Line incl SP report."""
    mls_number: Optional[str] = None
    status: Optional[str] = None
    area_code: Optional[str] = None
    address: Optional[str] = None
    list_price: Optional[int] = None
    sold_price: Optional[int] = None
    sold_date: Optional[date] = None          # the report's "Date" column
    dom: Optional[int] = None
    property_type_code: Optional[str] = None
    style_code: Optional[str] = None
    year_built: Optional[int] = None
    living_area_sqft: Optional[int] = None
    source_page: Optional[int] = None
    parser_version: str = ""


@dataclass
class ClientFullRecord:
    """One property page of the Residential Client Full report.

    Only non-private, model-relevant fields are captured. Agent/brokerage/
    contact/remarks are intentionally excluded (see privacy_exclusions in the
    data contract) and never populated here.
    """
    mls_number: Optional[str] = None
    linc_number: Optional[str] = None
    address: Optional[str] = None
    postal_code: Optional[str] = None
    area_code: Optional[str] = None
    neighbourhood: Optional[str] = None
    status: Optional[str] = None
    list_price: Optional[int] = None
    sold_price: Optional[int] = None
    sold_date: Optional[date] = None          # the report's explicit "Sell Date"
    dom: Optional[int] = None
    property_type_code: Optional[str] = None
    style_code: Optional[str] = None
    year_built: Optional[int] = None
    living_area_sqft: Optional[int] = None
    bedrooms_above_grade: Optional[int] = None
    bedrooms_total: Optional[int] = None
    full_bathrooms: Optional[int] = None
    half_bathrooms: Optional[int] = None
    basement_type: Optional[str] = None
    lot_front_ft: Optional[float] = None
    lot_depth_ft: Optional[float] = None
    gross_tax: Optional[float] = None
    tax_year: Optional[int] = None
    source_page: Optional[int] = None
    parser_version: str = ""


# ---------------------------------------------------------------------------
# Reconciled canonical record + provenance
# ---------------------------------------------------------------------------
@dataclass
class CanonicalSale:
    mls_number: Optional[str] = None
    linc_number: Optional[str] = None
    address: Optional[str] = None
    postal_code: Optional[str] = None
    area_code: Optional[str] = None
    neighbourhood: Optional[str] = None
    list_price: Optional[int] = None
    sold_price: Optional[int] = None
    sold_date: Optional[date] = None
    sale_year: Optional[int] = None           # DERIVED FROM sold_date only
    sale_month: Optional[int] = None          # DERIVED FROM sold_date only
    dom: Optional[int] = None
    property_type_code: Optional[str] = None
    style_code: Optional[str] = None
    year_built: Optional[int] = None
    living_area_sqft: Optional[int] = None
    bedrooms_above_grade: Optional[int] = None
    bedrooms_total: Optional[int] = None
    full_bathrooms: Optional[int] = None
    half_bathrooms: Optional[int] = None
    basement_type: Optional[str] = None
    lot_front_ft: Optional[float] = None
    lot_depth_ft: Optional[float] = None
    gross_tax: Optional[float] = None
    tax_year: Optional[int] = None
    mls_year_hint: Optional[int] = None       # from MLS number; NEVER a date source
    source_batch_id: str = ""
    parser_version: str = ""
    data_quality_status: str = "candidate"    # candidate | accepted | rejected
    issues: list[Issue] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["sold_date"] = self.sold_date.isoformat() if self.sold_date else None
        d["issue_codes"] = "|".join(sorted({i.code for i in self.issues}))
        d.pop("issues", None)
        return d


@dataclass
class PageDiagnostic:
    source_file: str = ""
    source_sha256: str = ""
    report_type: str = ""
    page_number: int = 0
    text_char_count: int = 0
    extraction_method: str = "text_layer"    # text_layer | ocr
    ocr_used: bool = False
    records_found: int = 0
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"                        # ok | needs_ocr | empty

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["warnings"] = "|".join(self.warnings)
        return d
