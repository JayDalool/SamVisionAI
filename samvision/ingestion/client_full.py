"""Parser for the WRREB "Residential Client Full" report.

Layout (verified against real exports): one property per page. Fields are
``Label:`` followed by a value (either on the same line or the next reading-order
line). We walk those pairs and map only non-private, model-relevant fields. The
explicit ``Sell Date`` is authoritative. Agent, brokerage, contact and remarks
text are intentionally never captured.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from .models import ClientFullRecord, PageDiagnostic, normalize_mls
from .pdf_text import iter_pages, PageLayout
from .provenance import sha256_file

PARSER_VERSION = "client_full/1.0.0"

_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /®#.\-]*?):\s*(.*)$")
_ADDR_RE = re.compile(r"^(\d+[^,]*?),?\s+Winnipeg\s+(R\d[A-Z]\s?\d[A-Z]\d)", re.I)
_LINC_RE = re.compile(r"\b\d{3}[A-Z]\d{9}\b")

# labels we deliberately do NOT carry into canonical output (privacy / noise)
_PRIVATE_LABELS = {"remarks", "dir/gps", "gds incl", "gds excl", "rnt eqp",
                   "legal", "add lgl", "site influ", "features"}


def _to_int(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    d = re.sub(r"[^\d]", "", raw)
    return int(d) if d else None


def _to_float(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    m = re.search(r"[\d,]+(?:\.\d+)?", raw)
    return float(m.group(0).replace(",", "")) if m else None


def _sold_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    m = re.search(r"(\d{2}/\d{2}/\d{4})", raw)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%m/%d/%Y").date()
    except ValueError:
        return None


def _walk_labels(lines: list[str]) -> dict[str, str]:
    """Build {label: value}. A bare ``Label:`` takes the next non-label line."""
    kv: dict[str, str] = {}
    i = 0
    while i < len(lines):
        m = _LABEL_RE.match(lines[i])
        if m:
            label, val = m.group(1).strip(), m.group(2).strip()
            if not val and i + 1 < len(lines) and not _LABEL_RE.match(lines[i + 1]):
                val = lines[i + 1].strip()
                i += 1
            kv[label] = val
        i += 1
    return kv


def _parse_page(page: PageLayout) -> Optional[ClientFullRecord]:
    lines = page.flat_text().split("\n")
    kv = {k: v for k, v in _walk_labels(lines).items() if k.lower() not in _PRIVATE_LABELS}

    mls = normalize_mls(kv.get("MLS® #"))
    if not mls:
        m = re.search(r"20\d{7}", page.flat_text())
        mls = m.group(0) if m else None
    if not mls:
        return None

    address = postal = None
    for ln in lines:
        m = _ADDR_RE.match(ln)
        if m:
            address = m.group(1).strip()
            postal = m.group(2).upper().replace("  ", " ")
            break

    linc = kv.get("Linc #")
    linc = linc if (linc and _LINC_RE.fullmatch(linc)) else (
        _LINC_RE.search(page.flat_text()).group(0) if _LINC_RE.search(page.flat_text()) else None)

    yr = kv.get("Yr Built/Age", "")
    year_built = int(re.match(r"(\d{4})", yr).group(1)) if re.match(r"(\d{4})", yr) else None

    liv = kv.get("Liv Area", "")
    sqft_m = re.search(r"([\d,]+)\s*SF", liv)
    living_area = int(sqft_m.group(1).replace(",", "")) if sqft_m else None

    baths = kv.get("Baths", "")
    fb = re.search(r"F(\d+)", baths)
    hb = re.search(r"H(\d+)", baths)

    def _feet(label: str) -> Optional[float]:
        m = re.search(r"/\s*([\d.]+)\s*F", kv.get(label, ""))
        return float(m.group(1)) if m else None

    return ClientFullRecord(
        mls_number=mls,
        linc_number=linc,
        address=address,
        postal_code=postal,
        area_code=kv.get("Area"),
        neighbourhood=kv.get("Nghbrhd") or None,
        status=kv.get("Status"),
        list_price=_to_int(kv.get("List Price")),
        sold_price=_to_int(kv.get("Sell Price")),
        sold_date=_sold_date(kv.get("Sell Date")),
        dom=_to_int(kv.get("DOM")),
        property_type_code=kv.get("Type") or None,
        style_code=kv.get("Style") or None,
        year_built=year_built,
        living_area_sqft=living_area,
        bedrooms_above_grade=_to_int(kv.get("BDA")),
        bedrooms_total=_to_int(kv.get("TBD")),
        full_bathrooms=int(fb.group(1)) if fb else None,
        half_bathrooms=int(hb.group(1)) if hb else None,
        basement_type=kv.get("Basement") or None,
        lot_front_ft=_feet("Lot Front"),
        lot_depth_ft=_feet("Lot Dpth"),
        gross_tax=_to_float(kv.get("Gross Tax")),
        tax_year=_to_int(kv.get("Tax Yr")),
        source_page=page.page_number,
        parser_version=PARSER_VERSION,
    )


def parse(path: str) -> tuple[list[ClientFullRecord], list[PageDiagnostic]]:
    records: list[ClientFullRecord] = []
    diags: list[PageDiagnostic] = []
    file_sha = sha256_file(path)

    for page in iter_pages(path):
        diag = PageDiagnostic(source_file=path, source_sha256=file_sha,
                              report_type="client_full", page_number=page.page_number,
                              text_char_count=page.char_count)
        if page.is_thin:
            diag.status = "needs_ocr"
            diag.warnings.append("thin/empty text layer — flagged, not parsed")
            diags.append(diag)
            continue
        rec = _parse_page(page)
        if rec:
            records.append(rec)
            diag.records_found = 1
        else:
            diag.warnings.append("no MLS number found on page")
        diags.append(diag)

    return records, diags
