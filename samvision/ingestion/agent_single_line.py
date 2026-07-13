"""Parser for the WRREB "Residential Agent Single Line incl SP" report.

Layout (verified against real exports): a fixed-column table repeated across
pages. Column x-anchors are read from each page's header row, then every data
row's positioned fragments are bucketed into columns by x. The "Date" column is
the real Sell Date. We never infer a missing date from today() or the MLS year.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from .models import SingleLineRecord, PageDiagnostic, normalize_mls
from .pdf_text import iter_pages, PageLayout, TextFragment
from .provenance import sha256_file

PARSER_VERSION = "single_line/1.0.0"

# header label -> canonical column key
_HEADER_MAP = {
    "MLS® #": "mls", "MLS # ": "mls", "MLS #": "mls",
    "S": "status",
    "Ar Address": "address", "Address": "address",
    "List Price": "list_price",
    "Sold Price": "sold_price",
    "Date": "sold_date",
    "DOM": "dom",
    "Ty": "property_type_code",
    "Style": "style_code",
    "YrBt": "year_built",
    "SqFt": "living_area_sqft",
}


def _to_int(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", raw)
    return int(digits) if digits else None


def _to_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", raw)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), "%m/%d/%Y").date()
    except ValueError:
        return None


def _find_header(page: PageLayout) -> Optional[list[tuple[float, str]]]:
    """Return [(x_anchor, column_key), ...] if this page has the table header."""
    for row in page.rows():
        texts = {f.text for f in row}
        if "MLS® #" in texts and "List Price" in texts and "Date" in texts:
            anchors = []
            for f in row:
                key = _HEADER_MAP.get(f.text)
                if key:
                    anchors.append((f.x0, key))
            return sorted(anchors)
    return None


def _bucket(fragment_x: float, anchors: list[tuple[float, str]]) -> Optional[str]:
    """Nearest column anchor; ignore fragments left of the first (row numbers)."""
    first_x = anchors[0][0]
    if fragment_x < first_x - 6:
        return None
    return min(anchors, key=lambda a: abs(a[0] - fragment_x))[1]


def _split_area_address(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not raw:
        return None, None
    parts = raw.split(None, 1)
    if len(parts) == 2 and re.fullmatch(r"\d+[A-Z]?", parts[0]):
        return parts[0], parts[1].strip()
    return None, raw.strip()


def parse_page(page: PageLayout) -> list[SingleLineRecord]:
    """Parse one table page into records (empty list if it has no header)."""
    anchors = _find_header(page)
    if not anchors:
        return []
    header_y = next(f.y for f in page.fragments if f.text == "MLS® #")
    out: list[SingleLineRecord] = []
    for row in page.rows():
        if row[0].y >= header_y - 1:
            continue  # header and anything above it
        cells: dict[str, str] = {}
        for f in row:
            key = _bucket(f.x0, anchors)
            if key:
                cells[key] = (cells.get(key, "") + " " + f.text).strip()
        mls = normalize_mls(cells.get("mls"))
        if not mls:
            continue
        area, addr = _split_area_address(cells.get("address"))
        out.append(SingleLineRecord(
            mls_number=mls,
            status=cells.get("status"),
            area_code=area,
            address=addr,
            list_price=_to_int(cells.get("list_price")),
            sold_price=_to_int(cells.get("sold_price")),
            sold_date=_to_date(cells.get("sold_date")),
            dom=_to_int(cells.get("dom")),
            property_type_code=cells.get("property_type_code"),
            style_code=cells.get("style_code"),
            year_built=_to_int(cells.get("year_built")),
            living_area_sqft=_to_int(cells.get("living_area_sqft")),
            source_page=page.page_number,
            parser_version=PARSER_VERSION,
        ))
    return out


def parse(path: str) -> tuple[list[SingleLineRecord], list[PageDiagnostic]]:
    records: list[SingleLineRecord] = []
    diags: list[PageDiagnostic] = []
    file_sha = sha256_file(path)

    for page in iter_pages(path):
        diag = PageDiagnostic(source_file=path, source_sha256=file_sha,
                              report_type="agent_single_line", page_number=page.page_number,
                              text_char_count=page.char_count)
        if page.is_thin:
            diag.status = "needs_ocr"
            diag.warnings.append("thin/empty text layer — flagged, not parsed")
            diags.append(diag)
            continue
        if not _find_header(page):
            diag.warnings.append("no table header found on page")
            diags.append(diag)
            continue
        page_records = parse_page(page)
        records.extend(page_records)
        diag.records_found = len(page_records)
        diags.append(diag)

    return records, diags
