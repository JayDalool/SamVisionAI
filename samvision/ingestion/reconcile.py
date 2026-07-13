"""Reconcile the two WRREB reports by normalized MLS number.

Authority rules (from the data contract):
* Client Full explicit ``Sell Date`` is authoritative; Agent Single Line ``Date``
  must normally match it — a mismatch raises SOLD_DATE_CONFLICT (never silently
  overwritten).
* Client Full is authoritative for LINC, neighbourhood and detailed
  characteristics; Single Line is the compact index and a cross-check source.
* sale_year / sale_month are derived ONLY from the real sold_date. The MLS-number
  year is stored solely as ``mls_year_hint`` and never substitutes for a date.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from .models import (CanonicalSale, ClientFullRecord, Issue, ReasonCode,
                     SingleLineRecord)


def _dupes(mls_list: list[str]) -> set[str]:
    return {m for m, c in Counter(m for m in mls_list if m).items() if c > 1}


def _norm_addr(a: Optional[str]) -> str:
    return " ".join((a or "").lower().replace(",", " ").split())


def _pick(primary, secondary):
    return primary if primary not in (None, "") else secondary


def reconcile(single: list[SingleLineRecord],
              full: list[ClientFullRecord],
              batch_id: str = "") -> tuple[list[CanonicalSale], dict]:
    sl_by = {r.mls_number: r for r in single if r.mls_number}
    cf_by = {r.mls_number: r for r in full if r.mls_number}
    sl_dupes = _dupes([r.mls_number for r in single])
    cf_dupes = _dupes([r.mls_number for r in full])

    canon: list[CanonicalSale] = []
    for mls in sorted(set(sl_by) | set(cf_by)):
        sl, cf = sl_by.get(mls), cf_by.get(mls)
        issues: list[Issue] = []

        if not cf:
            issues.append(Issue(ReasonCode.MISSING_CLIENT_FULL_RECORD, mls_number=mls))
        if not sl:
            issues.append(Issue(ReasonCode.MISSING_SINGLE_LINE_RECORD, mls_number=mls))
        if mls in sl_dupes or mls in cf_dupes:
            issues.append(Issue(ReasonCode.DUPLICATE_MLS, mls_number=mls,
                                detail="MLS appears more than once in a source report"))

        # --- sold_date: Client Full authoritative, cross-checked vs Single Line
        sold_date = (cf.sold_date if cf and cf.sold_date else (sl.sold_date if sl else None))
        if sl and cf and sl.sold_date and cf.sold_date and sl.sold_date != cf.sold_date:
            issues.append(Issue(ReasonCode.SOLD_DATE_CONFLICT, "sold_date", mls_number=mls,
                                detail=f"single={sl.sold_date} full={cf.sold_date}"))

        # --- prices: cross-check when both present
        def _cross(field, a, b, code):
            if a is not None and b is not None and a != b:
                issues.append(Issue(code, field, mls_number=mls, detail=f"single={a} full={b}"))
        _cross("sold_price", sl and sl.sold_price, cf and cf.sold_price, ReasonCode.SOLD_PRICE_CONFLICT)
        _cross("list_price", sl and sl.list_price, cf and cf.list_price, ReasonCode.LIST_PRICE_CONFLICT)

        if sl and cf and sl.address and cf.address and _norm_addr(sl.address) not in _norm_addr(cf.address) \
                and _norm_addr(cf.address) not in _norm_addr(sl.address):
            issues.append(Issue(ReasonCode.ADDRESS_CONFLICT, "address", mls_number=mls,
                                detail=f"single={sl.address!r} full={cf.address!r}"))

        cs = CanonicalSale(
            mls_number=mls,
            linc_number=cf.linc_number if cf else None,
            address=_pick(cf and cf.address, sl and sl.address),
            postal_code=cf.postal_code if cf else None,
            area_code=_pick(cf and cf.area_code, sl and sl.area_code),
            neighbourhood=cf.neighbourhood if cf else None,
            list_price=_pick(cf and cf.list_price, sl and sl.list_price),
            sold_price=_pick(cf and cf.sold_price, sl and sl.sold_price),
            sold_date=sold_date,
            sale_year=sold_date.year if sold_date else None,
            sale_month=sold_date.month if sold_date else None,
            dom=_pick(cf and cf.dom, sl and sl.dom),
            property_type_code=_pick(cf and cf.property_type_code, sl and sl.property_type_code),
            style_code=_pick(cf and cf.style_code, sl and sl.style_code),
            year_built=_pick(cf and cf.year_built, sl and sl.year_built),
            living_area_sqft=_pick(cf and cf.living_area_sqft, sl and sl.living_area_sqft),
            bedrooms_above_grade=cf.bedrooms_above_grade if cf else None,
            bedrooms_total=cf.bedrooms_total if cf else None,
            full_bathrooms=cf.full_bathrooms if cf else None,
            half_bathrooms=cf.half_bathrooms if cf else None,
            basement_type=cf.basement_type if cf else None,
            lot_front_ft=cf.lot_front_ft if cf else None,
            lot_depth_ft=cf.lot_depth_ft if cf else None,
            gross_tax=cf.gross_tax if cf else None,
            tax_year=cf.tax_year if cf else None,
            mls_year_hint=int(mls[:4]) if mls and mls[:2] == "20" else None,
            source_batch_id=batch_id,
            parser_version=f"{getattr(sl, 'parser_version', '')}+{getattr(cf, 'parser_version', '')}".strip("+"),
            issues=issues,
        )
        canon.append(cs)

    summary = {
        "single_line_records": len(single),
        "client_full_records": len(full),
        "unique_mls_single": len(sl_by),
        "unique_mls_full": len(cf_by),
        "matched_both": len(set(sl_by) & set(cf_by)),
        "single_only": len(set(sl_by) - set(cf_by)),
        "full_only": len(set(cf_by) - set(sl_by)),
        "duplicate_mls": sorted(sl_dupes | cf_dupes),
    }
    return canon, summary
