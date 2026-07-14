"""Turn a dry-run pipeline-output directory into a reviewable load plan.

Reads the manifest that ``samvision.ingestion.import_wrreb_batch`` writes
(``summary.json`` + the five CSVs), validates it, and builds the ``import_batches``
row, ``staging_sales`` rows and ``import_issues`` rows — without touching a
database. Structural problems (missing files) raise ``ManifestError``. Unsafe-batch
conditions (count mismatch, critical conflicts, failed reconciliation,
unrecognized contract version, duplicate MLS) are collected as ``refusals`` so a
dry run can report them; the loader must not write when any refusal is present.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from ..contract import SUPPORTED_CONTRACT_VERSIONS
from .fingerprints import (
    batch_fingerprint,
    normalized_property_id,
    record_fingerprint,
)

REQUIRED_FILES = (
    "summary.json",
    "accepted.csv",
    "rejected.csv",
    "conflicts.csv",
    "warnings.csv",
    "extraction_diagnostics.csv",
)

# accepted.csv column -> staging_sales column (codes become plain columns).
_CSV_TO_STAGING = {
    "mls_number": "mls_number",
    "linc_number": "linc_number",
    "address": "address",
    "postal_code": "postal_code",
    "area_code": "area_code",
    "neighbourhood": "neighbourhood",
    "list_price": "list_price",
    "sold_price": "sold_price",
    "sold_date": "sold_date",
    "dom": "dom",
    "property_type_code": "property_type",
    "style_code": "style",
    "year_built": "year_built",
    "living_area_sqft": "living_area_sqft",
    "bedrooms_above_grade": "bedrooms_above_grade",
    "bedrooms_total": "bedrooms_total",
    "full_bathrooms": "full_bathrooms",
    "half_bathrooms": "half_bathrooms",
    "basement_type": "basement_type",
    "lot_front_ft": "lot_front_ft",
    "lot_depth_ft": "lot_depth_ft",
    "gross_tax": "gross_tax",
    "tax_year": "tax_year",
    "parser_version": "parser_version",
}
_INT_FIELDS = {
    "dom", "year_built", "living_area_sqft", "bedrooms_above_grade",
    "bedrooms_total", "full_bathrooms", "half_bathrooms", "tax_year",
}
_DECIMAL_FIELDS = {"list_price", "sold_price", "lot_front_ft", "lot_depth_ft", "gross_tax"}


class ManifestError(Exception):
    """A pipeline-output directory is missing files or is structurally invalid."""


@dataclass
class LoadPlan:
    input_dir: str
    batch: dict[str, Any]
    staging_rows: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    counts: dict[str, Any]
    refusals: list[str] = field(default_factory=list)

    @property
    def safe_to_write(self) -> bool:
        return not self.refusals


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _to_int(value: Any) -> Optional[int]:
    s = _clean(value)
    if s is None:
        return None
    try:
        return int(float(s)) if ("." in s or "e" in s.lower()) else int(s)
    except (TypeError, ValueError):
        return None


def _to_decimal(value: Any) -> Optional[Decimal]:
    s = _clean(value)
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "")
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def _to_date(value: Any) -> Optional[date]:
    s = _clean(value)
    if s is None:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size == 0:
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _coerce_staging_row(csv_row: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for csv_col, staging_col in _CSV_TO_STAGING.items():
        raw = csv_row.get(csv_col)
        if staging_col == "sold_date":
            row[staging_col] = _to_date(raw)
        elif staging_col in _INT_FIELDS:
            row[staging_col] = _to_int(raw)
        elif staging_col in _DECIMAL_FIELDS:
            row[staging_col] = _to_decimal(raw)
        else:
            row[staging_col] = _clean(raw)
    row["normalized_property_id"] = normalized_property_id(
        row.get("linc_number"), row.get("address"), row.get("postal_code")
    )
    row["record_fingerprint"] = record_fingerprint(row)
    row["validation_status"] = "warning" if _clean(csv_row.get("issue_codes")) else "accepted"
    return row


def build_load_plan(input_dir: str | Path) -> LoadPlan:
    """Parse + validate a pipeline-output directory into a LoadPlan."""
    root = Path(input_dir)
    if not root.is_dir():
        raise ManifestError(f"input directory not found: {root}")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ManifestError("missing manifest files: " + ", ".join(missing))

    summary = json.loads((root / "summary.json").read_text())
    accepted = _read_csv(root / "accepted.csv")
    rejected = _read_csv(root / "rejected.csv")
    conflicts = _read_csv(root / "conflicts.csv")
    warnings = _read_csv(root / "warnings.csv")
    diagnostics = _read_csv(root / "extraction_diagnostics.csv")

    refusals: list[str] = []

    # contract_version is REQUIRED — no silent default. A missing, empty, or
    # unsupported value refuses the batch (design §2 / §6).
    contract_version = summary.get("contract_version")
    if contract_version is None or not str(contract_version).strip():
        refusals.append("missing or empty contract_version in summary.json")
        contract_version = ""
    elif contract_version not in SUPPORTED_CONTRACT_VERSIONS:
        refusals.append(f"unsupported contract_version: {contract_version!r}")

    if summary.get("critical_reconciliation_failed"):
        refusals.append("critical_reconciliation_failed is true in summary.json")

    # Summary counts must match the actual CSV contents.
    if summary.get("accepted") != len(accepted):
        refusals.append(
            f"summary.accepted={summary.get('accepted')} != accepted.csv rows={len(accepted)}"
        )
    if summary.get("rejected") != len(rejected):
        refusals.append(
            f"summary.rejected={summary.get('rejected')} != rejected.csv rows={len(rejected)}"
        )
    if summary.get("conflicts") != len(conflicts):
        refusals.append(
            f"summary.conflicts={summary.get('conflicts')} != conflicts.csv rows={len(conflicts)}"
        )

    if conflicts:
        refusals.append(f"batch has {len(conflicts)} unresolved critical conflict(s)")

    sources = summary.get("sources", {})
    try:
        sl = sources["single_line"]
        cf = sources["client_full"]
        single_line_sha = sl["sha256"]
        client_full_sha = cf["sha256"]
    except (KeyError, TypeError):
        raise ManifestError("summary.json is missing source file hashes")

    # Parser version: stable, derived from the accepted rows (fall back to rejected).
    parser_versions = sorted(
        {r.get("parser_version", "").strip() for r in (accepted + rejected)} - {""}
    )
    parser_version = "|".join(parser_versions)

    fingerprint = batch_fingerprint(
        single_line_sha, client_full_sha, parser_version, contract_version
    )

    # Duplicate MLS within the accepted set.
    seen: set[str] = set()
    for r in accepted:
        mls = (r.get("mls_number") or "").strip()
        if mls and mls in seen:
            refusals.append(f"duplicate MLS within batch: {mls}")
        seen.add(mls)

    recon = summary.get("reconciliation", {})
    pages = summary.get("pages", {})
    batch = {
        "batch_fingerprint": fingerprint,
        "contract_version": contract_version,
        "parser_version": parser_version,
        "single_line_filename": Path(str(sl.get("file", ""))).name,
        "single_line_sha256": single_line_sha,
        "client_full_filename": Path(str(cf.get("file", ""))).name,
        "client_full_sha256": client_full_sha,
        "status": "parsed",
        "single_line_count": recon.get("single_line_records", 0),
        "client_full_count": recon.get("client_full_records", 0),
        "matched_count": recon.get("matched_both", 0),
        "accepted_count": summary.get("accepted", 0),
        "rejected_count": summary.get("rejected", 0),
        "warning_count": summary.get("warnings", 0),
        "conflict_count": summary.get("conflicts", 0),
        "needs_ocr_count": pages.get("needs_ocr", 0),
    }
    date_range = summary.get("sold_date_range")
    if isinstance(date_range, list) and len(date_range) == 2:
        batch["search_start_date"] = _to_date(date_range[0])
        batch["search_end_date"] = _to_date(date_range[1])

    staging_rows = [_coerce_staging_row(r) for r in accepted]

    # A non-empty but unparseable sold_date is a corruption, not a silent NULL.
    for raw, coerced in zip(accepted, staging_rows):
        if _clean(raw.get("sold_date")) and coerced.get("sold_date") is None:
            refusals.append(
                f"accepted row MLS {coerced.get('mls_number')} has an invalid sold_date"
            )

    issues: list[dict[str, Any]] = []
    for r in conflicts:
        issues.append({
            "mls_number": _clean(r.get("mls_number")),
            "severity": "conflict",
            "reason_code": _clean(r.get("code")) or "CONFLICT",
            "field_name": _clean(r.get("field")),
            "message": _clean(r.get("detail")),
        })
    for r in warnings:
        issues.append({
            "mls_number": _clean(r.get("mls_number")),
            "severity": "warning",
            "reason_code": _clean(r.get("code")) or "WARNING",
            "field_name": _clean(r.get("field")),
            "message": _clean(r.get("detail")),
        })
    for r in rejected:
        codes = (_clean(r.get("issue_codes")) or "REJECTED").split("|")
        for code in codes:
            issues.append({
                "mls_number": _clean(r.get("mls_number")),
                "severity": "rejection",
                "reason_code": code or "REJECTED",
            })
    for d in diagnostics:
        if _clean(d.get("status")) == "needs_ocr":
            issues.append({
                "mls_number": None,
                "severity": "needs_ocr",
                "reason_code": "NEEDS_OCR",
                "source_page": _to_int(d.get("page_number")),
                "message": _clean(d.get("report_type")),
            })

    counts = {
        "batch_fingerprint": fingerprint,
        "single_line_count": batch["single_line_count"],
        "client_full_count": batch["client_full_count"],
        "matched_count": batch["matched_count"],
        "accepted_count": batch["accepted_count"],
        "rejected_count": batch["rejected_count"],
        "warning_count": batch["warning_count"],
        "conflict_count": batch["conflict_count"],
        "needs_ocr_count": batch["needs_ocr_count"],
        "staging_rows": len(staging_rows),
        "issue_rows": len(issues),
    }
    return LoadPlan(
        input_dir=str(root), batch=batch, staging_rows=staging_rows,
        issues=issues, counts=counts, refusals=refusals,
    )
