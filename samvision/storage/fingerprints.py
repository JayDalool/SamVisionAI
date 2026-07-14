"""Deterministic fingerprints and identity helpers for canonical storage.

Pure stdlib so it is unit-testable without a database. See
``docs/wrreb_canonical_storage_design.md`` §4 (fingerprints) and §3 (identity).
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Optional

# The contract version participates in the batch fingerprint so a parser or
# contract change makes re-imports distinguishable rather than colliding. The
# string itself lives in samvision.contract (single source of truth).
from ..contract import CONTRACT_VERSION, SUPPORTED_CONTRACT_VERSIONS

# Back-compat alias for existing importers.
RECOGNIZED_CONTRACT_VERSIONS = SUPPORTED_CONTRACT_VERSIONS

# Fields that define a record's identity + critical values. Used for the record
# fingerprint that drives promotion idempotency and conflict detection.
RECORD_FINGERPRINT_FIELDS = (
    "mls_number",
    "linc_number",
    "normalized_property_id",
    "sold_date",
    "sold_price",
    "list_price",
    "address",
    "postal_code",
)


def _norm_text(value: Optional[str]) -> str:
    """Lower-case, collapse whitespace, keep only alphanumerics. Stable key."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def normalized_property_id(
    linc_number: Optional[str],
    address: Optional[str],
    postal_code: Optional[str],
) -> Optional[str]:
    """Preferred property identity: LINC if present, else normalized addr+postal.

    Returns ``None`` when neither a LINC nor any address/postal is available, so
    callers never key on an empty string.
    """
    if linc_number:
        linc = _norm_text(linc_number)
        if linc:
            return f"linc:{linc}"
    addr = _norm_text(address)
    postal = _norm_text(postal_code)
    if addr or postal:
        return f"addr:{addr}|{postal}"
    return None


def _canonical_value(value: Any) -> str:
    """Stable string form so fingerprints do not depend on int/float/str typing."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def batch_fingerprint(
    single_line_sha256: str,
    client_full_sha256: str,
    parser_version: str,
    contract_version: str,
) -> str:
    """Deterministic, collision-resistant id for a report pair + versions.

    Order-independent w.r.t. which file is 'single line' vs 'client full' would
    be wrong (they are distinct roles), so the two hashes are kept positional.
    """
    payload = json.dumps(
        {
            "single_line_sha256": single_line_sha256,
            "client_full_sha256": client_full_sha256,
            "parser_version": parser_version,
            "contract_version": contract_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_fingerprint(record: Mapping[str, Any]) -> str:
    """Deterministic id for a sale from its identifying + critical value fields."""
    payload = json.dumps(
        {f: _canonical_value(record.get(f)) for f in RECORD_FINGERPRINT_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
