"""Single source of truth for the WRREB canonical contract version.

Both the ingestion CLI (which emits ``contract_version`` into ``summary.json``)
and the storage loader (which requires and validates it) import this constant so
the version string is never repeated across unrelated modules.
"""
from __future__ import annotations

CONTRACT_VERSION = "wrreb-canonical/1.0"

# The set of contract versions the current storage loader understands.
SUPPORTED_CONTRACT_VERSIONS = frozenset({CONTRACT_VERSION})
