"""SQLAlchemy Core schema for WRREB canonical storage.

Mirrors ``db/migrations/0001_wrreb_canonical_storage.up.sql``. Generic column
types map onto PostgreSQL (production) and SQLite (synthetic tests). Timestamps
use ``CURRENT_TIMESTAMP`` (portable) rather than ``now()``.

Keep this in sync with the SQL migration; the migration owns production DDL,
this module is the runtime/test schema.
"""
from __future__ import annotations

from typing import Any, Mapping

import sqlalchemy as sa

metadata = sa.MetaData()

# ---------------------------------------------------------------------------
# Status / severity vocabularies (also enforced by CHECK constraints).
# ---------------------------------------------------------------------------
BATCH_STATUSES = ("parsed", "staged", "review_required", "approved", "rejected", "failed")
VALIDATION_STATUSES = ("accepted", "warning")
ISSUE_SEVERITIES = ("warning", "conflict", "rejection", "needs_ocr")
CANONICAL_QUALITY_STATUSES = ("approved", "flagged")

_now = sa.text("CURRENT_TIMESTAMP")

# BIGINT on PostgreSQL, INTEGER on SQLite. SQLite only autoincrements an
# INTEGER PRIMARY KEY (rowid alias); a BIGINT PK would come back NULL.
_PK = sa.BigInteger().with_variant(sa.Integer(), "sqlite")


def _property_columns() -> list[sa.Column]:
    """Property/value columns shared by staging_sales and canonical_sales.

    Optional property fields are nullable with no default (no fabricated data).
    Prices/tax are NUMERIC; sold_date is a real DATE.
    """
    return [
        sa.Column("mls_number", sa.Text, nullable=False),
        sa.Column("linc_number", sa.Text),
        sa.Column("normalized_property_id", sa.Text),
        sa.Column("address", sa.Text),
        sa.Column("postal_code", sa.Text),
        sa.Column("area_code", sa.Text),
        sa.Column("neighbourhood", sa.Text),
        sa.Column("list_price", sa.Numeric(12, 2)),
        sa.Column("sold_price", sa.Numeric(12, 2)),
        sa.Column("sold_date", sa.Date),
        sa.Column("dom", sa.Integer),
        sa.Column("property_type", sa.Text),
        sa.Column("style", sa.Text),
        sa.Column("year_built", sa.Integer),
        sa.Column("living_area_sqft", sa.Integer),
        sa.Column("bedrooms_above_grade", sa.Integer),
        sa.Column("bedrooms_total", sa.Integer),
        sa.Column("full_bathrooms", sa.Integer),
        sa.Column("half_bathrooms", sa.Integer),
        sa.Column("basement_type", sa.Text),
        sa.Column("basement_development", sa.Text),
        sa.Column("finished_basement_sqft", sa.Integer),
        sa.Column("lot_front_ft", sa.Numeric(8, 2)),
        sa.Column("lot_depth_ft", sa.Numeric(8, 2)),
        sa.Column("lot_area_sqft", sa.Numeric(12, 2)),
        sa.Column("garage_type", sa.Text),
        sa.Column("parking_description", sa.Text),
        sa.Column("parking_spaces", sa.Integer),
        sa.Column("new_construction", sa.Boolean),
        sa.Column("gross_tax", sa.Numeric(12, 2)),
        sa.Column("tax_year", sa.Integer),
        sa.Column("parser_version", sa.Text),
        sa.Column("record_fingerprint", sa.Text, nullable=False),
    ]


import_batches = sa.Table(
    "import_batches",
    metadata,
    sa.Column("id", _PK, primary_key=True, autoincrement=True),
    sa.Column("batch_uuid", sa.Text, nullable=False, unique=True),
    sa.Column("batch_fingerprint", sa.Text, nullable=False, unique=True),
    sa.Column("contract_version", sa.Text, nullable=False),
    sa.Column("parser_version", sa.Text, nullable=False),
    sa.Column("single_line_filename", sa.Text, nullable=False),
    sa.Column("single_line_sha256", sa.Text, nullable=False),
    sa.Column("client_full_filename", sa.Text, nullable=False),
    sa.Column("client_full_sha256", sa.Text, nullable=False),
    sa.Column("search_start_date", sa.Date),
    sa.Column("search_end_date", sa.Date),
    sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
    sa.Column("imported_by", sa.Text),
    sa.Column("status", sa.Text, nullable=False, server_default=sa.text("'parsed'")),
    sa.Column("single_line_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("client_full_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("matched_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("accepted_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("rejected_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("warning_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("conflict_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("needs_ocr_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    sa.Column("approved_at", sa.DateTime(timezone=True)),
    sa.Column("approved_by", sa.Text),
    sa.Column("notes", sa.Text),
    sa.CheckConstraint(
        "status IN ('parsed','staged','review_required','approved','rejected','failed')",
        name="ck_import_batches_status",
    ),
)

staging_sales = sa.Table(
    "staging_sales",
    metadata,
    sa.Column("id", _PK, primary_key=True, autoincrement=True),
    sa.Column(
        "batch_id",
        sa.BigInteger,
        sa.ForeignKey("import_batches.id", ondelete="CASCADE"),
        nullable=False,
    ),
    *_property_columns(),
    sa.Column("source_single_line_page", sa.Integer),
    sa.Column("source_client_full_page", sa.Integer),
    sa.Column("validation_status", sa.Text, nullable=False, server_default=sa.text("'accepted'")),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
    sa.UniqueConstraint("batch_id", "mls_number", name="uq_staging_batch_mls"),
    sa.CheckConstraint(
        "validation_status IN ('accepted','warning')",
        name="ck_staging_validation_status",
    ),
    sa.Index("ix_staging_sold_date", "sold_date"),
    sa.Index("ix_staging_linc", "linc_number"),
    sa.Index("ix_staging_normalized_prop_id", "normalized_property_id"),
    sa.Index("ix_staging_record_fingerprint", "record_fingerprint"),
)

import_issues = sa.Table(
    "import_issues",
    metadata,
    sa.Column("id", _PK, primary_key=True, autoincrement=True),
    sa.Column(
        "batch_id",
        sa.BigInteger,
        sa.ForeignKey("import_batches.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column(
        "staging_sale_id",
        sa.BigInteger,
        sa.ForeignKey("staging_sales.id", ondelete="CASCADE"),
    ),
    sa.Column("mls_number", sa.Text),
    sa.Column("severity", sa.Text, nullable=False),
    sa.Column("reason_code", sa.Text, nullable=False),
    sa.Column("field_name", sa.Text),
    sa.Column("single_line_value", sa.Text),
    sa.Column("client_full_value", sa.Text),
    sa.Column("message", sa.Text),
    sa.Column("source_page", sa.Integer),
    sa.Column("resolved", sa.Boolean, nullable=False, server_default=sa.text("0")),
    sa.Column("resolution_notes", sa.Text),
    sa.Column("resolved_at", sa.DateTime(timezone=True)),
    sa.Column("resolved_by", sa.Text),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
    sa.CheckConstraint(
        "severity IN ('warning','conflict','rejection','needs_ocr')",
        name="ck_import_issues_severity",
    ),
    sa.Index("ix_issues_batch_sev_resolved", "batch_id", "severity", "resolved"),
)

canonical_sales = sa.Table(
    "canonical_sales",
    metadata,
    sa.Column("id", _PK, primary_key=True, autoincrement=True),
    *_property_columns(),
    sa.Column(
        "source_batch_id",
        sa.BigInteger,
        sa.ForeignKey("import_batches.id", ondelete="RESTRICT"),
    ),
    sa.Column("data_quality_status", sa.Text, nullable=False, server_default=sa.text("'approved'")),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=_now),
    sa.UniqueConstraint("mls_number", name="uq_canonical_mls"),
    sa.CheckConstraint(
        "data_quality_status IN ('approved','flagged')",
        name="ck_canonical_quality_status",
    ),
    sa.Index("ix_canonical_sold_date", "sold_date"),
    sa.Index("ix_canonical_neighbourhood", "neighbourhood"),
    sa.Index("ix_canonical_area_code", "area_code"),
    sa.Index("ix_canonical_linc", "linc_number"),
    sa.Index("ix_canonical_normalized_prop_id", "normalized_property_id"),
    sa.Index("ix_canonical_sold_price", "sold_price"),
    sa.Index("ix_canonical_property_type", "property_type"),
)


# ---------------------------------------------------------------------------
# Privacy guard: banned fields must never enter storage model objects.
# ---------------------------------------------------------------------------
PRIVACY_BANNED_FIELDS = frozenset({
    "agent", "agent_name", "listing_agent", "selling_agent", "agent_phone",
    "agent_email", "brokerage", "brokerage_name", "office", "office_phone",
    "phone", "phone_number", "email", "showing_instructions", "showing",
    "appointment", "appointment_instructions", "lockbox", "lockbox_info",
    "remarks", "public_remarks", "private_remarks", "agent_remarks",
    "realtor_remarks", "broker_remarks", "notes_private", "contact",
    "contact_name", "contact_phone",
})

# Columns a staging/canonical insert is allowed to set from parsed data.
_STAGING_COLUMNS = frozenset(c.name for c in staging_sales.columns) - {"id", "created_at"}
_CANONICAL_COLUMNS = frozenset(c.name for c in canonical_sales.columns) - {
    "id", "created_at", "updated_at",
}


class PrivacyViolation(ValueError):
    """Raised when a banned private field is offered to a storage model object."""


def _reject_private_keys(record: Mapping[str, Any]) -> None:
    banned = {k for k in record if k.lower() in PRIVACY_BANNED_FIELDS}
    if banned:
        # Never echo the values — only the offending key names.
        raise PrivacyViolation(
            "private fields are not allowed in canonical storage: "
            + ", ".join(sorted(banned))
        )


def build_staging_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist a parsed mapping into a staging_sales insert dict.

    Rejects private fields and silently drops unknown non-private keys so a
    widened parser cannot inject columns. ``record_fingerprint`` is required.
    """
    _reject_private_keys(record)
    row = {k: v for k, v in record.items() if k in _STAGING_COLUMNS}
    if not row.get("mls_number"):
        raise ValueError("staging row requires an mls_number")
    if not row.get("record_fingerprint"):
        raise ValueError("staging row requires a record_fingerprint")
    return row


def build_canonical_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist a parsed/staged mapping into a canonical_sales insert dict."""
    _reject_private_keys(record)
    row = {k: v for k, v in record.items() if k in _CANONICAL_COLUMNS}
    if not row.get("mls_number"):
        raise ValueError("canonical row requires an mls_number")
    if not row.get("record_fingerprint"):
        raise ValueError("canonical row requires a record_fingerprint")
    return row


def create_all(engine: sa.Engine) -> None:
    """Create the four tables on an engine (used by tests on SQLite)."""
    metadata.create_all(engine)
