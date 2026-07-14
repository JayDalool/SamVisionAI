"""Repository layer for WRREB canonical storage.

All data operations take an explicit SQLAlchemy ``Connection`` so the caller owns
the transaction boundary (``engine.begin()`` → commit/rollback). Nothing here
reads ``SAMVISION_DB_*`` or connects implicitly; ``create_engine_from_url``
refuses an empty URL.
"""
from __future__ import annotations

import uuid
from typing import Any, Iterable, Mapping, Optional

import sqlalchemy as sa

from . import models
from .fingerprints import record_fingerprint


class InvalidDatabaseUrl(ValueError):
    """Raised when no explicit, non-empty database URL is provided."""


class DuplicateBatch(Exception):
    """Raised when a batch fingerprint already exists."""


class DuplicateMls(Exception):
    """Raised when the same MLS appears twice within one batch."""


class PromotionBlocked(Exception):
    """Raised when a batch cannot be promoted, or a record conflicts."""


def assert_explicit_database_url(url: Optional[str]) -> str:
    """Guard: the URL must be explicitly provided and non-empty."""
    if url is None or not str(url).strip():
        raise InvalidDatabaseUrl(
            "an explicit, non-empty --database-url is required; "
            "storage never infers credentials from the environment"
        )
    return str(url).strip()


def create_engine_from_url(url: str, *, echo: bool = False) -> sa.Engine:
    """Build an engine from an explicit URL. Never falls back to env vars."""
    url = assert_explicit_database_url(url)
    engine = sa.create_engine(url, echo=echo, future=True)
    if engine.dialect.name == "sqlite":
        # Enforce FK constraints (incl. ON DELETE) on SQLite, off by default.
        @sa.event.listens_for(engine, "connect")
        def _fk_pragma(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


# ---------------------------------------------------------------------------
# import_batches
# ---------------------------------------------------------------------------
def batch_exists_by_fingerprint(conn: sa.Connection, fingerprint: str) -> bool:
    stmt = sa.select(models.import_batches.c.id).where(
        models.import_batches.c.batch_fingerprint == fingerprint
    )
    return conn.execute(stmt).first() is not None


def insert_batch(conn: sa.Connection, batch: Mapping[str, Any]) -> int:
    """Insert an import_batches row; raises DuplicateBatch on fingerprint clash."""
    fingerprint = batch["batch_fingerprint"]
    if batch_exists_by_fingerprint(conn, fingerprint):
        raise DuplicateBatch(f"batch fingerprint already imported: {fingerprint}")
    values = dict(batch)
    values.setdefault("batch_uuid", str(uuid.uuid4()))
    stmt = sa.insert(models.import_batches).values(**values).returning(
        models.import_batches.c.id
    )
    return int(conn.execute(stmt).scalar_one())


def update_batch_status(
    conn: sa.Connection, batch_id: int, status: str, **fields: Any
) -> None:
    if status not in models.BATCH_STATUSES:
        raise ValueError(f"unknown batch status: {status}")
    stmt = (
        sa.update(models.import_batches)
        .where(models.import_batches.c.id == batch_id)
        .values(status=status, **fields)
    )
    conn.execute(stmt)


def get_batch(conn: sa.Connection, *, batch_id: Optional[int] = None,
              batch_uuid: Optional[str] = None) -> Optional[sa.Row]:
    t = models.import_batches
    if batch_id is not None:
        stmt = sa.select(t).where(t.c.id == batch_id)
    elif batch_uuid is not None:
        stmt = sa.select(t).where(t.c.batch_uuid == batch_uuid)
    else:
        raise ValueError("get_batch requires batch_id or batch_uuid")
    return conn.execute(stmt).first()


# ---------------------------------------------------------------------------
# staging_sales
# ---------------------------------------------------------------------------
def insert_staging_rows(
    conn: sa.Connection, batch_id: int, rows: Iterable[Mapping[str, Any]]
) -> dict[str, int]:
    """Insert accepted rows for a batch. Returns {mls_number: staging_id}.

    Raises DuplicateMls if the same MLS appears twice in the input.
    """
    mls_to_id: dict[str, int] = {}
    for raw in rows:
        row = models.build_staging_row(raw)
        mls = row["mls_number"]
        if mls in mls_to_id:
            raise DuplicateMls(f"duplicate MLS within batch: {mls}")
        stmt = (
            sa.insert(models.staging_sales)
            .values(batch_id=batch_id, **row)
            .returning(models.staging_sales.c.id)
        )
        mls_to_id[mls] = int(conn.execute(stmt).scalar_one())
    return mls_to_id


def count_staging_rows(conn: sa.Connection, batch_id: int) -> int:
    stmt = sa.select(sa.func.count()).select_from(models.staging_sales).where(
        models.staging_sales.c.batch_id == batch_id
    )
    return int(conn.execute(stmt).scalar_one())


# ---------------------------------------------------------------------------
# import_issues
# ---------------------------------------------------------------------------
def insert_issues(
    conn: sa.Connection,
    batch_id: int,
    issues: Iterable[Mapping[str, Any]],
    mls_to_staging_id: Optional[Mapping[str, int]] = None,
) -> int:
    """Insert issue rows. Links to a staging row when the MLS was staged."""
    mls_to_staging_id = mls_to_staging_id or {}
    payload = []
    for issue in issues:
        severity = issue.get("severity")
        if severity not in models.ISSUE_SEVERITIES:
            raise ValueError(f"unknown issue severity: {severity!r}")
        mls = issue.get("mls_number")
        payload.append({
            "batch_id": batch_id,
            "staging_sale_id": mls_to_staging_id.get(mls) if mls else None,
            "mls_number": mls,
            "severity": severity,
            "reason_code": issue.get("reason_code", ""),
            "field_name": issue.get("field_name"),
            "single_line_value": issue.get("single_line_value"),
            "client_full_value": issue.get("client_full_value"),
            "message": issue.get("message"),
            "source_page": issue.get("source_page"),
        })
    if payload:
        conn.execute(sa.insert(models.import_issues), payload)
    return len(payload)


def count_unresolved_issues(
    conn: sa.Connection, batch_id: int, severities: Iterable[str]
) -> int:
    stmt = (
        sa.select(sa.func.count())
        .select_from(models.import_issues)
        .where(
            models.import_issues.c.batch_id == batch_id,
            models.import_issues.c.severity.in_(list(severities)),
            models.import_issues.c.resolved.is_(False),
        )
    )
    return int(conn.execute(stmt).scalar_one())


def batch_staging_summary(conn: sa.Connection, batch_id: int) -> dict[str, Any]:
    """Post-load summary read straight from the tables (verification aid)."""
    return {
        "batch_id": batch_id,
        "staging_rows": count_staging_rows(conn, batch_id),
        "unresolved_conflicts": count_unresolved_issues(conn, batch_id, ("conflict",)),
        "unresolved_rejections": count_unresolved_issues(conn, batch_id, ("rejection",)),
        "unresolved_needs_ocr": count_unresolved_issues(conn, batch_id, ("needs_ocr",)),
    }


# ---------------------------------------------------------------------------
# Promotion (staging -> canonical). See design §7.
# ---------------------------------------------------------------------------
def _canonical_by_mls(conn: sa.Connection, mls: str) -> Optional[sa.Row]:
    stmt = sa.select(models.canonical_sales).where(
        models.canonical_sales.c.mls_number == mls
    )
    return conn.execute(stmt).first()


def promote_batch(
    conn: sa.Connection, batch_id: int, *, approved_by: Optional[str] = None
) -> dict[str, Any]:
    """Promote a staged batch's rows into canonical_sales.

    Gates (design §7): batch status must be 'staged', no unresolved conflicts or
    rejections, and accepted_count must equal the staged row count. Per record:
      - same MLS + same fingerprint  -> idempotent no-op
      - same MLS + different values   -> block + create a 'conflict' issue
      - different MLS + same LINC     -> preserved as another transaction
    Never overwrites an existing canonical row silently.
    """
    batch = get_batch(conn, batch_id=batch_id)
    if batch is None:
        raise PromotionBlocked(f"batch {batch_id} not found")
    if batch.status != "staged":
        raise PromotionBlocked(
            f"batch status must be 'staged' to promote, is '{batch.status}'"
        )
    if count_unresolved_issues(conn, batch_id, ("conflict",)):
        raise PromotionBlocked("batch has unresolved conflicts")
    if count_unresolved_issues(conn, batch_id, ("rejection",)):
        raise PromotionBlocked("batch has unresolved rejections")

    staged = list(conn.execute(
        sa.select(models.staging_sales).where(
            models.staging_sales.c.batch_id == batch_id
        )
    ))
    if batch.accepted_count != len(staged):
        raise PromotionBlocked(
            f"accepted_count {batch.accepted_count} != staged rows {len(staged)}"
        )

    inserted = skipped = 0
    for row in staged:
        mapping = dict(row._mapping)
        existing = _canonical_by_mls(conn, mapping["mls_number"])
        if existing is not None:
            if existing.record_fingerprint == mapping["record_fingerprint"]:
                skipped += 1  # idempotent no-op
                continue
            # Same MLS, different critical values -> block, record a conflict.
            insert_issues(conn, batch_id, [{
                "mls_number": mapping["mls_number"],
                "severity": "conflict",
                "reason_code": "CANONICAL_FINGERPRINT_CONFLICT",
                "field_name": "record_fingerprint",
                "message": "existing canonical sale differs from staged record",
            }])
            raise PromotionBlocked(
                f"canonical conflict for MLS {mapping['mls_number']}; "
                "resolve before promoting"
            )
        canonical_row = models.build_canonical_row({
            **mapping,
            "source_batch_id": batch_id,
        })
        canonical_row["source_batch_id"] = batch_id
        conn.execute(sa.insert(models.canonical_sales).values(**canonical_row))
        inserted += 1

    update_batch_status(
        conn, batch_id, "approved",
        approved_by=approved_by,
        approved_at=sa.text("CURRENT_TIMESTAMP"),
    )
    return {"batch_id": batch_id, "inserted": inserted, "idempotent_skipped": skipped}
