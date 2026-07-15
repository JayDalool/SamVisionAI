"""Read-only access to approved canonical sales.

For future comparables and training. Returns **only approved** canonical sales by
default. This layer performs no writes and does not change the production model or
the current Streamlit prediction path.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

import sqlalchemy as sa

from . import models, repository

# Higher quality first. "minimum data-quality" includes everything at least this good.
_QUALITY_ORDER = ("approved", "flagged")


def _allowed_qualities(minimum: str) -> list[str]:
    if minimum not in _QUALITY_ORDER:
        raise ValueError(f"unknown data-quality status: {minimum!r}")
    idx = _QUALITY_ORDER.index(minimum)
    return list(_QUALITY_ORDER[: idx + 1])


def query_canonical_sales(
    conn: sa.Connection,
    *,
    sold_start: Optional[date] = None,
    sold_end: Optional[date] = None,
    neighbourhood: Optional[str] = None,
    area_code: Optional[str] = None,
    property_type: Optional[str] = None,
    exclude_mls: Optional[str] = None,
    min_data_quality: str = "approved",
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return approved canonical sales matching the given filters.

    ``exclude_mls`` drops a specific MLS (e.g. the subject property itself). This
    function only ever SELECTs; it performs no writes.
    """
    t = models.canonical_sales
    stmt = sa.select(t).where(t.c.data_quality_status.in_(_allowed_qualities(min_data_quality)))
    if sold_start is not None:
        stmt = stmt.where(t.c.sold_date >= sold_start)
    if sold_end is not None:
        stmt = stmt.where(t.c.sold_date <= sold_end)
    if neighbourhood is not None:
        stmt = stmt.where(t.c.neighbourhood == neighbourhood)
    if area_code is not None:
        stmt = stmt.where(t.c.area_code == area_code)
    if property_type is not None:
        stmt = stmt.where(t.c.property_type == property_type)
    if exclude_mls is not None:
        stmt = stmt.where(t.c.mls_number != exclude_mls)
    # Stable ordering: sold_date desc, then mls asc (deterministic tie-break).
    stmt = stmt.order_by(t.c.sold_date.desc(), t.c.mls_number.asc())
    if limit is not None:
        stmt = stmt.limit(limit)
    return [dict(row._mapping) for row in conn.execute(stmt)]


class CanonicalDataset:
    """Read-only convenience wrapper over an explicit database URL."""

    def __init__(self, database_url: str):
        self._engine = repository.create_engine_from_url(database_url)

    def sales(self, **filters: Any) -> list[dict[str, Any]]:
        with self._engine.connect() as conn:
            return query_canonical_sales(conn, **filters)

    def count(self, **filters: Any) -> int:
        return len(self.sales(**filters))

    def dispose(self) -> None:
        self._engine.dispose()
