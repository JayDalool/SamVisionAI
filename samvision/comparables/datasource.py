"""Read-only data sources for the comparable engine.

The service depends on the ``ComparableDataSource`` protocol, not on raw SQL.
Two implementations ship: an in-memory one (tests, no PostgreSQL) and a
PostgreSQL-backed one over ``samvision.storage.dataset`` (approved canonical
records only). Neither ever writes.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional, Protocol, Sequence


def _as_date(value: Any) -> Optional[date]:
    if value is None or isinstance(value, date):
        return value if isinstance(value, date) else None
    return date.fromisoformat(str(value))


class ComparableDataSource(Protocol):
    """Read-only candidate source. Implementations return approved rows only."""

    def fetch(
        self,
        *,
        property_type: str,
        sold_start: date,
        sold_end: date,
        neighbourhood: Optional[str] = None,
        area_code: Optional[str] = None,
        exclude_mls: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        ...


class InMemoryDataSource:
    """Filters a list of canonical-row dicts. Enforces approved-only."""

    def __init__(self, rows: Sequence[dict[str, Any]]):
        self._rows = list(rows)

    def fetch(self, *, property_type, sold_start, sold_end, neighbourhood=None,
              area_code=None, exclude_mls=None, limit=None):
        out = []
        for r in self._rows:
            if r.get("data_quality_status") != "approved":
                continue
            if r.get("property_type") != property_type:
                continue
            sd = _as_date(r.get("sold_date"))
            if sd is None or sd < sold_start or sd > sold_end:
                continue
            if neighbourhood is not None and r.get("neighbourhood") != neighbourhood:
                continue
            if area_code is not None and r.get("area_code") != area_code:
                continue
            if exclude_mls is not None and r.get("mls_number") == exclude_mls:
                continue
            out.append(dict(r))
        # Stable deterministic order: sold_date desc, then mls asc.
        out.sort(key=lambda r: str(r.get("mls_number") or ""))
        out.sort(key=lambda r: _as_date(r.get("sold_date")) or date.min, reverse=True)
        if limit is not None:
            out = out[:limit]
        return out


class PostgresDataSource:
    """Read-only PostgreSQL/SQLite source over the canonical dataset layer."""

    def __init__(self, database_url: str):
        # Imported lazily so the in-memory path needs no SQLAlchemy driver.
        from samvision.storage.dataset import CanonicalDataset
        self._ds = CanonicalDataset(database_url)

    def fetch(self, *, property_type, sold_start, sold_end, neighbourhood=None,
              area_code=None, exclude_mls=None, limit=None):
        return self._ds.sales(
            property_type=property_type,
            sold_start=sold_start,
            sold_end=sold_end,
            neighbourhood=neighbourhood,
            area_code=area_code,
            exclude_mls=exclude_mls,
            min_data_quality="approved",
            limit=limit,
        )

    def dispose(self) -> None:
        self._ds.dispose()
