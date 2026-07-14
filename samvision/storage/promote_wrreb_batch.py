"""Promote a staged batch into canonical_sales (explicitly authorized).

    # Preview gates only (default): no writes.
    python -m samvision.storage.promote_wrreb_batch --batch-id <id> \
        --database-url "$STAGING_DATABASE_URL"

    # Promote: requires the authorization flag as well.
    python -m samvision.storage.promote_wrreb_batch --batch-id <id> \
        --database-url "$STAGING_DATABASE_URL" \
        --authorize-canonical-write

Promotion rules and idempotency live in ``repository.promote_batch`` (design §7).
A promotion runs in one transaction and rolls back on any failure. There is no
production flag here; the URL is always explicit.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import sqlalchemy as sa

from . import repository


def run(*, batch_id: Optional[int] = None, batch_uuid: Optional[str] = None,
        database_url: Optional[str] = None,
        authorize_canonical_write: bool = False,
        approved_by: Optional[str] = None) -> int:
    try:
        url = repository.assert_explicit_database_url(database_url)
    except repository.InvalidDatabaseUrl as exc:
        print(json.dumps({"error": "authorization", "reason": str(exc)}, indent=2))
        return 2

    engine = repository.create_engine_from_url(url)
    try:
        with engine.connect() as conn:
            batch = repository.get_batch(conn, batch_id=batch_id, batch_uuid=batch_uuid)
            if batch is None:
                print(json.dumps({"error": "not_found",
                                  "reason": "batch not found"}, indent=2))
                return 1
            summary = repository.batch_staging_summary(conn, batch.id)
            gate_ok = (
                batch.status == "staged"
                and summary["unresolved_conflicts"] == 0
                and summary["unresolved_rejections"] == 0
                and batch.accepted_count == summary["staging_rows"]
            )

        if not authorize_canonical_write:
            print(json.dumps({
                "dry_run": True,
                "batch_id": batch.id,
                "batch_status": batch.status,
                "gates_pass": gate_ok,
                "summary": summary,
                "safety": "no canonical writes were made",
            }, indent=2, default=str))
            return 0 if gate_ok else 1

        # ---- authorized write path (single transaction) ----
        with engine.begin() as conn:
            result = repository.promote_batch(conn, batch.id, approved_by=approved_by)
        print(json.dumps({"dry_run": False, **result,
                          "safety": "promotion committed in a single transaction"},
                         indent=2, default=str))
        return 0
    except repository.PromotionBlocked as exc:
        print(json.dumps({"error": "promotion_blocked", "reason": str(exc),
                          "wrote_to_database": False}, indent=2))
        return 1
    except Exception as exc:  # rolled back
        print(json.dumps({"error": "promotion_failed_rolled_back",
                          "reason": type(exc).__name__,
                          "wrote_to_database": False}, indent=2))
        return 1
    finally:
        engine.dispose()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Promote a staged WRREB batch to canonical.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch-id", type=int)
    g.add_argument("--batch-uuid")
    p.add_argument("--database-url", default=None,
                   help="Explicit SQLAlchemy URL. Never inferred from the environment.")
    p.add_argument("--authorize-canonical-write", action="store_true", default=False)
    p.add_argument("--approved-by", default=None)
    a = p.parse_args(argv)
    return run(batch_id=a.batch_id, batch_uuid=a.batch_uuid,
               database_url=a.database_url,
               authorize_canonical_write=a.authorize_canonical_write,
               approved_by=a.approved_by)


if __name__ == "__main__":
    sys.exit(main())
