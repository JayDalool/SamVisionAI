"""Safe staging loader for WRREB pipeline output (dry-run by default).

    # Dry run (default): validates + shows intended inserts, writes nothing.
    python -m samvision.storage.load_wrreb_batch --input pipeline_output/<batch>

    # Staging write: requires BOTH an explicit URL and the authorization flag.
    python -m samvision.storage.load_wrreb_batch \
        --input pipeline_output/<batch> \
        --database-url "$STAGING_DATABASE_URL" \
        --authorize-staging-write

There is no --authorize-production flag. The database URL is never inferred from
the environment. A write runs in one transaction; any failure rolls back
everything and exits non-zero with a privacy-safe reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

from . import repository
from .batch_loader import LoadPlan, ManifestError, build_load_plan


def _preview(plan: LoadPlan, limit: int = 5) -> list[dict[str, Any]]:
    """Aggregate, privacy-safe preview of intended staging inserts."""
    out = []
    for row in plan.staging_rows[:limit]:
        out.append({
            "mls_number": row.get("mls_number"),
            "sold_date": str(row.get("sold_date")) if row.get("sold_date") else None,
            "sold_price": str(row.get("sold_price")) if row.get("sold_price") is not None else None,
            "normalized_property_id": row.get("normalized_property_id"),
            "validation_status": row.get("validation_status"),
        })
    return out


def _report(plan: LoadPlan, *, dry_run: bool, duplicate: Optional[bool],
            wrote: bool, write_result: Optional[dict] = None) -> dict[str, Any]:
    return {
        "input_dir": plan.input_dir,
        "dry_run": dry_run,
        "counts": plan.counts,
        "duplicate_batch": duplicate,
        "refusals": plan.refusals,
        "intended_staging_preview": _preview(plan),
        "wrote_to_database": wrote,
        "write_result": write_result,
        "safety": "no database writes were made" if not wrote
                  else "staging write committed in a single transaction",
    }


def run(input_dir: str, *, database_url: Optional[str] = None,
        authorize_staging_write: bool = False, dry_run: bool = True) -> int:
    try:
        plan = build_load_plan(input_dir)
    except ManifestError as exc:
        print(json.dumps({"error": "manifest_invalid", "reason": str(exc)}, indent=2))
        return 2

    # A write is gated on BOTH an explicit --database-url and
    # --authorize-staging-write (design §5). The authorization flag is the intent
    # to write; dry-run is the default only when the flag is absent, so callers
    # need not also pass --no-dry-run. Without the flag, this stays a dry run.
    want_write = authorize_staging_write
    duplicate: Optional[bool] = None

    # Read-only duplicate check when a URL is available (both modes).
    if database_url and str(database_url).strip():
        engine = repository.create_engine_from_url(database_url)
        try:
            with engine.connect() as conn:
                duplicate = repository.batch_exists_by_fingerprint(
                    conn, plan.batch["batch_fingerprint"]
                )
        finally:
            engine.dispose()
        if duplicate:
            plan.refusals.append("duplicate batch fingerprint already in database")

    if not want_write:
        print(json.dumps(_report(plan, dry_run=True, duplicate=duplicate,
                                  wrote=False), indent=2))
        return 0 if plan.safe_to_write else 1

    # ---- write path (authorized) ----
    try:
        url = repository.assert_explicit_database_url(database_url)
    except repository.InvalidDatabaseUrl as exc:
        print(json.dumps({"error": "authorization", "reason": str(exc)}, indent=2))
        return 2

    if not plan.safe_to_write:
        print(json.dumps(_report(plan, dry_run=False, duplicate=duplicate,
                                  wrote=False), indent=2))
        print("[safety] refusing staging write: unsafe batch (see refusals).")
        return 1

    engine = repository.create_engine_from_url(url)
    try:
        with engine.begin() as conn:  # single transaction; rolls back on error
            if repository.batch_exists_by_fingerprint(conn, plan.batch["batch_fingerprint"]):
                raise repository.DuplicateBatch("duplicate batch fingerprint")
            batch_id = repository.insert_batch(conn, plan.batch)
            mls_to_id = repository.insert_staging_rows(conn, batch_id, plan.staging_rows)
            repository.insert_issues(conn, batch_id, plan.issues, mls_to_id)
            staged = repository.count_staging_rows(conn, batch_id)
            unresolved = (
                repository.count_unresolved_issues(conn, batch_id, ("rejection",))
                + repository.count_unresolved_issues(conn, batch_id, ("warning",))
                + repository.count_unresolved_issues(conn, batch_id, ("needs_ocr",))
            )
            status = "review_required" if unresolved else "staged"
            repository.update_batch_status(conn, batch_id, status)
            write_result = {"batch_id": batch_id, "batch_status": status,
                            "staged_rows": staged, "issue_rows": len(plan.issues)}
    except (repository.DuplicateBatch, repository.DuplicateMls,
            repository.InvalidDatabaseUrl) as exc:
        print(json.dumps({"error": "write_refused", "reason": str(exc),
                          "wrote_to_database": False}, indent=2))
        return 1
    except Exception as exc:  # transaction already rolled back
        # Privacy-safe: report the exception type, not row values.
        print(json.dumps({"error": "write_failed_rolled_back",
                          "reason": type(exc).__name__,
                          "wrote_to_database": False}, indent=2))
        return 1
    finally:
        engine.dispose()

    print(json.dumps(_report(plan, dry_run=False, duplicate=duplicate,
                             wrote=True, write_result=write_result), indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Safe WRREB staging loader (dry-run by default)."
    )
    p.add_argument("--input", required=True, help="pipeline_output/<batch> directory")
    p.add_argument("--database-url", default=None,
                   help="Explicit SQLAlchemy URL. Never inferred from the environment.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    p.add_argument("--authorize-staging-write", action="store_true", default=False,
                   help="Required (with --database-url and --no-dry-run) to write.")
    a = p.parse_args(argv)
    return run(a.input, database_url=a.database_url,
               authorize_staging_write=a.authorize_staging_write, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
