"""Read-only leave-one-out comparable-sales backtest CLI.

Example::

    .venv/bin/python -m samvision.backtesting.run_comparable_backtest \\
      --database-url "$STAGING_DATABASE_URL" \\
      --output-dir /tmp/samvision-comparable-backtest

The database URL and output directory are both required and explicit — no
production URL is ever inferred and no default database is used. The path only
issues SELECTs (no write-authorization flag exists), consults no model, and
refuses to write into tracked source directories. Credentials are never printed.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from . import reporting
from .runner import BacktestRunner


def _sanitized_target(database_url: str) -> str:
    """host:port/database only — never the user or password."""
    try:
        from sqlalchemy.engine import make_url
        u = make_url(database_url)
        return f"{u.host}:{u.port}/{u.database}"
    except Exception:
        return "<database>"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="samvision.backtesting.run_comparable_backtest",
        description="Read-only leave-one-out backtest of the comparable-sales engine.",
    )
    p.add_argument("--database-url", required=True,
                   help="Explicit read-only database URL (required; no default, no inference).")
    p.add_argument("--output-dir", required=True,
                   help="Explicit output directory (must not be a tracked source dir).")
    p.add_argument("--min-group", type=int, default=3,
                   help="Minimum samples before a grouped metric is marked stable (default 3).")
    p.add_argument("--top-failures", type=int, default=15,
                   help="How many largest-error cases to include in the failure report.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Validate the output directory *before* touching the database.
    try:
        reporting.guard_output_dir(args.output_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Lazy imports so unit tests need no SQLAlchemy driver.
    from samvision.comparables.datasource import PostgresDataSource
    from samvision.comparables.service import ComparableService
    from samvision.storage.dataset import CanonicalDataset

    subjects_ds = CanonicalDataset(args.database_url)
    engine_ds = PostgresDataSource(args.database_url)
    try:
        subjects = subjects_ds.sales(min_data_quality="approved")
        service = ComparableService(engine_ds)
        report = BacktestRunner(service).run(subjects, min_group=args.min_group)
        written = reporting.write_reports(report, args.output_dir)
    finally:
        engine_ds.dispose()
        subjects_ds.dispose()

    print(f"target: {_sanitized_target(args.database_url)}")
    print(reporting.terminal_summary(report))
    print("wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
