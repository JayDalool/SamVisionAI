"""Serialize backtest results to JSON + CSV, sanitized and privacy-safe.

MLS is masked, private fields are never emitted, and output is refused into
tracked source directories. Writes files only under an explicit output dir.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import AggregateReport, BacktestCase

# Repo directories that must never receive report artifacts.
TRACKED_SOURCE_DIRS = frozenset({
    "samvision", "tests", "docs", "app", "utils", "ml", "data", "db",
    "scripts", ".streamlit", "tools", "parsed_csv", ".git",
})

_REPO_ROOT = Path(__file__).resolve().parents[2]


def guard_output_dir(output_dir: str, *, repo_root: Path = _REPO_ROOT) -> Path:
    """Resolve and validate the output directory. Refuse tracked source dirs."""
    resolved = Path(output_dir).resolve()
    root = repo_root.resolve()
    if resolved == root or root in resolved.parents:
        rel = resolved.relative_to(root)
        top = rel.parts[0] if rel.parts else ""
        if top == "" or top in TRACKED_SOURCE_DIRS:
            raise ValueError(
                f"refusing to write reports into tracked source path: {resolved} "
                f"(use /tmp or a gitignored output directory)"
            )
    return resolved


def failure_cases(report: AggregateReport, *, top_n: int = 15) -> list[dict[str, Any]]:
    """Largest-absolute-error valued cases, sanitized per the design's privacy rules."""
    ranked = sorted(report.cases, key=lambda c: c.absolute_error, reverse=True)[:top_n]
    rows: list[dict[str, Any]] = []
    for i, c in enumerate(ranked, start=1):
        d = c.sanitized_dict()
        d = {"index": i, **d}
        rows.append(d)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")  # empty but present
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_reports(report: AggregateReport, output_dir: str) -> list[str]:
    """Write aggregate JSON, per-case CSV, grouped CSVs, and failure CSV.

    Returns the list of written file paths. No private data is emitted.
    """
    out = guard_output_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    agg = out / "backtest_aggregate.json"
    agg.write_text(json.dumps(report.summary_dict(), indent=2, sort_keys=True))
    written.append(str(agg))

    cases_csv = out / "backtest_cases.csv"
    _write_csv(cases_csv, [c.sanitized_dict() for c in report.cases])
    written.append(str(cases_csv))

    skips_csv = out / "backtest_skips.csv"
    _write_csv(skips_csv, [s.sanitized_dict() for s in report.skips])
    written.append(str(skips_csv))

    for dim, groups in report.grouped.items():
        gpath = out / f"backtest_group_{dim}.csv"
        _write_csv(gpath, [g.to_dict() for g in groups])
        written.append(str(gpath))

    fpath = out / "backtest_failure_cases.csv"
    _write_csv(fpath, failure_cases(report))
    written.append(str(fpath))

    return written


def terminal_summary(report: AggregateReport) -> str:
    o = report.overall
    lines = [
        "Comparable-sales backtest summary",
        f"  records={report.total_records} eligible={report.eligible} "
        f"valued={report.valued} insufficient={report.insufficient} "
        f"skipped_ineligible={report.skipped_ineligible} engine_errors={report.engine_errors}",
        f"  coverage_rate={report.coverage_rate:.1%}",
        f"  MAE={o.get('mae')} median_AE={o.get('median_absolute_error')} RMSE={o.get('rmse')}",
        f"  MAPE={o.get('mape_pct')}%  median_APE={o.get('median_ape_pct')}%",
        f"  within: 5%={report.thresholds.get('within_5pct')}  "
        f"10%={report.thresholds.get('within_10pct')}  "
        f"15%={report.thresholds.get('within_15pct')}  "
        f"20%={report.thresholds.get('within_20pct')}",
    ]
    return "\n".join(lines)
