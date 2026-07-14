"""Safe command-line WRREB batch import (dry-run by default).

    python -m samvision.ingestion.import_wrreb_batch \
        --single-line "/srv/server/private/samvisionai/wrreb/incoming/....pdf" \
        --client-full "/srv/server/private/samvisionai/wrreb/incoming/....pdf" \
        --output "pipeline_output/<timestamp>" \
        --dry-run

The dry run makes NO database changes and NO model changes; it writes only into
the gitignored output directory and returns non-zero if critical reconciliation
fails. Production loading is intentionally not implemented here.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from . import agent_single_line, client_full
from .models import ReasonCode
from .provenance import make_batch_id, sha256_file
from .reconcile import reconcile
from .validate import validate, ValidationLimits

_CRITICAL_CONFLICTS = {ReasonCode.SOLD_DATE_CONFLICT, ReasonCode.SOLD_PRICE_CONFLICT,
                       ReasonCode.DUPLICATE_MLS}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def run(single_path: str, full_path: str, out_dir: str,
        dry_run: bool = True, authorize_production: bool = False,
        min_match_rate: float = 1.0) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batch_id = make_batch_id(single_path, full_path)

    sl_records, sl_diag = agent_single_line.parse(single_path)
    cf_records, cf_diag = client_full.parse(full_path)
    canon, summary = reconcile(sl_records, cf_records, batch_id=batch_id)
    canon = validate(canon, ValidationLimits())

    accepted = [c for c in canon if c.data_quality_status == "accepted"]
    rejected = [c for c in canon if c.data_quality_status == "rejected"]
    conflicts, warnings = [], []
    for c in canon:
        for i in c.issues:
            (conflicts if i.code in _CRITICAL_CONFLICTS else warnings).append(i.as_row())

    _write_csv(out / "accepted.csv", [c.to_row() for c in accepted])
    _write_csv(out / "rejected.csv", [c.to_row() for c in rejected])
    _write_csv(out / "conflicts.csv", conflicts)
    _write_csv(out / "warnings.csv", warnings)
    _write_csv(out / "extraction_diagnostics.csv",
               [d.as_row() for d in (sl_diag + cf_diag)])

    dates = sorted(c.sold_date for c in accepted if c.sold_date)
    hoods = sorted({c.neighbourhood for c in accepted if c.neighbourhood})
    match_rate = (summary["matched_both"] / max(summary["unique_mls_full"], 1))
    critical_fail = (
        summary["single_only"] > 0 or summary["full_only"] > 0
        or bool(summary["duplicate_mls"])
        or any(i["code"] in _CRITICAL_CONFLICTS for i in conflicts)
        or match_rate < min_match_rate
    )

    summary_out = {
        "batch_id": batch_id,
        "dry_run": dry_run,
        "authorize_production": authorize_production,
        "production_load": "not-implemented (MVP is dry-run only)",
        "sources": {
            "single_line": {"file": Path(single_path).name, "sha256": sha256_file(single_path)},
            "client_full": {"file": Path(full_path).name, "sha256": sha256_file(full_path)},
        },
        "reconciliation": summary,
        "match_rate_vs_client_full": round(match_rate, 4),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "conflicts": len(conflicts),
        "warnings": len(warnings),
        "sold_date_range": [str(dates[0]), str(dates[-1])] if dates else None,
        "neighbourhood_count": len(hoods),
        "pages": {
            "single_line": len(sl_diag),
            "client_full": len(cf_diag),
            "needs_ocr": sum(1 for d in (sl_diag + cf_diag) if d.status == "needs_ocr"),
        },
        "critical_reconciliation_failed": critical_fail,
    }
    (out / "summary.json").write_text(json.dumps(summary_out, indent=2))

    # stdout is aggregate-only (never prints remarks / contact / lockbox data)
    print(json.dumps({k: summary_out[k] for k in
                      ("batch_id", "reconciliation", "accepted", "rejected", "conflicts",
                       "warnings", "sold_date_range", "neighbourhood_count", "pages",
                       "critical_reconciliation_failed")}, indent=2))
    print(f"\n[output] {out.resolve()}")
    if not dry_run and not authorize_production:
        print("[safety] --no-dry-run requires --authorize-production; refusing any write.")
    print("[safety] no database or model changes were made.")
    return 1 if critical_fail else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dry-run WRREB two-report import + reconciliation.")
    p.add_argument("--single-line", required=True)
    p.add_argument("--client-full", required=True)
    p.add_argument("--output", default=f"pipeline_output/{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    p.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    p.add_argument("--authorize-production", action="store_true", default=False,
                   help="Reserved; production loading is not implemented in the MVP.")
    p.add_argument("--min-match-rate", type=float, default=1.0)
    a = p.parse_args(argv)
    return run(a.single_line, a.client_full, a.output, a.dry_run,
               a.authorize_production, a.min_match_rate)


if __name__ == "__main__":
    sys.exit(main())
