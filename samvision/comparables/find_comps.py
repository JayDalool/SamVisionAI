"""Read-only comparable-sales CLI.

    .venv/bin/python -m samvision.comparables.find_comps \
        --database-url "$STAGING_DATABASE_URL" \
        --subject-json /path/to/subject.json \
        --limit 10

This command can only READ. There is no write-authorization flag. It validates
the subject before querying, never writes to the database, and never consults a
trained model. The terminal summary never prints addresses or full listing rows;
pass --show-address to include addresses in the JSON payload only (for a future
realtor UI). --output must be given an explicit path (it never defaults inside
tracked source directories).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import constants
from .datasource import PostgresDataSource
from .models import SubjectProperty, SubjectValidationError
from .service import ComparableService


def _load_subject(path: str) -> SubjectProperty:
    with open(path) as fh:
        data = json.load(fh)
    return SubjectProperty.from_mapping(data).validate()


def _terminal_summary(result_dict: dict) -> str:
    r = result_dict
    lines = [
        "WRREB comparable-sales (decision support, NOT an appraisal)",
        f"  status:            {r['status']}",
        f"  tier:              {r['tier']} (widened={r['tier_widened']})",
        f"  candidates/return: {r['candidate_count']} / {r['selected_comparable_count']}",
        f"  median similarity: {r['median_similarity']}",
        f"  indicated value:   {r['indicated_value']}"
        f"  (range {r['indicated_range_low']} – {r['indicated_range_high']})",
        f"  weighted med ppsf: {r['weighted_median_ppsf']}",
        f"  price spread:      {r['price_spread']}",
        f"  confidence:        {r['confidence']}",
        f"  sold-date range:   {r['sold_date_range']}",
        f"  warnings:          {r['warnings']}",
    ]
    return "\n".join(lines)


def run(*, database_url: str, subject_json: str, limit: Optional[int] = None,
        output: Optional[str] = None, show_address: bool = False) -> int:
    try:
        subject = _load_subject(subject_json)
    except (SubjectValidationError, ValueError, KeyError) as exc:
        print(json.dumps({"error": "invalid_subject", "reason": str(exc)}))
        return 2

    max_returned = limit if limit is not None else constants.MAX_RETURNED_COMPS
    source = PostgresDataSource(database_url)  # read-only
    try:
        service = ComparableService(source, max_returned=max_returned)
        result = service.find_comparables(subject)
    finally:
        source.dispose()

    payload = result.to_dict(include_address=show_address)
    if output:
        with open(output, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[output] wrote result JSON to {output}")
    else:
        # stdout JSON never includes addresses unless explicitly requested.
        print(json.dumps(result.to_dict(include_address=False), indent=2))

    print(_terminal_summary(payload), file=sys.stderr)
    return 0 if result.status == "ok" else 1


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Read-only WRREB comparable-sales finder.")
    p.add_argument("--database-url", required=True,
                   help="Explicit SQLAlchemy URL. Never inferred from the environment.")
    p.add_argument("--subject-json", required=True, help="Path to a subject-property JSON file.")
    p.add_argument("--limit", type=int, default=None, help="Max comparables returned.")
    p.add_argument("--output", default=None,
                   help="Explicit path for the JSON result (never defaults into the repo).")
    p.add_argument("--show-address", action="store_true", default=False,
                   help="Include addresses in the JSON payload only (never in the terminal).")
    a = p.parse_args(argv)
    return run(database_url=a.database_url, subject_json=a.subject_json,
               limit=a.limit, output=a.output, show_address=a.show_address)


if __name__ == "__main__":
    sys.exit(main())
