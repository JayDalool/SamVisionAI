"""Tests for the WRREB canonical storage layer (samvision/storage).

Synthetic data only — no real WRREB listing data is used or committed. Each test
builds a fake pipeline-output directory and runs against a throwaway SQLite
database, so no PostgreSQL, psycopg2, or production credentials are involved.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import sqlalchemy as sa

from samvision.storage import (
    batch_loader,
    dataset,
    fingerprints,
    load_wrreb_batch,
    models,
    repository,
)

ACCEPTED_FIELDS = [
    "mls_number", "linc_number", "address", "postal_code", "area_code",
    "neighbourhood", "list_price", "sold_price", "sold_date", "dom",
    "property_type_code", "style_code", "year_built", "living_area_sqft",
    "bedrooms_above_grade", "bedrooms_total", "full_bathrooms", "half_bathrooms",
    "basement_type", "lot_front_ft", "lot_depth_ft", "gross_tax", "tax_year",
    "parser_version", "issue_codes",
]


def _accepted_row(**over):
    base = {
        "mls_number": "202600001",
        "linc_number": "012R021018000",
        "address": "123 Main St",
        "postal_code": "R2M1A1",
        "area_code": "1A",
        "neighbourhood": "River Heights",
        "list_price": "500000",
        "sold_price": "482500",
        "sold_date": "2026-05-19",
        "dom": "7",
        "property_type_code": "RES",
        "style_code": "BUNGALOW",
        "year_built": "1975",
        "living_area_sqft": "1100",
        "bedrooms_above_grade": "3",
        "bedrooms_total": "4",
        "full_bathrooms": "1",
        "half_bathrooms": "1",
        "basement_type": "Full",
        "lot_front_ft": "50",
        "lot_depth_ft": "100",
        "gross_tax": "3200.50",
        "tax_year": "2025",
        "parser_version": "single_line/1.0.0+client_full/1.0.0",
        "issue_codes": "",
    }
    base.update(over)
    return base


def _write_csv(path: Path, fieldnames, rows):
    import csv
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def make_pipeline_output(
    root: Path,
    *,
    accepted=None,
    rejected=None,
    conflicts=None,
    warnings=None,
    diagnostics=None,
    sl_sha="a" * 64,
    cf_sha="b" * 64,
    critical_failed=False,
    contract_version=fingerprints.CONTRACT_VERSION,
    summary_overrides=None,
    omit=(),
):
    root.mkdir(parents=True, exist_ok=True)
    accepted = accepted or []
    rejected = rejected or []
    conflicts = conflicts or []
    warnings = warnings or []
    diagnostics = diagnostics or []

    _write_csv(root / "accepted.csv", ACCEPTED_FIELDS, accepted)
    _write_csv(root / "rejected.csv", ["mls_number", "issue_codes"], rejected)
    _write_csv(root / "conflicts.csv", ["mls_number", "code", "field", "detail"], conflicts)
    _write_csv(root / "warnings.csv", ["mls_number", "code", "field", "detail"], warnings)
    _write_csv(
        root / "extraction_diagnostics.csv",
        ["report_type", "page_number", "status"],
        diagnostics,
    )
    summary = {
        "batch_id": "test",
        "sources": {
            "single_line": {"file": "single.pdf", "sha256": sl_sha},
            "client_full": {"file": "full.pdf", "sha256": cf_sha},
        },
        "reconciliation": {
            "single_line_records": len(accepted),
            "client_full_records": len(accepted),
            "matched_both": len(accepted),
        },
        "accepted": len(accepted),
        "rejected": len(rejected),
        "conflicts": len(conflicts),
        "warnings": len(warnings),
        "sold_date_range": ["2026-05-19", "2026-05-19"],
        "pages": {"needs_ocr": sum(1 for d in diagnostics if d.get("status") == "needs_ocr")},
        "critical_reconciliation_failed": critical_failed,
    }
    # contract_version is required; None omits the key entirely (missing case).
    if contract_version is not None:
        summary["contract_version"] = contract_version
    if summary_overrides:
        summary.update(summary_overrides)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    for name in omit:
        (root / name).unlink()
    return root


class StorageTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db_path = self.tmp / "test.db"
        self.url = f"sqlite:///{self.db_path}"
        engine = repository.create_engine_from_url(self.url)
        models.create_all(engine)
        engine.dispose()

    def tearDown(self):
        self._tmp.cleanup()

    def out_dir(self, name="batch", **kw):
        return make_pipeline_output(self.tmp / name, **kw)

    def canonical_count(self, **filters):
        engine = repository.create_engine_from_url(self.url)
        try:
            with engine.connect() as conn:
                return len(dataset.query_canonical_sales(conn, **filters))
        finally:
            engine.dispose()


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------
class FingerprintTests(unittest.TestCase):
    def test_batch_fingerprint_deterministic(self):
        a = fingerprints.batch_fingerprint("s1", "c1", "p", "k")
        b = fingerprints.batch_fingerprint("s1", "c1", "p", "k")
        self.assertEqual(a, b)

    def test_different_source_hashes_differ(self):
        a = fingerprints.batch_fingerprint("s1", "c1", "p", "k")
        b = fingerprints.batch_fingerprint("s2", "c1", "p", "k")
        self.assertNotEqual(a, b)

    def test_plan_batch_fingerprint_stable(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(Path(d) / "b", accepted=[_accepted_row()])
            fp1 = batch_loader.build_load_plan(out).batch["batch_fingerprint"]
            fp2 = batch_loader.build_load_plan(out).batch["batch_fingerprint"]
            self.assertEqual(fp1, fp2)

    def test_record_fingerprint_changes_with_price(self):
        r1 = {"mls_number": "1", "sold_price": "100"}
        r2 = {"mls_number": "1", "sold_price": "200"}
        self.assertNotEqual(
            fingerprints.record_fingerprint(r1), fingerprints.record_fingerprint(r2)
        )


# ---------------------------------------------------------------------------
# Manifest / plan validation (dry-run, no DB)
# ---------------------------------------------------------------------------
class ManifestValidationTests(unittest.TestCase):
    def test_missing_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(Path(d) / "b", accepted=[_accepted_row()],
                                       omit=("summary.json",))
            with self.assertRaises(batch_loader.ManifestError):
                batch_loader.build_load_plan(out)

    def test_mismatched_summary_count_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(Path(d) / "b", accepted=[_accepted_row()],
                                       summary_overrides={"accepted": 5})
            plan = batch_loader.build_load_plan(out)
            self.assertFalse(plan.safe_to_write)
            self.assertTrue(any("accepted" in r for r in plan.refusals))

    def test_conflicts_block_staging(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(
                Path(d) / "b", accepted=[_accepted_row()],
                conflicts=[{"mls_number": "202600001", "code": "SOLD_DATE_CONFLICT",
                            "field": "sold_date", "detail": "mismatch"}],
            )
            plan = batch_loader.build_load_plan(out)
            self.assertFalse(plan.safe_to_write)
            self.assertTrue(any("conflict" in r for r in plan.refusals))

    def test_critical_reconciliation_failed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(Path(d) / "b", accepted=[_accepted_row()],
                                       critical_failed=True)
            plan = batch_loader.build_load_plan(out)
            self.assertFalse(plan.safe_to_write)

    def test_supported_contract_version_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(Path(d) / "b", accepted=[_accepted_row()],
                                       contract_version=fingerprints.CONTRACT_VERSION)
            plan = batch_loader.build_load_plan(out)
            self.assertTrue(plan.safe_to_write, plan.refusals)

    def test_missing_contract_version_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(Path(d) / "b", accepted=[_accepted_row()],
                                       contract_version=None)  # key omitted
            plan = batch_loader.build_load_plan(out)
            self.assertFalse(plan.safe_to_write)
            self.assertTrue(any("contract_version" in r for r in plan.refusals))

    def test_empty_contract_version_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(Path(d) / "b", accepted=[_accepted_row()],
                                       contract_version="   ")
            plan = batch_loader.build_load_plan(out)
            self.assertFalse(plan.safe_to_write)
            self.assertTrue(any("contract_version" in r for r in plan.refusals))

    def test_unsupported_contract_version_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(Path(d) / "b", accepted=[_accepted_row()],
                                       contract_version="bogus/9.9")
            plan = batch_loader.build_load_plan(out)
            self.assertFalse(plan.safe_to_write)
            self.assertTrue(any("unsupported contract_version" in r for r in plan.refusals))

    def test_invalid_sold_date_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(
                Path(d) / "b",
                accepted=[_accepted_row(sold_date="not-a-date")],
            )
            plan = batch_loader.build_load_plan(out)
            self.assertFalse(plan.safe_to_write)
            self.assertTrue(any("sold_date" in r for r in plan.refusals))

    def test_duplicate_mls_within_batch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out = make_pipeline_output(
                Path(d) / "b",
                accepted=[_accepted_row(mls_number="202600001"),
                          _accepted_row(mls_number="202600001", address="9 Other St")],
            )
            plan = batch_loader.build_load_plan(out)
            self.assertFalse(plan.safe_to_write)
            self.assertTrue(any("duplicate MLS" in r for r in plan.refusals))


# ---------------------------------------------------------------------------
# Loader authorization + dry-run safety
# ---------------------------------------------------------------------------
class LoaderAuthTests(StorageTestBase):
    def test_dry_run_performs_no_writes(self):
        out = self.out_dir(accepted=[_accepted_row()])
        rc = load_wrreb_batch.run(str(out), database_url=self.url, dry_run=True)
        self.assertEqual(rc, 0)
        engine = repository.create_engine_from_url(self.url)
        with engine.connect() as conn:
            n = conn.execute(sa.select(sa.func.count()).select_from(models.staging_sales)).scalar_one()
        engine.dispose()
        self.assertEqual(n, 0)

    def test_staging_authorization_required(self):
        out = self.out_dir(accepted=[_accepted_row()])
        # --no-dry-run but WITHOUT the authorize flag -> no write.
        rc = load_wrreb_batch.run(str(out), database_url=self.url,
                                  authorize_staging_write=False, dry_run=False)
        self.assertEqual(rc, 0)
        self.assertEqual(self.canonical_count(), 0)
        engine = repository.create_engine_from_url(self.url)
        with engine.connect() as conn:
            n = conn.execute(sa.select(sa.func.count()).select_from(models.staging_sales)).scalar_one()
        engine.dispose()
        self.assertEqual(n, 0)

    def test_missing_database_url_rejected(self):
        out = self.out_dir(accepted=[_accepted_row()])
        rc = load_wrreb_batch.run(str(out), database_url=None,
                                  authorize_staging_write=True, dry_run=False)
        self.assertEqual(rc, 2)

    def test_empty_database_url_rejected(self):
        with self.assertRaises(repository.InvalidDatabaseUrl):
            repository.assert_explicit_database_url("   ")


# ---------------------------------------------------------------------------
# Staging write path
# ---------------------------------------------------------------------------
class StagingWriteTests(StorageTestBase):
    def _load(self, out):
        return load_wrreb_batch.run(str(out), database_url=self.url,
                                    authorize_staging_write=True, dry_run=False)

    def test_valid_staging_load(self):
        out = self.out_dir(accepted=[_accepted_row()])
        self.assertEqual(self._load(out), 0)
        engine = repository.create_engine_from_url(self.url)
        with engine.connect() as conn:
            rows = list(conn.execute(sa.select(models.staging_sales)))
            batch = conn.execute(sa.select(models.import_batches)).first()
        engine.dispose()
        self.assertEqual(len(rows), 1)
        self.assertEqual(batch.status, "staged")
        self.assertEqual(batch.accepted_count, 1)

    def test_authorize_flag_writes_without_no_dry_run(self):
        # Regression: --authorize-staging-write + URL must write even though
        # --dry-run defaults True (no --no-dry-run required).
        out = self.out_dir(accepted=[_accepted_row()])
        rc = load_wrreb_batch.run(str(out), database_url=self.url,
                                  authorize_staging_write=True)  # dry_run defaults True
        self.assertEqual(rc, 0)
        engine = repository.create_engine_from_url(self.url)
        with engine.connect() as conn:
            n = conn.execute(sa.select(sa.func.count()).select_from(models.staging_sales)).scalar_one()
        engine.dispose()
        self.assertEqual(n, 1)

    def test_duplicate_batch_rejected(self):
        out = self.out_dir(accepted=[_accepted_row()])
        self.assertEqual(self._load(out), 0)
        # Same directory again -> same fingerprint -> refused.
        rc = self._load(out)
        self.assertEqual(rc, 1)
        engine = repository.create_engine_from_url(self.url)
        with engine.connect() as conn:
            n = conn.execute(sa.select(sa.func.count()).select_from(models.import_batches)).scalar_one()
        engine.dispose()
        self.assertEqual(n, 1)

    def test_null_optional_fields_preserved(self):
        out = self.out_dir(accepted=[_accepted_row(
            neighbourhood="", year_built="", basement_type="", gross_tax="")])
        self.assertEqual(self._load(out), 0)
        engine = repository.create_engine_from_url(self.url)
        with engine.connect() as conn:
            row = conn.execute(sa.select(models.staging_sales)).first()
        engine.dispose()
        self.assertIsNone(row.neighbourhood)
        self.assertIsNone(row.year_built)
        self.assertIsNone(row.basement_type)
        self.assertIsNone(row.gross_tax)

    def test_prices_preserved_accurately(self):
        out = self.out_dir(accepted=[_accepted_row(sold_price="482500", list_price="500000.50")])
        self.assertEqual(self._load(out), 0)
        engine = repository.create_engine_from_url(self.url)
        with engine.connect() as conn:
            row = conn.execute(sa.select(models.staging_sales)).first()
        engine.dispose()
        self.assertEqual(float(row.sold_price), 482500.0)
        self.assertEqual(float(row.list_price), 500000.50)

    def test_unique_mls_within_batch_enforced(self):
        # Repository-level guard (loader refuses earlier; this proves the guard).
        engine = repository.create_engine_from_url(self.url)
        try:
            with engine.begin() as conn:
                bid = repository.insert_batch(conn, _min_batch("f1"))
                with self.assertRaises(repository.DuplicateMls):
                    repository.insert_staging_rows(conn, bid, [
                        {"mls_number": "X", "record_fingerprint": "a"},
                        {"mls_number": "X", "record_fingerprint": "b"},
                    ])
        finally:
            engine.dispose()

    def test_transaction_rollback_on_failure(self):
        engine = repository.create_engine_from_url(self.url)
        try:
            with self.assertRaises(RuntimeError):
                with engine.begin() as conn:
                    repository.insert_batch(conn, _min_batch("f-rollback"))
                    raise RuntimeError("boom after insert")
            with engine.connect() as conn:
                n = conn.execute(sa.select(sa.func.count()).select_from(models.import_batches)).scalar_one()
            self.assertEqual(n, 0)  # rolled back
        finally:
            engine.dispose()


# ---------------------------------------------------------------------------
# Privacy guard
# ---------------------------------------------------------------------------
class PrivacyTests(unittest.TestCase):
    def test_private_field_rejected_from_staging_object(self):
        with self.assertRaises(models.PrivacyViolation):
            models.build_staging_row({
                "mls_number": "1", "record_fingerprint": "f",
                "agent_name": "Jane Realtor",
            })

    def test_private_field_rejected_from_canonical_object(self):
        for banned in ("remarks", "lockbox", "phone_number", "showing_instructions"):
            with self.assertRaises(models.PrivacyViolation):
                models.build_canonical_row({
                    "mls_number": "1", "record_fingerprint": "f", banned: "secret",
                })


# ---------------------------------------------------------------------------
# Promotion + canonical query
# ---------------------------------------------------------------------------
def _min_batch(fp, **over):
    base = {
        "batch_fingerprint": fp,
        "contract_version": fingerprints.CONTRACT_VERSION,
        "parser_version": "p/1.0",
        "single_line_filename": "single.pdf", "single_line_sha256": "s",
        "client_full_filename": "full.pdf", "client_full_sha256": "c",
        "status": "parsed", "accepted_count": 0,
    }
    base.update(over)
    return base


class PromotionTests(StorageTestBase):
    def _stage_clean(self, out_name, sl_sha, accepted):
        out = self.out_dir(out_name, accepted=accepted, sl_sha=sl_sha)
        rc = load_wrreb_batch.run(str(out), database_url=self.url,
                                  authorize_staging_write=True, dry_run=False)
        self.assertEqual(rc, 0)
        engine = repository.create_engine_from_url(self.url)
        with engine.connect() as conn:
            b = conn.execute(sa.select(models.import_batches)
                             .where(models.import_batches.c.single_line_sha256 == sl_sha)).first()
        engine.dispose()
        self.assertEqual(b.status, "staged")
        return b.id

    def _promote(self, batch_id):
        engine = repository.create_engine_from_url(self.url)
        try:
            with engine.begin() as conn:
                return repository.promote_batch(conn, batch_id, approved_by="tester")
        finally:
            engine.dispose()

    def test_different_mls_same_linc_preserved(self):
        bid = self._stage_clean("b1", "s1" + "0" * 62, [
            _accepted_row(mls_number="202600001", linc_number="012R021018000"),
            _accepted_row(mls_number="202600002", linc_number="012R021018000",
                          address="55 Elm St"),
        ])
        res = self._promote(bid)
        self.assertEqual(res["inserted"], 2)
        self.assertEqual(self.canonical_count(), 2)

    def test_same_mls_same_fingerprint_idempotent(self):
        row = _accepted_row(mls_number="202600009")
        bid1 = self._stage_clean("b1", "s1" + "0" * 62, [dict(row)])
        self.assertEqual(self._promote(bid1)["inserted"], 1)
        # Second batch, identical critical values -> same record fingerprint.
        bid2 = self._stage_clean("b2", "s2" + "0" * 62, [dict(row)])
        res = self._promote(bid2)
        self.assertEqual(res["inserted"], 0)
        self.assertEqual(res["idempotent_skipped"], 1)
        self.assertEqual(self.canonical_count(), 1)

    def test_same_mls_conflicting_fingerprint_blocked(self):
        bid1 = self._stage_clean("b1", "s1" + "0" * 62,
                                 [_accepted_row(mls_number="202600009", sold_price="482500")])
        self.assertEqual(self._promote(bid1)["inserted"], 1)
        bid2 = self._stage_clean("b2", "s2" + "0" * 62,
                                 [_accepted_row(mls_number="202600009", sold_price="999999")])
        with self.assertRaises(repository.PromotionBlocked):
            self._promote(bid2)
        self.assertEqual(self.canonical_count(), 1)  # unchanged

    def test_canonical_query_returns_approved_only(self):
        bid = self._stage_clean("b1", "s1" + "0" * 62, [_accepted_row(mls_number="202600001")])
        self._promote(bid)
        # Insert a flagged row directly.
        engine = repository.create_engine_from_url(self.url)
        with engine.begin() as conn:
            conn.execute(sa.insert(models.canonical_sales).values(
                mls_number="FLAGGED1", record_fingerprint="z",
                data_quality_status="flagged", sold_date=None,
            ))
        engine.dispose()
        self.assertEqual(self.canonical_count(), 1)  # approved only by default
        self.assertEqual(self.canonical_count(min_data_quality="flagged"), 2)


if __name__ == "__main__":
    unittest.main()
