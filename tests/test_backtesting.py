"""Synthetic tests for the read-only comparable-sales backtest.

No database, no model, no scoring-weight changes. Uses the in-memory data source
and the unmodified production comparable engine.
"""
from __future__ import annotations

import json
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from samvision.backtesting import metrics, models, reporting
from samvision.backtesting.models import BacktestCase
from samvision.backtesting.runner import BacktestRunner
from samvision.comparables.datasource import InMemoryDataSource
from samvision.comparables.service import ComparableService


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_row(mls, sold_date, price, sqft=1000, *, ptype="RD", area="A",
             nb="N", status="approved", **opts):
    row = {
        "mls_number": mls,
        "sold_date": sold_date,
        "sold_price": price,
        "living_area_sqft": sqft,
        "area_code": area,
        "neighbourhood": nb,
        "property_type": ptype,
        "style": opts.get("style", "bungalow"),
        "year_built": opts.get("year_built", 1975),
        "bedrooms_total": opts.get("bedrooms_total", 3),
        "full_bathrooms": opts.get("full_bathrooms", 2),
        "half_bathrooms": opts.get("half_bathrooms", 0),
        "basement_type": opts.get("basement_type", "full"),
        "garage_type": opts.get("garage_type"),  # absent like real data
        "data_quality_status": status,
    }
    for k in ("mls_number", "sold_date", "sold_price", "living_area_sqft",
              "area_code", "neighbourhood", "property_type"):
        if k in opts:
            row[k] = opts[k]
    row.update({k: v for k, v in opts.items() if k in row or k in (
        "address", "linc_number", "postal_code", "normalized_property_id")})
    return row


def pool_rows():
    """Six approved RD sales in the same neighbourhood, increasing dates/prices."""
    return [
        make_row("202600001", date(2026, 3, 1), 300000),
        make_row("202600002", date(2026, 3, 10), 305000),
        make_row("202600003", date(2026, 3, 20), 310000),
        make_row("202600004", date(2026, 4, 1), 315000),
        make_row("202600005", date(2026, 4, 10), 320000),
        make_row("202600006", date(2026, 4, 20), 325000),
    ]


def make_case(indicated, actual, **kw):
    ind, act = Decimal(str(indicated)), Decimal(str(actual))
    return BacktestCase(
        mls_number=kw.get("mls", "202600999"),
        area_code=kw.get("area_code", "A"),
        neighbourhood=kw.get("neighbourhood", "N"),
        property_type=kw.get("property_type", "RD"),
        sold_date=kw.get("sold_date", date(2026, 5, 1)),
        actual_sold_price=act,
        indicated_value=ind,
        absolute_error=metrics.absolute_error(ind, act),
        percentage_error=metrics.percentage_error(ind, act),
        confidence=kw.get("confidence", "medium"),
        tier=kw.get("tier", "tier1_neighbourhood_365"),
        tier_widened=kw.get("tier_widened", False),
        selected_comparable_count=kw.get("n", 5),
        median_similarity=kw.get("median_similarity", 70.0),
        price_spread=kw.get("price_spread", 0.1),
        mean_coverage=kw.get("mean_coverage", 0.83),
    )


# --------------------------------------------------------------------------
# chronology & leakage (engine-level, exactly as the backtest drives it)
# --------------------------------------------------------------------------
class ChronologyLeakageTests(unittest.TestCase):
    def _service(self, rows):
        return ComparableService(InMemoryDataSource(rows))

    def _subject_from(self, row):
        from samvision.comparables.models import SubjectProperty
        m = dict(row)
        m["valuation_date"] = row["sold_date"]
        return SubjectProperty.from_mapping(m)

    def test_future_sale_excluded(self):
        rows = pool_rows()
        subject = self._subject_from(rows[3])  # 2026-04-01
        result = self._service(rows).find_comparables(subject)
        for c in result.comparables:
            self.assertLessEqual(c.candidate.sold_date, subject.valuation_date)

    def test_held_out_mls_excluded(self):
        rows = pool_rows()
        subject = self._subject_from(rows[5])  # latest, 5 priors
        result = self._service(rows).find_comparables(subject)
        self.assertNotIn("202600006", [c.candidate.mls_number for c in result.comparables])

    def test_same_valuation_date_comp_included(self):
        # documented inclusive bound: a different sale on the same date is eligible
        rows = pool_rows()
        rows.append(make_row("202600050", date(2026, 4, 20), 322000))  # same date as row6
        subject = self._subject_from(rows[5])  # 2026-04-20
        result = self._service(rows).find_comparables(subject)
        self.assertIn("202600050", [c.candidate.mls_number for c in result.comparables])

    def test_no_future_even_if_present(self):
        rows = pool_rows()
        subject = self._subject_from(rows[2])  # 2026-03-20 -> only 2 priors, insufficient
        result = self._service(rows).find_comparables(subject)
        self.assertEqual(result.status, "insufficient_data")
        for c in result.comparables:
            self.assertLessEqual(c.candidate.sold_date, subject.valuation_date)

    def test_db_query_upper_bound_enforced(self):
        ds = InMemoryDataSource(pool_rows())
        out = ds.fetch(property_type="RD", sold_start=date(2026, 1, 1),
                       sold_end=date(2026, 3, 15), neighbourhood="N")
        for r in out:
            self.assertLessEqual(r["sold_date"], date(2026, 3, 15))
        self.assertNotIn("202600004", [r["mls_number"] for r in out])


# --------------------------------------------------------------------------
# subject eligibility
# --------------------------------------------------------------------------
class EligibilityTests(unittest.TestCase):
    def _run(self, subject_rows, pool=None):
        pool = pool if pool is not None else pool_rows()
        service = ComparableService(InMemoryDataSource(pool))
        return BacktestRunner(service).run(subject_rows)

    def test_valid_subject_included(self):
        rep = self._run([pool_rows()[5]])
        self.assertEqual(rep.valued, 1)
        self.assertEqual(rep.skipped_ineligible, 0)

    def test_missing_mls_skipped(self):
        row = make_row(None, date(2026, 5, 1), 300000)
        rep = self._run([row])
        self.assertEqual(rep.skip_reason_counts.get(models.MISSING_MLS), 1)

    def test_missing_sold_date_skipped(self):
        row = make_row("202600100", None, 300000)
        rep = self._run([row])
        self.assertEqual(rep.skip_reason_counts.get(models.MISSING_SOLD_DATE), 1)

    def test_invalid_sold_price_skipped(self):
        rep = self._run([make_row("202600101", date(2026, 5, 1), 0)])
        self.assertEqual(rep.skip_reason_counts.get(models.INVALID_SOLD_PRICE), 1)

    def test_invalid_living_area_skipped(self):
        rep = self._run([make_row("202600102", date(2026, 5, 1), 300000, sqft=0)])
        self.assertEqual(rep.skip_reason_counts.get(models.INVALID_LIVING_AREA), 1)

    def test_missing_property_type_skipped(self):
        rep = self._run([make_row("202600103", date(2026, 5, 1), 300000, ptype=None)])
        self.assertEqual(rep.skip_reason_counts.get(models.MISSING_PROPERTY_TYPE), 1)

    def test_missing_geography_skipped(self):
        rep = self._run([make_row("202600104", date(2026, 5, 1), 300000, area=None, nb=None)])
        self.assertEqual(rep.skip_reason_counts.get(models.MISSING_GEOGRAPHY), 1)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
class MetricsTests(unittest.TestCase):
    def test_absolute_error(self):
        self.assertEqual(metrics.absolute_error(Decimal("110"), Decimal("100")), Decimal("10.00"))

    def test_percentage_error(self):
        self.assertEqual(metrics.percentage_error(Decimal("110"), Decimal("100")), Decimal("0.100000"))

    def test_zero_actual_rejected(self):
        with self.assertRaises(ValueError):
            metrics.percentage_error(Decimal("100"), Decimal("0"))
        with self.assertRaises(ValueError):
            metrics.percentage_error(Decimal("100"), Decimal("-5"))

    def test_decimal_rounding_deterministic(self):
        self.assertEqual(metrics.absolute_error(Decimal("100.005"), Decimal("100")), Decimal("0.01"))

    def _cases(self):
        return [
            make_case(110000, 100000),
            make_case(90000, 100000),
            make_case(105000, 100000),
            make_case(95000, 100000),
        ]

    def test_mae_and_median(self):
        o = metrics.overall_metrics(self._cases())
        self.assertEqual(o["mae"], "7500.00")
        self.assertEqual(o["median_absolute_error"], "7500.00")

    def test_rmse(self):
        o = metrics.overall_metrics(self._cases())
        self.assertEqual(o["rmse"], "7905.69")

    def test_mape_and_median_ape(self):
        o = metrics.overall_metrics(self._cases())
        self.assertEqual(o["mape_pct"], 7.5)
        self.assertEqual(o["median_ape_pct"], 7.5)

    def test_within_thresholds(self):
        t = metrics.threshold_table(self._cases())
        self.assertEqual(t["within_5pct"], 50.0)
        self.assertEqual(t["within_10pct"], 100.0)
        self.assertEqual(t["within_15pct"], 100.0)
        self.assertEqual(t["within_20pct"], 100.0)

    def test_percentile_linear(self):
        vals = [Decimal("5"), Decimal("5"), Decimal("10"), Decimal("10")]
        self.assertEqual(metrics.percentile(vals, Decimal("0.5")), Decimal("7.50"))
        self.assertEqual(metrics.percentile(vals, Decimal("0.25")), Decimal("5.00"))
        self.assertEqual(metrics.percentile(vals, Decimal("0.90")), Decimal("10.00"))
        self.assertIsNone(metrics.percentile([], Decimal("0.5")))


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
class _RaisingService:
    def find_comparables(self, subject):
        raise RuntimeError("boom")


class RunnerTests(unittest.TestCase):
    def _service(self, pool=None):
        return ComparableService(InMemoryDataSource(pool if pool is not None else pool_rows()))

    def test_successful_valuation_recorded(self):
        rep = BacktestRunner(self._service()).run([pool_rows()[5]])
        self.assertEqual(rep.valued, 1)
        self.assertEqual(len(rep.cases), 1)
        self.assertIsNotNone(rep.cases[0].indicated_value)

    def test_insufficient_data_recorded(self):
        # earliest subject has no prior sales -> insufficient
        rep = BacktestRunner(self._service()).run([pool_rows()[0]])
        self.assertEqual(rep.valued, 0)
        self.assertEqual(rep.insufficient, 1)
        self.assertEqual(rep.coverage_rate, 0.0)

    def test_engine_exception_captured(self):
        rep = BacktestRunner(_RaisingService()).run([pool_rows()[5]])
        self.assertEqual(rep.engine_errors, 1)
        self.assertEqual(rep.valued, 0)

    def test_no_database_write_methods(self):
        ds = InMemoryDataSource(pool_rows())
        for banned in ("insert", "update", "delete", "commit", "execute"):
            self.assertFalse(hasattr(ds, banned))

    def test_no_write_sql_in_backtest_source(self):
        pkg = Path(__file__).resolve().parents[1] / "samvision" / "backtesting"
        for py in pkg.glob("*.py"):
            text = py.read_text().upper()
            for kw in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "CREATE TABLE"):
                self.assertNotIn(kw, text, f"{py.name} contains write SQL {kw!r}")

    def test_skip_reason_counts(self):
        subjects = [
            make_row(None, date(2026, 5, 1), 300000),                 # MISSING_MLS
            make_row("202600200", None, 300000),                      # MISSING_SOLD_DATE
            make_row("202600201", date(2026, 5, 1), 0),               # INVALID_SOLD_PRICE
            pool_rows()[5],                                           # valued
        ]
        rep = BacktestRunner(self._service()).run(subjects)
        self.assertEqual(rep.skip_reason_counts.get(models.MISSING_MLS), 1)
        self.assertEqual(rep.skip_reason_counts.get(models.MISSING_SOLD_DATE), 1)
        self.assertEqual(rep.skip_reason_counts.get(models.INVALID_SOLD_PRICE), 1)
        self.assertEqual(rep.skipped_ineligible, 3)

    def test_coverage_correct(self):
        # subjects: 1 valued (latest), 1 insufficient (earliest) => eligible 2, coverage 0.5
        rep = BacktestRunner(self._service()).run([pool_rows()[0], pool_rows()[5]])
        self.assertEqual(rep.eligible, 2)
        self.assertEqual(rep.valued, 1)
        self.assertEqual(rep.coverage_rate, 0.5)

    def test_grouped_metrics_and_unstable_flag(self):
        cases = [make_case(110000, 100000, neighbourhood="Big") for _ in range(4)]
        cases += [make_case(120000, 100000, neighbourhood="Small")]
        groups = metrics.grouped_metrics(cases, lambda c: c.neighbourhood, min_group=3)
        by = {g.group_key: g for g in groups}
        self.assertEqual(by["Big"].n, 4)
        self.assertTrue(by["Big"].stable)
        self.assertEqual(by["Small"].n, 1)
        self.assertFalse(by["Small"].stable)

    def test_grouped_stable_ordering(self):
        cases = ([make_case(110000, 100000, neighbourhood="B") for _ in range(2)]
                 + [make_case(110000, 100000, neighbourhood="A") for _ in range(5)])
        groups = metrics.grouped_metrics(cases, lambda c: c.neighbourhood)
        self.assertEqual([g.group_key for g in groups], ["A", "B"])  # -n then key


# --------------------------------------------------------------------------
# privacy
# --------------------------------------------------------------------------
class PrivacyTests(unittest.TestCase):
    def _report_with_address(self):
        pool = pool_rows()
        subj = make_row("202600777", date(2026, 5, 1), 330000,
                        address="123 Secret St", linc_number="LINC-XYZ",
                        postal_code="R3M0A1")
        service = ComparableService(InMemoryDataSource(pool + [subj]))
        return BacktestRunner(service).run([subj])

    def test_address_omitted_from_case_output(self):
        rep = self._report_with_address()
        self.assertEqual(rep.valued, 1)
        d = rep.cases[0].sanitized_dict()
        self.assertNotIn("address", d)
        self.assertNotIn("123 Secret St", json.dumps(d))

    def test_mls_masked_in_failure_report(self):
        rep = self._report_with_address()
        fcs = reporting.failure_cases(rep)
        self.assertTrue(fcs)
        self.assertEqual(fcs[0]["mls_masked"], "******777")
        self.assertTrue(fcs[0]["mls_masked"].endswith("777"))
        self.assertNotIn("202600777", json.dumps(fcs))

    def test_banned_fields_not_serialized(self):
        rep = self._report_with_address()
        blob = json.dumps(rep.summary_dict()) + json.dumps([c.sanitized_dict() for c in rep.cases])
        for banned in ("address", "linc_number", "postal_code", "LINC-XYZ", "R3M0A1"):
            self.assertNotIn(banned, blob)

    def test_written_reports_have_no_credentials(self):
        rep = self._report_with_address()
        with TemporaryDirectory() as td:
            written = reporting.write_reports(rep, td)
            blob = "".join(Path(p).read_text() for p in written)
        for secret in ("password", "OCBz", "123 Secret St", "LINC-XYZ"):
            self.assertNotIn(secret, blob)


# --------------------------------------------------------------------------
# output-dir guard
# --------------------------------------------------------------------------
class OutputGuardTests(unittest.TestCase):
    def test_refuses_tracked_source_dir(self):
        repo = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            reporting.guard_output_dir(str(repo / "samvision" / "reports"))
        with self.assertRaises(ValueError):
            reporting.guard_output_dir(str(repo / "tests" / "out"))

    def test_allows_tmp(self):
        with TemporaryDirectory() as td:
            self.assertEqual(reporting.guard_output_dir(td), Path(td).resolve())


if __name__ == "__main__":
    unittest.main()
