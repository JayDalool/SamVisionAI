"""Tests for the comparable-sales engine (samvision/comparables).

Synthetic data only — no real WRREB listings. Everything runs in memory; no
PostgreSQL, no trained model, no database writes.
"""
from __future__ import annotations

import unittest
from datetime import date
from decimal import Decimal

from samvision.comparables import constants, scoring, statistics
from samvision.comparables.datasource import InMemoryDataSource
from samvision.comparables.models import (
    CandidateSale, SubjectProperty, SubjectValidationError,
)
from samvision.comparables.service import ComparableService
from samvision.storage.models import PRIVACY_BANNED_FIELDS


def crow(mls, sold, *, price=400000, sqft=1100, hood="A", area="X", ptype="RES",
         status="approved", **extra):
    d = dict(
        mls_number=mls, sold_date=sold, sold_price=Decimal(price), living_area_sqft=sqft,
        neighbourhood=hood, area_code=area, property_type=ptype, data_quality_status=status,
        bedrooms_total=3, full_bathrooms=1, half_bathrooms=0, year_built=1975,
        style="BUNG", garage_type="Dbl", basement_type="Full",
    )
    d.update(extra)
    return d


def subj(valuation="2026-06-01", *, hood="A", area="X", sqft=1100, ptype="RES", **extra):
    kw = dict(
        valuation_date=date.fromisoformat(valuation), property_type=ptype, living_area_sqft=sqft,
        area_code=area, neighbourhood=hood, bedrooms_total=3, full_bathrooms=1, year_built=1975,
        style="BUNG", garage_type="Dbl", basement_type="Full",
    )
    kw.update(extra)
    return SubjectProperty(**kw)


def cand(**over):
    base = dict(
        mls_number="202600001", sold_date=date(2026, 5, 1), sold_price=Decimal(400000),
        living_area_sqft=1100, area_code="X", neighbourhood="A", property_type="RES",
        style="BUNG", year_built=1975, bedrooms_total=3, full_bathrooms=1, half_bathrooms=0,
        basement_type="Full", garage_type="Dbl", data_quality_status="approved",
    )
    base.update(over)
    return CandidateSale.from_mapping(base)


# ---------------------------------------------------------------------------
class SubjectValidationTests(unittest.TestCase):
    def test_valid_minimum_subject(self):
        s = SubjectProperty(valuation_date=date(2026, 6, 1), property_type="RES",
                            living_area_sqft=1000, area_code="X")
        self.assertIs(s.validate(), s)

    def test_missing_geography_rejected(self):
        s = SubjectProperty(valuation_date=date(2026, 6, 1), property_type="RES",
                            living_area_sqft=1000)
        with self.assertRaises(SubjectValidationError):
            s.validate()

    def test_non_positive_sqft_rejected(self):
        with self.assertRaises(SubjectValidationError):
            subj(sqft=0).validate()
        with self.assertRaises(SubjectValidationError):
            subj(sqft=-5).validate()

    def test_future_year_built_rejected(self):
        with self.assertRaises(SubjectValidationError):
            subj(year_built=2030).validate()  # after valuation year 2026

    def test_negative_counts_rejected(self):
        with self.assertRaises(SubjectValidationError):
            subj(bedrooms_total=-1).validate()
        with self.assertRaises(SubjectValidationError):
            subj(full_bathrooms=-2).validate()

    def test_valuation_date_never_defaults_to_today(self):
        with self.assertRaises(SubjectValidationError):
            SubjectProperty.from_mapping({"property_type": "RES", "living_area_sqft": 1000,
                                          "area_code": "X"})


# ---------------------------------------------------------------------------
class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.subject = subj()

    def test_future_sale_excluded(self):
        ok, reason = scoring.eligibility(cand(sold_date=date(2026, 7, 1)), self.subject)
        self.assertFalse(ok)
        self.assertEqual(reason, "future_sale")

    def test_sale_on_valuation_date_included(self):
        ok, _ = scoring.eligibility(cand(sold_date=date(2026, 6, 1)), self.subject)
        self.assertTrue(ok)

    def test_unapproved_record_excluded(self):
        ok, reason = scoring.eligibility(cand(data_quality_status="flagged"), self.subject)
        self.assertFalse(ok)
        self.assertEqual(reason, "not_approved")

    def test_subject_mls_excluded(self):
        s = subj(mls_number="202600001")
        ok, reason = scoring.eligibility(cand(mls_number="202600001"), s)
        self.assertFalse(ok)
        self.assertEqual(reason, "subject_self")

    def test_wrong_property_type_excluded(self):
        ok, reason = scoring.eligibility(cand(property_type="CONDO"), self.subject)
        self.assertFalse(ok)
        self.assertEqual(reason, "property_type_mismatch")

    def test_outside_lookback_excluded(self):
        ok, reason = scoring.eligibility(cand(sold_date=date(2023, 1, 1)), self.subject)
        self.assertFalse(ok)
        self.assertEqual(reason, "outside_lookback")


# ---------------------------------------------------------------------------
class SearchTierTests(unittest.TestCase):
    def _service(self, rows, **kw):
        return ComparableService(InMemoryDataSource(rows), **kw)

    def test_exact_neighbourhood_selected_first(self):
        rows = [crow(f"2026000{i:02d}", f"2026-04-{i:02d}", hood="A", area="X") for i in range(1, 6)]
        res = self._service(rows).find_comparables(subj())
        self.assertEqual(res.tier, constants.SEARCH_TIERS[0].name)
        self.assertFalse(res.tier_widened)

    def test_area_widening_only_when_necessary(self):
        rows = [crow("20260001", "2026-04-01", hood="A", area="X"),
                crow("20260002", "2026-04-02", hood="A", area="X")]
        rows += [crow(f"2026010{i}", f"2026-04-1{i}", hood="B", area="X") for i in range(1, 4)]
        res = self._service(rows).find_comparables(subj())
        self.assertEqual(res.tier, constants.SEARCH_TIERS[1].name)
        self.assertTrue(res.tier_widened)
        self.assertIn("geographic_or_temporal_widening", res.warnings)

    def test_730_widening_only_when_necessary(self):
        # all sold ~500 days before valuation -> outside 365 tiers, inside 730
        rows = [crow(f"2026000{i}", f"2025-01-0{i}", hood="A", area="X") for i in range(1, 5)]
        res = self._service(rows).find_comparables(subj())
        self.assertEqual(res.tier, constants.SEARCH_TIERS[2].name)
        self.assertTrue(res.tier_widened)

    def test_no_duplicate_mls_and_unique(self):
        rows = [crow("20260001", "2026-04-01", hood="A", area="X"),
                crow("20260002", "2026-04-02", hood="A", area="X")]
        rows += [crow(f"2026010{i}", f"2026-04-1{i}", hood="B", area="X") for i in range(1, 4)]
        res = self._service(rows).find_comparables(subj())
        mls = [c.candidate.mls_number for c in res.comparables]
        self.assertEqual(len(mls), len(set(mls)))

    def test_insufficient_data_status(self):
        rows = [crow("20260001", "2026-04-01"), crow("20260002", "2026-04-02")]
        res = self._service(rows).find_comparables(subj())
        self.assertEqual(res.status, "insufficient_data")
        self.assertIsNone(res.indicated_value)
        self.assertEqual(res.confidence, "insufficient")
        self.assertIn("insufficient_data", res.warnings)


# ---------------------------------------------------------------------------
class ScoringTests(unittest.TestCase):
    def test_identical_scores_highest(self):
        s = subj(valuation="2026-05-01")
        identical = scoring.score(cand(sold_date=date(2026, 5, 1)), s, "t")
        different = scoring.score(cand(sold_date=date(2026, 5, 1), living_area_sqft=1600,
                                       bedrooms_total=6, style="TWO"), s, "t")
        self.assertGreater(identical.similarity, different.similarity)
        self.assertAlmostEqual(identical.similarity, 100.0, delta=0.01)

    def test_larger_size_difference_lowers_score(self):
        s = subj()
        near = scoring.score(cand(living_area_sqft=1100), s, "t")
        far = scoring.score(cand(living_area_sqft=1600), s, "t")
        self.assertGreater(near.components["living_area"], far.components["living_area"])

    def test_older_sale_lowers_recency(self):
        s = subj(valuation="2026-06-01")
        recent = scoring.score(cand(sold_date=date(2026, 5, 15)), s, "t")
        old = scoring.score(cand(sold_date=date(2025, 7, 1)), s, "t")
        self.assertGreater(recent.components["recency"], old.components["recency"])

    def test_exact_beds_baths_score_higher(self):
        s = subj(bedrooms_total=3, full_bathrooms=1)
        exact = scoring.score(cand(bedrooms_total=3, full_bathrooms=1), s, "t")
        off = scoring.score(cand(bedrooms_total=5, full_bathrooms=3), s, "t")
        self.assertGreater(exact.components["bedrooms"], off.components["bedrooms"])
        self.assertGreater(exact.components["bathrooms"], off.components["bathrooms"])

    def test_age_uses_historical_dates_not_current_year(self):
        s = subj(valuation="2026-06-01", year_built=2000)  # subject age 26
        same_age = cand(sold_date=date(2020, 1, 1), year_built=1994)  # age at sold = 26
        diff_age = cand(sold_date=date(2020, 1, 1), year_built=2000)  # age at sold = 20
        self.assertEqual(scoring.year_built_subscore(same_age, s), 1.0)
        self.assertLess(scoring.year_built_subscore(diff_age, s), 1.0)

    def test_unknown_optional_no_exact_credit(self):
        s = subj(style="BUNG")
        self.assertEqual(scoring.score(cand(style=None), s, "t").components["style"], 0.0)

    def test_sparse_record_lower_coverage(self):
        full = cand()
        sparse = cand(bedrooms_total=None, full_bathrooms=None, half_bathrooms=None,
                      year_built=None, style=None, garage_type=None, basement_type=None)
        self.assertEqual(scoring.data_coverage(full), 1.0)
        self.assertEqual(scoring.data_coverage(sparse), 0.0)

    def test_stable_tie_breaking(self):
        s = subj()
        # Two identical-similarity comps; expect sold_date desc then mls asc.
        a = scoring.score(cand(mls_number="20260002", sold_date=date(2026, 4, 1)), s, "t")
        b = scoring.score(cand(mls_number="20260001", sold_date=date(2026, 5, 1)), s, "t")
        ranked = ComparableService._rank([a, b])
        self.assertEqual([c.candidate.mls_number for c in ranked], ["20260001", "20260002"])


# ---------------------------------------------------------------------------
class StatisticsTests(unittest.TestCase):
    def test_ppsf_decimal_safe(self):
        self.assertEqual(statistics.price_per_sqft(Decimal(440000), 1100), Decimal(400))

    def test_zero_sqft_rejected(self):
        with self.assertRaises(statistics.StatisticsError):
            statistics.price_per_sqft(Decimal(400000), 0)

    def test_weighted_median(self):
        pairs = [(Decimal(100), Decimal(1)), (Decimal(200), Decimal(3))]
        self.assertEqual(statistics.weighted_median(pairs), Decimal(200))

    def test_weighted_quantiles(self):
        pairs = [(Decimal(v), Decimal(1)) for v in (100, 200, 300, 400)]
        self.assertEqual(statistics.weighted_quantile(pairs, 0.25), Decimal(100))
        self.assertEqual(statistics.weighted_quantile(pairs, 0.75), Decimal(300))

    def test_size_normalized_price(self):
        v = statistics.size_normalized_price(Decimal(400000), 1200, 1100)
        self.assertEqual(v, Decimal(400000) * Decimal(1200) / Decimal(1100))

    def test_size_normalized_zero_sqft_rejected(self):
        with self.assertRaises(statistics.StatisticsError):
            statistics.size_normalized_price(Decimal(400000), 1200, 0)

    def test_deterministic_rounding(self):
        self.assertEqual(statistics.round_to_nearest(Decimal("415250"), 500), Decimal(415500))
        self.assertEqual(statistics.round_to_nearest(Decimal("415249"), 500), Decimal(415000))

    def test_extreme_values_handled(self):
        # a huge price does not overflow (Decimal is arbitrary precision)
        v = statistics.size_normalized_price(Decimal("99999999999"), 1000, 1000)
        self.assertEqual(v, Decimal("99999999999"))


# ---------------------------------------------------------------------------
class OutputTests(unittest.TestCase):
    def _rows(self, n, **kw):
        return [crow(f"202600{i:03d}", f"2026-04-{(i % 27) + 1:02d}", **kw) for i in range(1, n + 1)]

    def test_top_count_respected(self):
        res = ComparableService(InMemoryDataSource(self._rows(12)), max_returned=5).find_comparables(subj())
        self.assertEqual(res.selected_comparable_count, 5)
        self.assertEqual(res.candidate_count, 12)

    def test_point_estimate_withheld_when_insufficient(self):
        res = ComparableService(InMemoryDataSource(self._rows(2))).find_comparables(subj())
        self.assertIsNone(res.indicated_value)

    def test_confidence_high(self):
        res = ComparableService(InMemoryDataSource(self._rows(8))).find_comparables(subj())
        self.assertEqual(res.confidence, "high")

    def test_confidence_medium(self):
        # 5 strong comps: fails High (needs >=7) but meets Medium.
        res = ComparableService(InMemoryDataSource(self._rows(5))).find_comparables(subj())
        self.assertEqual(res.confidence, "medium")

    def test_confidence_low_with_three(self):
        res = ComparableService(InMemoryDataSource(self._rows(3))).find_comparables(subj())
        self.assertEqual(res.confidence, "low")

    def test_component_scores_sum_consistently(self):
        res = ComparableService(InMemoryDataSource(self._rows(6))).find_comparables(subj())
        for c in res.comparables:
            self.assertAlmostEqual(sum(c.weighted_components.values()), c.similarity, places=2)

    def test_no_private_fields_exposed(self):
        res = ComparableService(InMemoryDataSource(self._rows(5))).find_comparables(subj())
        d = res.to_dict()  # default: no address
        for comp in d["comparables"]:
            leaked = {k for k in comp if k.lower() in PRIVACY_BANNED_FIELDS}
            self.assertFalse(leaked)
            self.assertNotIn("address", comp)
        # opt-in address only
        d2 = res.to_dict(include_address=True)
        self.assertIn("address", d2["comparables"][0])

    def test_service_and_cli_cannot_write(self):
        import inspect
        from samvision.comparables import find_comps, service
        for mod in (service, find_comps):
            src = inspect.getsource(mod).lower()
            for banned in ("insert(", "update(", "sa.insert", "sa.update",
                           "authorize", "commit(", "engine.begin"):
                self.assertNotIn(banned, src, f"{mod.__name__} must be read-only ({banned})")


# ---------------------------------------------------------------------------
class DatasetTests(unittest.TestCase):
    def test_only_approved_returned(self):
        ds = InMemoryDataSource([
            crow("20260001", "2026-04-01", status="approved"),
            crow("20260002", "2026-04-02", status="flagged"),
            crow("20260003", "2026-04-03", status="rejected"),
        ])
        rows = ds.fetch(property_type="RES", sold_start=date(2026, 1, 1), sold_end=date(2026, 6, 1))
        self.assertEqual([r["mls_number"] for r in rows], ["20260001"])

    def test_sold_date_boundaries_enforced(self):
        ds = InMemoryDataSource([
            crow("20260001", "2026-03-01"), crow("20260002", "2026-05-01"),
            crow("20260003", "2026-07-01"),
        ])
        rows = ds.fetch(property_type="RES", sold_start=date(2026, 4, 1), sold_end=date(2026, 6, 1))
        self.assertEqual([r["mls_number"] for r in rows], ["20260002"])

    def test_filters_compose(self):
        ds = InMemoryDataSource([
            crow("20260001", "2026-04-01", hood="A", area="X"),
            crow("20260002", "2026-04-02", hood="B", area="X"),
            crow("20260003", "2026-04-03", hood="A", area="Y"),
        ])
        rows = ds.fetch(property_type="RES", sold_start=date(2026, 1, 1), sold_end=date(2026, 6, 1),
                        neighbourhood="A", area_code="X", exclude_mls="20260099")
        self.assertEqual([r["mls_number"] for r in rows], ["20260001"])

    def test_exclude_mls(self):
        ds = InMemoryDataSource([crow("20260001", "2026-04-01"), crow("20260002", "2026-04-02")])
        rows = ds.fetch(property_type="RES", sold_start=date(2026, 1, 1), sold_end=date(2026, 6, 1),
                        exclude_mls="20260001")
        self.assertEqual([r["mls_number"] for r in rows], ["20260002"])


if __name__ == "__main__":
    unittest.main()
