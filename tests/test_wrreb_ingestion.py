"""WRREB ingestion tests using SYNTHETIC fixtures only.

No real WRREB/listing data is used here — every page is hand-built to imitate the
report structure. This guards the safety-critical behaviours: real sold dates
are parsed, dates are never fabricated from today()/MLS year, private fields are
excluded, conflicts/duplicates surface, and the dry-run performs no writes.
"""
import unittest
from datetime import date, timedelta

from samvision.ingestion import agent_single_line as sl
from samvision.ingestion import client_full as cf
from samvision.ingestion import import_wrreb_batch as cli
from samvision.ingestion.models import (CanonicalSale, ClientFullRecord, ReasonCode,
                                        SingleLineRecord, normalize_mls)
from samvision.ingestion.pdf_text import PageLayout, TextFragment
from samvision.ingestion.reconcile import reconcile
from samvision.ingestion.validate import validate


def F(y, x, text):
    return TextFragment(y=y, x0=x, x1=x + 10, text=text)


def single_line_page(include_date=True):
    frags = [
        F(100, 47, "MLS® #"), F(100, 97, "S"), F(100, 113, "Ar Address"),
        F(100, 305, "List Price"), F(100, 370, "Sold Price"), F(100, 425, "Date"),
        F(100, 458, "DOM"), F(100, 481, "Ty"), F(100, 496, "Style"),
        F(100, 532, "YrBt"), F(100, 561, "SqFt"),
        F(92, 28, "1"), F(92, 47, "202512345"), F(92, 97, "S"),
        F(92, 112, "1A 76 Test Bay"), F(92, 304, "$479,000"), F(92, 371, "$482,500"),
        F(92, 458, "7"), F(92, 481, "DP"), F(92, 496, "TWO"),
        F(92, 531, "1959"), F(92, 561, "1904"),
    ]
    if include_date:
        frags.append(F(92, 415, "01/10/2026"))
    return PageLayout(page_number=1, fragments=frags)


def client_full_page():
    rows = [
        (100, 28, "76 Test Bay , Winnipeg R3T 3L2"),
        (95, 28, "MLS® #:"), (94, 28, "202512345"),
        (93, 28, "Nghbrhd:"), (92, 28, "Riverview"),
        (91, 28, "Linc #:"), (90, 28, "012R014506100"),
        (89, 459, "List Price: $479,000"), (88, 459, "Sell Price: $482,500"),
        (87, 459, "Sell Date: 05/28/2026"), (86, 459, "DOM:"), (85, 459, "7"),
        (84, 28, "Type:"), (83, 28, "DP"), (82, 28, "Style:"), (81, 28, "TWO"),
        (80, 28, "Yr Built/Age:"), (79, 28, "1959/"),
        (78, 28, "Liv Area:"), (77, 28, "176.89 M2/1,904 SF"),
        (76, 28, "BDA: 4"), (75, 28, "TBD: 4"), (74, 28, "Baths: F2/H0"),
        (73, 28, "Tax Yr:"), (72, 28, "2025"), (71, 28, "Gross Tax: $6,000.76"),
        # private content that must NEVER be captured:
        (70, 28, "Remarks:"), (69, 28, "PRIVATE agent note call 204-555-1234"),
    ]
    return PageLayout(page_number=1, fragments=[F(*r) for r in rows])


class TestSingleLine(unittest.TestCase):
    def test_parses_real_sold_date_and_fields(self):
        rec = sl.parse_page(single_line_page())[0]
        self.assertEqual(rec.mls_number, "202512345")
        self.assertEqual(rec.sold_date, date(2026, 1, 10))
        self.assertEqual(rec.list_price, 479000)
        self.assertEqual(rec.sold_price, 482500)
        self.assertEqual(rec.area_code, "1A")
        self.assertEqual(rec.address, "76 Test Bay")
        self.assertEqual(rec.year_built, 1959)
        self.assertEqual(rec.living_area_sqft, 1904)
        self.assertEqual(rec.dom, 7)

    def test_missing_date_is_none_never_today(self):
        rec = sl.parse_page(single_line_page(include_date=False))[0]
        self.assertIsNone(rec.sold_date)


class TestClientFull(unittest.TestCase):
    def setUp(self):
        self.rec = cf._parse_page(client_full_page())

    def test_authoritative_sell_date_and_details(self):
        self.assertEqual(self.rec.mls_number, "202512345")
        self.assertEqual(self.rec.sold_date, date(2026, 5, 28))
        self.assertEqual(self.rec.linc_number, "012R014506100")
        self.assertEqual(self.rec.neighbourhood, "Riverview")
        self.assertEqual(self.rec.postal_code, "R3T 3L2")
        self.assertEqual(self.rec.sold_price, 482500)
        self.assertEqual(self.rec.bedrooms_total, 4)
        self.assertEqual(self.rec.full_bathrooms, 2)
        self.assertEqual(self.rec.living_area_sqft, 1904)
        self.assertEqual(self.rec.tax_year, 2025)

    def test_private_fields_excluded(self):
        blob = repr(self.rec)
        self.assertNotIn("PRIVATE agent note", blob)
        self.assertNotIn("204-555-1234", blob)


class TestReconcile(unittest.TestCase):
    def _pair(self, **cf_over):
        s = SingleLineRecord(mls_number="202512345", sold_date=date(2026, 5, 28),
                             sold_price=482500, list_price=479000, address="76 Test Bay")
        c = ClientFullRecord(mls_number="202512345", sold_date=date(2026, 5, 28),
                             sold_price=482500, list_price=479000, address="76 Test Bay",
                             linc_number="012R014506100", property_type_code="DP")
        for k, v in cf_over.items():
            setattr(c, k, v)
        return [s], [c]

    def test_full_match_clean(self):
        canon, summ = reconcile(*self._pair())
        self.assertEqual(summ["matched_both"], 1)
        codes = {i.code for i in canon[0].issues}
        self.assertNotIn(ReasonCode.SOLD_DATE_CONFLICT, codes)

    def test_sale_year_from_sold_date_not_mls(self):
        # MLS year is 2025 but the real sale is in 2026
        canon, _ = reconcile(*self._pair())
        cs = canon[0]
        self.assertEqual(cs.sale_year, 2026)      # from sold_date
        self.assertEqual(cs.mls_year_hint, 2025)  # from MLS number, stored only as a hint

    def test_sold_date_conflict(self):
        canon, _ = reconcile(*self._pair(sold_date=date(2026, 5, 29)))
        self.assertIn(ReasonCode.SOLD_DATE_CONFLICT, {i.code for i in canon[0].issues})

    def test_sold_price_conflict(self):
        canon, _ = reconcile(*self._pair(sold_price=999999))
        self.assertIn(ReasonCode.SOLD_PRICE_CONFLICT, {i.code for i in canon[0].issues})

    def test_missing_counterpart(self):
        s = [SingleLineRecord(mls_number="202512345", sold_date=date(2026, 5, 28), sold_price=1)]
        canon, summ = reconcile(s, [])
        self.assertEqual(summ["single_only"], 1)
        self.assertIn(ReasonCode.MISSING_CLIENT_FULL_RECORD, {i.code for i in canon[0].issues})

    def test_duplicate_mls(self):
        c = ClientFullRecord(mls_number="202512345", sold_date=date(2026, 5, 28), sold_price=1)
        canon, summ = reconcile([], [c, c])
        self.assertEqual(summ["duplicate_mls"], ["202512345"])
        self.assertIn(ReasonCode.DUPLICATE_MLS, {i.code for i in canon[0].issues})


class TestValidate(unittest.TestCase):
    def _cs(self, **over):
        base = dict(mls_number="202512345", sold_price=482500, sold_date=date(2026, 1, 10),
                    property_type_code="DP", address="76 Test Bay", living_area_sqft=1904)
        base.update(over)
        return CanonicalSale(**base)

    def test_accepts_valid(self):
        out = validate([self._cs()], today=date(2026, 7, 12))
        self.assertEqual(out[0].data_quality_status, "accepted")

    def test_rejects_future_date(self):
        out = validate([self._cs(sold_date=date(2027, 1, 1))], today=date(2026, 7, 12))
        self.assertEqual(out[0].data_quality_status, "rejected")
        self.assertIn(ReasonCode.FUTURE_SOLD_DATE, {i.code for i in out[0].issues})

    def test_rejects_missing_required(self):
        out = validate([self._cs(sold_price=None)], today=date(2026, 7, 12))
        self.assertEqual(out[0].data_quality_status, "rejected")

    def test_no_today_substitution_leaves_year_none(self):
        # a sale with no date must not inherit the current year
        cs = self._cs(sold_date=None, sale_year=None)
        out = validate([cs], today=date(2026, 7, 12))
        self.assertIsNone(out[0].sale_year)
        self.assertEqual(out[0].data_quality_status, "rejected")


class TestNormalizeMls(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(normalize_mls("MLS® # 202611485"), "202611485")
        self.assertEqual(normalize_mls("202611485"), "202611485")
        self.assertIsNone(normalize_mls("N/A"))
        self.assertIsNone(normalize_mls(None))


class TestDryRunSafety(unittest.TestCase):
    def test_import_module_has_no_db_or_model_deps(self):
        import inspect
        src = inspect.getsource(cli)
        for banned in ("psycopg2", "sqlalchemy", "to_sql", "joblib", "train_model"):
            self.assertNotIn(banned, src)

    def test_dry_run_writes_only_to_output(self, ):
        import tempfile, os, json
        from samvision.ingestion import import_wrreb_batch as m

        s_recs = [SingleLineRecord(mls_number="202512345", sold_date=date(2026, 1, 10),
                                   sold_price=482500, list_price=479000, address="76 Test Bay",
                                   property_type_code="DP")]
        c_recs = [ClientFullRecord(mls_number="202512345", sold_date=date(2026, 1, 10),
                                   sold_price=482500, list_price=479000, address="76 Test Bay",
                                   linc_number="012R014506100", property_type_code="DP")]
        orig = (m.agent_single_line.parse, m.client_full.parse, m.make_batch_id, m.sha256_file)
        m.agent_single_line.parse = lambda p: (s_recs, [])
        m.client_full.parse = lambda p: (c_recs, [])
        m.make_batch_id = lambda a, b: "test_batch"
        m.sha256_file = lambda p: "deadbeef"
        try:
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, "run")
                code = m.run("s.pdf", "c.pdf", out, dry_run=True)
                self.assertEqual(code, 0)
                with open(os.path.join(out, "summary.json")) as fh:
                    summary = json.load(fh)
                self.assertEqual(summary["accepted"], 1)
                self.assertIn("not-implemented", summary["production_load"])
        finally:
            (m.agent_single_line.parse, m.client_full.parse, m.make_batch_id, m.sha256_file) = orig


if __name__ == "__main__":
    unittest.main()
