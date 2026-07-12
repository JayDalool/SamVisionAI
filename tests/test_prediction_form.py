from __future__ import annotations

import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from utils.prediction_form import (
    BASEMENT_TYPES,
    DEFAULT_MULTI_OFFER_PREMIUM,
    KEY_BEDROOMS,
    KEY_IS_MULTI_OFFER,
    KEY_NEIGHBORHOOD,
    KEY_PREMIUM_PCT,
    PREDICTION_STATE_KEYS,
    basement_adjustment,
    ensure_choice,
    field_defaults,
    init_prediction_state,
    reset_prediction_state,
    suggested_multi_offer_premium,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "app" / "streamlit_app.py"


# ---------------------------------------------------------------------------
# Pure-logic tests (no Streamlit runtime required).
# ---------------------------------------------------------------------------
class SuggestedPremiumTests(unittest.TestCase):
    def test_zero_when_not_multi_offer(self) -> None:
        self.assertEqual(suggested_multi_offer_premium("Tuxedo", False), 0.0)

    def test_known_neighborhood_premium(self) -> None:
        self.assertEqual(suggested_multi_offer_premium("Tuxedo", True), 9.0)
        self.assertEqual(suggested_multi_offer_premium("Sage Creek", True), 6.5)

    def test_unknown_neighborhood_uses_default(self) -> None:
        self.assertEqual(
            suggested_multi_offer_premium("Nowhere-ville", True),
            DEFAULT_MULTI_OFFER_PREMIUM,
        )

    def test_none_neighborhood_uses_default(self) -> None:
        self.assertEqual(
            suggested_multi_offer_premium(None, True), DEFAULT_MULTI_OFFER_PREMIUM
        )


class BasementAdjustmentTests(unittest.TestCase):
    def test_known_types(self) -> None:
        self.assertEqual(basement_adjustment("Full (Finished)"), 25000)
        self.assertEqual(basement_adjustment("None"), -45000)
        self.assertEqual(basement_adjustment("Walkout"), 0)

    def test_unknown_type_is_zero(self) -> None:
        self.assertEqual(basement_adjustment("Something Else"), 0)
        self.assertEqual(basement_adjustment(None), 0)


class FieldDefaultsTests(unittest.TestCase):
    def test_reads_from_row(self) -> None:
        row = pd.Series(
            {"bedrooms": 4, "bathrooms": 2.5, "sqft": 1800, "built_year": 1995, "list_price": 450000}
        )
        d = field_defaults(row)
        self.assertEqual(d[KEY_BEDROOMS], 4)
        self.assertEqual(d["prediction_bathrooms"], 2.5)
        self.assertEqual(d["prediction_sqft"], 1800)
        self.assertEqual(d["prediction_built_year"], 1995)
        self.assertEqual(d["prediction_list_price"], 450000)

    def test_missing_and_bad_values_fall_back(self) -> None:
        d = field_defaults(pd.Series({"bedrooms": None, "sqft": "not-a-number"}))
        self.assertEqual(d[KEY_BEDROOMS], 3)
        self.assertEqual(d["prediction_sqft"], 1200)
        self.assertEqual(d["prediction_list_price"], 300000)

    def test_none_row(self) -> None:
        d = field_defaults(None)
        self.assertEqual(d[KEY_BEDROOMS], 3)

    def test_values_are_clamped_to_widget_ranges(self) -> None:
        d = field_defaults(pd.Series({"bedrooms": 99, "sqft": 10, "list_price": 9_999_999}))
        self.assertEqual(d[KEY_BEDROOMS], 10)
        self.assertEqual(d["prediction_sqft"], 300)
        self.assertEqual(d["prediction_list_price"], 2_000_000)


class StateSeedingTests(unittest.TestCase):
    def _defaults(self) -> dict:
        return field_defaults(
            pd.Series({"bedrooms": 3, "bathrooms": 2.0, "sqft": 1200, "built_year": 2000, "list_price": 300000})
        )

    def test_init_seeds_all_keys_once(self) -> None:
        state: dict = {}
        init_prediction_state(
            state,
            neighborhoods=["A", "B"],
            house_types=["H"],
            styles=["S"],
            garage_types=["G"],
            defaults=self._defaults(),
        )
        for key in PREDICTION_STATE_KEYS:
            self.assertIn(key, state)
        self.assertEqual(state[KEY_NEIGHBORHOOD], "A")
        self.assertEqual(state[KEY_PREMIUM_PCT], 0.0)
        self.assertIs(state[KEY_IS_MULTI_OFFER], False)

    def test_init_never_overwrites_existing_user_input(self) -> None:
        state = {KEY_NEIGHBORHOOD: "B", KEY_PREMIUM_PCT: 12.0, KEY_BEDROOMS: 5}
        init_prediction_state(
            state,
            neighborhoods=["A", "B"],
            house_types=["H"],
            styles=["S"],
            garage_types=["G"],
            defaults=self._defaults(),
        )
        self.assertEqual(state[KEY_NEIGHBORHOOD], "B")
        self.assertEqual(state[KEY_PREMIUM_PCT], 12.0)
        self.assertEqual(state[KEY_BEDROOMS], 5)

    def test_ensure_choice_reseeds_when_value_no_longer_valid(self) -> None:
        state = {KEY_NEIGHBORHOOD: "Gone"}
        ensure_choice(state, KEY_NEIGHBORHOOD, ["A", "B"])
        self.assertEqual(state[KEY_NEIGHBORHOOD], "A")

    def test_reset_clears_all_keys(self) -> None:
        state = {key: "x" for key in PREDICTION_STATE_KEYS}
        state["unrelated"] = "keep"
        reset_prediction_state(state)
        for key in PREDICTION_STATE_KEYS:
            self.assertNotIn(key, state)
        self.assertEqual(state["unrelated"], "keep")

    def test_basement_and_season_defaults(self) -> None:
        state: dict = {}
        init_prediction_state(
            state,
            neighborhoods=["A"],
            house_types=["H"],
            styles=["S"],
            garage_types=["G"],
            defaults=self._defaults(),
        )
        self.assertIn(state["prediction_basement_type"], BASEMENT_TYPES)
        self.assertEqual(state["prediction_season"], "Summer")


# ---------------------------------------------------------------------------
# End-to-end regression test through the real Streamlit app (AppTest).
# The app has no login gate, so the tabs (and the prediction form) render on the
# first run. get_engine() still needs DB env vars to build its URL, but the
# actual query is patched, so these values are never used to connect.
# ---------------------------------------------------------------------------
def _db_env() -> dict[str, str]:
    return {
        "SAMVISION_DB_NAME": "SamVision",
        "SAMVISION_DB_USER": "postgres",
        "SAMVISION_DB_PASSWORD": "test-only",
        "SAMVISION_DB_HOST": "127.0.0.1",
        "SAMVISION_DB_PORT": "5432",
    }


def _fake_housing_df() -> pd.DataFrame:
    rows = []
    for hood in ["Tuxedo", "Sage Creek", "Charleswood"]:
        rows.append(
            {
                "mls_number": f"MLS-{hood}", "address": f"1 {hood} St",
                "neighborhood": hood, "house_type": "Single Family Detached",
                "style": "Bungalow", "garage_type": "attached",
                "bedrooms": 3, "bathrooms": 2.0, "sqft": 1200, "built_year": 2000,
                "age": 25, "season": "Summer", "list_price": 300000, "sold_price": 320000,
                "sell_list_ratio": 1.06, "dom_days": 5, "region": "SW",
            }
        )
    return pd.DataFrame(rows)


@contextmanager
def _app_context():
    import streamlit as st

    rental_stub = types.ModuleType("utils.rental_income_tool")
    rental_stub.render_rental_income_tab = lambda: None
    # load_data() is @st.cache_data; clear it so a cached frame from another test
    # or a real DB read can't leak in here.
    st.cache_data.clear()
    with patch.dict(os.environ, _db_env(), clear=False), patch.dict(
        sys.modules, {"utils.rental_income_tool": rental_stub}
    ), patch("pandas.read_sql_query", return_value=_fake_housing_df()):
        yield


def _start_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    return app


class PredictionFormStateRegressionTests(unittest.TestCase):
    """Guards against the disappearing/resetting-field bug."""

    def test_all_prediction_widgets_have_stable_keys(self) -> None:
        with _app_context():
            app = _start_app()
            # Every widget is addressable by its explicit key (would raise otherwise).
            for key in PREDICTION_STATE_KEYS:
                self.assertIn(key, app.session_state)

    def test_changing_one_field_does_not_reset_others(self) -> None:
        with _app_context():
            app = _start_app()

            app.selectbox(key=KEY_NEIGHBORHOOD).set_value("Tuxedo")
            app.checkbox(key=KEY_IS_MULTI_OFFER).set_value(True)
            app.run()

            # Manually set the premium slider, then change unrelated fields.
            app.slider(key=KEY_PREMIUM_PCT).set_value(15.0)
            app.number_input(key=KEY_BEDROOMS).set_value(5)
            app.run()

            app.selectbox(key=KEY_NEIGHBORHOOD).set_value("Sage Creek")
            app.run()
            self.assertEqual(app.slider(key=KEY_PREMIUM_PCT).value, 15.0)
            self.assertEqual(app.number_input(key=KEY_BEDROOMS).value, 5)

            app.number_input(key=KEY_BEDROOMS).set_value(2)
            app.run()
            self.assertEqual(app.slider(key=KEY_PREMIUM_PCT).value, 15.0)
            self.assertEqual(app.selectbox(key=KEY_NEIGHBORHOOD).value, "Sage Creek")

    def test_premium_slider_not_auto_overwritten_on_rerun(self) -> None:
        with _app_context():
            app = _start_app()
            # Enabling multi-offer must NOT silently jump the slider to the
            # neighborhood suggestion (that was the reset bug).
            app.selectbox(key=KEY_NEIGHBORHOOD).set_value("Tuxedo")
            app.checkbox(key=KEY_IS_MULTI_OFFER).set_value(True)
            app.run()
            self.assertEqual(app.slider(key=KEY_PREMIUM_PCT).value, 0.0)

    def test_use_suggested_button_applies_premium(self) -> None:
        with _app_context():
            app = _start_app()
            app.selectbox(key=KEY_NEIGHBORHOOD).set_value("Tuxedo")
            app.checkbox(key=KEY_IS_MULTI_OFFER).set_value(True)
            app.run()
            next(b for b in app.button if b.label == "Use suggested").click()
            app.run()
            self.assertEqual(app.slider(key=KEY_PREMIUM_PCT).value, 9.0)

    def test_reset_button_restores_defaults(self) -> None:
        with _app_context():
            app = _start_app()
            app.number_input(key=KEY_BEDROOMS).set_value(7)
            app.slider(key=KEY_PREMIUM_PCT).set_value(20.0)
            app.run()

            next(b for b in app.button if "Reset Form" in b.label).click()
            app.run()
            self.assertEqual(app.number_input(key=KEY_BEDROOMS).value, 3)
            self.assertEqual(app.slider(key=KEY_PREMIUM_PCT).value, 0.0)


if __name__ == "__main__":
    unittest.main()
