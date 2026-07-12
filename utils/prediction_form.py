# file: utils/prediction_form.py
"""State + field-schema helpers for the Price Prediction form.

This module holds the *pure* logic behind the Streamlit prediction form so it can
be unit-tested without a running Streamlit/browser session. The Streamlit layer
(`app/streamlit_app.py`) only wires widgets to these helpers.

Design rules (see docs / commit message "Fix prediction form field state"):
  * Every persistent widget has a stable, explicit key defined here.
  * Widget keys never embed changing indexes or field values.
  * Session state is seeded only when a key is absent; user input is never
    overwritten on a rerun.
  * The multi-offer premium is a *suggestion* the user can apply explicitly; it
    is not silently written into the slider on every rerun (that was the bug that
    reset the slider whenever an unrelated field changed).
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

# ---------------------------------------------------------------------------
# Stable widget keys (never interpolate options/indexes/values into these).
# ---------------------------------------------------------------------------
KEY_NEIGHBORHOOD = "prediction_neighborhood"
KEY_HOUSE_TYPE = "prediction_house_type"
KEY_STYLE = "prediction_style"
KEY_GARAGE_TYPE = "prediction_garage_type"
KEY_BASEMENT_TYPE = "prediction_basement_type"
KEY_BEDROOMS = "prediction_bedrooms"
KEY_BATHROOMS = "prediction_bathrooms"
KEY_SQFT = "prediction_sqft"
KEY_BUILT_YEAR = "prediction_built_year"
KEY_SEASON = "prediction_season"
KEY_LIST_PRICE = "prediction_list_price"
KEY_IS_MULTI_OFFER = "prediction_is_multi_offer"
KEY_PREMIUM_PCT = "prediction_premium_pct"

# Every key that belongs to the prediction form (used for reset).
PREDICTION_STATE_KEYS: tuple[str, ...] = (
    KEY_NEIGHBORHOOD,
    KEY_HOUSE_TYPE,
    KEY_STYLE,
    KEY_GARAGE_TYPE,
    KEY_BASEMENT_TYPE,
    KEY_BEDROOMS,
    KEY_BATHROOMS,
    KEY_SQFT,
    KEY_BUILT_YEAR,
    KEY_SEASON,
    KEY_LIST_PRICE,
    KEY_IS_MULTI_OFFER,
    KEY_PREMIUM_PCT,
)

# ---------------------------------------------------------------------------
# Static option lists / lookups.
# ---------------------------------------------------------------------------
BASEMENT_TYPES: tuple[str, ...] = (
    "None",
    "Crawl Space",
    "Full (Unfinished)",
    "Full (Finished)",
    "Walkout",
)

SEASONS: tuple[str, ...] = ("Winter", "Spring", "Summer", "Fall")

BASEMENT_ADJUSTMENTS: dict[str, int] = {
    "None": -45000,
    "Crawl Space": -45000,
    "Full (Unfinished)": -15000,
    "Full (Finished)": 25000,
    "Walkout": 0,
}

DEFAULT_MULTI_OFFER_PREMIUM = 5.0

# Educated-guess premiums; any area not listed defaults to DEFAULT_MULTI_OFFER_PREMIUM.
MULTI_OFFER_PREMIUMS: dict[str, float] = {
    "Bridgwater Trails": 8.0, "Bridgwater Lakes": 7.5, "Bridgwater Forest": 7.5, "Bridgwater Centre": 7.5,
    "Tuxedo": 9.0, "Old Tuxedo": 9.0, "Crescentwood": 8.0, "Crescent Park": 8.0,
    "Osborne Village": 8.5, "River-Osborne": 8.0, "West Wolseley": 7.5,
    "Waverley Heights": 7.0, "Waverley West B": 7.0, "Fort Richmond": 7.0, "Linden Woods": 8.0,
    "River Heights": 6.5, "North River Heights": 6.5, "Sage Creek": 6.5,
    "Richmond West": 6.5, "Grant Park": 6.0, "Jameswood": 6.0,
    "Kensington": 6.0, "King Edward": 6.0, "University": 6.0,
    "Central River Heights": 6.5, "Exchange District": 6.0, "Prairie Pointe": 6.0,
    "Charleswood": 5.5, "St. Vital": 5.5, "St. Norbert": 5.5,
    "Amber Trails": 5.5, "Garden City": 5.5, "Maples": 5.5,
    "Westwood": 5.5, "Elm Park": 5.5,
}


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------
def suggested_multi_offer_premium(
    neighborhood: str | None,
    is_multi_offer: bool,
    premiums: Mapping[str, float] = MULTI_OFFER_PREMIUMS,
) -> float:
    """Suggested premium (%) for a neighborhood.

    Returns 0.0 when the listing is not flagged multi-offer. Otherwise returns the
    neighborhood-specific premium, falling back to ``DEFAULT_MULTI_OFFER_PREMIUM``
    for unknown neighborhoods. This is only a *suggestion* — it is never written
    into the slider automatically.
    """
    if not is_multi_offer:
        return 0.0
    return float(premiums.get(neighborhood or "", DEFAULT_MULTI_OFFER_PREMIUM))


def basement_adjustment(basement_type: str | None) -> int:
    """Dollar adjustment applied for a basement type (0 for unknown types)."""
    return BASEMENT_ADJUSTMENTS.get(basement_type or "", 0)


def _to_int(value: Any, fallback: int) -> int:
    try:
        if value is None:
            return fallback
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _to_float(value: Any, fallback: float) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def field_defaults(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Seed values for the numeric fields, taken from a representative data row.

    Missing/blank/non-numeric values fall back to sensible constants so the form
    always initializes to a valid state.
    """
    if row is None:
        row = {}

    def get(key: str) -> Any:
        try:
            return row.get(key)  # type: ignore[union-attr]
        except AttributeError:
            return None

    return {
        KEY_BEDROOMS: max(0, min(10, _to_int(get("bedrooms"), 3))),
        KEY_BATHROOMS: max(0.5, min(6.0, _to_float(get("bathrooms"), 2.0))),
        KEY_SQFT: max(300, min(6000, _to_int(get("sqft"), 1200))),
        KEY_BUILT_YEAR: _to_int(get("built_year"), 2000),
        KEY_LIST_PRICE: max(50000, min(2000000, _to_int(get("list_price"), 300000))),
    }


def ensure_choice(
    session_state: MutableMapping[str, Any],
    key: str,
    options: Sequence[Any],
    default_index: int = 0,
) -> None:
    """Seed a selectbox key with a valid option.

    Seeds only when the key is absent OR its current value is no longer a valid
    option (e.g. the underlying data was reloaded and the previously chosen value
    disappeared). Otherwise the user's existing choice is preserved.
    """
    if not options:
        return
    if key not in session_state or session_state[key] not in options:
        idx = default_index if 0 <= default_index < len(options) else 0
        session_state[key] = options[idx]


def ensure_value(session_state: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Seed a numeric/boolean key only when it is not already present."""
    if key not in session_state:
        session_state[key] = value


def init_prediction_state(
    session_state: MutableMapping[str, Any],
    *,
    neighborhoods: Sequence[str],
    house_types: Sequence[str],
    styles: Sequence[str],
    garage_types: Sequence[str],
    defaults: Mapping[str, Any],
) -> None:
    """Idempotently seed all prediction-form widget state.

    Safe to call on every rerun: existing user input is never overwritten. Only
    absent keys (or selectbox choices that no longer exist in their options) are
    (re)seeded.
    """
    ensure_choice(session_state, KEY_NEIGHBORHOOD, neighborhoods)
    ensure_choice(session_state, KEY_HOUSE_TYPE, house_types)
    ensure_choice(session_state, KEY_STYLE, styles)
    ensure_choice(session_state, KEY_GARAGE_TYPE, garage_types)
    ensure_choice(session_state, KEY_BASEMENT_TYPE, BASEMENT_TYPES)
    # Season default = Summer (index 2), matching prior behaviour.
    ensure_choice(session_state, KEY_SEASON, SEASONS, default_index=2)

    ensure_value(session_state, KEY_BEDROOMS, defaults[KEY_BEDROOMS])
    ensure_value(session_state, KEY_BATHROOMS, defaults[KEY_BATHROOMS])
    ensure_value(session_state, KEY_SQFT, defaults[KEY_SQFT])
    ensure_value(session_state, KEY_BUILT_YEAR, defaults[KEY_BUILT_YEAR])
    ensure_value(session_state, KEY_LIST_PRICE, defaults[KEY_LIST_PRICE])
    ensure_value(session_state, KEY_IS_MULTI_OFFER, False)
    ensure_value(session_state, KEY_PREMIUM_PCT, 0.0)


def reset_prediction_state(session_state: MutableMapping[str, Any]) -> None:
    """Clear every prediction-form key so the next render reseeds defaults."""
    for key in PREDICTION_STATE_KEYS:
        session_state.pop(key, None)
