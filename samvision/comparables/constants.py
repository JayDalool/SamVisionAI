"""Tunable constants for the comparable-sales engine.

Everything configurable lives here as a named constant — no scattered literals in
scoring/service code. See ``docs/comparable_sales_engine_design.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- comparable-count / lookback policy -----------------------------------
MIN_USABLE_COMPS = 3        # below this: no point estimate (insufficient data)
PREFERRED_COMPS = 5
MAX_RETURNED_COMPS = 10
MAX_LOOKBACK_DAYS = 730     # hard ceiling on how old a comparable may be


@dataclass(frozen=True)
class SearchTier:
    """One geographic/temporal search tier. Narrowest tiers come first."""
    name: str
    geography: str          # "neighbourhood" | "area_code"
    lookback_days: int


# Narrowest first. Tier 1 requires a subject neighbourhood; if absent it is
# skipped and Tier 2 (area code) becomes the entry tier.
SEARCH_TIERS: tuple[SearchTier, ...] = (
    SearchTier("tier1_neighbourhood_365", "neighbourhood", 365),
    SearchTier("tier2_area_365", "area_code", 365),
    SearchTier("tier3_area_730", "area_code", 730),
)

# --- scoring weights (sum == 100) -----------------------------------------
WEIGHTS: dict[str, int] = {
    "geography": 25,
    "living_area": 25,
    "recency": 20,
    "bedrooms": 8,
    "bathrooms": 8,
    "year_built": 6,
    "style": 4,
    "garage": 2,
    "basement": 2,
}
assert sum(WEIGHTS.values()) == 100, "scoring weights must sum to 100"

# Optional dimensions that count toward data coverage.
COVERAGE_DIMENSIONS = ("bedrooms", "bathrooms", "year_built", "style", "garage", "basement")

# --- component tolerances (deterministic decay) ---------------------------
GEO_SAME_AREA_SUBSCORE = 0.5     # same area code but not same neighbourhood
LIVING_AREA_ZERO_AT_PCT = 0.50   # subscore hits 0 at 50% relative size difference
BEDROOM_ZERO_AT_DIFF = 3.0
BATHROOM_ZERO_AT_DIFF = 3.0
YEAR_BUILT_ZERO_AT_DIFF = 50.0   # years of age difference

# --- price-indicator guards -----------------------------------------------
PPSF_SANITY_LOW = 0.2            # exclude comps whose ppsf is <0.2x subject band
PPSF_SANITY_HIGH = 5.0           # ... or >5x — protects against extreme ratios
INDICATED_VALUE_ROUND_TO = 500   # round indicated value/range to nearest $500

# --- confidence thresholds (rule-based, documented) -----------------------
CONFIDENCE_HIGH = {
    "min_comps": 7, "min_median_similarity": 75.0,
    "min_mean_coverage": 0.75, "max_spread": 0.20, "require_tier1": True,
}
CONFIDENCE_MEDIUM = {
    "min_comps": 4, "min_median_similarity": 60.0,
    "min_mean_coverage": 0.50, "max_spread": 0.35, "require_tier1": False,
}

# --- plausibility bounds for subject validation ---------------------------
MIN_PLAUSIBLE_YEAR_BUILT = 1800
