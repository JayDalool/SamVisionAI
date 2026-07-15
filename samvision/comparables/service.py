"""Comparable-sales service: query → widen → score → rank → price → confidence.

Deterministic and read-only. Depends on a ``ComparableDataSource`` (never raw
SQL); performs no writes. No trained model is consulted.
"""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Optional

from . import constants, explain, scoring, statistics
from .datasource import ComparableDataSource
from .models import CandidateSale, ComparableResult, ScoredComparable, SubjectProperty


class ComparableService:
    def __init__(
        self,
        datasource: ComparableDataSource,
        *,
        min_usable: int = constants.MIN_USABLE_COMPS,
        max_returned: int = constants.MAX_RETURNED_COMPS,
    ):
        self._ds = datasource
        self._min_usable = min_usable
        self._max_returned = max_returned

    # -- tier search --------------------------------------------------------
    def _applicable_tiers(self, subject: SubjectProperty):
        tiers = []
        for t in constants.SEARCH_TIERS:
            if t.geography == "neighbourhood" and not subject.neighbourhood:
                continue  # no neighbourhood -> skip Tier 1
            if t.geography == "area_code" and not subject.area_code:
                continue  # no area code -> area tiers impossible
            tiers.append(t)
        return tiers

    def _search_tier(self, subject: SubjectProperty, tier) -> list[ScoredComparable]:
        sold_end = subject.valuation_date
        sold_start = subject.valuation_date - timedelta(days=tier.lookback_days)
        kwargs = dict(
            property_type=subject.property_type,
            sold_start=sold_start,
            sold_end=sold_end,
            exclude_mls=subject.mls_number,
        )
        if tier.geography == "neighbourhood":
            kwargs["neighbourhood"] = subject.neighbourhood
        else:
            kwargs["area_code"] = subject.area_code
        rows = self._ds.fetch(**kwargs)

        scored: list[ScoredComparable] = []
        seen: set[str] = set()
        for row in rows:
            cand = CandidateSale.from_mapping(row)
            if not cand.mls_number or cand.mls_number in seen:
                continue
            eligible, _reason = scoring.eligibility(cand, subject, tier.lookback_days)
            if not eligible:
                continue
            seen.add(cand.mls_number)
            sc = scoring.score(cand, subject, tier.name)
            sc.reasons = explain.comparable_reasons(sc, subject)
            sc.warnings = explain.comparable_warnings(sc)
            scored.append(sc)
        return scored

    # -- ranking ------------------------------------------------------------
    @staticmethod
    def _rank(comps: list[ScoredComparable]) -> list[ScoredComparable]:
        # Stable tie-break: similarity desc, sold_date desc, mls asc.
        return sorted(
            comps,
            key=lambda c: (
                -c.similarity,
                -(c.candidate.sold_date.toordinal() if c.candidate.sold_date else 0),
                str(c.candidate.mls_number or ""),
            ),
        )

    # -- price indicators ---------------------------------------------------
    def _price_indicators(self, subject: SubjectProperty,
                          comps: list[ScoredComparable]) -> tuple[dict, list[str]]:
        warnings: list[str] = []
        # compute per-comp ppsf + size-normalized price
        for c in comps:
            c.ppsf = statistics.price_per_sqft(c.candidate.sold_price, c.candidate.living_area_sqft)
            c.size_normalized_price = statistics.size_normalized_price(
                c.candidate.sold_price, subject.living_area_sqft, c.candidate.living_area_sqft
            )
        # outlier guard on ppsf relative to the plain median ppsf
        med_ppsf = statistics.median([float(c.ppsf) for c in comps])
        band_low = Decimal(str(constants.PPSF_SANITY_LOW)) * Decimal(str(med_ppsf))
        band_high = Decimal(str(constants.PPSF_SANITY_HIGH)) * Decimal(str(med_ppsf))
        usable = [c for c in comps if band_low <= c.ppsf <= band_high]
        if len(usable) < len(comps):
            warnings.append("extreme_ppsf_excluded")
        if len(usable) < self._min_usable:
            usable = comps  # don't let the guard starve the estimate

        def pairs(attr):
            return [(getattr(c, attr), Decimal(str(c.similarity))) for c in usable]

        sn_pairs = pairs("size_normalized_price")
        indicated = statistics.round_to_nearest(
            statistics.weighted_median(sn_pairs), constants.INDICATED_VALUE_ROUND_TO)
        low = statistics.round_to_nearest(
            statistics.weighted_quantile(sn_pairs, 0.25), constants.INDICATED_VALUE_ROUND_TO)
        high = statistics.round_to_nearest(
            statistics.weighted_quantile(sn_pairs, 0.75), constants.INDICATED_VALUE_ROUND_TO)
        direct = statistics.weighted_median(
            [(c.candidate.sold_price, Decimal(str(c.similarity))) for c in usable])
        wmed_ppsf = statistics.weighted_median(pairs("ppsf"))
        spread = statistics.relative_spread(low, high, indicated)

        return {
            "indicated_value": indicated,
            "indicated_range_low": low,
            "indicated_range_high": high,
            "direct_weighted_median_price": direct.quantize(Decimal("0.01")),
            "weighted_median_ppsf": wmed_ppsf.quantize(Decimal("0.01")),
            "price_spread": round(spread, 4),
        }, warnings

    # -- confidence ---------------------------------------------------------
    def _confidence(self, comps, median_similarity, mean_coverage, spread,
                    tier_widened) -> tuple[str, list[str]]:
        n = len(comps)
        reasons = [
            f"{n} comparables", f"median similarity {median_similarity:.1f}",
            f"mean coverage {mean_coverage:.2f}", f"price spread {spread:.2f}",
            "geography widened beyond neighbourhood" if tier_widened else "no geographic widening",
        ]
        h, m = constants.CONFIDENCE_HIGH, constants.CONFIDENCE_MEDIUM
        if (n >= h["min_comps"] and median_similarity >= h["min_median_similarity"]
                and mean_coverage >= h["min_mean_coverage"] and spread <= h["max_spread"]
                and not tier_widened):
            return "high", reasons
        if (n >= m["min_comps"] and median_similarity >= m["min_median_similarity"]
                and mean_coverage >= m["min_mean_coverage"] and spread <= m["max_spread"]):
            return "medium", reasons
        return "low", reasons

    # -- public entry -------------------------------------------------------
    def find_comparables(self, subject: SubjectProperty) -> ComparableResult:
        subject.validate()
        tiers = self._applicable_tiers(subject)

        selected: list[ScoredComparable] = []
        selected_tier: Optional[str] = None
        tier_index = 0
        for idx, tier in enumerate(tiers):
            found = self._search_tier(subject, tier)
            selected, selected_tier, tier_index = found, tier.name, idx
            if len(found) >= self._min_usable:
                break

        tier_widened = tier_index > 0
        warnings: list[str] = []
        if tier_widened:
            warnings.append("geographic_or_temporal_widening")

        ranked = self._rank(selected)
        candidate_count = len(ranked)
        comps = ranked[: self._max_returned]

        sold_dates = sorted(c.candidate.sold_date for c in comps if c.candidate.sold_date)
        date_range = ((sold_dates[0].isoformat(), sold_dates[-1].isoformat())
                      if sold_dates else None)

        # insufficient data -> return records, no point estimate
        if candidate_count < self._min_usable:
            warnings.append("insufficient_data")
            return ComparableResult(
                subject=subject, status="insufficient_data", tier=selected_tier,
                tier_widened=tier_widened, candidate_count=candidate_count,
                comparables=comps, confidence="insufficient",
                confidence_reasons=[f"only {candidate_count} usable comparables "
                                    f"(minimum {self._min_usable})"],
                warnings=warnings, sold_date_range=date_range,
            )

        indicators, ind_warnings = self._price_indicators(subject, comps)
        warnings.extend(ind_warnings)
        median_similarity = statistics.median([c.similarity for c in comps])
        mean_coverage = sum(c.data_coverage for c in comps) / len(comps)
        confidence, conf_reasons = self._confidence(
            comps, median_similarity, mean_coverage, indicators["price_spread"], tier_widened)

        return ComparableResult(
            subject=subject, status="ok", tier=selected_tier, tier_widened=tier_widened,
            candidate_count=candidate_count, comparables=comps,
            indicated_value=indicators["indicated_value"],
            indicated_range_low=indicators["indicated_range_low"],
            indicated_range_high=indicators["indicated_range_high"],
            direct_weighted_median_price=indicators["direct_weighted_median_price"],
            weighted_median_ppsf=indicators["weighted_median_ppsf"],
            median_similarity=round(median_similarity, 2),
            price_spread=indicators["price_spread"],
            confidence=confidence, confidence_reasons=conf_reasons,
            warnings=warnings, sold_date_range=date_range,
        )
