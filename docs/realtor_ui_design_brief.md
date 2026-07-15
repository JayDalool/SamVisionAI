# Realtor UI Design Brief (SamVisionAI — Phase 14B)

Status: design brief. Input for the **Fable-built** professional realtor UI that
replaces the retired Streamlit prototype. Companion to
[`samvision_v2_upgrade_tracker.md`](samvision_v2_upgrade_tracker.md) and
[`comparable_sales_engine_design.md`](comparable_sales_engine_design.md).
See ADR-21 in [`samvision_v2_upgrade_decisions.md`](samvision_v2_upgrade_decisions.md).

## 1. Audience & purpose

Winnipeg **realtors**, daily use, in-office and on-device. The product helps a
realtor produce a **transparent comparable-sales opinion of value** for a subject
property in minutes and defend it to a client. It is **decision support, not an
appraisal** — that framing is non-negotiable and appears on every valuation
surface.

## 2. Product principles

1. **Professional & trustworthy** — clean, calm, brokerage-grade; nothing "beta".
2. **Transparent** — every number is explainable; the realtor can see *why* each
   comparable was chosen and how the value was derived.
3. **Fast** — subject → valuation in a few clicks; no dead ends.
4. **Honest about uncertainty** — confidence and warnings are first-class, not
   buried; weak results are visibly gated for manual review.
5. **Privacy-safe** — no private operational data ever shown (see §7).
6. **Not Streamlit** — a real, responsive web app; light/dark; accessible (WCAG AA).

## 3. Core workflows / screens

1. **Subject entry** — a focused form for the property being valued: valuation
   date (defaults to today for *convenience only*, always editable), area /
   neighbourhood, property type, living area (sqft), and optional beds / baths /
   year built / style / basement. Inline validation mirrors the engine's required
   fields (valuation date, property type, living area > 0, area or neighbourhood).
2. **Valuation result** — the hero screen (see §4).
3. **Comparable detail** — expand any comparable to see its similarity breakdown,
   key differences vs the subject, sold price, and price-per-sqft.
4. **Adjust & finalize** — include/exclude comparables, add realtor notes, and
   export/share a clean client-facing summary (later phase).

## 4. Valuation result — the hero screen

- **Indicated value** shown as a **range first** (`indicated_range_low –
  indicated_range_high`) with the point estimate secondary — never a lone number
  that reads as an appraisal.
- **Confidence badge** (High / Medium / Low / Insufficient) with a plain-language
  explanation and the `confidence_reasons`.
- **Supporting indicators:** weighted median price-per-sqft, direct weighted
  median price, median similarity, price spread, selected comparable count, search
  tier, and whether geography was widened.
- **Comparables table:** one row per selected comparable — masked MLS, area /
  neighbourhood, property type, sold date, sold price, ppsf, size-normalized
  price, similarity (0–100), data coverage, and a compact similarity-component
  bar. Include/exclude toggle per row (recompute is client-driven, engine stays
  authoritative).
- **Warnings panel:** surfaces engine warnings (extreme ppsf excluded, widening,
  insufficient data, etc.) prominently.
- **Decision-support disclaimer** always visible.

## 5. Confidence-gated presentation (from backtest evidence)

The backtest showed high confidence ≈ **2×** the within-10% hit rate of low, and
that widened/low-similarity/wide-spread cases are unreliable. So the UI must:

- **High confidence, not widened:** present as a usable indicated range.
- **Medium / Low / geographically widened / insufficient:** visibly **flag for
  manual review** — de-emphasize the point value, lead with the range and the
  reasons, and prompt the realtor to verify comparables by hand.
- **Insufficient (< 3 comps):** no point estimate at all — show the available
  records and why coverage was insufficient.

(Exact thresholds/warnings finalize in **Phase 12 guardrails**; the UI build 14B
starts after that so this presentation is stable.)

## 6. Data available to the UI (read-only contract, Phase 14A)

The UI consumes a **sanitized** valuation response (subject in → result out), never
the engine internals or raw DB rows. Available fields: `indicated_value`,
`indicated_range_low/high`, `direct_weighted_median_price`,
`weighted_median_ppsf`, `median_similarity`, `price_spread`, `confidence`,
`confidence_reasons`, `tier`, `tier_widened`, `warnings`, `sold_date_range`, and
per comparable: masked MLS, area_code, neighbourhood, property_type, sold_date,
sold_price, living_area_sqft, ppsf, size_normalized_price, similarity, component
subscores, data_coverage, differences, reasons, warnings.

## 7. Privacy & compliance (hard constraints)

- **Never display:** full street address, LINC, postal code,
  normalized_property_id, agent / brokerage / contact info, remarks, showing /
  lockbox instructions. MLS is **masked** (or shown only per an explicit, approved
  display policy).
- **Approved canonical data only**; read-only initially (no writes from the UI).
- **No appraisal claims** anywhere; "decision support" framing throughout.

## 8. Non-functional

Responsive (desktop-first, tablet-friendly), light/dark themes, fast perceived
performance, accessible (keyboard, contrast, semantics), and a professional,
brand-neutral visual system Fable can extend. No dependency on Streamlit.

## 9. Out of scope (for this brief)

Write-back / CRM sync, Model V2 outputs, production auth/SSO, and multi-user
collaboration — these are later phases. This brief covers the read-only comparable
valuation experience only.

## 10. Handoff to Fable

Fable produces the design system + high-fidelity mockups (all screens, light/dark,
responsive states, empty/loading/error/insufficient states) from this brief. The
engineering team wires the mockups to the Phase 14A read-only API. Build begins
after Phase 12 guardrails are merged.
