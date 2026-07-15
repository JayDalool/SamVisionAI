# SamVisionAI V2 — Architectural Decision Log

Concise, dated decision records for the V2 upgrade. Companion to
[`samvision_v2_upgrade_tracker.md`](samvision_v2_upgrade_tracker.md). No secrets
or private listing details appear here.

Each record: **Decision · Reason · Consequence · Revisit when**.

---

### ADR-01 — Use *Agent Single Line incl SP* + *Client Full* reports
- **Phase:** 2 (contract)
- **Decision:** Ingest exactly two WRREB Matrix reports per batch, joined by MLS.
- **Reason:** Together they provide a verified **Sell Date** and full property
  detail; the previous Agent Multi-Row report did not give a reliable sold date.
- **Consequence:** Two-file batches with cross-report verification; Agent
  Multi-Row retired.
- **Revisit when:** WRREB changes report formats or adds a more authoritative
  sold-date source.

### ADR-02 — MLS number is the transaction identity
- **Phase:** 2 / 4
- **Decision:** A sale is identified by its MLS number.
- **Reason:** MLS uniquely identifies a transaction; address/price collisions do not.
- **Consequence:** Deduplication and subject exclusion key on MLS.
- **Revisit when:** A dataset arrives without reliable MLS numbers.

### ADR-03 — LINC is the preferred property identity
- **Phase:** 2 / 4
- **Decision:** Property identity prefers LINC, then normalized address/postal.
- **Reason:** LINC is a stable land identifier; addresses vary in formatting.
- **Consequence:** `normalized_property_id` derived with LINC precedence.
- **Revisit when:** A jurisdiction lacks LINC coverage.

### ADR-04 — `sold_date` is the temporal truth
- **Phase:** 4
- **Decision:** All chronology uses the real `sold_date` only.
- **Reason:** Legacy pipelines fabricated/`listing_date`-based dates, causing
  leakage and wrong ages.
- **Consequence:** Eligibility, recency, age, and backtest valuation dates all use
  `sold_date`; never today / import / model-training date.
- **Revisit when:** A verified transaction timestamp finer than date is available.

### ADR-05 — Never fabricate missing dates
- **Phase:** 1 / 4
- **Decision:** Missing dates stay NULL; rows are flagged, never back-filled.
- **Reason:** Fabricated dates silently corrupt chronology and training.
- **Consequence:** Some rows are ineligible rather than wrong.
- **Revisit when:** Never (fabrication is prohibited).

### ADR-06 — Address + price is not an identity
- **Phase:** 4
- **Decision:** Remove address+sold_price as a dedup/identity key.
- **Reason:** Same address sells repeatedly; price is not unique — false merges.
- **Consequence:** Identity relies on MLS (transaction) and LINC (property).
- **Revisit when:** Never without a new unique key.

### ADR-07 — Keep legacy `housing_data` separate
- **Phase:** 1 / 5
- **Decision:** The canonical tables are independent of legacy `housing_data`.
- **Reason:** Preserve production behaviour during the rebuild; avoid contaminating
  verified data with legacy rows.
- **Consequence:** Parallel schemas; the legacy Streamlit path is untouched.
- **Revisit when:** Production fully migrates to canonical data.

### ADR-08 — Stage before canonical promotion
- **Phase:** 5 / 8
- **Decision:** Parsed rows land in `staging_sales`; promotion to `canonical_sales`
  is a separate authorized step.
- **Reason:** Review/approval must precede any row becoming decision data.
- **Consequence:** Two-step load with an approval gate.
- **Revisit when:** A fully automated, audited approval is trusted.

### ADR-09 — Require explicit database-write authorization
- **Phase:** 6
- **Decision:** Writes require an explicit URL and an explicit write-authorization
  flag; no production credential inference.
- **Reason:** Prevent accidental writes to the wrong database.
- **Consequence:** Read paths (dataset, comparables, backtest) hold no write flag.
- **Revisit when:** Never relaxed.

### ADR-10 — Deterministic batch and record fingerprints
- **Phase:** 6
- **Decision:** Every batch and record has a deterministic content fingerprint.
- **Reason:** Idempotency and duplicate detection without trusting filenames.
- **Consequence:** Re-importing an identical batch is safely blocked.
- **Revisit when:** The fingerprint input fields change (bump `contract_version`).

### ADR-11 — Canonical sales must be approved before use
- **Phase:** 8 / 9
- **Decision:** Only `data_quality_status = approved` rows feed comparables/models.
- **Reason:** Unreviewed rows must not influence valuations.
- **Consequence:** The dataset layer filters to approved by default.
- **Revisit when:** A finer quality taxonomy is introduced.

### ADR-12 — Comparable engine is the transparent baseline
- **Phase:** 9
- **Decision:** Valuation baseline is a deterministic, explainable comparable
  engine — **no ML** in scoring.
- **Reason:** Realtors need reproducible, defensible numbers; ML opacity failed before.
- **Consequence:** Fixed documented weights; every result is hand-reproducible.
- **Revisit when:** A model demonstrably beats it *and* stays interpretable (Phase 16).

### ADR-13 — No future-sale leakage
- **Phase:** 9 / 10
- **Decision:** Candidates must have `sold_date <= valuation_date`.
- **Reason:** Using future sales inflates apparent accuracy and is invalid.
- **Consequence:** Enforced in SQL upper bound and eligibility; backtest inherits it.
- **Revisit when:** Never relaxed.

### ADR-14 — Same-day sales allowed (no intraday ordering)
- **Phase:** 10
- **Decision:** The `sold_date <= valuation_date` bound is inclusive, so non-subject
  same-day sales are eligible.
- **Reason:** The dataset has no intraday timestamp; excluding same-day would also
  drop legitimately-prior evidence.
- **Consequence:** A small, documented optimism in backtest coverage/accuracy.
- **Revisit when:** An intraday transaction timestamp becomes available.

### ADR-15 — Do not auto-tune weights on 100 sales
- **Phase:** 10
- **Decision:** Backtest results inform recommendations only; weights are not
  automatically changed.
- **Reason:** 100 single-batch sales cannot support stable calibration; overfitting risk.
- **Consequence:** Weight changes stay human-reviewed; backtest re-runs after any rule change.
- **Revisit when:** A materially larger, multi-batch approved dataset exists.

### ADR-16 — Results are decision support, not an appraisal
- **Phase:** 9 / 10 / 14
- **Decision:** Output is framed as decision support; no appraisal claims.
- **Reason:** Legal/professional boundary; measured precision is not appraisal-grade.
- **Consequence:** UI and reports carry explicit disclaimers.
- **Revisit when:** Never removed; wording reviewed if regulations change.

### ADR-17 — Medium / low / widened results require manual review
- **Phase:** 10 / 14
- **Decision:** Only high-confidence, non-widened results are surfaced as usable;
  others are flagged for manual review.
- **Reason:** Backtest shows high confidence ≈ 2× better within-10% than low.
- **Consequence:** The pilot gates presentation by confidence and tier.
- **Revisit when:** Confidence calibration improves with more data.

### ADR-18 — Model V2 must beat or complement the comparable baseline
- **Phase:** 16
- **Decision:** A candidate model is adopted only if it beats or usefully
  complements the transparent baseline on held-out data.
- **Reason:** Avoid replacing an explainable baseline with an opaque, no-better model.
- **Consequence:** Benchmarking is a hard gate before any promotion.
- **Revisit when:** Baseline or model methodology changes.

### ADR-19 — Production model replacement requires registry, approval, rollback
- **Phase:** 17
- **Decision:** No model reaches production without a registry entry
  (dataset+config+commit fingerprints, metrics), explicit approval, and rollback.
- **Reason:** Prevent silent, unauditable model swaps.
- **Consequence:** `trained_price_model.pkl` is never overwritten silently.
- **Revisit when:** Never relaxed.

### ADR-20 — Private WRREB reports and runtime outputs are never committed
- **Phase:** all
- **Decision:** Licensed PDFs, parsed private rows, staging credentials, and
  backtest report artifacts stay out of Git.
- **Reason:** Licensing and privacy; outputs can contain listing data.
- **Consequence:** `.gitignore` covers them; reports write only to `/tmp` or a
  gitignored dir; a tracked-dir guard refuses report writes into source.
- **Revisit when:** Coordinated remediation of the historical PDFs is planned (with
  approval; do not rewrite shared history unilaterally).
