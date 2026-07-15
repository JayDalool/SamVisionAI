# SamVisionAI V2 Upgrade Tracker

## Document purpose

This file is the **source of truth** for the SamVisionAI V2 upgrade — the move
from the unreliable legacy pipeline (fabricated dates, weak identity, opaque model
lineage) to a verified **canonical-data → deterministic comparable-sales →
controlled-model** system. It records what was planned, what is done, what is in
progress, the architectural decisions, the validation evidence, the operational
risks, and the exact next task.

- **Last updated:** 2026-07-14
- **Current branch:** `feature/comparable-sales-backtest`
- **Current phase:** Phase 10 (leave-one-out backtesting) complete locally
- **Current status:** Phases 1–9 merged to `main`; Phase 10 complete on the
  feature branch (5 commits, not pushed); staging validated at 100 canonical.
- **Immediate next task:** Push, review, and merge
  `feature/comparable-sales-backtest` **without changing comparable weights**
  (Phase 11).

> **Update rule:** this document **must be updated at the end of every future
> phase, before the phase branch is pushed.** See *Update procedure* at the end.

## Status legend

- `[x]` Completed and merged
- `[✓ staging]` Completed and validated in non-production staging
- `[~]` Completed locally but not merged
- `[ ]` Not started
- `[!]` Operational risk or follow-up

## Original target architecture

```
WRREB reports
  → canonical parsing
  → reconciliation and validation
  → staging database
  → review and approval
  → canonical sold-sales dataset
  → deterministic comparable-sales engine
  → backtesting and calibration decisions
  → realtor review interface
  → candidate Model V2
  → benchmark and model registry
  → controlled production promotion
```

## Reference commits (verified against `origin/main`)

| Milestone | PR | Merge commit |
|---|---|---|
| Legacy training-data lineage audit | #3 | `40644ae` |
| Canonical ingestion pipeline | #4 | `99ca90c` |
| Canonical storage / staging | #5 | `e43069d` |
| Deterministic comparable-sales engine | #6 | `4cb7bd6` |
| Leave-one-out backtest (local, unmerged) | — | `f88e486 aad6618 c006c6b 5c8a82d e8bed22` |

Local `main` is currently behind `origin/main` (not fast-forwarded);
`origin/main` (`4cb7bd6`) is authoritative. `feature/comparable-sales-backtest`
is based on `origin/main` and has **not** been pushed.
`feature/auth-runtime-hardening` (`29499f4`, 4 commits) remains **local and
separate**; backup `backup/comparable-plus-auth-20260714` preserves the earlier
mixed history.

---

## Phase checklist

### PHASE 1 — Legacy audit — `[x]` (PR #3, `40644ae`)
- [x] inspect legacy parsers
- [x] identify fabricated dates
- [x] identify unreliable identity / deduplication
- [x] identify unreliable model lineage
- [x] pause retraining until verified data exists
- [x] preserve legacy production behaviour during rebuild

### PHASE 2 — WRREB report contract — `[x]` (PR #4, `b19bfc8`)
- [x] Residential Agent Single Line incl SP
- [x] Residential Client Full
- [x] Date / Sell Date cross-verification
- [x] normalized MLS join
- [x] MLS transaction identity
- [x] LINC preferred property identity
- [x] normalized address / postal fallback
- [x] Agent Multi-Row removed
- [x] explicit `contract_version`
- [x] explicit `parser_version`

### PHASE 3 — Canonical ingestion — `[x]` (PR #4, `99ca90c`)
- [x] coordinate-aware parsing
- [x] Agent Single Line parser
- [x] Client Full parser
- [x] reconciliation
- [x] validation
- [x] conflicts and reason codes
- [x] NEEDS_OCR handling
- [x] dry-run CLI
- [x] safe output files
- [x] no fabricated dates
- [x] synthetic tests
- [x] real 100 + 100 report validation → 100 matched, 100 accepted, 0 conflicts, 0 NEEDS_OCR

### PHASE 4 — Chronology and identity correction — `[x]` (PR #4, `6f9ed99`)
- [x] real `sold_date` as temporal truth
- [x] age calculated at `sold_date`
- [x] no use of today / import / model-training date for chronology
- [x] `listing_date` nullable
- [x] valid-MLS deduplication
- [x] missing-MLS rows preserved
- [x] MLS year only a diagnostic hint
- [x] address + sold_price identity removed

### PHASE 5 — Canonical schema and storage — `[x]` (PR #5, `e43069d`)
- [x] `import_batches` / `staging_sales` / `import_issues` / `canonical_sales`
- [x] PostgreSQL DATE `sold_date`
- [x] `NUMERIC(12,2)` money
- [x] nullable optional fields
- [x] foreign keys / unique constraints / check constraints / indexes
- [x] reversible migrations
- [x] explicit transaction ownership (`3f4ef34`)
- [x] PostgreSQL 17 up/down validation

### PHASE 6 — Safe staging loader — `[x]` (PR #5)
- [x] deterministic batch fingerprint
- [x] deterministic record fingerprint
- [x] explicit contract validation
- [x] duplicate batch prevention
- [x] duplicate MLS prevention
- [x] transactional writes / rollback on failure
- [x] explicit database URL
- [x] explicit staging-write authorization
- [x] no production credential inference
- [x] conflict and manifest refusal rules

### PHASE 7 — Real PostgreSQL staging validation — `[✓ staging]`
- [✓ staging] isolated PostgreSQL 17 container, localhost-only port binding
- [✓ staging] migration applied
- [✓ staging] one batch inserted, 100 staging rows, zero issues
- [✓ staging] duplicate import blocked
- [✓ staging] legacy `housing_data` untouched, no production connection

### PHASE 8 — Canonical promotion — `[✓ staging]`
- [✓ staging] promotion preview
- [✓ staging] explicit canonical authorization
- [✓ staging] 100 canonical rows inserted, approved batch status
- [✓ staging] staging rows preserved
- [✓ staging] MLS / fingerprint one-to-one validation
- [✓ staging] no duplicates or orphan records
- [✓ staging] repeat promotion safely blocked
- [✓ staging] approved-only dataset interface

**Current validated staging counts (read-only, 2026-07-14):**
`import_batches = 1` · `staging_sales = 100` · `import_issues = 0` ·
`canonical_sales = 100` · batch status **approved**.

### PHASE 9 — Deterministic comparable-sales engine — `[x]` (PR #6, `4cb7bd6`)
- [x] typed subject contract
- [x] eligibility rules
- [x] future-sale exclusion
- [x] subject MLS exclusion
- [x] approved-only canonical source
- [x] progressive geographic tiers
- [x] deterministic score
- [x] stable ranking
- [x] coverage score
- [x] Decimal-safe statistics / weighted median
- [x] size-normalized indicators / indicated range
- [x] rule-based confidence
- [x] insufficient-data behaviour
- [x] read-only CLI
- [x] PostgreSQL smoke validation
- [x] 45 comparable tests

### PHASE 10 — Leave-one-out backtesting — `[~]` (local, not merged)
- [~] 100 approved records evaluated
- [~] `valuation_date` equals held-out `sold_date`
- [~] held-out MLS excluded
- [~] future sales excluded
- [~] same-date limitation documented
- [~] read-only staging execution
- [~] metrics and grouped reports
- [~] privacy-safe failure reporting
- [~] no automatic weight tuning
- [~] 36 backtest tests

**Verified results (read-only staging, 74 valued of 100):**

*Records* — total approved 100 · eligible 100 · valued 74 · insufficient
comparables 26 · coverage **74.0%** · engine errors 0.

*Accuracy* — MAE **$92,981.96** · median absolute error **$63,000.00** ·
RMSE **$121,959.38** · MAPE **19.1%** · median APE **15.52%**.

*Thresholds* — within 5% **12.16%** · within 10% **33.78%** · within 15%
**47.30%** · within 20% **56.76%**.

*Observed behaviour* — higher similarity correlates with lower error; high
confidence outperforms medium and low; neighbourhood-tier outperforms
area-widened tier; pools of 4–5 comparables underperform larger pools; non-RD
property types are under-represented; size mismatch and wide price spread are the
major failure categories.

*Conclusion* — design behaviour is directionally valid; precision is **not**
strong enough for an appraisal-like single number; suitable for a **guarded
realtor decision-support pilot**; medium / low / widened / insufficient results
require manual review; more data and richer property fields are needed; **do not
automatically tune weights using only 100 sales**.

*Commits* — `f88e486` · `aad6618` · `c006c6b` · `5c8a82d` · `e8bed22`.

### PHASE 11 — Backtest PR — `[ ]` (immediate next task)
- [ ] final verification
- [ ] push `feature/comparable-sales-backtest`
- [ ] open PR
- [ ] review metrics and privacy behaviour
- [ ] merge into `main`

### PHASE 12 — Comparable guardrail improvements — `[ ]` (pending backtest review)
- [ ] stronger warning for tier-2 widening
- [ ] stronger warning or refusal below similarity 60
- [ ] stronger warning for wide price spread
- [ ] stronger handling for non-RD property types
- [ ] consider higher minimum comparable count
- [ ] do **not** automatically tune scoring weights
- [ ] repeat backtest after every rule change

> Recommendations in Phase 12 must be **reviewed before implementation**.

### PHASE 13 — Parser / data enrichment — `[ ]`
- [ ] garage fields
- [ ] parking fields
- [ ] basement development
- [ ] lot area and dimensions
- [ ] construction / new-build fields
- [ ] renovation / condition signals when legally and reliably available
- [ ] preserve provenance for every new field
- [ ] rerun parsing and completeness validation

> Missing optional fields currently cap data coverage near **0.833**
> (garage / lot / parking / basement-development are 100% NULL in the current
> canonical batch).

### PHASE 14 — Realtor-facing comparable review screen — `[ ]`
- [ ] feature-flagged Streamlit page
- [ ] subject property form
- [ ] comparable selection results
- [ ] similarity components / key differences
- [ ] sold price and price-per-square-foot
- [ ] indicated range
- [ ] confidence and warnings
- [ ] include / exclude controls
- [ ] realtor notes
- [ ] no appraisal claims
- [ ] approved canonical data only
- [ ] read-only initially
- [ ] suppress private operational data
- [ ] manual review required for medium / low / widened results

### PHASE 15 — Candidate Model V2 dataset — `[ ]`
- [ ] approved canonical data only
- [ ] real `sold_date` chronology
- [ ] chronological train / validation / test split
- [ ] prevent future leakage
- [ ] dataset fingerprint
- [ ] feature provenance
- [ ] missing-value policy
- [ ] no use of legacy unreliable training rows

### PHASE 16 — Candidate Model V2 benchmark — `[ ]`
- [ ] train candidate model
- [ ] preserve production model
- [ ] compare against transparent comparable baseline
- [ ] MAE / median APE / coverage / confidence calibration
- [ ] neighbourhood and property-type performance
- [ ] failure-case comparison
- [ ] realtor interpretability review

### PHASE 17 — Model registry and promotion — `[ ]`
- [ ] candidate registry
- [ ] dataset fingerprint / configuration fingerprint / Git commit / metrics
- [ ] approval state
- [ ] production / candidate comparison
- [ ] explicit promotion authorization
- [ ] rollback support
- [ ] never overwrite `trained_price_model.pkl` silently

### PHASE 18 — Production integration — `[ ]`
- [ ] canonical dataset production configuration
- [ ] approved comparable UI
- [ ] approved candidate model
- [ ] health check / monitoring
- [ ] session / authentication validation
- [ ] Cloudflare validation
- [ ] release checklist / rollback checklist
- [ ] controlled deployment
- [ ] post-deployment smoke tests

---

## Parallel operational checklist

**Authentication / runtime**
- [x] live authentication currently works
- [x] Streamlit global development-mode runtime issue fixed
- [~] auth/runtime changes exist on local `feature/auth-runtime-hardening`
- [ ] review auth branch
- [ ] push separate auth branch
- [ ] open separate PR
- [ ] add application health check
- [ ] confirm permanent environment configuration without committing secrets

**Cloudflare / security**
- [!] Cloudflare tunnel token appeared in terminal output
- [ ] rotate tunnel token
- [ ] store replacement privately
- [ ] confirm old token is invalid
- [ ] monitor tunnel errors
- [!] historical Docker `exitCode=139` event must remain documented
- [ ] continue restart monitoring

**Environment**
- [!] host `.venv` uses Python 3.14 while application Docker uses Python 3.11
- [ ] standardize local development / testing on Python 3.11
- [ ] recreate `.venv` with Python 3.11
- [ ] install exact requirements
- [ ] rerun complete suite
- [ ] confirm Streamlit AppTests pass in supported environment

**Privacy**
- [x] new PDFs and pipeline output ignored
- [x] staging credentials remain outside Git
- [!] old licensed WRREB PDFs remain in shared Git history
- [ ] prepare coordinated remediation plan
- [ ] do **not** rewrite shared history without approval

---

## Current tests (verified 2026-07-14)

| Suite | Result |
|---|---|
| backtesting | 36 / 36 passing |
| comparables | 45 / 45 passing |
| storage | 31 / 31 passing |
| ingestion | 17 / 17 passing |
| migration | 6 / 6 passing |
| **full host suite** | **163 total, 157 passing, 6 environment-related Streamlit failures** |

The 6 failures are Streamlit `AppTest` cases that import the full app body and
require compiled dependencies (`sklearn`, `lightgbm`) with no cp314 wheels in the
Python 3.14 host venv. They pass in the Python 3.11 Docker runtime and are **not**
comparable/backtest regressions.

## Current source-of-truth status

- Core phases 1–9 are **merged**.
- Phase 10 is **complete locally**.
- Phase 11 is the **immediate next task**.
- **No model training should begin yet.**
- **No production deployment should begin yet.**

## Exact next action

> **Push, review, and merge `feature/comparable-sales-backtest` without changing
> the comparable weights.**

## Update procedure

At the end of **every** future phase:

1. Verify Git branch and commit state.
2. Run the relevant tests.
3. Record staging / production safety results.
4. Update this tracker.
5. Add commit and PR references.
6. Record metrics and decisions.
7. Mark the next task.
8. Confirm no secrets or private data were added.
9. Commit the tracker update **before** pushing the phase branch.
