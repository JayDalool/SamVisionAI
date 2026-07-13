# WRREB Report Data Contract (SamVisionAI v2 ingestion)

Status: MVP / canonical. Supersedes the legacy "Side By Side Plus" ingestion.

## 1. Permanent import package

Two WRREB **Matrix** reports form the permanent import package. Both are
exported together for the same batch of sold listings:

| Report | Role |
|---|---|
| **Residential Agent Single Line incl SP** | Compact batch index + cross-check source. One row per listing. |
| **Residential Client Full** | Authoritative detail source. One property per page. |

The **Agent Multi-Row** report is **no longer required** and is not part of the
contract.

The two reports are joined by **normalized MLS number** (`20` + 7 digits, ®/
spaces/punctuation stripped). For the validated sample batch the control totals
were **100 Single Line rows / 100 Client Full pages / 100 MLS matches / 0
duplicates / 0 orphans**. `100` is the sample expectation, **not** a general
business rule.

## 2. Canonical date rules (the correction that motivated v2)

- `sold_date` comes from the **real report value**, never from today, the
  report-generated/printed time, or the MLS number.
- **Client Full `Sell Date` is authoritative.** Agent Single Line `Date` is the
  same Sell Date and must normally match; a mismatch raises `SOLD_DATE_CONFLICT`
  and is never silently overwritten.
- `sale_year` and `sale_month` are **derived from `sold_date`** only.
- `listing_date` remains **NULL** unless a future approved report explicitly
  provides a real listing date.
- `mls_year_hint` (year embedded in the MLS number) **may be stored** but must
  **never replace a missing `sold_date`** and must never drive a temporal split.
- `imported_at` records ingestion time only.
- Chronological model splitting must use the real `sold_date`.

## 3. Field mapping

### Agent Single Line incl SP → canonical
| Report field | Canonical | Notes |
|---|---|---|
| `MLS® #` | `mls_number` | normalized |
| `S` | `status` | e.g. `S` = Sold |
| `Ar` (in "Ar Address") | `area_code` | leading token |
| `Address` | `address` | |
| `List Price` | `list_price` | |
| `Sold Price` | `sold_price` | |
| **`Date`** | **`sold_date`** | **real Sell Date** |
| `DOM` | `dom` | |
| `Ty` | `property_type_code` | |
| `Style` | `style_code` | |
| `YrBt` | `year_built` | |
| `SqFt` | `living_area_sqft` | |

### Client Full → canonical
| Report field | Canonical | Notes |
|---|---|---|
| `MLS® #` | `mls_number` | normalized |
| `Linc #` | `linc_number` | preferred property identity |
| address line | `address`, `postal_code` | |
| `Area` | `area_code` | |
| `Nghbrhd` | `neighbourhood` | |
| `Status` | `status` | |
| `List Price` | `list_price` | |
| `Sell Price` | `sold_price` | |
| **`Sell Date`** | **`sold_date`** | **authoritative** |
| `DOM` | `dom` | |
| `Type` | `property_type_code` | |
| `Style` | `style_code` | |
| `Yr Built/Age` | `year_built` | leading 4 digits |
| `Liv Area` | `living_area_sqft` | the `... SF` value |
| `BDA` / `TBD` | `bedrooms_above_grade` / `bedrooms_total` | |
| `Baths: F/H` | `full_bathrooms` / `half_bathrooms` | |
| `Lot Front` / `Lot Dpth` | `lot_front_ft` / `lot_depth_ft` | imperial `F` value |
| `Basement` | `basement_type` | |
| `Gross Tax` / `Tax Yr` | `gross_tax` / `tax_year` | |
| `New Const` | `new_construction` | |

## 4. Source authority & conflict resolution

- **Sold date:** Client Full authoritative; Single Line must match → else
  `SOLD_DATE_CONFLICT`.
- **Prices:** cross-checked; disagreement → `SOLD_PRICE_CONFLICT` /
  `LIST_PRICE_CONFLICT`.
- **LINC, neighbourhood, detailed characteristics:** Client Full authoritative.
- **Compact index / control counts:** Single Line.
- Conflicts are recorded, never silently overwritten. A blocking conflict marks
  the record `rejected` (not accepted for training).

## 5. Required vs optional (for a training candidate)

Required: `mls_number`, `sold_price`, `sold_date`, `property_type_code`, and
`address` **or** `linc_number`, plus positive `living_area_sqft` when present.
Optional → warnings (never zero/`none`/today defaults): lot dimensions, finished
basement, garage, taxes, room detail, renovations.

## 6. Identity & deduplication

- **Listing/transaction identity:** normalized `mls_number`.
- **Preferred property identity:** `linc_number`.
- **Fallback property identity:** normalized `address` + `postal_code`.
- Do **not** use `address + listing_date` or `address + sold_price` as canonical
  identity. Keep all legitimate historical/repeat sales — never collapse
  separate transactions for the same property.

## 7. Privacy & licensing

- Raw WRREB PDFs are **licensed board data**: never commit them, never use them
  as test fixtures. Store under `/srv/server/private/samvisionai/wrreb/incoming/`.
- Excluded from canonical output, model features, and logs: agent/brokerage
  names, phone/email, lockbox/showing/appointment instructions, and remarks/
  free-text. (Remarks may only ever live in restricted raw storage, never git,
  never logged in full.)
- See `docs/current_model_audit.md` follow-up for the pre-existing
  `pdf_uploads/*.pdf` committed-history remediation item.

## 8. Provenance & versioning

Every canonical row carries `source_batch_id`, `parser_version`, `source_page`
(via diagnostics), and each source file's `sha256`. Parser versions:
`single_line/1.0.0`, `client_full/1.0.0`.

## 9. Reason codes

`MISSING_CLIENT_FULL_RECORD`, `MISSING_SINGLE_LINE_RECORD`, `SOLD_DATE_CONFLICT`,
`SOLD_PRICE_CONFLICT`, `LIST_PRICE_CONFLICT`, `ADDRESS_CONFLICT`, `DUPLICATE_MLS`,
`INVALID_PRICE`, `INVALID_DATE`, `FUTURE_SOLD_DATE`, `MISSING_REQUIRED_FIELD`,
`OCR_UNVERIFIED_FIELD`, `NEEDS_OCR`, `OUT_OF_RANGE`.

## 10. Extraction reliability

Text-layer-first (pdfminer, coordinate-aware). These exports have a clean text
layer, so OCR is **not** used; a page whose text layer is empty/corrupt is
flagged `NEEDS_OCR` and never guessed. Cross-report reconciliation by MLS is the
primary safety control. Dry-run CLI writes only to the gitignored
`pipeline_output/<batch>/` and makes no database or model changes.
