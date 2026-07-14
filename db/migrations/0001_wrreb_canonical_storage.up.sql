-- Migration: 0001_wrreb_canonical_storage (UP)
-- Creates the WRREB canonical staging + storage schema:
--   import_batches, staging_sales, import_issues, canonical_sales
--
-- REVIEW / APPLY MANUALLY against a NON-production database first. This file is
-- never applied automatically and never at application startup. It is additive
-- only (no changes to existing tables such as housing_data). PostgreSQL dialect.
--
-- This file intentionally does NOT open its own transaction. Apply it with
-- `psql -v ON_ERROR_STOP=1 -1` so psql is the single transaction owner and a
-- partial apply cannot leave a half-built schema behind.

-- ---------------------------------------------------------------------------
-- import_batches: one row per imported WRREB report pair.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS import_batches (
    id                    BIGSERIAL PRIMARY KEY,
    batch_uuid            TEXT        NOT NULL UNIQUE,
    batch_fingerprint     TEXT        NOT NULL UNIQUE,
    contract_version      TEXT        NOT NULL,
    parser_version        TEXT        NOT NULL,
    single_line_filename  TEXT        NOT NULL,
    single_line_sha256    TEXT        NOT NULL,
    client_full_filename  TEXT        NOT NULL,
    client_full_sha256    TEXT        NOT NULL,
    search_start_date     DATE,
    search_end_date       DATE,
    imported_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    imported_by           TEXT,
    status                TEXT        NOT NULL DEFAULT 'parsed'
        CHECK (status IN ('parsed', 'staged', 'review_required',
                          'approved', 'rejected', 'failed')),
    single_line_count     INTEGER     NOT NULL DEFAULT 0,
    client_full_count     INTEGER     NOT NULL DEFAULT 0,
    matched_count         INTEGER     NOT NULL DEFAULT 0,
    accepted_count        INTEGER     NOT NULL DEFAULT 0,
    rejected_count        INTEGER     NOT NULL DEFAULT 0,
    warning_count         INTEGER     NOT NULL DEFAULT 0,
    conflict_count        INTEGER     NOT NULL DEFAULT 0,
    needs_ocr_count       INTEGER     NOT NULL DEFAULT 0,
    approved_at           TIMESTAMPTZ,
    approved_by           TEXT,
    notes                 TEXT
);

-- ---------------------------------------------------------------------------
-- staging_sales: accepted candidate rows for one batch (pre-approval).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS staging_sales (
    id                       BIGSERIAL PRIMARY KEY,
    batch_id                 BIGINT   NOT NULL
        REFERENCES import_batches (id) ON DELETE CASCADE,
    mls_number               TEXT     NOT NULL,
    linc_number              TEXT,
    normalized_property_id   TEXT,
    address                  TEXT,
    postal_code              TEXT,
    area_code                TEXT,
    neighbourhood            TEXT,
    list_price               NUMERIC(12, 2),
    sold_price               NUMERIC(12, 2),
    sold_date                DATE,
    dom                      INTEGER,
    property_type            TEXT,
    style                    TEXT,
    year_built               INTEGER,
    living_area_sqft         INTEGER,
    bedrooms_above_grade     INTEGER,
    bedrooms_total           INTEGER,
    full_bathrooms           INTEGER,
    half_bathrooms           INTEGER,
    basement_type            TEXT,
    basement_development     TEXT,
    finished_basement_sqft   INTEGER,
    lot_front_ft             NUMERIC(8, 2),
    lot_depth_ft             NUMERIC(8, 2),
    lot_area_sqft            NUMERIC(12, 2),
    garage_type              TEXT,
    parking_description      TEXT,
    parking_spaces           INTEGER,
    new_construction         BOOLEAN,
    gross_tax                NUMERIC(12, 2),
    tax_year                 INTEGER,
    source_single_line_page  INTEGER,
    source_client_full_page  INTEGER,
    parser_version           TEXT,
    validation_status        TEXT     NOT NULL DEFAULT 'accepted'
        CHECK (validation_status IN ('accepted', 'warning')),
    record_fingerprint       TEXT     NOT NULL,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_staging_batch_mls UNIQUE (batch_id, mls_number)
);

CREATE INDEX IF NOT EXISTS ix_staging_sold_date          ON staging_sales (sold_date);
CREATE INDEX IF NOT EXISTS ix_staging_linc               ON staging_sales (linc_number);
CREATE INDEX IF NOT EXISTS ix_staging_normalized_prop_id ON staging_sales (normalized_property_id);
CREATE INDEX IF NOT EXISTS ix_staging_record_fingerprint ON staging_sales (record_fingerprint);

-- ---------------------------------------------------------------------------
-- import_issues: warnings / conflicts / rejections / needs-OCR for review.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS import_issues (
    id                 BIGSERIAL PRIMARY KEY,
    batch_id           BIGINT   NOT NULL
        REFERENCES import_batches (id) ON DELETE CASCADE,
    staging_sale_id    BIGINT
        REFERENCES staging_sales (id) ON DELETE CASCADE,
    mls_number         TEXT,
    severity           TEXT     NOT NULL
        CHECK (severity IN ('warning', 'conflict', 'rejection', 'needs_ocr')),
    reason_code        TEXT     NOT NULL,
    field_name         TEXT,
    single_line_value  TEXT,
    client_full_value  TEXT,
    message            TEXT,
    source_page        INTEGER,
    resolved           BOOLEAN  NOT NULL DEFAULT FALSE,
    resolution_notes   TEXT,
    resolved_at        TIMESTAMPTZ,
    resolved_by        TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_issues_batch_sev_resolved
    ON import_issues (batch_id, severity, resolved);

-- ---------------------------------------------------------------------------
-- canonical_sales: approved sales. Promotion target + read source.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS canonical_sales (
    id                       BIGSERIAL PRIMARY KEY,
    mls_number               TEXT     NOT NULL UNIQUE,
    linc_number              TEXT,
    normalized_property_id   TEXT,
    address                  TEXT,
    postal_code              TEXT,
    area_code                TEXT,
    neighbourhood            TEXT,
    list_price               NUMERIC(12, 2),
    sold_price               NUMERIC(12, 2),
    sold_date                DATE,
    dom                      INTEGER,
    property_type            TEXT,
    style                    TEXT,
    year_built               INTEGER,
    living_area_sqft         INTEGER,
    bedrooms_above_grade     INTEGER,
    bedrooms_total           INTEGER,
    full_bathrooms           INTEGER,
    half_bathrooms           INTEGER,
    basement_type            TEXT,
    basement_development     TEXT,
    finished_basement_sqft   INTEGER,
    lot_front_ft             NUMERIC(8, 2),
    lot_depth_ft             NUMERIC(8, 2),
    lot_area_sqft            NUMERIC(12, 2),
    garage_type              TEXT,
    parking_description      TEXT,
    parking_spaces           INTEGER,
    new_construction         BOOLEAN,
    gross_tax                NUMERIC(12, 2),
    tax_year                 INTEGER,
    source_batch_id          BIGINT
        REFERENCES import_batches (id) ON DELETE RESTRICT,
    parser_version           TEXT,
    record_fingerprint       TEXT     NOT NULL,
    data_quality_status      TEXT     NOT NULL DEFAULT 'approved'
        CHECK (data_quality_status IN ('approved', 'flagged')),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_canonical_sold_date          ON canonical_sales (sold_date);
CREATE INDEX IF NOT EXISTS ix_canonical_neighbourhood      ON canonical_sales (neighbourhood);
CREATE INDEX IF NOT EXISTS ix_canonical_area_code          ON canonical_sales (area_code);
CREATE INDEX IF NOT EXISTS ix_canonical_linc               ON canonical_sales (linc_number);
CREATE INDEX IF NOT EXISTS ix_canonical_normalized_prop_id ON canonical_sales (normalized_property_id);
CREATE INDEX IF NOT EXISTS ix_canonical_sold_price         ON canonical_sales (sold_price);
CREATE INDEX IF NOT EXISTS ix_canonical_property_type      ON canonical_sales (property_type);
