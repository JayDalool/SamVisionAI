-- Migration: 0001_wrreb_canonical_storage (DOWN / rollback)
-- Drops exactly what 0001_*.up.sql created, in reverse dependency order.
-- Review before running. Dropping these tables discards any staged/canonical
-- rows they hold, so take a backup first if the tables are non-empty.
--
-- Only these four tables are removed; no existing table (e.g. housing_data) is
-- touched.
--
-- This file intentionally does NOT open its own transaction. Apply it with
-- `psql -v ON_ERROR_STOP=1 -1` so psql is the single transaction owner and the
-- whole rollback is atomic (all-or-nothing).

DROP TABLE IF EXISTS canonical_sales;
DROP TABLE IF EXISTS import_issues;
DROP TABLE IF EXISTS staging_sales;
DROP TABLE IF EXISTS import_batches;
