# Database migrations

Hand-written, reviewable SQL migrations for SamVisionAI. There is intentionally
**no** migration runner and **no** automatic application: the codebase does not
use Alembic/Flyway, and migrations must never run at application startup or
against production without a human.

## Layout

Each migration is a pair, numbered and reversible:

- `NNNN_<name>.up.sql`   — forward migration (additive, transactional)
- `NNNN_<name>.down.sql` — exact rollback, reverse dependency order

## Applying (manual, non-production first)

```
# Review the SQL, then apply to a NON-production database you control:
psql "$STAGING_DATABASE_URL" -1 -f db/migrations/0001_wrreb_canonical_storage.up.sql

# Roll back:
psql "$STAGING_DATABASE_URL" -1 -f db/migrations/0001_wrreb_canonical_storage.down.sql
```

`-1` runs the file in a single transaction (the files also wrap themselves in
`BEGIN/COMMIT`). Never point these at the production database as part of an
automated deploy.

## Migrations

| # | Name | Adds |
|---|------|------|
| 0001 | `wrreb_canonical_storage` | `import_batches`, `staging_sales`, `import_issues`, `canonical_sales` |

The SQLAlchemy Core table definitions in `samvision/storage/models.py` mirror
these migrations and are the schema used by tests (on SQLite) and by the
repository layer at runtime. **Keep the two in sync**: if you change a table,
update both the migration pair and `models.py`.
