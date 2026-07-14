# Database migrations

Hand-written, reviewable SQL migrations for SamVisionAI. There is intentionally
**no** migration runner and **no** automatic application: the codebase does not
use Alembic/Flyway, and migrations must never run at application startup or
against production without a human.

## Layout

Each migration is a pair, numbered and reversible:

- `NNNN_<name>.up.sql`   — forward migration (additive)
- `NNNN_<name>.down.sql` — exact rollback, reverse dependency order

The migration files **intentionally do not self-manage transactions** — they
contain no top-level `BEGIN`/`COMMIT`. `psql -1` is the single transaction
owner, which keeps ownership unambiguous and avoids "there is already a
transaction in progress" warnings from a nested `BEGIN`.

## Applying (manual, non-production first)

```
# Review the SQL, then apply to a NON-production database you control.
# -v ON_ERROR_STOP=1 aborts on the first error; -1 wraps the whole file in one
# transaction (all-or-nothing):
psql "$STAGING_DATABASE_URL" -v ON_ERROR_STOP=1 -1 -f db/migrations/0001_wrreb_canonical_storage.up.sql

# Roll back:
psql "$STAGING_DATABASE_URL" -v ON_ERROR_STOP=1 -1 -f db/migrations/0001_wrreb_canonical_storage.down.sql
```

Always pass `-1` (never rely on the file to open a transaction). Never point
these at the production database as part of an automated deploy.

## Migrations

| # | Name | Adds |
|---|------|------|
| 0001 | `wrreb_canonical_storage` | `import_batches`, `staging_sales`, `import_issues`, `canonical_sales` |

The SQLAlchemy Core table definitions in `samvision/storage/models.py` mirror
these migrations and are the schema used by tests (on SQLite) and by the
repository layer at runtime. **Keep the two in sync**: if you change a table,
update both the migration pair and `models.py`.
