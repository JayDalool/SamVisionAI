"""Regression tests for the WRREB canonical-storage migration files.

The migration files must NOT manage their own transactions (no top-level
BEGIN/COMMIT) — `psql -1` is the single transaction owner. These checks operate
on comment-stripped SQL statements, not on arbitrary comment text, so a mention
of "begin"/"commit" in a comment does not trip them.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parents[1] / "db" / "migrations"
UP = MIGRATIONS / "0001_wrreb_canonical_storage.up.sql"
DOWN = MIGRATIONS / "0001_wrreb_canonical_storage.down.sql"

EXPECTED_TABLES = {
    "import_batches", "staging_sales", "import_issues", "canonical_sales",
}
_TXN_KEYWORDS = {"BEGIN", "COMMIT", "ROLLBACK", "START", "END"}


def _statements(sql_path: Path) -> list[str]:
    """Return non-empty, comment-stripped SQL statements (split on ';')."""
    lines = []
    for line in sql_path.read_text().splitlines():
        # Drop full-line and trailing `--` comments (no string literals use `--`
        # in these migrations).
        code = line.split("--", 1)[0]
        lines.append(code)
    text = "\n".join(lines)
    return [s.strip() for s in text.split(";") if s.strip()]


def _first_word(statement: str) -> str:
    m = re.match(r"[A-Za-z]+", statement)
    return m.group(0).upper() if m else ""


class MigrationTransactionOwnershipTests(unittest.TestCase):
    def test_files_exist_and_are_paired(self):
        self.assertTrue(UP.is_file(), f"missing {UP.name}")
        self.assertTrue(DOWN.is_file(), f"missing {DOWN.name}")

    def test_up_has_no_top_level_transaction_control(self):
        for stmt in _statements(UP):
            self.assertNotIn(
                _first_word(stmt), _TXN_KEYWORDS,
                msg=f"up migration must not manage transactions: {stmt[:40]!r}",
            )

    def test_down_has_no_top_level_transaction_control(self):
        for stmt in _statements(DOWN):
            self.assertNotIn(
                _first_word(stmt), _TXN_KEYWORDS,
                msg=f"down migration must not manage transactions: {stmt[:40]!r}",
            )

    def test_up_creates_all_expected_tables(self):
        creates = {
            m.group(1)
            for stmt in _statements(UP)
            for m in [re.match(
                r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+([a-z_]+)",
                stmt, re.IGNORECASE)]
            if m
        }
        self.assertEqual(creates, EXPECTED_TABLES)

    def test_down_drops_all_expected_tables(self):
        drops = {
            m.group(1)
            for stmt in _statements(DOWN)
            for m in [re.match(
                r"DROP\s+TABLE(?:\s+IF\s+EXISTS)?\s+([a-z_]+)",
                stmt, re.IGNORECASE)]
            if m
        }
        self.assertEqual(drops, EXPECTED_TABLES)

    def test_migrations_are_reversible(self):
        # Every table the up migration creates is dropped by the down migration.
        creates = {
            m.group(1)
            for stmt in _statements(UP)
            for m in [re.match(
                r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+([a-z_]+)",
                stmt, re.IGNORECASE)]
            if m
        }
        drops = {
            m.group(1)
            for stmt in _statements(DOWN)
            for m in [re.match(
                r"DROP\s+TABLE(?:\s+IF\s+EXISTS)?\s+([a-z_]+)",
                stmt, re.IGNORECASE)]
            if m
        }
        self.assertEqual(creates, drops)


if __name__ == "__main__":
    unittest.main()
