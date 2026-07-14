"""WRREB canonical storage layer.

Turns dry-run pipeline output (see ``samvision.ingestion``) into reviewable
staging records and, once approved, canonical sales. Nothing in this package
connects to a database implicitly: every entry point requires an explicit
database URL, and writes require an explicit authorization flag. Production
credentials from ``SAMVISION_DB_*`` are never read here.
"""
