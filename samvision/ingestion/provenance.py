"""File provenance helpers: content hashes and batch identifiers.

Hashes let us prove which physical report a canonical row came from without ever
storing the licensed PDF in the repo.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path


def sha256_file(path: str | Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def make_batch_id(single_line: str | Path, client_full: str | Path) -> str:
    """Deterministic batch id from the two source hashes + a timestamp prefix."""
    combined = (sha256_file(single_line) + sha256_file(client_full)).encode()
    digest = hashlib.sha256(combined).hexdigest()[:12]
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{digest}"
