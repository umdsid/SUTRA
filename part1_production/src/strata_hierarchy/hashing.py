"""Small deterministic hashing helpers for hierarchy provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


__all__ = ["sha256_file"]
