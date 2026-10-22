"""Stable ID utilities for Qdrant (avoids Python's randomised hash())."""

from __future__ import annotations

import hashlib


def stable_qdrant_id(doc_id: str | int) -> int:
    """Convert any doc_id to a deterministic Qdrant-compatible integer.

    Tries int cast first (works for numeric BEIR IDs like '1234'),
    then falls back to MD5 truncated to 63 bits.
    """
    if isinstance(doc_id, int):
        return doc_id
    try:
        return int(doc_id)
    except ValueError:
        return int(hashlib.md5(doc_id.encode()).hexdigest()[:16], 16) % (2**63)
