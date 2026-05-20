"""B1: BM25 over text-only chunks. `rank_bm25` library."""
from __future__ import annotations


def build_index(text_elements: list[dict]):
    """Build BM25Okapi index over tokenized text."""
    raise NotImplementedError


def search(index, query: str, k: int = 50) -> list[tuple[str, float]]:
    """Return [(element_id, score), ...]."""
    raise NotImplementedError
