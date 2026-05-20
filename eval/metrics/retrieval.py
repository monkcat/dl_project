"""Retrieval metrics: Recall@k, MRR, NDCG@k.

All metrics operate at the element level. For page-level baselines, the caller
should expand a retrieved page into its constituent element ids before calling.
"""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    hits = sum(1 for e in retrieved[:k] if e in gt)
    return hits / len(gt)


def mrr(retrieved: list[str], gt: set[str]) -> float:
    for i, e in enumerate(retrieved, start=1):
        if e in gt:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevance_map: dict[str, int], k: int) -> float:
    """relevance_map: element_id → grade ∈ {0, 1, 2}."""
    dcg = 0.0
    for i, e in enumerate(retrieved[:k], start=1):
        rel = relevance_map.get(e, 0)
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    ideal_grades = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal_grades, start=1))
    return dcg / idcg if idcg > 0 else 0.0
