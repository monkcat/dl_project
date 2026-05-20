"""SPIQA loader.

Source: https://huggingface.co/datasets/google/spiqa
Paper:  https://arxiv.org/abs/2407.09413

270K+ QA pairs from arXiv CS papers (2018-2023). Test-A/B/C splits.
GT: answer + relevant figure id. Use for retrieval recall + downstream QA.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "spiqa"


def load(split: str = "test-A") -> list[dict]:
    """Return list of {query, answer, doc_id, gt_figure_ids, ...}."""
    raise NotImplementedError
