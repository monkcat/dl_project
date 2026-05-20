"""MMDocIR loader.

Source: https://huggingface.co/MMDocIR
Paper:  https://arxiv.org/abs/2501.08828

GT 단위: page-level relevance (1,658 expert-annotated QA on 313 long docs).
우리 매핑: page-hit rate; element 단위는 GT 페이지 안의 element를 retrieved으로 카운트.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "mmdocir"


def load(split: str = "test") -> list[dict]:
    """Return list of {query, doc_id, gt_pages: [int], ...}."""
    raise NotImplementedError
