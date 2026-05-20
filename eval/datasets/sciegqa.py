"""SciEGQA loader.

Source: https://yuwenhan07.github.io/SciEGQA-project/
Paper:  https://arxiv.org/abs/2511.15090

Each query has ground-truth evidence as bbox-level crops on specific pages.
Use bbox to compute element-IoU against our retrieved elements.

Expected layout under data/benchmarks/sciegqa/ (verify on download):
    annotations.json  — query, answer, evidence list with bbox + page
    pdfs/             — source PDFs
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "sciegqa"


def load(split: str = "test") -> list[dict]:
    """Return list of {query, answer, doc_id, evidence: [{page, bbox}], ...}."""
    raise NotImplementedError("Confirm format on download, then implement.")
