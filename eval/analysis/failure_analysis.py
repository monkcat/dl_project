"""Failure case classifier — 60 sample inspection (20 per dataset).

Categories:
  extraction_fail   — MinerU 잘못 추출
  edge_fail         — caption_of/references 누락 또는 오결합
  retrieval_fail    — top-50 안에 GT 없음
  rerank_fail       — top-50에 있었으나 score 낮음
  expansion_miss    — re-rank 후 GT가 pack에 안 들어감 (token budget / graph 끊김)
  generation_fail   — pack에 GT 있는데 VLM이 잘못 답함
"""
from __future__ import annotations


CATEGORIES = [
    "extraction_fail",
    "edge_fail",
    "retrieval_fail",
    "rerank_fail",
    "expansion_miss",
    "generation_fail",
]


def classify_failure(query_record: dict) -> str:
    """Walk through the pipeline trace to classify why this query failed."""
    raise NotImplementedError
