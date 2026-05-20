"""B4: VisRAG page-level retrieval.

Repo: https://github.com/openbmb/visrag
"""
from __future__ import annotations


def build_index(page_images: list[str]):
    raise NotImplementedError


def search(index, query: str, k: int = 50):
    raise NotImplementedError
