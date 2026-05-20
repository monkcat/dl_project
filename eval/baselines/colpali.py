"""B3: ColPali page-level retrieval.

Repo: https://huggingface.co/vidore/colpali-v1.2
Library: pip install colpali-engine
"""
from __future__ import annotations


def build_index(page_images: list[str]):
    raise NotImplementedError


def search(index, query: str, k: int = 50):
    raise NotImplementedError
