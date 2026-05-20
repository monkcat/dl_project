"""Generation metrics: EM, F1, LLM-as-judge."""
from __future__ import annotations

import re
import string


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred: str, gt: str) -> float:
    return 1.0 if normalize(pred) == normalize(gt) else 0.0


def f1_score(pred: str, gt: str) -> float:
    p_tokens = normalize(pred).split()
    g_tokens = normalize(gt).split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    p = len(common) / len(p_tokens)
    r = len(common) / len(g_tokens)
    return 2 * p * r / (p + r)


JUDGE_PROMPT = """\
Question: {q}
Reference answer: {gt}
Predicted answer: {pred}
Is the predicted answer factually correct given the reference?
Answer with one word: yes / no / partial.
"""


def llm_judge_score(verdict: str) -> float:
    v = verdict.strip().lower()
    if v.startswith("yes"):
        return 1.0
    if v.startswith("partial"):
        return 0.5
    return 0.0


def llm_judge(query: str, gt: str, pred: str, model: str = "claude-sonnet-4-6") -> float:
    """Call judge LLM, return score ∈ {0, 0.5, 1}."""
    raise NotImplementedError("Wire up Anthropic / OpenAI client here.")
