"""Paired bootstrap for ablation row vs row comparisons (1000 resamples)."""
from __future__ import annotations


def paired_bootstrap_ci(scores_a: list[float], scores_b: list[float],
                        n_resample: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Return (mean_diff, lo, hi) for scores_a - scores_b."""
    raise NotImplementedError("Use scipy.stats.bootstrap or numpy resample.")
