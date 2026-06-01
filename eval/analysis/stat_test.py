"""Paired bootstrap significance tests for ablation row-vs-row comparisons.

`paired_bootstrap_ci` is a standalone utility. Run as a script to compute CIs
for key comparisons (e.g. h vs a) from the per-query files written by
`eval_full --dump_per_query` (eval/results/experiments/<row>/eval.perquery.json).

    python -m eval.analysis.stat_test \
        --exp_root eval/results/experiments \
        --metric coverage@10 --variant no_prop \
        --compare h:a h:e f:e --out eval/results/figures/STAT_TESTS.md
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    n_resample: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Bootstrap the mean of (a - b) over paired per-query scores.

    Returns (mean_diff, lo, hi, p_two_sided). `p` is the bootstrap fraction of
    resamples whose mean diff crosses zero, doubled (two-sided) — a bootstrap
    analogue of a paired test. Requires len(a) == len(b) (same queries, order).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"unpaired lengths: {len(scores_a)} != {len(scores_b)}")
    n = len(scores_a)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n
    rng = random.Random(seed)
    boot = []
    for _ in range(n_resample):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boot.append(s / n)
    boot.sort()
    lo = boot[int((1 - ci) / 2 * n_resample)]
    hi = boot[int((1 - (1 - ci) / 2) * n_resample) - 1]
    if mean_diff >= 0:
        p = sum(1 for b in boot if b <= 0) / n_resample
    else:
        p = sum(1 for b in boot if b >= 0) / n_resample
    return mean_diff, lo, hi, min(1.0, 2 * p)


def _load_perquery(exp_root: Path, row: str) -> dict | None:
    """Find <row>'s eval.perquery.json (dir name starts with the row id)."""
    for d in sorted(exp_root.glob(f"{row}_*/")) + [exp_root / row]:
        pq = d / "eval.perquery.json"
        if pq.exists():
            try:
                return json.loads(pq.read_text())
            except Exception:
                return None
    return None


def _series(pq: dict, dataset_name: str, variant: str, metric: str):
    ds = pq.get(dataset_name)
    if not ds or variant not in ds:
        return None
    return ds[variant].get(metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="eval/results/experiments")
    ap.add_argument("--metric", default="coverage@10")
    ap.add_argument("--variant", default="no_prop")
    ap.add_argument("--compare", nargs="+", default=["h:a", "h:e", "f:e", "g:f"],
                    help="row pairs 'A:B' → tests mean(A)-mean(B)")
    ap.add_argument("--out", default="eval/results/figures/STAT_TESTS.md")
    args = ap.parse_args()

    exp_root = (REPO / args.exp_root) if not Path(args.exp_root).is_absolute() else Path(args.exp_root)
    out = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Paired-bootstrap significance tests (auto-generated)\n",
             f"metric=`{args.metric}`  variant=`{args.variant}`  (1000 resamples, 95% CI)\n"]
    any_data = False
    for pair in args.compare:
        if ":" not in pair:
            continue
        ra, rb = pair.split(":", 1)
        pqa, pqb = _load_perquery(exp_root, ra), _load_perquery(exp_root, rb)
        if not pqa or not pqb:
            lines.append(f"- **{ra} vs {rb}**: per-query data missing (skipped)")
            continue
        datasets = sorted(set(pqa) & set(pqb))
        lines.append(f"\n## {ra} − {rb}\n")
        lines.append("| dataset | Δmean | 95% CI | p | sig |")
        lines.append("|---|---|---|---|---|")
        for ds in datasets:
            sa = _series(pqa, ds, args.variant, args.metric)
            sb = _series(pqb, ds, args.variant, args.metric)
            if not sa or not sb or len(sa) != len(sb):
                lines.append(f"| {ds} | — | unpaired/missing | — | |")
                continue
            md, lo, hi, p = paired_bootstrap_ci(sa, sb)
            sig = "✓" if (lo > 0 or hi < 0) else ""
            lines.append(f"| {ds} | {md:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f} | {sig} |")
            any_data = True

    if not any_data:
        lines.append("\n_No per-query files found. Re-run eval_full with "
                     "`--dump_per_query` to enable these tests._")
    out.write_text("\n".join(lines) + "\n")
    print(f"[stat_test] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
