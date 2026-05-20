"""Aggregate per-experiment eval results into a single markdown table.

Reads `eval/results/experiments/*/eval.json` for every row that completed and
emits a comparison table for REPORT_KR.md §7 (Results).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ROW_ORDER = ["a", "e", "f", "h", "gme"]
DATASET_ORDER = ["spiqa_testA", "sciegqa", "mmdocir"]
DATASET_LABEL = {
    "spiqa_testA": "SPIQA test-A",
    "sciegqa":     "SciEGQA",
    "mmdocir":     "MMDocIR",
}
ROW_LABEL = {
    "a":   "(a) baseline InfoNCE",
    "e":   "(e) GRCL no GPE",
    "f":   "(f) GRCL + GPE",
    "h":   "(h) Full (GRCL + GPE + L_cov + L_cons)",
    "gme": "(gme) GME zero-shot",
}
VARIANT_ORDER = ["no_prop", "prop_a03_T2"]
VARIANT_LABEL = {
    "no_prop":    "encoder only",
    "prop_a03_T2": "+ propagation",
}
METRICS = ["recall@5", "recall@10", "mrr", "coverage@10", "perfect@10", "cross_page_hit"]
METRIC_LABEL = {
    "recall@5":       "R@5",
    "recall@10":      "R@10",
    "mrr":            "MRR",
    "coverage@10":    "Cov@10",
    "perfect@10":     "Pf@10",
    "cross_page_hit": "Xpg",
}


def load_results(exp_root: Path) -> dict:
    out = {}
    if not exp_root.exists():
        return out
    for run_dir in sorted(exp_root.iterdir()):
        if not run_dir.is_dir():
            continue
        eval_log = run_dir / "eval.json"
        if not eval_log.exists():
            continue
        try:
            data = json.loads(eval_log.read_text())
        except json.JSONDecodeError:
            continue
        row_id = run_dir.name.split("_", 1)[0]
        out[row_id] = data.get("results", {})
    return out


def fmt(v):
    if v is None:
        return "  —  "
    return f"{v*100:5.1f}"


def render_table(results: dict) -> str:
    lines = []
    lines.append("# Experiment Results — v2.1 element graph ablation")
    lines.append("")
    lines.append(f"Rows compared: {', '.join(ROW_ORDER)}")
    lines.append(f"Datasets: {', '.join(DATASET_LABEL[d] for d in DATASET_ORDER)}")
    lines.append(f"Variants: {', '.join(VARIANT_LABEL[v] for v in VARIANT_ORDER)}")
    lines.append("")

    for ds in DATASET_ORDER:
        lines.append(f"## {DATASET_LABEL[ds]}")
        lines.append("")
        header_metrics = "  ".join(f"{METRIC_LABEL[m]:>6s}" for m in METRICS)
        lines.append(f"| Row | Variant | {' | '.join(METRIC_LABEL[m] for m in METRICS)} | n |")
        lines.append("|" + "---|" * (3 + len(METRICS)))
        for row in ROW_ORDER:
            if row not in results:
                continue
            ds_data = results[row].get(ds, {})
            for var in VARIANT_ORDER:
                if var not in ds_data:
                    continue
                m = ds_data[var]
                cells = [fmt(m.get(metric)) for metric in METRICS]
                n_q = m.get("n_queries", "—")
                lines.append(
                    f"| {ROW_LABEL[row]} | {VARIANT_LABEL[var]} | "
                    + " | ".join(cells) + f" | {n_q} |"
                )
        lines.append("")

    # Summary takeaway
    lines.append("## Key comparisons (R@10, no propagation unless noted)")
    lines.append("")
    for ds in DATASET_ORDER:
        lines.append(f"### {DATASET_LABEL[ds]}")
        for row in ROW_ORDER:
            if row not in results:
                continue
            ds_data = results[row].get(ds, {})
            r_no = ds_data.get("no_prop", {}).get("recall@10")
            r_pr = ds_data.get("prop_a03_T2", {}).get("recall@10")
            lines.append(
                f"  - {ROW_LABEL[row]:<45s}  R@10 = "
                f"{fmt(r_no)} (no-prop)  /  {fmt(r_pr)} (+prop)"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", type=str, default="eval/results/experiments")
    ap.add_argument("--out", type=str, default="eval/results/experiments/SUMMARY.md")
    args = ap.parse_args()

    exp_root = REPO / args.exp_root
    print(f"Reading from {exp_root}")
    results = load_results(exp_root)
    print(f"Loaded {len(results)} runs: {list(results.keys())}")

    md = render_table(results)
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\nWrote summary → {out_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
