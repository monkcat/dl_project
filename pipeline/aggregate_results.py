"""Aggregate per-experiment eval results into a single markdown summary.

Reads `eval/results/experiments/*/eval.json` for every row that completed.
Sections in output (REPORT_KR §7):
  §7.1 Main ablation (a-h, gme)              — Tier 1
  §7.2 Negative controls (m, n, o, p)        — Tier 2
  §7.3 Sub-ablation sweeps (γ, λ_cov, λ_cons, lora, edge) — Tier 3
  §7.4 Propagation α/T sweep (on (h))        — separate eval pass
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DATASET_ORDER = ["spiqa_testA", "sciegqa", "mmdocir"]
DATASET_LABEL = {
    "spiqa_testA": "SPIQA test-A",
    "sciegqa":     "SciEGQA",
    "mmdocir":     "MMDocIR",
}

# Tier grouping for the output
TIER1_ROWS = ["a", "b", "c", "d", "e", "f", "g", "h", "gme"]
TIER2_ROWS = ["m", "n", "o", "p"]
TIER3_ROWS = (
    ["gamma_03", "gamma_07"]
    + [f"cov_{int(l*10):02d}" for l in (0.0, 0.1, 0.5, 1.0)]
    + [f"cons_{int(l*10):02d}" for l in (0.0, 0.1, 0.3, 1.0)]
    + [f"lora_r{r}" for r in (4, 16, 32)]
    + [f"edge_{e}" for e in ("caption_of", "refer_to", "contains")]
    + [f"tokens_{n:03d}" for n in (16, 64)]
)

ROW_LABEL = {
    "a":   "(a) baseline InfoNCE",
    "b":   "(b) +GPE type only, InfoNCE",
    "c":   "(c) +GPE type+role, InfoNCE",
    "d":   "(d) +GPE full, InfoNCE",
    "e":   "(e) GRCL, no GPE",
    "f":   "(f) GRCL + GPE",
    "g":   "(g) GRCL + GPE + L_cov",
    "h":   "(h) Full method",
    "gme": "(k) GME zero-shot",
    "m":   "(m) shuffled section_role",
    "n":   "(n) random section_role",
    "o":   "(o) no query PE dropout",
    "p":   "(p) encoder swap CLIP-L/14",
}
for rid in TIER3_ROWS:
    ROW_LABEL.setdefault(rid, f"(sub) {rid}")

# Default inference variants per row
VARIANT_ORDER = ["no_prop", "prop_a03_T2"]
VARIANT_LABEL = {
    "no_prop":      "enc only",
    "prop_a03_T2":  "+prop α=.3 T=2",
    "prop_a01_T2":  "+prop α=.1 T=2",
    "prop_a05_T2":  "+prop α=.5 T=2",
    "prop_a07_T2":  "+prop α=.7 T=2",
    "prop_a03_T1":  "+prop α=.3 T=1",
    "prop_a03_T3":  "+prop α=.3 T=3",
}
PROP_SWEEP_VARIANTS = ["prop_a01_T2", "prop_a05_T2", "prop_a07_T2",
                       "prop_a03_T1", "prop_a03_T3"]

METRICS = ["recall@5", "recall@10", "mrr", "coverage@10", "perfect@10", "cross_page_hit"]
METRIC_LABEL = {
    "recall@5":       "R@5",
    "recall@10":      "R@10",
    "mrr":            "MRR",
    "coverage@10":    "Cov@10",
    "perfect@10":     "Pf@10",
    "cross_page_hit": "Xpg",
}


def load_results(exp_root: Path) -> dict[str, dict]:
    """Map row_id → per-dataset eval results.

    Looks for any subdir of exp_root with name <row_id>_*/eval.json.
    """
    out: dict[str, dict] = {}
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
        # Folder name is <row_id>_<run_name>. Recover row_id by matching against known set.
        name = run_dir.name
        # row_id is the part before the first "_" that matches a known prefix.
        row_id = name.split("_", 1)[0]
        # Handle prefixes that contain underscores (e.g., "lora_r4", "gamma_03", "edge_caption_of")
        for prefix in ROW_LABEL.keys():
            if name == prefix or name.startswith(prefix + "_") or name == prefix + "_" + ROW_LABEL.get(prefix, "").split()[0]:
                row_id = prefix
                break
        # Direct prefix match for multi-word row ids
        for prefix in TIER3_ROWS:
            if name.startswith(prefix):
                row_id = prefix
                break
        out[row_id] = data.get("results", {})
    return out


def fmt(v) -> str:
    if v is None:
        return "  —  "
    return f"{v*100:5.1f}"


def render_per_dataset_table(results: dict, rows: list[str], variants: list[str]) -> list[str]:
    lines = []
    for ds in DATASET_ORDER:
        lines.append(f"### {DATASET_LABEL[ds]}")
        lines.append("")
        header = "| Row | Variant | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " | n |"
        sep = "|" + "---|" * (3 + len(METRICS))
        lines.append(header)
        lines.append(sep)
        for row in rows:
            if row not in results:
                continue
            ds_data = results[row].get(ds, {})
            for var in variants:
                if var not in ds_data:
                    continue
                m = ds_data[var]
                cells = [fmt(m.get(metric)) for metric in METRICS]
                n_q = m.get("n_queries", "—")
                lines.append(
                    f"| {ROW_LABEL.get(row, row)} | {VARIANT_LABEL.get(var, var)} | "
                    + " | ".join(cells) + f" | {n_q} |"
                )
        lines.append("")
    return lines


def render_summary(results: dict) -> str:
    lines = []
    lines.append("# Experiment Results — v2.1 element graph ablation")
    lines.append("")
    n_done = len(results)
    lines.append(f"Loaded {n_done} completed runs.")
    lines.append("")

    # §7.1 Tier 1 — main
    lines.append("## §7.1 Main ablation (Tier 1, rows a-h, k=gme)")
    lines.append("")
    lines.extend(render_per_dataset_table(results, TIER1_ROWS, VARIANT_ORDER))

    # §7.2 Tier 2 — negative controls
    lines.append("## §7.2 Negative controls (Tier 2, rows m-p)")
    lines.append("")
    lines.extend(render_per_dataset_table(results, TIER2_ROWS, VARIANT_ORDER))

    # §7.3 Tier 3 — sub-ablation sweeps (group by family for readability)
    lines.append("## §7.3 Sub-ablation sweeps (Tier 3)")
    lines.append("")
    for label, sub in [
        ("γ sweep (GRCL 2-hop decay)",
         ["gamma_03", "h", "gamma_07"]),
        ("λ_cov sweep (coverage weight)",
         ["cov_00", "cov_01", "h", "cov_05", "cov_10"]),
        ("λ_cons sweep (consistency weight)",
         ["cons_00", "cons_01", "cons_03", "h", "cons_10"]),
        ("LoRA rank sweep",
         ["lora_r4", "h", "lora_r16", "lora_r32"]),
        ("Edge type isolation",
         ["edge_caption_of", "edge_refer_to", "edge_contains", "h"]),
        ("Token count per visual element",
         ["tokens_016", "tokens_064", "h"]),
    ]:
        lines.append(f"### {label}")
        lines.append("")
        lines.extend(render_per_dataset_table(results, sub, VARIANT_ORDER))

    # §7.4 Propagation α/T sweep (on (h) checkpoint, separate eval pass)
    if "h_prop_sweep" in results or any(
        v in (results.get("h", {}).get("spiqa_testA", {}) or {}) for v in PROP_SWEEP_VARIANTS
    ):
        lines.append("## §7.4 Propagation α/T sweep (on (h) checkpoint)")
        lines.append("")
        target = "h_prop_sweep" if "h_prop_sweep" in results else "h"
        # Show all PROP_SWEEP_VARIANTS for (h)
        lines.extend(render_per_dataset_table(results, [target], ["no_prop"] + PROP_SWEEP_VARIANTS))

    # Quick comparison summary
    lines.append("## Quick comparison: R@10 (no_prop) on every dataset")
    lines.append("")
    lines.append("| Row | " + " | ".join(DATASET_LABEL[d] for d in DATASET_ORDER) + " |")
    lines.append("|" + "---|" * (1 + len(DATASET_ORDER)))
    all_rows = TIER1_ROWS + TIER2_ROWS + TIER3_ROWS
    for row in all_rows:
        if row not in results:
            continue
        cells = []
        for ds in DATASET_ORDER:
            v = results[row].get(ds, {}).get("no_prop", {}).get("recall@10")
            cells.append(fmt(v))
        lines.append(f"| {ROW_LABEL.get(row, row)} | " + " | ".join(cells) + " |")
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
    print(f"Loaded {len(results)} runs: {sorted(results.keys())}")

    md = render_summary(results)
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\nWrote summary → {out_path}")


if __name__ == "__main__":
    main()
