"""Aggregate per-row eval results into a single markdown summary.

Reads `eval/results/experiments/*/eval.json` for every row that completed.

Output sections (mapping to REPORT_KR §7):
    §7.1  Tier 1 — main ablation (a-h, gme)
    §7.2  Tier 2 — negative controls (m, n, o, p)
    §7.3  Tier 3 — sub-ablation sweeps (γ, λ_cov, λ_cons, lora, lr, τ, anchor,
                                          edge type, visual tokens)
    §7.4  Tier 4 — follow-up extensions (best-combo, GPE-strong, seeds)
    §7.5  Propagation sub-ablation on (h) — 3 groups (uniform / weighted / PPR)
    §7.6  Quick comparison table (R@10 no-prop on every dataset)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────────
# Import tier definitions from configs.py (single source of truth)
# ──────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(REPO))
from pipeline.configs import (
    TIER1_ROWS, TIER1_NO_TRAIN, TIER2_ROWS, TIER3_ROWS, TIER4_ROWS,
    INFERENCE_VARIANTS,
    VARIANTS_GROUP_A_UNIFORM, VARIANTS_GROUP_B_WEIGHTS, VARIANTS_GROUP_C_PPR,
    DEFAULT_VARIANTS,
)


DATASET_ORDER = ["spiqa_testA", "sciegqa", "mmdocir"]
DATASET_LABEL = {
    "spiqa_testA": "SPIQA test-A",
    "sciegqa":     "SciEGQA",
    "mmdocir":     "MMDocIR",
}

# Human-readable row labels (auto-fallback to row_id for any not listed)
ROW_LABEL = {
    # Tier 1
    "a":   "(a) baseline InfoNCE",
    "b":   "(b) +GPE type only, InfoNCE",
    "c":   "(c) +GPE type+role, InfoNCE",
    "d":   "(d) +GPE full, InfoNCE",
    "e":   "(e) GRCL, no GPE",
    "f":   "(f) GRCL + GPE",
    "g":   "(g) GRCL + GPE + L_cov",
    "h":   "(h) Full method",
    "gme": "(k) GME zero-shot",
    # Tier 2
    "m":   "(m) shuffled section_role",
    "n":   "(n) random section_role",
    "o":   "(o) no query PE dropout",
    "p":   "(p) encoder swap CLIP-L/14",
    # Tier 4
    "h_best_combo":     "h + best-HP combo",
    "h_best_combo_16k": "h + best-HP combo, 16k steps",
    "h_gpe_strong":     "h + stronger GPE init",
    "h_seed_43":        "h (seed=43)",
    "h_seed_44":        "h (seed=44)",
}
for rid in TIER3_ROWS:
    ROW_LABEL.setdefault(rid, f"(sub) {rid}")

# Default inference variants shown in main per-row tables
VARIANT_ORDER = list(DEFAULT_VARIANTS)

# Auto-build variant labels from configs (so they stay in sync)
def _variant_label(name: str) -> str:
    cfg = INFERENCE_VARIANTS.get(name, {})
    method = cfg.get("method", "none")
    if method == "none":
        return "enc only"
    weights = cfg.get("weights", "full")
    alpha = cfg.get("alpha", 0.0)
    if method == "ppr":
        wsuf = " (uniform)" if weights == "uniform" else ""
        return f"PPR α={alpha:.2f}{wsuf}"
    # diffusion
    T = cfg.get("T", 2)
    return f"{weights} α={alpha:.1f} T={T}"


VARIANT_LABEL = {v: _variant_label(v) for v in INFERENCE_VARIANTS}

METRICS = ["recall@5", "recall@10", "mrr", "coverage@10", "perfect@10", "cross_page_hit"]
METRIC_LABEL = {
    "recall@5":       "R@5",
    "recall@10":      "R@10",
    "mrr":            "MRR",
    "coverage@10":    "Cov@10",
    "perfect@10":     "Pf@10",
    "cross_page_hit": "Xpg",
}


# ──────────────────────────────────────────────────────────────────────────
# Load eval results from eval/results/experiments/*/eval.json
# ──────────────────────────────────────────────────────────────────────────

def load_results(exp_root: Path) -> dict[str, dict]:
    """Map row_id → per-dataset eval results.

    Folder names are <row_id>_<run_name> (e.g. `h_full_method`); we recover
    row_id by trying every known prefix. Multi-token row ids (`lora_r4`,
    `edge_caption_of`) are handled by longest-prefix match.
    """
    out: dict[str, dict] = {}
    if not exp_root.exists():
        return out

    # All known row ids; longest first so `edge_caption_of` matches before `edge`
    known_rows = sorted(
        list(ROW_LABEL.keys()) + TIER3_ROWS + TIER4_ROWS + TIER1_NO_TRAIN + ["h_prop_sweep"],
        key=lambda x: -len(x),
    )

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

        name = run_dir.name
        row_id = None
        for prefix in known_rows:
            if name == prefix or name.startswith(prefix + "_"):
                row_id = prefix
                break
        if row_id is None:
            row_id = name.split("_", 1)[0]
        out[row_id] = data.get("results", {})

    return out


# ──────────────────────────────────────────────────────────────────────────
# Markdown rendering helpers
# ──────────────────────────────────────────────────────────────────────────

def fmt(v) -> str:
    if v is None:
        return "  —  "
    return f"{v*100:5.1f}"


def render_per_dataset_table(results: dict, rows: list[str], variants: list[str]) -> list[str]:
    lines = []
    for ds in DATASET_ORDER:
        lines.append(f"### {DATASET_LABEL[ds]}")
        lines.append("")
        lines.append("| Row | Variant | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " | n |")
        lines.append("|" + "---|" * (3 + len(METRICS)))
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
    lines.append("# Experiment Results — element-graph v2.1 ablation")
    lines.append("")
    n_done = len(results)
    lines.append(f"Loaded {n_done} completed runs.")
    lines.append("")

    # §7.1 Tier 1
    lines.append("## §7.1 Main ablation (Tier 1, rows a-h + k=gme)")
    lines.append("")
    lines.extend(render_per_dataset_table(results, TIER1_ROWS + TIER1_NO_TRAIN, VARIANT_ORDER))

    # §7.2 Tier 2
    lines.append("## §7.2 Negative controls (Tier 2, rows m-p)")
    lines.append("")
    lines.extend(render_per_dataset_table(results, TIER2_ROWS, VARIANT_ORDER))

    # §7.3 Tier 3 — grouped by hyperparameter family
    lines.append("## §7.3 Sub-ablation sweeps (Tier 3)")
    lines.append("")
    for label, sub in [
        ("γ sweep (GRCL 2-hop decay)",       ["gamma_03", "h", "gamma_07"]),
        ("λ_cov sweep",                       ["cov_00", "cov_01", "h", "cov_05", "cov_10"]),
        ("λ_cons sweep",                      ["cons_00", "cons_01", "cons_03", "h", "cons_10"]),
        ("LoRA rank sweep",                   ["lora_r4", "h", "lora_r16", "lora_r32"]),
        ("Learning rate sweep",               ["lr_1e5", "h", "lr_1e4"]),
        ("Temperature τ sweep",               ["tau_005", "h", "tau_010"]),
        ("Anchor kind sweep",                 ["anchor_caption", "anchor_refer", "anchor_nlqa", "h"]),
        ("Edge type isolation in GRCL",       ["edge_caption_of", "edge_refer_to", "edge_contains", "h"]),
        ("Visual tokens per element",         ["tokens_016", "tokens_064", "h"]),
    ]:
        lines.append(f"### {label}")
        lines.append("")
        lines.extend(render_per_dataset_table(results, sub, VARIANT_ORDER))

    # §7.4 Tier 4
    lines.append("## §7.4 Follow-up extensions (Tier 4)")
    lines.append("")
    lines.extend(render_per_dataset_table(results, TIER4_ROWS + ["h"], VARIANT_ORDER))

    # §7.5 Propagation sub-ablation (3 groups)
    has_prop_sweep = (
        "h_prop_sweep" in results
        or any(v in (results.get("h", {}).get("spiqa_testA", {}) or {})
               for v in VARIANTS_GROUP_A_UNIFORM + VARIANTS_GROUP_B_WEIGHTS + VARIANTS_GROUP_C_PPR)
    )
    if has_prop_sweep:
        target = "h_prop_sweep" if "h_prop_sweep" in results else "h"
        lines.append("## §7.5 Propagation sub-ablation on (h) — 3 regimes × hyperparam sweep")
        lines.append("")
        lines.append("All variants applied to the (h) checkpoint (eval-only).")
        lines.append("")
        lines.append("### Group A — Uniform weights (structure only, modifier-design ablation)")
        lines.append("")
        lines.extend(render_per_dataset_table(
            results, [target], ["no_prop"] + list(VARIANTS_GROUP_A_UNIFORM)))
        lines.append("### Group B — Weighted diffusion (modifier ablation + α/T sweep)")
        lines.append("")
        lines.extend(render_per_dataset_table(
            results, [target], ["no_prop"] + list(VARIANTS_GROUP_B_WEIGHTS)))
        lines.append("### Group C — Personalized PageRank")
        lines.append("")
        lines.extend(render_per_dataset_table(
            results, [target], ["no_prop"] + list(VARIANTS_GROUP_C_PPR)))

    # §7.6 Quick comparison
    lines.append("## §7.6 Quick comparison — R@10 (no propagation)")
    lines.append("")
    lines.append("| Row | " + " | ".join(DATASET_LABEL[d] for d in DATASET_ORDER) + " |")
    lines.append("|" + "---|" * (1 + len(DATASET_ORDER)))
    all_rows = TIER1_ROWS + TIER1_NO_TRAIN + TIER2_ROWS + TIER3_ROWS + TIER4_ROWS
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


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

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
