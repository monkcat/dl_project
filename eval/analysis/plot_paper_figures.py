"""Paper-grade figures — SPIQA-only, only the rows that appear in REPORT_KR_v3.

Reads `eval/results/figures/results_long.csv` (long-format aggregator output)
and emits paper figures into `eval/results/figures/paper/`.

Rows are restricted to the ones shown in §4.7 Summary of REPORT_KR_v3:
    (a) baseline InfoNCE
    (e) GRCL
    edge_refer_to    (GRCL with refer_to-only edges)
    lr_1e4           (lr=1e-4)
    h_best_combo     (lr=1e-4 + τ=0.10 + λ_cov=0.5)
    p (CLIP-L/14)    (backbone swap)
    GME zero-shot
    GME + propagation

Dataset: SPIQA test-A only. SciEGQA / MMDocIR are not shown.

Usage:
    python -m eval.analysis.plot_paper_figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent

# Rows that appear in the v3 paper, in display order (left to right).
# (row_id, display label, variant to read)
PAPER_ROWS = [
    ("a",                 "(a) InfoNCE baseline",        "no_prop"),
    ("e",                 "(e) GRCL",                    "no_prop"),
    ("edge_refer_to",     "GRCL\nrefer_to only",         "no_prop"),
    ("lr_1e4",            "GRCL\nlr=1e-4",               "no_prop"),
    ("h_best_combo",      "GRCL\nbest HP combo",         "no_prop"),
    ("p",                 "CLIP-L/14\nbackbone",         "no_prop"),
    ("gme",               "GME\nzero-shot",              "no_prop"),
    ("gme",               "GME +\ngraph propagation",    "wfull_a03_T2"),
]

# Three-regime propagation comparison on GME (§4.5)
PROP_REGIMES_ON_GME = [
    ("no_prop",          "no\npropagation"),
    ("uniform_a05_T2",   "(A) Uniform\nα=0.5, T=2"),
    ("wfull_a03_T2",     "(B) Weighted\nα=0.3, T=2"),
    ("wfull_a03_T3",     "(B) Weighted\nα=0.3, T=3"),
    ("ppr_a030",         "(C) PPR\nα=0.30"),
    ("ppr_a050",         "(C) PPR\nα=0.50"),
]


def load_results() -> pd.DataFrame:
    """Long-form (row_id, dataset, variant, metric, value)."""
    csv_path = REPO / "eval/results/figures/results_long.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}; run plot_results.py first")
    return pd.read_csv(csv_path)


def pick(df: pd.DataFrame, row_id: str, variant: str, metric: str) -> float | None:
    sel = df[
        (df["dataset"] == "spiqa_testA")
        & (df["row_id"] == row_id)
        & (df["variant"] == variant)
        & (df["metric"] == metric)
    ]
    if len(sel) == 0:
        return None
    return float(sel["value"].iloc[0])


def make_bar(values: list[float], labels: list[str], title: str,
             ylabel: str, ylim: tuple[float, float], out: Path,
             highlight: list[int] | None = None) -> None:
    """Single-axis bar chart; values in [0, 1], rendered as percent."""
    fig, ax = plt.subplots(figsize=(9, 4.2))

    colors = ["#888888"] * len(values)
    if highlight is not None:
        for i in highlight:
            colors[i] = "#1f6feb"   # blue accent for the highlighted bars

    bars = ax.bar(range(len(values)), [v * 100 for v in values], color=colors, edgecolor="black", linewidth=0.6)

    for i, v in enumerate(values):
        ax.text(i, v * 100 + 0.6, f"{v*100:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(ylim[0] * 100, ylim[1] * 100)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def fig_recall(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig: R@10 across the 8 paper rows."""
    labels = [r[1] for r in PAPER_ROWS]
    values = [pick(df, r[0], r[2], "recall@10") or 0.0 for r in PAPER_ROWS]
    # Highlight: the two big positive results (best in-domain + GME+prop)
    highlight = [
        values.index(max(values[:7])),   # best trained row
        len(values) - 1,                  # GME + prop (last bar)
    ]
    make_bar(
        values, labels,
        title="Recall@10 on SPIQA test-A",
        ylabel="Recall@10 (%)",
        ylim=(0.55, 0.95),
        out=out_dir / "spiqa_recallat10.png",
        highlight=highlight,
    )


def fig_mrr(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig: MRR across the 8 paper rows."""
    labels = [r[1] for r in PAPER_ROWS]
    values = [pick(df, r[0], r[2], "mrr") or 0.0 for r in PAPER_ROWS]
    highlight = [
        values.index(max(values[:7])),   # best trained
        len(values) - 1,                  # GME + prop
    ]
    make_bar(
        values, labels,
        title="MRR on SPIQA test-A",
        ylabel="MRR (%)",
        ylim=(0.30, 0.65),
        out=out_dir / "spiqa_mrr.png",
        highlight=highlight,
    )


def fig_prop_regimes_gme(df: pd.DataFrame, out_dir: Path) -> None:
    """Fig: 3 propagation regimes are equally effective on GME."""
    labels = [r[1] for r in PROP_REGIMES_ON_GME]
    r10 = [pick(df, "gme", r[0], "recall@10") or 0.0 for r in PROP_REGIMES_ON_GME]
    mrr = [pick(df, "gme", r[0], "mrr") or 0.0 for r in PROP_REGIMES_ON_GME]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(labels))
    w = 0.38

    # Colors: gray for no-prop, regime-tinted for A/B/C
    base_color = "#888888"
    a_color = "#FDC15B"  # uniform — amber
    b_color = "#1f6feb"  # weighted — blue
    c_color = "#2DA02D"  # PPR — green
    regime_colors = {"uniform": a_color, "wfull": b_color, "ppr": c_color}
    r10_colors = [base_color] + [
        regime_colors[r[0].split("_")[0]] for r in PROP_REGIMES_ON_GME[1:]
    ]

    bars_r = ax.bar([i - w / 2 for i in x], [v * 100 for v in r10], w, color=r10_colors,
                    edgecolor="black", linewidth=0.6, label="R@10")
    bars_m = ax.bar([i + w / 2 for i in x], [v * 100 for v in mrr], w, color=r10_colors,
                    edgecolor="black", linewidth=0.6, alpha=0.45, label="MRR")

    for i, (rv, mv) in enumerate(zip(r10, mrr)):
        ax.text(i - w / 2, rv * 100 + 1.0, f"{rv*100:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, mv * 100 + 1.0, f"{mv*100:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("score (%)", fontsize=11)
    ax.set_title("Three propagation regimes on GME (SPIQA test-A)", fontsize=12)
    ax.set_ylim(25, 95)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out = out_dir / "spiqa_gme_prop_regimes.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="eval/results/figures/paper")
    args = ap.parse_args()

    df = load_results()
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"output dir: {out_dir.relative_to(REPO)}/")
    fig_recall(df, out_dir)
    fig_mrr(df, out_dir)
    fig_prop_regimes_gme(df, out_dir)
    print("done")


if __name__ == "__main__":
    main()
