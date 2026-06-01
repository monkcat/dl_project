"""Deterministic result plots + digest for the experiment suite.

Reads every eval/results/experiments/<row>/eval.json (aggregate metrics) and
emits figures + a long-form CSV + a markdown digest. Pure post-hoc: it only
reads outputs, so it is safe to run any time after rows finish. No GPU.

Usage:
    python -m eval.analysis.plot_results
    python -m eval.analysis.plot_results --exp_root eval/results/experiments \
        --out_dir eval/results/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent

# Tier-1 main ablation order (matches REPORT §6.3)
TIER1 = ["a", "b", "c", "d", "e", "f", "g", "h", "gme"]
KEY_METRICS = ["recall@10", "coverage@10", "hit@10", "mrr", "perfect@10"]
PROP_REGIMES = {
    "uniform": "Group A (uniform)",
    "wbase": "Group B (weighted)",
    "wfull": "Group B (weighted)",
    "ppr": "Group C (PPR)",
    "prop": "diffusion",
    "no_prop": "no propagation",
}


def load_all(exp_root: Path) -> list[dict]:
    """Flatten every eval.json into long-form records."""
    rows = []
    for d in sorted(exp_root.glob("*/")):
        ej = d / "eval.json"
        sj = d / "summary.json"
        if not ej.exists():
            continue
        try:
            ev = json.loads(ej.read_text())
        except Exception:
            continue
        row_id = name = d.name
        if sj.exists():
            try:
                cfg = json.loads(sj.read_text()).get("config", {})
                row_id = cfg.get("row_id", d.name)
                name = cfg.get("name", d.name)
            except Exception:
                pass
        results = ev.get("results", ev)  # tolerate both shapes
        for dataset, variants in results.items():
            if not isinstance(variants, dict):
                continue
            for variant, metrics in variants.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, value in metrics.items():
                    if metric == "n_queries" or not isinstance(value, (int, float)):
                        continue
                    rows.append({
                        "row_id": row_id, "name": name, "dir": d.name,
                        "dataset": dataset, "variant": variant,
                        "metric": metric, "value": float(value),
                    })
    return rows


def _lookup(records, row_id, dataset, variant, metric):
    for r in records:
        if (r["row_id"] == row_id and r["dataset"] == dataset
                and r["variant"] == variant and r["metric"] == metric):
            return r["value"]
    return None


def plot_main_ablation(records, out_dir: Path):
    """Grouped bars: tier-1 rows (a..h, gme) per dataset, no_prop, per key metric."""
    datasets = sorted({r["dataset"] for r in records})
    rows_present = [r for r in TIER1 if any(x["row_id"] == r for x in records)]
    if not rows_present or not datasets:
        return []
    made = []
    for metric in KEY_METRICS:
        fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(rows_present)), 4.2))
        width = 0.8 / max(1, len(datasets))
        for di, ds in enumerate(datasets):
            vals = [_lookup(records, r, ds, "no_prop", metric) or 0.0 for r in rows_present]
            xs = [i + di * width for i in range(len(rows_present))]
            ax.bar(xs, vals, width=width, label=ds)
        ax.set_xticks([i + width * (len(datasets) - 1) / 2 for i in range(len(rows_present))])
        ax.set_xticklabels(rows_present)
        ax.set_ylabel(metric)
        ax.set_title(f"Tier-1 main ablation — {metric} (no_prop)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        p = out_dir / f"ablation_{metric.replace('@','at')}.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        made.append(p)
    return made


def plot_prop_sweep(records, out_dir: Path):
    """For each row that has many variants, plot metric across variants by regime."""
    made = []
    by_row = {}
    for r in records:
        by_row.setdefault(r["row_id"], set()).add(r["variant"])
    target = max(by_row, key=lambda k: len(by_row[k])) if by_row else None
    if not target or len(by_row[target]) < 3:
        return made
    datasets = sorted({r["dataset"] for r in records if r["row_id"] == target})
    for metric in ["recall@10", "coverage@10"]:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        variants = sorted({r["variant"] for r in records if r["row_id"] == target})
        for ds in datasets:
            vals = [_lookup(records, target, ds, v, metric) for v in variants]
            xs = list(range(len(variants)))
            ax.plot(xs, [v if v is not None else float("nan") for v in vals],
                    marker="o", label=ds)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(f"Propagation variants — {metric} (row {target})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        p = out_dir / f"propagation_{metric.replace('@','at')}.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        made.append(p)
    return made


def plot_training_curves(exp_root: Path, out_dir: Path):
    """Overlay L_tot vs step for rows that wrote train.json."""
    curves = []
    for d in sorted(exp_root.glob("*/")):
        tj = d / "train.json"
        if not tj.exists():
            continue
        try:
            t = json.loads(tj.read_text())
        except Exception:
            continue
        hist = t.get("history") or t.get("log") or t.get("steps")
        if isinstance(hist, list) and hist and isinstance(hist[0], dict):
            def _loss(h):
                return h.get("L_total", h.get("L_tot", h.get("loss")))
            pts = [(h["step"], _loss(h)) for h in hist
                   if "step" in h and _loss(h) is not None]
            if pts:
                xs, ys = zip(*pts)
                curves.append((d.name, list(xs), list(ys)))
    if not curves:
        return []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, xs, ys in curves[:20]:
        ax.plot(xs, ys, label=name, alpha=0.8, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("L_tot"); ax.set_yscale("log")
    ax.set_title("Training loss curves"); ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    p = out_dir / "training_curves.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return [p]


def write_csv_and_digest(records, out_dir: Path):
    csv_path = out_dir / "results_long.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "name", "dataset", "variant", "metric", "value"])
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in w.fieldnames})

    # digest: no_prop recall@10 + coverage@10 table across rows × datasets
    datasets = sorted({r["dataset"] for r in records})
    rows = sorted({r["row_id"] for r in records},
                  key=lambda x: (TIER1.index(x) if x in TIER1 else 99, x))
    lines = ["# Results digest (auto-generated)\n",
             f"Rows with eval results: {len(rows)}  |  datasets: {', '.join(datasets)}\n"]
    for metric in ["recall@10", "coverage@10", "hit@10", "mrr"]:
        lines.append(f"\n## {metric} (no_prop)\n")
        lines.append("| row | " + " | ".join(datasets) + " |")
        lines.append("|" + "---|" * (len(datasets) + 1))
        for r in rows:
            cells = []
            for ds in datasets:
                v = _lookup(records, r, ds, "no_prop", metric)
                cells.append(f"{v:.3f}" if v is not None else "—")
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
    digest = out_dir / "RESULTS_DIGEST.md"
    digest.write_text("\n".join(lines) + "\n")
    return csv_path, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="eval/results/experiments")
    ap.add_argument("--out_dir", default="eval/results/figures")
    args = ap.parse_args()
    exp_root = (REPO / args.exp_root) if not Path(args.exp_root).is_absolute() else Path(args.exp_root)
    out_dir = (REPO / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_all(exp_root)
    if not records:
        print(f"[plot_results] no eval.json found under {exp_root} — nothing to plot")
        return 0
    print(f"[plot_results] loaded {len(records)} metric records from "
          f"{len({r['row_id'] for r in records})} rows")

    made = []
    made += plot_main_ablation(records, out_dir)
    made += plot_prop_sweep(records, out_dir)
    made += plot_training_curves(exp_root, out_dir)
    csv_path, digest = write_csv_and_digest(records, out_dir)

    print(f"[plot_results] wrote {len(made)} figures + {csv_path.name} + {digest.name} → {out_dir}")
    for p in made:
        try:
            print("   -", p.relative_to(REPO))
        except ValueError:
            print("   -", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
