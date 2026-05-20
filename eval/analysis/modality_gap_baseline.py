"""Modality gap baseline — measure caption ↔ figure cosine on SPIQA test-A
caption_of edges, using SigLIPv2 zero-shot. This is the "before fine-tuning"
number that GRCL+caption_of supervision should improve.

Three measurements per (caption, figure) pair from v2.1 graph:
  1. caption-figure cosine          — direct cross-modal pair similarity
  2. caption-other_figure cosine    — distribution of unrelated cross-modal cos
  3. figure-other_figure cosine     — distribution of within-modal visual cos
  4. caption-other_caption cosine   — distribution of within-modal text cos

Saves:
  - eval/results/modality_gap_baseline/<encoder>_<dataset>.json    (summary stats)
  - eval/results/figures/modality_gap_baseline/<encoder>_<dataset>_hist.png

Usage:
    python -m eval.analysis.modality_gap_baseline \\
        --dataset spiqa_testA --encoder siglipv2-base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from eval.analysis.motivation_modality_gap import SigLIPLikeEncoder  # type: ignore


DATASETS = {
    "spiqa_testA": {
        "graph": REPO / "data/benchmarks/spiqa/test-A/element_graph_v2.json",
        "elements": REPO / "data/benchmarks/spiqa/test-A/elements_v2.jsonl",
        "image_root": REPO / "data/benchmarks/spiqa/test-A/images_224px/SPIQA_testA_Images_224px",
    },
}

ENCODERS = {
    "siglipv2-base": ("google/siglip2-base-patch16-224", SigLIPLikeEncoder),
    "clip-l-14":    ("openai/clip-vit-large-patch14",    SigLIPLikeEncoder),
}


def load_caption_figure_pairs(graph_path: Path, elements_path: Path, image_root: Path) -> list[dict]:
    """Return [{caption_text, figure_path, doc_id, fname, label}, ...] for all caption_of edges."""
    graphs = json.loads(graph_path.read_text())
    elements: dict[str, dict] = {}
    with open(elements_path) as f:
        for line in f:
            r = json.loads(line)
            elements[r["id"]] = r

    pairs: list[dict] = []
    for doc_id, g in graphs.items():
        for edge in g.get("edges", []):
            if edge["type"] != "caption_of":
                continue
            # caption_of: src=caption, dst=figure/table
            cap_id = edge["src"]
            fig_id = edge["dst"]
            cap_rec = elements.get(cap_id, {})
            fig_rec = elements.get(fig_id, {})
            cap_text = cap_rec.get("text", "")
            if not cap_text:
                continue
            # figure image path: image_root / doc_id / fig_id (which is the SPIQA filename)
            fig_path = image_root / doc_id / fig_id
            if not fig_path.exists():
                continue
            pairs.append({
                "doc_id": doc_id,
                "caption_id": cap_id,
                "figure_id": fig_id,
                "caption_text": cap_text,
                "figure_path": str(fig_path),
                "label": fig_rec.get("label") or cap_rec.get("label"),
                "type": fig_rec.get("type"),  # figure or table
            })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="spiqa_testA", choices=list(DATASETS.keys()))
    ap.add_argument("--encoder", default="siglipv2-base", choices=list(ENCODERS.keys()))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_pairs", type=int, default=None, help="cap for quick test")
    args = ap.parse_args()

    ds_cfg = DATASETS[args.dataset]
    hf_id, encoder_cls = ENCODERS[args.encoder]

    out_dir = REPO / "eval/results/modality_gap_baseline"
    fig_dir = REPO / "eval/results/figures/modality_gap_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.encoder}_{args.dataset}.json"
    hist_path = fig_dir / f"{args.encoder}_{args.dataset}_hist.png"

    print(f"[1/4] loading caption_of pairs from {ds_cfg['graph']}")
    pairs = load_caption_figure_pairs(ds_cfg["graph"], ds_cfg["elements"], ds_cfg["image_root"])
    print(f"  {len(pairs)} caption_of pairs with both caption text and figure image")
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
        print(f"  capped at {len(pairs)} (--max_pairs)")
    if not pairs:
        raise SystemExit("no pairs")

    print(f"[2/4] loading encoder {args.encoder}")
    encoder = encoder_cls(args.encoder, hf_id, device=args.device)

    print(f"[3/4] encoding {len(pairs)} caption texts + figure images")
    captions = [p["caption_text"] for p in pairs]
    text_embs = encoder.encode_text(captions, batch_size=64)        # (N, D)

    images = []
    valid_idx = []
    for i, p in enumerate(pairs):
        try:
            img = Image.open(p["figure_path"]).convert("RGB")
            images.append(img)
            valid_idx.append(i)
        except Exception as e:
            print(f"  WARN: failed to load {p['figure_path']}: {e}")
    img_embs = encoder.encode_images(images, batch_size=16)         # (M, D)

    # Filter to only valid pairs
    text_embs = text_embs[valid_idx]
    pairs_v = [pairs[i] for i in valid_idx]
    N = len(pairs_v)
    print(f"  encoded {N} valid pairs")

    print(f"[4/4] computing cosine distributions")

    # 1. matched caption ↔ figure cosine (diagonal)
    diag_cos = (text_embs * img_embs).sum(dim=-1).numpy()  # (N,)

    # 2. caption ↔ random other figure (off-diagonal sample)
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    # Avoid self-pair: shift by 1
    perm_self_avoid = (perm + 1) % N
    text_to_other_fig = (text_embs * img_embs[perm_self_avoid]).sum(dim=-1).numpy()

    # 3. figure ↔ figure (within-visual)
    img_to_other_img = (img_embs * img_embs[perm_self_avoid]).sum(dim=-1).numpy()

    # 4. caption ↔ caption (within-text)
    text_to_other_text = (text_embs * text_embs[perm_self_avoid]).sum(dim=-1).numpy()

    # Stats
    def stats(arr):
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary = {
        "encoder": args.encoder,
        "dataset": args.dataset,
        "n_pairs": N,
        "matched_caption_figure": stats(diag_cos),
        "caption_to_other_figure": stats(text_to_other_fig),
        "figure_to_other_figure": stats(img_to_other_img),
        "caption_to_other_caption": stats(text_to_other_text),
        "gap_intra_minus_cross": {
            "text_intra_minus_cross": float(text_to_other_text.mean() - diag_cos.mean()),
            "visual_intra_minus_cross": float(img_to_other_img.mean() - diag_cos.mean()),
        },
        "by_type": {},
    }

    # Breakdown by figure vs table
    for t in ("figure", "table"):
        mask = np.array([p["type"] == t for p in pairs_v])
        if mask.sum() > 0:
            summary["by_type"][t] = {
                "n": int(mask.sum()),
                "matched_mean": float(diag_cos[mask].mean()),
                "matched_std": float(diag_cos[mask].std()),
            }

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary ({args.encoder} on {args.dataset}) ===")
    print(f"  caption ↔ matched figure:    mean={summary['matched_caption_figure']['mean']:+.4f}  std={summary['matched_caption_figure']['std']:.4f}")
    print(f"  caption ↔ other figure:      mean={summary['caption_to_other_figure']['mean']:+.4f}  std={summary['caption_to_other_figure']['std']:.4f}")
    print(f"  figure  ↔ other figure:      mean={summary['figure_to_other_figure']['mean']:+.4f}  std={summary['figure_to_other_figure']['std']:.4f}")
    print(f"  caption ↔ other caption:     mean={summary['caption_to_other_caption']['mean']:+.4f}  std={summary['caption_to_other_caption']['std']:.4f}")
    print(f"  text intra − matched cross:  {summary['gap_intra_minus_cross']['text_intra_minus_cross']:+.4f}")
    print(f"  visual intra − matched cross: {summary['gap_intra_minus_cross']['visual_intra_minus_cross']:+.4f}")
    print(f"  by type:")
    for t, s in summary["by_type"].items():
        print(f"    {t}: n={s['n']}, matched mean={s['matched_mean']:+.4f}")
    print(f"\n  saved → {summary_path}")

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-0.1, 1.0, 60)
    ax.hist(diag_cos, bins=bins, alpha=0.7, label=f"caption ↔ matched figure (μ={diag_cos.mean():.3f})", color="#D97706")
    ax.hist(text_to_other_fig, bins=bins, alpha=0.5, label=f"caption ↔ other figure (μ={text_to_other_fig.mean():.3f})", color="#9CA3AF")
    ax.hist(text_to_other_text, bins=bins, alpha=0.5, label=f"caption ↔ other caption (μ={text_to_other_text.mean():.3f})", color="#3B82F6")
    ax.hist(img_to_other_img, bins=bins, alpha=0.5, label=f"figure ↔ other figure (μ={img_to_other_img.mean():.3f})", color="#10B981")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    ax.set_title(f"Modality gap baseline (zero-shot)\n{args.encoder} on {args.dataset} — {N} caption_of pairs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=120)
    print(f"  plot saved → {hist_path}")


if __name__ == "__main__":
    main()
