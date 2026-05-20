"""Hypersphere visualization of modality gap.

L2-normalized multimodal embeddings live on a unit hypersphere. Standard t-SNE
distorts geodesic distances. This script produces three modality-gap-aware
visualizations:

  A. 3D PCA + sphere projection — embeddings projected to top-3 PCs, then
     L2-normalized to the unit 3-sphere. Cones for text/visual become
     directly visible (Liang et al. NeurIPS 2022 style).

  B. Pairwise angle (cosine) histograms — distribution of cos(text_i, text_j),
     cos(visual_i, visual_j), cos(text_i, visual_j). Shows the full gap
     distribution, not just averages.

  C. Centroid-angle 2D scatter — for each element, plot (cos to text centroid,
     cos to visual centroid). Modality cones appear as two diagonal clusters.

Usage:
    python eval/analysis/hypersphere_viz.py \\
        --dataset spiqa --encoder siglipv2-base \\
        --sample-text 800 --sample-visual 400 \\
        --out-dir eval/results/figures/hypersphere
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from PIL import Image

from eval.datasets.loaders import load_mmdocir, load_sciegqa, load_spiqa_enhanced
from eval.analysis.motivation_modality_gap import build_encoder


def pretty_encoder(name: str) -> str:
    return {
        "siglipv2-base": "SigLIPv2",
        "clip-l-14":     "CLIP-L/14",
        "jina-clip-v2":  "Jina-CLIP-v2",
    }.get(name, name)


def pretty_dataset(name: str) -> str:
    return {"sciegqa": "SciEGQA", "mmdocir": "MMDocIR", "spiqa": "SPIQA test-A"}.get(name, name)


def plot_3d_sphere(text_emb: torch.Tensor, visual_emb: torch.Tensor,
                    q_emb: torch.Tensor, title: str, out_path: Path):
    """3D PCA → unit sphere projection."""
    # Compute PCA on all embeddings together
    all_emb = torch.cat([text_emb, visual_emb, q_emb], dim=0).numpy()
    centered = all_emb - all_emb.mean(axis=0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    top3 = centered @ Vt[:3].T  # (N, 3)
    # Project to unit sphere
    norms = np.linalg.norm(top3, axis=1, keepdims=True)
    top3 = top3 / np.clip(norms, 1e-9, None)

    n_t = len(text_emb); n_v = len(visual_emb); n_q = len(q_emb)
    text_3d = top3[:n_t]
    visual_3d = top3[n_t:n_t+n_v]
    q_3d = top3[n_t+n_v:]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Light reference sphere wireframe
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    sx = np.cos(u) * np.sin(v)
    sy = np.sin(u) * np.sin(v)
    sz = np.cos(v)
    ax.plot_wireframe(sx, sy, sz, color="lightgray", alpha=0.15, linewidth=0.4)

    ax.scatter(text_3d[:, 0], text_3d[:, 1], text_3d[:, 2],
                c="steelblue", s=8, alpha=0.5, label=f"text (n={n_t})")
    ax.scatter(visual_3d[:, 0], visual_3d[:, 1], visual_3d[:, 2],
                c="coral", s=14, alpha=0.7, label=f"visual (n={n_v})")
    ax.scatter(q_3d[:, 0], q_3d[:, 1], q_3d[:, 2],
                c="darkgreen", s=20, alpha=0.8, marker="^", label=f"query (n={n_q})")

    var_explained = (S[:3]**2 / (S**2).sum())
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.0f}%)")
    ax.set_zlabel(f"PC3 ({var_explained[2]*100:.0f}%)")
    ax.set_title(title + "\n3D PCA + unit-sphere projection")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_angle_histogram(text_emb: torch.Tensor, visual_emb: torch.Tensor,
                          title: str, out_path: Path, n_sample: int = 1500):
    """Histogram of pairwise cosine (or angle) for intra-text, intra-visual, cross."""
    rng = np.random.default_rng(42)

    def sample_pairs(a: torch.Tensor, b: torch.Tensor, n: int) -> np.ndarray:
        i = rng.integers(0, len(a), size=n)
        j = rng.integers(0, len(b), size=n)
        return (a[i] * b[j]).sum(dim=-1).numpy()

    text_text   = sample_pairs(text_emb, text_emb, n_sample)
    vis_vis     = sample_pairs(visual_emb, visual_emb, n_sample) if len(visual_emb) > 1 else np.array([])
    text_vis    = sample_pairs(text_emb, visual_emb, n_sample) if len(visual_emb) else np.array([])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(-0.2, 1.0, 50)
    ax.hist(text_text,  bins=bins, alpha=0.5, color="steelblue", label=f"text–text (mean={text_text.mean():.3f})")
    if len(vis_vis):
        ax.hist(vis_vis, bins=bins, alpha=0.5, color="coral", label=f"visual–visual (mean={vis_vis.mean():.3f})")
    if len(text_vis):
        ax.hist(text_vis, bins=bins, alpha=0.5, color="gray", label=f"text–visual (mean={text_vis.mean():.3f})")
    ax.axvline(0, color="black", linestyle="--", alpha=0.3, linewidth=0.7)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("# pairs")
    ax.set_title(title + "\npairwise cosine distribution (gap = intra modes far from cross mode)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_centroid_angle_scatter(text_emb: torch.Tensor, visual_emb: torch.Tensor,
                                  q_emb: torch.Tensor, title: str, out_path: Path):
    """For each element, plot (cos to text centroid, cos to visual centroid)."""
    text_centroid = F.normalize(text_emb.mean(dim=0, keepdim=True), dim=-1)  # (1, D)
    visual_centroid = F.normalize(visual_emb.mean(dim=0, keepdim=True), dim=-1) if len(visual_emb) else None

    def proj(emb):
        cos_t = (emb @ text_centroid.T).squeeze(-1).numpy()
        cos_v = (emb @ visual_centroid.T).squeeze(-1).numpy() if visual_centroid is not None else np.zeros(len(emb))
        return cos_t, cos_v

    t_t, t_v = proj(text_emb)
    v_t, v_v = proj(visual_emb) if len(visual_emb) else (np.array([]), np.array([]))
    q_t, q_v = proj(q_emb)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(t_t, t_v, c="steelblue", s=6, alpha=0.4, label=f"text element (n={len(text_emb)})")
    if len(visual_emb):
        ax.scatter(v_t, v_v, c="coral", s=15, alpha=0.7, label=f"visual element (n={len(visual_emb)})")
    ax.scatter(q_t, q_v, c="darkgreen", s=20, alpha=0.8, marker="^", label=f"query (n={len(q_emb)})")

    # Diagonal reference
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]) - 0.05,
           max(ax.get_xlim()[1], ax.get_ylim()[1]) + 0.05]
    ax.plot(lim, lim, "k--", alpha=0.3, linewidth=0.7, label="diagonal (no modality bias)")

    ax.set_xlabel("cos(·, text centroid)")
    ax.set_ylabel("cos(·, visual centroid)")
    ax.set_title(title + "\nelements near their own modality centroid ⇒ cones")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sciegqa", "mmdocir", "spiqa"], required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--sample-text", type=int, default=800, help="random sample of text elements")
    ap.add_argument("--sample-visual", type=int, default=400, help="random sample of visual elements")
    ap.add_argument("--sample-query", type=int, default=300, help="random sample of queries")
    ap.add_argument("--out-dir", default="eval/results/figures/hypersphere")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.encoder}"

    print(f"\n=== Hypersphere viz: {pretty_dataset(args.dataset)} × {pretty_encoder(args.encoder)} ===")

    loader = {"sciegqa": load_sciegqa, "mmdocir": load_mmdocir, "spiqa": load_spiqa_enhanced}[args.dataset]
    elements, queries = loader()
    text_idx = [i for i, e in enumerate(elements) if e["type"] == "text"]
    visual_idx = [i for i, e in enumerate(elements) if e["type"] != "text"]
    print(f"  pool: {len(text_idx)} text, {len(visual_idx)} visual, {len(queries)} queries")

    # Sample
    rng = np.random.default_rng(42)
    t_sample = rng.choice(text_idx, size=min(args.sample_text, len(text_idx)), replace=False)
    v_sample = rng.choice(visual_idx, size=min(args.sample_visual, len(visual_idx)), replace=False) if visual_idx else np.array([], dtype=int)
    q_sample_idx = rng.choice(len(queries), size=min(args.sample_query, len(queries)), replace=False)

    encoder = build_encoder(args.encoder, device=args.device)

    print(f"  encoding {len(t_sample)} text elements...")
    t_emb = encoder.encode_text([elements[i]["text"][:512] for i in t_sample])

    if len(v_sample):
        print(f"  encoding {len(v_sample)} visual elements...")
        v_emb = encoder.encode_images([elements[i]["image_loader"]() for i in v_sample])
    else:
        v_emb = torch.zeros(0, t_emb.shape[1])

    print(f"  encoding {len(q_sample_idx)} queries...")
    q_emb = encoder.encode_text([queries[i]["query"] for i in q_sample_idx])

    # Plots
    title = f"{pretty_dataset(args.dataset)} × {pretty_encoder(args.encoder)}"
    print("\nGenerating plots:")
    plot_3d_sphere(t_emb, v_emb, q_emb, title, out_dir / f"sphere3d_{tag}.png")
    plot_angle_histogram(t_emb, v_emb, title, out_dir / f"anglehist_{tag}.png")
    plot_centroid_angle_scatter(t_emb, v_emb, q_emb, title, out_dir / f"centroid2d_{tag}.png")

    print(f"\nDone. Output: {out_dir}/")


if __name__ == "__main__":
    main()
