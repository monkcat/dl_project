"""Cleaner system architecture diagram."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("eval/results/figures")
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, color="#E8F0FE", edge="#3367D6", fontsize=9, weight="normal"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                           edgecolor=edge, facecolor=color, linewidth=1.4)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight)


def arrow(ax, x1, y1, x2, y2, color="black", style="->", lw=1.2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=12,
                        color=color, linewidth=lw,
                        connectionstyle="arc3,rad=0")
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(7, 9.7, "Layout-/Structure-aware Element-level Multimodal RAG",
             ha="center", fontsize=14, weight="bold")

    # ============ OFFLINE banner ============
    ax.add_patch(FancyBboxPatch((0.2, 6.0), 13.6, 3.2,
                                  boxstyle="round,pad=0.05",
                                  facecolor="#FAFAFA", edgecolor="#888", linestyle="--", linewidth=1.3))
    ax.text(0.4, 9.0, "OFFLINE  (per corpus, one-time)",
             fontsize=11, weight="bold", color="#555")

    # PDFs
    box(ax, 0.6, 7.4, 1.6, 0.8, "Academic\nPDFs", color="#FFE0B2", edge="#E65100", fontsize=9)

    # Docling
    box(ax, 3.0, 7.4, 1.9, 0.8, "Docling\nextractor", color="#C8E6C9", edge="#2E7D32",
        fontsize=9, weight="bold")

    # Three outputs of docling
    box(ax, 5.6, 8.1, 2.1, 0.7, "Elements\n(text/fig/table)", color="#E8F0FE", fontsize=8)
    box(ax, 5.6, 7.1, 2.1, 0.7, "Captions + bboxes", color="#E8F0FE", fontsize=8)
    box(ax, 5.6, 6.1, 2.1, 0.7, "Section headers\n→ section_path π(e)", color="#E8F0FE", fontsize=8)

    # Three offline artifacts
    box(ax, 8.6, 8.1, 2.3, 0.7, "Element graph G\n(caption_of, refs, next)",
        color="#FFF9C4", edge="#F9A825", fontsize=8)
    box(ax, 8.6, 7.1, 2.3, 0.7, "HS-PE vectors\n(parameter-free)",
        color="#FFF9C4", edge="#F9A825", fontsize=8)
    box(ax, 8.6, 6.1, 2.3, 0.7, "SigLIPv2 + LoRA",
        color="#FFF9C4", edge="#F9A825", fontsize=8, weight="bold")
    box(ax, 11.4, 6.1, 2.0, 0.7, "embeddings.pt\n(cached)",
        color="#E0E0E0", edge="#424242", fontsize=8)

    # Offline arrows (flat, left-to-right)
    arrow(ax, 2.2, 7.8, 3.0, 7.8)
    arrow(ax, 4.9, 8.0, 5.6, 8.4)
    arrow(ax, 4.9, 7.8, 5.6, 7.4)
    arrow(ax, 4.9, 7.5, 5.6, 6.4)
    arrow(ax, 7.7, 8.4, 8.6, 8.4)        # elements + captions → graph
    arrow(ax, 7.7, 7.4, 8.6, 8.4, color="#1976D2")  # captions help build graph
    arrow(ax, 7.7, 6.4, 8.6, 7.4)        # section_path → HS-PE
    arrow(ax, 7.7, 8.4, 8.6, 6.4, color="#1976D2")  # elements → encoder
    arrow(ax, 10.9, 6.4, 11.4, 6.4)

    # ============ ONLINE banner ============
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 13.6, 5.3,
                                  boxstyle="round,pad=0.05",
                                  facecolor="#F0F8FF", edgecolor="#888", linestyle="--", linewidth=1.3))
    ax.text(0.4, 5.4, "ONLINE  (per query)",
             fontsize=11, weight="bold", color="#555")

    # Query
    box(ax, 0.6, 4.3, 1.6, 0.7, "Query text", color="#FFE0B2", edge="#E65100", fontsize=10, weight="bold")
    box(ax, 0.6, 3.3, 1.6, 0.7, "Encode\n(SigLIPv2)", color="#C8E6C9", edge="#2E7D32", fontsize=9)
    box(ax, 0.6, 2.3, 1.6, 0.7, "q · corpus\n(matmul)", color="#E8F0FE", fontsize=9)
    box(ax, 0.6, 1.3, 1.6, 0.7, "RRF (per-modality)\n+ per-paper mask", color="#E8F0FE", fontsize=8)
    arrow(ax, 1.4, 4.3, 1.4, 4.0)
    arrow(ax, 1.4, 3.3, 1.4, 3.0)
    arrow(ax, 1.4, 2.3, 1.4, 2.0)

    # Anchor selection
    box(ax, 3.0, 1.3, 1.7, 0.7, "top-N anchors\nA_q ⊂ doc",
        color="#FFCDD2", edge="#C62828", fontsize=9, weight="bold")
    arrow(ax, 2.2, 1.65, 3.0, 1.65)

    # Three score paths
    box(ax, 5.5, 4.3, 3.0, 0.85,
        "α · content_RRF(q,e)\n(cross-modal content)",
        color="#E8F0FE", edge="#3367D6", fontsize=9)
    box(ax, 5.5, 2.85, 3.0, 0.95,
        "β · B̃_C2(q,e)\nmax_a cos(HS-PE(a), HS-PE(e))\n+ RRF normalize",
        color="#FFF9C4", edge="#F9A825", fontsize=9)
    box(ax, 5.5, 1.3, 3.0, 0.95,
        "γ · B_C1(q,e)\nPQ-BFS expansion\nfrom A_q (≤2 hops, ρ=0.6)",
        color="#FFF9C4", edge="#F9A825", fontsize=9)

    # Lines from anchors → score paths
    arrow(ax, 4.7, 1.65, 5.5, 1.78, color="#F9A825")    # → C1
    arrow(ax, 4.7, 1.65, 5.5, 3.33, color="#F9A825")    # → C2
    arrow(ax, 2.2, 1.65, 5.5, 4.6, color="#3367D6")     # → content

    # Combiner
    box(ax, 9.5, 2.55, 2.0, 1.0, "S(q,e) =\nα·c + β·c2 + γ·c1",
        color="#FFCDD2", edge="#C62828", fontsize=10, weight="bold")
    arrow(ax, 8.5, 4.7, 9.5, 3.3, color="#3367D6")
    arrow(ax, 8.5, 3.3, 9.5, 3.0, color="#F9A825")
    arrow(ax, 8.5, 1.8, 9.5, 2.8, color="#F9A825")

    # Top-k
    box(ax, 12.3, 2.55, 1.4, 1.0, "top-k\nresults",
        color="#C8E6C9", edge="#2E7D32", fontsize=10, weight="bold")
    arrow(ax, 11.5, 3.05, 12.3, 3.05)

    # Dashed lines from offline artifacts to online use
    arrow(ax, 12.4, 6.1, 1.4, 2.6, color="#999", lw=0.8)
    ax.text(7.5, 6.0, "cached", fontsize=7, color="#999", style="italic")
    arrow(ax, 9.7, 8.1, 7.0, 2.3, color="#999", lw=0.8)
    arrow(ax, 9.7, 7.1, 7.0, 3.8, color="#999", lw=0.8)

    plt.tight_layout()
    out = OUT / "system_diagram.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
