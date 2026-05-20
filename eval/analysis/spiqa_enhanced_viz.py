"""Visualize the SPIQA augmentation: show how docling enriches SPIQA with
page + bbox + section_path for each figure.

Outputs:
  - eval/results/figures/spiqa_enhanced_stats.png  (overview stats)
  - eval/results/figures/spiqa_enhanced_example_*.png  (per-paper viz)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import matplotlib.patches as patches


ROOT = Path("data/benchmarks/spiqa/test-A")
PDF_DIR = ROOT / "pdfs"
OUT_DIR = Path("eval/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def render_overview_stats():
    spiqa = json.loads((ROOT / "SPIQA_testA.json").read_text())
    mapping = json.loads((ROOT / "spiqa_fig_to_element.json").read_text())
    sections = json.loads((ROOT / "section_paths_docling.json").read_text())
    elements = defaultdict(list)
    with open(ROOT / "elements_docling.jsonl") as f:
        for line in f:
            r = json.loads(line)
            for e in r["elements"]:
                elements[r["doc_id"]].append({**e, "page": r["page"]})

    # Per-paper distribution of matched figures
    matched_per_paper = []
    figures_per_paper = []
    paper_with_section = 0
    matched_with_section = 0
    fig_section_role = Counter()
    sim_dist = []

    # Roles inferred from section_path level-1 idx is unreliable;
    # use heuristic: classify section header text via simple keywords
    # For simplicity, just report depth distribution and matched count

    for paper_id, paper in spiqa.items():
        figs = list(paper["all_figures"].keys())
        figures_per_paper.append(len(figs))
        n_matched = sum(1 for f in figs if f in mapping)
        matched_per_paper.append(n_matched)
        if paper_id in sections and sections[paper_id]:
            paper_with_section += 1
        for f in figs:
            if f in mapping:
                m = mapping[f]
                sim_dist.append(m["caption_similarity"])
                # Section depth of this figure's element
                sp = sections.get(paper_id, {}).get(m["eid"])
                if sp:
                    matched_with_section += 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Matched per paper
    axes[0,0].hist(matched_per_paper, bins=15, color="tab:green", alpha=0.7, label="matched")
    axes[0,0].hist(figures_per_paper, bins=15, color="tab:gray", alpha=0.4, label="total")
    axes[0,0].set_xlabel("# figures/tables per paper")
    axes[0,0].set_ylabel("# papers")
    axes[0,0].legend()
    axes[0,0].set_title(f"Per-paper match coverage\n"
                        f"({sum(matched_per_paper)}/{sum(figures_per_paper)} = "
                        f"{sum(matched_per_paper)/sum(figures_per_paper):.1%} matched)")

    # 2. Similarity distribution
    axes[0,1].hist(sim_dist, bins=20, color="tab:blue")
    axes[0,1].set_xlabel("caption similarity score (SequenceMatcher)")
    axes[0,1].set_ylabel("# figures")
    axes[0,1].set_title(f"Match quality\nmean={sum(sim_dist)/len(sim_dist):.3f}, "
                        f"≥0.9: {sum(1 for s in sim_dist if s>=0.9)} ({sum(1 for s in sim_dist if s>=0.9)/len(sim_dist):.0%})")

    # 3. Added annotations per figure
    added_info = ["page", "bbox", "section_path", "section_role", "caption_block_id",
                   "neighboring_paragraphs", "associated_caption"]
    bar_labels = ["Original\nSPIQA", "+ docling enhanced"]
    original_info = ["caption_text", "figure_image", "content_type"]
    counts = [len(original_info), len(original_info) + len(added_info)]
    axes[1,0].bar(bar_labels, counts, color=["tab:gray", "tab:green"])
    for i, c in enumerate(counts):
        axes[1,0].text(i, c, str(c), ha="center", va="bottom")
    axes[1,0].set_ylabel("# annotation fields per figure")
    axes[1,0].set_title("Annotation richness: original vs enhanced")

    # 4. Section path depth distribution
    depths = []
    for paper_id, paths in sections.items():
        for p in paths.values():
            depths.append(len(p))
    if depths:
        depth_count = Counter(depths)
        ks = sorted(depth_count.keys())
        axes[1,1].bar([str(k) for k in ks], [depth_count[k] for k in ks], color="tab:purple")
        axes[1,1].set_xlabel("section_path depth (level)")
        axes[1,1].set_ylabel("# elements")
        axes[1,1].set_title(f"Section hierarchy depth\n(total {len(depths)} elements with section_path)")

    plt.suptitle("SPIQA Augmentation: original annotations + docling-derived metadata", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "spiqa_enhanced_stats.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def render_example(paper_id: str):
    """For one paper: show page with SPIQA figure overlaid with docling annotations."""
    spiqa = json.loads((ROOT / "SPIQA_testA.json").read_text())
    mapping = json.loads((ROOT / "spiqa_fig_to_element.json").read_text())
    sections = json.loads((ROOT / "section_paths_docling.json").read_text())

    elements = []
    with open(ROOT / "elements_docling.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["doc_id"] == paper_id:
                for e in r["elements"]:
                    elements.append({**e, "page": r["page"]})

    paper = spiqa[paper_id]
    paths = sections.get(paper_id, {})

    # Pick one figure that was matched, render its page
    target_fname = None; target_mapping = None
    for fname, m in mapping.items():
        if fname.startswith(paper_id + "-"):
            target_fname = fname; target_mapping = m
            break
    if target_fname is None:
        print(f"no matched figure for {paper_id}")
        return

    target_page = target_mapping["page"]
    target_eid = target_mapping["eid"]

    # Render PDF page
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    doc = fitz.open(pdf_path)
    page = doc[target_page - 1]
    pix = page.get_pixmap(dpi=120)
    img_bytes = pix.tobytes("png")
    import io
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(img_bytes))
    scale_x = img.size[0] / page.rect.width
    scale_y = img.size[1] / page.rect.height

    fig, ax = plt.subplots(figsize=(8, 11))
    ax.imshow(img)

    # Overlay all elements on this page
    for e in elements:
        if e["page"] != target_page: continue
        b = e["bbox_pdf"]
        # docling bbox is (l, b, r, t) in PDF coords; flip y for image
        x0, x1 = b[0] * scale_x, b[2] * scale_x
        # PDF y from bottom; PIL/matplotlib y from top
        pdf_h = page.rect.height
        y0 = (pdf_h - b[3]) * scale_y  # top of bbox in image coords
        y1 = (pdf_h - b[1]) * scale_y  # bottom of bbox in image coords
        color_map = {"text": "tab:green", "image": "tab:blue", "table": "tab:orange",
                      "caption": "tab:red"}
        col = color_map.get(e["type"], "gray")
        is_target = (e["eid"] == target_eid)
        lw = 3.5 if is_target else 0.8
        alpha = 1.0 if is_target else 0.5
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        linewidth=lw, edgecolor=col, facecolor="none",
                                        alpha=alpha))
        if is_target:
            sp = paths.get(e["eid"], [])
            ax.text(x0, y0 - 5, f"⭐ SPIQA: '{target_fname}'\n   section_path={sp}",
                    fontsize=9, color="red", va="bottom", weight="bold")

    spiqa_cap = paper["all_figures"][target_fname]["caption"][:120]
    sp_label = paths.get(target_eid, [])
    title = (f"SPIQA enhancement example — {paper_id} (page {target_page})\n"
             f"⭐ matched element: {target_eid}, section_path={sp_label}, "
             f"caption sim={target_mapping['caption_similarity']}")
    subtitle = f"SPIQA caption: '{spiqa_cap}'"
    ax.set_title(f"{title}\n{subtitle}", fontsize=9, loc="left")
    ax.axis("off")
    plt.tight_layout()
    out = OUT_DIR / f"spiqa_enhanced_example_{paper_id}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def main():
    print("Generating overview stats...")
    render_overview_stats()
    # Pick a few diverse papers
    spiqa = json.loads((ROOT / "SPIQA_testA.json").read_text())
    chosen = list(spiqa.keys())[:3]
    print(f"Rendering examples for: {chosen}")
    for pid in chosen:
        render_example(pid)


if __name__ == "__main__":
    main()
