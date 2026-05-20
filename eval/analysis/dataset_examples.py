"""Render dataset example visualizations: page image + GT bbox + matched element bboxes.

For each of MMDocIR and SciEGQA, pick representative queries and visualize:
  - GT semantic region (red)
  - Each contained element (green by type)
  - Query + answer text

Output: eval/results/figures/dataset_examples_{dataset}_{i}.png
"""
from __future__ import annotations

import base64
import io
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


OUT_DIR = Path("eval/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TYPE_COLOR = {
    "text": "tab:green",
    "image": "tab:blue",
    "table": "tab:orange",
    "drawing": "tab:purple",
    "equation": "tab:cyan",
}


def wrap(s: str, width: int = 90) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(s, width=width))


def render_sciegqa_examples(n_examples: int = 4):
    root = Path("data/benchmarks/sciegqa")
    # Load elements (with bbox_img) per (doc, page)
    pages_to_elems = defaultdict(list)
    eid_to_elem = {}
    eid_counter = defaultdict(int)
    with open(root / "elements.jsonl") as f:
        for line in f:
            r = json.loads(line)
            for e in r["elements"]:
                eid = f"{r['doc_id']}__p{r['page']:03d}__e{eid_counter[r['doc_id']]:04d}"
                eid_counter[r["doc_id"]] += 1
                e2 = {**e, "eid": eid, "page": r["page"], "doc_id": r["doc_id"]}
                pages_to_elems[(r["doc_id"], r["page"])].append(e2)
                eid_to_elem[eid] = e2

    # Find category dir
    img_root = root / "Images"
    doc_cat = {}
    for cat_dir in img_root.iterdir():
        if cat_dir.is_dir():
            for doc_dir in cat_dir.iterdir():
                doc_cat[doc_dir.name] = cat_dir.name

    # Pick queries spanning different subimg types
    queries = []
    with open(root / "queries_with_gt.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["gt_element_ids"]:
                queries.append(r)

    # Sample: prefer queries with 3-8 GT elements on 1 page
    def score(q):
        n = len(q["gt_element_ids"])
        single_page = len(q["evidence_pages"]) == 1
        return (3 <= n <= 8, single_page, n)
    queries_sorted = sorted(queries, key=score, reverse=True)

    # Take examples from different categories
    chosen = []
    seen_cats = set()
    for q in queries_sorted:
        if q["category"] in seen_cats:
            continue
        seen_cats.add(q["category"])
        chosen.append(q)
        if len(chosen) >= n_examples:
            break

    for idx, q in enumerate(chosen):
        doc_id = q["doc_id"]
        page = q["evidence_pages"][0]
        cat = doc_cat.get(doc_id, "")
        png = img_root / cat / doc_id / f"{doc_id}_{page}.png"
        if not png.exists():
            continue
        img = Image.open(png).convert("RGB")

        # GT region bboxes (from SciEGQA_Bench.jsonl) for this query
        gt_regions = []
        with open(root / "SciEGQA_Bench.jsonl") as f:
            qi = 0
            for line in f:
                if qi == int(q["qid"].split("_")[1]):
                    sci = json.loads(line)
                    for pi, p in enumerate(sci["evidence_page"]):
                        if p == page:
                            for bb in sci["bbox"][pi]:
                                gt_regions.append(bb)
                    break
                qi += 1

        fig, ax = plt.subplots(figsize=(8, 11))
        ax.imshow(img)
        # GT region (red)
        for bb in gt_regions:
            ax.add_patch(patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                                            linewidth=3, edgecolor="red", facecolor="none", label="GT region"))

        # Matched elements (colored by type)
        for eid in q["gt_element_ids"]:
            e = eid_to_elem.get(eid)
            if e is None or e["page"] != page:
                continue
            bb = e["bbox_img"]
            color = TYPE_COLOR.get(e["type"], "gray")
            ax.add_patch(patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                                            linewidth=2, edgecolor=color, facecolor="none", alpha=0.7))

        title = f"SciEGQA / {q['category']} / {doc_id} (page {page})"
        subtitle = f"Q: {wrap(q['query'], 110)}\nA: {wrap(q['answer'], 110)}"
        subtitle += f"\nGT elements: {len(q['gt_element_ids'])} ({len(gt_regions)} region(s))"
        ax.set_title(f"{title}\n\n{subtitle}", fontsize=9, loc="left")
        ax.axis("off")
        plt.tight_layout()
        out = OUT_DIR / f"dataset_examples_sciegqa_{idx:02d}_{doc_id}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  saved {out}")


def render_mmdocir_examples(n_examples: int = 4):
    root = Path("data/benchmarks/mmdocir")
    # Build element lookup by eid (image_b64 + bbox_pdf + page_size_pdf)
    eid_to_elem = {}
    doc_layouts = defaultdict(list)
    with open(root / "academic_elements.jsonl") as f:
        for line in f:
            r = json.loads(line)
            eid_to_elem[r["eid"]] = r
            doc_layouts[r["doc_id"]].append(r)

    # Load full annotations (need original GT bbox in screenshot space + page_size)
    full_annos = []
    with open(root / "MMDocIR_annotations.jsonl") as f:
        for line in f:
            full_annos.append(json.loads(line))

    queries = []
    with open(root / "academic_queries.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["gt_element_ids"]:
                queries.append(r)

    # Sample diverse queries (variety of types)
    chosen = []
    seen_types = set()
    for q in queries:
        t = q.get("type", "")
        if t in seen_types: continue
        seen_types.add(t)
        chosen.append(q)
        if len(chosen) >= n_examples: break

    # For each chosen, fetch original anno to get GT bbox + assemble image
    for idx, q in enumerate(chosen):
        # parse qid: mmdocir_DDD_QQ
        _, di_s, qi_s = q["qid"].split("_")
        di, qi = int(di_s), int(qi_s)
        # Find academic-only index (qids are sequential over academic docs)
        academic_docs = [d for d in full_annos if "Academic" in d["domain"]]
        anno_doc = academic_docs[di]
        qa_item = anno_doc["questions"][qi]

        # Render each GT region (one per layout_mapping entry, possibly different pages)
        for bi, box_item in enumerate(qa_item["layout_mapping"]):
            page = box_item["page"]
            gt_bbox = box_item["bbox"]  # screenshot coords
            gt_size = box_item["page_size"]

            # We need the page screenshot. MMDocIR pages.parquet has it; we skipped that download.
            # Instead, render bounding boxes ON A BLANK PAGE matching gt_size scaled
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.set_xlim(0, gt_size[0])
            ax.set_ylim(gt_size[1], 0)  # flip y
            ax.set_aspect("equal")
            ax.set_facecolor("white")
            # GT bbox
            ax.add_patch(patches.Rectangle(
                (gt_bbox[0], gt_bbox[1]), gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1],
                linewidth=3, edgecolor="red", facecolor="none", label="GT region"))
            # All layouts on this page (annotated by type)
            for L in doc_layouts[q["doc_id"]]:
                if L["page"] != page: continue
                ps = L["page_size_pdf"]
                sx = gt_size[0] / ps[0]; sy = gt_size[1] / ps[1]
                bb = L["bbox_pdf"]
                bb_scr = [bb[0]*sx, bb[1]*sy, bb[2]*sx, bb[3]*sy]
                color = TYPE_COLOR.get(L["type"], "gray")
                is_gt = L["eid"] in q["gt_element_ids"]
                lw = 2.5 if is_gt else 0.8
                alpha = 1.0 if is_gt else 0.5
                ax.add_patch(patches.Rectangle(
                    (bb_scr[0], bb_scr[1]), bb_scr[2]-bb_scr[0], bb_scr[3]-bb_scr[1],
                    linewidth=lw, edgecolor=color, facecolor="none", alpha=alpha))
                if is_gt:
                    txt_short = (L.get("text") or "")[:40].replace("\n", " ")
                    ax.text(bb_scr[0]+3, bb_scr[1]-3, f"{L['type']}: {txt_short}",
                            fontsize=6, color=color, va="bottom")

            title = f"MMDocIR / {q['doc_id']} (page {page}, GT region {bi+1}/{len(qa_item['layout_mapping'])})"
            subtitle = f"Q: {wrap(q['query'], 110)}\nA: {wrap(q['answer'], 110)}"
            subtitle += f"\ntype={qa_item['type']}, GT elements: {len(q['gt_element_ids'])}"
            ax.set_title(f"{title}\n\n{subtitle}", fontsize=9, loc="left")
            ax.axis("off")
            plt.tight_layout()
            out = OUT_DIR / f"dataset_examples_mmdocir_{idx:02d}_p{page}_{q['doc_id'][:20]}.png"
            plt.savefig(out, dpi=110, bbox_inches="tight")
            plt.close()
            print(f"  saved {out}")
            break  # one image per query


if __name__ == "__main__":
    print("Rendering SciEGQA examples...")
    render_sciegqa_examples()
    print("\nRendering MMDocIR examples...")
    render_mmdocir_examples()
