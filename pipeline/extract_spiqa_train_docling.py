"""Run docling on the downloaded SPIQA train PDFs + match captions + filter.

Output:
  data/benchmarks/spiqa/train_subset/elements_docling.jsonl
  data/benchmarks/spiqa/train_subset/section_paths_docling.json
  data/benchmarks/spiqa/train_subset/element_graph_docling.json
  data/benchmarks/spiqa/train_subset/spiqa_fig_to_element.json (only matched figures)
  data/benchmarks/spiqa/train_subset/kept_papers.json (papers passing quality gate)
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

warnings.filterwarnings("ignore")


ROOT = Path("data/benchmarks/spiqa")
TRAIN_JSON = ROOT / "train_val" / "SPIQA_train.json"
PDF_DIR = ROOT / "train_subset" / "pdfs"
OUT_DIR = ROOT / "train_subset"


ROMAN_RE = re.compile(r"^\s*([IVXLCDM]+)\.\s+", re.IGNORECASE)
LETTER_RE = re.compile(r"^\s*([A-Z])\.\s+")
NUMERIC_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+")
CAPTION_PREFIX_RE = re.compile(r"^\s*(figure|fig\.?|table|tab\.?)\s*([0-9]+(?:\.[0-9]+)?)[:\.]?\s*", re.IGNORECASE)
LABEL_RE = re.compile(r"(?i)\b(figure|fig\.?|table|tab\.?)\s*([0-9]+(?:\.[0-9]+)?)\b")


def infer_section_level(title: str) -> int:
    if NUMERIC_RE.match(title):
        return len(NUMERIC_RE.match(title).group(1).split("."))
    if ROMAN_RE.match(title): return 1
    if LETTER_RE.match(title): return 2
    return 1


def normalize_caption(s: str) -> str:
    if not s: return ""
    s = re.sub(r"\\(cite|ref|label|footnote|citep|citet)\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "")
    s = CAPTION_PREFIX_RE.sub("", s, count=1)
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_label(text: str) -> str | None:
    m = LABEL_RE.search(text or "")
    if not m: return None
    kind = "figure" if m.group(1).lower().startswith("fig") else "table"
    return f"{kind}{m.group(2)}"


def bbox_vertical_dist(b1, b2):
    if b1[3] < b2[1]: return b2[1] - b1[3]
    if b2[3] < b1[1]: return b1[1] - b2[3]
    return 0


def extract_doc(pdf_path: Path, conv):
    """Returns (page_records, eid->section_path, visual_to_caption_list)."""
    result = conv.convert(pdf_path)
    doc = result.document
    doc_id = pdf_path.stem

    pages = sorted(doc.pages.keys())
    by_page = defaultdict(list)
    eid_paths: dict[str, list[int]] = {}

    cur_path: list[int] = []
    h_counters: dict[int, int] = {}
    elem_idx_in_section = 0
    eid_counter = 0

    all_items = []
    for t in doc.texts:
        if t.prov: all_items.append(("text", t, t.prov[0]))
    for p in doc.pictures:
        if p.prov: all_items.append(("picture", p, p.prov[0]))
    for tbl in doc.tables:
        if tbl.prov: all_items.append(("table", tbl, tbl.prov[0]))
    all_items.sort(key=lambda x: (x[2].page_no, -x[2].bbox.t))

    pending_caption: dict[int, str] = {}
    visual_to_caption: list[dict] = []

    for kind, item, prov in all_items:
        page_num = prov.page_no
        bbox_pdf = [prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t]

        if kind == "text":
            label = str(getattr(item, "label", "")).lower()
            text = (item.text or "").strip()
            if "header" in label and "page_header" not in label:
                level = max(1, infer_section_level(text))
                h_counters = {k: v for k, v in h_counters.items() if k < level}
                h_counters[level] = h_counters.get(level, 0) + 1
                cur_path = [h_counters[k] for k in sorted(h_counters)]
                elem_idx_in_section = 0
                continue
            if label == "page_header" or "footnote" in label or "reference" in label:
                continue
            if not text or len(text) < 5:
                continue
            is_caption = bool(CAPTION_PREFIX_RE.match(text))
            eid = f"{doc_id}__p{page_num:03d}__e{eid_counter:04d}"
            eid_counter += 1
            elem_idx_in_section += 1
            elem_type = "caption" if is_caption else "text"
            by_page[page_num].append({
                "type": elem_type, "bbox_pdf": bbox_pdf, "text": text, "eid": eid,
            })
            eid_paths[eid] = list(cur_path) + [elem_idx_in_section]
            if is_caption:
                pending_caption[page_num] = text
        elif kind in ("picture", "table"):
            elem_type = "image" if kind == "picture" else "table"
            eid = f"{doc_id}__p{page_num:03d}__e{eid_counter:04d}"
            eid_counter += 1
            elem_idx_in_section += 1
            caption_text = pending_caption.pop(page_num, "")
            by_page[page_num].append({
                "type": elem_type, "bbox_pdf": bbox_pdf, "text": "",
                "eid": eid, "associated_caption": caption_text,
            })
            eid_paths[eid] = list(cur_path) + [elem_idx_in_section]
            visual_to_caption.append({
                "eid": eid, "type": elem_type, "page": page_num,
                "caption": caption_text, "doc_id": doc_id,
            })

    records = [{"doc_id": doc_id, "page": p, "elements": by_page.get(p, [])} for p in pages]
    return records, eid_paths, visual_to_caption


def match_spiqa_figures(spiqa_paper: dict, docling_elements: list[dict],
                          captions_blocks: list[dict]) -> dict[str, dict]:
    """For each SPIQA figure in this paper, find best docling element via caption text similarity."""
    mapping = {}
    for fname, meta in spiqa_paper.get("all_figures", {}).items():
        spiqa_cap = normalize_caption(meta.get("caption", ""))
        spiqa_label = parse_label(meta.get("caption", ""))
        spiqa_type = meta.get("content_type", "figure")
        target_type = "image" if spiqa_type == "figure" else "table"

        # Step 1: best caption block
        best_cap = None; best_score = 0.0
        for c in captions_blocks:
            label = parse_label(c["text"])
            base = 1.0 if (spiqa_label and label == spiqa_label) else 0.0
            cap_norm = normalize_caption(c["text"])
            if not cap_norm or not spiqa_cap:
                score = base
            else:
                score = base * 0.3 + SequenceMatcher(None, spiqa_cap[:200], cap_norm[:200]).ratio() * 0.7
            if score > best_score:
                best_score = score; best_cap = c
        if not best_cap or best_score < 0.3:
            continue

        # Step 2: nearest visual on same page
        cap_page = best_cap["page"]
        cap_bbox = best_cap["bbox_pdf"]
        cands = [e for e in docling_elements
                 if e.get("page") == cap_page and e["type"] == target_type]
        if not cands:
            cands = [e for e in docling_elements
                     if e.get("page") == cap_page and e["type"] in ("image", "table")]
            if not cands: continue
        best_v = min(cands, key=lambda v: bbox_vertical_dist(v["bbox_pdf"], cap_bbox))
        mapping[fname] = {
            "eid": best_v["eid"],
            "caption_eid": best_cap["eid"],
            "page": cap_page,
            "caption_similarity": round(best_score, 3),
        }
    return mapping


EDGE_WEIGHTS = {"caption_of": 1.0, "references": 0.3}


def build_graph_for_paper(elements: list[dict], paper_id: str) -> list[dict]:
    """Build C1 graph edges for one paper."""
    by_page = defaultdict(list)
    for e in elements:
        by_page[e.get("page", 0)].append(e)

    # Label → visual map (from caption blocks)
    visuals = [e for e in elements if e["type"] in ("image", "table")]
    label_to_visual: dict[str, dict] = {}

    edges = []
    captions = [e for e in elements if e["type"] == "caption"]

    # caption_of edges
    for cap in captions:
        lbl = parse_label(cap.get("text", ""))
        page = cap.get("page", 0)
        cap_bbox = cap.get("bbox_pdf", [0, 0, 0, 0])
        # match label to visual
        if lbl and lbl in label_to_visual:
            tgt = label_to_visual[lbl]
            if tgt.get("page") == page:
                edges.append({"src": cap["eid"], "dst": tgt["eid"], "type": "caption_of",
                              "weight": 1.0, "confidence": 1.0})
                continue
        # nearest visual on same page
        same_page_visuals = [v for v in visuals if v.get("page") == page]
        if not same_page_visuals: continue
        nearest = min(same_page_visuals,
                       key=lambda v: bbox_vertical_dist(v["bbox_pdf"], cap_bbox))
        d = bbox_vertical_dist(nearest["bbox_pdf"], cap_bbox)
        if d < 100:
            edges.append({"src": cap["eid"], "dst": nearest["eid"], "type": "caption_of",
                          "weight": 1.0, "confidence": 0.7})
            if lbl: label_to_visual[lbl] = nearest

    # references edges: text body mentions "Figure N" -> figure
    for e in elements:
        if e["type"] != "text": continue
        body = e.get("text", "")
        if not body: continue
        for m in LABEL_RE.finditer(body):
            kind = "figure" if m.group(1).lower().startswith("fig") else "table"
            lbl = f"{kind}{m.group(2)}"
            tgt = label_to_visual.get(lbl)
            if tgt is None or tgt["eid"] == e["eid"]: continue
            edges.append({"src": e["eid"], "dst": tgt["eid"], "type": "references",
                          "weight": 0.8, "confidence": 0.9})

    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match_threshold", type=float, default=0.5,
                    help="Min caption similarity to keep a figure")
    ap.add_argument("--min_matched_per_paper", type=int, default=2,
                    help="Drop papers with fewer matched figures")
    args = ap.parse_args()

    print("Loading SPIQA train metadata...")
    train = json.load(open(TRAIN_JSON))
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"PDFs to process: {len(pdfs)}")

    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()

    out_elements = []   # all page records to write
    section_paths_all: dict[str, dict[str, list[int]]] = {}
    fig_mapping: dict[str, dict] = {}
    kept_papers: list[str] = []
    docs_skipped = 0
    docs_low_match = 0
    n_total_spiqa_figs = 0
    n_matched_figs = 0

    t0 = time.time()
    for i, pdf in enumerate(pdfs, 1):
        paper_id = pdf.stem
        if paper_id not in train:
            docs_skipped += 1
            continue
        try:
            records, paths, visuals = extract_doc(pdf, conv)
        except Exception as e:
            print(f"  [{i}/{len(pdfs)}] FAILED {paper_id}: {e}")
            docs_skipped += 1
            continue

        # Flatten elements for graph building
        flat_elements: list[dict] = []
        for r in records:
            for e in r["elements"]:
                flat_elements.append({**e, "page": r["page"]})
        # Captions as separate list
        captions_blocks = [e for e in flat_elements if e["type"] == "caption"]

        # Match SPIQA figures
        paper_mapping = match_spiqa_figures(train[paper_id], flat_elements, captions_blocks)
        # Filter low-similarity matches
        paper_mapping = {fn: info for fn, info in paper_mapping.items()
                           if info["caption_similarity"] >= args.match_threshold}
        n_spiqa = len(train[paper_id].get("all_figures", {}))
        n_total_spiqa_figs += n_spiqa
        n_matched_figs += len(paper_mapping)

        if len(paper_mapping) < args.min_matched_per_paper:
            docs_low_match += 1
            continue  # skip this paper entirely

        # Keep
        kept_papers.append(paper_id)
        section_paths_all[paper_id] = paths
        for fname, info in paper_mapping.items():
            fig_mapping[fname] = info

        for r in records:
            out_elements.append(r)

        if i % 20 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(pdfs) - i) / rate
            print(f"  [{i}/{len(pdfs)}] kept={len(kept_papers)} "
                  f"low_match={docs_low_match} skip={docs_skipped} "
                  f"figs={n_matched_figs}/{n_total_spiqa_figs} "
                  f"({rate*60:.1f}/min, ETA {eta/60:.1f}m)")

    # Build graph from kept elements (per paper)
    print("\nBuilding C1 graph for kept papers...")
    graph_all: dict[str, dict] = {}
    by_paper_elems: dict[str, list[dict]] = defaultdict(list)
    for r in out_elements:
        for e in r["elements"]:
            by_paper_elems[r["doc_id"]].append({**e, "page": r["page"]})
    for paper_id, elems in by_paper_elems.items():
        edges = build_graph_for_paper(elems, paper_id)
        nodes = [e["eid"] for e in elems]
        graph_all[paper_id] = {"nodes": nodes, "edges": edges}

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    elem_path = OUT_DIR / "elements_docling.jsonl"
    section_path = OUT_DIR / "section_paths_docling.json"
    graph_path = OUT_DIR / "element_graph_docling.json"
    mapping_path = OUT_DIR / "spiqa_fig_to_element.json"
    kept_path = OUT_DIR / "kept_papers.json"

    with open(elem_path, "w") as f:
        for r in out_elements:
            f.write(json.dumps(r) + "\n")
    section_path.write_text(json.dumps(section_paths_all))
    graph_path.write_text(json.dumps(graph_all))
    mapping_path.write_text(json.dumps(fig_mapping, indent=2))
    kept_path.write_text(json.dumps(kept_papers, indent=2))

    print(f"\n=== Train subset built ===")
    print(f"  PDFs processed: {len(pdfs)}")
    print(f"  Papers kept (>= {args.min_matched_per_paper} matched figs): {len(kept_papers)}")
    print(f"  Papers low-match dropped: {docs_low_match}")
    print(f"  Papers extraction failed: {docs_skipped}")
    print(f"  SPIQA figures matched: {n_matched_figs}/{n_total_spiqa_figs} "
          f"({n_matched_figs/max(1,n_total_spiqa_figs):.1%})")
    n_edges = sum(len(g["edges"]) for g in graph_all.values())
    print(f"  Graph edges: {n_edges} (avg {n_edges/max(1,len(kept_papers)):.0f}/paper)")
    print(f"  Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
