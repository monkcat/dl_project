"""Build per-document element graph for C1 (evidence pack expansion).

For each doc, extract edges:
  - caption_of: caption text block ↔ figure/table (label + bbox proximity)
  - references: text → figure/table (regex on body)
  - same_section: elements in same section (page-based proximity if no headings)
  - next/prev: reading order

Works for both MMDocIR (academic_elements.jsonl) and SciEGQA (elements.jsonl).

Output: data/benchmarks/{dataset}/element_graph.json
    {doc_id: {nodes: [eid...], edges: [{src, dst, type, weight, confidence}]}}
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


LABEL_RE = re.compile(r"(?i)\b(figure|fig\.?|table|tab\.?)\s*([0-9]+(?:\.[0-9]+)?)\b")
CAPTION_PREFIX_RE = re.compile(r"(?i)^\s*(figure|fig\.?|table|tab\.?)\s*([0-9]+(?:\.[0-9]+)?)[:\.]\s*")
CITATION_NEARBY_RE = re.compile(r"\[[\d, ]+\]|\bet\s+al\b", re.IGNORECASE)

EDGE_WEIGHT = {
    "caption_of":   1.0,
    "references":   0.8,
    "appendix_link":0.7,
    "contains":     0.4,
    "next":         0.3,
    "prev":         0.3,
    "same_section": 0.2,
}


def parse_label(text: str) -> str | None:
    if not text: return None
    m = LABEL_RE.search(text)
    if not m: return None
    kind = "figure" if m.group(1).lower().startswith("fig") else "table"
    return f"{kind}{m.group(2)}"


def parse_caption_label(text: str) -> str | None:
    """Only return label if caption-shaped (starts with 'Figure N:')."""
    if not text: return None
    m = CAPTION_PREFIX_RE.match(text)
    if not m: return None
    kind = "figure" if m.group(1).lower().startswith("fig") else "table"
    return f"{kind}{m.group(2)}"


def bbox_vertical_distance(b1, b2):
    """Vertical pixel distance between two bboxes (0 if overlapping)."""
    if b1[3] < b2[1]: return b2[1] - b1[3]
    if b2[3] < b1[1]: return b1[1] - b2[3]
    return 0


def build_graph_for_doc(elements: list[dict]) -> dict:
    """elements: list of {eid, type, text, bbox (page coords), page}"""
    nodes = [e["eid"] for e in elements]
    edges = []

    # Index by page
    by_page = defaultdict(list)
    for e in elements:
        by_page[e["page"]].append(e)

    # Estimate proximity threshold from page extent (max y across all elements)
    max_y = max((e["bbox"][3] for e in elements), default=1000)
    proximity_thresh = max_y * 0.05  # 5% of page height

    visuals = [e for e in elements if e["type"] in ("image", "table", "equation", "drawing")]
    label_to_visual = {}  # label -> visual element

    # 1a. Label from visual's own text (MMDocIR ocr_text/vlm_text)
    for e in visuals:
        for src in ("text", "ocr_text", "vlm_text"):
            if e.get(src):
                lbl = parse_label(e[src])
                if lbl and lbl not in label_to_visual:
                    label_to_visual[lbl] = e
                    break

    # 1b. Captions: text blocks starting with "Figure N:" — match to nearest visual on same page.
    #     This ALSO populates label_to_visual (critical for SciEGQA where visuals have no text).
    captions = []
    for e in elements:
        if e["type"] != "text": continue
        lbl = parse_caption_label(e.get("text", ""))
        if lbl:
            captions.append((e, lbl))

    for cap_elem, lbl in captions:
        cap_page = cap_elem["page"]
        cap_bbox = cap_elem["bbox"]
        # Prefer existing label match if same page
        tgt = label_to_visual.get(lbl)
        if tgt and tgt["page"] == cap_page:
            edges.append({"src": cap_elem["eid"], "dst": tgt["eid"], "type": "caption_of",
                          "weight": EDGE_WEIGHT["caption_of"], "confidence": 1.0})
            continue
        # Fallback: nearest visual on same page
        same_page_visuals = [v for v in visuals if v["page"] == cap_page]
        if not same_page_visuals:
            continue
        nearest = None; min_d = 1e18
        for v in same_page_visuals:
            d = bbox_vertical_distance(v["bbox"], cap_bbox)
            if d < min_d:
                min_d = d; nearest = v
        if nearest and min_d < proximity_thresh:
            edges.append({"src": cap_elem["eid"], "dst": nearest["eid"], "type": "caption_of",
                          "weight": EDGE_WEIGHT["caption_of"], "confidence": 0.7})
            # Register label→visual so references can use it
            if lbl not in label_to_visual:
                label_to_visual[lbl] = nearest

    # 2. references — text mentions "Figure N" / "Table N" → edge to that visual
    for e in elements:
        if e["type"] != "text": continue
        body = e.get("text", "") or e.get("ocr_text", "") or ""
        if not body: continue
        # All label matches
        for m in LABEL_RE.finditer(body):
            kind = "figure" if m.group(1).lower().startswith("fig") else "table"
            lbl = f"{kind}{m.group(2)}"
            # Filter: skip if within 50 chars of [N] (likely citation)
            start = max(0, m.start() - 50)
            end = min(len(body), m.end() + 50)
            ctx = body[start:end]
            if CITATION_NEARBY_RE.search(ctx) and abs(start - m.start()) < 20:
                # Could be a citation context; partial confidence
                conf = 0.5
            else:
                conf = 0.9
            tgt = label_to_visual.get(lbl)
            if tgt is None: continue
            # Skip self-edge: text element that IS the caption
            if tgt["eid"] == e["eid"]: continue
            edges.append({"src": e["eid"], "dst": tgt["eid"], "type": "references",
                          "weight": EDGE_WEIGHT["references"], "confidence": conf})

    # 3. same_section — elements on same page (proxy for same section)
    for page, page_elems in by_page.items():
        for i, a in enumerate(page_elems):
            for b in page_elems[i+1:]:
                edges.append({"src": a["eid"], "dst": b["eid"], "type": "same_section",
                              "weight": EDGE_WEIGHT["same_section"], "confidence": 0.6})

    # 4. next/prev — element order by (page, y_top) — only consecutive pairs
    sorted_elems = sorted(elements, key=lambda e: (e["page"], e["bbox"][1]))
    for i in range(len(sorted_elems) - 1):
        a = sorted_elems[i]; b = sorted_elems[i+1]
        edges.append({"src": a["eid"], "dst": b["eid"], "type": "next",
                      "weight": EDGE_WEIGHT["next"], "confidence": 1.0})
        edges.append({"src": b["eid"], "dst": a["eid"], "type": "prev",
                      "weight": EDGE_WEIGHT["prev"], "confidence": 1.0})

    return {"nodes": nodes, "edges": edges}


def normalize_elements_mmdocir(jsonl_path: Path) -> dict[str, list[dict]]:
    """MMDocIR: bbox_pdf is in PDF coords. Use bbox_pdf as bbox for graph builder."""
    by_doc = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            by_doc[r["doc_id"]].append({
                "eid": r["eid"], "type": r["type"], "page": r["page"],
                "bbox": r["bbox_pdf"],
                "text": r["text"], "ocr_text": r["ocr_text"], "vlm_text": r["vlm_text"],
            })
    return by_doc


def normalize_elements_sciegqa(jsonl_path: Path) -> dict[str, list[dict]]:
    """SciEGQA: bbox_img in page-pixel coords. Handles both legacy and docling format.
    Falls back to bbox_pdf if bbox_img missing (SPIQA case)."""
    by_doc = defaultdict(list)
    eid_counter = defaultdict(int)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            for e in r["elements"]:
                if "eid" in e:
                    eid = e["eid"]
                else:
                    eid = f"{r['doc_id']}__p{r['page']:03d}__e{eid_counter[r['doc_id']]:04d}"
                    eid_counter[r["doc_id"]] += 1
                bbox = e.get("bbox_img") or e.get("bbox_pdf") or [0, 0, 0, 0]
                # SPIQA extraction separates "caption" type; treat as text so
                # parse_caption_label() picks it up for caption_of edges.
                etype = "text" if e["type"] == "caption" else e["type"]
                by_doc[r["doc_id"]].append({
                    "eid": eid, "type": etype, "page": r["page"],
                    "bbox": bbox,
                    "text": e.get("text", ""), "ocr_text": "", "vlm_text": "",
                })
    return by_doc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mmdocir", "sciegqa", "spiqa"], required=True)
    args = ap.parse_args()

    if args.dataset == "mmdocir":
        elems = normalize_elements_mmdocir(Path("data/benchmarks/mmdocir/academic_elements.jsonl"))
        out_path = Path("data/benchmarks/mmdocir/element_graph.json")
    elif args.dataset == "spiqa":
        # SPIQA test-A docling extraction (uses same per-page jsonl format with eid)
        elems = normalize_elements_sciegqa(Path("data/benchmarks/spiqa/test-A/elements_docling.jsonl"))
        out_path = Path("data/benchmarks/spiqa/element_graph_docling.json")
    else:
        # SciEGQA: prefer docling
        docling_path = Path("data/benchmarks/sciegqa/elements_docling.jsonl")
        if docling_path.exists():
            elems = normalize_elements_sciegqa(docling_path)
            out_path = Path("data/benchmarks/sciegqa/element_graph_docling.json")
        else:
            elems = normalize_elements_sciegqa(Path("data/benchmarks/sciegqa/elements.jsonl"))
            out_path = Path("data/benchmarks/sciegqa/element_graph.json")

    graphs = {}
    edge_type_total = defaultdict(int)
    for doc_id, doc_elems in elems.items():
        g = build_graph_for_doc(doc_elems)
        graphs[doc_id] = g
        for e in g["edges"]:
            edge_type_total[e["type"]] += 1

    out_path.write_text(json.dumps(graphs))
    print(f"docs: {len(graphs)}, total edges: {sum(edge_type_total.values())}")
    for t, c in sorted(edge_type_total.items()):
        print(f"  {t}: {c}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
