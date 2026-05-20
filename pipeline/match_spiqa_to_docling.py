"""Better SPIQA figure → docling element matching.

Strategy:
  1. For each SPIQA figure: find docling caption block(s) across the doc whose
     normalized text best matches SPIQA caption (across all pages, all caption blocks)
  2. On the page of best caption, find the nearest visual element (image/table)
     by vertical proximity
  3. Confidence = caption similarity × normalized proximity score
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("data/benchmarks/spiqa/test-A")
SPIQA_JSON = ROOT / "SPIQA_testA.json"
ELEMENTS_JSONL = ROOT / "elements_docling.jsonl"
OUT_MAPPING = ROOT / "spiqa_fig_to_element.json"

CAPTION_PREFIX_RE = re.compile(r"^\s*(figure|fig\.?|table|tab\.?)\s*([0-9]+(?:\.[0-9]+)?)[:\.]?\s*", re.IGNORECASE)
LABEL_RE = re.compile(r"(?i)\b(figure|fig\.?|table|tab\.?)\s*([0-9]+(?:\.[0-9]+)?)\b")


def normalize_caption(s: str) -> str:
    if not s: return ""
    s = re.sub(r"\\(cite|ref|label|footnote|citep|citet)\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "")
    s = CAPTION_PREFIX_RE.sub("", s, count=1)
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_label(text: str) -> str | None:
    if not text: return None
    m = LABEL_RE.search(text)
    if not m: return None
    kind = "figure" if m.group(1).lower().startswith("fig") else "table"
    return f"{kind}{m.group(2)}"


def bbox_vertical_dist(b1, b2):
    if b1[3] < b2[1]: return b2[1] - b1[3]
    if b2[3] < b1[1]: return b1[1] - b2[3]
    return 0


def main():
    spiqa = json.loads(SPIQA_JSON.read_text())

    # Load all docling elements grouped by doc
    by_doc: dict[str, list[dict]] = defaultdict(list)
    with open(ELEMENTS_JSONL) as f:
        for line in f:
            r = json.loads(line)
            for e in r["elements"]:
                e2 = {**e, "doc_id": r["doc_id"], "page": r["page"]}
                by_doc[r["doc_id"]].append(e2)
    print(f"Loaded elements for {len(by_doc)} docs")

    mapping = {}
    n_total = 0
    n_matched = 0
    n_no_docling = 0
    sim_dist = []

    for paper_id, paper in spiqa.items():
        doc_elems = by_doc.get(paper_id, [])
        if not doc_elems:
            n_no_docling += len(paper["all_figures"])
            continue
        # Captions and visuals in this doc
        captions = [e for e in doc_elems if e["type"] == "caption"]
        visuals = [e for e in doc_elems if e["type"] in ("image", "table")]

        # Pre-compute normalized text for each caption
        for c in captions:
            c["_norm"] = normalize_caption(c["text"])
            c["_label"] = parse_label(c["text"])

        for fname, meta in paper["all_figures"].items():
            n_total += 1
            spiqa_cap_raw = meta.get("caption", "")
            spiqa_norm = normalize_caption(spiqa_cap_raw)
            spiqa_label = parse_label(spiqa_cap_raw)
            spiqa_type = meta.get("content_type", "figure")
            target_visual_type = "image" if spiqa_type == "figure" else "table"

            # === Step 1: find best caption match ===
            best_cap = None; best_cap_score = 0.0
            for c in captions:
                # Boost score if label matches exactly
                if spiqa_label and c["_label"] == spiqa_label:
                    base = 1.0
                else:
                    base = 0.0
                if not c["_norm"] or not spiqa_norm:
                    score = base
                else:
                    score = base * 0.3 + SequenceMatcher(None, spiqa_norm[:200], c["_norm"][:200]).ratio() * 0.7
                if score > best_cap_score:
                    best_cap_score = score
                    best_cap = c
            if not best_cap or best_cap_score < 0.3:
                continue

            # === Step 2: nearest visual element on same page ===
            cap_page = best_cap["page"]
            cap_bbox = best_cap["bbox_pdf"]
            cands = [v for v in visuals if v["page"] == cap_page and v["type"] == target_visual_type]
            if not cands:
                # Fall back to any visual on the same page
                cands = [v for v in visuals if v["page"] == cap_page]
                if not cands:
                    continue

            best_v = None; best_dist = 1e18
            for v in cands:
                d = bbox_vertical_dist(v["bbox_pdf"], cap_bbox)
                if d < best_dist:
                    best_dist = d; best_v = v

            if best_v is None:
                continue
            mapping[fname] = {
                "eid": best_v["eid"],
                "caption_eid": best_cap["eid"],
                "page": cap_page,
                "caption_similarity": round(best_cap_score, 3),
                "proximity": round(best_dist, 1),
            }
            n_matched += 1
            sim_dist.append(best_cap_score)

    OUT_MAPPING.write_text(json.dumps(mapping, indent=2))
    print(f"SPIQA figures total: {n_total}")
    print(f"  matched: {n_matched} ({n_matched/n_total:.1%})")
    print(f"  papers with no docling output: contributed {n_no_docling} unmatched")
    import statistics as st
    if sim_dist:
        print(f"  caption similarity distribution:")
        print(f"    mean={st.mean(sim_dist):.3f}, median={st.median(sim_dist):.3f}")
        print(f"    ≥0.9: {sum(1 for s in sim_dist if s>=0.9)}, "
              f"≥0.7: {sum(1 for s in sim_dist if s>=0.7)}, "
              f"≥0.5: {sum(1 for s in sim_dist if s>=0.5)}, "
              f"≥0.3: {sum(1 for s in sim_dist if s>=0.3)}")
    print(f"saved to {OUT_MAPPING}")


if __name__ == "__main__":
    main()
