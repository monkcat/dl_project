"""Run docling on SPIQA test-A PDFs and match each SPIQA figure to a docling
element by caption similarity.

Output:
  data/benchmarks/spiqa/test-A/elements_docling.jsonl  (all docling elements per page)
  data/benchmarks/spiqa/test-A/section_paths_docling.json  (section_path per element)
  data/benchmarks/spiqa/test-A/spiqa_fig_to_element.json  (SPIQA fname -> eid mapping)
"""
from __future__ import annotations

import json
import re
import time
import warnings
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

warnings.filterwarnings("ignore")


ROOT = Path("data/benchmarks/spiqa/test-A")
PDF_DIR = ROOT / "pdfs"
SPIQA_JSON = ROOT / "SPIQA_testA.json"
OUT_ELEMENTS = ROOT / "elements_docling.jsonl"
OUT_SECTIONS = ROOT / "section_paths_docling.json"
OUT_MAPPING = ROOT / "spiqa_fig_to_element.json"


ROMAN_RE = re.compile(r"^\s*([IVXLCDM]+)\.\s+", re.IGNORECASE)
LETTER_RE = re.compile(r"^\s*([A-Z])\.\s+")
NUMERIC_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+")
CAPTION_PREFIX_RE = re.compile(r"^\s*(figure|fig\.?|table|tab\.?)\s*([0-9]+(?:\.[0-9]+)?)[:\.]?\s*", re.IGNORECASE)


def infer_section_level(title: str) -> int:
    if NUMERIC_RE.match(title):
        m = NUMERIC_RE.match(title)
        return len(m.group(1).split("."))
    if ROMAN_RE.match(title): return 1
    if LETTER_RE.match(title): return 2
    return 1


def normalize_caption(s: str) -> str:
    """Strip LaTeX commands + label prefix, normalize whitespace."""
    s = re.sub(r"\\(cite|ref|label|footnote|citep|citet|emph|textbf|textit)\{[^}]*\}", "", s or "")
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "")
    s = CAPTION_PREFIX_RE.sub("", s, count=1)
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_doc(pdf_path: Path, conv) -> tuple[list[dict], dict[str, list[int]], list[dict]]:
    """Returns (page_records, section_paths_by_eid, visual_elements_with_caption)"""
    result = conv.convert(pdf_path)
    doc = result.document
    doc_id = pdf_path.stem  # e.g. 1611.04684v1

    pages = sorted(doc.pages.keys())
    by_page = defaultdict(list)
    eid_paths: dict[str, list[int]] = {}

    cur_path: list[int] = []
    h_counters: dict[int, int] = {}
    elem_idx_in_section = 0
    eid_counter = 0

    all_items = []
    for t in doc.texts:
        if not t.prov: continue
        all_items.append(("text_like", t, t.prov[0]))
    for p in doc.pictures:
        if not p.prov: continue
        all_items.append(("picture", p, p.prov[0]))
    for tbl in doc.tables:
        if not tbl.prov: continue
        all_items.append(("table", tbl, tbl.prov[0]))

    all_items.sort(key=lambda x: (x[2].page_no, -x[2].bbox.t))

    # Track captions to associate with the immediately preceding/following figure/table
    # (captions in docling are tagged as 'caption' label)
    pending_caption: dict[int, str] = {}  # page_no -> last unbound caption text

    visual_to_caption: list[dict] = []  # for SPIQA matching

    for kind, item, prov in all_items:
        page_num = prov.page_no
        bbox_pdf = [prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t]

        if kind == "text_like":
            label = str(getattr(item, "label", "")).lower()
            text = (item.text or "").strip()
            if "header" in label and "page_header" not in label:
                level = infer_section_level(text)
                if level <= 0: level = 1
                h_counters = {k: v for k, v in h_counters.items() if k < level}
                h_counters[level] = h_counters.get(level, 0) + 1
                cur_path = [h_counters[k] for k in sorted(h_counters)]
                elem_idx_in_section = 0
                continue
            if label == "page_header" or "footnote" in label or "reference" in label:
                continue
            if not text or len(text) < 5:
                continue
            # Caption detection: starts with "Figure N:" or "Table N:"
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
            # Attach pending caption (if any) for SPIQA matching
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


def main():
    spiqa = json.loads(SPIQA_JSON.read_text())
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"PDFs: {len(pdfs)}")

    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()

    all_section_paths: dict[str, dict[str, list[int]]] = {}
    all_visuals: dict[str, list[dict]] = {}  # doc_id -> list of visual elements

    OUT_ELEMENTS.unlink(missing_ok=True)
    n_pages = n_text = n_img = n_tbl = n_cap = 0

    with open(OUT_ELEMENTS, "a") as fout:
        for i, pdf in enumerate(pdfs, 1):
            t0 = time.time()
            try:
                records, paths, visuals = extract_doc(pdf, conv)
            except Exception as e:
                print(f"  [{i}/{len(pdfs)}] FAILED {pdf.name}: {e}")
                continue
            doc_id = pdf.stem
            all_section_paths[doc_id] = paths
            all_visuals[doc_id] = visuals
            for r in records:
                fout.write(json.dumps(r) + "\n")
                n_pages += 1
                for e in r["elements"]:
                    t = e["type"]
                    if t == "text": n_text += 1
                    elif t == "image": n_img += 1
                    elif t == "table": n_tbl += 1
                    elif t == "caption": n_cap += 1
            if i % 10 == 0 or i == len(pdfs):
                print(f"  [{i}/{len(pdfs)}] {pdf.name} ({time.time()-t0:.1f}s)  "
                      f"pages={n_pages} text={n_text} img={n_img} tbl={n_tbl} cap={n_cap}")

    OUT_SECTIONS.write_text(json.dumps(all_section_paths))
    print(f"\nDone. pages={n_pages} text={n_text} img={n_img} tbl={n_tbl} cap={n_cap}")

    # === Match SPIQA figures to docling visual elements ===
    print(f"\n=== Matching SPIQA figures to docling elements ===")
    mapping = {}     # spiqa_fname -> {eid, similarity}
    n_papers = 0
    n_spiqa_total = 0
    n_matched = 0
    for paper_id, paper in spiqa.items():
        n_papers += 1
        docling_visuals = all_visuals.get(paper_id, [])
        if not docling_visuals:
            continue
        for fname, meta in paper["all_figures"].items():
            n_spiqa_total += 1
            spiqa_cap = normalize_caption(meta.get("caption", ""))
            spiqa_type = meta.get("content_type", "figure")
            best_eid = None
            best_score = 0.0
            for v in docling_visuals:
                if v["type"] != ("image" if spiqa_type == "figure" else "table"):
                    continue
                if not v["caption"]:
                    continue
                v_cap = normalize_caption(v["caption"])
                if not v_cap or not spiqa_cap:
                    continue
                score = SequenceMatcher(None, spiqa_cap[:200], v_cap[:200]).ratio()
                if score > best_score:
                    best_score = score
                    best_eid = v["eid"]
            if best_eid and best_score >= 0.5:
                mapping[fname] = {"eid": best_eid, "similarity": round(best_score, 3)}
                n_matched += 1

    OUT_MAPPING.write_text(json.dumps(mapping, indent=2))
    print(f"papers with docling output: {len(all_visuals)}/{n_papers}")
    print(f"SPIQA figures matched (sim ≥ 0.5): {n_matched}/{n_spiqa_total} ({n_matched/n_spiqa_total:.1%})")
    print(f"\nsaved elements to {OUT_ELEMENTS}")
    print(f"saved section_paths to {OUT_SECTIONS}")
    print(f"saved SPIQA→element mapping to {OUT_MAPPING}")


if __name__ == "__main__":
    main()
