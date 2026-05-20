"""Re-extract SciEGQA elements preserving docling labels (section_header, caption, ...).

Previous extraction (`extract_sciegqa_docling.py`) collapsed all docling label
types into {text, image, table} and SKIPPED section_header items entirely.
This destroyed:
  - section_header text (needed for clean section labels and role classification)
  - caption label (caption nodes were inferred later via heuristic instead of
    being identified directly by docling)
  - formula label (equations were merged into text)
  - list_item label (could be useful for structural prior)

This v2 extractor:
  - Preserves docling label → element type mapping (no skipping).
  - Outputs:
      data/benchmarks/sciegqa/elements_docling_v2.jsonl
      data/benchmarks/sciegqa/section_paths_docling_v2.json
  - Preserves the same eid scheme (doc__pPPP__eEEEE) so v1 GT mappings work
    after a small re-mapping (we'll publish a converter for this).

Run:
    python -m pipeline.extract_sciegqa_docling_v2
"""
from __future__ import annotations

import io
import json
import re
import tarfile
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

from PIL import Image


ROOT = Path("data/benchmarks/sciegqa")
PDF_DIR = ROOT / "PDF"
IMAGES_TAR = ROOT / "Images.tar"
OUT_JSONL = ROOT / "elements_docling_v2.jsonl"
SECTION_PATHS_JSON = ROOT / "section_paths_docling_v2.json"


NUMERIC_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+")
ROMAN_RE = re.compile(r"^\s*([IVXLCDM]+)\.\s+", re.IGNORECASE)
LETTER_RE = re.compile(r"^\s*([A-Z])\.\s+")


def infer_section_level(title: str) -> int:
    if NUMERIC_RE.match(title):
        m = NUMERIC_RE.match(title)
        return len(m.group(1).split("."))
    if ROMAN_RE.match(title):
        return 1
    if LETTER_RE.match(title):
        return 2
    return 1


def collect_page_sizes() -> dict[tuple[str, int], tuple[int, int]]:
    sizes: dict[tuple[str, int], tuple[int, int]] = {}
    with tarfile.open(IMAGES_TAR) as t:
        for m in t.getmembers():
            if not m.name.endswith(".png"):
                continue
            stem = Path(m.name).stem
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            doc_id, page_s = parts
            try:
                page = int(page_s)
            except ValueError:
                continue
            f = t.extractfile(m)
            if f is None:
                continue
            img = Image.open(io.BytesIO(f.read()))
            sizes[(doc_id, page)] = img.size
    return sizes


def bbox_to_img(bbox, page_size_pdf, page_size_img):
    pdf_w, pdf_h = page_size_pdf
    img_w, img_h = page_size_img
    sx = img_w / pdf_w
    sy = img_h / pdf_h
    x0 = bbox.l * sx
    x1 = bbox.r * sx
    y0 = (pdf_h - bbox.t) * sy
    y1 = (pdf_h - bbox.b) * sy
    return [x0, y0, x1, y1]


# Map docling label string → our element type
DOCLING_LABEL_MAP = {
    "section_header": "section_header",
    "caption":        "caption",
    "formula":        "equation",
    "list_item":      "text",     # keep as text (could be separate type later)
    "text":           "text",
    "footnote":       None,       # drop
    "page_header":    None,       # drop (page numbers)
    "page_footer":    None,
    "title":          "section_header",
}


def extract_doc(pdf_path: Path, page_sizes: dict, conv) -> tuple[list[dict], dict[str, list[int]]]:
    """Returns (page_records, section_paths_for_eids)."""
    result = conv.convert(pdf_path)
    doc = result.document
    doc_id = pdf_path.name.split("_")[0]

    pages = sorted(doc.pages.keys())
    by_page = defaultdict(list)
    eid_paths: dict[str, list[int]] = {}

    cur_path: list[int] = []
    h_counters: dict[int, int] = {}
    elem_idx_in_section = 0
    eid_counter = 0

    # Aggregate items
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

    # Reading order: by (page, decreasing y_top in PDF coords)
    all_items.sort(key=lambda x: (x[2].page_no, -x[2].bbox.t))

    for kind, item, prov in all_items:
        page_num = prov.page_no
        page_size_pdf = (doc.pages[page_num].size.width, doc.pages[page_num].size.height)
        page_size_img = page_sizes.get((doc_id, page_num))
        if page_size_img is None:
            continue
        bbox_img = bbox_to_img(prov.bbox, page_size_pdf, page_size_img)

        if kind == "text_like":
            raw_label = str(getattr(item, "label", "")).lower()
            text = (item.text or "").strip()
            mapped = DOCLING_LABEL_MAP.get(raw_label)
            if mapped is None:
                continue
            if mapped == "section_header":
                # Update section hierarchy
                level = infer_section_level(text)
                if level <= 0: level = 1
                h_counters = {k: v for k, v in h_counters.items() if k < level}
                h_counters[level] = h_counters.get(level, 0) + 1
                cur_path = [h_counters[k] for k in sorted(h_counters)]
                elem_idx_in_section = 0
            else:
                elem_idx_in_section += 1
            if not text or len(text) < 2:
                continue
            eid = f"{doc_id}__p{page_num:03d}__e{eid_counter:04d}"
            eid_counter += 1
            entry = {
                "type": mapped,
                "docling_label": raw_label,
                "bbox_img": bbox_img,
                "bbox_pdf": [prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t],
                "text": text,
                "eid": eid,
            }
            by_page[page_num].append(entry)
            eid_paths[eid] = list(cur_path) + [elem_idx_in_section]
        elif kind in ("picture", "table"):
            elem_idx_in_section += 1
            eid = f"{doc_id}__p{page_num:03d}__e{eid_counter:04d}"
            eid_counter += 1
            etype = "figure" if kind == "picture" else "table"
            by_page[page_num].append({
                "type": etype,
                "docling_label": kind,
                "bbox_img": bbox_img,
                "bbox_pdf": [prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t],
                "text": "",
                "eid": eid,
            })
            eid_paths[eid] = list(cur_path) + [elem_idx_in_section]

    records = [{
        "doc_id": doc_id, "page": p, "elements": by_page.get(p, []),
    } for p in pages]
    return records, eid_paths


def main():
    print("Collecting page image sizes...")
    page_sizes = collect_page_sizes()
    print(f"  page sizes for {len(page_sizes)} pages")

    pdfs = sorted(PDF_DIR.rglob("*.pdf"))
    print(f"PDFs to process: {len(pdfs)}")

    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()

    all_section_paths: dict[str, dict[str, list[int]]] = {}
    type_counts: dict[str, int] = {}
    n_pages = 0

    OUT_JSONL.unlink(missing_ok=True)
    with open(OUT_JSONL, "a") as fout:
        for i, pdf in enumerate(pdfs, 1):
            t0 = time.time()
            try:
                records, paths = extract_doc(pdf, page_sizes, conv)
            except Exception as e:
                print(f"  [{i}/{len(pdfs)}] FAILED {pdf.name[:50]}: {e}")
                continue
            doc_id = pdf.name.split("_")[0]
            all_section_paths[doc_id] = paths
            for r in records:
                fout.write(json.dumps(r) + "\n")
                n_pages += 1
                for e in r["elements"]:
                    type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
            if i % 5 == 0 or i == len(pdfs):
                print(f"  [{i}/{len(pdfs)}] {pdf.name[:50]} ({time.time()-t0:.1f}s)  "
                      f"types={type_counts}")

    SECTION_PATHS_JSON.write_text(json.dumps(all_section_paths))
    print(f"\nDone. Total pages={n_pages}")
    print(f"final element type counts: {type_counts}")
    print(f"saved elements to {OUT_JSONL}")
    print(f"saved section_paths to {SECTION_PATHS_JSON}")


if __name__ == "__main__":
    main()
