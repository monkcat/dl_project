"""Re-extract SciEGQA elements using docling with proper section hierarchy.

Output: data/benchmarks/sciegqa/elements_docling.jsonl
   Each line: one page record matching elements.jsonl format, PLUS section_path
   inferred from docling's section_header items.
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
OUT_JSONL = ROOT / "elements_docling.jsonl"
SECTION_PATHS_JSON = ROOT / "section_paths_docling.json"


# Prefix patterns to infer section depth
ROMAN_RE = re.compile(r"^\s*([IVXLCDM]+)\.\s+", re.IGNORECASE)
LETTER_RE = re.compile(r"^\s*([A-Z])\.\s+")
NUMERIC_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+")


def infer_section_level(title: str) -> int:
    """Infer hierarchical level (1, 2, 3, ...) from header prefix."""
    if NUMERIC_RE.match(title):
        m = NUMERIC_RE.match(title)
        return len(m.group(1).split("."))
    if ROMAN_RE.match(title):
        return 1  # Roman numerals: top-level
    if LETTER_RE.match(title):
        return 2  # Letters: subsection
    return 1  # Default: top-level


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
    """Docling bbox is in PDF coords with y-flipped (origin bottom-left).
    Convert to image pixel coords (origin top-left).
    """
    pdf_w, pdf_h = page_size_pdf
    img_w, img_h = page_size_img
    sx = img_w / pdf_w
    sy = img_h / pdf_h
    # Docling bbox: (x0, y0_from_bottom, x1, y1_from_bottom)
    x0 = bbox.l * sx
    x1 = bbox.r * sx
    # Flip y: top = pdf_h - y_from_bottom
    y0 = (pdf_h - bbox.t) * sy
    y1 = (pdf_h - bbox.b) * sy
    return [x0, y0, x1, y1]


def extract_doc(pdf_path: Path, page_sizes: dict, conv) -> tuple[list[dict], dict[str, list[int]]]:
    """Returns (page_records, section_paths_for_eids).

    page_records: same shape as old elements.jsonl per-page records.
    section_paths_for_eids: eid -> section_path.
    """
    result = conv.convert(pdf_path)
    doc = result.document
    doc_id = pdf_path.name.split("_")[0]

    pages = sorted(doc.pages.keys())
    by_page = defaultdict(list)
    eid_paths: dict[str, list[int]] = {}

    # Walk doc.texts in body order. Track section_path.
    cur_path: list[int] = []
    h_counters: dict[int, int] = {}
    elem_idx_in_section = 0
    eid_counter = 0

    all_items = []
    # Combine texts + pictures + tables, sorted by (page, y_top in pdf)
    for t in doc.texts:
        if not t.prov: continue
        all_items.append(("text_like", t, t.prov[0]))
    for p in doc.pictures:
        if not p.prov: continue
        all_items.append(("picture", p, p.prov[0]))
    for tbl in doc.tables:
        if not tbl.prov: continue
        all_items.append(("table", tbl, tbl.prov[0]))

    # Sort by (page, -y_top)  (PDF y is bottom-up, top means largest y)
    all_items.sort(key=lambda x: (x[2].page_no, -x[2].bbox.t))

    for kind, item, prov in all_items:
        page_num = prov.page_no
        page_size_pdf = (doc.pages[page_num].size.width, doc.pages[page_num].size.height)
        page_size_img = page_sizes.get((doc_id, page_num))
        if page_size_img is None:
            continue
        bbox_img = bbox_to_img(prov.bbox, page_size_pdf, page_size_img)

        if kind == "text_like":
            label = str(getattr(item, "label", "")).lower()
            text = (item.text or "").strip()
            if "header" in label and "page_header" not in label:
                # Section header — update hierarchy
                level = infer_section_level(text)
                if level <= 0: level = 1
                h_counters = {k: v for k, v in h_counters.items() if k < level}
                h_counters[level] = h_counters.get(level, 0) + 1
                cur_path = [h_counters[k] for k in sorted(h_counters)]
                elem_idx_in_section = 0
                # Headers themselves don't go into elements list (skip)
                continue
            if label == "page_header":
                # Skip page numbers/running heads
                continue
            if "footnote" in label or "reference" in label:
                continue
            if not text or len(text) < 5:
                continue
            # Regular text
            eid = f"{doc_id}__p{page_num:03d}__e{eid_counter:04d}"
            eid_counter += 1
            elem_idx_in_section += 1
            by_page[page_num].append({
                "type": "text", "bbox_img": bbox_img,
                "bbox_pdf": [prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t],
                "text": text, "eid": eid,
            })
            eid_paths[eid] = list(cur_path) + [elem_idx_in_section]
        elif kind == "picture":
            eid = f"{doc_id}__p{page_num:03d}__e{eid_counter:04d}"
            eid_counter += 1
            elem_idx_in_section += 1
            by_page[page_num].append({
                "type": "image", "bbox_img": bbox_img,
                "bbox_pdf": [prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t],
                "text": "", "eid": eid,
            })
            eid_paths[eid] = list(cur_path) + [elem_idx_in_section]
        elif kind == "table":
            eid = f"{doc_id}__p{page_num:03d}__e{eid_counter:04d}"
            eid_counter += 1
            elem_idx_in_section += 1
            by_page[page_num].append({
                "type": "table", "bbox_img": bbox_img,
                "bbox_pdf": [prov.bbox.l, prov.bbox.b, prov.bbox.r, prov.bbox.t],
                "text": "", "eid": eid,
            })
            eid_paths[eid] = list(cur_path) + [elem_idx_in_section]

    # Convert to page records
    records = []
    for page in pages:
        records.append({
            "doc_id": doc_id, "page": page,
            "elements": by_page.get(page, []),
        })
    return records, eid_paths


def main():
    print("Collecting page image sizes...")
    page_sizes = collect_page_sizes()
    print(f"  page sizes for {len(page_sizes)} pages")

    pdfs = sorted(PDF_DIR.rglob("*.pdf"))
    print(f"PDFs to process: {len(pdfs)}")

    # Init docling
    from docling.document_converter import DocumentConverter
    conv = DocumentConverter()

    all_section_paths: dict[str, dict[str, list[int]]] = {}
    n_pages = n_text = n_img = n_tbl = 0

    OUT_JSONL.unlink(missing_ok=True)
    with open(OUT_JSONL, "a") as fout:
        for i, pdf in enumerate(pdfs, 1):
            t0 = time.time()
            try:
                records, paths = extract_doc(pdf, page_sizes, conv)
            except Exception as e:
                print(f"  [{i}/{len(pdfs)}] FAILED {pdf.name}: {e}")
                continue
            doc_id = pdf.name.split("_")[0]
            all_section_paths[doc_id] = paths
            for r in records:
                fout.write(json.dumps(r) + "\n")
                n_pages += 1
                for e in r["elements"]:
                    if e["type"] == "text": n_text += 1
                    elif e["type"] == "image": n_img += 1
                    elif e["type"] == "table": n_tbl += 1
            if i % 5 == 0 or i == len(pdfs):
                print(f"  [{i}/{len(pdfs)}] {pdf.name[:60]} ({time.time()-t0:.1f}s)  "
                      f"pages={n_pages} text={n_text} img={n_img} tbl={n_tbl}")

    SECTION_PATHS_JSON.write_text(json.dumps(all_section_paths))
    print(f"\nDone. Total pages={n_pages} text={n_text} img={n_img} tbl={n_tbl}")
    print(f"saved elements to {OUT_JSONL}")
    print(f"saved section_paths to {SECTION_PATHS_JSON}")


if __name__ == "__main__":
    main()
