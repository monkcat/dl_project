"""Unified loaders for MMDocIR (academic) and SciEGQA element-level retrieval.

Both loaders return:
    elements: list[dict] with keys {id, doc_id, type, text, image_loader}
        image_loader: callable returning PIL.Image (or None for pure-text)
    queries: list[dict] with keys {qid, doc_id, query, gt_element_ids: set[str]}

Only queries with ≥1 GT element are returned (gt-filtered).
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Callable

from PIL import Image


def _crop_image_loader(page_png_path: Path, bbox_img: list[float]) -> Callable[[], Image.Image]:
    def loader():
        img = Image.open(page_png_path).convert("RGB")
        x0, y0, x1, y1 = [max(0, int(v)) for v in bbox_img]
        x1 = min(x1, img.size[0])
        y1 = min(y1, img.size[1])
        if x1 - x0 < 8 or y1 - y0 < 8:
            return img  # crop too small, fall back to full page
        crop = img.crop((x0, y0, x1, y1))
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        return crop
    return loader


def _b64_image_loader(b64: str) -> Callable[[], Image.Image]:
    def loader():
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    return loader


def load_mmdocir(root: str | Path = "data/benchmarks/mmdocir") -> tuple[list[dict], list[dict]]:
    root = Path(root)
    elements = []
    with open(root / "academic_elements.jsonl") as f:
        for line in f:
            r = json.loads(line)
            t = r["type"]
            # Encoder modality: text uses text field; image/table/equation use image
            text_for_encoding = r["text"] or r["ocr_text"] or r["vlm_text"] or ""
            elements.append({
                "id": r["eid"],
                "doc_id": r["doc_id"],
                "type": t,
                "text": text_for_encoding,
                "image_loader": _b64_image_loader(r["image_b64"]) if r["image_b64"] else None,
                "page": r["page"],
            })

    queries = []
    with open(root / "academic_queries.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if not r["gt_element_ids"]:
                continue
            queries.append({
                "qid": r["qid"],
                "doc_id": r["doc_id"],
                "query": r["query"],
                "answer": r["answer"],
                "gt_element_ids": set(r["gt_element_ids"]),
                "type": r.get("type", ""),
            })
    return elements, queries


def load_sciegqa(root: str | Path = "data/benchmarks/sciegqa",
                  source: str = "docling") -> tuple[list[dict], list[dict]]:
    """source: 'docling' (default, has section hierarchy) or 'pymupdf' (legacy)."""
    root = Path(root)
    img_root = root / "Images"
    doc_cat = {}
    for cat_dir in img_root.iterdir():
        if cat_dir.is_dir():
            for doc_dir in cat_dir.iterdir():
                if doc_dir.is_dir():
                    doc_cat[doc_dir.name] = cat_dir.name

    if source == "docling":
        elements_file = root / "elements_docling.jsonl"
        queries_file = root / "queries_with_gt_docling.jsonl"
    else:
        elements_file = root / "elements.jsonl"
        queries_file = root / "queries_with_gt.jsonl"

    elements = []
    eid_counter = {}
    with open(elements_file) as f:
        for line in f:
            r = json.loads(line)
            doc_id = r["doc_id"]
            page = r["page"]
            for e in r["elements"]:
                if "eid" in e:
                    # docling pre-assigns eid
                    eid = e["eid"]
                else:
                    eid_counter[doc_id] = eid_counter.get(doc_id, 0)
                    eid = f"{doc_id}__p{page:03d}__e{eid_counter[doc_id]:04d}"
                    eid_counter[doc_id] += 1

                if e["type"] == "text":
                    image_loader = None
                    text = e["text"]
                else:
                    cat = doc_cat.get(doc_id, "")
                    page_png = img_root / cat / doc_id / f"{doc_id}_{page}.png"
                    image_loader = _crop_image_loader(page_png, e["bbox_img"])
                    text = ""

                elements.append({
                    "id": eid,
                    "doc_id": doc_id,
                    "type": e["type"],
                    "text": text,
                    "image_loader": image_loader,
                    "page": page,
                })

    queries = []
    with open(queries_file) as f:
        for line in f:
            r = json.loads(line)
            if not r["gt_element_ids"]:
                continue
            queries.append({
                "qid": r["qid"],
                "doc_id": r["doc_id"],
                "query": r["query"],
                "answer": r["answer"],
                "gt_element_ids": set(r["gt_element_ids"]),
                "category": r.get("category", ""),
            })
    return elements, queries


def _pdf_crop_loader(pdf_path: Path, page_no: int, bbox_pdf: list[float]) -> Callable[[], Image.Image]:
    """Render PDF page and crop the bbox region (docling bbox: l/b/r/t with y bottom-up)."""
    def loader():
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[page_no - 1]
        pix = page.get_pixmap(dpi=120)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pdf_w, pdf_h = page.rect.width, page.rect.height
        sx = img.size[0] / pdf_w
        sy = img.size[1] / pdf_h
        l, b, r, t = bbox_pdf
        # docling bbox y is bottom-up; image y is top-down
        x0 = max(0, int(l * sx))
        x1 = min(img.size[0], int(r * sx))
        y0 = max(0, int((pdf_h - t) * sy))
        y1 = min(img.size[1], int((pdf_h - b) * sy))
        doc.close()
        if x1 - x0 < 8 or y1 - y0 < 8:
            return img
        return img.crop((x0, y0, x1, y1))
    return loader


def load_spiqa_enhanced(root: str | Path = "data/benchmarks/spiqa/test-A") -> tuple[list[dict], list[dict]]:
    """SPIQA test-A with docling-derived element-level annotations.

    GT for each query = the matched docling element id (from spiqa_fig_to_element.json).
    Pool = all docling elements (text/image/table/caption).
    """
    root = Path(root)
    spiqa = json.loads((root / "SPIQA_testA.json").read_text())
    mapping = json.loads((root / "spiqa_fig_to_element.json").read_text())

    elements = []
    eid_to_idx = {}
    with open(root / "elements_docling.jsonl") as f:
        for line in f:
            r = json.loads(line)
            doc_id = r["doc_id"]
            for e in r["elements"]:
                etype = e["type"]
                if etype == "caption":
                    etype = "text"  # treat caption as text for retrieval
                if etype == "text":
                    image_loader = None
                else:
                    pdf_path = root / "pdfs" / f"{doc_id}.pdf"
                    image_loader = _pdf_crop_loader(pdf_path, r["page"], e["bbox_pdf"])
                elements.append({
                    "id": e["eid"],
                    "doc_id": doc_id,
                    "type": etype,
                    "text": e.get("text", ""),
                    "image_loader": image_loader,
                    "page": r["page"],
                })
                eid_to_idx[e["eid"]] = len(elements) - 1

    queries = []
    for paper_id, paper in spiqa.items():
        for qi, qa in enumerate(paper.get("qa", [])):
            ref = qa.get("reference")
            m = mapping.get(ref)
            if not m:
                continue  # GT figure couldn't be matched to a docling element
            queries.append({
                "qid": f"spiqa_{paper_id}_{qi:02d}",
                "doc_id": paper_id,
                "query": qa["question"],
                "answer": qa["answer"],
                "gt_element_ids": {m["eid"]},
                "type": "figure",
            })
    return elements, queries


def stats(elements, queries):
    from collections import Counter
    print(f"  elements: {len(elements)}")
    types = Counter(e["type"] for e in elements)
    print(f"    by type: {dict(types)}")
    print(f"  queries: {len(queries)}")
    docs = len({e["doc_id"] for e in elements})
    qdocs = len({q["doc_id"] for q in queries})
    print(f"    docs: {docs}, queried docs: {qdocs}")
    avg_gt = sum(len(q["gt_element_ids"]) for q in queries) / max(1, len(queries))
    print(f"    avg GT per query: {avg_gt:.2f}")


if __name__ == "__main__":
    print("=== MMDocIR (academic) ===")
    el, q = load_mmdocir()
    stats(el, q)
    print()
    print("=== SciEGQA ===")
    el, q = load_sciegqa()
    stats(el, q)
    print()
    print("=== SPIQA enhanced ===")
    el, q = load_spiqa_enhanced()
    stats(el, q)
