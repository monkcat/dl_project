"""Extract elements (text blocks + figure bboxes) from SciEGQA PDFs using PyMuPDF.

Output: data/benchmarks/sciegqa/elements.jsonl
Each line: {doc_id, page, elements: [{eid, type, bbox_pdf, bbox_img, text}]}

bbox_pdf is in PDF point coords; bbox_img is scaled to page-image pixel coords
(matching SciEGQA GT bbox space). Page image size read from first PNG per doc.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path("data/benchmarks/sciegqa")
PDFS_DIR = ROOT / "PDF"
IMAGES_TAR = ROOT / "Images.tar"
OUT_JSONL = ROOT / "elements.jsonl"


def collect_page_sizes() -> dict[tuple[str, int], tuple[int, int]]:
    """Walk Images.tar once to read page-image dimensions per (doc_id, page)."""
    sizes: dict[tuple[str, int], tuple[int, int]] = {}
    with tarfile.open(IMAGES_TAR) as t:
        for m in t.getmembers():
            if not m.name.endswith(".png"):
                continue
            # name: Images/<cat>/<doc_id>/<doc_id>_<page>.png
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


def extract_doc(pdf_path: Path, page_sizes: dict) -> list[dict]:
    """Return list of pages, each with elements (text blocks + image regions)."""
    doc = fitz.open(pdf_path)
    doc_id = pdf_path.stem.split("_")[0]  # arxiv id prefix
    out = []
    for pi in range(len(doc)):
        page_num = pi + 1  # 1-based to match SciEGQA evidence_page
        p = doc[pi]
        img_size = page_sizes.get((doc_id, page_num))
        if img_size is None:
            sx = sy = 1.0
        else:
            sx = img_size[0] / p.rect.width
            sy = img_size[1] / p.rect.height

        elements = []
        # Text blocks
        for b in p.get_text("blocks"):
            x0, y0, x1, y1, text, bno, btype = b
            if btype != 0:  # 0 = text, 1 = image
                continue
            text = text.strip()
            if len(text) < 20:  # skip tiny fragments (line numbers etc)
                continue
            elements.append({
                "type": "text",
                "bbox_pdf": [x0, y0, x1, y1],
                "bbox_img": [x0 * sx, y0 * sy, x1 * sx, y1 * sy],
                "text": text,
            })

        # Embedded images
        for ii in p.get_image_info():
            bb = ii.get("bbox")
            if bb is None:
                continue
            x0, y0, x1, y1 = bb
            elements.append({
                "type": "image",
                "bbox_pdf": [x0, y0, x1, y1],
                "bbox_img": [x0 * sx, y0 * sy, x1 * sx, y1 * sy],
                "text": "",
            })

        # Drawings → group into figure regions if numerous
        # (theory papers often have figures as vector drawings)
        drawings = p.get_drawings()
        if drawings:
            # Compute union bbox of drawings, treat as one figure if total area is meaningful
            xs0 = [d["rect"].x0 for d in drawings if d.get("rect")]
            ys0 = [d["rect"].y0 for d in drawings if d.get("rect")]
            xs1 = [d["rect"].x1 for d in drawings if d.get("rect")]
            ys1 = [d["rect"].y1 for d in drawings if d.get("rect")]
            if xs0:
                ux0, uy0, ux1, uy1 = min(xs0), min(ys0), max(xs1), max(ys1)
                if (ux1 - ux0) * (uy1 - uy0) > 1000:  # min area threshold
                    elements.append({
                        "type": "drawing",
                        "bbox_pdf": [ux0, uy0, ux1, uy1],
                        "bbox_img": [ux0 * sx, uy0 * sy, ux1 * sx, uy1 * sy],
                        "text": "",
                    })

        out.append({
            "doc_id": doc_id,
            "page": page_num,
            "page_size_pdf": [p.rect.width, p.rect.height],
            "page_size_img": list(img_size) if img_size else None,
            "elements": elements,
        })
    doc.close()
    return out


def main():
    print("Collecting page image sizes...")
    page_sizes = collect_page_sizes()
    print(f"  page sizes for {len(page_sizes)} pages")

    pdfs = sorted(PDFS_DIR.rglob("*.pdf"))
    print(f"PDFs: {len(pdfs)}")

    n_pages = n_text = n_img = n_draw = 0
    with open(OUT_JSONL, "w") as fout:
        for i, pdf in enumerate(pdfs, 1):
            pages = extract_doc(pdf, page_sizes)
            for page_rec in pages:
                n_pages += 1
                for e in page_rec["elements"]:
                    if e["type"] == "text": n_text += 1
                    elif e["type"] == "image": n_img += 1
                    elif e["type"] == "drawing": n_draw += 1
                fout.write(json.dumps(page_rec) + "\n")
            if i % 20 == 0:
                print(f"  [{i}/{len(pdfs)}] {pdf.stem}: pages={n_pages} text={n_text} img={n_img} draw={n_draw}")
    print(f"\nDone. pages={n_pages} text={n_text} img={n_img} draw={n_draw}")
    print(f"saved to {OUT_JSONL}")


if __name__ == "__main__":
    main()
