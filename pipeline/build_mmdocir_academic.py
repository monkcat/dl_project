"""Build element + query files for MMDocIR academic subset.

Output:
  data/benchmarks/mmdocir/academic_elements.jsonl
    each line: {eid, doc_id, layout_id, page, type, text, ocr_text, vlm_text,
                bbox_pdf, page_size_pdf, image_b64}
  data/benchmarks/mmdocir/academic_queries.jsonl
    each line: {qid, doc_id, query, answer, type, gt_element_ids, gt_pages}
"""
from __future__ import annotations

import base64
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("data/benchmarks/mmdocir")
ANNO_JSONL = ROOT / "MMDocIR_annotations.jsonl"
LAYOUTS_PARQUET = ROOT / "MMDocIR_layouts.parquet"
OUT_ELEMENTS = ROOT / "academic_elements.jsonl"
OUT_QUERIES = ROOT / "academic_queries.jsonl"


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def main():
    # Load academic doc names
    with open(ANNO_JSONL) as f:
        docs = [json.loads(l) for l in f]
    academic = [d for d in docs if "Academic" in d["domain"]]
    ac_doc_names = {d["doc_name"].replace(".pdf", "") for d in academic}
    print(f"academic docs: {len(academic)}")

    # Load layouts filtered to academic
    pf = pq.ParquetFile(LAYOUTS_PARQUET)
    layouts_by_doc = defaultdict(list)
    print("Reading layouts...")
    for batch in pf.iter_batches(batch_size=20000):
        names = batch.column("doc_name").to_pylist()
        lids = batch.column("layout_id").to_pylist()
        pages = batch.column("page_id").to_pylist()
        types = batch.column("type").to_pylist()
        texts = batch.column("text").to_pylist()
        ocrs = batch.column("ocr_text").to_pylist()
        vlms = batch.column("vlm_text").to_pylist()
        bboxes = batch.column("bbox").to_pylist()
        psizes = batch.column("page_size").to_pylist()
        imgs = batch.column("image_binary").to_pylist()
        for dn, lid, pg, typ, txt, ocr, vlm, bb, ps, ib in zip(
            names, lids, pages, types, texts, ocrs, vlms, bboxes, psizes, imgs
        ):
            if dn not in ac_doc_names:
                continue
            layouts_by_doc[dn].append({
                "layout_id": lid, "page": pg, "type": typ,
                "text": txt or "", "ocr_text": ocr or "", "vlm_text": vlm or "",
                "bbox_pdf": list(bb), "page_size_pdf": list(ps),
                "image_b64": base64.b64encode(ib).decode("ascii") if ib else None,
            })
    print(f"  layouts in academic: {sum(len(v) for v in layouts_by_doc.values())}")

    # Write elements (assign eid)
    eid_lookup = defaultdict(dict)  # doc_id -> {layout_id: eid}
    with open(OUT_ELEMENTS, "w") as fout:
        for doc_id, layouts in layouts_by_doc.items():
            for k, L in enumerate(layouts):
                eid = f"{doc_id}__l{L['layout_id']:05d}"
                eid_lookup[doc_id][L["layout_id"]] = eid
                rec = {"eid": eid, "doc_id": doc_id, **L}
                fout.write(json.dumps(rec) + "\n")
    print(f"saved elements to {OUT_ELEMENTS}")

    # Match queries to layouts via IoU (relative coords)
    n_match = 0
    n_total_box = 0
    elements_per_query = []
    with open(OUT_QUERIES, "w") as fout:
        for di, d in enumerate(academic):
            doc_id = d["doc_name"].replace(".pdf", "")
            layouts = layouts_by_doc.get(doc_id, [])
            by_page = defaultdict(list)
            for L in layouts:
                by_page[L["page"]].append(L)
            for qi, q in enumerate(d["questions"]):
                gt_eids = set()
                for boxitem in q["layout_mapping"]:
                    page = boxitem["page"]
                    gt_bbox = boxitem["bbox"]
                    gt_size = boxitem["page_size"]
                    # Normalize GT to relative
                    gb_rel = [gt_bbox[0] / gt_size[0], gt_bbox[1] / gt_size[1],
                              gt_bbox[2] / gt_size[0], gt_bbox[3] / gt_size[1]]
                    cands = by_page.get(page, [])
                    best_iou = 0.0
                    best_L = None
                    for L in cands:
                        ps = L["page_size_pdf"]
                        Lb_rel = [L["bbox_pdf"][0] / ps[0], L["bbox_pdf"][1] / ps[1],
                                  L["bbox_pdf"][2] / ps[0], L["bbox_pdf"][3] / ps[1]]
                        v = iou(gb_rel, Lb_rel)
                        if v > best_iou:
                            best_iou = v
                            best_L = L
                    n_total_box += 1
                    if best_iou >= 0.5 and best_L is not None:
                        gt_eids.add(eid_lookup[doc_id][best_L["layout_id"]])
                        n_match += 1
                elements_per_query.append(len(gt_eids))
                rec = {
                    "qid": f"mmdocir_{di:03d}_{qi:02d}",
                    "doc_id": doc_id,
                    "query": q["Q"],
                    "answer": q["A"],
                    "type": q["type"],
                    "gt_element_ids": sorted(gt_eids),
                    "gt_pages": q["page_id"],
                }
                fout.write(json.dumps(rec) + "\n")

    print(f"\nQueries: {sum(len(d['questions']) for d in academic)}")
    print(f"GT bboxes matched: {n_match}/{n_total_box} ({n_match/n_total_box:.1%})")
    n_with_gt = sum(1 for n in elements_per_query if n > 0)
    print(f"Queries with ≥1 GT element: {n_with_gt}/{len(elements_per_query)}")
    print(f"Elements per query: mean={sum(elements_per_query)/len(elements_per_query):.2f}, "
          f"max={max(elements_per_query)}")
    print(f"saved queries to {OUT_QUERIES}")


if __name__ == "__main__":
    main()
