"""Match SciEGQA GT bboxes to docling-extracted elements via containment.

Output: data/benchmarks/sciegqa/queries_with_gt_docling.jsonl
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data/benchmarks/sciegqa")
ELEMENTS_JSONL = ROOT / "elements_docling.jsonl"
BENCH_JSONL = ROOT / "SciEGQA_Bench.jsonl"
OUT_JSONL = ROOT / "queries_with_gt_docling.jsonl"

CONTAIN_THRESHOLD = 0.7


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersect_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def main():
    pages = defaultdict(list)
    with open(ELEMENTS_JSONL) as f:
        for line in f:
            r = json.loads(line)
            for e in r["elements"]:
                e2 = {**e, "doc_id": r["doc_id"], "page": r["page"]}
                pages[(r["doc_id"], r["page"])].append(e2)
    print(f"loaded {sum(len(v) for v in pages.values())} elements across {len(pages)} pages")

    out_queries = []
    n_gt_total = 0
    n_gt_matched = 0
    elements_per_query = []
    matched_type_counts = Counter()

    with open(BENCH_JSONL) as f:
        for qi, line in enumerate(f):
            d = json.loads(line)
            doc_id = d["doc_name"]
            gt_elements = set()
            for pi, page in enumerate(d["evidence_page"]):
                for bb, st in zip(d["bbox"][pi], d["subimg_type"][pi]):
                    n_gt_total += 1
                    cands = pages.get((doc_id, page), [])
                    found = False
                    for e in cands:
                        eb = e["bbox_img"]
                        ea = area(eb)
                        if ea <= 0:
                            continue
                        inter = intersect_area(eb, bb)
                        if inter / ea >= CONTAIN_THRESHOLD:
                            gt_elements.add(e["eid"])
                            matched_type_counts[e["type"]] += 1
                            found = True
                    if found:
                        n_gt_matched += 1
            elements_per_query.append(len(gt_elements))
            out_queries.append({
                "qid": f"sciegqa_{qi:05d}",
                "doc_id": doc_id,
                "query": d["query"],
                "answer": d["answer"],
                "category": d["category"],
                "gt_element_ids": sorted(gt_elements),
                "evidence_pages": d["evidence_page"],
            })

    with open(OUT_JSONL, "w") as fout:
        for q in out_queries:
            fout.write(json.dumps(q) + "\n")

    print(f"Queries: {len(out_queries)}")
    print(f"GT bbox regions: {n_gt_total}, matched (≥1 element): {n_gt_matched} ({n_gt_matched/n_gt_total:.1%})")
    n_with = sum(1 for n in elements_per_query if n > 0)
    print(f"Queries with ≥1 GT element: {n_with}/{len(out_queries)} ({n_with/len(out_queries):.1%})")
    print(f"Elements per query: mean={sum(elements_per_query)/len(elements_per_query):.2f}")
    print(f"Matched element types: {dict(matched_type_counts)}")
    print(f"saved to {OUT_JSONL}")


if __name__ == "__main__":
    main()
