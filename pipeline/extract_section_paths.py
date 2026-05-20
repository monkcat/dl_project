"""Extract section_path per element for MMDocIR/SciEGQA via heuristic header detection.

Section header detected when text matches: `^\d+(\.\d+)*\s+[A-Z][a-zA-Z\s]{3,40}$`
or contains common section keywords as title-cased short line.

section_path = [section_idx, subsection_idx, ..., position_in_section]

Output: data/benchmarks/{dataset}/section_paths.json
   {doc_id: {eid: [section_path...]}}
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


# Numeric header: "1 Introduction", "2.1 Methods", "3.1.2 Architecture"
NUM_HEADER_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\s+([A-Z][A-Za-z\s\-:,&/]{3,60})\s*$")
# Keyword header: short line of all-titlecase words
KEY_HEADERS = {
    "abstract", "introduction", "background", "related work", "preliminaries",
    "method", "methods", "approach", "model", "framework", "architecture",
    "experiments", "experimental setup", "results", "evaluation",
    "discussion", "analysis", "ablation", "limitations", "conclusion",
    "future work", "references", "acknowledgments", "appendix",
}
KEY_HEADER_RE = re.compile(r"^\s*([A-Z][A-Za-z\s\-:&]+?)\s*$")
ROLE_KEYWORDS = [
    ("intro",       ["introduction", "background"]),
    ("related",     ["related work", "prior work", "preliminari"]),
    ("method",      ["method", "approach", "model", "framework", "algorithm",
                     "architecture", "formulation", "design"]),
    ("experiment",  ["experiment", "experimental", "evaluation", "evaluat", "result",
                     "ablation", "study", "training"]),
    ("discussion",  ["discussion", "analysis", "limitation"]),
    ("conclusion",  ["conclusion", "summary", "future work"]),
    ("appendix",    ["appendix"]),
    ("abstract",    ["abstract"]),
    ("references",  ["references", "bibliography", "acknowledg"]),
]


def classify_role(title: str) -> str:
    t = title.lower()
    for role, keys in ROLE_KEYWORDS:
        if any(k in t for k in keys):
            return role
    return "other"


def detect_header(text: str) -> tuple[list[int] | None, str | None]:
    """Returns (level_path, normalized_title) if text is a header, else (None, None).

    Strict rules to avoid false positives:
      - Single line only
      - Either matches numeric pattern '1.2 Title' OR exact keyword match (after strip)
    """
    text = text.strip()
    if not text or "\n" in text:
        return None, None
    if len(text) < 4 or len(text) > 80:
        return None, None
    # Strict numeric pattern
    m = NUM_HEADER_RE.match(text)
    if m:
        parts = [int(x) for x in m.group(1).split(".")]
        return parts, m.group(2).strip()
    # Strict keyword: text must be exactly a header keyword (modulo trailing colon)
    low = text.lower().rstrip(":").strip()
    if low in KEY_HEADERS:
        return None, text
    return None, None


def assign_section_paths(elements: list[dict]) -> dict[str, list[int]]:
    """Walk elements in reading order, build section_path per element.

    cur_path tracks section hierarchy. When a header is found:
      - numeric path => set cur_path to that path
      - keyword header (no number) => start new section at level-1 (next counter)

    Element section_path = cur_path + [position_in_section].
    """
    sorted_elems = sorted(elements, key=lambda e: (e["page"], e["bbox"][1]))
    cur_path: list[int] = []
    h_counters: dict[int, int] = {}
    section_role = "other"
    elem_idx_in_section = 0
    out: dict[str, list[int]] = {}

    for e in sorted_elems:
        text = e.get("text", "")
        path, title = detect_header(text) if e["type"] == "text" else (None, None)
        if path is not None:
            # Numeric header — set hierarchy
            level = len(path)
            # Update counters: outer levels stay, set this level to path[-1]
            h_counters = {k: v for k, v in h_counters.items() if k < level}
            for i, v in enumerate(path, start=1):
                h_counters[i] = v
            cur_path = [h_counters[k] for k in sorted(h_counters)]
            section_role = classify_role(title or "")
            elem_idx_in_section = 0
            out[e["eid"]] = list(cur_path)
            continue
        if title is not None:
            # Keyword header — new level-1 section
            new_l1 = h_counters.get(1, 0) + 1
            h_counters = {1: new_l1}
            cur_path = [new_l1]
            section_role = classify_role(title)
            elem_idx_in_section = 0
            out[e["eid"]] = list(cur_path)
            continue
        # Regular element
        elem_idx_in_section += 1
        out[e["eid"]] = cur_path + [elem_idx_in_section]
    return out


def normalize_mmdocir():
    by_doc = defaultdict(list)
    with open("data/benchmarks/mmdocir/academic_elements.jsonl") as f:
        for line in f:
            r = json.loads(line)
            by_doc[r["doc_id"]].append({
                "eid": r["eid"], "type": r["type"], "page": r["page"],
                "bbox": r["bbox_pdf"], "text": r["text"] or r["ocr_text"] or r["vlm_text"] or "",
            })
    return by_doc


def normalize_sciegqa():
    by_doc = defaultdict(list)
    eid_counter = defaultdict(int)
    with open("data/benchmarks/sciegqa/elements.jsonl") as f:
        for line in f:
            r = json.loads(line)
            for e in r["elements"]:
                eid = f"{r['doc_id']}__p{r['page']:03d}__e{eid_counter[r['doc_id']]:04d}"
                eid_counter[r["doc_id"]] += 1
                by_doc[r["doc_id"]].append({
                    "eid": eid, "type": e["type"], "page": r["page"],
                    "bbox": e["bbox_img"], "text": e.get("text", "") or "",
                })
    return by_doc


def assign_section_paths_from_toc(elements: list[dict], toc: list[list]) -> dict[str, list[int]]:
    """toc entries: [level, title, page]. Build section_path from level structure."""
    if not toc:
        return assign_section_paths(elements)

    # Build entries with level counters
    h_counters: dict[int, int] = {}
    entries = []  # (page, path, role)
    for level, title, page in toc:
        h_counters = {k: v for k, v in h_counters.items() if k < level}
        h_counters[level] = h_counters.get(level, 0) + 1
        path = [h_counters[k] for k in sorted(h_counters)]
        role = classify_role(title)
        entries.append({"page": page, "path": path, "role": role})

    # For each element, find latest entry whose page <= elem.page
    sorted_elems = sorted(elements, key=lambda e: (e["page"], e["bbox"][1]))
    out: dict[str, list[int]] = {}
    cur_path: list[int] = []
    elem_idx = 0
    entry_idx = 0
    for e in sorted_elems:
        # Advance entry_idx while next entry's page <= elem page
        while entry_idx < len(entries) and entries[entry_idx]["page"] <= e["page"]:
            new_path = entries[entry_idx]["path"]
            if new_path != cur_path:
                cur_path = new_path
                elem_idx = 0
            entry_idx += 1
        if cur_path:
            elem_idx += 1
            out[e["eid"]] = list(cur_path) + [elem_idx]
        else:
            elem_idx += 1
            out[e["eid"]] = [elem_idx]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mmdocir", "sciegqa"], required=True)
    args = ap.parse_args()

    # Optional: TOC source for SciEGQA
    tocs: dict[str, list] = {}
    if args.dataset == "mmdocir":
        elems = normalize_mmdocir()
        out_path = Path("data/benchmarks/mmdocir/section_paths.json")
    else:
        elems = normalize_sciegqa()
        out_path = Path("data/benchmarks/sciegqa/section_paths.json")
        # Load TOCs from PDFs
        import fitz, glob
        print("Loading PDF TOCs...")
        pdfs = glob.glob("data/benchmarks/sciegqa/PDF/**/*.pdf", recursive=True)
        for p in pdfs:
            doc_id = Path(p).name.split("_")[0]
            try:
                d = fitz.open(p)
                tocs[doc_id] = d.get_toc()
                d.close()
            except Exception:
                tocs[doc_id] = []
        n_with_toc = sum(1 for t in tocs.values() if t)
        print(f"  {n_with_toc}/{len(tocs)} docs with TOC")

    all_paths = {}
    actual_headers_per_doc = {}  # docs -> #true headers via detect_header
    n_total = 0
    path_depths = []
    for doc_id, doc_elems in elems.items():
        toc = tocs.get(doc_id, [])
        if args.dataset == "sciegqa" and toc:
            paths = assign_section_paths_from_toc(doc_elems, toc)
        else:
            paths = assign_section_paths(doc_elems)
        all_paths[doc_id] = paths
        # Count *true* headers (numeric or keyword) by re-detecting
        true_h = 0
        for e in doc_elems:
            if e["type"] != "text": continue
            p, t = detect_header(e["text"])
            if p is not None or t is not None:
                true_h += 1
        actual_headers_per_doc[doc_id] = true_h
        n_total += len(paths)
        for p in paths.values():
            path_depths.append(len(p))

    out_path.write_text(json.dumps(all_paths))
    import statistics as st
    print(f"docs: {len(all_paths)}")
    print(f"total elements: {n_total}")
    print(f"actual headers (via detect_header): {sum(actual_headers_per_doc.values())}")
    print(f"actual headers/doc: mean={st.mean(actual_headers_per_doc.values()):.1f}, "
          f"median={st.median(actual_headers_per_doc.values())}, "
          f"max={max(actual_headers_per_doc.values())}, min={min(actual_headers_per_doc.values())}")
    if path_depths:
        print(f"section_path depth: mean={st.mean(path_depths):.2f}, max={max(path_depths)}")
    # How many docs have ANY headers detected
    n_with_headers = sum(1 for v in actual_headers_per_doc.values() if v > 0)
    print(f"docs with ≥1 header: {n_with_headers}/{len(actual_headers_per_doc)}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
