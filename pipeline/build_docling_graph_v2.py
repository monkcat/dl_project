"""Convert v1 docling-based graphs (SciEGQA, MMDocIR) to schema v2.1.

The v1 graphs already carry the bulk of structural information:
    - caption_of edges (layout proximity + label match)
    - references edges (regex text→target)
    - same_section edges (sibling grouping; v2.1 drops these but uses for inference)
    - next/prev edges (reading order)

This converter:
    1. Renames `references` → `refer_to`.
    2. Drops `same_section` (v2.1 omits — encoder content similarity + contains hierarchy carry it).
    3. Limits `reading_next` to **intra-section** transitions (cross-section becomes `section_next`).
    4. Adds `section_header` nodes for each distinct `section_path` group.
    5. Adds `contains` edges from section_header → child elements.
    6. Relabels caption-side text nodes as `caption` type (via existing caption_of src).
    7. Promotes node attributes (type, page, section_path, section_role, label) from
       elements_docling.jsonl (SciEGQA) or academic_elements.jsonl (MMDocIR).

Run:
    python -m pipeline.build_docling_graph_v2 --dataset sciegqa
    python -m pipeline.build_docling_graph_v2 --dataset mmdocir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


DATASETS = {
    "sciegqa": {
        "root": REPO / "data/benchmarks/sciegqa",
        "v1_graph": REPO / "data/benchmarks/sciegqa/element_graph_docling.json",
        "elements": REPO / "data/benchmarks/sciegqa/elements_docling_v2.jsonl",
        "elements_per_page": True,                  # records grouped by (doc_id, page) with .elements list
        "section_paths": REPO / "data/benchmarks/sciegqa/section_paths_docling_v2.json",
        "out_graph": REPO / "data/benchmarks/sciegqa/element_graph_v2.json",
        "out_elements": REPO / "data/benchmarks/sciegqa/elements_v2.jsonl",
        "reclassify_text_to_header": False,         # use docling section_header labels directly
    },
    "sciegqa_v1": {
        # Fallback to old extraction (without docling labels)
        "root": REPO / "data/benchmarks/sciegqa",
        "v1_graph": REPO / "data/benchmarks/sciegqa/element_graph_docling.json",
        "elements": REPO / "data/benchmarks/sciegqa/elements_docling.jsonl",
        "elements_per_page": True,
        "section_paths": REPO / "data/benchmarks/sciegqa/section_paths_docling.json",
        "out_graph": REPO / "data/benchmarks/sciegqa/element_graph_v2_legacy.json",
        "out_elements": REPO / "data/benchmarks/sciegqa/elements_v2_legacy.jsonl",
        "reclassify_text_to_header": True,
    },
    "mmdocir": {
        "root": REPO / "data/benchmarks/mmdocir",
        "v1_graph": REPO / "data/benchmarks/mmdocir/element_graph.json",
        "elements": REPO / "data/benchmarks/mmdocir/academic_elements.jsonl",
        "elements_per_page": False,                 # flat per-element records
        "section_paths": REPO / "data/benchmarks/mmdocir/section_paths.json",
        "out_graph": REPO / "data/benchmarks/mmdocir/element_graph_v2.json",
        "out_elements": REPO / "data/benchmarks/mmdocir/elements_v2.jsonl",
        "reclassify_text_to_header": True,          # MMDocIR has heading text inline as 'text' type
    },
}

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

LABEL_RE = re.compile(
    r"^\s*(Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Regex for finding inline references in paragraph text (e.g., "as shown in Figure 3",
# "see Table 2", "Eq. 5"). Matches anywhere in text, not just at the start.
REF_IN_TEXT_RE = re.compile(
    r"\b(Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?|Equation|Eq\.?)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Citation pattern to filter false positives (e.g., "as in [12]")
CITATION_NEAR_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")

ROLE_KEYWORDS = {
    # 키워드는 normalize_heading_text 통과 후의 lowercase 형태와 매칭됨
    "intro":          ("introduction", "motivation"),
    "related_work":   ("related work", "background", "literature", "preliminar", "prior work"),
    "method":         ("method", "approach", "model", "framework", "algorithm", "architecture",
                       "design", "implementation", "training", "loss", "task", "datasets",
                       "dataset", "selection"),
    "result":         ("experiment", "result", "evaluation", "benchmark", "setup", "ablation",
                       "performance"),
    "discussion":     ("discussion", "analysis", "limitation", "limitations"),
    "conclusion":     ("conclusion", "future"),
    "appendix":       ("appendix", "supplementary", "supplement"),
    "references":     ("reference", "bibliography", "acknowledg"),
    "front_matter":   ("abstract", "front matter", "frontmatter", "title page"),
}

# Section heading regex (heuristic): "1 Introduction", "3.2 Method", "Appendix A" etc.
HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:[1-9]\d?\.?)+\s+[A-Z]"           # numeric prefix + Cap
    r"|"
    r"(?:Section|Sec\.?|Appendix|App\.?)\s+[A-Z\d]"   # "Section 3", "Appendix B"
    r"|"
    r"(?:" + "|".join(
        kw.title() for kws in ROLE_KEYWORDS.values() for kw in kws
    ) + r")"                                # known section names (capitalized)
    r")",
    re.IGNORECASE,
)


def looks_like_heading(text: str) -> bool:
    """Heuristic: short text that matches heading patterns."""
    if not text:
        return False
    t = text.strip()
    if len(t) > 120:
        return False
    if len(t) < 3:
        return False
    return bool(HEADING_RE.match(t))


SEC_PREFIX_RE = re.compile(
    r"^\s*(?:Sec\.?|Section|Appendix|App\.?|Chap\.?|Chapter)\s*\d*(?:\.\d+)*\s*[.:]?\s*",
    re.IGNORECASE,
)
NUM_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+){0,3}\s+")
LETTER_PREFIX_RE = re.compile(r"^\s*[A-Z]\.\s+")


def normalize_heading_text(title: str) -> str:
    """Strip section prefixes and OCR-style letter spacing for role classification.

    Examples:
        "Sec. 3.4 In-context Few-shot Learning"  → "in-context few-shot learning"
        "3.1 Object detection set prediction"     → "object detection set prediction"
        "A. Graph Reduction"                       → "graph reduction"
        "1 I N TRO DUCT ION"                       → "introduction"
    """
    if not title:
        return ""
    t = title.strip()
    # Strip "Sec. 3.4", "Section 5", "Appendix A" prefixes
    t = SEC_PREFIX_RE.sub("", t)
    # Strip "3.1", "4.4.2" prefixes
    t = NUM_PREFIX_RE.sub("", t)
    # Strip "A.", "B." prefixes
    t = LETTER_PREFIX_RE.sub("", t)
    # Fix OCR letter-spacing ("I N TRO DUCT ION" → "INTRODUCTION")
    # Heuristic: if the text has many single-char tokens separated by spaces, glue them.
    tokens = t.split()
    if tokens and sum(1 for tok in tokens if len(tok) == 1) >= len(tokens) * 0.4:
        t = "".join(tokens)
    return t.strip().lower()


def classify_section_role(title: str | None) -> str:
    if not title:
        return "other"
    t = normalize_heading_text(title)
    if not t:
        return "other"
    for role, kws in ROLE_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return role
    return "other"


def parse_label(text: str | None) -> str | None:
    if not text:
        return None
    m = LABEL_RE.match(text)
    if not m:
        return None
    kind = m.group(1).lower().rstrip(".")
    if kind.startswith("fig"):
        kind = "figure"
    elif kind.startswith("tab"):
        kind = "table"
    elif kind.startswith("alg"):
        kind = "algorithm"
    return f"{kind} {m.group(2)}"


# ────────────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────────────

def load_elements(path: Path, per_page: bool) -> dict[str, dict[str, dict]]:
    """Return {doc_id: {eid: element_dict}}.

    The eid uses different fields per dataset; we normalize via:
        SciEGQA: r["doc_id"] + per-element "eid" inside r["elements"]
        MMDocIR: r["eid"] directly (already includes doc_id prefix)
    """
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if per_page:
                doc = r["doc_id"]
                page = r.get("page")
                for e in r.get("elements", []):
                    eid = e["eid"]
                    out[doc][eid] = {
                        "id": eid,
                        "doc_id": doc,
                        "type": e.get("type", "text"),
                        "text": e.get("text") or None,
                        "page": page,
                        "bbox_pdf": e.get("bbox_pdf"),
                    }
            else:
                doc = r["doc_id"]
                eid = r["eid"]
                out[doc][eid] = {
                    "id": eid,
                    "doc_id": doc,
                    "type": r.get("type", "text"),
                    "text": r.get("text") or r.get("ocr_text") or r.get("vlm_text") or None,
                    "page": r.get("page"),
                    "bbox_pdf": r.get("bbox_pdf"),
                }
    return out


def load_section_paths(path: Path) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return {doc_id: {eid: section_path_tuple}}."""
    raw = json.loads(path.read_text())
    out: dict[str, dict[str, tuple[int, ...]]] = {}
    for doc_id, mp in raw.items():
        out[doc_id] = {eid: tuple(v) for eid, v in mp.items()}
    return out


def load_v1_graph(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text())


# ────────────────────────────────────────────────────────────────────────────
# Per-doc conversion
# ────────────────────────────────────────────────────────────────────────────

# Map docling element types → v2.1 ElementType
TYPE_NORMALIZE = {
    "text": "text",
    "image": "figure",
    "figure": "figure",
    "table": "table",
    "equation": "equation",
    "caption": "caption",
}


def reading_order_key(eid: str) -> tuple[int, int]:
    """eid pattern: doc__pPPP__eEEEE  or  doc__lLLLLL.
    Return sortable tuple (page_or_layout, elem_idx).
    """
    parts = eid.split("__")
    if len(parts) >= 3 and parts[1].startswith("p"):
        try:
            return int(parts[1][1:]), int(parts[2][1:])
        except ValueError:
            return 0, 0
    if len(parts) >= 2 and parts[1].startswith("l"):
        try:
            return 0, int(parts[1][1:])
        except ValueError:
            return 0, 0
    return 0, 0


def convert_doc(
    doc_id: str,
    v1_doc: dict,
    elements_doc: dict[str, dict],
    section_paths_doc: dict[str, tuple[int, ...]],
    reclassify_text_to_header: bool = True,
) -> tuple[dict, list[dict]]:
    """Convert v1 graph + element records → v2.1 doc graph + element records.

    If `reclassify_text_to_header=True`, scan text elements for heading-like
    patterns (numeric prefix, known section names) and re-type them as
    `section_header`. Use this for MMDocIR (no inline headings) and as a
    fallback for SciEGQA v1 elements (which threw away docling headers).

    If False, trust the input element types (use this with the v2 extractor
    that preserves docling labels).
    """

    # ── 0. Optionally reclassify inline heading-like text → section_header ──
    if reclassify_text_to_header:
        for eid, e in elements_doc.items():
            if e.get("type") == "text" and looks_like_heading(e.get("text") or ""):
                e["type"] = "section_header"

    # ── 1. Caption-side text elements → relabel as 'caption' type ──
    caption_src_set: set[str] = set()
    for edge in v1_doc.get("edges", []):
        if edge["type"] == "caption_of":
            caption_src_set.add(edge["src"])

    # ── 2. Find section grouping ──
    # Preferred: use explicit section_header elements (from docling labels or reclassify pass).
    # Walk elements in reading order; each section_header starts a new section that
    # owns all following elements until the next header.
    # Fallback: v1 same_section connected components (when no headers present).
    sorted_eids = sorted(elements_doc.keys(), key=reading_order_key)
    header_eids = [eid for eid in sorted_eids
                   if elements_doc[eid].get("type") == "section_header"]
    section_groups: dict[int, list[str]] = {}

    if header_eids:
        # Section boundaries from explicit headers
        # Elements before the first header → "front_matter" section (title/abstract/author)
        current_section_id = 0
        current_members: list[str] = []
        current_header_text: str | None = None
        section_headers_for_groups: dict[int, str | None] = {}

        first_header_pos = sorted_eids.index(header_eids[0])
        if first_header_pos > 0:
            section_groups[0] = sorted_eids[:first_header_pos]
            section_headers_for_groups[0] = "Front Matter"   # explicit so role classifier picks front_matter
            current_section_id = 1

        for eid in sorted_eids[first_header_pos:]:
            e = elements_doc[eid]
            if e.get("type") == "section_header":
                # Start a new section; the header itself becomes a member
                if current_members:
                    section_groups[current_section_id] = current_members
                    section_headers_for_groups[current_section_id] = current_header_text
                    current_section_id += 1
                current_members = [eid]
                current_header_text = e.get("text") or None
            else:
                current_members.append(eid)
        # Flush the last group
        if current_members:
            section_groups[current_section_id] = current_members
            section_headers_for_groups[current_section_id] = current_header_text
    else:
        # Fallback: same_section connected components
        adj: dict[str, set[str]] = defaultdict(set)
        for e in v1_doc.get("edges", []):
            if e["type"] == "same_section":
                adj[e["src"]].add(e["dst"])
                adj[e["dst"]].add(e["src"])
        visited_cc: set[str] = set()
        next_section_id = 0
        for eid in sorted_eids:
            if eid in visited_cc:
                continue
            stack = [eid]
            comp: list[str] = []
            while stack:
                x = stack.pop()
                if x in visited_cc:
                    continue
                visited_cc.add(x)
                comp.append(x)
                stack.extend(n for n in adj.get(x, ()) if n not in visited_cc)
            section_groups[next_section_id] = comp
            next_section_id += 1
        section_headers_for_groups = {k: None for k in section_groups}

    # Sort groups by min reading-order within them (= document order of sections)
    sorted_section_keys = sorted(
        section_groups.keys(),
        key=lambda k: min(reading_order_key(e) for e in section_groups[k]),
    )

    # ── 2b. Merge small consecutive sections to match SPIQA-like granularity ──
    # Docling section trees are over-segmented (avg ~20 sections/doc) vs LaTeX
    # \section{} (~7/doc). Merge sections of size < MIN_SECTION_SIZE into the
    # previous section (or next, if no previous), preserving reading order.
    MIN_SECTION_SIZE = 8
    merged_keys: list[int] = []
    for k in sorted_section_keys:
        if not merged_keys:
            merged_keys.append(k)
            continue
        prev_size = len(section_groups[merged_keys[-1]])
        cur_size = len(section_groups[k])
        if cur_size < MIN_SECTION_SIZE and prev_size >= MIN_SECTION_SIZE:
            # absorb current into previous
            section_groups[merged_keys[-1]].extend(section_groups[k])
            del section_groups[k]
        else:
            merged_keys.append(k)
    # Second pass: handle cases where small section followed by another small section
    final_keys: list[int] = []
    for k in merged_keys:
        if not final_keys:
            final_keys.append(k); continue
        if len(section_groups[k]) < MIN_SECTION_SIZE:
            section_groups[final_keys[-1]].extend(section_groups[k])
            del section_groups[k]
        else:
            final_keys.append(k)
    sorted_section_keys = final_keys

    # ── 3. Build section_header nodes ──
    nodes: list[dict] = []
    records: list[dict] = []
    section_id_by_key: dict[tuple, str] = {}
    section_order: list[str] = []

    for i, key in enumerate(sorted_section_keys):
        sid = f"{doc_id}__sec{i:03d}"
        members = sorted(section_groups[key], key=reading_order_key)
        # Preferred label source: the section_header element of this group
        heading_text = section_headers_for_groups.get(key) if header_eids else None
        if not heading_text:
            # Fallback: look for a heading-like text element among first 10 members
            for eid in members[:10]:
                e = elements_doc.get(eid)
                if not e:
                    continue
                t = (e.get("text") or "").strip()
                if e.get("type") == "section_header" or looks_like_heading(t):
                    heading_text = t
                    break

        if heading_text:
            role = classify_section_role(heading_text)
            label = heading_text[:80]
        else:
            scan_text = ""
            for eid in members[:5]:
                e = elements_doc.get(eid)
                if e and e.get("type") == "text" and e.get("text"):
                    scan_text = e["text"][:300]
                    break
            role = classify_section_role(scan_text) if scan_text else "other"
            label = f"section_{i:03d}"

        nodes.append({
            "id": sid,
            "type": "section_header",
            "page": None,
            "section_path": [i],
            "section_role": role,
            "label": label,
        })
        records.append({
            "id": sid,
            "doc_id": doc_id,
            "type": "section_header",
            "text": label,
            "section_role": role,
            "section_path": [i],
        })
        section_id_by_key[key] = sid
        section_order.append(sid)

    # ── 4. Build element nodes + records ──
    # Map eid → section_id
    eid_to_section_id: dict[str, str] = {}
    for key, members in section_groups.items():
        for eid in members:
            eid_to_section_id[eid] = section_id_by_key[key]

    # Reuse normalized type from elements; if eid is caption_of src → override to "caption"
    sorted_eids = sorted(elements_doc.keys(), key=reading_order_key)
    for eid in sorted_eids:
        e = elements_doc[eid]
        ntype_raw = e.get("type", "text")
        ntype = TYPE_NORMALIZE.get(ntype_raw, "text")
        if eid in caption_src_set and ntype == "text":
            ntype = "caption"  # relabel
        sec_id = eid_to_section_id.get(eid)
        sec_role = next((n["section_role"] for n in nodes if n["id"] == sec_id), "other") if sec_id else "other"
        label = parse_label(e.get("text")) if ntype in ("caption", "figure", "table") else None
        # For figures/tables, derive label from caption (if linked) — best effort, leave None for now
        nodes.append({
            "id": eid,
            "type": ntype,
            "page": e.get("page"),
            "section_path": list(section_paths_doc.get(eid, ())) or None,
            "section_role": sec_role,
            "label": label,
        })
        records.append({
            "id": eid,
            "doc_id": doc_id,
            "type": ntype,
            "text": e.get("text"),
            "page": e.get("page"),
            "bbox_pdf": e.get("bbox_pdf"),
            "label": label,
            "section_role": sec_role,
            "section_path": list(section_paths_doc.get(eid, ())) or None,
        })

    # ── 5. Build edges ──
    edges: list[dict] = []

    # 5a. contains: section_header ↔ each member (bidirectional)
    for key, members in section_groups.items():
        sid = section_id_by_key[key]
        for eid in members:
            edges.append({
                "src": sid,
                "dst": eid,
                "type": "contains",
                "confidence": 1.0,
                "bidirectional": True,
            })

    # 5b. section_next: sequential section_headers
    for i in range(len(section_order) - 1):
        edges.append({
            "src": section_order[i],
            "dst": section_order[i + 1],
            "type": "section_next",
            "confidence": 1.0,
            "bidirectional": False,
        })

    # Build set of valid eids for filtering v1 edges (which may reference old eids)
    valid_eids = set(elements_doc.keys())

    def both_valid(e):
        return e["src"] in valid_eids and e["dst"] in valid_eids

    # 5c. caption_of: copy from v1 (skip edges referencing missing eids)
    for e in v1_doc.get("edges", []):
        if e["type"] == "caption_of" and both_valid(e):
            edges.append({
                "src": e["src"],
                "dst": e["dst"],
                "type": "caption_of",
                "confidence": e.get("confidence", 0.7),
                "bidirectional": True,
                "evidence": "v1 caption_of (layout proximity + label match)",
            })

    # 5d. refer_to: rename from v1 'references'
    for e in v1_doc.get("edges", []):
        if e["type"] == "references" and both_valid(e):
            edges.append({
                "src": e["src"],
                "dst": e["dst"],
                "type": "refer_to",
                "confidence": e.get("confidence", 0.9),
                "bidirectional": False,
                "evidence": "v1 references",
            })

    # 5e. reading_next: extract directly from sorted reading order (intra-section only)
    # This is more reliable than v1 'next' edges (which may have stale eids).
    sorted_eids_for_next = sorted(elements_doc.keys(), key=reading_order_key)
    existing_next = set()  # for dedup
    for i in range(len(sorted_eids_for_next) - 1):
        a = sorted_eids_for_next[i]
        b = sorted_eids_for_next[i + 1]
        sa = eid_to_section_id.get(a)
        sb = eid_to_section_id.get(b)
        if sa is None or sb is None or sa != sb:
            continue  # cross-section → handled by section_next
        if (a, b) in existing_next:
            continue
        existing_next.add((a, b))
        edges.append({
            "src": a,
            "dst": b,
            "type": "reading_next",
            "confidence": 1.0,
            "bidirectional": True,
        })

    # ── 5f. Augment caption_of edges via reading-order proximity ──
    # If a caption element exists in the new extraction but no v1 caption_of edge
    # connects it to a figure/table, infer caption_of via reading-order proximity:
    # caption ↔ nearest adjacent figure/table (within 3 reading positions).
    existing_capof_src = {e["src"] for e in edges if e["type"] == "caption_of"}
    existing_capof_dst = {e["dst"] for e in edges if e["type"] == "caption_of"}
    captions = [eid for eid, e in elements_doc.items()
                 if e.get("type") == "caption" and eid not in existing_capof_src]
    visuals = [eid for eid, e in elements_doc.items()
                if e.get("type") in ("figure", "table") and eid not in existing_capof_dst]
    if captions and visuals:
        sorted_all = sorted(elements_doc.keys(), key=reading_order_key)
        pos = {eid: i for i, eid in enumerate(sorted_all)}
        for cap_eid in captions:
            ci = pos.get(cap_eid)
            if ci is None: continue
            best = None; best_d = 4
            for v_eid in visuals:
                vi = pos.get(v_eid)
                if vi is None: continue
                d = abs(ci - vi)
                if d < best_d:
                    best_d = d; best = v_eid
            if best is not None:
                edges.append({
                    "src": cap_eid,
                    "dst": best,
                    "type": "caption_of",
                    "confidence": 0.7,
                    "bidirectional": True,
                    "evidence": "reading-order proximity (docling caption label)",
                })

    # ── 5g. Augment refer_to edges via regex on paragraph text ──
    # Build label → fname map from captions (parsed label) AND figure/table elements
    label_to_fname: dict[str, str] = {}
    # First pass: parse labels from caption element text
    for eid, e in elements_doc.items():
        if e.get("type") != "caption":
            continue
        lbl = parse_label(e.get("text") or "")
        if not lbl:
            continue
        # Find the figure/table linked to this caption (via existing caption_of edge or proximity)
        linked_visual = None
        for edge in edges:
            if edge.get("type") == "caption_of" and edge.get("src") == eid:
                if elements_doc.get(edge.get("dst"), {}).get("type") in ("figure", "table", "equation"):
                    linked_visual = edge["dst"]
                    break
        if linked_visual:
            label_to_fname[lbl] = linked_visual

    # Second pass: also map labels to visual elements directly if their own text has a label
    for eid, e in elements_doc.items():
        if e.get("type") in ("figure", "table", "equation"):
            lbl = parse_label(e.get("text") or "")
            if lbl and lbl not in label_to_fname:
                label_to_fname[lbl] = eid

    # Now scan text/paragraph elements for inline references
    existing_refto = {(e["src"], e["dst"]) for e in edges if e["type"] == "refer_to"}
    n_refer_added = 0
    for eid, e in elements_doc.items():
        if e.get("type") != "text":
            continue
        text = e.get("text") or ""
        if len(text) < 10:
            continue
        seen_in_this_para: set[str] = set()
        for match in REF_IN_TEXT_RE.finditer(text):
            kind = match.group(1).lower().rstrip(".")
            num = match.group(2)
            if kind.startswith("fig"):
                kind = "figure"
            elif kind.startswith("tab"):
                kind = "table"
            elif kind.startswith("alg"):
                kind = "algorithm"
            elif kind.startswith("eq"):
                kind = "equation"
            key = f"{kind} {num}"
            if key in seen_in_this_para:
                continue
            seen_in_this_para.add(key)
            dst = label_to_fname.get(key)
            if not dst:
                continue
            # Filter false positives: citation token within ±50 chars of the match
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            window = text[start:end]
            if CITATION_NEAR_RE.search(window):
                # Could be external paper reference like "[12] Figure 3 of ..."
                # We keep cautiously: only drop if citation is RIGHT before the match
                pre_window = text[max(0, match.start() - 8):match.start()]
                if CITATION_NEAR_RE.search(pre_window):
                    continue
            if (eid, dst) in existing_refto:
                continue
            existing_refto.add((eid, dst))
            edges.append({
                "src": eid,
                "dst": dst,
                "type": "refer_to",
                "confidence": 0.9,
                "bidirectional": False,
                "evidence": f"regex '{key}' in paragraph",
            })
            n_refer_added += 1

    graph = {
        "doc_id": doc_id,
        "schema_version": "v2.1",
        "nodes": nodes,
        "edges": edges,
    }
    return graph, records


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]

    print(f"=== building v2.1 graph for {args.dataset} ===")
    print(f"  loading v1 graph: {cfg['v1_graph'].name}")
    v1 = load_v1_graph(cfg["v1_graph"])
    print(f"  loading elements: {cfg['elements'].name}")
    elements = load_elements(cfg["elements"], cfg["elements_per_page"])
    print(f"  loading section_paths: {cfg['section_paths'].name}")
    section_paths = load_section_paths(cfg["section_paths"])
    print(f"  docs in v1: {len(v1)}, elements: {sum(len(d) for d in elements.values())}")

    out_graphs: dict[str, dict] = {}
    n_records = 0
    edge_counts: dict[str, int] = {}
    node_type_counts: dict[str, int] = {}

    with open(cfg["out_elements"], "w") as ef:
        for doc_id in sorted(v1.keys()):
            v1_doc = v1[doc_id]
            elements_doc = elements.get(doc_id, {})
            section_paths_doc = section_paths.get(doc_id, {})
            if not elements_doc:
                print(f"  WARN: no elements for {doc_id}, skipping")
                continue
            graph, records = convert_doc(
                doc_id, v1_doc, elements_doc, section_paths_doc,
                reclassify_text_to_header=cfg.get("reclassify_text_to_header", True),
            )
            out_graphs[doc_id] = graph
            for r in records:
                ef.write(json.dumps(r) + "\n")
                n_records += 1
            for n in graph["nodes"]:
                node_type_counts[n["type"]] = node_type_counts.get(n["type"], 0) + 1
            for e in graph["edges"]:
                edge_counts[e["type"]] = edge_counts.get(e["type"], 0) + 1

    print(f"\nwriting graphs: {cfg['out_graph']}")
    with open(cfg["out_graph"], "w") as f:
        json.dump(out_graphs, f)

    print(f"\n=== {args.dataset} v2.1 summary ===")
    print(f"  docs:    {len(out_graphs)}")
    print(f"  records: {n_records:,}")
    print(f"  nodes by type: {node_type_counts}")
    print(f"  edges by type: {edge_counts}")


if __name__ == "__main__":
    main()
