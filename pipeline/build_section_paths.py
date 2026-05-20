"""Build section_path metadata for SPIQA test-A elements using MinerU output.

For each paper, we walk MinerU's content_list (in reading order) and assign
a section_path tuple to every element based on the most recent headings.

Output:
    data/benchmarks/spiqa/test-A/section_paths.json
    {
        paper_id: {
            "headings": [{"level": 1|2|3, "text": ..., "page_idx": ...}, ...],
            "elements": [
                {"id": ..., "type": ..., "section_path": [s1, s2, ...],
                 "section_role": "intro/method/result/...",
                 "page_idx": ...,
                 "label": "Figure 1" | "Table 2" | None,
                 "spiqa_fname": ... | None}
            ]
        }, ...
    }

Mapping logic:
    - Heading at level L resets all deeper levels
    - L1 (title) gets index 0 (we ignore for elements)
    - L2 headings get sequential index 1, 2, ...
    - Sub-elements get section_path = [L2_idx, position_within_section, ...]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("data/benchmarks/spiqa/test-A")
SPIQA_JSON = ROOT / "SPIQA_testA.json"
MINERU_DIR = ROOT / "mineru_out"
OUT_JSON = ROOT / "section_paths.json"

LABEL_RE = re.compile(r"(?i)\b(figure|fig\.?|table|tab\.?)\s*([0-9]+(?:\.[0-9]+)?)\b")

# Heuristic mapping of section title keywords → semantic role
ROLE_KEYWORDS = [
    ("intro",       ["introduction", "background"]),
    ("related",     ["related work", "prior work"]),
    ("method",      ["method", "approach", "model", "framework", "algorithm",
                     "architecture", "formulation", "design"]),
    ("experiment",  ["experiment", "experimental", "evaluation", "evaluat", "result",
                     "ablation", "study"]),
    ("discussion",  ["discussion", "analysis"]),
    ("conclusion",  ["conclusion", "summary", "future work"]),
    ("appendix",    ["appendix"]),
    ("abstract",    ["abstract"]),
    ("references",  ["references", "bibliography"]),
    ("acks",        ["acknowledgment", "acknowledgement"]),
]


def classify_section_role(title: str) -> str:
    t = title.lower()
    for role, keys in ROLE_KEYWORDS:
        if any(k in t for k in keys):
            return role
    return "other"


def parse_label(caption: str) -> str | None:
    if not caption:
        return None
    m = LABEL_RE.search(caption)
    if not m:
        return None
    kind = "figure" if m.group(1).lower().startswith("fig") else "table"
    return f"{kind}{m.group(2)}"


def map_to_spiqa_fname(label: str | None, paper_id: str, spiqa_paper: dict) -> str | None:
    """Try to find the SPIQA filename whose caption matches label."""
    if not label:
        return None
    for fname, meta in spiqa_paper["all_figures"].items():
        if parse_label(meta.get("caption", "")) == label:
            return fname
    return None


def build_for_paper(paper_id: str, spiqa_paper: dict) -> dict | None:
    cl_path = MINERU_DIR / paper_id / "auto" / f"{paper_id}_content_list.json"
    if not cl_path.exists():
        return None

    cl = json.loads(cl_path.read_text())
    headings: list[dict] = []          # [{level, text, role, idx_in_doc}]
    elements: list[dict] = []          # output

    # Walk in reading order
    cur_path: list[int] = []           # current section_path stack (indices)
    h_counters: dict[int, int] = {}    # level → counter (so each L2 heading gets idx 1, 2, ...)
    elem_counter_in_section = 0        # paragraph/figure index within current section
    cur_role = "other"

    for entry in cl:
        etype = entry.get("type")
        if etype == "text" and "text_level" in entry:
            level = entry["text_level"]
            text = entry.get("text", "")
            # Reset deeper levels
            cur_path = cur_path[: level - 1]
            h_counters = {k: v for k, v in h_counters.items() if k < level}
            h_counters[level] = h_counters.get(level, 0) + 1
            cur_path.append(h_counters[level])
            cur_role = classify_section_role(text) if level <= 2 else cur_role
            elem_counter_in_section = 0
            headings.append({
                "level": level, "text": text, "role": cur_role,
                "page_idx": entry.get("page_idx", -1),
                "section_path": list(cur_path),
            })
            continue

        if etype == "text" and entry.get("text_level") is None:
            elem_counter_in_section += 1
            elements.append({
                "id": f"{paper_id}__para_{len(elements):03d}",
                "type": "text",
                "section_path": list(cur_path) + [elem_counter_in_section],
                "section_role": cur_role,
                "page_idx": entry.get("page_idx", -1),
                "label": None,
                "spiqa_fname": None,
                "text": entry.get("text", ""),
            })

        elif etype in ("image", "table"):
            cap = (entry.get("image_caption") if etype == "image"
                   else entry.get("table_caption")) or [""]
            cap_str = cap[0] if cap else ""
            label = parse_label(cap_str)
            elem_counter_in_section += 1
            elements.append({
                "id": f"{paper_id}__{etype}_{len(elements):03d}",
                "type": "figure" if etype == "image" else "table",
                "section_path": list(cur_path) + [elem_counter_in_section],
                "section_role": cur_role,
                "page_idx": entry.get("page_idx", -1),
                "label": label,
                "spiqa_fname": map_to_spiqa_fname(label, paper_id, spiqa_paper),
                "caption": cap_str,
            })

    return {
        "headings": headings,
        "elements": elements,
        "n_elements": len(elements),
        "n_headings": len(headings),
    }


def main():
    spiqa = json.loads(SPIQA_JSON.read_text())
    out = {}
    n_papers_with_mineru = 0
    n_visuals_mapped = 0
    n_visuals_total = 0

    for paper_id in spiqa.keys():
        rec = build_for_paper(paper_id, spiqa[paper_id])
        if rec is None:
            continue
        out[paper_id] = rec
        n_papers_with_mineru += 1
        for e in rec["elements"]:
            if e["type"] in ("figure", "table"):
                n_visuals_total += 1
                if e["spiqa_fname"]:
                    n_visuals_mapped += 1

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"papers with MinerU output: {n_papers_with_mineru}")
    print(f"visual elements: {n_visuals_total} (mapped to SPIQA: {n_visuals_mapped})")
    print(f"saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
