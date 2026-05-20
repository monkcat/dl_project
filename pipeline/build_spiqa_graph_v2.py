"""Build element graph v2 for SPIQA papers using ONLY original SPIQA data.

Sources (all original, no docling/PDF parsing):
  - SPIQA_testA.json (or train/val): paper_id → {all_figures: {fname: {caption, content_type}}, qa: [...]}
  - paragraphs/{paper_id}.txt:        body text, paragraph-split by blank lines
  - raw_tex/{paper_id}.tex (optional): for section_role classification via \\section{} matching

Outputs:
  - elements_v2.jsonl:        one row per element {id, doc_id, type, text, label, section_role}
  - element_graph_v2.json:    {doc_id: DocumentGraph} per v2 schema

Element ID conventions:
  - figure/table:   "<paper_id>-Figure1-1.png"  (original SPIQA filename, kept as-is)
  - caption:        "<fname>__caption"
  - paragraph:      "<paper_id>__p<idx:03d>"

Run:
  python -m pipeline.build_spiqa_graph_v2 --testA \\
      --papers 1804.04410v2 1611.05742v3 ...   # specific papers
  python -m pipeline.build_spiqa_graph_v2 --testA --limit 10
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

SPIQA_ROOT = REPO / "data" / "benchmarks" / "spiqa"
TESTA_DIR = SPIQA_ROOT / "test-A"
TESTA_JSON = TESTA_DIR / "SPIQA_testA.json"
PARAGRAPHS_DIR = TESTA_DIR / "paragraphs" / "SPIQA_train_val_test-A_extracted_paragraphs"
RAW_TEX_DIR = TESTA_DIR / "raw_tex"

# Unified location for train/val papers (extracted from the two zips at SPIQA_ROOT)
ALL_EXTRACTED = SPIQA_ROOT / "all_extracted"
ALL_PARAGRAPHS_DIR = ALL_EXTRACTED / "SPIQA_train_val_test-A_extracted_paragraphs"
ALL_RAW_TEX_DIR = ALL_EXTRACTED / "SPIQA_train_val_test-A_raw_tex"

# Train/val paths
TRAIN_DIR = SPIQA_ROOT / "train_val"
TRAIN_JSON = TRAIN_DIR / "SPIQA_train.json"
VAL_JSON = TRAIN_DIR / "SPIQA_val.json"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

LABEL_RE = re.compile(
    r"^\s*(Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

REF_RE = re.compile(r"\b(Figure|Fig\.?|Table|Tab\.?|Algorithm|Alg\.?)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

# SPIQA figure filename pattern: e.g. "1804.04410v2-Figure1-1.png", "...-Table3-2.png"
FILENAME_LABEL_RE = re.compile(
    r"-(Figure|Fig|Table|Tab|Algorithm|Alg)\s*(\d+(?:\.\d+)?)-",
    re.IGNORECASE,
)

SECTION_RE = re.compile(r"\\(sub)?section\*?\s*\{([^}]+)\}")


def _normalize_kind(raw: str) -> str:
    k = raw.lower().rstrip(".")
    if k.startswith("fig"):
        return "figure"
    if k.startswith("tab"):
        return "table"
    if k.startswith("alg"):
        return "algorithm"
    return k


def parse_label_from_caption(caption: str) -> str | None:
    """'Figure 1: A telescoping...' → 'figure 1'  (normalized lowercase)"""
    m = LABEL_RE.match(caption or "")
    if not m:
        return None
    return f"{_normalize_kind(m.group(1))} {m.group(2)}"


def parse_label_from_filename(filename: str) -> str | None:
    """'1804.04410v2-Figure1-1.png' → 'figure 1'  (most reliable for SPIQA)."""
    m = FILENAME_LABEL_RE.search(filename or "")
    if not m:
        return None
    return f"{_normalize_kind(m.group(1))} {m.group(2)}"


def parse_label(filename: str, caption: str) -> str | None:
    """Prefer filename-derived label (deterministic), fall back to caption."""
    return parse_label_from_filename(filename) or parse_label_from_caption(caption)


def classify_section_role(title: str | None) -> str:
    if not title:
        return "other"
    t = title.lower()
    if any(k in t for k in ("introduc", "motivation")):
        return "intro"
    if any(k in t for k in ("related work", "background", "literature", "preliminar")):
        return "related_work"
    if any(k in t for k in ("method", "approach", "model", "framework", "algorithm", "system", "architecture", "design")):
        return "method"
    if any(k in t for k in ("experiment", "result", "evaluation", "data and", "benchmark", "setup", "ablation")):
        return "result"
    if any(k in t for k in ("discussion", "analysis", "limitation")):
        return "discussion"
    if any(k in t for k in ("conclusion", "future work")):
        return "conclusion"
    if any(k in t for k in ("appendix", "supplementary", "supplement")):
        return "appendix"
    if any(k in t for k in ("reference", "bibliography", "acknowledg")):
        return "references"
    if any(k in t for k in ("abstract",)):
        return "front_matter"
    return "other"


_LATEX_DROP_CMDS = re.compile(
    r"\\(cite[a-zA-Z]*|ref|eqref|pageref|autoref|label|footnote|includegraphics|"
    r"input|include|bibliography|bibliographystyle|usepackage|documentclass|"
    r"begin|end|noindent|newpage|clearpage|smallskip|medskip|bigskip|vspace|hspace)"
    r"\*?(\[[^\]]*\])?(\{[^{}]*\})?"
)
# NOTE: \caption{...} intentionally NOT dropped — its argument (figure caption text)
# is needed for figure→section attachment via caption-position lookup in raw_tex.
_LATEX_KEEP_ARG = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^{}]*)\}")
_LATEX_BARE_CMD = re.compile(r"\\[a-zA-Z]+\*?")


def normalize_text(s: str) -> str:
    """Drop LaTeX noise commands (cite/ref/label), keep argument of formatting
    commands (\\textbf{X} → X), then strip remaining non-alphanumeric and lowercase.
    """
    s = _LATEX_DROP_CMDS.sub(" ", s)
    # Repeat to handle nested formatting: \textbf{\emph{X}} → \textbf{X} → X
    for _ in range(3):
        new_s = _LATEX_KEEP_ARG.sub(lambda m: " " + m.group(2) + " ", s)
        if new_s == s:
            break
        s = new_s
    s = _LATEX_BARE_CMD.sub(" ", s)
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_section_titles_ordered(raw_tex: str | None) -> list[str]:
    """Return LaTeX section titles in document order (only top-level \\section, not subsections)."""
    if not raw_tex:
        return []
    titles: list[str] = []
    for m in SECTION_RE.finditer(raw_tex):
        if m.group(1) is not None:
            # this is \subsection — skip top-level for now
            continue
        titles.append(m.group(2).strip())
    return titles


def _section_at_norm_offset(sec_starts_norm: list[tuple[int, str]], idx: int) -> str | None:
    """Given normalized offsets [(start, title), ...] in document order, return the
    title of the latest section whose start <= idx, else None."""
    assigned = None
    for start, title in reversed(sec_starts_norm):
        if idx >= start:
            assigned = title
            break
    return assigned


def assign_figures_to_sections(
    all_figures: dict[str, dict],
    raw_tex: str,
    refer_to_edges: list[tuple[str, str]],
) -> dict[str, str | None]:
    """For each figure filename, return the LaTeX section title it falls in (or None).

    Strategy 1 (primary): locate the figure's caption text in raw_tex.
    Strategy 2 (fallback): plurality vote of refer_to edge source sections.
                          (caller computes para→section beforehand; this fn just
                          accepts pre-resolved (src_section_title, fname) tuples.)
    """
    out: dict[str, str | None] = {fname: None for fname in all_figures}
    if not raw_tex:
        return out

    sec_matches = [m for m in SECTION_RE.finditer(raw_tex) if m.group(1) is None]
    if not sec_matches:
        return out

    tex_norm = normalize_text(raw_tex)
    sec_starts_norm: list[tuple[int, str]] = []
    for m in sec_matches:
        prefix_norm_len = len(normalize_text(raw_tex[: m.start()]))
        sec_starts_norm.append((prefix_norm_len, m.group(2).strip()))

    # Primary: caption text match
    for fname, meta in all_figures.items():
        cap = meta.get("caption", "") or ""
        # Strip "Figure 1:" prefix to get the descriptive part
        m = LABEL_RE.match(cap)
        if m:
            cap_body = cap[m.end():].lstrip(": ").strip()
        else:
            cap_body = cap.strip()
        if not cap_body:
            continue
        cap_norm = normalize_text(cap_body)
        if not cap_norm:
            continue
        snippet = cap_norm[:40]
        idx = tex_norm.find(snippet)
        if idx < 0 and len(snippet) > 25:
            idx = tex_norm.find(snippet[:25])
        if idx < 0:
            continue
        out[fname] = _section_at_norm_offset(sec_starts_norm, idx)

    # Fallback: plurality vote from refer_to edges
    from collections import Counter
    refer_votes: dict[str, Counter] = {}
    for src_section_title, fname in refer_to_edges:
        if fname in out and out[fname] is None and src_section_title:
            refer_votes.setdefault(fname, Counter())[src_section_title] += 1
    for fname, votes in refer_votes.items():
        if votes:
            out[fname] = votes.most_common(1)[0][0]

    return out


def assign_paragraph_sections(paras: list[str], raw_tex: str) -> list[str | None]:
    """For each paragraph, return the LaTeX section title it falls in (or None).

    Strategy: locate paragraph's first-50-char-normalized substring in normalized raw_tex,
    then find the latest top-level \\section{} before that position.
    """
    sec_matches = [m for m in SECTION_RE.finditer(raw_tex) if m.group(1) is None]
    if not sec_matches:
        return [None] * len(paras)

    tex_norm = normalize_text(raw_tex)
    sec_starts_norm: list[tuple[int, str]] = []
    for m in sec_matches:
        prefix_norm_len = len(normalize_text(raw_tex[: m.start()]))
        sec_starts_norm.append((prefix_norm_len, m.group(2).strip()))

    out: list[str | None] = []
    for p in paras:
        p_norm = normalize_text(p)
        if not p_norm:
            out.append(None)
            continue
        snippet = p_norm[:50]
        idx = tex_norm.find(snippet)
        if idx < 0:
            idx = tex_norm.find(snippet[:30])
        if idx < 0:
            out.append(None)
            continue
        # find latest section start <= idx
        assigned = None
        for start_norm, title in reversed(sec_starts_norm):
            if idx >= start_norm:
                assigned = title
                break
        out.append(assigned)

    # Fallback: for paragraphs that failed direct matching (heavy math/special chars),
    # inherit from the nearest previously-assigned neighbor. Since paragraphs are in
    # reading order, this is almost always correct.
    last_assigned: str | None = None
    for i in range(len(out)):
        if out[i] is not None:
            last_assigned = out[i]
        elif last_assigned is not None:
            out[i] = last_assigned
    # Tail-fill from the right for any paragraphs before the first assignment
    next_assigned: str | None = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is not None:
            next_assigned = out[i]
        elif next_assigned is not None:
            out[i] = next_assigned

    return out


# ────────────────────────────────────────────────────────────────────────────
# Graph build
# ────────────────────────────────────────────────────────────────────────────

def build_paper_graph(paper_id: str, paper: dict, paras: list[str], raw_tex: str | None) -> tuple[dict, list[dict]]:
    """Returns (graph_dict, element_records).

    graph_dict is per v2 DocumentGraph schema.
    element_records is the flat list of element-record dicts for elements_v2.jsonl.
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    records: list[dict] = []

    # ── 1. Visual + caption elements (from all_figures, GT-quality) ──
    label_to_fname: dict[str, str] = {}
    for fname, meta in paper.get("all_figures", {}).items():
        ftype = "figure" if meta.get("content_type") != "table" else "table"
        if str(meta.get("content_type", "")).lower() == "table":
            ftype = "table"
        elif str(meta.get("content_type", "")).lower() == "figure":
            ftype = "figure"
        label = parse_label(fname, meta.get("caption", ""))
        # figure / table node
        nodes.append({
            "id": fname,
            "type": ftype,
            "page": None,
            "section_path": None,
            "section_role": None,
            "label": label,
        })
        records.append({
            "id": fname,
            "doc_id": paper_id,
            "type": ftype,
            "text": None,
            "image_path": f"images_224px/{fname}",
            "label": label,
        })
        if label:
            label_to_fname[label.lower()] = fname

        # caption node
        cap_id = f"{fname}__caption"
        nodes.append({
            "id": cap_id,
            "type": "caption",
            "page": None,
            "section_path": None,
            "section_role": None,
            "label": label,
        })
        records.append({
            "id": cap_id,
            "doc_id": paper_id,
            "type": "caption",
            "text": meta.get("caption", ""),
            "label": label,
        })
        # caption_of edge (bidirectional, GT confidence)
        edges.append({
            "src": cap_id,
            "dst": fname,
            "type": "caption_of",
            "confidence": 1.0,
            "bidirectional": True,
            "evidence": "all_figures dict (SPIQA original)",
        })

    # ── 2. Section_header nodes (v2.1) ──
    section_titles_ordered = extract_section_titles_ordered(raw_tex)
    # Fallback: papers without raw_tex (no section_header detectable) get a single
    # "Untitled" section so all elements are still attached via contains.
    used_fallback_section = not section_titles_ordered
    if used_fallback_section:
        section_titles_ordered = ["Untitled"]

    section_id_by_title: dict[str, str] = {}
    section_id_in_order: list[str] = []
    for i, title in enumerate(section_titles_ordered):
        sid = f"{paper_id}__sec{i:02d}"
        role = classify_section_role(title)
        nodes.append({
            "id": sid,
            "type": "section_header",
            "page": None,
            "section_path": [i],
            "section_role": role,
            "label": title,
        })
        records.append({
            "id": sid,
            "doc_id": paper_id,
            "type": "section_header",
            "text": title,
            "section_title": title,
            "section_role": role,
        })
        # last-write-wins on duplicate section titles
        section_id_by_title[title] = sid
        section_id_in_order.append(sid)

    # section_next edges (sequential, directional)
    for i in range(len(section_id_in_order) - 1):
        edges.append({
            "src": section_id_in_order[i],
            "dst": section_id_in_order[i + 1],
            "type": "section_next",
            "confidence": 1.0,
            "bidirectional": False,
        })

    # ── 3. Paragraph elements ──
    if raw_tex:
        section_assignments = assign_paragraph_sections(paras, raw_tex)
    elif used_fallback_section:
        # No raw_tex available — attach every paragraph to the single "Untitled" section
        section_assignments = ["Untitled"] * len(paras)
    else:
        section_assignments = [None] * len(paras)
    para_ids: list[str] = []
    para_section_ids: list[str | None] = []
    for i, p in enumerate(paras):
        pid = f"{paper_id}__p{i:03d}"
        sec_title = section_assignments[i] if i < len(section_assignments) else None
        role = classify_section_role(sec_title)
        nodes.append({
            "id": pid,
            "type": "text",
            "page": None,
            "section_path": None,
            "section_role": role,
            "label": None,
        })
        records.append({
            "id": pid,
            "doc_id": paper_id,
            "type": "text",
            "text": p,
            "section_title": sec_title,
            "section_role": role,
        })
        para_ids.append(pid)
        para_section_ids.append(section_id_by_title.get(sec_title) if sec_title else None)

    # contains edges (section_header ↔ paragraph) — bidirectional
    for pid, sid in zip(para_ids, para_section_ids):
        if sid is None:
            continue
        edges.append({
            "src": sid,
            "dst": pid,
            "type": "contains",
            "confidence": 1.0,
            "bidirectional": True,
        })

    # ── 3.5 Attach figures/tables/captions to sections via contains ──
    # Build (paragraph_section_title, fname) list for refer_to-based fallback voting.
    refer_to_section_votes: list[tuple[str, str]] = []
    for i, p in enumerate(paras):
        para_sec = section_assignments[i] if section_assignments and i < len(section_assignments) else None
        if not para_sec:
            continue
        seen = set()
        for kind, num in REF_RE.findall(p):
            kind_l = kind.lower().rstrip(".")
            if kind_l.startswith("fig"):
                kind_l = "figure"
            elif kind_l.startswith("tab"):
                kind_l = "table"
            elif kind_l.startswith("alg"):
                kind_l = "algorithm"
            key = f"{kind_l} {num}"
            if key in seen:
                continue
            seen.add(key)
            dst = label_to_fname.get(key)
            if dst:
                refer_to_section_votes.append((para_sec, dst))

    fig_section_title = assign_figures_to_sections(
        paper.get("all_figures", {}), raw_tex or "", refer_to_section_votes
    )

    for fname in paper.get("all_figures", {}):
        sec_title = fig_section_title.get(fname)
        if not sec_title and used_fallback_section:
            sec_title = "Untitled"
        if not sec_title:
            continue
        sid = section_id_by_title.get(sec_title)
        if not sid:
            continue
        # section → figure
        edges.append({
            "src": sid,
            "dst": fname,
            "type": "contains",
            "confidence": 0.9,
            "bidirectional": True,
            "evidence": "caption text position in raw_tex",
        })
        # section → caption (mirror)
        cap_id = f"{fname}__caption"
        edges.append({
            "src": sid,
            "dst": cap_id,
            "type": "contains",
            "confidence": 0.9,
            "bidirectional": True,
            "evidence": "via figure's section",
        })

    # ── 4. refer_to edges (regex on paragraph text → figure/table) ──
    for i, p in enumerate(paras):
        seen = set()
        for kind, num in REF_RE.findall(p):
            kind_l = kind.lower().rstrip(".")
            if kind_l.startswith("fig"):
                kind_l = "figure"
            elif kind_l.startswith("tab"):
                kind_l = "table"
            elif kind_l.startswith("alg"):
                kind_l = "algorithm"
            key = f"{kind_l} {num}"
            if key in seen:
                continue
            seen.add(key)
            dst = label_to_fname.get(key)
            if not dst:
                continue
            edges.append({
                "src": para_ids[i],
                "dst": dst,
                "type": "refer_to",
                "confidence": 0.9,
                "bidirectional": False,
                "evidence": f"regex match '{key}'",
            })

    # ── 5. reading_next edges (paragraph sequential, INTRA-SECTION ONLY, bidirectional) ──
    for i in range(len(para_ids) - 1):
        s1, s2 = para_section_ids[i], para_section_ids[i + 1]
        if s1 is None or s2 is None or s1 != s2:
            continue  # cross-section transition handled by section_next, not reading_next
        edges.append({
            "src": para_ids[i],
            "dst": para_ids[i + 1],
            "type": "reading_next",
            "confidence": 1.0,
            "bidirectional": True,
        })

    graph = {
        "doc_id": paper_id,
        "schema_version": "v2.1",
        "nodes": nodes,
        "edges": edges,
    }
    return graph, records


# ────────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────────

def load_paragraphs(paper_id: str, base_dir: Path = PARAGRAPHS_DIR) -> list[str]:
    path = base_dir / f"{paper_id}.txt"
    if not path.exists():
        # Fallback to the unified extracted dir (covers train/val)
        path = ALL_PARAGRAPHS_DIR / f"{paper_id}.txt"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def load_raw_tex(paper_id: str, base_dir: Path = RAW_TEX_DIR) -> str | None:
    path = base_dir / f"{paper_id}.txt"
    if not path.exists():
        path = ALL_RAW_TEX_DIR / f"{paper_id}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


SPLIT_CONFIG = {
    "testA": {"json": TESTA_JSON, "out_dir": TESTA_DIR},
    "train": {"json": TRAIN_JSON, "out_dir": TRAIN_DIR},
    "val":   {"json": VAL_JSON,   "out_dir": TRAIN_DIR},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default=None, choices=("testA", "train", "val"),
                    help="which SPIQA split to process")
    ap.add_argument("--testA", action="store_true", help="(deprecated alias for --split testA)")
    ap.add_argument("--papers", nargs="+", default=None, help="specific paper_ids to process")
    ap.add_argument("--limit", type=int, default=None, help="cap on # papers")
    ap.add_argument("--output_dir", type=str, default=None, help="where to write outputs (default: split-specific)")
    ap.add_argument("--suffix", type=str, default="_v2",
                    help="filename suffix (default _v2; use e.g. _v2_train for train split)")
    ap.add_argument("--verbose", action="store_true", help="print per-paper progress (default: every 100)")
    args = ap.parse_args()

    if args.split is None:
        args.split = "testA" if args.testA else None
    if args.split is None:
        print("must specify --split testA|train|val", file=sys.stderr)
        sys.exit(1)

    cfg = SPLIT_CONFIG[args.split]
    json_path = cfg["json"]
    out_dir = Path(args.output_dir) if args.output_dir else cfg["out_dir"]

    print(f"loading {json_path}")
    with open(json_path) as f:
        split_data = json.load(f)
    print(f"  {len(split_data):,} papers in {args.split}")

    if args.papers:
        target_ids = [p for p in args.papers if p in split_data]
        missing = [p for p in args.papers if p not in split_data]
        if missing:
            print(f"WARNING: missing papers: {missing[:5]}", file=sys.stderr)
    else:
        target_ids = list(split_data.keys())
        if args.limit:
            # Pick papers with paragraphs available and 2-5 figures (small for visualization)
            picked = []
            for pid in target_ids:
                paper = split_data[pid]
                n_fig = len(paper.get("all_figures", {}))
                if 2 <= n_fig <= 5:
                    para_path = PARAGRAPHS_DIR / f"{pid}.txt"
                    if not para_path.exists():
                        para_path = ALL_PARAGRAPHS_DIR / f"{pid}.txt"
                    if para_path.exists():
                        picked.append(pid)
                if len(picked) >= args.limit:
                    break
            target_ids = picked

    print(f"Processing {len(target_ids):,} papers")

    out_dir.mkdir(parents=True, exist_ok=True)
    graphs: dict[str, dict] = {}
    elements_path = out_dir / f"elements{args.suffix}.jsonl"
    n_records = 0
    n_skipped = 0
    edge_counts: dict[str, int] = {}

    with open(elements_path, "w") as ef:
        for i, pid in enumerate(target_ids):
            paper = split_data[pid]
            paras = load_paragraphs(pid)
            if not paras:
                n_skipped += 1
                if args.verbose:
                    print(f"  {pid}: no paragraphs, skipped")
                continue
            raw_tex = load_raw_tex(pid)
            graph, records = build_paper_graph(pid, paper, paras, raw_tex)
            graphs[pid] = graph
            for r in records:
                ef.write(json.dumps(r) + "\n")
                n_records += 1
            for e in graph["edges"]:
                edge_counts[e["type"]] = edge_counts.get(e["type"], 0) + 1

            if args.verbose:
                print(f"  {pid}: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges "
                      f"(cap={sum(1 for e in graph['edges'] if e['type']=='caption_of')}, "
                      f"ref={sum(1 for e in graph['edges'] if e['type']=='refer_to')}, "
                      f"next={sum(1 for e in graph['edges'] if e['type']=='reading_next')})"
                      f"{'  [no tex]' if not raw_tex else ''}")
            elif (i + 1) % 500 == 0:
                print(f"  [{i+1:,}/{len(target_ids):,}] processed (skipped={n_skipped})")

    graph_path = out_dir / f"element_graph{args.suffix}.json"
    with open(graph_path, "w") as f:
        json.dump(graphs, f)

    print()
    print(f"Wrote {len(graphs)} graphs → {graph_path}")
    print(f"Wrote {n_records} element records → {elements_path}")
    print(f"Edge totals: {edge_counts}")


if __name__ == "__main__":
    main()
