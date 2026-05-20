"""Parse SPIQA test-A raw_tex (LaTeX) to extract:
  - Section / subsection structure
  - Figure / table environments with captions and labels
  - Section_path for each figure/table by position
  - Section_path for each text paragraph by position

Output:
  data/benchmarks/spiqa/test-A/section_paths_latex.json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("data/benchmarks/spiqa/test-A")
RAW_TEX_DIR = ROOT / "raw_tex"
SPIQA_JSON = ROOT / "SPIQA_testA.json"
OUT_JSON = ROOT / "section_paths_latex.json"

# LaTeX section commands → level
SECTION_LEVEL = {
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
    "subparagraph": 5,
}

# Heading regex: matches \section*?{...}, \subsection*?{...}, etc.
HEADING_RE = re.compile(
    r"\\(sub)?(?:sub)?(?:section|paragraph)\*?\s*\{([^{}]*)\}"
)
# More precise: capture the command name
HEADING_NAME_RE = re.compile(
    r"\\((?:sub){0,2}section|paragraph|subparagraph)\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}"
)
# Figure / table environment start
ENV_START_RE = re.compile(r"\\begin\{(figure|table)\*?\}")
ENV_END_RE = re.compile(r"\\end\{(figure|table)\*?\}")
CAPTION_RE = re.compile(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")

ROLE_KEYWORDS = [
    ("intro", ["introduction"]),
    ("background", ["background", "preliminari"]),
    ("related", ["related work", "prior work"]),
    ("method", ["method", "approach", "model", "framework", "algorithm",
                "architecture", "formulation", "proposed"]),
    ("experiment", ["experiment", "evaluation", "result", "ablation", "study"]),
    ("discussion", ["discussion", "analysis", "limitation"]),
    ("conclusion", ["conclusion", "summary", "future work"]),
    ("appendix", ["appendix"]),
    ("abstract", ["abstract"]),
    ("references", ["references", "bibliography", "acknowledg"]),
]


def classify_role(title: str) -> str:
    t = title.lower()
    for role, keys in ROLE_KEYWORDS:
        if any(k in t for k in keys):
            return role
    return "other"


def normalize_caption(s: str) -> str:
    """Strip LaTeX commands and normalize whitespace for matching."""
    # Remove \cite{...}, \ref{...}, \label{...}, \footnote{...} etc.
    s = re.sub(r"\\(cite|ref|label|footnote|citep|citet|emph|textbf|textit)\{[^}]*\}", "", s)
    # Remove other commands like \\, \&, \%
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    # Strip { }
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_paper(content: str, paper_id: str, spiqa_paper: dict) -> dict:
    """Walk LaTeX content in order, build section_path for each element."""
    headings: list[dict] = []
    elements: list[dict] = []

    # State
    cur_path: list[int] = []
    h_counters: dict[int, int] = {}
    cur_role = "other"

    # Tokenize: scan for section commands and figure/table environments
    # We walk char-by-char and find matches at each position
    pos = 0
    n = len(content)
    elem_in_section = 0
    text_buffer = ""
    last_text_pos = 0

    def flush_text(end_pos: int):
        nonlocal elem_in_section, text_buffer, last_text_pos
        text = content[last_text_pos:end_pos]
        # Split into paragraphs by double newline
        for para in re.split(r"\n\s*\n", text):
            stripped = re.sub(r"\s+", " ", para).strip()
            if len(stripped) < 50:  # skip very short fragments (LaTeX commands etc.)
                continue
            elem_in_section += 1
            elements.append({
                "id": f"{paper_id}__para_{len(elements):03d}",
                "type": "text",
                "section_path": list(cur_path) + [elem_in_section],
                "section_role": cur_role,
                "label": None,
                "caption": None,
                "spiqa_fname": None,
                "preview": stripped[:120],
            })
        last_text_pos = end_pos

    while pos < n:
        # Try heading match
        m_head = HEADING_NAME_RE.match(content, pos)
        m_env = ENV_START_RE.match(content, pos)

        if m_head:
            flush_text(pos)
            cmd = m_head.group(1)
            title = m_head.group(2)
            level = SECTION_LEVEL.get(cmd, 1)
            # Save prev counter at this level (so we increment, not reset)
            prev_at_level = h_counters.get(level, 0)
            # Drop counters at this level and deeper (k >= level)
            h_counters = {k: v for k, v in h_counters.items() if k < level}
            h_counters[level] = prev_at_level + 1
            # Rebuild cur_path: ancestor levels' counters + current
            cur_path = [h_counters[k] for k in sorted(h_counters)]
            if level == 1:
                cur_role = classify_role(title)
            elem_in_section = 0
            headings.append({
                "level": level, "title": normalize_caption(title), "role": cur_role,
                "section_path": list(cur_path),
            })
            pos = m_head.end()
            last_text_pos = pos
            continue

        if m_env:
            flush_text(pos)
            env_kind = m_env.group(1)  # figure or table
            # Find matching \end{env_kind} (handle nested envs simply)
            end_re = re.compile(rf"\\end\{{{env_kind}\*?\}}")
            m_end = end_re.search(content, m_env.end())
            if m_end is None:
                pos = m_env.end()
                continue
            inner = content[m_env.end(): m_end.start()]
            cap_m = CAPTION_RE.search(inner)
            caption = normalize_caption(cap_m.group(1)) if cap_m else ""
            lbl_m = LABEL_RE.search(inner)
            label = lbl_m.group(1) if lbl_m else None

            elem_in_section += 1
            etype = "figure" if env_kind == "figure" else "table"
            elements.append({
                "id": f"{paper_id}__{etype}_{len(elements):03d}",
                "type": etype,
                "section_path": list(cur_path) + [elem_in_section],
                "section_role": cur_role,
                "label": label,
                "caption": caption,
                "spiqa_fname": None,    # filled in later by matching
                "preview": caption[:120],
            })
            pos = m_end.end()
            last_text_pos = pos
            continue

        pos += 1

    flush_text(n)

    # Now match each LaTeX figure/table element to SPIQA fname by caption similarity
    # SPIQA captions look like "Figure 1: Architecture..." and LaTeX captions are just "Architecture..."
    # We strip "Figure N:" / "Table N:" prefix from SPIQA before matching.
    spiqa_visuals = []
    for fname, meta in spiqa_paper["all_figures"].items():
        sp_cap = meta.get("caption", "")
        # Strip "Figure N: " or "Table N: " prefix
        sp_cap_stripped = re.sub(r"^(figure|fig\.?|table|tab\.?)\s*[0-9]+(\.[0-9]+)?[:.]?\s*",
                                  "", sp_cap, flags=re.IGNORECASE)
        spiqa_visuals.append((fname, normalize_caption(sp_cap_stripped),
                              meta.get("content_type", "figure")))

    used_spiqa = set()
    for elem in elements:
        if elem["type"] not in ("figure", "table"):
            continue
        best_fname = None
        best_score = 0.0
        for fname, sp_cap, sp_type in spiqa_visuals:
            if fname in used_spiqa:
                continue
            if sp_type != elem["type"]:
                continue
            if not sp_cap or not elem["caption"]:
                continue
            # Sequence similarity
            score = SequenceMatcher(None, elem["caption"][:200], sp_cap[:200]).ratio()
            if score > best_score:
                best_score = score
                best_fname = fname
        if best_score >= 0.6:
            elem["spiqa_fname"] = best_fname
            elem["match_score"] = round(best_score, 3)
            used_spiqa.add(best_fname)

    return {
        "headings": headings,
        "elements": elements,
        "n_elements": len(elements),
        "n_visuals_matched": sum(1 for e in elements
                                  if e["type"] in ("figure", "table") and e["spiqa_fname"]),
        "n_spiqa_visuals": len(spiqa_visuals),
    }


def main():
    spiqa = json.loads(SPIQA_JSON.read_text())
    out = {}
    n_with_tex = 0
    n_visuals_matched = 0
    n_visuals_total_spiqa = 0
    n_visuals_total_latex = 0

    for paper_id in spiqa.keys():
        tex_path = RAW_TEX_DIR / f"{paper_id}.txt"
        if not tex_path.exists():
            continue
        n_with_tex += 1
        content = tex_path.read_text(errors="ignore")
        rec = parse_paper(content, paper_id, spiqa[paper_id])
        out[paper_id] = rec
        n_visuals_matched += rec["n_visuals_matched"]
        n_visuals_total_spiqa += rec["n_spiqa_visuals"]
        n_visuals_total_latex += sum(1 for e in rec["elements"]
                                       if e["type"] in ("figure", "table"))

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"papers with raw_tex: {n_with_tex}/{len(spiqa)}")
    print(f"latex visuals: {n_visuals_total_latex}, spiqa visuals: {n_visuals_total_spiqa}")
    print(f"latex→spiqa matched (caption ratio ≥0.6): "
          f"{n_visuals_matched}/{n_visuals_total_spiqa} "
          f"({n_visuals_matched/n_visuals_total_spiqa:.2%})")
    print(f"saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
