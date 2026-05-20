"""Prepare local HuggingFace-style dataset export.

Output layout:
    data/hf_export/
      element-graph-v2.1/
        README.md
        spiqa-v2.1/
          train/{element_graph.json, elements.jsonl, qa.json}
          val/{element_graph.json, elements.jsonl, qa.json}
          test-A/{element_graph.json, elements.jsonl, qa.json}
        sciegqa-v2.1/
          {element_graph.json, elements.jsonl, queries.jsonl, README.md}
        mmdocir-v2.1/
          {element_graph.json, elements.jsonl, queries.jsonl, README.md}
        SCHEMA.md   # v2.1 schema documentation

Images are NOT copied (too large). Users download from original dataset
sources and use {doc_id, eid} keys to look them up.

Run:
    python -m pipeline.prepare_hf_export
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/hf_export/element-graph-v2.1"

SPIQA_ROOT = REPO / "data/benchmarks/spiqa"
SCIEGQA_ROOT = REPO / "data/benchmarks/sciegqa"
MMDOCIR_ROOT = REPO / "data/benchmarks/mmdocir"


def hardlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except (OSError, AttributeError):
        shutil.copy(src, dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ── SPIQA ──
    spiqa_dir = OUT / "spiqa-v2.1"
    for split, src_g, src_e, src_qa in [
        ("train", SPIQA_ROOT / "train_val/element_graph_v2.json",
                  SPIQA_ROOT / "train_val/elements_v2.jsonl",
                  SPIQA_ROOT / "train_val/SPIQA_train.json"),
        ("val",   SPIQA_ROOT / "train_val/element_graph_v2_val.json",
                  SPIQA_ROOT / "train_val/elements_v2_val.jsonl",
                  SPIQA_ROOT / "train_val/SPIQA_val.json"),
        ("test-A", SPIQA_ROOT / "test-A/element_graph_v2.json",
                   SPIQA_ROOT / "test-A/elements_v2.jsonl",
                   SPIQA_ROOT / "test-A/SPIQA_testA.json"),
    ]:
        d = spiqa_dir / split
        hardlink_or_copy(src_g, d / "element_graph.json")
        hardlink_or_copy(src_e, d / "elements.jsonl")
        hardlink_or_copy(src_qa, d / "qa.json")
        print(f"  spiqa-v2.1/{split}/")

    # ── SciEGQA ──
    sciegqa_dir = OUT / "sciegqa-v2.1"
    hardlink_or_copy(SCIEGQA_ROOT / "element_graph_v2.json", sciegqa_dir / "element_graph.json")
    hardlink_or_copy(SCIEGQA_ROOT / "elements_v2.jsonl", sciegqa_dir / "elements.jsonl")
    hardlink_or_copy(SCIEGQA_ROOT / "queries_with_gt_docling.jsonl", sciegqa_dir / "queries.jsonl")
    print(f"  sciegqa-v2.1/")

    # ── MMDocIR ──
    mmdocir_dir = OUT / "mmdocir-v2.1"
    hardlink_or_copy(MMDOCIR_ROOT / "element_graph_v2.json", mmdocir_dir / "element_graph.json")
    hardlink_or_copy(MMDOCIR_ROOT / "elements_v2.jsonl", mmdocir_dir / "elements.jsonl")
    hardlink_or_copy(MMDOCIR_ROOT / "academic_queries.jsonl", mmdocir_dir / "queries.jsonl")
    print(f"  mmdocir-v2.1/")

    # ── Schema documentation ──
    (OUT / "SCHEMA.md").write_text(_schema_md())
    (OUT / "README.md").write_text(_readme_md())
    print(f"\nwrote SCHEMA.md + README.md")

    # ── Stats summary ──
    print("\n=== File sizes ===")
    total = 0
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total += size
            rel = p.relative_to(OUT)
            if size > 1_000_000:
                print(f"  {size/1e6:>8.1f} MB  {rel}")
            else:
                print(f"  {size/1e3:>8.1f} KB  {rel}")
    print(f"\n  TOTAL: {total/1e6:.1f} MB ({total/1e9:.2f} GB)")


def _schema_md() -> str:
    return """# Element Graph Schema v2.1

A typed document element graph used for multimodal retrieval research.

## Node types (`ElementType`)
- `text` — paragraph
- `figure` — image element
- `table` — table (rendered image or HTML)
- `equation` — display equation
- `caption` — caption text (separate node from its referenced figure/table)
- `section_header` — LaTeX `\\section{}` (SPIQA) or docling section node (SciEGQA, MMDocIR)

## Node attributes (`GraphNode`)
```json
{
  "id": "paperA__p003__e0012",
  "type": "text",
  "page": 3,
  "section_path": [2, 1],
  "section_role": "method",
  "label": "figure 2" | null
}
```
`section_role` ∈ {`front_matter`, `intro`, `related_work`, `method`, `result`,
`discussion`, `conclusion`, `references`, `appendix`, `other`}.

## Edge types (`EdgeType`)
| Type | Direction | Base weight | Source |
|---|---|---|---|
| `caption_of` | bidirectional | 1.0 | dataset metadata (SPIQA `all_figures`) or layout proximity (others) |
| `refer_to`   | directional   | 0.8 | regex match in paragraph text |
| `contains`   | bidirectional | 0.5 | section_header → child element |
| `reading_next` | bidirectional | 0.3 | consecutive paragraphs WITHIN same section |
| `section_next` | directional   | 0.2 | section_header → next section_header |

## Edge attributes (`ElementEdge`)
```json
{
  "src": "...",
  "dst": "...",
  "type": "caption_of",
  "confidence": 1.0,
  "bidirectional": true,
  "evidence": "all_figures dict (SPIQA original)"
}
```

## Target-attribute modifiers (applied at training/inference time only)
For `refer_to` edges:
- `SECTION_ROLE_MODIFIER[appendix] = 1.3` (cross-page evidence)
- `SECTION_ROLE_MODIFIER[references] = 0.5` (bibliography)
- `VISUAL_TARGET_MODIFIER = 1.1` if target is figure/table/equation

These are NOT stored on edges — they are computed from target node attributes
during graph_relevance / graph_propagate computation. See `pipeline/types.py`.

## File formats
- `element_graph.json`: `{doc_id: {schema_version, nodes, edges}}` (single JSON file)
- `elements.jsonl`: one element per line with all attributes (denormalized)
- `qa.json` or `queries.jsonl`: original query format from each dataset
  (preserved with GT element_id mapping into v2.1 ids)
"""


def _readme_md() -> str:
    return """# Element Graph v2.1 — Multimodal Academic Document Retrieval

Typed element graphs derived from three multimodal academic document QA datasets:
SPIQA, SciEGQA, and MMDocIR. Each document is decomposed into a typed graph
with cross-modal edges (`caption_of`), reference edges (`refer_to`), and
section-aware structural edges (`contains`, `reading_next`, `section_next`).

The graphs are designed for retrieval research:
- Training supervision via **Graph-Relevance Contrastive Loss** (GRCL)
- Inference-time **graph propagation** of retrieval scores
- Same edge-weight schema drives both (single source of truth)

## Contents

| Subset | Papers | Elements | Caption-figure pairs | Refer_to edges |
|---|---:|---:|---:|---:|
| **spiqa-v2.1/train**  | 25,459 | 1,873,124 | 262,524 | 231,101 |
| spiqa-v2.1/val        | 200    | 14,537    | 2,085   | 1,947   |
| spiqa-v2.1/test-A     | 118    | 9,039     | 1,115   | 1,191   |
| **sciegqa-v2.1**      | 80     | 21,817    | 444     | 1,103   |
| **mmdocir-v2.1**      | 75     | 15,434    | 203     | 475     |

## Per-document section count

| Dataset | min | median | mean | max |
|---|---:|---:|---:|---:|
| spiqa-v2.1 (test-A; LaTeX-based) | 1 | 7 | 7.2 | 24 |
| sciegqa-v2.1 (docling-based) | 1 | 10 | 12.8 | 54 |
| mmdocir-v2.1 (docling-based) | 3 | 9 | 13.1 | 122 |

## Schema
See `SCHEMA.md` for full schema definition (node types, edge types, attributes,
target-attribute modifiers).

## Provenance
- **SPIQA**: graph extracted from original LaTeX (`\\section{}`, `\\ref{}`) and
  `all_figures` metadata. Caption-figure pairs are author-provided.
- **SciEGQA**: graph derived from docling layout parsing + heuristic extraction.
- **MMDocIR**: graph derived from MMDocIR_layouts.parquet + heuristic extraction.

## Reference (original datasets)
- SPIQA: Pramanick et al., NeurIPS 2024 D&B
- SciEGQA: arXiv:2511.15090
- MMDocIR: Dong et al., EMNLP 2025

## Images
Image files are NOT included. Download from original dataset sources and
look up by (doc_id, eid) — the eid is preserved as the figure filename
(SPIQA) or layout_id (MMDocIR).

## License
Each subset inherits the license of the original dataset. See per-subset
README files.
"""


if __name__ == "__main__":
    main()
