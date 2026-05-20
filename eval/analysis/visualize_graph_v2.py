"""Visualize a per-document element graph (v2 schema) as an interactive HTML page.

Uses pyvis (vis.js wrapper). Output is a standalone .html file viewable in any browser.

Visual conventions:
  - Nodes:
      text         small grey dot
      figure       blue square (large)
      table        green square (large)
      caption      yellow triangle
      equation     red diamond
  - Edges:
      caption_of    yellow, thick, bidirectional arrows
      refer_to      blue, medium, single-arrow
      reading_next  dashed grey, thin, bidirectional
  - Hover over a node → shows id, type, label, section_role, text preview
  - Hover over an edge → shows type, confidence, evidence

Run:
  python -m eval.analysis.visualize_graph_v2 \\
      --graph data/benchmarks/spiqa/test-A/element_graph_v2.json \\
      --elements data/benchmarks/spiqa/test-A/elements_v2.jsonl \\
      --paper 1804.04410v2 \\
      --out output/graph_1804.04410v2.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyvis.network import Network

REPO = Path(__file__).resolve().parent.parent.parent

# ────────────────────────────────────────────────────────────────────────────
# Styles
# ────────────────────────────────────────────────────────────────────────────

NODE_STYLE = {
    "text": {
        "color": {"background": "#E5E7EB", "border": "#6B7280", "highlight": {"background": "#D1D5DB", "border": "#374151"}},
        "shape": "dot", "size": 14,
        "font": {"color": "#374151", "size": 11},
    },
    "section_header": {
        "color": {"background": "#FFFFFF", "border": "#111827",
                  "highlight": {"background": "#F3F4F6", "border": "#000000"}},
        "shape": "box", "size": 40,
        "font": {"color": "#111827", "size": 14, "bold": True, "face": "Inter, Arial, sans-serif"},
        "margin": 14,
        "widthConstraint": {"minimum": 120, "maximum": 240},
        "borderWidth": 2.5,
    },
    "figure": {
        "color": {"background": "#3B82F6", "border": "#1E40AF", "highlight": {"background": "#60A5FA", "border": "#1E3A8A"}},
        "shape": "box", "size": 30,
        "font": {"color": "#FFFFFF", "size": 14, "bold": True},
    },
    "table": {
        "color": {"background": "#10B981", "border": "#047857", "highlight": {"background": "#34D399", "border": "#065F46"}},
        "shape": "box", "size": 30,
        "font": {"color": "#FFFFFF", "size": 14, "bold": True},
    },
    "caption": {
        "color": {"background": "#F59E0B", "border": "#B45309", "highlight": {"background": "#FBBF24", "border": "#92400E"}},
        "shape": "triangle", "size": 22,
        "font": {"color": "#78350F", "size": 12, "bold": True},
    },
    "equation": {
        "color": {"background": "#A855F7", "border": "#6B21A8", "highlight": {"background": "#C084FC", "border": "#581C87"}},
        "shape": "diamond", "size": 22,
        "font": {"color": "#FFFFFF", "size": 12, "bold": True},
    },
}

EDGE_STYLE = {
    "caption_of": {
        "color": {"color": "#D97706", "highlight": "#92400E", "hover": "#B45309"},
        "width": 3.5,
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.7},
                   "from": {"enabled": True, "scaleFactor": 0.7}},
    },
    "refer_to": {
        "color": {"color": "#2563EB", "highlight": "#1E3A8A", "hover": "#1E40AF"},
        "width": 2.0,
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.9}},
    },
    "contains": {
        "color": {"color": "#1F2937", "highlight": "#000000", "hover": "#374151"},
        "width": 1.8,
        "arrows": {"to": {"enabled": False}, "from": {"enabled": False}},
    },
    "section_next": {
        "color": {"color": "#7C3AED", "highlight": "#4C1D95", "hover": "#5B21B6"},
        "width": 3.0,
        "arrows": {"to": {"enabled": True, "scaleFactor": 1.2}},
    },
    "reading_next": {
        "color": {"color": "#9CA3AF", "highlight": "#4B5563", "hover": "#6B7280"},
        "width": 1.2,
        "dashes": True,
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.5},
                   "from": {"enabled": True, "scaleFactor": 0.5}},
    },
}


def load_elements_index(elements_jsonl: Path, doc_id: str) -> dict[str, dict]:
    """Return {element_id: record} for one doc."""
    out: dict[str, dict] = {}
    with open(elements_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("doc_id") == doc_id:
                out[r["id"]] = r
    return out


def short_label(node: dict, record: dict | None) -> str:
    """Pick a short label to display ON the node."""
    ntype = node.get("type")
    if ntype == "section_header":
        title = node.get("label") or ""
        return f"§ {title}" if title else "§"
    if node.get("label"):
        return node["label"]
    nid = node["id"]
    if ntype == "text" and record and record.get("text"):
        return record["text"][:30].rstrip() + "…"
    if ntype == "caption":
        return "caption"
    # for figures/tables, strip the paper prefix
    short = nid.split("-")[-2] if "-" in nid else nid
    return short[:25]


def title_html(node: dict, record: dict | None) -> str:
    """Detailed hover-tooltip content."""
    lines = [
        f"<b>id</b>: {node['id']}",
        f"<b>type</b>: {node.get('type')}",
    ]
    if node.get("label"):
        lines.append(f"<b>label</b>: {node['label']}")
    if node.get("section_role"):
        lines.append(f"<b>section_role</b>: {node['section_role']}")
    if record:
        if record.get("section_title"):
            lines.append(f"<b>section</b>: {record['section_title']}")
        if record.get("text"):
            preview = record["text"][:300].replace("\n", " ")
            if len(record["text"]) > 300:
                preview += "…"
            lines.append(f"<b>text</b>: {preview}")
    return "<br>".join(lines)


def edge_title(edge: dict) -> str:
    parts = [f"<b>{edge['type']}</b>"]
    parts.append(f"confidence: {edge.get('confidence', 1.0):.2f}")
    if edge.get("bidirectional"):
        parts.append("bidirectional")
    if edge.get("evidence"):
        parts.append(f"evidence: {edge['evidence']}")
    return "<br>".join(parts)


def visualize(graph: dict, elements: dict[str, dict], out_path: Path):
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="#1F2937",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    # Barnes-Hut force-directed: good for discovering section clusters via the
    # `contains` edge pulling each section's paragraphs toward it.
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.2,
        spring_length=150,
        spring_strength=0.05,
        damping=0.95,
    )
    net.set_options(
        """
        var options = {
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "stabilization": {"iterations": 200}
          },
          "edges": {
            "smooth": {"enabled": true, "type": "dynamic"}
          }
        }
        """
    )

    for node in graph["nodes"]:
        nid = node["id"]
        ntype = node.get("type", "text")
        style = dict(NODE_STYLE.get(ntype, NODE_STYLE["text"]))
        rec = elements.get(nid)
        net.add_node(
            nid,
            label=short_label(node, rec),
            title=title_html(node, rec),
            **style,
        )

    for edge in graph["edges"]:
        style = dict(EDGE_STYLE.get(edge["type"], {}))
        net.add_edge(
            edge["src"],
            edge["dst"],
            title=edge_title(edge),
            **style,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = net.generate_html(notebook=False)
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=str, required=True, help="path to element_graph_v2.json")
    ap.add_argument("--elements", type=str, default=None, help="path to elements_v2.jsonl (for hover text)")
    ap.add_argument("--paper", type=str, required=True, help="paper_id to visualize")
    ap.add_argument("--out", type=str, required=True, help="output html file")
    args = ap.parse_args()

    with open(args.graph) as f:
        all_graphs = json.load(f)
    if args.paper not in all_graphs:
        raise SystemExit(f"paper {args.paper} not in graph file (have {len(all_graphs)})")
    graph = all_graphs[args.paper]

    elements: dict[str, dict] = {}
    if args.elements:
        elements = load_elements_index(Path(args.elements), args.paper)

    visualize(graph, elements, Path(args.out))


if __name__ == "__main__":
    main()
