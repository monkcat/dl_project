"""Graph-Relevance computation for GRCL.

For an anchor node `q` and candidate `e`, compute:

    g(q, e) = max_{π: q ⇝ e, |π| ≤ max_hops} ∏_{r ∈ π} α_τ(r) · conf(r) · γ^(|π|-1)

where:
    α_τ            = BASE_EDGE_WEIGHTS[edge type]
    conf           = edge.confidence
    γ              = path-length decay (default 0.5)
    target modifier = SECTION_ROLE_MODIFIER × VISUAL_TARGET_MODIFIER  (refer_to only)

All weights / modifiers are read from `pipeline.types` — single source of truth
for both GRCL supervision and inference-time graph propagation.

Usage:
    from pipeline.graph_relevance import graph_relevance, build_neighbor_index

    nb = build_neighbor_index(doc_graph)
    g_vec = graph_relevance("paperA__p007", candidate_ids, doc_graph,
                            gamma=0.5, max_hops=2, neighbor_index=nb)
"""
from __future__ import annotations

from typing import Optional

from .types import (
    BASE_EDGE_WEIGHTS,
    SECTION_ROLE_MODIFIER,
    VISUAL_TARGET_MODIFIER,
    DocumentGraph,
    ElementEdge,
)


def _edge_effective_weight(
    edge: ElementEdge,
    target_node: dict | None,
) -> float:
    """Compute effective weight for an edge given target node attributes.

    For `refer_to` edges, apply target's section_role and visual modifiers.
    For other edge types, only `base_weight × confidence` (no modifier).
    For bidirectional edges, the caller invokes this twice with src/dst swapped
    and takes the max — matches `pipeline.types.graph_propagate` semantics.
    """
    base = BASE_EDGE_WEIGHTS.get(edge["type"], 0.0)
    conf = edge.get("confidence", 1.0)
    w = base * conf
    if edge["type"] == "refer_to" and target_node is not None:
        role = target_node.get("section_role") or "other"
        w *= SECTION_ROLE_MODIFIER.get(role, 1.0)
        if target_node.get("type") in ("figure", "table", "equation"):
            w *= VISUAL_TARGET_MODIFIER
    return w


def build_neighbor_index(graph: DocumentGraph) -> dict[str, list[tuple[str, ElementEdge]]]:
    """Return `{node_id: [(neighbor_id, edge_dict), ...]}` for fast traversal.

    Bidirectional edges contribute to both src→dst and dst→src entries.
    """
    idx: dict[str, list[tuple[str, ElementEdge]]] = {}
    for edge in graph.get("edges", []):
        idx.setdefault(edge["src"], []).append((edge["dst"], edge))
        if edge.get("bidirectional", False):
            idx.setdefault(edge["dst"], []).append((edge["src"], edge))
    return idx


def graph_relevance(
    anchor_id: str,
    candidate_ids: list[str],
    graph: DocumentGraph,
    gamma: float = 0.5,
    max_hops: int = 2,
    neighbor_index: Optional[dict[str, list[tuple[str, ElementEdge]]]] = None,
) -> list[float]:
    """For each candidate, return max-product path weight from anchor (≤ `max_hops`).

    Args:
        anchor_id: source node id. If equal to a candidate id, returns 1.0 for self.
        candidate_ids: ordered list of candidate node ids.
        graph: per-document DocumentGraph (v2.1 schema).
        gamma: path-length decay. With γ=0.5, a 2-hop path has weight halved.
        max_hops: maximum path length to consider (default 2).
        neighbor_index: precomputed via `build_neighbor_index`. If None, computed here.

    Returns:
        List of float relevances (same length as candidate_ids, range [0, ∞)).
        Unreachable nodes get 0.0.

    Notes:
        - For `refer_to` edges, target_attribute_modifier is applied via the
          neighbor (dst) node's `section_role` and `type`.
        - For bidirectional edges, we traverse both directions; modifier is
          computed using whichever node is the *target* of the traversal step.
        - Self-loops are not produced; anchor's own entry in the result is 1.0
          if `anchor_id` is in `candidate_ids`, via the initial `max_weight` map.
    """
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    if neighbor_index is None:
        neighbor_index = build_neighbor_index(graph)

    # max_weight[node_id] = best max-product weight to reach this node from anchor
    max_weight: dict[str, float] = {anchor_id: 1.0}

    # BFS layer by layer, applying γ to each *additional* hop beyond the first.
    current_layer: list[tuple[str, float]] = [(anchor_id, 1.0)]
    for hop in range(max_hops):
        next_layer: list[tuple[str, float]] = []
        gamma_factor = gamma if hop > 0 else 1.0
        for node_id, w_so_far in current_layer:
            for nbr_id, edge in neighbor_index.get(node_id, []):
                if nbr_id == anchor_id:
                    continue  # don't bounce back to anchor
                target_node = nodes.get(nbr_id)
                edge_w = _edge_effective_weight(edge, target_node)
                new_w = w_so_far * edge_w * gamma_factor
                if new_w > max_weight.get(nbr_id, 0.0):
                    max_weight[nbr_id] = new_w
                    next_layer.append((nbr_id, new_w))
        current_layer = next_layer
        if not current_layer:
            break

    return [max_weight.get(cid, 0.0) for cid in candidate_ids]


# ────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    REPO = Path(__file__).resolve().parent.parent
    GRAPH_PATH = REPO / "data/benchmarks/spiqa/test-A/element_graph_v2.json"

    if not GRAPH_PATH.exists():
        print(f"graph not found: {GRAPH_PATH}", file=sys.stderr)
        sys.exit(1)

    graphs = json.loads(GRAPH_PATH.read_text())
    paper_id = "1611.05742v3"
    if paper_id not in graphs:
        paper_id = next(iter(graphs))
    g = graphs[paper_id]
    print(f"=== smoke test on {paper_id} ===")
    print(f"nodes: {len(g['nodes'])}, edges: {len(g['edges'])}")

    # Find a caption node and check relevance to its figure (should be 1.0 via caption_of)
    cap_id = None
    fig_id = None
    for edge in g["edges"]:
        if edge["type"] == "caption_of":
            cap_id = edge["src"]
            fig_id = edge["dst"]
            break

    if cap_id and fig_id:
        all_ids = [n["id"] for n in g["nodes"]]
        rels = graph_relevance(cap_id, all_ids, g, gamma=0.5, max_hops=2)
        cap_to_fig = rels[all_ids.index(fig_id)]
        print(f"\nanchor: {cap_id} (caption)")
        print(f"target: {fig_id} (figure)")
        print(f"  g(caption, figure) = {cap_to_fig:.4f}  (should be ~1.0 via caption_of)")

        # Show top-5 by relevance from this anchor
        ranked = sorted(zip(all_ids, rels), key=lambda x: -x[1])[:8]
        print("\ntop-8 relevances from anchor:")
        for nid, r in ranked:
            ntype = next((n["type"] for n in g["nodes"] if n["id"] == nid), "?")
            print(f"  {r:.4f}  {ntype:18s} {nid}")
