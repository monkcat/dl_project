"""Graph Position Embedding — node-level structural prior for retrieval.

GPE composes multiple facets of element-graph node attributes into a single
D-dim vector. Each facet has a sigmoid-bounded learnable gate (init small) so
the model can learn how much to rely on each.

Facets (cross-document-safe; absolute section_path explicitly excluded):
  1. element type      — learned nn.Embedding(6 types, D)
  2. section role      — learned nn.Embedding(10 roles, D)
  3. subsection depth  — sinusoidal of int depth (depth-of-nesting)
  4. intra-section pos — sinusoidal of float ratio ∈ [0, 1]

Usage (training):
    gpe = GraphPositionEmbedding(d_model=768)
    features = extract_node_features(doc_graph)
    # ... build batch of node ids → look up features → tensorize → forward
    pe_vec = gpe(type_ids, role_ids, depths, intra_pos)   # (B, D)
    z = F.normalize(LayerNorm(content_emb + beta * pe_vec), dim=-1)
"""
from __future__ import annotations

import math
from typing import Any, get_args

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import DocumentGraph, ElementType, SectionRole

# Stable enum-to-id mapping. Order is determined by Literal definition order
# in pipeline/types.py.
ELEMENT_TYPES: list[str] = list(get_args(ElementType))  # 6 entries
SECTION_ROLES: list[str] = list(get_args(SectionRole))  # 10 entries

TYPE_TO_ID: dict[str, int] = {t: i for i, t in enumerate(ELEMENT_TYPES)}
ROLE_TO_ID: dict[str, int] = {r: i for i, r in enumerate(SECTION_ROLES)}


def sinusoidal_pe(values: torch.Tensor, d_model: int, max_period: float = 1000.0) -> torch.Tensor:
    """Standard sinusoidal positional encoding adapted for continuous values.

    Args:
        values: (B,) float tensor — any continuous value (depth, ratio, etc.).
        d_model: output dimensionality.
        max_period: max wavelength (controls frequency spectrum).

    Returns:
        (B, d_model) tensor with sin/cos interleaved.
    """
    half = max(d_model // 2, 1)
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=values.device, dtype=values.dtype) / half
    )
    args = values.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    pe = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, 2*half)
    if pe.shape[-1] < d_model:
        pe = F.pad(pe, (0, d_model - pe.shape[-1]))
    elif pe.shape[-1] > d_model:
        pe = pe[..., :d_model]
    return pe


class GraphPositionEmbedding(nn.Module):
    """Gated additive composition of graph-derived structural PE facets.

    Forward signature:
        forward(type_ids, role_ids, depth, intra_pos) -> (B, d_model)

    Each facet has a `raw_alpha_*` parameter; the effective gate is
    `sigmoid(raw_alpha)`, bounded in (0, 1). Initialized at raw = -3.0
    → sigmoid(-3) ≈ 0.047 (gentle PE contribution at start).
    """

    def __init__(self, d_model: int, alpha_init_raw: float = -3.0):
        super().__init__()
        self.d_model = d_model

        # Learned embeddings
        self.type_emb = nn.Embedding(len(ELEMENT_TYPES), d_model)
        self.role_emb = nn.Embedding(len(SECTION_ROLES), d_model)
        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.role_emb.weight, mean=0.0, std=0.02)

        # Per-facet learnable gates (raw values, sigmoid-bounded)
        self.raw_alpha_type = nn.Parameter(torch.tensor(alpha_init_raw))
        self.raw_alpha_role = nn.Parameter(torch.tensor(alpha_init_raw))
        self.raw_alpha_depth = nn.Parameter(torch.tensor(alpha_init_raw))
        self.raw_alpha_pos = nn.Parameter(torch.tensor(alpha_init_raw))

    @property
    def alphas(self) -> dict[str, float]:
        """Current sigmoid-bounded gate values, for logging."""
        return {
            "type": torch.sigmoid(self.raw_alpha_type).item(),
            "role": torch.sigmoid(self.raw_alpha_role).item(),
            "depth": torch.sigmoid(self.raw_alpha_depth).item(),
            "pos": torch.sigmoid(self.raw_alpha_pos).item(),
        }

    def forward(
        self,
        type_ids: torch.LongTensor,    # (B,) ∈ [0, len(ELEMENT_TYPES))
        role_ids: torch.LongTensor,    # (B,) ∈ [0, len(SECTION_ROLES))
        depth: torch.Tensor,           # (B,) float, e.g. 1.0, 2.0, 3.0
        intra_pos: torch.Tensor,       # (B,) float ∈ [0, 1]
    ) -> torch.Tensor:                 # (B, d_model)
        alpha_type = torch.sigmoid(self.raw_alpha_type)
        alpha_role = torch.sigmoid(self.raw_alpha_role)
        alpha_depth = torch.sigmoid(self.raw_alpha_depth)
        alpha_pos = torch.sigmoid(self.raw_alpha_pos)

        pe = (
            alpha_type * self.type_emb(type_ids)
            + alpha_role * self.role_emb(role_ids)
            + alpha_depth * sinusoidal_pe(depth, self.d_model)
            # Scale intra_pos to a similar wavelength as depth so its sin/cos varies meaningfully.
            + alpha_pos * sinusoidal_pe(intra_pos * 100.0, self.d_model)
        )
        return pe


# ────────────────────────────────────────────────────────────────────────────
# Feature extraction from a DocumentGraph
# ────────────────────────────────────────────────────────────────────────────

def _para_index_in_id(node_id: str) -> int | None:
    """Extract paragraph index from a node id like 'paperX__p007'. None otherwise."""
    if "__p" not in node_id:
        return None
    try:
        return int(node_id.split("__p")[-1])
    except ValueError:
        return None


def extract_node_features(graph: DocumentGraph) -> dict[str, dict[str, Any]]:
    """Precompute (type_id, role_id, depth, intra_pos) for every node in `graph`.

    `intra_pos` is derived from `contains` edges + paragraph index within section:
      - For each section_header, list its `contains` children (paragraphs and figures).
      - Sort paragraphs by their parsed index; intra_pos = i / total.
      - Non-paragraph children get intra_pos = 0.0 by default.

    Returns:
        {node_id: {"type_id": int, "role_id": int, "depth": float, "intra_pos": float}}
    """
    nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}

    # Build section → list of contained children (sorted by paragraph index)
    section_children: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        if edge["type"] != "contains":
            continue
        # contains edges in v2.1 are bidirectional with src=section_header by convention.
        # Use whichever endpoint IS a section_header.
        src_node = nodes_by_id.get(edge["src"])
        dst_node = nodes_by_id.get(edge["dst"])
        if src_node is None or dst_node is None:
            continue
        if src_node.get("type") == "section_header":
            section_id, child_id = edge["src"], edge["dst"]
        elif dst_node.get("type") == "section_header":
            section_id, child_id = edge["dst"], edge["src"]
        else:
            continue
        section_children.setdefault(section_id, []).append(child_id)

    # intra_pos = (index in section, sorted by paragraph idx) / total
    intra_pos_lookup: dict[str, float] = {}
    # role inheritance: figure/caption nodes don't carry section_role themselves;
    # they inherit from their enclosing section_header.
    inherited_role: dict[str, str] = {}
    for sec_id, children in section_children.items():
        sec_node = nodes_by_id.get(sec_id, {})
        sec_role = sec_node.get("section_role") or "other"
        # Sort by paragraph index (paragraphs first ordered by idx); non-paragraphs follow.
        children_sorted = sorted(
            children,
            key=lambda cid: (_para_index_in_id(cid) is None, _para_index_in_id(cid) or 0),
        )
        n = max(1, len(children_sorted))
        for i, cid in enumerate(children_sorted):
            intra_pos_lookup[cid] = i / n
            inherited_role.setdefault(cid, sec_role)

    # Build feature dict
    features: dict[str, dict[str, Any]] = {}
    for node_id, node in nodes_by_id.items():
        type_str = node.get("type", "text")
        # Use node's own section_role if present, else inherit from parent section.
        role_str = node.get("section_role") or inherited_role.get(node_id) or "other"
        # Depth: for section_headers, length of section_path; for others, default 1
        sp = node.get("section_path")
        depth = float(len(sp)) if sp else 1.0
        intra_pos = intra_pos_lookup.get(node_id, 0.0)
        features[node_id] = {
            "type_id": TYPE_TO_ID.get(type_str, TYPE_TO_ID["text"]),
            "role_id": ROLE_TO_ID.get(role_str, ROLE_TO_ID["other"]),
            "depth": depth,
            "intra_pos": intra_pos,
        }
    return features


def features_to_tensors(
    node_ids: list[str],
    features: dict[str, dict[str, Any]],
    device: str | torch.device = "cpu",
) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
    """Convert a batch of node_ids + per-node feature dict into 4 stacked tensors
    matching `GraphPositionEmbedding.forward` arg order.

    Returns:
        (type_ids, role_ids, depth, intra_pos)
    """
    type_ids = []
    role_ids = []
    depth = []
    intra_pos = []
    for nid in node_ids:
        f = features.get(nid)
        if f is None:
            # Default: text, other, depth=1, pos=0
            type_ids.append(TYPE_TO_ID["text"])
            role_ids.append(ROLE_TO_ID["other"])
            depth.append(1.0)
            intra_pos.append(0.0)
        else:
            type_ids.append(f["type_id"])
            role_ids.append(f["role_id"])
            depth.append(f["depth"])
            intra_pos.append(f["intra_pos"])
    return (
        torch.tensor(type_ids, dtype=torch.long, device=device),
        torch.tensor(role_ids, dtype=torch.long, device=device),
        torch.tensor(depth, dtype=torch.float32, device=device),
        torch.tensor(intra_pos, dtype=torch.float32, device=device),
    )


# ────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from pathlib import Path

    REPO = Path(__file__).resolve().parent.parent
    GRAPH_PATH = REPO / "data/benchmarks/spiqa/test-A/element_graph_v2.json"
    graphs = json.loads(GRAPH_PATH.read_text())
    paper_id = "1611.05742v3"
    g = graphs[paper_id]

    print(f"=== GPE smoke test on {paper_id} ===")
    features = extract_node_features(g)
    print(f"extracted {len(features)} node feature sets")

    # Show a few examples
    print("\nsample features:")
    for nid in list(features)[:5]:
        ntype = next((n["type"] for n in g["nodes"] if n["id"] == nid), "?")
        f = features[nid]
        print(f"  {nid:55s} type={ntype:18s} type_id={f['type_id']} role_id={f['role_id']} depth={f['depth']:.1f} intra_pos={f['intra_pos']:.2f}")

    # Forward through GPE
    gpe = GraphPositionEmbedding(d_model=128)
    node_ids = list(features.keys())[:8]
    tensors = features_to_tensors(node_ids, features)
    pe_out = gpe(*tensors)
    print(f"\nGPE forward output shape: {pe_out.shape}")
    print(f"per-facet alphas (init): {gpe.alphas}")
    print(f"PE norm per element: {pe_out.norm(dim=-1).tolist()}")
