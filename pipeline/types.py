"""Central interface definitions for the multimodal element retrieval pipeline.

This file is the **contract** between modules. Any module that produces or consumes
elements, queries, graphs, or token tensors imports from here.

Design principles:
  - TypedDict for compatibility with existing dict-based loaders (no migration needed).
  - Ragged tensor layout for tokens.pt (memory-efficient, GPU-friendly).
  - Pure-function APIs (no class state) for retrieval/scoring/propagation.

Module organization:
  - pipeline/element_encoder.py — ElementTokenEncoder produces tokens.pt
  - pipeline/set_contrastive.py — set_contrastive_loss + multi-positive sampler
  - pipeline/retrieve.py        — late_interaction_score + graph_propagate
  - eval/runners/               — orchestrators (training, evaluation)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, TypedDict

import torch
from torch import Tensor, LongTensor


# ────────────────────────────────────────────────────────────────────────────
# Element / Query / Graph schemas (TypedDict)
# ────────────────────────────────────────────────────────────────────────────

ElementType = Literal["text", "figure", "table", "caption", "equation", "section_header"]

SectionRole = Literal[
    "front_matter",   # title, authors, abstract
    "intro",          # introduction
    "related_work",   # related work / background
    "method",         # methodology / approach
    "result",         # experiments / results
    "discussion",     # discussion / analysis
    "conclusion",     # conclusion / future work
    "references",     # bibliography
    "appendix",       # supplementary / appendix
    "other",          # unclassified
]

# Schema v2.1: minimal edge set (5 types).
# Rationale:
#   - caption_of:   structural binding of caption to its visual element (cross-modal).
#   - refer_to:     text mention of any target (figure/table/eq/section/appendix). Target
#                   attributes (type, section_role) modulate effective weight.
#   - contains:     section_header ↔ child element (paragraph / figure / etc.).
#                   Restores explicit section structure dropped in initial v2.
#   - section_next: directional link between consecutive section_headers in
#                   reading order. Section-level flow without paragraph noise.
#   - reading_next: paragraph-level sequential link, INTRA-SECTION ONLY (cross-section
#                   transitions are represented by section_next, not reading_next).
#
# Edges deliberately collapsed:
#   - body_to_visual_ref / section_ref / text_to_appendix_ref → all `refer_to`,
#     differentiated via target node attributes.
#   - same_section / sibling: replaced by section_header + contains hierarchy
#     (siblings share a `contains` parent, propagation reaches them in 2 hops).
EdgeType = Literal[
    "caption_of",     # caption ↔ figure/table (structural binding, bidirectional)
    "refer_to",       # text → mentioned target (any element)
    "contains",       # section_header ↔ child element (bidirectional)
    "section_next",   # section_header → next section_header (directional)
    "reading_next",   # consecutive paragraphs WITHIN one section (bidirectional)
]

BASE_EDGE_WEIGHTS: dict[str, float] = {
    "caption_of":   1.0,
    "refer_to":     0.8,
    "contains":     0.5,
    "reading_next": 0.3,
    "section_next": 0.2,
}

# Target-attribute modifiers for refer_to edges only.
# Applied at propagation time, not stored on the edge.
SECTION_ROLE_MODIFIER: dict[str, float] = {
    "appendix":     1.3,  # cross-page evidence boost (M2)
    "method":       1.0,
    "result":       1.0,
    "discussion":   1.0,
    "intro":        1.0,
    "related_work": 1.0,
    "conclusion":   1.0,
    "front_matter": 0.8,
    "references":   0.5,  # bibliography rarely relevant evidence
    "other":        1.0,
}

VISUAL_TARGET_MODIFIER = 1.1  # extra boost for refer_to → figure/table/equation (cross-modal)

# Backward-compatibility alias (deprecated, do not use in new code)
EDGE_TYPE_WEIGHTS = BASE_EDGE_WEIGHTS


class Element(TypedDict, total=False):
    """A single document element.

    Required fields: id, doc_id, type. Others are optional but conventional.
    `text` must be non-None for text-type elements; `image_loader` must be
    callable for visual-type elements.
    """
    id: str                                 # globally unique element ID, e.g. "paperA_p3_e7"
    doc_id: str                             # paper identifier
    type: ElementType
    text: str | None                        # body text (text/caption); None for visual
    image_loader: Callable[[], object] | None  # () → PIL.Image (lazy); None for text
    page: int                               # 0-indexed PDF page
    bbox: list[float] | None                # [x0, y0, x1, y1] in PDF coords
    section_path: list[int] | None          # e.g. [3, 2] for §3.2
    section_role: str | None                # "intro" | "method" | "result" | "discussion" | "appendix"
    label: str | None                       # "Figure 2", "Table 1", etc. (None if not labeled)


class Query(TypedDict):
    """A retrieval query with multi-element ground truth."""
    qid: str
    doc_id: str
    query: str
    answer: str | None
    gt_element_ids: set[str]                # set of element IDs that satisfy this query
    category: str | None                    # dataset-specific tag (e.g. "single-region", "multi-page")


class GraphNode(TypedDict, total=False):
    """A node in the per-document element graph.

    Carries enough element attributes for propagation-time weight modulation.
    `section_path` is informational (not used by graph propagation); `section_role`
    drives target-attribute weight modifiers for refer_to edges.
    """
    id: str                                 # required, = Element.id
    type: ElementType                       # required, for target-modality routing
    page: int                               # required, for cross-page reasoning
    section_path: list[int] | None          # e.g. [3, 2] for §3.2
    section_role: SectionRole | None        # semantic role
    label: str | None                       # "Figure 2", "Table 1" if applicable


class ElementEdge(TypedDict, total=False):
    """A typed edge in the per-document element graph (schema v2).

    Confidence semantics:
        1.0  — structural extraction (layout-parser tree, exact label+proximity)
        0.9  — strong pattern match (regex match, no ambiguous context)
        0.7  — weak match (proximity only, no label; or pattern with mild ambiguity)
        0.5  — ambiguous (citation-adjacent reference, etc.)
        <0.5 — DROPPED (not emitted)

    Effective propagation weight (computed at propagation time):
        w = BASE_EDGE_WEIGHTS[type] * confidence
            * SECTION_ROLE_MODIFIER[target.section_role]   (refer_to only)
            * (VISUAL_TARGET_MODIFIER if target.type ∈ {figure, table, equation} else 1)
                                                            (refer_to only)
    """
    src: str
    dst: str
    type: EdgeType
    confidence: float                       # in [0.5, 1.0]
    bidirectional: bool                     # default False, propagation treats as symmetric
    evidence: str | None                    # debug snippet (optional)


class DocumentGraph(TypedDict, total=False):
    """Per-document element graph (schema v2).

    `schema_version` is required to enable forward-compatible migration.
    """
    doc_id: str                             # required
    schema_version: str                     # required, currently "v2"
    nodes: list[GraphNode]                  # full node attributes, not just IDs
    edges: list[ElementEdge]


GRAPH_SCHEMA_VERSION = "v2.1"


# ────────────────────────────────────────────────────────────────────────────
# tokens.pt format (ragged tensor layout)
# ────────────────────────────────────────────────────────────────────────────
#
# Saved via torch.save(d, path) where d is:
#
#   {
#     "tokens":       Tensor(N_total_tokens, D),  L2-normalized
#     "offsets":      LongTensor(N_elements + 1), tokens[offsets[i]:offsets[i+1]] = element i tokens
#     "element_ids":  list[str],                  element_ids[i] for slot i
#     "encoder_name": str,                        e.g. "siglipv2-base-loraR16"
#     "dim":          int,                        D
#     "version":      str,                        schema version, currently "v1"
#   }
#
# Loading: see token_index.load_tokens / save_tokens
# ────────────────────────────────────────────────────────────────────────────

TOKEN_INDEX_VERSION = "v1"


class TokenIndex(TypedDict):
    tokens: Tensor                          # (N_total_tokens, D)
    offsets: LongTensor                     # (N_elements + 1,)
    element_ids: list[str]
    encoder_name: str
    dim: int
    version: str


def save_tokens(path: str | Path, idx: TokenIndex) -> None:
    """Write a token index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(idx, path)


def load_tokens(path: str | Path) -> TokenIndex:
    """Read a token index from disk."""
    return torch.load(path, weights_only=False)


def get_element_tokens(idx: TokenIndex, element_id: str) -> Tensor:
    """Retrieve token tensor for a specific element by id.

    Returns Tensor of shape (K_e, D).
    """
    i = idx["element_ids"].index(element_id)
    s = idx["offsets"][i].item()
    e = idx["offsets"][i + 1].item()
    return idx["tokens"][s:e]


# ────────────────────────────────────────────────────────────────────────────
# Scoring API (late interaction)
# ────────────────────────────────────────────────────────────────────────────

def late_interaction_score(
    q_tokens: Tensor,                       # (K_q, D)
    e_tokens: Tensor,                       # (K_e, D)
) -> float:
    """ColBERT MaxSim score: sum_i max_j <q_i, e_j>.

    Both tensors assumed L2-normalized along dim=-1.
    """
    sim = q_tokens @ e_tokens.T             # (K_q, K_e)
    return sim.max(dim=-1).values.sum().item()


def late_interaction_batch(
    q_tokens: Tensor,                       # (K_q, D)
    idx: TokenIndex,
    candidate_indices: list[int] | None = None,  # element slot indices to score; None = all
) -> Tensor:
    """Compute MaxSim score for query against all (or candidate) elements.

    Returns Tensor of shape (N_candidates,) or (N_elements,) — scores in element-slot order.
    """
    tokens = idx["tokens"]
    offsets = idx["offsets"]
    K_q = q_tokens.shape[0]
    N = len(idx["element_ids"]) if candidate_indices is None else len(candidate_indices)

    # Full similarity matrix: (K_q, N_total_tokens) — may be large; OK for small corpora
    sim_all = q_tokens @ tokens.T

    scores = torch.zeros(N, dtype=q_tokens.dtype, device=q_tokens.device)
    cand = range(len(idx["element_ids"])) if candidate_indices is None else candidate_indices
    for out_i, e_i in enumerate(cand):
        s = offsets[e_i].item()
        e = offsets[e_i + 1].item()
        # MaxSim: for each query token, max over this element's tokens, then sum
        scores[out_i] = sim_all[:, s:e].max(dim=-1).values.sum()
    return scores


# ────────────────────────────────────────────────────────────────────────────
# Graph propagation API
# ────────────────────────────────────────────────────────────────────────────

def _build_transition_matrix(
    element_ids: list[str],
    doc_graph: DocumentGraph,
    weights_mode: str,
    edge_weight_overrides: dict[str, float] | None = None,
    dtype=torch.float32,
    device=None,
) -> Tensor:
    """Build the row-normalized P matrix: P[i,j] = transition i → j.

    weights_mode (sub-ablation axis from REPORT_KR §6.5):
        "uniform"      — every edge has weight 1.0; structure only, no semantic boost
        "base"         — BASE_EDGE_WEIGHTS only (caption_of=1.0, refer_to=0.8, ...)
        "base+role"    — base × SECTION_ROLE_MODIFIER (appendix/references boost)
        "base+visual"  — base × VISUAL_TARGET_MODIFIER (refer_to → figure/table boost)
        "full"         — base × role × visual modifiers (default, REPORT_KR §4.9)

    For bidirectional edges (caption_of, contains), forward/backward weights are
    computed independently then max-ed (matches §4.9 design choice).
    """
    base = dict(BASE_EDGE_WEIGHTS)
    if edge_weight_overrides:
        base.update(edge_weight_overrides)

    use_role   = weights_mode in ("base+role", "full")
    use_visual = weights_mode in ("base+visual", "full")
    uniform    = weights_mode == "uniform"

    N = len(element_ids)
    id_to_idx = {eid: i for i, eid in enumerate(element_ids)}

    node_attrs: dict[str, GraphNode] = {}
    raw_nodes = doc_graph.get("nodes", [])
    if raw_nodes and isinstance(raw_nodes[0], dict):
        for n in raw_nodes:
            node_attrs[n["id"]] = n

    def target_modifier(target_id: str, edge_type: str) -> float:
        # Role/visual modifiers only apply to refer_to (per REPORT_KR §4.9 design).
        if edge_type != "refer_to" or (not use_role and not use_visual):
            return 1.0
        n = node_attrs.get(target_id)
        if not n:
            return 1.0
        mod = 1.0
        if use_role:
            mod *= SECTION_ROLE_MODIFIER.get(n.get("section_role") or "other", 1.0)
        if use_visual and n.get("type") in ("figure", "table", "equation"):
            mod *= VISUAL_TARGET_MODIFIER
        return mod

    W = torch.zeros(N, N, dtype=dtype, device=device)
    for edge in doc_graph.get("edges", []):
        i = id_to_idx.get(edge["src"])
        j = id_to_idx.get(edge["dst"])
        if i is None or j is None:
            continue
        if uniform:
            w_fwd = 1.0
            w_bwd = 1.0
        else:
            edge_base = base.get(edge["type"], 0.0) * edge.get("confidence", 1.0)
            w_fwd = edge_base * target_modifier(edge["dst"], edge["type"])
            w_bwd = edge_base * target_modifier(edge["src"], edge["type"])
        if w_fwd > W[i, j].item():
            W[i, j] = w_fwd
        if edge.get("bidirectional", False):
            w_back = max(w_fwd, w_bwd) if not uniform else 1.0
            if w_back > W[j, i].item():
                W[j, i] = w_back

    deg_out = W.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return W / deg_out                       # row-normalized P[i,j] = transition i→j


def graph_propagate(
    initial_scores: Tensor,                 # (N,) scores for top-N elements
    element_ids: list[str],                 # element_ids[i] corresponds to initial_scores[i]
    doc_graph: DocumentGraph,
    *,
    method: str = "diffusion",              # "none" | "diffusion" | "ppr"
    weights: str = "full",                  # "uniform" | "base" | "base+role" | "base+visual" | "full"
    alpha: float = 0.3,
    T: int = 2,                             # diffusion only
    max_iter: int = 30,                     # ppr only
    tol: float = 1e-4,                      # ppr early-stop tolerance
    edge_weight_overrides: dict[str, float] | None = None,
    # legacy positional args (back-compat for old callers)
    legacy_alpha: float | None = None,
    legacy_T: int | None = None,
) -> Tensor:                                # (N,) refined scores
    """Graph-aware score propagation. Three regimes:

    method="none":
        Identity — return scores unchanged.

    method="diffusion" (original):
        s^{(t+1)} = (1 - α) * s^{(t)} + α * P^T @ s^{(t)}
        Iterate exactly T steps. Local smoothing, no query teleport.

    method="ppr" (Personalized PageRank):
        s^{(t+1)} = α * q + (1 - α) * P^T @ s^{(t)}
        where q = initial_scores is the teleport (personalization) vector.
        Iterate up to max_iter, early-stop on tol convergence. α here is the
        teleport probability (small α = many random walks; standard PPR uses
        α≈0.15, i.e. 85% walk / 15% return-to-query).

    weights axis (sub-ablation over edge-weight components):
        "uniform" / "base" / "base+role" / "base+visual" / "full"
        See _build_transition_matrix.

    Elements in element_ids but not in doc_graph.nodes retain initial score
    (isolated rows of P are zero → no propagation in/out).
    """
    if method == "none":
        return initial_scores

    if legacy_alpha is not None:
        alpha = legacy_alpha
    if legacy_T is not None:
        T = legacy_T

    P = _build_transition_matrix(
        element_ids, doc_graph, weights, edge_weight_overrides,
        dtype=initial_scores.dtype, device=initial_scores.device,
    )
    PT = P.T

    s = initial_scores.clone()
    if method == "diffusion":
        for _ in range(T):
            s = (1.0 - alpha) * s + alpha * (PT @ s)
        return s

    if method == "ppr":
        q = initial_scores.clone()
        for _ in range(max_iter):
            s_new = alpha * q + (1.0 - alpha) * (PT @ s)
            if (s_new - s).abs().max() < tol:
                return s_new
            s = s_new
        return s

    raise ValueError(f"unknown propagation method: {method!r}")


# ────────────────────────────────────────────────────────────────────────────
# Set-contrastive loss API
# ────────────────────────────────────────────────────────────────────────────

def set_contrastive_loss(
    q_embs: list[Tensor],                   # B query token tensors, each (K_q_b, D)
    pos_sets: list[list[Tensor]],           # B positive sets; pos_sets[b] = list of (K_e, D)
    neg_embs: list[Tensor],                 # B negative pools; neg_embs[b] = (N_neg_b, K_e, D) flattened or list
    score_fn: Callable[[Tensor, Tensor], float] = late_interaction_score,
    tau: float = 1.0 / 0.07,
    lambda_cov: float = 0.5,
    K_cov: int = 10,
) -> Tensor:                                # scalar loss
    """Multi-positive InfoNCE with coverage-at-K term.

    For each (q, S_+) pair:
      L_NCE = -1/|S+| Σ_p log[ exp(score(q,p)/τ) / Σ_x exp(score(q,x)/τ) ]
      L_cov = -1/|S+| Σ_p log σ( score(q,p) - score(q, x_K) )
      L = L_NCE + λ_cov L_cov

    The coverage term rewards every positive crossing rank K boundary, not just
    being closer than negatives on average.

    NOTE: This is the signature. Implementation lives in pipeline/set_contrastive.py.
    This file only defines the contract.
    """
    raise NotImplementedError("Implement in pipeline/set_contrastive.py")


# ────────────────────────────────────────────────────────────────────────────
# Top-level retrieve API (composition of components)
# ────────────────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query_tokens: Tensor,                   # (K_q, D)
    query_doc_id: str | None,               # for per-doc restriction; None = cross-doc
    idx: TokenIndex,
    doc_graphs: dict[str, DocumentGraph],
    elements: list[Element],                # parallel to idx.element_ids (for doc_id lookup)
    k: int = 10,
    top_n_for_propagation: int = 50,
    use_late_interaction: bool = True,
    use_graph_propagation: bool = True,
    restrict_to_paper: bool = True,
    alpha: float = 0.3,
    T: int = 2,
    edge_weight_overrides: dict[str, float] | None = None,
) -> list[tuple[str, float]]:               # top-k (element_id, score) sorted desc
    """End-to-end retrieval: late interaction → graph propagation → top-k.

    NOTE: Signature only. Implementation in pipeline/retrieve.py.
    """
    raise NotImplementedError("Implement in pipeline/retrieve.py")


# ────────────────────────────────────────────────────────────────────────────
# Element encoder API
# ────────────────────────────────────────────────────────────────────────────

class ElementTokenEncoder:
    """Abstract base for element token encoders.

    Concrete subclasses live in pipeline/element_encoder.py.
    They produce a TokenIndex from a list of Elements.
    """
    encoder_name: str
    dim: int

    def encode_elements(self, elements: list[Element], batch_size: int = 16) -> TokenIndex:
        raise NotImplementedError

    def encode_query(self, query_text: str) -> Tensor:
        """Returns (K_q, D) L2-normalized query token tensor."""
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────────────────
# Convenience: load doc graphs from JSON
# ────────────────────────────────────────────────────────────────────────────

def load_doc_graphs(path: str | Path) -> dict[str, DocumentGraph]:
    """Load element_graph.json (per-corpus dict of doc_id → DocumentGraph).

    Handles both v1 (legacy) and v2 schema:
      v1: { "doc_id": {"nodes": [str], "edges": [{src, dst, type, weight, confidence}]} }
      v2: { "doc_id": {"schema_version": "v2", "nodes": [GraphNode], "edges": [ElementEdge]} }

    v1 graphs are loaded as-is for backward compat. v1 edge types (references,
    same_section, next, prev, appendix_link, contains) are NOT remapped here —
    use a migration script for that.
    """
    import json
    raw = json.loads(Path(path).read_text())
    out = {}
    for doc_id, g in raw.items():
        nodes = g.get("nodes", [])
        edges = []
        for e in g.get("edges", []):
            ed = ElementEdge(
                src=e["src"],
                dst=e["dst"],
                type=e["type"],
                confidence=e.get("confidence", 1.0),
                bidirectional=e.get("bidirectional", False),
            )
            if "evidence" in e:
                ed["evidence"] = e["evidence"]
            edges.append(ed)
        out[doc_id] = DocumentGraph(
            doc_id=doc_id,
            schema_version=g.get("schema_version", "v1"),
            nodes=nodes,
            edges=edges,
        )
    return out
