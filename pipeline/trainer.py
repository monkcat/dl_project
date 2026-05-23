"""Proper trainer for ElementTokenEncoder (GRCL + GPE + L_cov + L_cons).

Training data:  SPIQA train (random sample of N papers, default 1000).
                Anchor types mixed: caption_of, refer_to, NL QA.
Held-out eval:  SPIQA test-A (118 papers, never seen at training).
                Metrics: caption-figure cosine, Recall@k on QA reference figures.

Run:
    python -m pipeline.trainer \\
        --train_sample 1000 --steps 2000 --eval_every 200 \\
        --batch 4 --pool 12 --lr 3e-5 \\
        --out eval/results/trainer/run_2k.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.element_encoder import ElementTokenEncoder
from pipeline.graph_pe import extract_node_features
from pipeline.graph_relevance import build_neighbor_index, graph_relevance
from pipeline.losses import consistency_loss, coverage_loss, grcl_loss, total_loss


# ────────────────────────────────────────────────────────────────────────────
# Data paths
# ────────────────────────────────────────────────────────────────────────────

SPIQA_ROOT = REPO / "data/benchmarks/spiqa"

TRAIN_GRAPH = SPIQA_ROOT / "train_val/element_graph_v2.json"
TRAIN_ELEMENTS = SPIQA_ROOT / "train_val/elements_v2.jsonl"
TRAIN_QA = SPIQA_ROOT / "train_val/SPIQA_train.json"
TRAIN_IMG_ROOT = SPIQA_ROOT / "train_val/SPIQA_train_val_Images"

TESTA_GRAPH = SPIQA_ROOT / "test-A/element_graph_v2.json"
TESTA_ELEMENTS = SPIQA_ROOT / "test-A/elements_v2.jsonl"
TESTA_QA = SPIQA_ROOT / "test-A/SPIQA_testA.json"
TESTA_IMG_ROOT = SPIQA_ROOT / "test-A/SPIQA_testA_Images_224px"


# ────────────────────────────────────────────────────────────────────────────
# Negative controls — section_role rewriting at load time
# ────────────────────────────────────────────────────────────────────────────

# All valid section_role labels (must match SectionRole Literal in types.py)
_SECTION_ROLES = ("intro", "method", "result", "discussion", "appendix",
                  "references", "abstract", "front_matter", "related_work", "other")


def _apply_section_role_mode(graphs: dict, mode: str, seed: int) -> None:
    """In-place mutation of `graphs` to rewrite `section_role` per neg-control mode.

    - 'shuffled': permute the multiset of section_roles within each doc (preserves
      per-doc role marginal but breaks structure↔role correspondence)
    - 'random':   uniform random role per node (also breaks marginal)
    """
    import random as _r
    rng = _r.Random(seed)
    for doc_id, g in graphs.items():
        nodes = g.get("nodes", [])
        if mode == "shuffled":
            roles = [n.get("section_role") or "other" for n in nodes]
            rng.shuffle(roles)
            for n, r in zip(nodes, roles):
                n["section_role"] = r
        elif mode == "random":
            for n in nodes:
                n["section_role"] = rng.choice(_SECTION_ROLES)
        else:
            raise ValueError(f"unknown section_role_mode: {mode!r}")


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

class SpiqaSplitData:
    """Per-split bundle: graphs + elements + (optional) QA + image root.

    Loaded lazily on first access via `subset_paper_ids` so we don't pay the
    750 MB JSON parse cost for SPIQA train when only 1000 papers are needed.
    """

    def __init__(
        self,
        graph_path: Path,
        elements_path: Path,
        qa_path: Path | None,
        image_root: Path,
        subset_paper_ids: set[str] | None = None,
        section_role_mode: str = "normal",
        role_shuffle_seed: int = 42,
    ):
        self.graph_path = graph_path
        self.elements_path = elements_path
        self.qa_path = qa_path
        self.image_root = image_root
        self.subset = subset_paper_ids

        print(f"  loading {graph_path.name}")
        all_graphs = json.loads(graph_path.read_text())
        if subset_paper_ids is not None:
            self.graphs = {k: v for k, v in all_graphs.items() if k in subset_paper_ids}
        else:
            self.graphs = all_graphs
        print(f"    {len(self.graphs):,} papers kept")

        # Negative controls (m, n): rewrite section_role at load time.
        if section_role_mode != "normal":
            _apply_section_role_mode(self.graphs, section_role_mode, role_shuffle_seed)
            print(f"    section_role mode = {section_role_mode!r} applied")

        # Precompute per-doc neighbor indices + features
        self.neighbor_index = {pid: build_neighbor_index(g) for pid, g in self.graphs.items()}
        self.features = {pid: extract_node_features(g) for pid, g in self.graphs.items()}
        # Per-doc lookup of nodes
        self.nodes_by_doc = {
            pid: {n["id"]: n for n in g["nodes"]} for pid, g in self.graphs.items()
        }
        self.doc_node_ids = {pid: list(self.nodes_by_doc[pid].keys()) for pid in self.graphs}

        print(f"  loading {elements_path.name}")
        self.elements: dict[str, dict] = {}
        n_bad = 0
        with open(elements_path) as f:
            for line in f:
                if not line.strip() or "\x00" in line:
                    n_bad += 1
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                if subset_paper_ids is None or r.get("doc_id") in subset_paper_ids:
                    self.elements[r["id"]] = r
        if n_bad:
            print(f"    [WARN] {n_bad:,} malformed lines skipped")
        print(f"    {len(self.elements):,} element records kept")

        # QA pairs (NL question → reference figure)
        self.qa: list[tuple[str, dict]] = []  # (doc_id, qa_record)
        if qa_path is not None and qa_path.exists():
            print(f"  loading {qa_path.name}")
            raw_qa = json.loads(qa_path.read_text())
            for doc_id, paper in raw_qa.items():
                if subset_paper_ids is not None and doc_id not in subset_paper_ids:
                    continue
                for qa in paper.get("qa", []):
                    if isinstance(qa.get("reference"), str):
                        self.qa.append((doc_id, qa))
            print(f"    {len(self.qa):,} QA pairs kept")

        # Available caption_of / refer_to anchors (only those with image present)
        self.caption_of_anchors: list[tuple[str, str, str]] = []
        self.refer_to_anchors: list[tuple[str, str, str]] = []
        missing_img = 0
        for pid, g in self.graphs.items():
            for e in g.get("edges", []):
                if e["type"] == "caption_of":
                    cap_id, fig_id = e["src"], e["dst"]
                    if (image_root / pid / fig_id).exists():
                        self.caption_of_anchors.append((pid, cap_id, fig_id))
                    else:
                        missing_img += 1
                elif e["type"] == "refer_to":
                    p_id, fig_id = e["src"], e["dst"]
                    if (image_root / pid / fig_id).exists():
                        self.refer_to_anchors.append((pid, p_id, fig_id))
                    else:
                        missing_img += 1
        print(f"    caption_of: {len(self.caption_of_anchors):,}  "
              f"refer_to: {len(self.refer_to_anchors):,}  "
              f"(skipped {missing_img:,} due to missing image)")

        self._image_cache: dict[str, Image.Image] = {}

    def get_image(self, doc_id: str, fname: str) -> Image.Image | None:
        key = f"{doc_id}/{fname}"
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached
        path = self.image_root / doc_id / fname
        if not path.exists():
            return None
        img = Image.open(path).convert("RGB")
        # LRU-lite: bound the cache
        if len(self._image_cache) > 5000:
            self._image_cache.pop(next(iter(self._image_cache)))
        self._image_cache[key] = img
        return img


class TrainDataset:
    """Wraps SpiqaSplitData for SPIQA train and provides sample_anchor()."""

    def __init__(
        self,
        data: SpiqaSplitData,
        seed: int = 0,
        gamma: float = 0.5,
        edge_types_only: set[str] | None = None,
    ):
        self.data = data
        self.rng = random.Random(seed)
        self.gamma = gamma
        self.edge_types_only = edge_types_only

    def _sample_caption_of(self) -> dict | None:
        if not self.data.caption_of_anchors:
            return None
        doc_id, anchor_id, target_id = self.rng.choice(self.data.caption_of_anchors)
        return self._build_pool(doc_id, anchor_id, target_id, "caption_of")

    def _sample_refer_to(self) -> dict | None:
        if not self.data.refer_to_anchors:
            return None
        doc_id, anchor_id, target_id = self.rng.choice(self.data.refer_to_anchors)
        return self._build_pool(doc_id, anchor_id, target_id, "refer_to")

    def _sample_nl_qa(self) -> dict | None:
        if not self.data.qa:
            return None
        doc_id, qa = self.rng.choice(self.data.qa)
        target_id = qa["reference"]
        # Use the reference element as graph_relevance anchor (since NL question is not in graph)
        if target_id not in self.data.nodes_by_doc.get(doc_id, {}):
            return None
        if not (self.data.image_root / doc_id / target_id).exists():
            return None
        inst = self._build_pool(doc_id, target_id, target_id, "nl_qa")
        if inst is None:
            return None
        # Override anchor: NL question text, no GPE on query side
        inst["anchor_text"] = qa["question"]
        inst["anchor_is_text"] = True
        inst["anchor_no_gpe"] = True
        return inst

    def _build_pool(self, doc_id: str, anchor_id: str, target_id: str, kind: str,
                    pool_size: int = 12) -> dict | None:
        nb = self.data.neighbor_index[doc_id]
        nodes_by_id = self.data.nodes_by_doc[doc_id]

        # 1-hop positives (excluding section_headers as candidates)
        positives_1hop = [
            nbr_id for nbr_id, _ in nb.get(anchor_id, [])
            if nbr_id != anchor_id
            and nodes_by_id.get(nbr_id, {}).get("type") != "section_header"
        ]
        if target_id not in positives_1hop and nodes_by_id.get(target_id, {}).get("type") != "section_header":
            positives_1hop.append(target_id)

        # Same-doc negatives
        excluded = set([anchor_id, *positives_1hop])
        same_doc_negs = [
            n for n in self.data.doc_node_ids[doc_id]
            if n not in excluded
            and nodes_by_id.get(n, {}).get("type") != "section_header"
        ]
        self.rng.shuffle(same_doc_negs)
        # Cross-doc negatives
        cross_negs: list[str] = []
        other_doc_ids = [d for d in self.data.graphs if d != doc_id]
        for _ in range(max(4, pool_size // 4)):
            if not other_doc_ids:
                break
            other = self.rng.choice(other_doc_ids)
            other_pool = [
                n for n in self.data.doc_node_ids[other]
                if self.data.nodes_by_doc[other].get(n, {}).get("type") != "section_header"
            ]
            if other_pool:
                cross_negs.append(self.rng.choice(other_pool))

        positives = positives_1hop[:4]
        n_neg = pool_size - len(positives)
        n_same = max(0, n_neg - len(cross_negs))
        negatives = same_doc_negs[:n_same] + cross_negs[:n_neg - n_same]
        candidates = (positives + negatives)[:pool_size]

        relevances = graph_relevance(
            anchor_id, candidates, self.data.graphs[doc_id],
            gamma=self.gamma, max_hops=2, neighbor_index=nb,
            edge_types_only=self.edge_types_only,
        )
        return {
            "doc_id": doc_id,
            "anchor_id": anchor_id,
            "anchor_kind": kind,
            "anchor_text": None,
            "anchor_is_text": False,
            "anchor_no_gpe": False,
            "target_id": target_id,
            "candidates": candidates,
            "relevances": relevances,
        }

    def sample(self, kind: str = "mixed", pool_size: int = 12) -> dict | None:
        if kind == "mixed":
            r = self.rng.random()
            if r < 0.33:
                kind = "caption_of"
            elif r < 0.67:
                kind = "refer_to"
            else:
                kind = "nl_qa"

        if kind == "caption_of":
            return self._sample_caption_of()
        if kind == "refer_to":
            return self._sample_refer_to()
        if kind == "nl_qa":
            return self._sample_nl_qa()
        raise ValueError(f"unknown anchor kind: {kind}")


# ────────────────────────────────────────────────────────────────────────────
# Encoding + scoring
# ────────────────────────────────────────────────────────────────────────────

def late_interaction_score(q_tok: torch.Tensor, q_mask: torch.Tensor,
                            e_tok: torch.Tensor, e_mask: torch.Tensor) -> torch.Tensor:
    sim = q_tok @ e_tok.T
    sim = sim.masked_fill(~e_mask.unsqueeze(0).bool(), -1e4)
    return (sim.max(dim=-1).values * q_mask.float()).sum()


def encode_element(
    encoder: ElementTokenEncoder,
    element_id: str,
    data: SpiqaSplitData,
    use_gpe: bool = True,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    rec = data.elements.get(element_id, {})
    ntype = rec.get("type")
    doc_id = rec.get("doc_id")

    gpe_feats = None
    if use_gpe and doc_id in data.features:
        f = data.features[doc_id].get(element_id)
        if f is not None:
            gpe_feats = encoder._stack_gpe_features([f])

    if ntype in ("text", "caption", "section_header"):
        text = rec.get("text") or "(empty)"
        tok = encoder.tokenize_text([text])
        z, m = encoder.forward_text(tok["input_ids"].to(device), tok["attention_mask"].to(device), gpe_feats)
        return z[0], m[0]
    elif ntype in ("figure", "table", "equation"):
        img = data.get_image(doc_id, element_id)
        if img is None:
            return (torch.zeros(196, encoder.proj_dim, device=device),
                    torch.ones(196, dtype=torch.long, device=device))
        px = encoder.preprocess_images([img]).to(device)
        z, m = encoder.forward_vision(px, gpe_feats)
        return z[0], m[0]
    return (torch.zeros(1, encoder.proj_dim, device=device),
            torch.ones(1, dtype=torch.long, device=device))


def encode_text_only(
    encoder: ElementTokenEncoder,
    text: str,
    use_gpe: bool = False,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode an NL query (no GPE always — query has no graph attribute)."""
    tok = encoder.tokenize_text([text])
    z, m = encoder.forward_text(tok["input_ids"].to(device), tok["attention_mask"].to(device), None)
    return z[0], m[0]


def batch_encode_elements(
    encoder: ElementTokenEncoder,
    element_ids: list[str],
    data: SpiqaSplitData,
    use_gpe: bool = True,
    device: str = "cuda",
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Encode elements in two grouped forward passes (one text batch, one vision batch).

    Deduplicates IDs so each unique element is encoded once. Returns a dict mapping
    element_id -> (tokens, mask) with gradients intact.
    """
    unique_ids: list[str] = list(dict.fromkeys(element_ids))

    text_ids: list[str] = []
    text_texts: list[str] = []
    text_gpes: list = []
    vis_ids: list[str] = []
    vis_imgs: list = []
    vis_gpes: list = []
    unk_ids: list[str] = []

    for eid in unique_ids:
        rec = data.elements.get(eid, {})
        ntype = rec.get("type")
        doc_id = rec.get("doc_id")

        gpe_feat = None
        if use_gpe and doc_id in data.features:
            f = data.features[doc_id].get(eid)
            if f is not None:
                gpe_feat = f

        if ntype in ("text", "caption", "section_header"):
            text_ids.append(eid)
            text_texts.append(rec.get("text") or "(empty)")
            text_gpes.append(gpe_feat)
        elif ntype in ("figure", "table", "equation"):
            vis_ids.append(eid)
            vis_imgs.append(data.get_image(doc_id, eid))
            vis_gpes.append(gpe_feat)
        else:
            unk_ids.append(eid)

    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    # Single forward pass for all text elements
    if text_ids:
        tok = encoder.tokenize_text(text_texts)
        gpe_batch = encoder._stack_gpe_features(text_gpes)
        z, m = encoder.forward_text(
            tok["input_ids"].to(device), tok["attention_mask"].to(device), gpe_batch
        )
        for i, eid in enumerate(text_ids):
            result[eid] = (z[i], m[i])

    # Single forward pass for all vision elements
    if vis_ids:
        valid_triples = [(eid, img, gpe) for eid, img, gpe in zip(vis_ids, vis_imgs, vis_gpes) if img is not None]
        missing_ids = [eid for eid, img in zip(vis_ids, vis_imgs) if img is None]

        if valid_triples:
            v_eids, v_imgs, v_gpes = zip(*valid_triples)
            px = encoder.preprocess_images(list(v_imgs)).to(device)
            gpe_batch = encoder._stack_gpe_features(list(v_gpes))
            z, m = encoder.forward_vision(px, gpe_batch)
            for i, eid in enumerate(v_eids):
                result[eid] = (z[i], m[i])

        for eid in missing_ids:
            result[eid] = (
                torch.zeros(196, encoder.proj_dim, device=device),
                torch.ones(196, dtype=torch.long, device=device),
            )

    for eid in unk_ids:
        result[eid] = (
            torch.zeros(1, encoder.proj_dim, device=device),
            torch.ones(1, dtype=torch.long, device=device),
        )

    return result


def compute_batch_losses(
    encoder: ElementTokenEncoder,
    batch: list[dict],
    data: SpiqaSplitData,
    tau: float,
    lambda_cov: float,
    lambda_cons: float,
    device: str = "cuda",
    use_gpe: bool = True,                # row toggle: include GPE in forward
    loss_type: str = "grcl",             # 'grcl' (graded) | 'infonce' (binary positives)
    infonce_threshold: float = 0.5,
    query_pe_dropout: float = 0.5,       # neg-control (o): 0 disables L_cons teacher/student
) -> tuple[torch.Tensor, dict[str, float]]:
    # ── Collect all unique element IDs for the whole batch ──────────────────
    anchor_eids = [inst["anchor_id"] for inst in batch if not inst.get("anchor_is_text")]
    cand_eids   = [cid for inst in batch for cid in inst["candidates"]]
    all_eids    = list(dict.fromkeys(anchor_eids + cand_eids))

    # Two grouped forward passes instead of ~200 individual calls
    encoded = batch_encode_elements(encoder, all_eids, data, use_gpe=use_gpe, device=device)

    # For L_cons: re-encode anchors without GPE (small batch, anchors only)
    encoded_no_gpe: dict = {}
    if use_gpe and lambda_cons > 0 and anchor_eids:
        encoded_no_gpe = batch_encode_elements(
            encoder, list(dict.fromkeys(anchor_eids)), data, use_gpe=False, device=device
        )

    # Batch-encode NL query texts (anchor_is_text instances)
    nl_instances = [(i, inst) for i, inst in enumerate(batch) if inst.get("anchor_is_text")]
    nl_encoded: dict[int, tuple] = {}
    if nl_instances:
        texts = [inst["anchor_text"] for _, inst in nl_instances]
        tok = encoder.tokenize_text(texts)
        z, m = encoder.forward_text(
            tok["input_ids"].to(device), tok["attention_mask"].to(device), None
        )
        for j, (i, _) in enumerate(nl_instances):
            nl_encoded[i] = (z[j], m[j])

    # ── Compute losses using cached tensors ─────────────────────────────────
    grcl_terms, cov_terms, cons_terms = [], [], []
    cap_fig_cos = []
    kind_counts = {"caption_of": 0, "refer_to": 0, "nl_qa": 0}

    for i, inst in enumerate(batch):
        kind_counts[inst["anchor_kind"]] = kind_counts.get(inst["anchor_kind"], 0) + 1

        if inst.get("anchor_is_text"):
            a_full, a_mask = nl_encoded[i]
            a_drop, _ = a_full, a_mask
            anchor_has_gpe = False
        else:
            a_full, a_mask = encoded[inst["anchor_id"]]
            # Compute a_drop teacher only when GPE active AND query_pe_dropout > 0.
            # query_pe_dropout=0 (neg-control o) disables the L_cons mechanism.
            if use_gpe and query_pe_dropout > 0 and lambda_cons > 0:
                a_drop, _ = encoded_no_gpe[inst["anchor_id"]]
            else:
                a_drop, _ = a_full, a_mask
            anchor_has_gpe = use_gpe

        cand_tokens = [encoded[cid][0] for cid in inst["candidates"]]
        cand_masks  = [encoded[cid][1] for cid in inst["candidates"]]

        scores_full = torch.stack([
            late_interaction_score(a_full, a_mask, ct, cm) for ct, cm in zip(cand_tokens, cand_masks)
        ])
        rels = torch.tensor(inst["relevances"], dtype=torch.float32, device=device)

        if loss_type == "infonce":
            rels_for_loss = (rels >= infonce_threshold).float()
        else:
            rels_for_loss = rels
        grcl_terms.append(grcl_loss(scores_full, rels_for_loss, tau=tau))
        cov_terms.append(coverage_loss(scores_full, rels, K=5))

        if anchor_has_gpe and lambda_cons > 0:
            scores_drop = torch.stack([
                late_interaction_score(a_drop, a_mask, ct, cm) for ct, cm in zip(cand_tokens, cand_masks)
            ])
            cons_terms.append(consistency_loss(scores_full, scores_drop, tau=tau))

        target_id = inst["target_id"]
        if target_id in inst["candidates"]:
            t_idx = inst["candidates"].index(target_id)
            t_pool = F.normalize(cand_tokens[t_idx].mean(0), dim=-1)
            a_pool = F.normalize(a_full.mean(0), dim=-1)
            cap_fig_cos.append((a_pool * t_pool).sum().item())

    L_grcl = torch.stack(grcl_terms).mean()
    L_cov  = torch.stack(cov_terms).mean()
    L_cons = torch.stack(cons_terms).mean() if cons_terms else torch.zeros(1, device=device).squeeze()
    L_tot  = total_loss(L_grcl, L_cov, L_cons, lambda_cov=lambda_cov, lambda_cons=lambda_cons)

    metrics = {
        "L_grcl": L_grcl.item(),
        "L_cov": L_cov.item(),
        "L_cons": L_cons.item() if isinstance(L_cons, torch.Tensor) else 0.0,
        "L_total": L_tot.item(),
        "diag_cos_mean": sum(cap_fig_cos) / max(1, len(cap_fig_cos)),
        **{f"n_{k}": v for k, v in kind_counts.items()},
    }
    return L_tot, metrics


# ────────────────────────────────────────────────────────────────────────────
# Held-out evaluation on SPIQA test-A
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_caption_figure_cos(
    encoder: ElementTokenEncoder,
    test_data: SpiqaSplitData,
    n_pairs: int = 200,
    device: str = "cuda",
) -> float:
    encoder.eval()
    pairs = random.Random(0).sample(test_data.caption_of_anchors, min(n_pairs, len(test_data.caption_of_anchors)))
    all_ids = list(dict.fromkeys(eid for pair in pairs for eid in (pair[1], pair[2])))
    encoded = batch_encode_elements(encoder, all_ids, test_data, use_gpe=False, device=device)
    sims = []
    for _, cap_id, fig_id in pairs:
        c_tok, c_mask = encoded[cap_id]
        f_tok, f_mask = encoded[fig_id]
        c_pool = F.normalize((c_tok * c_mask.unsqueeze(-1).float()).sum(0) / c_mask.sum().clamp(min=1).float(), dim=-1)
        f_pool = F.normalize((f_tok * f_mask.unsqueeze(-1).float()).sum(0) / f_mask.sum().clamp(min=1).float(), dim=-1)
        sims.append((c_pool * f_pool).sum().item())
    encoder.train()
    return sum(sims) / len(sims)


@torch.no_grad()
def eval_recall_at_k(
    encoder: ElementTokenEncoder,
    test_data: SpiqaSplitData,
    ks: list[int] = (1, 5, 10),
    n_queries: int | None = None,
    device: str = "cuda",
) -> dict[int, float]:
    """Per-paper pool. For each QA: encode question (no GPE), score all elements
    in that paper, check if reference figure is in top-k."""
    encoder.eval()
    qa_list = test_data.qa
    if n_queries is not None:
        qa_list = random.Random(0).sample(qa_list, min(n_queries, len(qa_list)))

    # Pre-encode all unique candidates across all QA pairs in one pass
    valid_qa = []
    for doc_id, qa in qa_list:
        target = qa["reference"]
        nodes_by_id = test_data.nodes_by_doc[doc_id]
        cand_ids = [
            e for e in test_data.doc_node_ids.get(doc_id, [])
            if nodes_by_id.get(e, {}).get("type") != "section_header"
        ]
        if target in cand_ids:
            valid_qa.append((doc_id, qa, cand_ids))

    all_cand_ids = list(dict.fromkeys(cid for _, _, cands in valid_qa for cid in cands))
    encoded_cands = batch_encode_elements(encoder, all_cand_ids, test_data, use_gpe=True, device=device)

    hits = {k: 0 for k in ks}
    n = 0
    for doc_id, qa, cand_ids in valid_qa:
        q_tok, q_mask = encode_text_only(encoder, qa["question"], device=device)
        scores = [
            late_interaction_score(q_tok, q_mask, encoded_cands[cid][0], encoded_cands[cid][1]).item()
            for cid in cand_ids
        ]
        ranked_ids = [x[0] for x in sorted(zip(cand_ids, scores), key=lambda x: -x[1])]
        target = qa["reference"]
        target_rank = ranked_ids.index(target) + 1 if target in ranked_ids else len(ranked_ids) + 1
        for k in ks:
            if target_rank <= k:
                hits[k] += 1
        n += 1

    encoder.train()
    return {k: (hits[k] / max(1, n)) for k in ks} | {"_n": n}


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None,
                    help="ablation row id from configs.ROW_CONFIGS (e.g. a, e, f, h)")
    ap.add_argument("--train_sample", type=int, default=1000,
                    help="random sample of papers from SPIQA train (0 = use all 25k)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--lambda_cov", type=float, default=0.3)
    ap.add_argument("--lambda_cons", type=float, default=0.5)
    ap.add_argument("--use_gpe", type=int, default=1)
    ap.add_argument("--loss_type", type=str, default="grcl", choices=("grcl", "infonce"))
    ap.add_argument("--infonce_threshold", type=float, default=0.5)
    ap.add_argument("--anchor_kind", type=str, default="mixed",
                    choices=("caption_of", "refer_to", "nl_qa", "mixed"))
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_cf_pairs", type=int, default=200)
    ap.add_argument("--eval_recall_n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="eval/results/trainer/run.json")
    ap.add_argument("--save_ckpt", type=str, default=None, help="if set, save state_dict here")
    # New flags for ablation rows
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="GRCL graph-relevance 2-hop decay (γ)")
    ap.add_argument("--gpe_facets", type=str, default="type,role,depth,pos",
                    help="comma list of active GPE facets (for ablation rows b,c,d)")
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--hf_id", type=str, default="google/siglip2-base-patch16-224",
                    help="HF model id for the encoder (override for (p) CLIP-L/14 swap)")
    ap.add_argument("--edge_types_only", type=str, default=None,
                    help="comma list of edges to keep in graph relevance (sub-ablation)")
    ap.add_argument("--query_pe_dropout", type=float, default=0.5,
                    help="prob. of dropping GPE on query/anchor side during L_cons (negative control o)")
    ap.add_argument("--section_role_mode", type=str, default="normal",
                    choices=("normal", "shuffled", "random"),
                    help="negative controls (m, n) — re-assign section_role at load time")
    ap.add_argument("--tokens_per_visual", type=int, default=0,
                    help="patch-pool visual tokens to this count (16, 64). 0 = no pool (full ~196)")
    args = ap.parse_args()

    # If --config provided, override args from preset (CLI args still take precedence
    # for fields not in preset).
    if args.config:
        from pipeline.configs import get_config
        cfg = get_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) == ap.get_default(k):
                setattr(args, k, v)
        # Force-apply key training fields
        for k in ("use_gpe", "loss_type", "lambda_cov", "lambda_cons", "steps",
                   "batch", "pool", "lr", "lora_rank", "lora_alpha", "tau", "train_sample",
                   "eval_every", "eval_cf_pairs", "eval_recall_n",
                   "anchor_kind", "seed", "gamma", "gpe_facets", "hf_id",
                   "edge_types_only", "query_pe_dropout", "section_role_mode",
                   "infonce_threshold", "tokens_per_visual",
                   "warmup_steps", "weight_decay"):
            if k in cfg:
                setattr(args, k, cfg[k])
        # use_gpe stored as bool in config, but argparse uses int
        if isinstance(getattr(args, "use_gpe", None), bool):
            args.use_gpe = int(args.use_gpe)
        # Default out path / ckpt path per row
        run_name = cfg["name"]
        if args.out == ap.get_default("out"):
            args.out = f"eval/results/trainer/run_{cfg['row_id']}_{run_name}.json"
        if not args.save_ckpt:
            args.save_ckpt = f"ckpt/{cfg['row_id']}_{run_name}.pt"

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\nargs: {vars(args)}\n")

    # ── load test-A held-out ──
    print("[1/3] loading SPIQA test-A (held-out eval)...")
    test_data = SpiqaSplitData(TESTA_GRAPH, TESTA_ELEMENTS, TESTA_QA, TESTA_IMG_ROOT)

    # ── load SPIQA train subsample ──
    print(f"\n[2/3] loading SPIQA train (sample N={args.train_sample})...")
    if args.train_sample > 0:
        # Read just the keys first to enable subsetting before parsing all
        all_train_keys = list(json.loads(TRAIN_QA.read_text()).keys())
        rng = random.Random(args.seed)
        rng.shuffle(all_train_keys)
        subset_ids = set(all_train_keys[: args.train_sample])
        print(f"  selected {len(subset_ids):,} paper ids")
    else:
        subset_ids = None
    train_data = SpiqaSplitData(TRAIN_GRAPH, TRAIN_ELEMENTS, TRAIN_QA, TRAIN_IMG_ROOT, subset_ids,
                                section_role_mode=args.section_role_mode,
                                role_shuffle_seed=args.seed)
    edge_set = set(args.edge_types_only.split(",")) if args.edge_types_only else None
    train_ds = TrainDataset(train_data, seed=args.seed,
                            gamma=args.gamma, edge_types_only=edge_set)

    # ── encoder ──
    print(f"\n[3/3] building encoder...")
    enc_kwargs = dict(device=device, lora_rank=args.lora_rank)
    if hasattr(args, "lora_alpha") and args.lora_alpha:
        enc_kwargs["lora_alpha"] = args.lora_alpha
    if hasattr(args, "hf_id") and args.hf_id:
        enc_kwargs["hf_id"] = args.hf_id
    if hasattr(args, "gpe_facets") and args.gpe_facets:
        enc_kwargs["gpe_active_facets"] = tuple(args.gpe_facets.split(","))
    if hasattr(args, "tokens_per_visual") and args.tokens_per_visual:
        enc_kwargs["tokens_per_visual"] = args.tokens_per_visual
    encoder = ElementTokenEncoder(**enc_kwargs)
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ── initial eval on test-A ──
    print("\n[eval 0] held-out test-A baseline...")
    cf0 = eval_caption_figure_cos(encoder, test_data, n_pairs=args.eval_cf_pairs, device=device)
    rec0 = eval_recall_at_k(encoder, test_data, ks=[1, 5, 10],
                              n_queries=args.eval_recall_n, device=device)
    print(f"  caption-figure cos: {cf0:+.4f}")
    print(f"  Recall@1/5/10 (n={rec0['_n']}): "
          f"{rec0[1]*100:.1f}% / {rec0[5]*100:.1f}% / {rec0[10]*100:.1f}%")

    history = [{
        "step": 0,
        "L_total": None,
        "testA_cap_fig_cos": cf0,
        "testA_recall@1": rec0[1], "testA_recall@5": rec0[5], "testA_recall@10": rec0[10],
        "testA_n": rec0["_n"],
        "alphas": encoder.gpe.alphas, "beta": encoder.beta.item(),
    }]
    best_recall10 = rec0[10]
    best_step = 0

    # ── training loop ──
    encoder.train()
    t0 = time.time()
    running = {"L_grcl": 0, "L_cov": 0, "L_cons": 0, "L_total": 0,
               "diag_cos_mean": 0, "n_caption_of": 0, "n_refer_to": 0, "n_nl_qa": 0,
               "n": 0}

    for step in range(1, args.steps + 1):
        batch = []
        while len(batch) < args.batch:
            inst = train_ds.sample(kind=args.anchor_kind, pool_size=args.pool)
            if inst is None:
                continue
            batch.append(inst)

        L, m = compute_batch_losses(encoder, batch, train_data,
                                     tau=args.tau, lambda_cov=args.lambda_cov,
                                     lambda_cons=args.lambda_cons, device=device,
                                     use_gpe=bool(args.use_gpe),
                                     loss_type=args.loss_type,
                                     infonce_threshold=getattr(args, "infonce_threshold", 0.5),
                                     query_pe_dropout=getattr(args, "query_pe_dropout", 0.5))
        optim.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()
        scheduler.step()

        for k in ["L_grcl", "L_cov", "L_cons", "L_total", "diag_cos_mean"]:
            running[k] += m[k]
        running["n_caption_of"] += m.get("n_caption_of", 0)
        running["n_refer_to"] += m.get("n_refer_to", 0)
        running["n_nl_qa"] += m.get("n_nl_qa", 0)
        running["n"] += 1

        if step % 20 == 0:
            n = running["n"]
            elapsed = time.time() - t0
            print(
                f"step {step:5d}  L_grcl={running['L_grcl']/n:6.2f}  L_cov={running['L_cov']/n:5.2f}  "
                f"L_cons={running['L_cons']/n:.3f}  L_tot={running['L_total']/n:6.2f}  "
                f"diag={running['diag_cos_mean']/n:+.3f}  "
                f"mix=[cap:{running['n_caption_of']}/ref:{running['n_refer_to']}/nl:{running['n_nl_qa']}]  "
                f"β={encoder.beta.item():.3f}  α_t={encoder.gpe.alphas['type']:.3f}  ({elapsed:.0f}s)"
            )
            running = {"L_grcl": 0, "L_cov": 0, "L_cons": 0, "L_total": 0,
                       "diag_cos_mean": 0, "n_caption_of": 0, "n_refer_to": 0, "n_nl_qa": 0, "n": 0}

        if step % args.eval_every == 0:
            print(f"\n  [eval @ step {step}] held-out test-A...")
            cf = eval_caption_figure_cos(encoder, test_data, n_pairs=args.eval_cf_pairs, device=device)
            rec = eval_recall_at_k(encoder, test_data, ks=[1, 5, 10],
                                    n_queries=args.eval_recall_n, device=device)
            d_cf = cf - cf0
            print(f"    cap-fig cos: {cf:+.4f}  (Δ {d_cf:+.4f})")
            print(f"    R@1/5/10: {rec[1]*100:.1f}% / {rec[5]*100:.1f}% / {rec[10]*100:.1f}%  "
                  f"(Δ {(rec[1]-rec0[1])*100:+.1f} / {(rec[5]-rec0[5])*100:+.1f} / {(rec[10]-rec0[10])*100:+.1f})")
            history.append({
                "step": step, "L_total": m["L_total"],
                "testA_cap_fig_cos": cf,
                "testA_recall@1": rec[1], "testA_recall@5": rec[5], "testA_recall@10": rec[10],
                "testA_n": rec["_n"],
                "alphas": encoder.gpe.alphas, "beta": encoder.beta.item(),
            })
            if rec[10] > best_recall10 and args.save_ckpt:
                best_recall10 = rec[10]; best_step = step
                Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save(encoder.state_dict(), args.save_ckpt)
                print(f"    [ckpt saved @ step {step} R@10={rec[10]*100:.1f}%]")
            print()

    # ── final ──
    print("\n[final] test-A:")
    cf_final = eval_caption_figure_cos(encoder, test_data, n_pairs=args.eval_cf_pairs, device=device)
    rec_final = eval_recall_at_k(encoder, test_data, ks=[1, 5, 10],
                                   n_queries=args.eval_recall_n, device=device)
    print(f"  cap-fig cos: {cf_final:+.4f}  (init {cf0:+.4f}, Δ {cf_final-cf0:+.4f})")
    print(f"  R@1/5/10: {rec_final[1]*100:.1f}% / {rec_final[5]*100:.1f}% / {rec_final[10]*100:.1f}%")
    print(f"  init R@1/5/10: {rec0[1]*100:.1f}% / {rec0[5]*100:.1f}% / {rec0[10]*100:.1f}%")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "args": vars(args), "history": history,
        "init": {"cap_fig_cos": cf0, **rec0},
        "final": {"cap_fig_cos": cf_final, **rec_final},
        "best_step": best_step, "total_seconds": time.time() - t0,
    }, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
