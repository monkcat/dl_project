"""PoC trainer — proves the pipeline can actually optimize modality gap.

Trains ElementTokenEncoder (LoRA + GPE) on SPIQA test-A's caption_of and
refer_to edges as a proof-of-concept. Periodically measures caption-figure
cosine to track modality gap closure.

This is NOT the final trainer — it uses test-A as both training and (proxy)
validation data, which would contaminate any real eval. The purpose is solely
to demonstrate that L_GRCL + L_cov + L_cons can move caption-figure cosine
above the 0.126 zero-shot baseline.

Run:
    python -m pipeline.trainer_poc --steps 300 --batch 4 --pool 12
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


GRAPH_PATH = REPO / "data/benchmarks/spiqa/test-A/element_graph_v2.json"
ELEMENTS_PATH = REPO / "data/benchmarks/spiqa/test-A/elements_v2.jsonl"
IMAGE_ROOT = REPO / "data/benchmarks/spiqa/test-A/images_224px/SPIQA_testA_Images_224px"


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────

class PocDataset:
    """Holds SPIQA test-A graphs + element records + image cache.

    Provides:
        sample_anchor() → returns one training instance
            {anchor_id, anchor_kind, anchor_doc, candidate_ids, relevances}
    """

    def __init__(self, graph_path: Path = GRAPH_PATH,
                 elements_path: Path = ELEMENTS_PATH,
                 image_root: Path = IMAGE_ROOT,
                 seed: int = 0):
        self.rng = random.Random(seed)
        print(f"[PocDataset] loading {graph_path}")
        self.graphs = json.loads(graph_path.read_text())
        print(f"  {len(self.graphs):,} papers")

        # Per-doc neighbor index for fast graph_relevance
        self.neighbor_index = {pid: build_neighbor_index(g) for pid, g in self.graphs.items()}
        # Per-doc features for GPE
        self.features = {pid: extract_node_features(g) for pid, g in self.graphs.items()}
        # Per-doc node list (for negative sampling)
        self.doc_node_ids = {pid: [n["id"] for n in g["nodes"]] for pid, g in self.graphs.items()}
        # All doc_ids for cross-doc negative sampling
        self.all_doc_ids = list(self.graphs.keys())

        # Element records
        self.elements: dict[str, dict] = {}
        with open(elements_path) as f:
            for line in f:
                r = json.loads(line)
                self.elements[r["id"]] = r

        # Available caption_of and refer_to anchor pairs
        self.caption_of_anchors: list[tuple[str, str, str]] = []  # (doc_id, caption_id, figure_id)
        self.refer_to_anchors: list[tuple[str, str, str]] = []    # (doc_id, paragraph_id, figure_id)
        for pid, g in self.graphs.items():
            for e in g.get("edges", []):
                if e["type"] == "caption_of":
                    cap_id = e["src"]
                    fig_id = e["dst"]
                    img_path = image_root / pid / fig_id
                    if img_path.exists():
                        self.caption_of_anchors.append((pid, cap_id, fig_id))
                elif e["type"] == "refer_to":
                    p_id = e["src"]
                    fig_id = e["dst"]
                    img_path = image_root / pid / fig_id
                    if img_path.exists():
                        self.refer_to_anchors.append((pid, p_id, fig_id))

        print(f"  caption_of anchors: {len(self.caption_of_anchors):,}")
        print(f"  refer_to anchors:   {len(self.refer_to_anchors):,}")

        self.image_root = image_root
        # Image cache: pid+fname → PIL.Image
        self._image_cache: dict[str, Image.Image] = {}

    def get_image(self, doc_id: str, fname: str) -> Image.Image | None:
        key = f"{doc_id}/{fname}"
        if key not in self._image_cache:
            path = self.image_root / doc_id / fname
            if not path.exists():
                return None
            self._image_cache[key] = Image.open(path).convert("RGB")
        return self._image_cache[key]

    def sample_anchor(self, kind: str = "mixed", pool_size: int = 12) -> dict | None:
        """Sample one training instance.

        Args:
            kind: 'caption_of' | 'refer_to' | 'mixed' (uniform mix)
            pool_size: total candidates (including positives)

        Returns:
            dict with anchor + pool info, or None if image missing.
        """
        if kind == "mixed":
            kind = "caption_of" if self.rng.random() < 0.5 else "refer_to"

        if kind == "caption_of":
            if not self.caption_of_anchors:
                return None
            doc_id, anchor_id, target_id = self.rng.choice(self.caption_of_anchors)
        else:  # refer_to
            if not self.refer_to_anchors:
                return None
            doc_id, anchor_id, target_id = self.rng.choice(self.refer_to_anchors)

        # Build candidate pool: positives (1-2 hop) + same-doc randoms + cross-doc randoms
        nb = self.neighbor_index[doc_id]
        doc_nodes = self.doc_node_ids[doc_id]
        # Strong positives: 1-hop neighbors of anchor (excluding section_headers as candidates)
        nodes_by_id = {n["id"]: n for n in self.graphs[doc_id]["nodes"]}
        positives_1hop = [nbr_id for nbr_id, _e in nb.get(anchor_id, [])
                          if nbr_id != anchor_id
                          and nodes_by_id.get(nbr_id, {}).get("type") != "section_header"]

        # Ensure target is in candidates
        if target_id not in positives_1hop:
            positives_1hop.append(target_id)

        # Sample same-doc negatives (must exclude anchor + 1-hop positives + section_headers)
        excluded = set([anchor_id, *positives_1hop])
        same_doc_negs = [
            n for n in doc_nodes
            if n not in excluded
            and nodes_by_id.get(n, {}).get("type") != "section_header"
        ]
        self.rng.shuffle(same_doc_negs)
        # Sample cross-doc negatives (random from other docs)
        cross_negs: list[str] = []
        for _ in range(max(4, pool_size // 4)):
            other_doc = self.rng.choice(self.all_doc_ids)
            if other_doc == doc_id:
                continue
            other_nodes = self.doc_node_ids[other_doc]
            other_nodes_filtered = [
                n for n in other_nodes
                if self.graphs[other_doc]["nodes"][other_nodes.index(n)].get("type") != "section_header"
            ]
            if other_nodes_filtered:
                cross_negs.append(self.rng.choice(other_nodes_filtered))

        # Compose pool: all positives (up to 4) + fill with same-doc + cross-doc negs
        positives = positives_1hop[:4]
        n_neg = pool_size - len(positives)
        n_same = max(0, n_neg - len(cross_negs))
        negatives = same_doc_negs[:n_same] + cross_negs[:n_neg - n_same]
        candidates = positives + negatives
        candidates = candidates[:pool_size]

        # Compute graph relevance for all candidates from anchor
        relevances = graph_relevance(anchor_id, candidates, self.graphs[doc_id],
                                      gamma=0.5, max_hops=2, neighbor_index=nb)

        return {
            "doc_id": doc_id,
            "anchor_id": anchor_id,
            "anchor_kind": kind,
            "target_id": target_id,
            "candidates": candidates,
            "relevances": relevances,
        }


# ────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ────────────────────────────────────────────────────────────────────────────

def late_interaction_score(q_tok: torch.Tensor, q_mask: torch.Tensor,
                            e_tok: torch.Tensor, e_mask: torch.Tensor) -> torch.Tensor:
    """ColBERT MaxSim with attention masks. Both shapes (K, D)."""
    sim = q_tok @ e_tok.T                                       # (K_q, K_e)
    e_mask_2d = e_mask.unsqueeze(0).bool()
    sim = sim.masked_fill(~e_mask_2d, -1e4)
    max_sim = sim.max(dim=-1).values
    max_sim = max_sim * q_mask.float()
    return max_sim.sum()


def encode_element(
    encoder: ElementTokenEncoder,
    element_id: str,
    dataset: PocDataset,
    use_gpe: bool = True,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single element to (K, D_proj) tokens + (K,) mask."""
    rec = dataset.elements.get(element_id, {})
    ntype = rec.get("type")
    doc_id = rec.get("doc_id")

    # Build GPE features
    gpe_feats = None
    if use_gpe and doc_id in dataset.features:
        f = dataset.features[doc_id].get(element_id)
        if f is not None:
            gpe_feats = encoder._stack_gpe_features([f])

    if ntype in ("text", "caption", "section_header"):
        text = rec.get("text") or "(empty)"
        tok = encoder.tokenize_text([text])
        input_ids = tok["input_ids"].to(device)
        attn_mask = tok["attention_mask"].to(device)
        z, m = encoder.forward_text(input_ids, attn_mask, gpe_feats)
        return z[0], m[0]
    elif ntype in ("figure", "table", "equation"):
        img = dataset.get_image(doc_id, element_id)
        if img is None:
            # fallback: zero tokens
            z = torch.zeros(196, encoder.proj_dim, device=device)
            m = torch.ones(196, dtype=torch.long, device=device)
            return z, m
        px = encoder.preprocess_images([img]).to(device)
        z, m = encoder.forward_vision(px, gpe_feats)
        return z[0], m[0]
    else:
        # Unknown type
        z = torch.zeros(1, encoder.proj_dim, device=device)
        m = torch.ones(1, dtype=torch.long, device=device)
        return z, m


def compute_batch_losses(
    encoder: ElementTokenEncoder,
    batch: list[dict],
    dataset: PocDataset,
    tau: float,
    lambda_cov: float,
    lambda_cons: float,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict[str, float]]:
    """For a batch of anchor instances, compute losses.

    Returns:
        L_total (scalar), metrics dict
    """
    grcl_terms = []
    cov_terms = []
    cons_terms = []
    diag_cosines = []                 # for monitoring caption-figure cosine

    for inst in batch:
        # Encode anchor (with + without GPE for L_cons)
        a_full, a_mask = encode_element(encoder, inst["anchor_id"], dataset,
                                         use_gpe=True, device=device)
        a_drop, _ = encode_element(encoder, inst["anchor_id"], dataset,
                                     use_gpe=False, device=device)

        # Encode candidates (always with GPE — they're the corpus side)
        cand_tokens = []
        cand_masks = []
        for cid in inst["candidates"]:
            ct, cm = encode_element(encoder, cid, dataset, use_gpe=True, device=device)
            cand_tokens.append(ct)
            cand_masks.append(cm)

        # Scores against all candidates
        scores_full = torch.stack([
            late_interaction_score(a_full, a_mask, ct, cm)
            for ct, cm in zip(cand_tokens, cand_masks)
        ])
        scores_drop = torch.stack([
            late_interaction_score(a_drop, a_mask, ct, cm)
            for ct, cm in zip(cand_tokens, cand_masks)
        ])
        rels = torch.tensor(inst["relevances"], dtype=torch.float32, device=device)

        grcl_terms.append(grcl_loss(scores_full, rels, tau=tau))
        cov_terms.append(coverage_loss(scores_full, rels, K=5))
        # Only apply L_cons when anchor has GPE (caption / paragraph element)
        cons_terms.append(consistency_loss(scores_full, scores_drop, tau=tau))

        # Diagnostic: cosine between anchor pool vector and target
        # (use mean-pooled tokens just for monitoring; training uses late int)
        target_id = inst["target_id"]
        if target_id in inst["candidates"]:
            t_idx = inst["candidates"].index(target_id)
            t_tok = cand_tokens[t_idx].mean(dim=0)
            t_tok = F.normalize(t_tok, dim=-1)
            a_pool = a_full.mean(dim=0)
            a_pool = F.normalize(a_pool, dim=-1)
            diag_cosines.append((a_pool * t_tok).sum().item())

    L_grcl = torch.stack(grcl_terms).mean()
    L_cov = torch.stack(cov_terms).mean()
    L_cons = torch.stack(cons_terms).mean()
    L_tot = total_loss(L_grcl, L_cov, L_cons, lambda_cov=lambda_cov, lambda_cons=lambda_cons)

    metrics = {
        "L_grcl": L_grcl.item(),
        "L_cov": L_cov.item(),
        "L_cons": L_cons.item(),
        "L_total": L_tot.item(),
        "diag_cos_mean": sum(diag_cosines) / max(1, len(diag_cosines)),
    }
    return L_tot, metrics


# ────────────────────────────────────────────────────────────────────────────
# Periodic diagnostic — caption-figure cosine baseline tracking
# ────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_caption_figure_gap(
    encoder: ElementTokenEncoder,
    dataset: PocDataset,
    n_pairs: int = 200,
    use_gpe: bool = False,            # default: as inference uses no query-GPE
    device: str = "cuda",
) -> dict[str, float]:
    """Measure caption ↔ figure mean cosine on a sample of caption_of pairs.

    Uses *mean-pooled* element vectors (not late interaction) so the number
    is directly comparable to the zero-shot baseline produced by
    `eval/analysis/modality_gap_baseline.py`.
    """
    encoder.eval()
    pairs = random.Random(0).sample(dataset.caption_of_anchors, min(n_pairs, len(dataset.caption_of_anchors)))
    sims = []
    for doc_id, cap_id, fig_id in pairs:
        c_tok, c_mask = encode_element(encoder, cap_id, dataset, use_gpe=use_gpe, device=device)
        f_tok, f_mask = encode_element(encoder, fig_id, dataset, use_gpe=use_gpe, device=device)
        # Mean-pool with mask
        c_pool = (c_tok * c_mask.unsqueeze(-1).float()).sum(0) / c_mask.sum().clamp(min=1).float()
        f_pool = (f_tok * f_mask.unsqueeze(-1).float()).sum(0) / f_mask.sum().clamp(min=1).float()
        c_pool = F.normalize(c_pool, dim=-1)
        f_pool = F.normalize(f_pool, dim=-1)
        sims.append((c_pool * f_pool).sum().item())
    encoder.train()
    return {
        "mean": sum(sims) / len(sims),
        "n_pairs": len(sims),
        "use_gpe": use_gpe,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main training loop
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--lambda_cov", type=float, default=0.3)
    ap.add_argument("--lambda_cons", type=float, default=0.5)
    ap.add_argument("--anchor_kind", type=str, default="mixed",
                    choices=("caption_of", "refer_to", "mixed"))
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_pairs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="eval/results/trainer_poc/run.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, args: {vars(args)}")

    # Data
    dataset = PocDataset(seed=args.seed)

    # Encoder
    encoder = ElementTokenEncoder(device=device, lora_rank=args.lora_rank)

    # Optimizer over trainable params
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # ── initial gap (baseline) ──
    print("\n[step 0] measuring caption-figure cosine baseline...")
    gap0 = measure_caption_figure_gap(encoder, dataset, n_pairs=args.eval_pairs, device=device)
    print(f"  cap-fig cos (init): {gap0['mean']:+.4f}  (zero-shot baseline)")

    history = [{
        "step": 0,
        "L_total": None,
        "cap_fig_cos": gap0["mean"],
        "alpha_type": encoder.gpe.alphas["type"],
        "alpha_role": encoder.gpe.alphas["role"],
        "alpha_depth": encoder.gpe.alphas["depth"],
        "alpha_pos": encoder.gpe.alphas["pos"],
        "beta": encoder.beta.item(),
    }]

    # ── training loop ──
    encoder.train()
    t_start = time.time()
    running = {"L_grcl": 0.0, "L_cov": 0.0, "L_cons": 0.0, "L_total": 0.0, "diag_cos_mean": 0.0, "n": 0}

    for step in range(1, args.steps + 1):
        batch = []
        while len(batch) < args.batch:
            inst = dataset.sample_anchor(kind=args.anchor_kind, pool_size=args.pool)
            if inst is None:
                continue
            batch.append(inst)

        L, m = compute_batch_losses(
            encoder, batch, dataset,
            tau=args.tau, lambda_cov=args.lambda_cov, lambda_cons=args.lambda_cons,
            device=device,
        )

        optim.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()

        running["L_grcl"] += m["L_grcl"]
        running["L_cov"] += m["L_cov"]
        running["L_cons"] += m["L_cons"]
        running["L_total"] += m["L_total"]
        running["diag_cos_mean"] += m["diag_cos_mean"]
        running["n"] += 1

        if step % 10 == 0:
            n = running["n"]
            elapsed = time.time() - t_start
            print(
                f"step {step:4d}  L_grcl={running['L_grcl']/n:6.2f}  L_cov={running['L_cov']/n:5.2f}  "
                f"L_cons={running['L_cons']/n:5.2f}  L_tot={running['L_total']/n:6.2f}  "
                f"diag_cos={running['diag_cos_mean']/n:+.3f}  β={encoder.beta.item():.3f}  "
                f"α_type={encoder.gpe.alphas['type']:.3f}  ({elapsed:.0f}s)"
            )
            running = {"L_grcl": 0.0, "L_cov": 0.0, "L_cons": 0.0, "L_total": 0.0, "diag_cos_mean": 0.0, "n": 0}

        if step % args.eval_every == 0:
            print(f"  [eval] caption-figure cosine on {args.eval_pairs} pairs:")
            gap = measure_caption_figure_gap(encoder, dataset, n_pairs=args.eval_pairs, device=device)
            delta = gap["mean"] - gap0["mean"]
            print(f"    mean = {gap['mean']:+.4f}  (Δ vs init = {delta:+.4f})")
            history.append({
                "step": step,
                "L_total": m["L_total"],
                "cap_fig_cos": gap["mean"],
                "alpha_type": encoder.gpe.alphas["type"],
                "alpha_role": encoder.gpe.alphas["role"],
                "alpha_depth": encoder.gpe.alphas["depth"],
                "alpha_pos": encoder.gpe.alphas["pos"],
                "beta": encoder.beta.item(),
            })

    # Final eval
    print("\n[final] caption-figure cosine:")
    gap_final = measure_caption_figure_gap(encoder, dataset, n_pairs=args.eval_pairs, device=device)
    print(f"  mean = {gap_final['mean']:+.4f}  (Δ vs init = {gap_final['mean']-gap0['mean']:+.4f})")
    print(f"  baseline (zero-shot SigLIPv2 from §motivation): +0.1257")

    # Save
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "args": vars(args),
        "history": history,
        "final_gap": gap_final,
        "init_gap": gap0,
        "total_seconds": time.time() - t_start,
    }, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
