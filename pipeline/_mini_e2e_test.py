"""End-to-end mini integration test — proves the full pipeline composes.

Picks ONE SPIQA test-A paper, encodes its caption + figure + paragraph candidates,
computes graph relevance from a caption anchor, and runs one forward+backward
through GRCL + L_cov + L_cons.

Purpose: catch shape/dtype/device bugs before scaling to the trainer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.element_encoder import ElementTokenEncoder
from pipeline.graph_pe import extract_node_features
from pipeline.graph_relevance import build_neighbor_index, graph_relevance
from pipeline.losses import consistency_loss, coverage_loss, grcl_loss, total_loss


def late_interaction_score(q_tokens: torch.Tensor, q_mask: torch.Tensor,
                            e_tokens: torch.Tensor, e_mask: torch.Tensor) -> torch.Tensor:
    """ColBERT MaxSim with masking. Both inputs (K, D); masks (K,)."""
    sim = q_tokens @ e_tokens.T                            # (K_q, K_e)
    # Mask out padding cols of e
    e_mask_2d = e_mask.unsqueeze(0).bool()
    sim = sim.masked_fill(~e_mask_2d, -1e4)
    max_sim = sim.max(dim=-1).values                       # (K_q,)
    # Mask out padding rows of q
    max_sim = max_sim * q_mask.float()
    return max_sim.sum()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load one paper
    graphs = json.loads((REPO / "data/benchmarks/spiqa/test-A/element_graph_v2.json").read_text())
    elements_path = REPO / "data/benchmarks/spiqa/test-A/elements_v2.jsonl"
    paper_id = "1611.05742v3"
    g = graphs[paper_id]
    print(f"paper: {paper_id}, {len(g['nodes'])} nodes, {len(g['edges'])} edges")

    # Load element records (text content / image paths)
    elements_by_id: dict[str, dict] = {}
    with open(elements_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("doc_id") == paper_id:
                elements_by_id[r["id"]] = r

    # Feature extraction (for GPE)
    features = extract_node_features(g)

    # Pick anchor: a caption
    anchor_id = next(
        (n["id"] for n in g["nodes"] if n["type"] == "caption"),
        None
    )
    if anchor_id is None:
        raise SystemExit("no caption in this paper")
    print(f"anchor: {anchor_id} (caption)")

    # Candidate pool: all elements except section_headers (just for test scale)
    candidate_ids = [n["id"] for n in g["nodes"] if n["type"] != "section_header"]
    # Limit to 16 candidates for a small clean test
    candidate_ids = candidate_ids[:16]
    if anchor_id not in candidate_ids:
        candidate_ids = [anchor_id] + candidate_ids[:15]
    print(f"candidates: {len(candidate_ids)}")

    # Graph relevance
    nb = build_neighbor_index(g)
    rels = graph_relevance(anchor_id, candidate_ids, g, gamma=0.5, max_hops=2, neighbor_index=nb)
    rels_t = torch.tensor(rels, dtype=torch.float32)
    print(f"\ngraph relevances (per candidate):")
    for cid, r in zip(candidate_ids, rels):
        ntype = next((n["type"] for n in g["nodes"] if n["id"] == cid), "?")
        print(f"  {r:.4f}  {ntype:18s} {cid[:50]}")

    # Build encoder
    print("\nbuilding encoder...")
    enc = ElementTokenEncoder(device=device, lora_rank=8)

    # Encode anchor (caption text)
    anchor_text = elements_by_id[anchor_id].get("text", "")
    print(f"\nanchor text: {anchor_text[:80]}...")
    tok = enc.tokenize_text([anchor_text])
    a_gpe = enc._stack_gpe_features([features[anchor_id]])
    a_tokens, a_mask = enc.forward_text(
        tok["input_ids"].to(device), tok["attention_mask"].to(device), a_gpe
    )
    # Drop pass (no GPE) for L_cons
    a_tokens_drop, _ = enc.forward_text(
        tok["input_ids"].to(device), tok["attention_mask"].to(device), None
    )
    print(f"anchor tokens (with GPE): {a_tokens.shape}")
    print(f"anchor tokens (no GPE):   {a_tokens_drop.shape}")

    # Encode each candidate (text or vision) + compute score
    image_root = REPO / "data/benchmarks/spiqa/test-A/images_224px/SPIQA_testA_Images_224px" / paper_id
    scores_full = []
    scores_drop = []
    for cid in candidate_ids:
        rec = elements_by_id.get(cid, {})
        ntype = rec.get("type")
        c_gpe = enc._stack_gpe_features([features[cid]])

        if ntype in ("text", "caption", "section_header"):
            text = rec.get("text", "") or "(empty)"
            ctok = enc.tokenize_text([text])
            c_tokens, c_mask = enc.forward_text(
                ctok["input_ids"].to(device), ctok["attention_mask"].to(device), c_gpe
            )
        elif ntype in ("figure", "table", "equation"):
            img_path = image_root / cid
            if not img_path.exists():
                # fallback: skip with random tokens (we only need the test to run)
                c_tokens = torch.randn(1, 196, enc.proj_dim, device=device)
                c_tokens = F.normalize(c_tokens, dim=-1)
                c_mask = torch.ones(1, 196, dtype=torch.long, device=device)
            else:
                img = Image.open(img_path).convert("RGB")
                px = enc.preprocess_images([img]).to(device)
                c_tokens, c_mask = enc.forward_vision(px, c_gpe)
        else:
            continue

        # Late interaction
        s_full = late_interaction_score(a_tokens[0], a_mask[0], c_tokens[0], c_mask[0])
        s_drop = late_interaction_score(a_tokens_drop[0], a_mask[0], c_tokens[0], c_mask[0])
        scores_full.append(s_full)
        scores_drop.append(s_drop)

    scores_full = torch.stack(scores_full)
    scores_drop = torch.stack(scores_drop)
    rels_t = rels_t.to(scores_full.device)

    print(f"\nscores_full: {scores_full.tolist()}")
    print(f"relevances:  {rels_t.tolist()}")

    # Compute losses
    L_grcl = grcl_loss(scores_full, rels_t, tau=0.07)
    L_cov = coverage_loss(scores_full, rels_t, K=4)
    L_cons = consistency_loss(scores_full, scores_drop, tau=0.07)
    L_tot = total_loss(L_grcl, L_cov, L_cons, lambda_cov=0.3, lambda_cons=0.5)

    print(f"\nlosses:")
    print(f"  L_grcl = {L_grcl.item():.4f}")
    print(f"  L_cov  = {L_cov.item():.4f}")
    print(f"  L_cons = {L_cons.item():.4f}")
    print(f"  total  = {L_tot.item():.4f}")

    # Backward
    L_tot.backward()

    # Check key gradients
    grad_summary = {}
    for name, p in enc.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if any(k in name for k in ("raw_beta", "raw_alpha", "proj_text.weight", "proj_vision.weight", "lora_A")):
            grad_summary[name] = p.grad.abs().mean().item()

    print(f"\ngradient flow (mean |grad|, key params):")
    for k, v in list(grad_summary.items())[:8]:
        print(f"  {v:.3e}  {k}")
    print(f"  ... ({len(grad_summary)} params have grad)")


if __name__ == "__main__":
    main()
