"""Multi-dataset evaluation orchestrator.

Loads a trained encoder checkpoint and runs retrieval evaluation on:
    - SPIQA test-A  (single-positive: 1 figure per QA)
    - SciEGQA       (multi-positive: multi-bbox, multi-page common)
    - MMDocIR       (multi-positive: GT element_ids list)

For each (dataset × inference variant) cell, reports:
    - Recall@1 / Recall@5 / Recall@10
    - Coverage@K (multi-pos: fraction of GT recovered)
    - PerfectSet@K (multi-pos: 1 if GT ⊆ top-K)
    - MRR (first-hit reciprocal rank)
    - Cross-page hit rate (when GT spans multiple pages)

Inference variants (from configs.INFERENCE_VARIANTS):
    - no_prop       : raw late-interaction scores
    - prop_a03_T2   : + graph_propagate(α=0.3, T=2) on top-50 candidates

Usage:
    # Evaluate a checkpoint on all 3 datasets, both inference variants
    python -m pipeline.eval_full --ckpt ckpt/h_full_method.pt --out eval/results/eval_h.json

    # Evaluate GME reference (no checkpoint, zero-shot encoder swap)
    python -m pipeline.eval_full --encoder gme --out eval/results/eval_gme.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.configs import EVAL_DATASETS, INFERENCE_VARIANTS
from pipeline.element_encoder import ElementTokenEncoder
from pipeline.graph_pe import extract_node_features
from pipeline.graph_relevance import build_neighbor_index
from pipeline.types import (
    BASE_EDGE_WEIGHTS, SECTION_ROLE_MODIFIER, VISUAL_TARGET_MODIFIER,
    graph_propagate,
)


# ────────────────────────────────────────────────────────────────────────────
# Query loaders
# ────────────────────────────────────────────────────────────────────────────

def load_queries(cfg: dict) -> list[dict]:
    """Return a uniform list of {qid, doc_id, query, gt_element_ids: [...], gt_pages: [...]}."""
    queries: list[dict] = []
    fmt = cfg["query_format"]
    qp = REPO / cfg["queries"]

    if fmt == "spiqa":
        data = json.loads(qp.read_text())
        for doc_id, paper in data.items():
            for i, qa in enumerate(paper.get("qa", [])):
                ref = qa.get("reference")
                if not isinstance(ref, str):
                    continue
                queries.append({
                    "qid": f"{doc_id}_q{i}",
                    "doc_id": doc_id,
                    "query": qa["question"],
                    "gt_element_ids": [ref],
                    "gt_pages": [],
                })
    elif fmt == "jsonl_gt_ids":
        with open(qp) as f:
            for line in f:
                q = json.loads(line)
                ids = q.get("gt_element_ids") or []
                if isinstance(ids, str):
                    ids = [ids]
                queries.append({
                    "qid": q.get("qid"),
                    "doc_id": q["doc_id"],
                    "query": q["query"],
                    "gt_element_ids": list(ids),
                    "gt_pages": list(q.get("gt_pages") or q.get("evidence_pages") or []),
                })
    else:
        raise ValueError(f"unknown query_format: {fmt}")
    return queries


# ────────────────────────────────────────────────────────────────────────────
# Encoder loading
# ────────────────────────────────────────────────────────────────────────────

def load_encoder(args) -> ElementTokenEncoder:
    enc = ElementTokenEncoder(
        hf_id=args.hf_id,
        proj_dim=args.proj_dim,
        use_lora=True,
        lora_rank=args.lora_rank,
        device=args.device,
    )
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=args.device)
        enc.load_state_dict(state, strict=False)
        print(f"[eval_full] loaded checkpoint: {args.ckpt}")
    enc.eval()
    return enc


# ────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ────────────────────────────────────────────────────────────────────────────

def _late_interaction(q_tok, q_mask, e_tok, e_mask) -> float:
    sim = q_tok @ e_tok.T
    sim = sim.masked_fill(~e_mask.unsqueeze(0).bool(), -1e4)
    return (sim.max(dim=-1).values * q_mask.float()).sum().item()


@torch.no_grad()
def encode_corpus_for_doc(
    encoder: ElementTokenEncoder,
    doc_id: str,
    elements_by_id: dict[str, dict],
    doc_node_ids: list[str],
    features: dict[str, dict],
    image_root: Path | None,
    device: str = "cuda",
    use_gpe: bool = True,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Encode every element of one paper. Returns {eid: (tokens (K,D), mask (K,))}."""
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for eid in doc_node_ids:
        rec = elements_by_id.get(eid, {})
        ntype = rec.get("type")
        if ntype == "section_header":
            continue

        gpe = None
        if use_gpe:
            f = features.get(eid)
            if f is not None:
                gpe = encoder._stack_gpe_features([f])

        if ntype in ("text", "caption"):
            text = rec.get("text") or "(empty)"
            tok = encoder.tokenize_text([text])
            z, m = encoder.forward_text(
                tok["input_ids"].to(device),
                tok["attention_mask"].to(device),
                gpe,
            )
            out[eid] = (z[0].cpu(), m[0].cpu())
        elif ntype in ("figure", "table", "equation"):
            img_path = (image_root / doc_id / eid) if image_root else None
            if img_path and img_path.exists():
                img = Image.open(img_path).convert("RGB")
            else:
                # MMDocIR keeps images inline as base64 in academic_elements.jsonl
                img_b64 = rec.get("image_b64")
                if img_b64:
                    import base64, io
                    img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
                else:
                    # Skip if no image available — score will be 0 against any query
                    out[eid] = (torch.zeros(1, encoder.proj_dim), torch.zeros(1, dtype=torch.long))
                    continue
            px = encoder.preprocess_images([img]).to(device)
            z, m = encoder.forward_vision(px, gpe)
            out[eid] = (z[0].cpu(), m[0].cpu())
    return out


@torch.no_grad()
def encode_query(encoder, text: str, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """NL query: no GPE."""
    tok = encoder.tokenize_text([text])
    z, m = encoder.forward_text(
        tok["input_ids"].to(device),
        tok["attention_mask"].to(device),
        None,
    )
    return z[0].cpu(), m[0].cpu()


# ────────────────────────────────────────────────────────────────────────────
# Per-query evaluation
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics_for_query(
    ranked_ids: list[str],
    gt_ids: list[str],
    gt_pages: list[int],
    elements_by_id: dict[str, dict],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall/Coverage/Perfect/MRR for one query."""
    gt_set = set(gt_ids)
    out: dict[str, float] = {}
    # MRR
    rank_first = None
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in gt_set:
            rank_first = i
            break
    out["mrr"] = 1.0 / rank_first if rank_first else 0.0

    for k in ks:
        topk = set(ranked_ids[:k])
        hit = len(topk & gt_set)
        out[f"recall@{k}"] = hit / max(1, len(gt_set))
        out[f"hit@{k}"] = 1.0 if hit > 0 else 0.0
        out[f"coverage@{k}"] = hit / max(1, len(gt_set))
        out[f"perfect@{k}"] = 1.0 if gt_set.issubset(topk) else 0.0

    # Cross-page hit (multi-page GT only)
    if len(gt_set) > 1:
        gt_page_set = set(int(p) for p in gt_pages if isinstance(p, (int, str)))
        if len(gt_page_set) > 1:
            # Did top-10 cover all GT pages?
            top10_pages = set()
            for rid in ranked_ids[:10]:
                p = elements_by_id.get(rid, {}).get("page")
                if p is not None:
                    top10_pages.add(int(p))
            out["cross_page_hit"] = 1.0 if gt_page_set.issubset(top10_pages) else 0.0
    return out


# ────────────────────────────────────────────────────────────────────────────
# Dataset eval loop
# ────────────────────────────────────────────────────────────────────────────

def eval_dataset(
    encoder: ElementTokenEncoder,
    ds_cfg: dict,
    inference_variants: dict[str, dict],
    args,
) -> dict[str, dict]:
    """Returns {variant_name: {metric: aggregate_value, ...}}."""
    device = args.device
    print(f"\n[eval] {ds_cfg['name']}")

    # Load graph + elements + queries
    graphs = json.loads((REPO / ds_cfg["graph"]).read_text())
    elements_by_id: dict[str, dict] = {}
    with open(REPO / ds_cfg["elements"]) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            elements_by_id[r["id"]] = r
    queries = load_queries(ds_cfg)
    if args.max_queries:
        queries = queries[: args.max_queries]
    print(f"  graph: {len(graphs)} docs, elements: {len(elements_by_id)}, queries: {len(queries)}")

    # Precompute features per doc + neighbor indices for propagation
    doc_features = {pid: extract_node_features(g) for pid, g in graphs.items()}
    doc_neighbors = {pid: build_neighbor_index(g) for pid, g in graphs.items()}
    doc_node_ids = {pid: [n["id"] for n in g["nodes"]] for pid, g in graphs.items()}

    image_root = REPO / ds_cfg["image_root"] if ds_cfg.get("image_root") else None

    # Per-variant accumulators
    aggregates: dict[str, dict[str, list[float]]] = {v: defaultdict(list) for v in inference_variants}

    # Group queries by doc to amortize per-doc corpus encoding
    queries_by_doc: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        queries_by_doc[q["doc_id"]].append(q)

    t0 = time.time()
    n_done = 0

    for doc_idx, (doc_id, qs) in enumerate(queries_by_doc.items()):
        if doc_id not in graphs:
            n_done += len(qs)
            continue
        node_ids = doc_node_ids[doc_id]
        features = doc_features[doc_id]
        # Encode all candidate elements once per doc (with GPE — corpus side always has it)
        corpus = encode_corpus_for_doc(
            encoder, doc_id, elements_by_id, node_ids, features, image_root,
            device=device, use_gpe=True,
        )
        # Move corpus tokens to GPU for fast scoring
        for eid in corpus:
            corpus[eid] = (corpus[eid][0].to(device), corpus[eid][1].to(device))
        candidate_ids = [e for e in corpus.keys()
                         if elements_by_id.get(e, {}).get("type") != "section_header"]

        for q in qs:
            q_tok, q_mask = encode_query(encoder, q["query"], device=device)
            q_tok = q_tok.to(device); q_mask = q_mask.to(device)

            # Initial scores (late interaction)
            scores = {}
            for eid in candidate_ids:
                e_tok, e_mask = corpus[eid]
                scores[eid] = _late_interaction(q_tok, q_mask, e_tok, e_mask)

            # For each inference variant, optionally apply graph_propagate
            for var_name, var_cfg in inference_variants.items():
                if var_cfg.get("graph_propagate"):
                    # Take top-N candidates, run propagation on them
                    top_n = sorted(scores.items(), key=lambda x: -x[1])[:50]
                    top_ids = [x[0] for x in top_n]
                    top_scores = torch.tensor([x[1] for x in top_n], dtype=torch.float32)
                    refined = graph_propagate(
                        top_scores,
                        top_ids,
                        graphs[doc_id],
                        alpha=var_cfg["alpha"],
                        T=var_cfg["T"],
                    )
                    refined_scores = dict(zip(top_ids, refined.tolist()))
                    # Re-rank top-50, append non-top with original scores
                    ranked = sorted(refined_scores.items(), key=lambda x: -x[1])
                    other_ids = [e for e in candidate_ids if e not in refined_scores]
                    other_ids.sort(key=lambda e: -scores[e])
                    ranked_ids = [x[0] for x in ranked] + other_ids
                else:
                    ranked_ids = sorted(scores.keys(), key=lambda e: -scores[e])

                m = compute_metrics_for_query(
                    ranked_ids, q["gt_element_ids"], q["gt_pages"], elements_by_id
                )
                for k, v in m.items():
                    aggregates[var_name][k].append(v)

            n_done += 1

        if (doc_idx + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{doc_idx+1}/{len(queries_by_doc)} docs, {n_done}/{len(queries)} queries] "
                  f"{elapsed:.0f}s")

    # Aggregate (mean)
    out: dict[str, dict] = {}
    for var_name, per_q in aggregates.items():
        out[var_name] = {k: sum(v) / max(1, len(v)) for k, v in per_q.items()}
        out[var_name]["n_queries"] = len(next(iter(per_q.values()))) if per_q else 0

    print(f"  done in {time.time() - t0:.0f}s")
    return out


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="encoder state_dict path; omit for zero-shot")
    ap.add_argument("--hf_id", type=str, default="google/siglip2-base-patch16-224")
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--datasets", nargs="+", default=list(EVAL_DATASETS.keys()),
                    choices=list(EVAL_DATASETS.keys()))
    ap.add_argument("--variants", nargs="+", default=list(INFERENCE_VARIANTS.keys()),
                    choices=list(INFERENCE_VARIANTS.keys()))
    ap.add_argument("--max_queries", type=int, default=None,
                    help="cap on queries per dataset (debug)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    print(f"device: {args.device}")
    encoder = load_encoder(args)

    inference_variants = {v: INFERENCE_VARIANTS[v] for v in args.variants}
    results: dict[str, dict] = {}
    for ds_id in args.datasets:
        results[ds_id] = eval_dataset(encoder, EVAL_DATASETS[ds_id], inference_variants, args)

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "datasets": args.datasets, "variants": args.variants,
        "results": results,
    }, indent=2))
    print(f"\nsaved → {out_path}")

    # Pretty print
    print("\n=== Summary ===")
    for ds_id, by_variant in results.items():
        print(f"\n{EVAL_DATASETS[ds_id]['name']}:")
        for var_name, metrics in by_variant.items():
            print(f"  [{var_name}] " + "  ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
                if k in ("recall@10", "hit@10", "mrr", "coverage@10", "perfect@10", "cross_page_hit")
            ))


if __name__ == "__main__":
    main()
