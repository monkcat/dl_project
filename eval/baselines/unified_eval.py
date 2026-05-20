"""Unified element-level retrieval evaluator for MMDocIR and SciEGQA.

Encodes:
  - text elements with SigLIPv2 text encoder
  - visual elements (image/table/equation/drawing) with SigLIPv2 image encoder
Computes per-modality RRF normalization to combat modality gap, optionally
restricts to same doc.

Multi-GT aware metrics:
  - Recall@k = |top-k ∩ GT| / |GT| (averaged over queries)
  - Hit@k    = 1 if any GT in top-k, else 0
  - MRR      = 1 / (rank of first GT)

Usage:
    python -m eval.baselines.unified_eval --dataset mmdocir
    python -m eval.baselines.unified_eval --dataset sciegqa --lora_adapter runs/c3_spiqa
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from eval.datasets.loaders import load_mmdocir, load_sciegqa, load_spiqa_enhanced


def _as_tensor(out):
    if hasattr(out, "last_hidden_state"):
        return out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
    return out


@torch.no_grad()
def encode_images(elements, model, processor, device, batch_size=32):
    embs = []
    for i in range(0, len(elements), batch_size):
        batch = elements[i:i+batch_size]
        imgs = []
        for e in batch:
            img = e["image_loader"]()
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs.append(img)
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        embs.append(F.normalize(out, dim=-1).cpu())
    return torch.cat(embs)


@torch.no_grad()
def encode_texts(texts, model, processor, device, batch_size=64, max_length=64):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = processor(text=batch, padding="max_length", truncation=True,
                           max_length=max_length, return_tensors="pt").to(device)
        out = model.get_text_features(**inputs)
        embs.append(F.normalize(out, dim=-1).cpu())
    return torch.cat(embs)


def normalize_per_modality(sim: torch.Tensor, groups: list[list[int]], method: str = "rrf"):
    """Apply RRF (per-query) within each modality group."""
    out = torch.zeros_like(sim)
    if method == "rrf":
        for group in groups:
            if not group: continue
            g_sim = sim[:, group]                          # (Nq, |G|)
            ranks = g_sim.argsort(descending=True, dim=-1).argsort(dim=-1) + 1
            out[:, group] = 1.0 / (60.0 + ranks.float())
        return out
    return sim


def evaluate(top_indices, queries, elem_ids, ks=(1, 3, 5, 10)):
    out = {f"R@{k}": 0.0 for k in ks}
    out.update({f"Hit@{k}": 0.0 for k in ks})
    out["MRR"] = 0.0
    n = len(queries)
    for qi, q in enumerate(queries):
        gt = q["gt_element_ids"]
        retrieved_idx = top_indices[qi].tolist()
        retrieved = [elem_ids[i] for i in retrieved_idx]
        # Recall@k and Hit@k
        for k in ks:
            topk_set = set(retrieved[:k])
            inter = topk_set & gt
            out[f"R@{k}"] += len(inter) / len(gt)
            out[f"Hit@{k}"] += 1.0 if inter else 0.0
        # MRR
        for rank, eid in enumerate(retrieved, 1):
            if eid in gt:
                out["MRR"] += 1.0 / rank
                break
    return {k: v / n for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mmdocir", "sciegqa", "spiqa"], required=True)
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--lora_adapter", default=None)
    ap.add_argument("--norm", choices=["none", "rrf"], default="rrf")
    ap.add_argument("--restrict-to-paper", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-text-tokens", type=int, default=64)
    args = ap.parse_args()

    loader_fn = {"mmdocir": load_mmdocir, "sciegqa": load_sciegqa, "spiqa": load_spiqa_enhanced}[args.dataset]
    print(f"Loading {args.dataset}...")
    t0 = time.time()
    elements, queries = loader_fn()
    print(f"  {len(elements)} elements, {len(queries)} queries  [{time.time()-t0:.1f}s]")

    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    if args.lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter).to(args.device).eval()
        print(f"  LoRA adapter: {args.lora_adapter}")

    # Encode visual elements (image_loader is set)
    text_idx = [i for i, e in enumerate(elements) if e["type"] == "text"]
    visual_idx = [i for i, e in enumerate(elements) if e["type"] != "text"]
    print(f"  text elements: {len(text_idx)}, visual: {len(visual_idx)}")

    print("Encoding visual elements...")
    t0 = time.time()
    vis_elems = [elements[i] for i in visual_idx]
    visual_emb = encode_images(vis_elems, model, processor, args.device, args.batch_size)
    print(f"  done [{time.time()-t0:.1f}s]")

    print("Encoding text elements...")
    t0 = time.time()
    text_emb = encode_texts([elements[i]["text"][:512] for i in text_idx],
                             model, processor, args.device, max_length=args.max_text_tokens)
    print(f"  done [{time.time()-t0:.1f}s]")

    # Build full corpus
    d = visual_emb.shape[1]
    corpus = torch.zeros(len(elements), d)
    for k, i in enumerate(visual_idx): corpus[i] = visual_emb[k]
    for k, i in enumerate(text_idx): corpus[i] = text_emb[k]

    print("Encoding queries...")
    q_emb = encode_texts([q["query"] for q in queries], model, processor, args.device,
                          max_length=args.max_text_tokens)

    sim = q_emb @ corpus.T  # (Nq, Nd)
    print(f"\n--- Eval (norm={args.norm}, restrict_to_paper={args.restrict_to_paper}) ---")

    if args.norm != "none":
        sim = normalize_per_modality(sim, [text_idx, visual_idx], method=args.norm)

    if args.restrict_to_paper:
        paper_d = [e["doc_id"] for e in elements]
        paper_q = [q["doc_id"] for q in queries]
        for qi, pq in enumerate(paper_q):
            for di, pd in enumerate(paper_d):
                if pd != pq:
                    sim[qi, di] = -1e9

    top = sim.topk(20, dim=-1).indices
    elem_ids = [e["id"] for e in elements]
    metrics = evaluate(top, queries, elem_ids)
    print(f"\nResults:")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")

    # Save
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.lora_adapter.split("/")[-1] if args.lora_adapter else "noft"
    fp = out_dir / f"{args.dataset}_{tag}_norm{args.norm}_perpaper{int(args.restrict_to_paper)}.json"
    json.dump({"args": vars(args), "metrics": metrics}, open(fp, "w"), indent=2, default=str)
    print(f"\nsaved to {fp}")


if __name__ == "__main__":
    main()
