"""Motivation experiment — quantify modality gap of off-the-shelf embedding models.

Claim: vanilla multimodal embedding models cannot adequately retrieve visual
elements when queries are mixed with text elements in the pool. This experiment
exposes the gap with no normalization tricks (no RRF) and no fine-tuning.

Four measurements:
  M-1. Per-GT-modality bucket recall — bucket queries by GT modality (text-only,
       visual-only, mixed) and report Hit@k / Recall@k per bucket.
  M-2. GT-element rank by modality — avg rank of first text GT vs first visual GT.
  M-3. Top-10 modality composition — fraction of text vs visual in top-10,
       compared to corpus modality ratio.
  M-4. Geometric modality gap — avg cosine similarity within text, within visual,
       and across modalities (Liang et al. NeurIPS 2022 style).

Encoders supported via `Encoder` abstraction. Currently:
  - siglipv2-base (google/siglip2-base-patch16-224)
  - clip-l-14    (openai/clip-vit-large-patch14)
  - jina-clip-v2 (jinaai/jina-clip-v2)
  - bge-vl       (BAAI/BGE-VL-base)
  - gme-qwen2-vl (Alibaba-NLP/gme-Qwen2-VL-2B-Instruct)

Usage:
    python eval/analysis/motivation_modality_gap.py \\
        --dataset sciegqa --encoder siglipv2-base \\
        --out eval/results/motivation/sciegqa_siglipv2.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval.datasets.loaders import load_mmdocir, load_sciegqa, load_spiqa_enhanced


# ──────────────────────────────────────────────────────────────────────────────
# Encoder abstraction
# ──────────────────────────────────────────────────────────────────────────────

class Encoder:
    """Unified encode_text / encode_images returning (N, D) L2-normalized tensors."""
    name: str
    dim: int

    def encode_text(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        raise NotImplementedError

    def encode_images(self, images: list[Image.Image], batch_size: int = 16) -> torch.Tensor:
        raise NotImplementedError


class SigLIPLikeEncoder(Encoder):
    """SigLIPv2, CLIP — same HF AutoModel API with get_text_features/get_image_features."""

    def __init__(self, name: str, hf_id: str, device: str = "cuda", max_text_len: int = 64):
        from transformers import AutoModel, AutoProcessor
        self.name = name
        self.device = device
        self.max_text_len = max_text_len
        print(f"  loading {hf_id}...")
        self.model = AutoModel.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        # Probe dim from a tiny forward
        with torch.no_grad():
            _ = self.processor(text=["test"], padding="max_length", truncation=True,
                                max_length=8, return_tensors="pt").to(device)
            self.dim = self.model.get_text_features(**_).shape[-1]
        print(f"  loaded {hf_id} (dim={self.dim})")

    @torch.no_grad()
    def encode_text(self, texts, batch_size=64):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, padding="max_length", truncation=True,
                                     max_length=self.max_text_len, return_tensors="pt").to(self.device)
            out = self.model.get_text_features(**inputs)
            embs.append(F.normalize(out, dim=-1).cpu())
        return torch.cat(embs)

    @torch.no_grad()
    def encode_images(self, images, batch_size=16):
        embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            imgs = [img.convert("RGB") if img.mode != "RGB" else img for img in batch]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            embs.append(F.normalize(out, dim=-1).cpu())
        return torch.cat(embs)


class RawQwen2VLEncoder(Encoder):
    """Raw Qwen2-VL-2B-Instruct without retrieval fine-tuning.

    Same backbone as GME but no contrastive retrieval tuning. Tests whether
    the modality-gap-closing is from MLLM architecture or from GME's retrieval-FT.

    Embedding: last hidden state of the last (EOS) token, L2-normalized.
    """

    def __init__(self, name: str, hf_id: str = "Qwen/Qwen2-VL-2B-Instruct",
                 device: str = "cuda"):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.name = name
        self.device = device
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        print(f"  loading {hf_id} (dtype={dtype})...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(hf_id)
        # Probe dim
        self.dim = self.model.config.hidden_size
        print(f"  loaded {hf_id} (dim={self.dim})")

    @torch.no_grad()
    def _encode_one(self, text: str | None = None, image: Image.Image | None = None) -> torch.Tensor:
        """Encode a single (text, image) item. At least one must be provided."""
        # Build chat-template prompt as Qwen2-VL expects
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if text is not None:
            content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[prompt],
            images=[image] if image is not None else None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = out.hidden_states[-1]  # (1, seq_len, dim)
        # Last token embedding
        emb = last_hidden[0, -1, :]  # (dim,)
        return F.normalize(emb.float().unsqueeze(0), dim=-1).cpu().squeeze(0)

    @torch.no_grad()
    def encode_text(self, texts, batch_size=1):
        embs = [self._encode_one(text=t).unsqueeze(0) for t in texts]
        return torch.cat(embs, dim=0)

    @torch.no_grad()
    def encode_images(self, images, batch_size=1):
        embs = []
        for img in images:
            img = img.convert("RGB") if img.mode != "RGB" else img
            # Qwen2-VL requires min 28×28; resize tiny crops up
            min_size = 56
            if img.width < min_size or img.height < min_size:
                scale = min_size / min(img.width, img.height)
                new_w = max(min_size, int(img.width * scale))
                new_h = max(min_size, int(img.height * scale))
                img = img.resize((new_w, new_h), Image.BICUBIC)
            embs.append(self._encode_one(image=img).unsqueeze(0))
        return torch.cat(embs, dim=0)


class GMEQwen2VLEncoder(Encoder):
    """GME-Qwen2-VL — Qwen2-VL based retrieval-specialized embedding.

    API: model.get_text_embeddings(texts=...) and model.get_image_embeddings(images=...)
    """

    def __init__(self, name: str, hf_id: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                 device: str = "cuda"):
        from transformers import AutoModel
        self.name = name
        self.device = device
        # bf16 for VRAM efficiency — zero-shot inference only, not training
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        print(f"  loading {hf_id} (dtype={dtype})...")
        self.model = AutoModel.from_pretrained(hf_id, trust_remote_code=True,
                                                torch_dtype=dtype).to(device).eval()
        self.dim = self.model.config.hidden_size if hasattr(self.model.config, "hidden_size") else 1536
        print(f"  loaded {hf_id} (dim={self.dim})")

    @torch.no_grad()
    def encode_text(self, texts, batch_size=8):
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            e = self.model.get_text_embeddings(texts=batch, batch_size=len(batch))
            embs.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(embs)

    @torch.no_grad()
    def encode_images(self, images, batch_size=4):
        embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            imgs = [img.convert("RGB") if img.mode != "RGB" else img for img in batch]
            e = self.model.get_image_embeddings(images=imgs, batch_size=len(imgs))
            embs.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(embs)


class JinaCLIPEncoder(Encoder):
    """Jina-CLIP-v2 — uses trust_remote_code, has .encode_text / .encode_image."""

    def __init__(self, name: str, hf_id: str = "jinaai/jina-clip-v2", device: str = "cuda"):
        from transformers import AutoModel
        self.name = name
        self.device = device
        print(f"  loading {hf_id}...")
        self.model = AutoModel.from_pretrained(hf_id, trust_remote_code=True,
                                                torch_dtype=torch.float32).to(device).eval()
        self.dim = self.model.config.projection_dim if hasattr(self.model.config, "projection_dim") else 1024
        print(f"  loaded {hf_id} (dim={self.dim})")

    @torch.no_grad()
    def encode_text(self, texts, batch_size=64):
        # Jina CLIP v2 has built-in encode_text method
        embs = self.model.encode_text(texts, batch_size=batch_size, convert_to_tensor=True)
        return F.normalize(embs.float(), dim=-1).cpu()

    @torch.no_grad()
    def encode_images(self, images, batch_size=16):
        imgs = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        embs = self.model.encode_image(imgs, batch_size=batch_size, convert_to_tensor=True)
        return F.normalize(embs.float(), dim=-1).cpu()


def build_encoder(name: str, device: str = "cuda") -> Encoder:
    """Factory."""
    presets = {
        "siglipv2-base": ("google/siglip2-base-patch16-224",      SigLIPLikeEncoder),
        "clip-l-14":     ("openai/clip-vit-large-patch14",        SigLIPLikeEncoder),
        "jina-clip-v2":  ("jinaai/jina-clip-v2",                  JinaCLIPEncoder),
        "bge-vl-base":   ("BAAI/BGE-VL-base",                     SigLIPLikeEncoder),
        "bge-vl-large":  ("BAAI/BGE-VL-large",                    SigLIPLikeEncoder),
        "gme-qwen2-vl":  ("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", GMEQwen2VLEncoder),
        "qwen2-vl-2b":   ("Qwen/Qwen2-VL-2B-Instruct",            RawQwen2VLEncoder),
    }
    if name not in presets:
        raise ValueError(f"unknown encoder: {name}. Available: {list(presets)}")
    hf_id, cls = presets[name]
    # Encoders with non-positional kwarg API
    if cls in (JinaCLIPEncoder, GMEQwen2VLEncoder, RawQwen2VLEncoder):
        return cls(name, hf_id=hf_id, device=device)
    return cls(name, hf_id, device)


# ──────────────────────────────────────────────────────────────────────────────
# Measurements
# ──────────────────────────────────────────────────────────────────────────────

def is_visual(elem_type: str) -> bool:
    return elem_type != "text"


def bucket_query_by_gt_modality(query, elem_type_map: dict[str, str]) -> str:
    """Returns 'text-only', 'visual-only', or 'mixed' based on GT element types."""
    gt = query["gt_element_ids"]
    has_text = any(elem_type_map.get(g) == "text" for g in gt)
    has_visual = any(is_visual(elem_type_map.get(g, "text")) for g in gt)
    if has_text and has_visual: return "mixed"
    if has_visual: return "visual-only"
    return "text-only"


def measure_M1_M2_M3(sim: torch.Tensor, queries: list[dict], elem_ids: list[str],
                      elem_types: list[str], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Compute M-1 (bucket recall), M-2 (rank-by-modality), M-3 (top-10 composition)."""
    elem_type_map = dict(zip(elem_ids, elem_types))
    top_idx = sim.topk(max(ks), dim=-1).indices  # (Nq, max_k)
    top_idx_full = sim.argsort(descending=True, dim=-1)  # (Nq, N) for rank lookup

    # Per-bucket metrics
    buckets = {b: {f"Hit@{k}": [] for k in ks} for b in ["text-only", "visual-only", "mixed", "all"]}
    for b in buckets:
        for k in ks:
            buckets[b][f"Recall@{k}"] = []
        buckets[b]["count"] = 0
        buckets[b]["MRR"] = []

    # M-2: GT rank by modality
    rank_by_mod = {"text_GT": [], "visual_GT": []}

    # M-3: top-10 modality composition
    top10_text_frac = []
    top10_vis_frac = []

    for qi, q in enumerate(queries):
        gt = q["gt_element_ids"]
        bucket = bucket_query_by_gt_modality(q, elem_type_map)
        retrieved_full = [elem_ids[i] for i in top_idx_full[qi].tolist()]
        retrieved_topk = retrieved_full[:max(ks)]

        # M-1: bucket recall
        for b in [bucket, "all"]:
            buckets[b]["count"] += 1
            for k in ks:
                topk_set = set(retrieved_full[:k])
                inter = topk_set & gt
                buckets[b][f"Hit@{k}"].append(1.0 if inter else 0.0)
                buckets[b][f"Recall@{k}"].append(len(inter) / len(gt))
            # MRR
            mrr = 0.0
            for rank, eid in enumerate(retrieved_full, 1):
                if eid in gt:
                    mrr = 1.0 / rank
                    break
            buckets[b]["MRR"].append(mrr)

        # M-2: GT rank by modality
        first_text_rank, first_vis_rank = None, None
        for rank, eid in enumerate(retrieved_full, 1):
            if eid in gt:
                if elem_type_map.get(eid) == "text" and first_text_rank is None:
                    first_text_rank = rank
                elif is_visual(elem_type_map.get(eid, "text")) and first_vis_rank is None:
                    first_vis_rank = rank
                if first_text_rank and first_vis_rank: break
        if first_text_rank: rank_by_mod["text_GT"].append(first_text_rank)
        if first_vis_rank: rank_by_mod["visual_GT"].append(first_vis_rank)

        # M-3: top-10 composition
        top10_types = [elem_type_map.get(eid, "?") for eid in retrieved_full[:10]]
        n_text = sum(1 for t in top10_types if t == "text")
        n_vis = sum(1 for t in top10_types if is_visual(t))
        top10_text_frac.append(n_text / 10.0)
        top10_vis_frac.append(n_vis / 10.0)

    # Aggregate
    result = {"M1_bucket_metrics": {}, "M2_rank_by_modality": {}, "M3_top10_composition": {}}
    for b, vals in buckets.items():
        result["M1_bucket_metrics"][b] = {
            "n_queries": vals["count"],
            **{m: float(np.mean(vals[m])) if vals[m] else 0.0
                for m in vals if m != "count"}
        }
    result["M2_rank_by_modality"] = {
        "avg_rank_first_text_GT": float(np.mean(rank_by_mod["text_GT"])) if rank_by_mod["text_GT"] else None,
        "avg_rank_first_visual_GT": float(np.mean(rank_by_mod["visual_GT"])) if rank_by_mod["visual_GT"] else None,
        "n_queries_with_text_GT": len(rank_by_mod["text_GT"]),
        "n_queries_with_visual_GT": len(rank_by_mod["visual_GT"]),
    }
    result["M3_top10_composition"] = {
        "avg_text_frac": float(np.mean(top10_text_frac)),
        "avg_visual_frac": float(np.mean(top10_vis_frac)),
    }
    return result


def measure_M4_geometric_gap(text_emb: torch.Tensor, visual_emb: torch.Tensor,
                              n_sample: int = 2000) -> dict:
    """Avg cosine within text, within visual, across modalities.
    Uses a random sample for tractability on large corpora.
    """
    rng = np.random.default_rng(42)

    def sample_pair_sim(a: torch.Tensor, b: torch.Tensor, n: int) -> float:
        idx_a = rng.choice(len(a), size=min(n, len(a)), replace=False)
        idx_b = rng.choice(len(b), size=min(n, len(b)), replace=False)
        a_s = a[idx_a]
        b_s = b[idx_b]
        sim = (a_s @ b_s.T).flatten()
        return float(sim.mean().item())

    return {
        "text_text_avg_cos":   sample_pair_sim(text_emb, text_emb, n_sample) if len(text_emb) > 1 else None,
        "visual_visual_avg_cos": sample_pair_sim(visual_emb, visual_emb, n_sample) if len(visual_emb) > 1 else None,
        "text_visual_avg_cos": sample_pair_sim(text_emb, visual_emb, n_sample) if len(text_emb) and len(visual_emb) else None,
        "n_text": len(text_emb),
        "n_visual": len(visual_emb),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sciegqa", "mmdocir", "spiqa"], required=True)
    ap.add_argument("--encoder", required=True,
                    help="siglipv2-base | clip-l-14 | jina-clip-v2 | bge-vl | gme-qwen2-vl")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--restrict-to-paper", action="store_true", default=True)
    ap.add_argument("--max-text-len", type=int, default=64)
    args = ap.parse_args()

    print(f"\n{'='*70}\nMotivation experiment: {args.dataset} × {args.encoder}\n{'='*70}")

    # Load dataset
    loader = {"sciegqa": load_sciegqa, "mmdocir": load_mmdocir, "spiqa": load_spiqa_enhanced}[args.dataset]
    print(f"\nLoading {args.dataset}...")
    t0 = time.time()
    elements, queries = loader()
    print(f"  {len(elements)} elements, {len(queries)} queries  [{time.time()-t0:.1f}s]")

    text_idx = [i for i, e in enumerate(elements) if e["type"] == "text"]
    visual_idx = [i for i, e in enumerate(elements) if e["type"] != "text"]
    print(f"  text elements: {len(text_idx)}, visual: {len(visual_idx)}")

    # Bucket query stats
    elem_type_map = {e["id"]: e["type"] for e in elements}
    bucket_counts = {"text-only": 0, "visual-only": 0, "mixed": 0}
    for q in queries:
        bucket_counts[bucket_query_by_gt_modality(q, elem_type_map)] += 1
    print(f"  query GT buckets: {bucket_counts}")

    # Build encoder
    print(f"\nBuilding encoder: {args.encoder}")
    encoder = build_encoder(args.encoder, device=args.device)

    # Encode
    print(f"\nEncoding visual elements ({len(visual_idx)})...")
    t0 = time.time()
    vis_imgs = [elements[i]["image_loader"]() for i in visual_idx]
    visual_emb = encoder.encode_images(vis_imgs)
    print(f"  done [{time.time()-t0:.1f}s]")

    print(f"\nEncoding text elements ({len(text_idx)})...")
    t0 = time.time()
    text_strs = [elements[i]["text"][:512] for i in text_idx]
    text_emb = encoder.encode_text(text_strs)
    print(f"  done [{time.time()-t0:.1f}s]")

    print(f"\nEncoding queries ({len(queries)})...")
    t0 = time.time()
    q_emb = encoder.encode_text([q["query"] for q in queries])
    print(f"  done [{time.time()-t0:.1f}s]")

    # Build full corpus tensor in original order
    D = visual_emb.shape[1]
    corpus = torch.zeros(len(elements), D)
    for k, i in enumerate(visual_idx): corpus[i] = visual_emb[k]
    for k, i in enumerate(text_idx): corpus[i] = text_emb[k]

    # Similarity (no RRF, no normalization — we want to expose the gap)
    sim = q_emb @ corpus.T  # (Nq, N)

    # Per-paper restriction
    if args.restrict_to_paper:
        paper_d = np.array([e["doc_id"] for e in elements])
        paper_q = np.array([q["doc_id"] for q in queries])
        for qi, pq in enumerate(paper_q):
            sim[qi, paper_d != pq] = -1e9

    elem_ids = [e["id"] for e in elements]
    elem_types = [e["type"] for e in elements]

    # M-1, M-2, M-3
    print("\nComputing M-1 (bucket recall), M-2 (rank by modality), M-3 (top-10 composition)...")
    m123 = measure_M1_M2_M3(sim, queries, elem_ids, elem_types)

    # M-4
    print("Computing M-4 (geometric modality gap)...")
    m4 = measure_M4_geometric_gap(text_emb, visual_emb)

    # Combined output
    result = {
        "encoder": args.encoder,
        "dataset": args.dataset,
        "n_elements": len(elements),
        "n_queries": len(queries),
        "query_buckets": bucket_counts,
        "corpus_modality": {"text_frac": len(text_idx)/len(elements),
                             "visual_frac": len(visual_idx)/len(elements)},
        "restrict_to_paper": args.restrict_to_paper,
        "embedding_dim": int(D),
        **m123,
        "M4_geometric_gap": m4,
    }

    # Summary print
    print(f"\n{'─'*70}\nResults:\n{'─'*70}")
    print(f"\n[M-1] Per-bucket Hit@10:")
    for b in ["text-only", "visual-only", "mixed", "all"]:
        v = m123["M1_bucket_metrics"].get(b, {})
        if v.get("n_queries", 0) > 0:
            print(f"  {b:>12}  n={v['n_queries']:>4}  Hit@10={v['Hit@10']:.4f}  R@10={v['Recall@10']:.4f}  MRR={v['MRR']:.4f}")

    print(f"\n[M-2] Avg rank of first GT element (lower = better):")
    print(f"  text GT  : rank {m123['M2_rank_by_modality']['avg_rank_first_text_GT']}")
    print(f"  visual GT: rank {m123['M2_rank_by_modality']['avg_rank_first_visual_GT']}")

    print(f"\n[M-3] Top-10 composition vs corpus:")
    print(f"  text frac  : top-10 {m123['M3_top10_composition']['avg_text_frac']:.3f}  vs corpus {result['corpus_modality']['text_frac']:.3f}")
    print(f"  visual frac: top-10 {m123['M3_top10_composition']['avg_visual_frac']:.3f}  vs corpus {result['corpus_modality']['visual_frac']:.3f}")

    print(f"\n[M-4] Geometric gap:")
    print(f"  text-text avg cos    : {m4['text_text_avg_cos']:.4f}")
    print(f"  visual-visual avg cos: {m4['visual_visual_avg_cos']:.4f}")
    print(f"  text-visual avg cos  : {m4['text_visual_avg_cos']:.4f}")
    print(f"  gap (intra - cross)  : {(m4['text_text_avg_cos'] + m4['visual_visual_avg_cos'])/2 - m4['text_visual_avg_cos']:.4f}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
