"""B2: SigLIP element zero-shot retrieval (no PE / no graph).

Floor baseline corresponding to row (a) of the C2 ablation: each element is
encoded by SigLIP (image for figures/tables, text for captions/text), and a
text query retrieves by cosine similarity over the whole corpus.

Run:
    python -m eval.baselines.siglip_elem --dataset spiqa --split test-A \
        --model google/siglip-base-patch16-224
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def load_spiqa_test_a(root: Path, include_text: bool = False):
    """Returns:
        elements: list[dict] with {id, paper_id, type, ...}
            - type='figure'/'table': has image_path + caption
            - type='text':           has text (paragraph string)
        queries:  list[dict] with {qid, paper_id, question, gt_figure_id}
    """
    json_path = root / "test-A" / "SPIQA_testA.json"
    image_root = root / "test-A" / "images_224px" / "SPIQA_testA_Images_224px"
    data = json.loads(json_path.read_text())

    elements: list[dict] = []
    # Visual elements (figures + tables)
    for paper_id, paper in data.items():
        for fname, meta in paper["all_figures"].items():
            elements.append({
                "id": fname,
                "paper_id": paper_id,
                "image_path": str(image_root / paper_id / fname),
                "caption": meta.get("caption", ""),
                "type": meta.get("content_type", "figure"),    # 'figure' | 'table'
            })

    # Text paragraphs
    if include_text:
        para_root = (root / "test-A" / "paragraphs"
                     / "SPIQA_train_val_test-A_extracted_paragraphs")
        n_text = 0
        for paper_id in data.keys():
            txt_path = para_root / f"{paper_id}.txt"
            if not txt_path.exists():
                continue
            content = txt_path.read_text()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for i, p in enumerate(paragraphs):
                elements.append({
                    "id": f"{paper_id}__para{i:03d}",
                    "paper_id": paper_id,
                    "text": p,
                    "type": "text",
                })
                n_text += 1
        print(f"  added {n_text} text paragraphs")

    queries: list[dict] = []
    qid = 0
    for paper_id, paper in data.items():
        for q in paper["qa"]:
            queries.append({
                "qid": qid,
                "paper_id": paper_id,
                "question": q["question"],
                "gt_figure_id": q["reference"],
            })
            qid += 1
    return elements, queries


def _as_tensor(out):
    """Handle transformers v5 returning BaseModelOutputWithPooling vs old tensor API."""
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


def encode_images(elements: list[dict], model, processor, device: str, batch_size: int = 32):
    """Returns (N, d) normalized image embeddings aligned with `elements`."""
    embs = []
    for i in tqdm(range(0, len(elements), batch_size), desc="enc image"):
        batch = elements[i : i + batch_size]
        imgs = [Image.open(e["image_path"]).convert("RGB") for e in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            v = _as_tensor(model.get_image_features(**inputs))
        v = F.normalize(v, dim=-1)
        embs.append(v.cpu())
    return torch.cat(embs, dim=0)


def encode_texts(texts: list[str], model, processor, device: str, batch_size: int = 64):
    """Returns (N, d) normalized text embeddings."""
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="enc text"):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt",
                           padding="max_length", truncation=True).to(device)
        with torch.no_grad():
            v = _as_tensor(model.get_text_features(**inputs))
        v = F.normalize(v, dim=-1)
        embs.append(v.cpu())
    return torch.cat(embs, dim=0)


def normalize_per_modality(sim: torch.Tensor, modality_groups: list[list[int]],
                            method: str = "none", k_rrf: int = 60) -> torch.Tensor:
    """Normalize/merge scores across modalities.

    Args:
        sim: (Nq, Nd) raw cosine similarity
        modality_groups: list of column-index lists, one per modality
        method: 'none' (raw cosine), 'rrf', or 'zscore'

    Returns:
        (Nq, Nd) tensor of normalized scores (higher = better, comparable across modalities).
    """
    if method == "none" or len(modality_groups) <= 1:
        return sim

    if method == "rrf":
        merged = torch.zeros_like(sim)
        for indices in modality_groups:
            if not indices:
                continue
            idx = torch.tensor(indices, device=sim.device)
            sub = sim.index_select(1, idx)               # (Nq, |group|)
            # Rank within this modality (1 = highest score)
            order = sub.argsort(dim=-1, descending=True)
            ranks = order.argsort(dim=-1) + 1            # (Nq, |group|)
            rrf = 1.0 / (k_rrf + ranks.float())
            merged.index_copy_(1, idx, rrf)
        return merged

    if method == "zscore":
        merged = torch.zeros_like(sim)
        for indices in modality_groups:
            if not indices:
                continue
            idx = torch.tensor(indices, device=sim.device)
            sub = sim.index_select(1, idx)
            mu = sub.mean(dim=-1, keepdim=True)
            sd = sub.std(dim=-1, keepdim=True) + 1e-8
            z = (sub - mu) / sd
            merged.index_copy_(1, idx, z)
        return merged

    raise ValueError(f"Unknown norm method: {method}")


def retrieve_topk(query_embs: torch.Tensor, doc_embs: torch.Tensor,
                  paper_ids_q: list[str], paper_ids_d: list[str],
                  elements: list[dict] | None = None,
                  k: int = 10, restrict_to_paper: bool = False,
                  norm: str = "none") -> torch.Tensor:
    """Returns (N_q, k) indices into doc_embs.

    norm ∈ {'none', 'rrf', 'zscore'} — score normalization across encoder modalities
    (text vs visual = figure/table). Applied AFTER paper restriction.
    """
    sim = query_embs @ doc_embs.T  # (Nq, Nd) — unmasked

    # Normalize FIRST (using corpus-wide stats — paper restriction comes after)
    if norm != "none" and elements is not None:
        text_idx = [i for i, e in enumerate(elements) if e["type"] == "text"]
        visual_idx = [i for i, e in enumerate(elements) if e["type"] != "text"]
        sim = normalize_per_modality(sim, [text_idx, visual_idx], method=norm)

    # Then apply per-paper restriction
    if restrict_to_paper:
        mask = torch.zeros_like(sim, dtype=torch.bool)
        for qi, pq in enumerate(paper_ids_q):
            for di, pd in enumerate(paper_ids_d):
                if pd == pq:
                    mask[qi, di] = True
        sim = sim.masked_fill(~mask, float("-inf"))

    return sim.topk(k, dim=-1).indices


def evaluate(top_indices: torch.Tensor, queries: list[dict],
             elements: list[dict], ks=(1, 3, 5, 10)) -> dict:
    """Returns Recall@k, MRR, and pool composition diagnostics."""
    elem_ids = [e["id"] for e in elements]
    elem_types = [e["type"] for e in elements]
    out = {f"R@{k}": 0.0 for k in ks}
    out["MRR"] = 0.0

    # Diagnostics: how many of top-10 are visual vs text
    type_counts: dict[str, int] = {}

    for qi, q in enumerate(queries):
        gt = q["gt_figure_id"]
        retrieved_idx = top_indices[qi].tolist()
        retrieved = [elem_ids[i] for i in retrieved_idx]

        for idx in retrieved_idx:
            t = elem_types[idx]
            type_counts[t] = type_counts.get(t, 0) + 1

        for k in ks:
            if gt in retrieved[:k]:
                out[f"R@{k}"] += 1
        try:
            rank = retrieved.index(gt) + 1
            out["MRR"] += 1.0 / rank
        except ValueError:
            pass

    n = len(queries)
    metrics = {k: v / n for k, v in out.items()}
    metrics["_pool_top10_composition"] = {t: c / (n * 10) for t, c in type_counts.items()}
    return metrics


def evaluate_full_rank(query_embs, doc_embs, queries, elements):
    """Full-rank diagnostic: rank of GT figure across the entire pool."""
    sim = query_embs @ doc_embs.T
    elem_ids = [e["id"] for e in elements]
    n_total = len(elements)

    ranks = []
    visual_only_ranks = []
    visual_indices = torch.tensor([i for i, e in enumerate(elements) if e["type"] != "text"])
    visual_id_set = {elem_ids[i] for i in visual_indices.tolist()}

    for qi, q in enumerate(queries):
        gt = q["gt_figure_id"]
        if gt not in elem_ids:
            continue
        gt_idx = elem_ids.index(gt)
        order = sim[qi].argsort(descending=True)
        rank = (order == gt_idx).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

        # Rank within visual-only subset
        v_sim = sim[qi, visual_indices]
        v_order = v_sim.argsort(descending=True)
        v_gt_pos = visual_indices.tolist().index(gt_idx)
        v_rank = (v_order == v_gt_pos).nonzero(as_tuple=True)[0].item() + 1
        visual_only_ranks.append(v_rank)

    import statistics as st
    return {
        "median_rank_full_pool": st.median(ranks),
        "median_rank_visual_only": st.median(visual_only_ranks),
        "rank_at_top1pct_full": sum(1 for r in ranks if r <= n_total / 100) / len(ranks),
        "n_visual": len(visual_indices),
        "n_total": n_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="spiqa")
    parser.add_argument("--split", default="test-A")
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--mode", choices=["image", "caption", "image+caption"], default="image")
    parser.add_argument("--restrict-to-paper", action="store_true",
                        help="Restrict retrieval to same paper as the query (per-doc setting)")
    parser.add_argument("--include-text", action="store_true",
                        help="Add SPIQA extracted paragraphs to the pool (mixed: figure+table+text)")
    parser.add_argument("--norm", choices=["none", "rrf", "zscore"], default="none",
                        help="Score normalization across modalities (text vs visual)")
    parser.add_argument("--lora_adapter", default=None,
                        help="Optional path to a C3-FT'd LoRA adapter directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path("data/benchmarks") / args.dataset
    elements, queries = load_spiqa_test_a(root, include_text=args.include_text)
    print(f"loaded {len(elements)} elements ({sum(1 for e in elements if e['type']=='text')} text, "
          f"{sum(1 for e in elements if e['type']!='text')} visual), {len(queries)} queries")

    print(f"loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    if args.lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter).to(args.device).eval()
        print(f"  loaded LoRA adapter from {args.lora_adapter}")

    # Split visual vs text elements; encode each appropriately, then concat in original order.
    visual_idx = [i for i, e in enumerate(elements) if e["type"] != "text"]
    text_idx   = [i for i, e in enumerate(elements) if e["type"] == "text"]

    visual = [elements[i] for i in visual_idx]
    if visual:
        if args.mode in ("image", "image+caption"):
            img_emb_v = encode_images(visual, model, processor, args.device)
        if args.mode in ("caption", "image+caption"):
            cap_emb_v = encode_texts([e["caption"] for e in visual], model, processor, args.device)

        if args.mode == "image":
            visual_emb = img_emb_v
        elif args.mode == "caption":
            visual_emb = cap_emb_v
        else:  # image+caption: average then re-normalize
            visual_emb = F.normalize(img_emb_v + cap_emb_v, dim=-1)
    else:
        visual_emb = torch.empty(0, model.config.projection_dim if hasattr(model.config, "projection_dim") else 768)

    if text_idx:
        text_emb = encode_texts([elements[i]["text"] for i in text_idx],
                                model, processor, args.device)
    else:
        text_emb = torch.empty(0, visual_emb.shape[1] if visual_emb.numel() else 768)

    # Reassemble in original element order
    d = visual_emb.shape[1] if visual_emb.numel() else text_emb.shape[1]
    doc_emb = torch.zeros(len(elements), d)
    for k, i in enumerate(visual_idx):
        doc_emb[i] = visual_emb[k]
    for k, i in enumerate(text_idx):
        doc_emb[i] = text_emb[k]

    q_emb = encode_texts([q["question"] for q in queries], model, processor, args.device)

    paper_ids_d = [e["paper_id"] for e in elements]
    paper_ids_q = [q["paper_id"] for q in queries]
    top_idx = retrieve_topk(q_emb, doc_emb, paper_ids_q, paper_ids_d,
                            elements=elements,
                            k=10, restrict_to_paper=args.restrict_to_paper,
                            norm=args.norm)

    metrics = evaluate(top_idx, queries, elements)
    diag = evaluate_full_rank(q_emb, doc_emb, queries, elements)

    print("\n=== Results ===")
    print(f"  model: {args.model}")
    print(f"  mode: {args.mode}, restrict_to_paper={args.restrict_to_paper}, include_text={args.include_text}, norm={args.norm}")
    for k, v in metrics.items():
        if k.startswith("_"):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")
    print(f"  --- diagnostics ---")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # Save
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace("/", "__")
    restrict_tag = "perpaper" if args.restrict_to_paper else "corpus"
    pool_tag = "mixed" if args.include_text else "visual"
    ft_tag = ""
    if args.lora_adapter:
        ft_tag = "_ft-" + args.lora_adapter.rstrip("/").split("/")[-1]
    out_file = (out_dir / f"baseline_{args.dataset}_{args.split}_"
                f"{model_tag}_{args.mode}_{restrict_tag}_{pool_tag}_{args.norm}{ft_tag}.json")
    out_file.write_text(json.dumps({
        "args": vars(args),
        "metrics": metrics,
        "n_elements": len(elements),
        "n_queries": len(queries),
    }, indent=2))
    print(f"  saved to {out_file}")


if __name__ == "__main__":
    main()
