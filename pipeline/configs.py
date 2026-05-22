"""Experiment configurations — full Tier 1+2+3 set from REPORT_KR §6.3-6.5 + Appendix B.

Tier 1 — Main ablation rows (REPORT_KR §6.3, rows a-h, gme=k):
  (a) baseline_infonce       LoRA + InfoNCE (binary 1-hop positives), no GPE
  (b) gpe_type_infonce       LoRA + InfoNCE + GPE (type only)
  (c) gpe_type_role_infonce  LoRA + InfoNCE + GPE (type+role)
  (d) gpe_full_infonce       LoRA + InfoNCE + GPE (all 4 facets)
  (e) grcl_no_gpe            LoRA + GRCL (graded), no GPE
  (f) grcl_gpe               LoRA + GRCL + GPE (all)
  (g) grcl_gpe_cov           LoRA + GRCL + GPE + L_cov (no L_cons)
  (h) full_method            LoRA + GRCL + GPE + L_cov + L_cons
  (gme) gme_zero_shot        GME-Qwen2-VL-2B zero-shot reference

Inference variants are applied to every row via `INFERENCE_VARIANTS`:
  no_prop      encoder only
  prop_a03_T2  + graph propagation α=0.3 T=2
This yields rows (i) = (a)+prop, (j) = (h)+prop, (l) = (gme)+prop automatically.

Tier 2 — Negative controls (REPORT_KR §6.4, rows m-p):
  (m) shuffled_role          (h) with section_role permuted within doc
  (n) random_role            (h) with section_role assigned uniform random
  (o) no_query_pe_dropout    (h) with query_pe_dropout=0 (train/infer mismatch)
  (p) encoder_clip_l14       (h) with SigLIPv2 swapped for CLIP-L/14

Tier 3 — Sub-ablations (REPORT_KR §6.5, all derive from (h)):
  γ sweep:   gamma_03, gamma_07         (γ=0.5 = h)
  λ_cov:     cov_0, cov_01, cov_05, cov_10   (λ_cov=0.3 = h)
  λ_cons:    cons_0, cons_01, cons_03, cons_10   (λ_cons=0.5 = h)
  LoRA rank: lora_r4, lora_r16, lora_r32   (r=8 = h)
  Edge iso:  edge_caption_of, edge_refer_to, edge_contains
  Token cnt: tokens_16, tokens_64           (default = 196)

Total training runs: 9 (Tier 1, excl gme not trained) + 4 (Tier 2) + 18 (Tier 3) - existing 4 done = 27.
Actually counting all unique training runs: 8 (a-h) + 4 (m,n,o,p) + (2+4+4+3+3+2) = 12 + 18 = 30 new training runs.
"""
from __future__ import annotations

# ───── Common hyperparameters shared by all FT runs ─────
COMMON = {
    "hf_id": "google/siglip2-base-patch16-224",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "proj_dim": 128,
    # Training
    "train_sample": 0,           # 0 = use all 25,459 SPIQA train papers
    "steps": 8000,
    "batch": 16,
    "pool": 12,
    "lr": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "tau": 0.07,
    "grad_clip": 1.0,
    "anchor_kind": "mixed",      # caption_of + refer_to + nl_qa
    "seed": 42,
    # GPE
    "gpe_alpha_init_raw": -3.0,
    "beta_init_raw": -3.0,
    "gpe_facets": "type,role,depth,pos",     # all facets
    # GRCL
    "gamma": 0.5,                # 2-hop decay
    # Eval cadence
    "eval_every": 500,
    "eval_cf_pairs": 200,
    "eval_recall_n": 200,
    # Negative-control flags (default: off)
    "query_pe_dropout": 0.5,
    "section_role_mode": "normal",
    "edge_types_only": None,
}


def _base_grcl_full():
    """The (h) preset — used as base for Tier 2 & 3 derivatives."""
    return {
        **COMMON,
        "use_gpe": True,
        "loss_type": "grcl",
        "lambda_cov": 0.3,
        "lambda_cons": 0.5,
    }


# ───── ROW CONFIGS ─────
ROW_CONFIGS: dict[str, dict] = {}


# Tier 1 — main ablation rows
ROW_CONFIGS["a"] = {
    **COMMON, "row_id": "a", "name": "baseline_infonce",
    "description": "LoRA + standard InfoNCE (binary 1-hop positives), no GPE",
    "use_gpe": False, "loss_type": "infonce", "infonce_threshold": 0.5,
    "lambda_cov": 0.0, "lambda_cons": 0.0,
}
ROW_CONFIGS["b"] = {
    **COMMON, "row_id": "b", "name": "gpe_type_infonce",
    "description": "LoRA + InfoNCE + GPE (type only)",
    "use_gpe": True, "loss_type": "infonce", "infonce_threshold": 0.5,
    "lambda_cov": 0.0, "lambda_cons": 0.0,
    "gpe_facets": "type",
}
ROW_CONFIGS["c"] = {
    **COMMON, "row_id": "c", "name": "gpe_type_role_infonce",
    "description": "LoRA + InfoNCE + GPE (type + role)",
    "use_gpe": True, "loss_type": "infonce", "infonce_threshold": 0.5,
    "lambda_cov": 0.0, "lambda_cons": 0.0,
    "gpe_facets": "type,role",
}
ROW_CONFIGS["d"] = {
    **COMMON, "row_id": "d", "name": "gpe_full_infonce",
    "description": "LoRA + InfoNCE + GPE (all 4 facets)",
    "use_gpe": True, "loss_type": "infonce", "infonce_threshold": 0.5,
    "lambda_cov": 0.0, "lambda_cons": 0.0,
}
ROW_CONFIGS["e"] = {
    **COMMON, "row_id": "e", "name": "grcl_no_gpe",
    "description": "LoRA + GRCL (graded graph-relevance), no GPE",
    "use_gpe": False, "loss_type": "grcl",
    "lambda_cov": 0.0, "lambda_cons": 0.0,
}
ROW_CONFIGS["f"] = {
    **COMMON, "row_id": "f", "name": "grcl_gpe",
    "description": "LoRA + GRCL + GPE (no L_cov, no L_cons)",
    "use_gpe": True, "loss_type": "grcl",
    "lambda_cov": 0.0, "lambda_cons": 0.0,
}
ROW_CONFIGS["g"] = {
    **COMMON, "row_id": "g", "name": "grcl_gpe_cov",
    "description": "LoRA + GRCL + GPE + L_cov (no L_cons)",
    "use_gpe": True, "loss_type": "grcl",
    "lambda_cov": 0.3, "lambda_cons": 0.0,
}
ROW_CONFIGS["h"] = {
    **COMMON, "row_id": "h", "name": "full_method",
    "description": "Full method: LoRA + GRCL + GPE + L_cov + L_cons",
    "use_gpe": True, "loss_type": "grcl",
    "lambda_cov": 0.3, "lambda_cons": 0.5,
}
ROW_CONFIGS["gme"] = {
    "row_id": "gme", "name": "gme_qwen2vl_zero_shot",
    "description": "GME-Qwen2-VL-2B zero-shot (no training); inference variants applied at eval time",
    "skip_training": True,
    "encoder_kind": "gme",
    "hf_id": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
}


# Tier 2 — Negative controls (derive from full method (h))
ROW_CONFIGS["m"] = {
    **_base_grcl_full(), "row_id": "m", "name": "shuffled_role",
    "description": "(h) with section_role permuted within each doc (PE-effect attribution)",
    "section_role_mode": "shuffled",
}
ROW_CONFIGS["n"] = {
    **_base_grcl_full(), "row_id": "n", "name": "random_role",
    "description": "(h) with section_role assigned uniform random per node",
    "section_role_mode": "random",
}
ROW_CONFIGS["o"] = {
    **_base_grcl_full(), "row_id": "o", "name": "no_query_pe_dropout",
    "description": "(h) without query-side PE dropout (train/inference mismatch)",
    "query_pe_dropout": 0.0,
}
ROW_CONFIGS["p"] = {
    **_base_grcl_full(), "row_id": "p", "name": "encoder_clip_l14",
    "description": "(h) with encoder swap: SigLIPv2 → CLIP-L/14 (encoder-agnostic test)",
    "hf_id": "openai/clip-vit-large-patch14",
}


# Tier 3 — Sub-ablation sweeps (all derive from (h) except varying ONE hyperparameter)
def _h_with(name: str, desc: str, **overrides) -> dict:
    return {**_base_grcl_full(), "row_id": name, "name": name, "description": desc, **overrides}


# γ sweep (γ=0.5 = h)
ROW_CONFIGS["gamma_03"] = _h_with(
    "gamma_03", "(h) with γ=0.3 (steeper 2-hop decay)",
    gamma=0.3,
)
ROW_CONFIGS["gamma_07"] = _h_with(
    "gamma_07", "(h) with γ=0.7 (flatter 2-hop decay)",
    gamma=0.7,
)

# λ_cov sweep (λ_cov=0.3 = h)
for lam in (0.0, 0.1, 0.5, 1.0):
    rid = f"cov_{int(lam*10):02d}"
    ROW_CONFIGS[rid] = _h_with(rid, f"(h) with λ_cov={lam}", lambda_cov=lam)

# λ_cons sweep (λ_cons=0.5 = h)
for lam in (0.0, 0.1, 0.3, 1.0):
    rid = f"cons_{int(lam*10):02d}"
    ROW_CONFIGS[rid] = _h_with(rid, f"(h) with λ_cons={lam}", lambda_cons=lam)

# LoRA rank sweep (r=8 = h)
for r in (4, 16, 32):
    rid = f"lora_r{r}"
    ROW_CONFIGS[rid] = _h_with(rid, f"(h) with LoRA rank={r}", lora_rank=r, lora_alpha=2 * r)

# Edge type isolation
for edge in ("caption_of", "refer_to", "contains"):
    rid = f"edge_{edge}"
    ROW_CONFIGS[rid] = _h_with(rid, f"(h) with only edges of type '{edge}' in GRCL",
                                edge_types_only=edge)

# Token count per visual (default = 196 = full patches for 224×224)
# NOTE: deferred — would require custom patch-pooling in element_encoder.forward_vision.
# Currently runs with default 196 tokens (no effect on results). Left here as a stub
# so the row id exists for the eventual sub-ablation.
# for ntok in (16, 64):
#     rid = f"tokens_{ntok:03d}"
#     ROW_CONFIGS[rid] = _h_with(rid, f"(h) with {ntok} tokens per visual element",
#                                 tokens_per_visual=ntok)


# ───── Inference-time evaluation variants ─────
# Each variant is applied to every row's checkpoint. (i)=(a)+prop, (j)=(h)+prop, (l)=(gme)+prop
# Propagation α / T sweep also exposed below for Tier 3 (eval-only, no retrain).
INFERENCE_VARIANTS: dict[str, dict] = {
    "no_prop":          {"graph_propagate": False, "alpha": 0.0, "T": 0},
    "prop_a03_T2":      {"graph_propagate": True,  "alpha": 0.3, "T": 2},  # default propagation
    # Propagation sub-ablation (eval-only on (h) checkpoint)
    "prop_a01_T2":      {"graph_propagate": True,  "alpha": 0.1, "T": 2},
    "prop_a05_T2":      {"graph_propagate": True,  "alpha": 0.5, "T": 2},
    "prop_a07_T2":      {"graph_propagate": True,  "alpha": 0.7, "T": 2},
    "prop_a03_T1":      {"graph_propagate": True,  "alpha": 0.3, "T": 1},
    "prop_a03_T3":      {"graph_propagate": True,  "alpha": 0.3, "T": 3},
}

# Variants ALWAYS run for every row (cheap, 12 cells extra per row)
DEFAULT_VARIANTS = ("no_prop", "prop_a03_T2")

# Variants run ONLY on (h) checkpoint (propagation α/T sweep, REPORT §6.5)
PROP_SWEEP_VARIANTS = ("prop_a01_T2", "prop_a05_T2", "prop_a07_T2",
                       "prop_a03_T1", "prop_a03_T3")


# ───── Evaluation datasets ─────
EVAL_DATASETS: dict[str, dict] = {
    "spiqa_testA": {
        "name": "SPIQA test-A",
        "graph":    "data/benchmarks/spiqa/test-A/element_graph_v2.json",
        "elements": "data/benchmarks/spiqa/test-A/elements_v2.jsonl",
        "queries":  "data/benchmarks/spiqa/test-A/SPIQA_testA.json",
        "query_format": "spiqa",
        "image_root": "data/benchmarks/spiqa/test-A/images_224px/SPIQA_testA_Images_224px",
    },
    "sciegqa": {
        "name": "SciEGQA",
        "graph":    "data/benchmarks/sciegqa/element_graph_v2.json",
        "elements": "data/benchmarks/sciegqa/elements_v2.jsonl",
        "queries":  "data/benchmarks/sciegqa/queries_with_gt_docling.jsonl",
        "query_format": "jsonl_gt_ids",
        "image_root": "data/benchmarks/sciegqa/Images",
    },
    "mmdocir": {
        "name": "MMDocIR",
        "graph":    "data/benchmarks/mmdocir/element_graph_v2.json",
        "elements": "data/benchmarks/mmdocir/elements_v2.jsonl",
        "queries":  "data/benchmarks/mmdocir/academic_queries.jsonl",
        "query_format": "jsonl_gt_ids",
        "image_root": None,
    },
}


# ───── Row groupings for the launcher ─────
TIER1_ROWS = ["a", "b", "c", "d", "e", "f", "g", "h"]
TIER1_NO_TRAIN = ["gme"]
TIER2_ROWS = ["m", "n", "o", "p"]
TIER3_ROWS = (
    ["gamma_03", "gamma_07"]
    + [f"cov_{int(l*10):02d}" for l in (0.0, 0.1, 0.5, 1.0)]
    + [f"cons_{int(l*10):02d}" for l in (0.0, 0.1, 0.3, 1.0)]
    + [f"lora_r{r}" for r in (4, 16, 32)]
    + [f"edge_{e}" for e in ("caption_of", "refer_to", "contains")]
    # tokens_per_visual ablation deferred (needs patch-pool in element_encoder)
)


def get_config(row_id: str) -> dict:
    if row_id not in ROW_CONFIGS:
        raise KeyError(f"unknown row_id: {row_id}. available: {list(ROW_CONFIGS.keys())[:20]} ...")
    return ROW_CONFIGS[row_id]


if __name__ == "__main__":
    print(f"Total rows: {len(ROW_CONFIGS)}")
    print(f"  Tier 1 (main, trained): {TIER1_ROWS}")
    print(f"  Tier 1 (no-train):       {TIER1_NO_TRAIN}")
    print(f"  Tier 2 (negative):       {TIER2_ROWS}")
    print(f"  Tier 3 (sub-ablation):   {len(TIER3_ROWS)} rows")
    print(f"  Total training runs:     {len(TIER1_ROWS) + len(TIER2_ROWS) + len(TIER3_ROWS)}")
    print(f"\n--- Sample row (h) ---")
    import json
    print(json.dumps(get_config("h"), indent=2, default=str))
    print(f"\n--- Inference variants ({len(INFERENCE_VARIANTS)}) ---")
    for v, vd in INFERENCE_VARIANTS.items():
        print(f"  {v}: {vd}")
