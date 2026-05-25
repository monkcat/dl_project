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

# ── Hyperparameter sweep — all based on full method (h) ──
ROW_CONFIGS["h_lr_low"] = {
    **COMMON, "row_id": "h_lr_low", "name": "hp_lr_1e5",
    "description": "Full method, lr=1e-5",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.3, "lambda_cons": 0.5,
    "lr": 1e-5,
}
ROW_CONFIGS["h_lr_high"] = {
    **COMMON, "row_id": "h_lr_high", "name": "hp_lr_1e4",
    "description": "Full method, lr=1e-4",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.3, "lambda_cons": 0.5,
    "lr": 1e-4,
}
ROW_CONFIGS["h_tau_low"] = {
    **COMMON, "row_id": "h_tau_low", "name": "hp_tau_005",
    "description": "Full method, tau=0.05",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.3, "lambda_cons": 0.5,
    "tau": 0.05,
}
ROW_CONFIGS["h_tau_high"] = {
    **COMMON, "row_id": "h_tau_high", "name": "hp_tau_010",
    "description": "Full method, tau=0.10",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.3, "lambda_cons": 0.5,
    "tau": 0.10,
}
ROW_CONFIGS["h_rank_low"] = {
    **COMMON, "row_id": "h_rank_low", "name": "hp_lora_rank4",
    "description": "Full method, lora_rank=4",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.3, "lambda_cons": 0.5,
    "lora_rank": 4,
}
ROW_CONFIGS["h_rank_high"] = {
    **COMMON, "row_id": "h_rank_high", "name": "hp_lora_rank16",
    "description": "Full method, lora_rank=16",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.3, "lambda_cons": 0.5,
    "lora_rank": 16,
}
ROW_CONFIGS["h_lcov_low"] = {
    **COMMON, "row_id": "h_lcov_low", "name": "hp_lcov_01",
    "description": "Full method, lambda_cov=0.1",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.1, "lambda_cons": 0.5,
}
ROW_CONFIGS["h_lcov_high"] = {
    **COMMON, "row_id": "h_lcov_high", "name": "hp_lcov_05",
    "description": "Full method, lambda_cov=0.5",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.5, "lambda_cons": 0.5,
}
ROW_CONFIGS["h_lcons_low"] = {
    **COMMON, "row_id": "h_lcons_low", "name": "hp_lcons_01",
    "description": "Full method, lambda_cons=0.1",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.3, "lambda_cons": 0.1,
}
ROW_CONFIGS["h_lcons_high"] = {
    **COMMON, "row_id": "h_lcons_high", "name": "hp_lcons_10",
    "description": "Full method, lambda_cons=1.0",
    "use_gpe": True, "loss_type": "grcl", "lambda_cov": 0.3, "lambda_cons": 1.0,
}

# ── Phase 3 follow-up experiments (post HP-sweep insights) ──
ROW_CONFIGS["h_best_combo"] = {
    **COMMON, "row_id": "h_best_combo", "name": "best_combo",
    "description": "Stacked HP winners: lr=1e-4 + tau=0.10 + lambda_cov=0.5",
    "use_gpe": True, "loss_type": "grcl",
    "lambda_cov": 0.5, "lambda_cons": 0.5,
    "lr": 1e-4, "tau": 0.10,
}
ROW_CONFIGS["h_best_combo_16k"] = {
    **COMMON, "row_id": "h_best_combo_16k", "name": "best_combo_16k",
    "description": "Best HP combo trained for 16k steps to see plateau",
    "use_gpe": True, "loss_type": "grcl",
    "lambda_cov": 0.5, "lambda_cons": 0.5,
    "lr": 1e-4, "tau": 0.10,
    "steps": 16000, "warmup_steps": 400, "eval_every": 1000,
}
ROW_CONFIGS["h_gpe_strong"] = {
    **COMMON, "row_id": "h_gpe_strong", "name": "gpe_strong_init",
    "description": "Full method with stronger GPE init (beta_init=0, alpha_init=0; sigma~0.5)",
    "use_gpe": True, "loss_type": "grcl",
    "lambda_cov": 0.3, "lambda_cons": 0.5,
    "gpe_alpha_init_raw": 0.0,
    "beta_init_raw": 0.0,
}

# GME reference — no training, just zero-shot retrieval.
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

# Token count per visual (default = 196 = full patches for 224×224).
# Implemented via 2D adaptive avg-pool on the patch grid in forward_vision.
# 14×14 → 8×8 (64) → 4×4 (16). Larger grids (CLIP 16×16=256) also handled.
for ntok in (16, 64):
    rid = f"tokens_{ntok:03d}"
    ROW_CONFIGS[rid] = _h_with(rid, f"(h) with {ntok} tokens per visual element",
                                tokens_per_visual=ntok)


# ───── Inference-time evaluation variants ─────
#
# Three propagation regimes (REPORT_KR §6.5 sub-ablation):
#   A. uniform_*        — diffusion with all edge weights = 1 (structure only)
#   B. wbase_* / wfull_*— diffusion with various weight modifier combinations
#   C. ppr_*            — Personalized PageRank (teleport-based) on full weights
#
# Each variant declares:
#   method:  "none" | "diffusion" | "ppr"
#   weights: "uniform" | "base" | "base+role" | "base+visual" | "full"
#   alpha:   propagation/teleport strength
#   T:       diffusion iterations (diffusion only)
#   max_iter: ppr iterations cap (ppr only)
#
INFERENCE_VARIANTS: dict[str, dict] = {
    # ── baseline (no propagation) ──
    "no_prop":           {"method": "none"},

    # ────────────────────────────────────────────────────────────────────
    # Group A — UNIFORM weights (structure only; ablates the modifier design)
    # ────────────────────────────────────────────────────────────────────
    "uniform_a01_T2":    {"method": "diffusion", "weights": "uniform", "alpha": 0.1, "T": 2},
    "uniform_a03_T1":    {"method": "diffusion", "weights": "uniform", "alpha": 0.3, "T": 1},
    "uniform_a03_T2":    {"method": "diffusion", "weights": "uniform", "alpha": 0.3, "T": 2},
    "uniform_a03_T3":    {"method": "diffusion", "weights": "uniform", "alpha": 0.3, "T": 3},
    "uniform_a05_T2":    {"method": "diffusion", "weights": "uniform", "alpha": 0.5, "T": 2},

    # ────────────────────────────────────────────────────────────────────
    # Group B — WEIGHTED diffusion (modifier ablation + α/T sweep)
    # ────────────────────────────────────────────────────────────────────
    # B1: which modifier components matter? (α=0.3, T=2 fixed)
    "wbase_a03_T2":      {"method": "diffusion", "weights": "base",        "alpha": 0.3, "T": 2},
    "wbase_role_a03_T2": {"method": "diffusion", "weights": "base+role",   "alpha": 0.3, "T": 2},
    "wbase_vis_a03_T2":  {"method": "diffusion", "weights": "base+visual", "alpha": 0.3, "T": 2},
    "wfull_a03_T2":      {"method": "diffusion", "weights": "full",        "alpha": 0.3, "T": 2},  # original default
    # B2: α sweep on full weights
    "wfull_a01_T2":      {"method": "diffusion", "weights": "full", "alpha": 0.1, "T": 2},
    "wfull_a05_T2":      {"method": "diffusion", "weights": "full", "alpha": 0.5, "T": 2},
    "wfull_a07_T2":      {"method": "diffusion", "weights": "full", "alpha": 0.7, "T": 2},
    # B3: T sweep on full weights
    "wfull_a03_T1":      {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 1},
    "wfull_a03_T3":      {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 3},

    # ────────────────────────────────────────────────────────────────────
    # Group C — Personalized PageRank
    # ────────────────────────────────────────────────────────────────────
    # C1: α (teleport prob) sweep on full weights
    "ppr_a015":          {"method": "ppr", "weights": "full",    "alpha": 0.15, "max_iter": 30},
    "ppr_a030":          {"method": "ppr", "weights": "full",    "alpha": 0.30, "max_iter": 30},
    "ppr_a050":          {"method": "ppr", "weights": "full",    "alpha": 0.50, "max_iter": 30},
    "ppr_a070":          {"method": "ppr", "weights": "full",    "alpha": 0.70, "max_iter": 30},
    "ppr_a085":          {"method": "ppr", "weights": "full",    "alpha": 0.85, "max_iter": 30},
    # C2: PPR with uniform weights (isolates effect of teleport vs weighting)
    "ppr_unif_a030":     {"method": "ppr", "weights": "uniform", "alpha": 0.30, "max_iter": 30},
    "ppr_unif_a050":     {"method": "ppr", "weights": "uniform", "alpha": 0.50, "max_iter": 30},

    # ── Backwards-compat aliases for legacy launcher scripts ──
    # Old name `prop_a03_T2` (and friends) now redirect to wfull_a03_T2.
    "prop_a03_T2":       {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 2},
    "prop_a01_T2":       {"method": "diffusion", "weights": "full", "alpha": 0.1, "T": 2},
    "prop_a05_T2":       {"method": "diffusion", "weights": "full", "alpha": 0.5, "T": 2},
    "prop_a07_T2":       {"method": "diffusion", "weights": "full", "alpha": 0.7, "T": 2},
    "prop_a03_T1":       {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 1},
    "prop_a03_T3":       {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 3},
}

# Compact "always run per row" set — keeps per-row eval fast.
DEFAULT_VARIANTS = ("no_prop", "wfull_a03_T2")

# Group bundles (used by launchers / aggregator for organized sweeps)
VARIANTS_GROUP_A_UNIFORM = (
    "uniform_a01_T2", "uniform_a03_T1", "uniform_a03_T2", "uniform_a03_T3", "uniform_a05_T2",
)
VARIANTS_GROUP_B_WEIGHTS = (
    "wbase_a03_T2", "wbase_role_a03_T2", "wbase_vis_a03_T2", "wfull_a03_T2",
    "wfull_a01_T2", "wfull_a05_T2", "wfull_a07_T2", "wfull_a03_T1", "wfull_a03_T3",
)
VARIANTS_GROUP_C_PPR = (
    "ppr_a015", "ppr_a030", "ppr_a050", "ppr_a070", "ppr_a085",
    "ppr_unif_a030", "ppr_unif_a050",
)

# Full sub-ablation sweep (run on (h) checkpoint, REPORT §6.5)
PROP_SWEEP_VARIANTS = VARIANTS_GROUP_A_UNIFORM + VARIANTS_GROUP_B_WEIGHTS + VARIANTS_GROUP_C_PPR


# ───── Evaluation datasets ─────
EVAL_DATASETS: dict[str, dict] = {
    "spiqa_testA": {
        "name": "SPIQA test-A",
        "graph":    "data/benchmarks/spiqa/test-A/element_graph_v2.json",
        "elements": "data/benchmarks/spiqa/test-A/elements_v2.jsonl",
        "queries": "data/benchmarks/spiqa/test-A/SPIQA_testA.json",
        "query_format": "spiqa",        # {paper_id: {qa: [{question, reference}, ...]}}
        "image_root": "data/benchmarks/spiqa/test-A/SPIQA_testA_Images_224px",
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
    + [f"tokens_{n:03d}" for n in (16, 64)]
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
