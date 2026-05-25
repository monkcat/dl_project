"""Experiment configurations — single source of truth for all ablation rows.

Each row is a preset dict consumed by `pipeline/run_experiment.py`.
Use `--config <row_id>` (e.g., `python -m pipeline.run_experiment --config h`) to launch.

The full experiment plan covers four tiers (REPORT_KR §6.3 - §6.5 + §10):

  Tier 1 — Main ablation (§6.3)
    (a-h)   eight training rows isolating one design dim each (InfoNCE vs GRCL,
            GPE facet inclusion, presence of L_cov / L_cons).
    (gme)   GME-Qwen2-VL-2B zero-shot, no training.

  Tier 2 — Negative controls (§6.4)
    (m, n)  shuffled / random section_role  — does GPE really use the role signal?
    (o)     no query-side PE dropout         — does L_cons actually help?
    (p)     encoder swap (SigLIPv2 → CLIP-L) — encoder-agnostic claim.

  Tier 3 — Sub-ablation sweeps (§6.5)
    Each row varies ONE hyperparameter on top of (h) for a sensitivity analysis.

  Tier 4 — Follow-up extensions
    Best-HP-combo, longer training, alt GPE init, seed variance.

Inference variants (§4.9) are applied at eval time on every row's checkpoint
and never require retraining:
  - no_prop                  baseline (encoder only)
  - wfull_a03_T2             default propagation (full edge modifiers, α=0.3 T=2)
  - 21-variant prop sub-ablation (uniform / weighted-diffusion / PPR groups)
    only applied to (h) checkpoint for the deep dive.
"""
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# 1. Common hyperparameters shared by all training runs
# ──────────────────────────────────────────────────────────────────────────

COMMON = {
    # Encoder + LoRA
    "hf_id":           "google/siglip2-base-patch16-224",
    "lora_rank":       8,
    "lora_alpha":      16,
    "lora_dropout":    0.05,
    "proj_dim":        128,

    # Training schedule
    "train_sample":    0,           # 0 = use all 25,459 SPIQA train papers
    "steps":           8000,
    "batch":           16,          # batch of anchors per step
    "pool":            12,          # candidates per anchor (pos + same-doc neg + cross-doc neg)
    "lr":              3e-5,
    "weight_decay":    0.01,
    "warmup_steps":    200,
    "tau":             0.07,        # contrastive temperature
    "grad_clip":       1.0,
    "anchor_kind":     "mixed",     # caption_of + refer_to + nl_qa, ⅓ each
    "seed":            42,

    # Graph Position Embedding
    "gpe_alpha_init_raw": -3.0,     # σ(-3) ≈ 0.05 — gentle PE contribution at start
    "beta_init_raw":      -3.0,     # σ(-3) ≈ 0.05 — gentle GPE-to-token mixing at start
    "gpe_facets":         "type,role,depth,pos",  # all four facets active

    # GRCL graph-relevance kernel
    "gamma":           0.5,         # 2-hop decay (g(a,b) = max-product path weight)

    # Eval cadence during training (held-out SPIQA test-A diagnostic)
    "eval_every":      500,
    "eval_cf_pairs":   200,
    "eval_recall_n":   200,

    # Negative-control flags (default = off)
    "query_pe_dropout":   0.5,
    "section_role_mode":  "normal",
    "edge_types_only":    None,
}


def _h_full() -> dict:
    """Return the (h) full-method preset — base for Tier 3 / Tier 4 derivatives."""
    return {
        **COMMON,
        "use_gpe": True,
        "loss_type": "grcl",
        "lambda_cov": 0.3,
        "lambda_cons": 0.5,
    }


def _h_with(name: str, desc: str, **overrides) -> dict:
    """Build a Tier-3/4 row by varying one or more hyperparameters from (h)."""
    return {**_h_full(), "row_id": name, "name": name, "description": desc, **overrides}


# ──────────────────────────────────────────────────────────────────────────
# 2. Row configurations
# ──────────────────────────────────────────────────────────────────────────

ROW_CONFIGS: dict[str, dict] = {}


# ───── Tier 1 — Main ablation (REPORT_KR §6.3) ─────

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
    "encoder_kind":  "gme",
    "hf_id":         "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
}


# ───── Tier 2 — Negative controls (REPORT_KR §6.4) ─────

ROW_CONFIGS["m"] = {
    **_h_full(), "row_id": "m", "name": "shuffled_role",
    "description": "(h) with section_role permuted within each doc — PE-effect attribution",
    "section_role_mode": "shuffled",
}
ROW_CONFIGS["n"] = {
    **_h_full(), "row_id": "n", "name": "random_role",
    "description": "(h) with section_role assigned uniform-random per node",
    "section_role_mode": "random",
}
ROW_CONFIGS["o"] = {
    **_h_full(), "row_id": "o", "name": "no_query_pe_dropout",
    "description": "(h) without query-side PE dropout — train/inference mismatch test",
    "query_pe_dropout": 0.0,
}
ROW_CONFIGS["p"] = {
    **_h_full(), "row_id": "p", "name": "encoder_clip_l14",
    "description": "(h) with encoder swap: SigLIPv2 → CLIP-L/14 — encoder-agnostic test",
    "hf_id": "openai/clip-vit-large-patch14",
}


# ───── Tier 3 — Sub-ablation sweeps (REPORT_KR §6.5) ─────
# Each row varies ONE hyperparameter from (h). (h) itself = the centre of the sweep.

# γ (GRCL 2-hop decay): (h) uses γ=0.5
ROW_CONFIGS["gamma_03"] = _h_with("gamma_03", "(h) with γ=0.3 (steeper 2-hop decay)", gamma=0.3)
ROW_CONFIGS["gamma_07"] = _h_with("gamma_07", "(h) with γ=0.7 (flatter 2-hop decay)", gamma=0.7)

# λ_cov sweep: (h) uses λ_cov=0.3
for _lam in (0.0, 0.1, 0.5, 1.0):
    rid = f"cov_{int(_lam * 10):02d}"
    ROW_CONFIGS[rid] = _h_with(rid, f"(h) with λ_cov={_lam}", lambda_cov=_lam)

# λ_cons sweep: (h) uses λ_cons=0.5
for _lam in (0.0, 0.1, 0.3, 1.0):
    rid = f"cons_{int(_lam * 10):02d}"
    ROW_CONFIGS[rid] = _h_with(rid, f"(h) with λ_cons={_lam}", lambda_cons=_lam)

# LoRA rank sweep: (h) uses rank=8
for _r in (4, 16, 32):
    rid = f"lora_r{_r}"
    ROW_CONFIGS[rid] = _h_with(rid, f"(h) with LoRA rank={_r}",
                                lora_rank=_r, lora_alpha=2 * _r)

# Learning-rate sweep: (h) uses lr=3e-5
ROW_CONFIGS["lr_1e5"] = _h_with("lr_1e5", "(h) with lr=1e-5 (conservative)", lr=1e-5)
ROW_CONFIGS["lr_1e4"] = _h_with("lr_1e4", "(h) with lr=1e-4 (aggressive)",   lr=1e-4)

# Temperature τ sweep: (h) uses τ=0.07
ROW_CONFIGS["tau_005"] = _h_with("tau_005", "(h) with τ=0.05 (sharper softmax)", tau=0.05)
ROW_CONFIGS["tau_010"] = _h_with("tau_010", "(h) with τ=0.10 (softer softmax)",  tau=0.10)

# Anchor kind sweep: (h) uses mixed
ROW_CONFIGS["anchor_caption"] = _h_with(
    "anchor_caption", "(h) with caption_of anchors only — visual-focused training",
    anchor_kind="caption_of",
)
ROW_CONFIGS["anchor_refer"] = _h_with(
    "anchor_refer", "(h) with refer_to anchors only — body-figure linkage focus",
    anchor_kind="refer_to",
)
ROW_CONFIGS["anchor_nlqa"] = _h_with(
    "anchor_nlqa", "(h) with NL-QA anchors only — query-document alignment focus",
    anchor_kind="nl_qa",
)

# Edge-type isolation in GRCL relevance kernel
for _edge in ("caption_of", "refer_to", "contains"):
    rid = f"edge_{_edge}"
    ROW_CONFIGS[rid] = _h_with(
        rid, f"(h) using only '{_edge}' edges in GRCL graph relevance",
        edge_types_only=_edge,
    )

# Visual token count: (h) uses full patches (≈196 for SigLIPv2-224)
for _ntok in (16, 64):
    rid = f"tokens_{_ntok:03d}"
    ROW_CONFIGS[rid] = _h_with(
        rid, f"(h) with {_ntok} visual tokens (2D adaptive-pooled patch grid)",
        tokens_per_visual=_ntok,
    )


# ───── Tier 4 — Follow-up extensions ─────

ROW_CONFIGS["h_best_combo"] = _h_with(
    "h_best_combo",
    "Stacked HP winners (lr=1e-4 + τ=0.10 + λ_cov=0.5) — Phase-3 best from Tier 3",
    lr=1e-4, tau=0.10, lambda_cov=0.5,
)
ROW_CONFIGS["h_best_combo_16k"] = _h_with(
    "h_best_combo_16k",
    "Best HP combo trained for 16 k steps to check convergence plateau",
    lr=1e-4, tau=0.10, lambda_cov=0.5,
    steps=16000, warmup_steps=400, eval_every=1000,
)
ROW_CONFIGS["h_gpe_strong"] = _h_with(
    "h_gpe_strong",
    "(h) with stronger GPE init (α/β raw=0 → σ≈0.5, ~10× the default contribution)",
    gpe_alpha_init_raw=0.0, beta_init_raw=0.0,
)
ROW_CONFIGS["h_seed_43"] = _h_with("h_seed_43", "(h) re-trained with seed=43 — variance check", seed=43)
ROW_CONFIGS["h_seed_44"] = _h_with("h_seed_44", "(h) re-trained with seed=44 — variance check", seed=44)


# ──────────────────────────────────────────────────────────────────────────
# 3. Tier groupings (used by launcher + aggregator)
# ──────────────────────────────────────────────────────────────────────────

TIER1_ROWS = ["a", "b", "c", "d", "e", "f", "g", "h"]
TIER1_NO_TRAIN = ["gme"]
TIER2_ROWS = ["m", "n", "o", "p"]
TIER3_ROWS = (
    # γ sweep
    ["gamma_03", "gamma_07"]
    # λ_cov sweep
    + [f"cov_{int(l * 10):02d}" for l in (0.0, 0.1, 0.5, 1.0)]
    # λ_cons sweep
    + [f"cons_{int(l * 10):02d}" for l in (0.0, 0.1, 0.3, 1.0)]
    # LoRA rank
    + [f"lora_r{r}" for r in (4, 16, 32)]
    # learning rate
    + ["lr_1e5", "lr_1e4"]
    # temperature τ
    + ["tau_005", "tau_010"]
    # anchor kind
    + ["anchor_caption", "anchor_refer", "anchor_nlqa"]
    # edge-type isolation
    + [f"edge_{e}" for e in ("caption_of", "refer_to", "contains")]
    # tokens per visual
    + [f"tokens_{n:03d}" for n in (16, 64)]
)
TIER4_ROWS = [
    "h_best_combo", "h_best_combo_16k", "h_gpe_strong",
    "h_seed_43", "h_seed_44",
]

ALL_TRAINED_ROWS = TIER1_ROWS + TIER2_ROWS + TIER3_ROWS + TIER4_ROWS
ALL_ROWS = ALL_TRAINED_ROWS + TIER1_NO_TRAIN


# ──────────────────────────────────────────────────────────────────────────
# 4. Inference-time evaluation variants
# ──────────────────────────────────────────────────────────────────────────
#
# Three propagation regimes (REPORT_KR §4.9 + §6.5):
#
#   A. uniform_*   — diffusion with all edge weights = 1 (graph STRUCTURE only)
#   B. wbase_* / wfull_*   — diffusion with edge-weight modifiers
#                            (BASE × {role, visual} subsets)
#   C. ppr_*       — Personalized PageRank: s = α·q + (1-α)·P^T·s with teleport
#
# Each variant declares:
#   method:    "none" | "diffusion" | "ppr"
#   weights:   "uniform" | "base" | "base+role" | "base+visual" | "full"
#   alpha:     propagation/teleport strength
#   T:         diffusion iterations (diffusion only)
#   max_iter:  PPR iteration cap (ppr only; early-stop on tol=1e-4)
#
INFERENCE_VARIANTS: dict[str, dict] = {
    "no_prop":           {"method": "none"},

    # ── Group A: uniform weights (structure-only — modifier-design ablation) ──
    "uniform_a01_T2":    {"method": "diffusion", "weights": "uniform", "alpha": 0.1, "T": 2},
    "uniform_a03_T1":    {"method": "diffusion", "weights": "uniform", "alpha": 0.3, "T": 1},
    "uniform_a03_T2":    {"method": "diffusion", "weights": "uniform", "alpha": 0.3, "T": 2},
    "uniform_a03_T3":    {"method": "diffusion", "weights": "uniform", "alpha": 0.3, "T": 3},
    "uniform_a05_T2":    {"method": "diffusion", "weights": "uniform", "alpha": 0.5, "T": 2},

    # ── Group B: weighted diffusion (modifier ablation + α/T sweep) ──
    # B1: which modifier components are active?
    "wbase_a03_T2":      {"method": "diffusion", "weights": "base",        "alpha": 0.3, "T": 2},
    "wbase_role_a03_T2": {"method": "diffusion", "weights": "base+role",   "alpha": 0.3, "T": 2},
    "wbase_vis_a03_T2":  {"method": "diffusion", "weights": "base+visual", "alpha": 0.3, "T": 2},
    "wfull_a03_T2":      {"method": "diffusion", "weights": "full",        "alpha": 0.3, "T": 2},  # default
    # B2: α sweep on full weights
    "wfull_a01_T2":      {"method": "diffusion", "weights": "full", "alpha": 0.1, "T": 2},
    "wfull_a05_T2":      {"method": "diffusion", "weights": "full", "alpha": 0.5, "T": 2},
    "wfull_a07_T2":      {"method": "diffusion", "weights": "full", "alpha": 0.7, "T": 2},
    # B3: T sweep on full weights
    "wfull_a03_T1":      {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 1},
    "wfull_a03_T3":      {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 3},

    # ── Group C: Personalized PageRank ──
    "ppr_a015":          {"method": "ppr", "weights": "full",    "alpha": 0.15, "max_iter": 30},
    "ppr_a030":          {"method": "ppr", "weights": "full",    "alpha": 0.30, "max_iter": 30},
    "ppr_a050":          {"method": "ppr", "weights": "full",    "alpha": 0.50, "max_iter": 30},
    "ppr_a070":          {"method": "ppr", "weights": "full",    "alpha": 0.70, "max_iter": 30},
    "ppr_a085":          {"method": "ppr", "weights": "full",    "alpha": 0.85, "max_iter": 30},
    # C2: PPR with uniform weights (isolates teleport effect from edge weighting)
    "ppr_unif_a030":     {"method": "ppr", "weights": "uniform", "alpha": 0.30, "max_iter": 30},
    "ppr_unif_a050":     {"method": "ppr", "weights": "uniform", "alpha": 0.50, "max_iter": 30},

    # ── Legacy aliases (back-compat for old launchers) ──
    "prop_a03_T2":       {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 2},
    "prop_a01_T2":       {"method": "diffusion", "weights": "full", "alpha": 0.1, "T": 2},
    "prop_a05_T2":       {"method": "diffusion", "weights": "full", "alpha": 0.5, "T": 2},
    "prop_a07_T2":       {"method": "diffusion", "weights": "full", "alpha": 0.7, "T": 2},
    "prop_a03_T1":       {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 1},
    "prop_a03_T3":       {"method": "diffusion", "weights": "full", "alpha": 0.3, "T": 3},
}

# Compact "always run per row" set (kept small for fast per-row eval)
DEFAULT_VARIANTS = ("no_prop", "wfull_a03_T2")

# Variant groups for the §7.4 propagation sub-ablation table
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

# Full sub-ablation sweep — run on (h) checkpoint only (eval-only, ~40 min total)
PROP_SWEEP_VARIANTS = VARIANTS_GROUP_A_UNIFORM + VARIANTS_GROUP_B_WEIGHTS + VARIANTS_GROUP_C_PPR


# ──────────────────────────────────────────────────────────────────────────
# 5. Evaluation datasets
# ──────────────────────────────────────────────────────────────────────────

EVAL_DATASETS: dict[str, dict] = {
    "spiqa_testA": {
        "name":         "SPIQA test-A",
        "graph":        "data/benchmarks/spiqa/test-A/element_graph_v2.json",
        "elements":     "data/benchmarks/spiqa/test-A/elements_v2.jsonl",
        "queries":      "data/benchmarks/spiqa/test-A/SPIQA_testA.json",
        "query_format": "spiqa",          # {paper_id: {qa: [{question, reference}, ...]}}
        "image_root":   "data/benchmarks/spiqa/test-A/SPIQA_testA_Images_224px",
    },
    "sciegqa": {
        "name":         "SciEGQA",
        "graph":        "data/benchmarks/sciegqa/element_graph_v2.json",
        "elements":     "data/benchmarks/sciegqa/elements_v2.jsonl",
        "queries":      "data/benchmarks/sciegqa/queries_with_gt_docling.jsonl",
        "query_format": "jsonl_gt_ids",   # per-line: {qid, doc_id, query, gt_element_ids, gt_pages}
        "image_root":   "data/benchmarks/sciegqa/Images",
    },
    "mmdocir": {
        "name":         "MMDocIR",
        "graph":        "data/benchmarks/mmdocir/element_graph_v2.json",
        "elements":     "data/benchmarks/mmdocir/elements_v2.jsonl",
        "queries":      "data/benchmarks/mmdocir/academic_queries.jsonl",
        "query_format": "jsonl_gt_ids",
        "image_root":   None,             # MMDocIR images are inline in element image_b64
    },
}


# ──────────────────────────────────────────────────────────────────────────
# 6. Helpers
# ──────────────────────────────────────────────────────────────────────────

def get_config(row_id: str) -> dict:
    if row_id not in ROW_CONFIGS:
        avail = sorted(ROW_CONFIGS.keys())
        raise KeyError(
            f"unknown row_id: {row_id!r}\n"
            f"available rows ({len(avail)}): {avail}"
        )
    return ROW_CONFIGS[row_id]


def list_rows_by_tier() -> dict[str, list[str]]:
    return {
        "Tier 1 — Main ablation":         TIER1_ROWS + TIER1_NO_TRAIN,
        "Tier 2 — Negative controls":     TIER2_ROWS,
        "Tier 3 — Sub-ablation sweeps":   TIER3_ROWS,
        "Tier 4 — Follow-up extensions":  TIER4_ROWS,
    }


if __name__ == "__main__":
    import json

    print(f"Total rows: {len(ROW_CONFIGS)} "
          f"({len(ALL_TRAINED_ROWS)} trained + {len(TIER1_NO_TRAIN)} zero-shot)\n")
    for tier, rows in list_rows_by_tier().items():
        print(f"{tier} — {len(rows)} rows")
        for r in rows:
            cfg = get_config(r)
            kind = "(no-train)" if cfg.get("skip_training") else "          "
            print(f"  {kind} {r:20s}  {cfg.get('description', '')[:70]}")
        print()

    print(f"Inference variants: {len(INFERENCE_VARIANTS)}")
    print(f"  Group A (uniform):  {len(VARIANTS_GROUP_A_UNIFORM)}")
    print(f"  Group B (weighted): {len(VARIANTS_GROUP_B_WEIGHTS)}")
    print(f"  Group C (PPR):      {len(VARIANTS_GROUP_C_PPR)}")
    print(f"  Default per-row:    {DEFAULT_VARIANTS}")
