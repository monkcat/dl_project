"""Experiment configurations.

Each ablation row is a preset dict consumed by `pipeline/run_experiment.py`.
Use `--config <row_id>` (e.g., `--config a`) to launch.

Naming follows REPORT_KR.md §6.3 ablation table:

  (a) baseline       LoRA + InfoNCE (binary 1-hop positives), no GPE
  (e) grcl_no_gpe    LoRA + GRCL (graded), no GPE
  (f) grcl_gpe       LoRA + GRCL + GPE
  (h) full           LoRA + GRCL + GPE + L_cov + L_cons  (main proposed method)
  (gme) gme_ref      GME zero-shot reference (no training)
"""
from __future__ import annotations

# Common hyperparameters shared by all FT runs
COMMON = {
    # Encoder + LoRA
    "hf_id": "google/siglip2-base-patch16-224",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "proj_dim": 128,
    # Training
    "train_sample": 0,          # 0 = use all 25,459 SPIQA train papers
    "steps": 8000,
    "batch": 16,                # tuned for A100 80GB; per-anchor pool=12 → ~200 elements/step
    "pool": 12,
    "lr": 3e-5,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "tau": 0.07,
    "grad_clip": 1.0,
    "anchor_kind": "mixed",     # caption_of + refer_to + nl_qa, ⅓ each
    "seed": 42,
    # GPE
    "gpe_alpha_init_raw": -3.0,  # σ(-3) ≈ 0.05 (gentle PE contribution at start)
    "beta_init_raw": -3.0,
    # GRCL graph-relevance
    "gamma": 0.5,                # 2-hop decay
    # Eval cadence
    "eval_every": 500,
    "eval_cf_pairs": 200,
    "eval_recall_n": 200,
}


ROW_CONFIGS: dict[str, dict] = {

    "a": {
        **COMMON,
        "row_id": "a",
        "name": "baseline_infonce",
        "description": "LoRA + standard InfoNCE (binary 1-hop positives), no GPE",
        "use_gpe": False,
        "loss_type": "infonce",        # binarize graph_relevance at threshold
        "infonce_threshold": 0.5,      # g(q,e) ≥ 0.5 → positive
        "lambda_cov": 0.0,
        "lambda_cons": 0.0,
    },

    "e": {
        **COMMON,
        "row_id": "e",
        "name": "grcl_no_gpe",
        "description": "LoRA + GRCL (graded graph-relevance), no GPE",
        "use_gpe": False,
        "loss_type": "grcl",
        "lambda_cov": 0.0,
        "lambda_cons": 0.0,
    },

    "f": {
        **COMMON,
        "row_id": "f",
        "name": "grcl_gpe",
        "description": "LoRA + GRCL + GPE (no L_cov, no L_cons)",
        "use_gpe": True,
        "loss_type": "grcl",
        "lambda_cov": 0.0,
        "lambda_cons": 0.0,
    },

    "h": {
        **COMMON,
        "row_id": "h",
        "name": "full_method",
        "description": "Full method: LoRA + GRCL + GPE + L_cov + L_cons",
        "use_gpe": True,
        "loss_type": "grcl",
        "lambda_cov": 0.3,
        "lambda_cons": 0.5,
    },

    # GME reference — no training, just zero-shot retrieval. Trainer skips
    # the FT loop and goes straight to eval.
    "gme": {
        "row_id": "gme",
        "name": "gme_qwen2vl_zero_shot",
        "description": "GME-Qwen2-VL-2B zero-shot (no training); apply propagation at inference",
        "skip_training": True,
        "encoder_kind": "gme",        # eval_full picks GMEQwen2VLEncoder
        "hf_id": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    },
}


# Inference-time evaluation variants — applied on top of each trained row
# (and on (gme)). Each variant produces its own result cell.
INFERENCE_VARIANTS: dict[str, dict] = {
    "no_prop":   {"graph_propagate": False, "alpha": 0.0, "T": 0},
    "prop_a03_T2": {"graph_propagate": True, "alpha": 0.3, "T": 2},
}


# Evaluation datasets — paths point to the local benchmarks dir.
# `query_format`: how to interpret the queries JSONL.
EVAL_DATASETS: dict[str, dict] = {
    "spiqa_testA": {
        "name": "SPIQA test-A",
        "graph": "data/benchmarks/spiqa/test-A/element_graph_v2.json",
        "elements": "data/benchmarks/spiqa/test-A/elements_v2.jsonl",
        "queries": "data/benchmarks/spiqa/test-A/SPIQA_testA.json",
        "query_format": "spiqa",        # {paper_id: {qa: [{question, reference}, ...]}}
        "image_root": "data/benchmarks/spiqa/test-A/images_224px/SPIQA_testA_Images_224px",
    },
    "sciegqa": {
        "name": "SciEGQA",
        "graph": "data/benchmarks/sciegqa/element_graph_v2.json",
        "elements": "data/benchmarks/sciegqa/elements_v2.jsonl",
        "queries": "data/benchmarks/sciegqa/queries_with_gt_docling.jsonl",
        "query_format": "jsonl_gt_ids", # per-line: {qid, doc_id, query, gt_element_ids: [...]}
        "image_root": "data/benchmarks/sciegqa/Images",
    },
    "mmdocir": {
        "name": "MMDocIR",
        "graph": "data/benchmarks/mmdocir/element_graph_v2.json",
        "elements": "data/benchmarks/mmdocir/elements_v2.jsonl",
        "queries": "data/benchmarks/mmdocir/academic_queries.jsonl",
        "query_format": "jsonl_gt_ids",
        "image_root": None,             # MMDocIR images are in element image_b64 inline
    },
}


def get_config(row_id: str) -> dict:
    if row_id not in ROW_CONFIGS:
        raise KeyError(f"unknown row_id: {row_id}. available: {list(ROW_CONFIGS.keys())}")
    return ROW_CONFIGS[row_id]


if __name__ == "__main__":
    import json
    for row_id, cfg in ROW_CONFIGS.items():
        print(f"\n=== row {row_id}: {cfg['name']} ===")
        print(f"  {cfg.get('description', '')}")
        if not cfg.get("skip_training"):
            print(f"  use_gpe={cfg['use_gpe']}, loss={cfg['loss_type']}, "
                  f"λ_cov={cfg.get('lambda_cov',0)}, λ_cons={cfg.get('lambda_cons',0)}")
    print("\n--- Inference variants ---")
    for vid, v in INFERENCE_VARIANTS.items():
        print(f"  {vid}: {v}")
    print("\n--- Eval datasets ---")
    for did, d in EVAL_DATASETS.items():
        print(f"  {did}: {d['name']}")
