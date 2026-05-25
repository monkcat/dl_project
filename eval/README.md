# `eval/` — Evaluation pipeline + baselines + analysis

This directory is **separate from `pipeline/`**: it holds dataset loaders,
metric implementations, baseline retrievers, and analysis/visualization
scripts. None of the runtime training code lives here.

```
eval/
├── analysis/      diagnostics, visualizations, failure analysis
├── baselines/     BM25, ColPali, VisRAG, SigLIP-only baselines (zero-shot)
├── datasets/      lightweight dataset loaders (not used by main pipeline)
├── metrics/       retrieval / localization / generation metric primitives
└── results/       (gitignored) runtime outputs
```

---

## `metrics/`

Reusable metric primitives. Used both by `pipeline/eval_full.py` and the
baseline scripts.

| File | Metrics |
|---|---|
| `retrieval.py` | Recall@k, MRR, nDCG, MAP for ranked lists |
| `localization.py` | Element-IoU, page-hit rate for bbox-grounded GT |
| `generation.py` | (placeholder) — Exact-Match / F1 / LLM-judge for end-to-end QA |

These are deliberately stateless functions so they can be called from any
evaluation harness.

---

## `baselines/`

Zero-shot reference retrievers — none of these are fine-tuned. Used as Tier-0
comparisons (REPORT_KR §6.2 baselines).

| Baseline | What it does | Key file |
|---|---|---|
| **BM25** | Text-only retrieval over chunks (no images, no graph) | `bm25.py` |
| **SigLIP element** | Zero-shot SigLIP encoder, element-level | `siglip_elem.py` |
| **ColPali** | Page-level visual retrieval (ColBERT × PaliGemma) | `colpali.py` |
| **VisRAG** | Page-level VLM-based retrieval | `visrag.py` |
| **`unified_eval.py`** | Common runner across all baselines |

These baselines run on the same `EVAL_DATASETS` paths as `pipeline.eval_full`,
so metrics are directly comparable.

---

## `datasets/`

Standalone dataset loaders. Mostly used by analysis scripts — the main
`pipeline/trainer.py` loads SPIQA via its own `SpiqaSplitData` class for
control over graph + element caching.

| File | Dataset |
|---|---|
| `spiqa.py` | SPIQA train/val/test-A loaders |
| `sciegqa.py` | SciEGQA loader |
| `mmdocir.py` | MMDocIR loader |
| `loaders.py` | Common iteration helpers |

---

## `analysis/`

One-off scripts for diagnostics, paper figures, and failure analysis. Run
manually (not part of the main suite).

| Script | What it shows |
|---|---|
| `motivation_modality_gap.py` | 6-encoder zero-shot diagnostic of text↔figure cosine gap (REPORT §2.3) |
| `modality_gap_baseline.py` | Caption ↔ figure cosine before/after FT |
| `hypersphere_viz.py` | Embedding distribution on the unit sphere (REPORT §2.3 footer) |
| `visualize_graph_v2.py` | pyvis interactive HTML of a doc graph |
| `dataset_examples.py` | Hand-picked motivating examples |
| `spiqa_enhanced_viz.py` | Per-paper element-graph view |
| `failure_analysis.py` | 20-query × 3-dataset failure taxonomy (extraction/retrieval/expansion/generation) |
| `stat_test.py` | Paired-bootstrap significance |
| `aggregate_motivation.py` | Tables for §2.3 motivation |
| `system_diagram.py` | Architecture diagram generator |

---

## `results/`

Gitignored. Populated by:
- `pipeline.run_experiment` → `eval/results/experiments/<rid>_<name>/{train,eval,summary}.json`
- `pipeline.aggregate_results` → `eval/results/experiments/SUMMARY.md`
- Analysis scripts → `eval/results/analysis/<name>.{json,png,html}`

Layout after a full suite run:

```
eval/results/
├── experiments/
│   ├── a_baseline_infonce/        train.json, eval.json, summary.json
│   ├── b_gpe_type_infonce/        ...
│   ├── ...
│   ├── h_full_method/             (main row)
│   ├── h_prop_sweep/              (21-variant propagation sub-ablation)
│   └── SUMMARY.md                  (aggregated markdown report)
└── analysis/                       (manual analysis outputs)
```
