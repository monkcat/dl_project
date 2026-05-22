# Element Graph for Academic Multimodal Retrieval

Typed document element graphs for retrieval research on academic papers,
covering **SPIQA**, **SciEGQA**, and **MMDocIR**. The graph plays two roles:

1. **Training supervision** via Graph-Relevance Contrastive Loss (GRCL) —
   a graph-induced graded relevance distribution replaces binary positives.
2. **Inference-time score propagation** via typed edges with target-attribute
   modifiers (appendix boost, cross-modal boost).

The same edge-weight schema (`pipeline/types.py:BASE_EDGE_WEIGHTS`) drives both
training and inference (single source of truth).

## Full design + results

See **[REPORT_KR.md](REPORT_KR.md)** — motivation, schema, methodology,
related work, dataset stats, experimental setup, and results.

---

## Quick start (new environment)

See **[scripts/README_new_env.md](scripts/README_new_env.md)** for the full
playbook (datasets, A100 launch, troubleshooting).

```bash
# 1. Clone + Python env  (conda OR venv — either is fine)
git clone https://github.com/monkcat/dl_project.git && cd dl_project
python3.10 -m venv .venv && source .venv/bin/activate    # or: conda create -n dl_hw2 python=3.10 -y && conda activate dl_hw2

# 2. Install
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. HF login + pull all data (v2.1 graphs + SPIQA + SciEGQA + MMDocIR)
huggingface-cli login
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1

# 4. Run all ablations (2× A100, ~12–14 hours)
bash scripts/run_all_experiments.sh

# 5. Aggregated results
cat eval/results/experiments/SUMMARY.md
```

---

## Repository layout

```
pipeline/                       core training + eval code
├── types.py                    schema v2.1 + graph_propagate + late_interaction (single source of truth)
├── configs.py                  ablation row presets (a / e / f / h / gme)
├── trainer.py                  end-to-end FT loop  (--config <row_id>)
├── eval_full.py                3-dataset multi-positive evaluation
├── run_experiment.py           train + eval orchestrator (one row)
├── aggregate_results.py        compile all rows into a markdown summary
├── graph_pe.py                 Graph Position Embedding (gated additive PE)
├── graph_relevance.py          g(q, e) graph-distance relevance for GRCL
├── losses.py                   GRCL + L_cov + L_cons
├── element_encoder.py          SigLIPv2 + LoRA + GPE forward
├── build_spiqa_graph_v2.py     SPIQA graph builder (LaTeX-based)
├── build_docling_graph_v2.py   SciEGQA + MMDocIR graph builder (docling labels)
├── extract_sciegqa_docling_v2.py   re-extract SciEGQA preserving docling labels
├── prepare_hf_export.py        materialize local hf_export/ layout
└── upload_to_hf.py             push v2.1 graphs to HF

eval/
├── analysis/                   diagnostics, viz, failure analysis
├── baselines/                  BM25, ColPali, SigLIP, VisRAG (zero-shot refs)
├── datasets/                   dataset loaders
└── metrics/                    retrieval / localization / generation metrics

scripts/
├── setup_data_from_hf.py       pull all data from HF, set up data/benchmarks/
├── run_all_experiments.sh      launch all ablations on 2× A100
└── README_new_env.md           new-environment playbook (KR)

data/benchmarks/                (not tracked — populated by setup_data_from_hf.py)
ckpt/                           (not tracked — checkpoints written during training)
eval/results/                   (not tracked — runtime outputs)
```

---

## HF datasets used

| Repo | Size | Purpose |
|---|---|---|
| `ljh38/element-graph-v2.1`     | ~2.4 GB | Our v2.1 graphs (SPIQA / SciEGQA / MMDocIR) |
| `google/spiqa`                 | ~33 GB  | Original SPIQA images + paragraphs |
| `Yuwh07/SciEGQA-Bench`         | ~1.3 GB | Original SciEGQA PDFs + page images + queries |
| `MMDocIR/MMDocIR-Challenge`    | ~2.5 GB | Original MMDocIR layout parquet + annotations |

`setup_data_from_hf.py` fetches and stitches all of the above into
`data/benchmarks/{spiqa,sciegqa,mmdocir}/` with the file names the trainer
and evaluator expect.

