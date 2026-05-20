# Element Graph for Academic Multimodal Retrieval

Typed document element graphs for retrieval research on academic papers,
covering SPIQA, SciEGQA, and MMDocIR. The graph plays two roles:

1. **Training supervision** via Graph-Relevance Contrastive Loss (GRCL) —
   a graph-induced graded relevance distribution replaces binary positives.
2. **Inference-time score propagation** via typed edges with target-attribute
   modifiers (appendix boost, cross-modal boost).

Same edge-weight schema (`pipeline/types.py:BASE_EDGE_WEIGHTS`) drives both.

## Full design + results
See **[REPORT_KR.md](REPORT_KR.md)** — methodology, related work, dataset stats,
experimental setup, and (pending) results.

---

## Quick start (new environment)

See **[scripts/README_new_env.md](scripts/README_new_env.md)** for the full setup
including HF download, conda setup, A100 launch.

```bash
# 1. install
conda create -n dl_hw2 python=3.10 -y && conda activate dl_hw2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57 peft pillow numpy pandas huggingface_hub pyvis matplotlib pytz

# 2. login + pull v2.1 element graphs + original datasets
huggingface-cli login
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1

# 3. run all ablation experiments (2× A100, ~12–14 hours)
bash scripts/run_all_experiments.sh

# 4. read aggregated results
cat eval/results/experiments/SUMMARY.md
```

---

## Layout

```
pipeline/
├── types.py                    schema v2.1 + graph_propagate + late_interaction (single source of truth)
├── configs.py                  ablation row presets (a / e / f / h / gme)
├── trainer.py                  end-to-end FT loop (--config <row_id>)
├── eval_full.py                3-dataset multi-positive evaluation
├── run_experiment.py           train + eval orchestrator
├── aggregate_results.py        compile all results into a markdown summary
├── graph_pe.py                 Graph Position Embedding (gated additive PE)
├── graph_relevance.py          g(q, e) graph-distance relevance for GRCL
├── losses.py                   GRCL + L_cov + L_cons
├── element_encoder.py          SigLIPv2 + LoRA + GPE forward
├── build_spiqa_graph_v2.py     SPIQA LaTeX-only graph builder
├── build_docling_graph_v2.py   SciEGQA + MMDocIR graph builder (docling labels)
├── extract_sciegqa_docling_v2.py  re-extract SciEGQA preserving docling labels
├── upload_to_hf.py             push v2.1 graphs to HF
└── prepare_hf_export.py        materialize local hf_export/ layout

eval/
├── analysis/
│   ├── motivation_modality_gap.py   6-encoder zero-shot diagnostic
│   ├── modality_gap_baseline.py     caption-figure cosine baseline
│   ├── visualize_graph_v2.py        pyvis interactive HTML graph viz
│   └── ...
├── baselines/                  BM25, ColPali, SigLIP, VisRAG, GME
└── metrics/                    retrieval / localization / generation

scripts/
├── run_all_experiments.sh      launch all ablations on 2× A100
├── setup_data_from_hf.py       pull data from HF, set up benchmarks/
└── README_new_env.md           new-env playbook

data/benchmarks/                (not tracked — pulled from HF)
ckpt/                           (not tracked — checkpoints)
eval/results/                   (not tracked — runtime outputs)
```

## Citation
TBD (DL project, KAIST 2026)
