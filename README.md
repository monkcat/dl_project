# Element Graph for Academic Figure Retrieval

학술 문서의 typed element graph (`figure` / `caption` / `paragraph` nodes + `caption_of` / `refer_to` / `contains` typed edges) 를 retrieval 시스템의 **단일 source of truth** 로 활용하는 framework. 동일 edge weight schema 가 (1) retriever 학습의 graded relevance supervision (GRCL) 과 (2) 임의의 base retriever 위에 얹는 inference-time score propagation prior 를 모두 정의한다.

> **전체 보고서**: [REPORT_KR_v3.md](REPORT_KR_v3.md)

---

## TL;DR — SPIQA test-A 핵심 결과

| Setup | R@5 | R@10 | MRR |
|---|---|---|---|
| (a) InfoNCE baseline | — | 84.4 | 32.8 |
| (e) **GRCL** (graded supervision, p=0.042 vs (a)) | — | 87.1 | 38.2 |
| **`refer_to`** edge 단독 GRCL | — | **90.4** | 41.9 |
| HP 최적화 (lr=1e-4, τ=0.10) | — | 88.6 | **42.6** |
| GME-Qwen2-VL-2B (외부 MLLM) zero-shot | 47.1 | 58.3 | 33.1 |
| **GME + graph propagation (plug-in)** | **78.2** | **84.4** | **63.3** |

가장 큰 효과는 우리 모델 학습이 아니라 **외부 retrieval-tuned MLLM 위에 graph propagation 을 plug-in** 했을 때 (+26pp R@10, +30pp MRR, MRR 거의 2배). 동일 schema 가 200M dual-encoder 의 supervised training 과 2B decoder MLLM 의 inference-time augmentation 양쪽을 구동한다.

---

## Five key findings

1. **GRCL > InfoNCE** — graph-induced graded supervision 이 binary contrastive 대비 R@10 +2.7pp, MRR +5.4pp, paired-bootstrap **p = 0.042**.
2. **`refer_to` edge dominates** — 5종 edge 중 paragraph→figure explicit reference 만 사용하면 R@10 **90.4** (전체 ablation 최고치).
3. **HP optimization** — lr=1e-4 + τ=0.10 으로 R@10 **88.6**, MRR **42.6**.
4. **Plug-in propagation on external MLLM** — GME-Qwen2-VL-2B + graph propagation 으로 R@10 **58.3 → 84.4 (+26pp)**, MRR **33.1 → 63.3 (+30pp)** — 본 study 최대 단일 효과.
5. **Three propagation regimes equivalent** — uniform diffusion / weighted diffusion / PPR 모두 GME 위에서 R@10 83-85%, MRR 59-63% 동등. **graph 구조 자체가 효용 원천**, modifier 정교함은 marginal.

---

## Quick start

### 1. Setup

```bash
git clone https://github.com/monkcat/dl_project.git && cd dl_project
python3.10 -m venv .venv && source .venv/bin/activate   # or conda env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
huggingface-cli login
```

### 2. Pull data (≈ 33 GB)

```bash
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1
```

자동 다운로드:
- `ljh38/element-graph-v2.1` — v2.1 element graph metadata
- `google/spiqa` — SPIQA train images + test-A
- (선택) `Yuwh07/SciEGQA-Bench`, `MMDocIR/MMDocIR-Challenge` — 추가 평가 dataset

### 3. (옵션) 학습된 adapter 받아서 평가만

처음부터 학습 (≈ 4-5 days) 대신, 우리가 학습한 LoRA adapter (42 rows, ~263 MB) 만 받아서 eval:

```bash
huggingface-cli download ljh38/element-graph-encoder-v2.1 --local-dir hf_models
python -m pipeline.eval_full \
    --ckpt hf_models/adapters/h_best_combo.pt \
    --lora_rank 8 --gpe_facets type,role,depth,pos \
    --datasets spiqa_testA \
    --out eval/results/quick.json
```

### 4. GME + graph propagation 재현 (학습 없음, ~10분)

```bash
python -m pipeline.eval_full \
    --hf_id Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
    --datasets spiqa_testA \
    --variants no_prop wfull_a03_T2 ppr_a030 uniform_a05_T2 \
    --out eval/results/gme_prop.json
```

### 5. 전체 학습 + ablation suite (≈ 4-5 days on 2× A100)

```bash
bash scripts/run_full_suite.sh
```

- Preflight `pipeline.sanity_check` 자동 실행
- 43 trained rows (4 tier) × 22 wave, 2× A100 병렬
- (h) checkpoint 위 21-variant propagation sub-ablation
- 결과 자동 aggregate → `eval/results/experiments/SUMMARY.md`

부분 실행: `--tier 1 / 2 / 3 / 4`, `--rows a,h,gme`, `--dry_run`.

---

## Method summary

세 component, **모두 동일 `BASE_EDGE_WEIGHTS` schema** 공유:

```python
# pipeline/types.py — single source of truth
BASE_EDGE_WEIGHTS = {
    "caption_of":   1.0,    # caption ↔ figure/table
    "refer_to":     0.8,    # body → figure (explicit reference, strongest signal)
    "contains":     0.5,    # section_header → child
    "reading_next": 0.3,
    "section_next": 0.2,
}
```

### Component 1 — Element token encoder + Late interaction

SigLIPv2-base-patch16-224 (200M) + LoRA rank 8 (~491K trainable params). token-level output `(K, D_proj)`. retrieval score 는 ColBERT-style MaxSim:
```
score(q, e) = Σ_i max_j  <q_tok[i], e_tok[j]>
```

### Component 2 — Graph-Relevance Contrastive Loss (GRCL)

graph relevance kernel `g(q, e)` = max-product path weight (2-hop, γ-decay). 이를 graded target 으로 한 listwise CE:
```
L_GRCL = -Σ_i (r_i/Σ_j r_j) · log( exp(s_i/τ) / Σ_j exp(s_j/τ) )
```
binary InfoNCE 의 strict generalization (r ∈ {0, 1} 이면 표준 InfoNCE).

### Component 3 — Inference-time graph propagation

같은 schema 가 transition matrix `P` 를 구성. 세 regime:

```
Uniform diffusion:    s ← (1-α)s + α P^T s     # edge weight = 1
Weighted diffusion:   s ← (1-α)s + α P^T s     # BASE × role × visual modifier
Personalized PageRank: s ← α q + (1-α) P^T s    # teleport
```

**재학습 없이 임의의 retriever 위에 plug-in** — §4.4 의 GME 결과가 핵심 증거.

---

## Repository layout

```
pipeline/                       core code (training + eval + graph machinery)
├── types.py                    schema v2.1 + BASE_EDGE_WEIGHTS (single source of truth)
├── configs.py                  43-row ablation presets, INFERENCE_VARIANTS (3 regimes × HP grid)
├── trainer.py                  GRCL training loop
├── eval_full.py                multi-dataset evaluation orchestrator
├── run_experiment.py           per-row train + eval
├── aggregate_results.py        SUMMARY.md generator
├── sanity_check.py             8-stage preflight check
├── element_encoder.py          SigLIPv2 / CLIP-L/14 + LoRA + late interaction
├── graph_relevance.py          g(q, e) kernel
├── graph_pe.py                 Graph Position Embedding (optional)
├── losses.py                   GRCL + auxiliary losses
└── gme_encoder.py              GME-Qwen2-VL-2B adapter

scripts/
├── setup_data_from_hf.py       single-command data download
├── run_full_suite.sh           one-shot launcher (preflight + 43 rows + prop sweep + aggregate)
├── download_all_to_local.py    laptop-side downloader (data + models, for off-line server transfer)
├── upload_archives_to_hf.py    publish archives (zip/tar) instead of 270k extracted files
└── README_new_env.md           GPU server setup playbook (KR)

eval/
├── analysis/                   plot scripts, statistical tests, paper figures
├── baselines/                  BM25, ColPali, VisRAG, SigLIP zero-shot baselines
├── datasets/                   standalone dataset loaders
├── metrics/                    Recall@k, MRR, Coverage@K, PerfectSet@K, IoU
└── results/                    (gitignored) experiment outputs + figures

data/benchmarks/                (gitignored) populated by setup_data_from_hf.py
ckpt/                           (gitignored) trained checkpoints
```

---

## Reproducibility — HF artifacts

| 자료 | HF repo | 용량 |
|---|---|---|
| Element graph metadata (v2.1) | [ljh38/element-graph-v2.1](https://huggingface.co/datasets/ljh38/element-graph-v2.1) | ~2.4 GB |
| **학습된 LoRA adapters** (42 rows) | [ljh38/element-graph-encoder-v2.1](https://huggingface.co/ljh38/element-graph-encoder-v2.1) | ~263 MB |
| SPIQA 원본 | [google/spiqa](https://huggingface.co/datasets/google/spiqa) | ~33 GB |

세 개 backbone 모델 (`google/siglip2-base-patch16-224`, `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`, `openai/clip-vit-large-patch14`) 은 HF cache 에서 자동 download.

Raw 결과 JSON (44 rows × eval / train / summary) 은 `eval/results/experiments/*/` 에 commit 되어 있음 → `pipeline.aggregate_results` 로 재집계 가능.

---

## Citation

```
@misc{dlproject2026elementgraph,
  title  = {Element Graph for Academic Figure Retrieval:
            Graded Supervision and Encoder-agnostic Score Propagation},
  author = {Lee, Jaehyeon and Lee, Seoyeon and Jun, Suhyeon and Kim, Minjun},
  year   = {2026},
  note   = {DL project, KAIST},
  url    = {https://github.com/monkcat/dl_project}
}
```

License: MIT (code). Backbone models / datasets 은 원본 license 유지.
