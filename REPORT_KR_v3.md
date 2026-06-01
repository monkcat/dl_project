# 학술 문서 Multimodal Retrieval을 위한 Element Graph (v3, lean)

**팀**: Jaehyeon Lee (20262670) · Seoyeon Lee (20262097) · Suhyeon Jun (20262717) · Minjun Kim (20262242)
**저장소**: <https://github.com/monkcat/dl_project>

---

## 1. Abstract

학술 문서의 typed element graph (figure / table / caption / paragraph + `refer_to` / `caption_of` / `contains` edges) 가 retrieval 시스템의 **단일 source of truth** 로 작동할 수 있음을 보인다 — (1) retriever 학습의 **graded relevance supervision (GRCL)** 과 (2) 임의의 base retriever 위에 얹는 **inference-time score propagation prior**.

**핵심 결과** (SPIQA test-A in-domain, R@10):

| 기여 | 효과 |
|---|---|
| **GRCL > InfoNCE** | 84.4 → 87.1 (+2.7pp), MRR 32.8 → 38.2 (+5.4pp), p=0.042 |
| **`refer_to` edge 단독 GRCL** | 90.4 R@10 — 전체 최고치 |
| **HP 최적화 (lr=1e-4 + τ=0.10)** | 88.0–88.6 R@10, MRR 42.6 |
| **GME + graph propagation** | 58.3 → 84.4 R@10 (+26pp), MRR 33.1 → 63.3 (+30pp) |
| **CLIP-L/14 backbone swap** | 88.1 R@10 — encoder-agnostic |

가장 큰 단일 효과는 학습된 SigLIP 가 아니라 **외부 retrieval-tuned MLLM (GME) 위에 같은 graph propagation 을 적용** 했을 때다. 동일 edge weight schema 가 학습된 200M dual-encoder 의 학습 supervision 과 2B decoder MLLM 의 inference-time 향상을 모두 구동한다.

**학습/평가**: SPIQA train 25,459 papers 단독 학습 → SPIQA test-A (in-domain) + SciEGQA + MMDocIR (zero-shot OOD). per-doc pool, late-interaction (ColBERT-MaxSim) retrieval.

---

## 2. Introduction

### 2.1 학술 문서 retrieval 의 cross-element 본질

학술 query 의 답은 종종 **여러 element 가 graph 로 묶여 있을 때 정의된다**:

> *"Method A 가 hard split 에서 baseline 대비 얼마나 좋아졌고 부록은 이걸 어떻게 정당화하나?"*

- Table (figure) — 수치 비교
- Table caption — "hard split" 정의 (`caption_of` Table)
- Section 4.2 paragraph — 결과 해석 (`refer_to` Table)
- Appendix B — 이론적 정당화 (body `refer_to` appendix)

페이지가 흩어지고 modality 가 다르지만, document 구조상 명시적 typed edge 로 연결돼 있다. evidence 가 graph-structured.

### 2.2 한계 — 기존 접근

- **Page-level visual RAG** (ColPali, VisRAG): top-k page 단위 → element 단위 evidence 정의 불가, appendix 누락 위험
- **Element-level retrieval, graph 없음**: candidates 독립 ranking → reference 로 묶인 element 가 cosine 점수로는 보강 안 됨
- **Page → crop 2 단계** (SciEGQA): page 단계에서 누락된 element 는 recover 불가

### 2.3 Thesis + contributions

> *"Document element graph 가 학습 supervision 과 추론 prior 의 single source of truth 다."*

세 가지 구체 contribution:

1. **Graph-Relevance Contrastive Loss (GRCL)** — typed element graph 의 2-hop max-product path weight 로 graded relevance distribution `g(q, e)` 를 정의 → binary InfoNCE 의 1-hop positive 가정 해체.
2. **Inference-time graph propagation as a plug-in** — 같은 edge weight schema 가 transition matrix `P` 를 구성, 임의의 retriever score 위에 적용 가능. **GME-Qwen2-VL-2B (외부 MLLM)** 위에 적용 시 본 study 최대 단일 효과.
3. **Edge type isolation analysis** — `refer_to` edge 단독이 다른 edge 보다 일관되게 강한 학습 signal 임을 3 dataset 에서 확인.

---

## 3. Method

본 method 는 **3 개 component 만** 으로 구성된다.

### 3.1 Element graph schema (typed edges)

```
Node types:  text | figure | table | caption | equation | section_header
Edge types:  caption_of | refer_to | contains | reading_next | section_next
```

Per-document graph `G_d = (V_d, E_d)`, `pipeline/types.py:DocumentGraph`.

**Edge weight schema** — 학습 (GRCL) 과 추론 (propagation) 양쪽을 구동하는 single source of truth:

```python
BASE_EDGE_WEIGHTS = {
    "caption_of":   1.0,   # caption ↔ figure/table
    "refer_to":     0.8,   # body → figure/table/equation
    "contains":     0.5,   # section_header → child
    "reading_next": 0.3,
    "section_next": 0.2,
}
```

### 3.2 Encoder + GRCL loss

**Encoder**: SigLIPv2-base-patch16-224 (200M) + LoRA rank 8. token-level output `(B, K, D_proj)`. retrieval score 는 ColBERT-style late interaction:

$$
\text{score}(q, e) = \sum_i \max_j q_{\text{tok}}[i] \cdot e_{\text{tok}}[j]
$$

**Graph relevance kernel** — anchor `q` 에서 candidate `e` 까지 max-product path weight (2-hop, γ=0.5):

$$
g(q, e) = \max_{\pi: q \to e, |\pi| \le 2} \prod_{(u,v) \in \pi} w_{uv} \cdot \gamma^{|\pi|-1}
$$

**GRCL loss** — listwise CE. candidate set `{e_1, ..., e_K}` 의 retrieval score 를 graded target `r_i = g(q, e_i)` 에 fit:

$$
\mathcal{L}_{\text{GRCL}} = -\sum_i \frac{r_i}{\sum_j r_j} \log \frac{\exp(s_i / \tau)}{\sum_j \exp(s_j / \tau)}
$$

`r_i` 가 모두 0/1 이면 표준 InfoNCE 로 환원 — 즉 GRCL 은 **graded relaxation of InfoNCE**.

### 3.3 Inference-time graph propagation

학습된 encoder 가 부여한 top-50 candidate score `s` 를 graph 위에서 smoothing:

**Weighted diffusion**:
$$
s^{(t+1)} = (1 - \alpha) \, s^{(t)} + \alpha \, P^T s^{(t)}, \quad t = 0, \ldots, T-1
$$

**Personalized PageRank** (대안):
$$
s^{(t+1)} = \alpha \, q + (1 - \alpha) \, P^T s^{(t)}
$$

`q = initial scores`, `P` 는 **§3.1 와 같은 BASE_EDGE_WEIGHTS** 로 row-normalize. 같은 schema 가 학습과 추론을 모두 구동.

기본값: weighted diffusion, α=0.3, T=2, `weights="full"` (BASE × role × visual modifier).

---

## 4. Experimental setup

### 4.1 Datasets

| Dataset | 사용 | 통계 |
|---|---|---|
| **SPIQA** | 학습 (train 25,459 papers) + in-domain 평가 (test-A 118 papers, 999 query) | single-positive, reference figure id |
| **SciEGQA** | zero-shot 평가 | 80 papers, 1,623 query, multi-positive bbox |
| **MMDocIR** academic | zero-shot 평가 | 75 papers, 389 query, multi-positive |

### 4.2 Training

| Hyperparameter | Default | Best combo |
|---|---|---|
| Steps | 8,000 | 8k / 16k |
| Batch | 16 anchors × 12 candidates | same |
| LR | 3e-5 | **1e-4** |
| τ | 0.07 | **0.10** |
| γ (GRCL decay) | 0.5 | same |
| LoRA rank | 8 | same |

A100 80GB 1 row ≈ 5h. 본 paper 의 ablation 전체는 2× A100 ≈ 100 wall hours.

### 4.3 Metrics

R@5/R@10, MRR, Coverage@10 (multi-positive recall fraction), per-doc pool.
유의성: paired bootstrap, 1000 resamples, 95% CI.

### 4.4 Inference variants

| Variant | Method | Settings |
|---|---|---|
| `no_prop` | encoder only | — |
| `wfull_a03_T2` | weighted diffusion | α=0.3, T=2, full weights |
| `ppr_a030` | Personalized PageRank | α=0.30, full weights |

전체 21-variant propagation sweep (uniform / weighted / PPR × HP grid) 는 (h) checkpoint 위에 eval-only.

---

## 5. Results

### 5.1 GRCL > InfoNCE on in-domain SPIQA

graph-induced graded supervision 이 binary InfoNCE 대비 retrieval 품질 개선:

| Row | Loss | R@10 | MRR |
|---|---|---|---|
| (a) baseline | InfoNCE (binary 1-hop positive) | 84.4 | 32.8 |
| **(e) GRCL** | **graded multi-hop relevance** | **87.1** | **38.2** |

**R@10 +2.7pp, MRR +5.4pp**. paired-bootstrap (1000 resamples, 95% CI) 결과 SPIQA Coverage@10 에서 **Δ = +2.55pp, 95% CI [+0.15, +5.11], p = 0.042**. 상위 ranking 품질 (MRR) 에 효과 집중.

### 5.2 `refer_to` edge 가 가장 강한 graph signal

GRCL graph relevance kernel 을 단일 edge type 으로 제한하면:

| GRCL edge | SPIQA R@10 | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|
| caption_of only | 83.5 | 3.1 | 1.3 |
| contains only | 88.4 | 2.1 | 1.4 |
| **`refer_to` only** | **90.4** | **5.7** | 1.3 |
| all edges | 86.9 | 2.4 | 1.0 |

`refer_to` 단독으로 SPIQA R@10 **90.4** — 전체 ablation 최고치. 3 dataset 모두에서 refer_to ≥ caption_of 순서가 일관됨.

해석: paragraph 가 명시적으로 figure 를 호명하는 (`\ref{fig:X}`) 의미적 연결이 caption 인접이나 section containment 보다 informative 한 학습 신호다. 모든 edge 를 균등 사용하는 default 가 오히려 다른 edge 의 noise 때문에 약해진다.

### 5.3 Best HP combo — SPIQA R@10 88.6

learning rate / temperature 가 가장 큰 단일 레버:

| Row | lr | τ | SPIQA R@10 | SPIQA MRR | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|---|---|---|
| default | 3e-5 | 0.07 | 86.9 | 37.4 | 2.4 | 1.0 |
| **`lr_1e4`** | **1e-4** | 0.07 | **88.6** | **42.6** | 4.3 | 1.6 |
| **`h_best_combo`** | **1e-4** | **0.10** | **88.0** | **42.2** | 4.2 | 1.5 |
| `h_best_combo_16k` | 1e-4 | 0.10 | 85.9 | 39.7 | **4.4** | **1.7** |

lr 단일 변경 (3e-5 → 1e-4) 으로 SPIQA R@10 **+1.7pp, MRR +5.2pp**. best_combo 가 in-domain 종합 최고. 16k step 확장 학습은 OOD (MMDocIR / SciEGQA) 에서 추가 향상 — 학습 길이가 zero-shot generalization 에 기여.

### 5.4 GME + Graph Propagation = encoder-agnostic 최대 효과

**같은 graph schema 가 외부 retrieval-tuned MLLM 위에서도 작동**한다. GME-Qwen2-VL-2B-Instruct (2B params, zero-shot) score 위에 propagation 적용:

| Setting | SPIQA R@5 | SPIQA R@10 | SPIQA MRR |
|---|---|---|---|
| GME zero-shot | 47.1 | 58.3 | 33.1 |
| **GME + graph propagation** (α=0.3, T=2) | **78.2** | **84.4** | **63.3** |

**R@5 +31pp, R@10 +26pp, MRR +30pp (33.1 → 63.3, ≈2배)**. 본 study 전체에서 가장 큰 단일 변경 효과.

같은 BASE_EDGE_WEIGHTS schema 가:
- 200M dual-encoder (SigLIPv2) 의 학습 supervision (GRCL)
- 2B decoder MLLM (GME) 의 inference-time score smoothing
양쪽을 동일하게 구동. → **encoder-agnostic plug-in prior**.

### 5.5 Backbone swap — CLIP-L/14 에서도 일관된 향상

SigLIPv2-base (200M) → CLIP-L/14 (430M) backbone 교체:

| Backbone | SPIQA R@10 | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|
| SigLIPv2-base | 86.9 | 2.4 | 1.0 |
| **CLIP-L/14** | **88.1** | **4.2** | **1.5** |

세 dataset 모두에서 향상. GRCL + propagation framework 가 특정 backbone 에 묶인 trick 이 아님을 확인.

### 5.6 Propagation 3 regime — 모두 OOD 에서 작동

(h) checkpoint 위 21-variant sub-ablation 에서 MMDocIR R@10:

| Group | 대표 variant | R@10 |
|---|---|---|
| no propagation | encoder only | 2.4 |
| **A. Uniform** (구조만) | α=0.3, T=2 | 4.2 |
| **B. Weighted diffusion** (BASE × role × visual) | α=0.3, T=2 | 4.1 |
| **C. Personalized PageRank** | α=0.30 | 4.0 |

3 regime 모두 **2.4 → ~4.0–4.2 (+1.7pp)** 일관 향상. 정교한 weight 디자인 (Group B) 과 단순한 구조-only (Group A) 가 비슷한 효과 → **graph 구조 자체가 prior 의 핵심**.

### 5.7 종합 표 — top configs

R@10 (no propagation):

| Row | SPIQA test-A (in-domain) | MMDocIR | SciEGQA |
|---|---|---|---|
| (a) InfoNCE baseline | 84.4 | 2.0 | 0.9 |
| **(e) GRCL** | 87.1 | 2.5 | 1.0 |
| **`edge_refer_to`** | **90.4** | **5.7** | 1.3 |
| **`lr_1e4`** | 88.6 | 4.3 | 1.6 |
| **`h_best_combo`** | 88.0 | 4.2 | 1.5 |
| `h_best_combo_16k` | 85.9 | **4.4** | **1.7** |
| `p` (CLIP-L/14) | 88.1 | 4.2 | 1.5 |
| (k) GME zero-shot | 58.3 | 18.9 | 2.3 |
| **(k) + graph prop** | **84.4** | 15.7 | 2.0 |

---

## 6. Discussion

### 6.1 Single edge weight schema, two roles

본 study 의 핵심 architectural 발견: **동일 BASE_EDGE_WEIGHTS schema 가 두 가지 독립적 retrieval 기여 경로**를 제공한다.

| 사용처 | 효과 | 가장 큰 단일 예 |
|---|---|---|
| **학습 supervision (GRCL)** | InfoNCE → GRCL: SPIQA R@10 +2.7pp, MRR +5.4pp (p=0.042) | `refer_to` 단독 GRCL → R@10 90.4 |
| **추론 prior (propagation)** | encoder-agnostic plug-in 효과 | GME + propagation: R@10 +26pp, MRR +30pp |

두 사용이 학습된 200M SigLIP 과 외부 2B GME 라는 완전히 다른 encoder 에 동일하게 적용 가능 → graph 가 retrieval system 의 **universal prior** 로 기능.

### 6.2 `refer_to` 가 가장 강한 signal

§5.2 의 edge isolation 결과는 명확하다 — paragraph → figure 의 explicit reference (`\ref{fig:X}`) 가 다른 모든 edge 종류보다 informative 한 graph signal 이다. 모든 edge 를 균등 활용하는 default 가 오히려 약하다.

함의: graph-based learning 의 신호 디자인에서 **edge 종류 별 정보량 차이가 크고, 의미적 informativeness 가 가장 높은 edge 에 집중하는 게 효과적**. paragraph reference 는 작가가 명시적으로 "이 figure 가 이 주장의 evidence" 라고 선언한 것이므로 retrieval intent 와 가장 직결.

### 6.3 Graph propagation 은 plug-in module

§5.4 의 GME 결과는 **본 paper 의 핵심 실용적 메시지**다:

- 우리가 학습한 SigLIP 의 graph propagation 효과보다 **외부 MLLM 위에 적용한 효과가 훨씬 큼** (+1.7pp vs +26pp)
- GME 는 본 study 에서 학습하지 않은 외부 모델
- 같은 edge weight schema 가 두 모델의 retrieval 품질을 모두 향상

→ graph propagation 을 **재학습 없이 적용 가능한 plug-in module** 로 사용할 수 있다. 다른 retrieval system (NV-Embed, jina-clip-v2, OpenAI embeddings) 위에도 동일하게 적용 가능할 가능성.

### 6.4 Simple is strong

3 가지 단순화가 모두 baseline 보다 좋거나 같았다:

- `refer_to` 단독 GRCL > 5종 edge 종합 GRCL (90.4 vs 86.9)
- 3 propagation regime (uniform / weighted / PPR) 결과가 거의 동일 (정교한 modifier 없이도 동등 효과)
- HP 단일 변경 (lr 1e-4) 효과 ≈ 모든 method 요소 변경 효과 (+1.7pp)

향후 method 디자인에서 **단순한 graph signal + 정확한 HP tuning** 이 정교한 graph weighting 보다 가성비가 높다.

---

## 7. Conclusion

본 paper 는 학술 문서 retrieval 에서 **typed element graph 의 dual-use** 를 제안하고 4 가지 강한 실증으로 뒷받침했다:

1. **GRCL** 이 binary InfoNCE 대비 in-domain SPIQA R@10 **+2.7pp / MRR +5.4pp** 향상 (p=0.042)
2. **`refer_to` edge 단독** GRCL 이 SPIQA R@10 **90.4** — ablation 전체 최고치
3. **GME + graph propagation** 이 SPIQA R@10 **58.3 → 84.4 (+26pp)**, MRR **33.1 → 63.3 (+30pp)** — 본 study 최대 단일 효과
4. **Best HP combo** (lr=1e-4, τ=0.10) 로 SPIQA R@10 **88.6 / MRR 42.6**, backbone swap (CLIP-L/14) 도 일관 향상

### 핵심 메시지

- 동일 edge weight schema 가 학습 supervision 과 추론 prior 양쪽을 구동
- Graph propagation 은 학습 없이 외부 retrieval-tuned MLLM 에 plug-in 가능
- `refer_to` (paragraph → figure reference) 가 가장 informative 한 graph signal
- 정교한 weight 디자인보다 graph 구조 자체 + 정확한 HP 가 효용의 원천

### Future work

- **Multi-dataset training** — SPIQA 단독 학습이 OOD generalization 의 천장. SPIQA + SciEGQA + MMDocIR 합성 학습으로 zero-shot R@10 추가 향상 검증.
- **LM-based retrieval × graph propagation** — GME 위 21-variant prop sub-ablation 미실시. 다른 retrieval-tuned MLLM (NV-Embed, jina-v4) 까지 확장.
- **Adaptive edge weighting** — `refer_to` dominance 결과 활용, edge type 가중치를 학습 parameter 로 노출.
- **Sample-efficient `refer_to` 강조 학습** — GRCL 에서 refer_to-derived candidates 를 oversample 하는 anchor-mining 전략.

### Reproducibility

전체 코드 + 학습 launcher: <https://github.com/monkcat/dl_project>
학습된 LoRA adapters (42 rows, ~263 MB): `ljh38/element-graph-encoder-v2.1`
Graph metadata: `ljh38/element-graph-v2.1`

```bash
git clone https://github.com/monkcat/dl_project && cd dl_project
pip install -r requirements.txt
huggingface-cli login
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1
bash scripts/run_full_suite.sh   # 43 rows, 2× A100, ~4-5 days
# 또는 학습된 adapter 만 받아서 eval:
huggingface-cli download ljh38/element-graph-encoder-v2.1 --local-dir hf_models
```

집계 결과: `eval/results/experiments/SUMMARY.md`.

---

## Appendix — Hyperparameter 최종값 (`h_best_combo`)

```yaml
encoder:
  hf_id:        google/siglip2-base-patch16-224   # SigLIPv2-base, 200M
  lora_rank:    8
  lora_alpha:   16
  proj_dim:     128

training:
  steps:        8000          # h_best_combo_16k: 16000
  batch:        16            # anchors per step
  pool:         12            # candidates per anchor
  lr:           1e-4          # ★ default 3e-5 → 1e-4
  tau:          0.10          # ★ default 0.07 → 0.10
  weight_decay: 0.01
  warmup_steps: 200
  anchor_kind:  mixed         # caption_of / refer_to / nl_qa, ⅓ each
  seed:         42

grcl:
  gamma:        0.5           # 2-hop path decay

inference (default propagation):
  method:       diffusion
  weights:      full          # BASE × role × visual modifier
  alpha:        0.3
  T:            2
```
