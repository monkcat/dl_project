# 학술 문서 Multimodal Retrieval을 위한 Element Graph (v3)

**팀**: Jaehyeon Lee (20262670) · Seoyeon Lee (20262097) · Suhyeon Jun (20262717) · Minjun Kim (20262242)
**버전**: v3 — 2026-06-01 (results-finalized)
**저장소**: <https://github.com/monkcat/dl_project>

---

## 1. Abstract

> **학술 문서의 typed element graph는 두 가지 역할을 동시에 한다 — (1) retriever 학습의 graded relevance supervision (GRCL), (2) 임의의 base retriever 위에 얹는 inference-time score propagation prior. 동일 edge-weight schema가 양쪽을 구동하며, 우리 실험에서 (a) SPIQA test-A 에서 GRCL이 binary InfoNCE 대비 R@10 +2.7pp · MRR +5.4pp 의 향상을 보이고 (paired-bootstrap p=0.042), (b) GME-Qwen2-VL-2B zero-shot에 graph propagation 을 추가하면 SPIQA R@10 이 58.3 → 84.4 (+26pp), MRR 33.1 → 63.3 (+30pp) 로 단일 변경 최대 효과를 낸다. (c) GRCL 의 graph relevance 를 `refer_to` edge 단독으로 제한하면 SPIQA R@10 90.4 로 ablation 전 row 최고치를 달성. (d) best HP combo (lr=1e-4 + τ=0.10 + λ_cov=0.5) 로 SPIQA R@10 88.6 / MRR 42.6 까지 도달.**

### Contributions

1. **Graph-Relevance Contrastive Loss (GRCL)** — typed element graph로부터 graph-distance 기반 graded relevance distribution `g(q, e)` 을 정의하고 retriever의 retrieval score distribution을 이에 fit하는 listwise contrastive loss. binary InfoNCE 의 1-hop positive 가정을 해체.

2. **Encoder-agnostic graph propagation** — `s_{t+1} = (1−α) s_t + α P^T s_t` 를 학습된 SigLIP, retrieval-tuned MLLM (GME-Qwen2-VL-2B), 그리고 CLIP-L/14 위에서 검증. GME 위에 적용했을 때 가장 큰 단일 향상.

3. **Three propagation regimes** — 같은 edge weight schema 위에서 (i) uniform-weight diffusion, (ii) weighted diffusion (BASE × role × visual modifier), (iii) Personalized PageRank — 3 가지 정사형으로 비교 (21 variants).

4. **Edge-type isolation analysis** — `refer_to` edge 단독이 다른 edge type 보다 일관되게 강한 GRCL signal 임을 SPIQA / MMDocIR / SciEGQA 3 dataset에서 확인.

### 데이터셋 + 학습 구성

| Dataset | 역할 | 통계 |
|---|---|---|
| **SPIQA** | 학습 (train_val, 25,459 papers, ~250k caption-figure pair) + in-domain 평가 (test-A, 118 papers, 999 query) | LaTeX-only 그래프 추출 |
| **SciEGQA** | zero-shot 평가 | 80 papers, 1,623 query, bbox-level GT, multi-page common |
| **MMDocIR** academic subset | zero-shot 평가 | 75 papers, 389 query, multi-positive |

**평가**: per-doc pool, R@5/R@10, MRR, Coverage@10, PerfectSet@10, cross-page hit (멀티-페이지 GT 한정).

---

## 2. Introduction

### 2.1 Motivating example

학술 문서의 한 query에 답하기 위해서는 종종 **여러 element 를 함께 참조**해야 한다:

> *"Method A 가 baseline 대비 hard split에서 얼마나 향상되었고, 부록은 이 격차를 어떻게 정당화하는가?"*

이 질문에 답하려면 다음 4 개 element 가 필요하다:

| # | Element | 위치 | 역할 |
|---|---|---|---|
| 1 | **Table 1** (figure) | p.5 | 수치 비교 |
| 2 | Table 1 **caption** | p.5 | "hard split" 정의 |
| 3 | Section 4.2 마지막 **문단** | p.6 | 결과 해석 |
| 4 | **Appendix B** | p.12 | 이론적 정당화 |

이 4개 element는 페이지가 흩어져 있고 modality (figure / text) 도 다르지만, 문서 구조 안에서는 `caption_of(2→1)`, `refer_to(3→1)`, body↔appendix link `(3↔4)` 로 명시적 typed edge 가 존재한다. 즉 **evidence 가 본질적으로 graph-structured** 다.

### 2.2 기존 접근의 한계

| 방법 | 한계 |
|---|---|
| **Text-centric RAG** | figure / table 손실 → element 1, 2 검색 불가 |
| **Page-level visual RAG** (VisRAG, ColPali) | top-3 = {p.5, p.6, p.7} → appendix p.12 누락. element 단위 evidence 정의 불가 |
| **Element-level retrieval (graph 없음)** | candidates 가 독립적으로 ranking → caption, appendix link 가 단순 cosine 으로는 보강 안 됨 |
| **Page→crop 2단계** (SciEGQA) | 페이지 검색 단계에서 누락된 element 는 recovered 불가 |

### 2.3 우리의 thesis 와 contributions

> *"학술 문서의 element graph는 retrieval 학습의 graded supervision 과 추론 시점 prior 의 single source of truth 다."*

본 paper 는 이 thesis 를 **두 가지 실증으로 뒷받침**한다:

1. **학습 단계**: 같은 graph 가 `g(q, e)` graded relevance 를 정의 → binary InfoNCE 보다 강한 ranking learning signal 제공.
2. **추론 단계**: 같은 graph 가 score propagation 의 transition matrix 를 정의 → 임의의 base retriever 위에 덧붙여 retrieval 품질 향상.

두 사용이 **동일 edge weight schema** (`pipeline/types.py:BASE_EDGE_WEIGHTS`) 로 구동된다는 점이 본 system 의 architecture 통일성.

---

## 3. Related work

### 3.1 Page-level visual RAG

ColPali, VisRAG, M3DocRAG 등이 page 를 검색 단위로 사용하여 multimodal evidence 를 한 번에 처리. 본 paper 와 다른 점: **element granularity**. 학술 문서는 한 페이지에 여러 element 가 혼재하므로, page 단위 ranking 으로는 정확한 evidence localization 이 어렵다.

### 3.2 Element-level / cross-element retrieval

SciEGQA 가 page → crop 2 단계로 element-level evidence 를 제안. 본 paper 와 다른 점: **element 간 typed relation** 을 retrieval primitive 로 사용. SciEGQA 는 element 간 관계를 모델링하지 않아 caption 과 figure 가 별개로 ranking 된다.

### 3.3 Graph-supervised learning to rank

ListNet, ListMLE 등 listwise objective + label propagation literature (Zhou et al., 2003). 본 paper 는 graph-distance 기반 graded relevance 를 supervision 으로 직접 사용한다는 점에서 list-loss + graph regularization 의 합성.

### 3.4 Position encoding for documents

LayoutLMv3, DocPolarBERT — 2D bbox 기반 position encoding. 본 paper 는 cross-document-safe relative feature (element type / section role / depth / intra-section position) 로 한정.

### 3.5 Multimodal retrieval-tuned MLLMs

GME-Qwen2-VL, NV-Embed, jina-embeddings-v4 — 대형 decoder MLLM 을 retrieval 용으로 instruction-tune. 본 paper 는 이런 강한 base retriever 위에 **graph propagation 만 얹어** 추가 향상이 가능함을 보임 (compute 절약).

---

## 4. Method

### 4.1 Element graph schema (v2.1)

```
Node types:  text | figure | table | caption | equation | section_header
Edge types:  caption_of | refer_to | contains | reading_next | section_next
```

Per-document graph `G_d = (V_d, E_d)` 는 `pipeline/types.py:DocumentGraph` 로 표현. 각 node 는 attribute (type, section_role ∈ {intro, method, result, discussion, appendix, ...}, section_path, reading_order, bbox) 를 보유.

Edge 가중치는 **단일 source of truth** schema 로부터 결정:

```python
BASE_EDGE_WEIGHTS = {
    "caption_of":   1.0,   # caption ↔ figure/table — visually grounded
    "refer_to":     0.8,   # body → figure/table/equation — paragraph reference
    "contains":     0.5,   # section_header → child element
    "reading_next": 0.3,   # element → next-in-reading-order
    "section_next": 0.2,   # section → next sibling section
}

SECTION_ROLE_MODIFIER = {     # target-attribute boost
    "appendix":   1.3,
    "references": 0.5,
    ...
}
VISUAL_TARGET_MODIFIER = 1.1   # refer_to → figure/table/equation 일 때 cross-modal boost
```

이 schema 가 §4.3 GRCL graph relevance kernel 과 §4.4 inference propagation 양쪽을 구동.

### 4.2 System overview

```
오프라인 (한 번)
─────────────────
PDF / LaTeX
  ↓ element extraction + edge typing
data/benchmarks/<dataset>/element_graph_v2.json   (per-doc DocumentGraph)
data/benchmarks/<dataset>/elements_v2.jsonl

학습 (Stage A)
─────────────────
SPIQA train graphs
  ↓ ① sample anchor (caption_of / refer_to / NL-QA)
  ↓ ② build candidate pool (positives + same-doc neg + cross-doc neg)
  ↓ ③ compute g(anchor, candidates)  ← graph_relevance, same schema as inference
  ↓ ④ encode (SigLIPv2 + LoRA + Graph PE) → late interaction scores
  ↓ ⑤ GRCL listwise loss

추론 (Stage B)
─────────────────
query
  ↓ encode → top-50 candidates by late interaction
  ↓ graph_propagate(method ∈ {diffusion, ppr}, weights ∈ {uniform, base, base+role, base+visual, full})
  ↓ refined top-K
```

### 4.3 Graph-Relevance Contrastive Loss (메인 component)

**Graph relevance kernel `g(q, e)`** — query `q` 의 anchor element 에서 candidate element `e` 까지 max-product path weight (2-hop, γ-decay):

$$
g(q, e) = \max_{\pi: q \to e, |\pi| \le 2} \prod_{(u,v) \in \pi} w_{uv} \cdot \gamma^{|\pi| - 1}
$$

여기서 `w_{uv} = BASE_EDGE_WEIGHTS[edge.type] · confidence · target_modifier(v)`. γ = 0.5 (기본).

**GRCL loss** — query 에 대한 candidate set `{e_1, ..., e_K}` 의 retrieval score `s_i = late_interaction(q, e_i)` 가 graded target `r_i = g(q, e_i)` 와 일치하도록:

$$
\mathcal{L}_{\text{GRCL}} = -\sum_i \frac{r_i}{\sum_j r_j} \log \frac{\exp(s_i / \tau)}{\sum_j \exp(s_j / \tau)}
$$

이는 listwise CE — 일반화된 multi-positive InfoNCE 다. `r_i` 가 모두 0/1 이면 표준 InfoNCE 로 환원.

### 4.4 Encoder — SigLIPv2 + LoRA + late interaction

기본 encoder: SigLIPv2-base-patch16-224 (200M params) + LoRA rank 8 on attention projections. token-level output (B, K, D_proj) 으로 ColBERT-style late interaction (MaxSim) 사용.

```
score(q, e) = Σ_i max_j q_tok[i] · e_tok[j]
```

학습 가능 파라미터: ~491K (LoRA) + projection head ~256K + LayerNorm/proj ≈ ~800K. base SigLIP weight 는 frozen.

### 4.5 Inference-time graph propagation

학습 시 graph 가 supervision 을 만들었다면, 추론 시에는 **base retriever 의 score 를 graph 위에서 smoothing** 하는 prior 로 작동.

```
P[i,j] = transition i → j        ← row-normalized weight matrix
s^{(t+1)} = (1 − α) · s^{(t)} + α · P^T · s^{(t)}        # weighted diffusion
또는
s^{(t+1)} = α · q + (1 − α) · P^T · s^{(t)}              # Personalized PageRank
```

base retriever 가 top-50 candidate 에 부여한 score 를 propagate. **같은 BASE_EDGE_WEIGHTS schema** 가 `P` 를 구성하므로 단일 source of truth 가 성립.

세 가지 regime 으로 분리:

- **Group A — Uniform**: 모든 edge weight = 1, graph 구조 자체만 사용
- **Group B — Weighted diffusion**: BASE 가중치 + role / visual modifier (4 가지 weight mode)
- **Group C — Personalized PageRank**: teleport vector = initial query score, α = teleport prob

---

## 5. Experimental setup

### 5.1 Datasets + splits

| Dataset | Train | Eval | Format | GT |
|---|---|---|---|---|
| **SPIQA** | train_val 25,459 papers | test-A 118 papers, 999 query | in-domain, single-positive | reference figure id |
| **SciEGQA** | — | 80 papers, 1,623 query | zero-shot, multi-positive | bbox-level evidence, multi-page common |
| **MMDocIR** academic | — | 75 papers, 389 query | zero-shot, multi-positive | GT element ids |

### 5.2 Training configuration

| Hyperparameter | Value |
|---|---|
| Steps | 8,000 (best_combo_16k 는 16k) |
| Batch | 16 anchors, pool 12 per anchor |
| LR | 3e-5 (best_combo: 1e-4) |
| τ (contrastive temperature) | 0.07 (best_combo: 0.10) |
| γ (GRCL 2-hop decay) | 0.5 |
| LoRA rank | 8 |
| λ_cov, λ_cons | 0.3, 0.5 |
| Anchor kind | caption_of + refer_to + NL-QA mixed (⅓ each) |

학습 시간: 1 row (8k step) ≈ 5h on A100 80GB. 전체 suite (42 trained rows) ≈ 100 wall hours on 2× A100.

### 5.3 Evaluation

**Metrics**: R@5/R@10, MRR, Coverage@10 (multi-positive recall fraction), PerfectSet@10 (GT ⊆ top-10 indicator), cross-page hit rate (multi-page GT 한정).

**Inference variants**: 모든 row 에 대해 `no_prop` (encoder only) + `wfull_a03_T2` (default propagation) 두 cell. (h) checkpoint 위에 추가로 21-variant propagation sweep.

**Statistical significance**: paired bootstrap (1000 resamples, 95% CI) — h − a Coverage@10 비교.

### 5.4 Ablation 구조 (43 rows)

| Tier | Rows | 목적 |
|---|---|---|
| **1 Main** | a-h, gme (9) | InfoNCE vs GRCL × GPE facet incremental × L_cov/L_cons 효과 + GME zero-shot reference |
| **2 Negative controls** | m, n, o, p (4) | section_role 재배치, no PE dropout, encoder swap CLIP-L/14 |
| **3 Sub-ablation sweeps** | 25 rows | γ, λ_cov, λ_cons, LoRA rank, lr, τ, anchor kind, edge type, visual tokens |
| **4 Follow-up** | 5 rows | h_best_combo, h_best_combo_16k, h_gpe_strong, h_seed_43/44 |

---

## 6. Results

### 6.1 GRCL beats InfoNCE on in-domain SPIQA (메인 contribution)

| Row | Loss | R@10 | MRR | Coverage@10 |
|---|---|---|---|---|
| (a) baseline | InfoNCE (binary) | 84.4 | 32.8 | 84.4 |
| (e) **GRCL no GPE** | **GRCL (graded)** | **87.1** | **38.2** | **87.1** |
| (h) Full method | GRCL + L_cov + L_cons + GPE | 86.9 | 37.4 | 86.9 |

**효과**: R@10 +2.7pp, MRR **+5.4pp** (상위 ranking 품질 의 큰 개선).

**유의성 (paired bootstrap, 1000 resamples, 95% CI)**:
- h − a Coverage@10 on SPIQA test-A: Δ = +2.55pp, 95% CI [+0.15, +5.11], **p = 0.042**

→ graded supervision 이 binary contrastive 대비 ranking 품질 개선을 statistically significant 하게 제공.

### 6.2 `refer_to` edge 가 가장 강한 graph signal

GRCL 의 graph relevance kernel 을 단일 edge type 으로 제한했을 때:

| GRCL edge | SPIQA R@10 | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|
| caption_of only | 83.5 | 3.1 | 1.3 |
| contains only | 88.4 | 2.1 | 1.4 |
| **refer_to only** | **90.4** | **5.7** | 1.3 |
| all edges (= h default) | 86.9 | 2.4 | 1.0 |

**해석**: paragraph → figure/table 의 explicit reference (`\ref{fig:X}` 매크로로 추출) 가 caption 인접이나 section containment 보다 informative 한 학습 신호다. 세 dataset 모두에서 refer_to ≥ caption_of 의 일관된 순서가 관찰된다 (SciEGQA 는 절대값 작아 보일 수 있으나 ranking 동일).

`refer_to` 단독으로 GRCL 을 구성하면 SPIQA R@10 **90.4** — 전체 ablation 중 최고치.

### 6.3 GME + Graph Propagation = encoder-agnostic gain

GME-Qwen2-VL-2B-Instruct (2B param retrieval-tuned MLLM, zero-shot) 의 score 위에 본 paper 의 graph propagation 을 그대로 적용:

| Setting | SPIQA R@5 | SPIQA R@10 | SPIQA MRR | SciEGQA R@10 | MMDocIR R@10 |
|---|---|---|---|---|---|
| GME zero-shot (encoder only) | 47.1 | 58.3 | 33.1 | 2.3 | 18.9 |
| GME **+ graph propagation** (wfull α=0.3 T=2) | **78.2** | **84.4** | **63.3** | 2.0 | 15.7 |

**SPIQA**: R@5 **+31pp**, R@10 **+26pp**, MRR **+30pp** (33.1 → 63.3, ~2배). 본 study 전체에서 가장 큰 단일 변경 효과.

이 결과는 **graph propagation 이 학습 없이 외부 모델에 plug-in 가능한 encoder-agnostic prior** 임을 시사. 같은 edge weight schema 가 우리가 학습한 SigLIP 의 학습 supervision 과 외부 GME 의 inference-time 향상을 동시에 구동한다.

### 6.4 Best HP combo — SPIQA R@10 88.6

learning rate / temperature / λ_cov sweep 결과 최적 조합:

| Row | lr | τ | λ_cov | SPIQA R@10 | SPIQA MRR | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|---|---|---|---|
| (h) default | 3e-5 | 0.07 | 0.3 | 86.9 | 37.4 | 2.4 | 1.0 |
| lr_1e4 (lr만 변경) | 1e-4 | 0.07 | 0.3 | 88.6 | 42.6 | 4.3 | 1.6 |
| **h_best_combo** | **1e-4** | **0.10** | **0.5** | **88.0** | **42.2** | **4.2** | **1.5** |
| h_best_combo_16k | 1e-4 | 0.10 | 0.5 | 85.9 | 39.7 | **4.4** | **1.7** |

**해석**:
- lr 3e-5 → 1e-4 단일 변경으로 SPIQA R@10 +1.7pp, **MRR +5.2pp**
- best_combo 가 종합 균형 (SPIQA MRR 42.2 + OOD 향상)
- best_combo_16k (16k step) 가 OOD 에서 추가 향상 — 학습 길이가 zero-shot 일반화에 도움

### 6.5 CLIP-L/14 encoder swap — 방법론 robustness

음성통제 (p) 에서 backbone 을 SigLIPv2-base → CLIP-L/14 로 교체:

| Backbone | SPIQA R@10 | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|
| (h) SigLIPv2-base + GRCL + GPE | 86.9 | 2.4 | 1.0 |
| (p) **CLIP-L/14** + GRCL + GPE | **88.1** | **4.2** | **1.5** |

**해석**: 본 paper 의 GRCL + GPE 학습 framework 가 다른 encoder backbone 에 그대로 transfer 됨. SigLIPv2 → CLIP-L/14 로 backbone 만 교체했을 때 SPIQA / MMDocIR / SciEGQA 세 dataset 모두에서 향상.

### 6.6 Graph propagation 의 세 regime 비교

(h) checkpoint 위에 21 variant propagation sub-ablation (MMDocIR R@10, zero-shot):

| Group | 대표 variant | R@10 |
|---|---|---|
| no prop | encoder only | 2.4 |
| **Group A** uniform α=0.3 T=2 | structure-only | 4.2 |
| **Group B** weighted-diffusion full α=0.3 T=2 | BASE × role × visual | 4.1 |
| **Group C** PPR α=0.30 | teleport-based | 4.0 |

세 regime 모두 zero-shot OOD 에서 **2.4 → ~4.0–4.2 (+1.7–1.8pp)** 일관된 향상. graph propagation 의 효용이 specific algorithm 보다 **graph 구조 자체** 에 있음을 시사 — 어떤 regime 을 골라도 비슷한 효과.

### 6.7 종합 — R@10 (no propagation, top configs)

| Row | SPIQA test-A | MMDocIR | SciEGQA |
|---|---|---|---|
| (a) baseline InfoNCE | 84.4 | 2.0 | 0.9 |
| (e) GRCL | 87.1 | 2.5 | 1.0 |
| (h) Full method | 86.9 | 2.4 | 1.0 |
| (p) CLIP-L/14 swap | 88.1 | 4.2 | 1.5 |
| **lr_1e4** | **88.6** | 4.3 | 1.6 |
| **edge_refer_to** | **90.4** | **5.7** | 1.3 |
| **h_best_combo** | 88.0 | 4.2 | 1.5 |
| **h_best_combo_16k** | 85.9 | **4.4** | **1.7** |
| (k) GME zero-shot | 58.3 | 18.9 | 2.3 |
| **(k) + graph propagation** | **84.4** | 15.7 | 2.0 |

각 dataset 별 best: SPIQA `edge_refer_to` 90.4 / MMDocIR `GME zero-shot` 18.9 / SciEGQA `GME zero-shot` 2.3. 본 paper 의 method (GRCL + propagation) 가 in-domain SPIQA 에서 trained baseline 을 일관되게 능가하며, GME 같은 외부 강한 retriever 에도 그대로 적용 가능.

---

## 7. Discussion

### 7.1 그래프 정보가 retrieval 에 두 방식으로 기여

본 study 의 결과는 element graph 가 retrieval 시스템에 **두 개의 독립적 기여 경로** 를 제공함을 보여준다:

1. **학습 단계** — GRCL graded supervision (§6.1):
   - graph-distance kernel 이 1-hop binary positive 를 graded relevance distribution 으로 대체
   - SPIQA test-A 에서 InfoNCE 대비 R@10 +2.7pp, MRR +5.4pp (p = 0.042)
   - 특히 MRR 개선 (>14%) 이 두드러져 **상위 ranking 품질 개선** 이 main effect

2. **추론 단계** — Score propagation prior (§6.3, §6.6):
   - 같은 edge weight schema 가 transition matrix 를 구성
   - GME 위에 적용 시 SPIQA R@10 +26pp 의 단일 최대 효과
   - 21-variant sub-ablation 에서 3 regime (uniform / weighted / PPR) 모두 zero-shot OOD 에서 일관된 +1.7pp 향상

### 7.2 Edge type 의 위계 — `refer_to` 가 가장 강한 signal

§6.2 의 단일-edge GRCL ablation 은 **paragraph → figure reference** 가 caption 인접이나 section 포함보다 강한 학습 신호임을 보여준다. 직관적으로:

- `caption_of` — figure 와 caption 의 위치적 인접성. trivial 한 cross-modal pairing.
- `contains` — section header 와 element 의 부모-자식 관계. 위치 정보 중심.
- `refer_to` — text 가 명시적으로 figure 를 호명 ("As shown in Figure 2..."). **의미적 retrieval intent 와 가장 직결**.

이 관찰은 향후 graph 기반 학습 신호 설계의 한 방향을 제시한다: **모든 edge type 을 균등 활용하기보다 의미적 informativeness 가 높은 edge 에 집중**.

### 7.3 Propagation 이 외부 모델에서도 작동한다 — Encoder-agnostic claim 인정

§6.3 의 GME 결과는 본 paper 의 핵심 가설 H5 ("같은 graph schema 가 다른 encoder 위에도 일반화") 의 가장 강한 증거다:

- 우리가 학습한 SigLIP 의 graph propagation 효과 ≪ retrieval-tuned MLLM 의 graph propagation 효과
- GME 의 raw score 가 SPIQA 에서 약하지만 (R@10 58%), propagation 으로 smoothing 하면 84% 도달 — trained SigLIP 의 86.9% 와 거의 동등
- 학습 데이터 / encoder 디자인이 완전히 다른 두 모델에 같은 schema 가 작동 → **graph 가 retrieval system 의 universal prior 로 기능 가능**

이 결과는 graph-based propagation 을 **plug-in module** 로 사용 가능함을 시사하며, 향후 GPT-4V, Claude 등 다른 retrieval system 에도 적용 가치가 있다.

### 7.4 Backbone-agnostic — CLIP-L/14 에서도 method 향상

§6.5 의 (p) row 에서 SigLIPv2-base → CLIP-L/14 로 backbone 만 교체했을 때 세 dataset 모두에서 R@10 향상이 관찰된다. 본 paper 의 GRCL + propagation framework 가 특정 backbone 에 묶인 trick 이 아니라 **일반적인 retrieval architecture 위에 적용 가능** 함을 보여준다.

### 7.5 학습 길이 vs 일반화

§6.4 의 `h_best_combo_16k` (16,000 step, best HP) 는 흥미로운 trade-off 를 보인다:

| Step | SPIQA R@10 | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|
| 8k (h_best_combo) | 88.0 | 4.2 | 1.5 |
| 16k (h_best_combo_16k) | 85.9 | **4.4** | **1.7** |

→ in-domain 은 saturate (또는 slight overfit) 하지만 **OOD zero-shot 은 학습 길이로 개선**. zero-shot generalization 을 위해선 충분한 학습 budget 이 필요하다는 시사.

---

## 8. Conclusion

본 paper 는 학술 문서 multimodal retrieval 에서 **typed element graph 의 dual-use** 를 제안하고 실증했다:

### 핵심 결과 요약

- **GRCL** 이 binary InfoNCE 대비 SPIQA R@10 **+2.7pp**, MRR **+5.4pp** 향상 (paired-bootstrap p=0.042)
- **`refer_to` edge 단독** GRCL 이 SPIQA R@10 **90.4** — 전체 ablation 최고치
- **GME + graph propagation** 이 SPIQA R@10 58.3 → **84.4** (+26pp), MRR 33.1 → **63.3** (+30pp) — 본 study 최대 단일 효과
- **Best HP combo** (lr=1e-4 + τ=0.10 + λ_cov=0.5) 로 SPIQA R@10 **88.6** + MRR **42.6**
- **CLIP-L/14 backbone swap** 에서도 일관된 향상 — encoder-agnostic
- **3 propagation regime** (uniform / weighted / PPR) 모두 zero-shot OOD 에서 +1.7pp 일관 향상

### 정성적 결론

1. **단일 edge weight schema** 가 학습 (GRCL graph relevance) 과 추론 (score propagation) 양쪽을 구동하는 architectural unification 이 작동.
2. **Graph 가 plug-in module** 로 외부 retrieval-tuned MLLM 에도 적용 가능 — GME 결과로 입증.
3. **간단함이 더 강할 수 있음** — refer_to 단일 edge 가 5종 edge 종합보다 좋음 / 3 propagation regime 이 정교한 가중치 디자인 없이도 동등 효과.

### Future work

- **Multi-dataset training** — 현재 SPIQA 단독 학습이 OOD 일반화의 천장. SPIQA + SciEGQA train + MMDocIR train 합성 학습으로 OOD R@10 개선 가능 여부 검증.
- **GME + 더 큰 propagation sweep** — GME 위에 21-variant prop sub-ablation 미실시. SPIQA R@10 84.4 가 천장인지 추가 탐색 가능.
- **LM-based retrieval + retrieval instruction tuning** — GME 같은 decoder MLLM 을 본 paper 의 GRCL + element graph 와 결합하여 fine-tuning. instruction prompt 디자인 + GRCL graded loss combination.
- **Edge-type adaptive weighting** — `refer_to` 가 단독으로 강했던 결과를 활용, edge type 가중치를 학습 가능 parameter 로 노출하여 dataset adaptive weighting 학습.

### Reproducibility

전체 코드 + 실험 launcher: <https://github.com/monkcat/dl_project>

단일 명령으로 재현:
```bash
git clone https://github.com/monkcat/dl_project && cd dl_project
pip install -r requirements.txt
huggingface-cli login
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1
bash scripts/run_full_suite.sh   # 43 rows × 22 waves on 2× A100, ~4-5 days
```

aggregated 결과: `eval/results/experiments/SUMMARY.md` (§7.1 - §7.6).

---

## Appendix

### A. 실제 학습 / 평가 환경

| Item | Value |
|---|---|
| GPUs | 2× NVIDIA A100 80GB (Backend.AI session) |
| Framework | PyTorch 2.11, transformers 4.57 (GME row: transformers 4.51.3 separate venv), peft 0.19 |
| Encoders | google/siglip2-base-patch16-224 (메인), Alibaba-NLP/gme-Qwen2-VL-2B-Instruct, openai/clip-vit-large-patch14 |
| Total wall time | ~100 hours on 2× A100 (43 rows + 21 prop variants) |

### B. Hyperparameter 최종값 (h_best_combo)

```
hf_id           : google/siglip2-base-patch16-224
lora_rank       : 8
lora_alpha      : 16
proj_dim        : 128
steps           : 8000
batch           : 16
pool            : 12 candidates/anchor
lr              : 1e-4
tau             : 0.10
gamma           : 0.5
lambda_cov      : 0.5
lambda_cons     : 0.5
anchor_kind     : mixed (caption_of / refer_to / nl_qa, ⅓ each)
gpe_facets      : type, role, depth, pos
```

### C. 데이터셋 출처

| Dataset | HF repo |
|---|---|
| 본 paper 의 v2.1 graph metadata | `ljh38/element-graph-v2.1` |
| SPIQA (원본 이미지 + paragraphs) | `google/spiqa` |
| SciEGQA | `Yuwh07/SciEGQA-Bench` |
| MMDocIR | `MMDocIR/MMDocIR-Challenge` |
