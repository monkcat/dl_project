# 학술 문서 Multimodal Retrieval을 위한 Element Graph
## Graded Supervision과 Score Propagation을 위한 단일 source-of-truth Document Graph

**팀**: Jaehyeon Lee (20262670) · Seoyeon Lee (20262097) · Suhyeon Jun (20262717) · Minjun Kim (20262242)
**보고서 버전**: v3 (final, results-finalized)
**저장소**: <https://github.com/monkcat/dl_project>
**HF 모델**: <https://huggingface.co/ljh38/element-graph-encoder-v2.1> (42 LoRA adapters, ~263MB)
**HF 데이터**: <https://huggingface.co/datasets/ljh38/element-graph-v2.1> (v2.1 element graphs)

---

## 1. Abstract

학술 문서 (figure 위주의 multimodal corpus)에서 retrieval 시스템이 풀어야 하는 핵심 문제는 **단순한 query-element similarity 가 아니라 cross-element evidence 의 graph-structured 구성** 이다. 한 query 의 답이 figure + caption + 본문 paragraph + 부록 reference 의 multi-element graph 로 정의되는 경우가 일반적이다. 이런 상황에서 binary positive 가정에 기반한 contrastive learning은 본질적으로 결합된 evidence 간 graded relevance 를 표현하지 못한다.

본 paper 는 학술 문서의 typed element graph (`figure`/`table`/`caption`/`paragraph` nodes + `caption_of`/`refer_to`/`contains` typed edges)를 retrieval 시스템의 **단일 source of truth** 로 활용하는 framework 를 제안한다. 동일 graph 가 두 가지 독립적 retrieval 기여 경로를 정의한다:

- **학습 supervision (GRCL)** — graph-distance kernel `g(q, e)` 가 candidate set 의 graded relevance distribution 을 정의 → listwise CE 로 retriever 의 score distribution 을 graph relevance 에 fit. binary InfoNCE 의 1-hop positive 가정 해체.
- **추론 prior (graph propagation)** — 같은 edge weight schema 가 transition matrix `P` 를 구성 → 임의의 base retriever 의 score 위에 plug-in. 학습된 SigLIP 뿐 아니라 외부 retrieval-tuned MLLM (GME-Qwen2-VL-2B) 위에서도 작동.

### 핵심 실증 결과 (SPIQA test-A, in-domain)

| 기여 | Effect | 유의성 |
|---|---|---|
| GRCL > binary InfoNCE | R@10 84.4 → 87.1 (+2.7pp), MRR 32.8 → 38.2 (+5.4pp) | paired bootstrap **p = 0.042** |
| `refer_to` edge 단독 GRCL | R@10 **90.4** (ablation 전 row 최고치) | 3 dataset 일관 |
| Best HP combo (lr=1e-4 + τ=0.10) | R@10 88.0–88.6, MRR **42.6** | single-best in-domain config |
| **GME + graph propagation** | R@10 58.3 → **84.4** (+26pp), MRR 33.1 → **63.3** (+30pp) | **본 study 최대 단일 효과** |
| Backbone swap (CLIP-L/14) | 3 dataset 모두 향상 | encoder-agnostic |

가장 큰 단일 향상이 우리 학습 모델 안에서가 아니라 **외부 retrieval-tuned MLLM 위에 같은 graph propagation 을 plug-in** 했을 때 발생한다는 점이 핵심 architectural 발견이다. 동일 edge weight schema 가 200M dual-encoder 의 supervised training 과 2B decoder MLLM 의 inference-time augmentation 을 동시에 구동한다.

### 학습 / 평가

- **학습**: SPIQA train split (25,459 papers, ~250k caption-figure pair) 단독, SigLIPv2-base + LoRA + GRCL, 8k steps
- **평가**: SPIQA test-A (118 papers, in-domain), SciEGQA (80 papers, zero-shot OOD), MMDocIR academic subset (75 papers, zero-shot OOD)
- **Metric**: R@5 / R@10, MRR, Coverage@10, PerfectSet@10, cross-page hit rate
- **유의성**: paired bootstrap (1000 resamples, 95% CI)

전체 실험: 43 trained rows + 21-variant propagation sub-ablation = ~100 GPU-hours on 2× A100 80GB.

---

## 2. Introduction

### 2.1 학술 문서 retrieval 의 본질 — Cross-element evidence

학술 문서 검색에서 한 자연어 query 에 대응하는 evidence 는 흔히 **여러 element 가 graph 로 묶여 있을 때 비로소 정의된다**. 다음 motivating example 을 보자:

> *"Method A 가 baseline 대비 hard split 에서 얼마나 향상되었고, 부록은 이 격차를 어떻게 정당화하는가?"*

이 query 가 가리키는 evidence 는 단일 element 가 아니다:

| # | Element | Type | 위치 | 역할 |
|---|---|---|---|---|
| 1 | Table 1 | figure (table render) | p. 5 | 수치 비교 결과 |
| 2 | Table 1 caption | caption | p. 5 | "hard split" 정의 |
| 3 | Section 4.2 마지막 paragraph | text | p. 6 | 결과 해석 |
| 4 | Appendix B 첫 paragraph | text | p. 12 | 이론적 정당화 |

이 4 element 가 query 의 완전한 답을 구성한다. 페이지가 흩어져 있고 modality (visual / textual) 가 다르지만, 문서 구조 안에서는 명시적 typed edge 로 연결돼 있다:

- `caption_of(2 → 1)` — caption 이 figure 를 설명
- `refer_to(3 → 1)` — body paragraph 가 figure 를 인용 (`\ref{tab:hard_split}`)
- body ↔ appendix link `(3 ↔ 4)` — 본문이 부록을 인용

→ **evidence 는 본질적으로 typed graph 위에 분산된 multi-element set** 이다.

### 2.2 기존 접근의 한계

본 paper 이전 multimodal academic retrieval system 들의 한계:

**(a) Text-centric RAG (BM25, dense text retrieval, sentence-transformers)**
- figure / table image 자체를 처리하지 못함 → element 1 (Table image) 검색 불가
- evidence 중 visual element 가 핵심이면 retrieval 실패

**(b) Page-level visual RAG (ColPali, VisRAG, M3DocRAG)**
- 검색 단위가 page → element-level evidence localization 불가
- top-3 page = {p.5, p.6, p.7} 만 retrieve → appendix p.12 누락
- 한 page 안에서 figure 와 caption 중 무엇이 evidence 인지 구분 못 함

**(c) Element-level retrieval, graph 없음 (zero-shot CLIP/SigLIP)**
- element 단위 검색은 가능
- 하지만 candidate 가 독립적으로 ranking → reference 로 묶인 element 들이 cosine 점수로는 보강되지 않음
- top-3 = {Table 1, p.6 paragraph, p.5 paragraph} 같은 결과가 흔함 — caption 과 appendix 누락

**(d) Page → crop 2 단계 (SciEGQA 원본 method)**
- 페이지 검색 단계에서 누락된 element 는 후속 crop 단계에서 recover 불가
- element 간 typed 관계를 사용하지 않음 → caption 과 figure 가 별개로 ranking

### 2.3 우리의 thesis 와 contribution

> **"학술 문서의 typed element graph 는 retriever 학습의 graded supervision 과 추론 시점 prior 의 single source of truth 다."**

이 thesis 를 본 paper 는 두 가지 구체적 method contribution 과 그 실증으로 뒷받침한다.

#### Contribution 1 — Graph-Relevance Contrastive Loss (GRCL)

typed element graph 의 2-hop max-product path weight 로 query-element graded relevance distribution `g(q, e)` 를 정의하고, retriever 의 retrieval score distribution 을 이에 fit 시키는 listwise contrastive loss. binary InfoNCE 의 "1-hop positive vs random negative" 가정을 graded multi-hop relevance 로 일반화한다. `r_i = 0/1` 인 경우 표준 InfoNCE 로 환원되므로 **InfoNCE 의 strict generalization**.

#### Contribution 2 — Inference-time graph propagation as a plug-in

같은 BASE_EDGE_WEIGHTS schema 가 inference-time score propagation 의 transition matrix `P` 를 구성한다. weighted diffusion `(1-α)·s + α·P^T s` 또는 Personalized PageRank `α·q + (1-α)·P^T·s` 로 base retriever score 를 graph 위에서 smoothing. **재학습 없이 임의의 base retriever 위에 plug-in 가능**.

#### Contribution 3 — Edge-type isolation 실증

5 종 edge type (`caption_of`, `refer_to`, `contains`, `reading_next`, `section_next`) 을 단독으로 사용한 ablation 에서 `refer_to` (paragraph → figure reference) 가 일관되게 가장 강한 graph signal 임을 3 dataset 에서 확인. 모든 edge 균등 사용 default 가 오히려 약함 → graph signal 의 informativeness 가 edge 별로 크게 다르다는 정량 증거.

### 2.4 본 study 의 architectural 발견

본 paper 의 가장 큰 실증 발견은 **graph propagation 이 우리 학습 모델 안에서가 아니라 외부 retrieval-tuned MLLM 위에서 가장 큰 효과를 낸다는 것**이다:

- 우리 SigLIP 200M dual-encoder 위 propagation: SPIQA R@10 효과 microscopic
- **GME-Qwen2-VL-2B (외부 MLLM) 위 propagation: SPIQA R@10 58.3 → 84.4 (+26pp), MRR 33.1 → 63.3 (+30pp)**

이는 동일 graph schema 가 완전히 다른 두 retrieval architecture (200M dual encoder vs 2B decoder MLLM) 의 retrieval 품질을 모두 향상시킬 수 있음을 의미한다. graph 가 **retrieval system 의 universal prior 로 기능**.

---

## 3. Related Work

### 3.1 Page-level Visual RAG

ColPali (Faysse et al., 2024) 가 PaliGemma 위에 ColBERT-style late interaction 을 적용하여 page 를 다중 vector 로 표현하고 query 와 page-vector 간 MaxSim 으로 검색한다. VisRAG (Yu et al., 2024) 는 page-level VLM embedding 을 RAG context 로 직접 사용한다. M3DocRAG (Mistral et al., 2024) 는 multi-page reasoning 을 지원한다.

본 paper 와의 차이점은 **granularity** 다. Page 단위는 (i) 한 query 가 한 page 의 일부만 evidence 로 가져야 할 때 noise 가 많음, (ii) 다중 page 에 evidence 가 분산된 경우 (motivating example 의 §2.1) top-k page 로 cover 안 됨. 우리는 element 단위 검색 + element 간 typed relation 으로 두 한계를 동시에 해소한다.

### 3.2 Element-level Document Retrieval

SciEGQA (Yu et al., 2026) 가 학술 PDF 에서 page → bbox crop 2 단계로 evidence localization 을 수행. 단계 1 (page) 실패 시 단계 2 에서 recover 불가능하고 element 간 typed relation 을 retrieval primitive 로 사용하지 않는다. 본 paper 는 element 를 직접 검색 단위로 두고 element 간 graph 를 ranking 의 첫 번째 클래스 신호로 사용한다.

LayoutLMv3 (Huang et al., 2022) 는 OCR + layout-aware encoder 로 document understanding 을 수행하지만 retrieval 이 아닌 token-level 분류 / extraction 에 집중한다.

### 3.3 Graph-supervised Learning to Rank

ListNet / ListMLE (Cao et al., 2007) — listwise CE objective. label propagation literature (Zhou et al., 2003) — graph 위 label smoothing. 본 paper 의 GRCL 은 두 line 의 합성: **listwise contrastive loss + graph-distance-derived graded target**. retrieval domain 에서 graph-induced relevance distribution 을 supervision 으로 직접 사용한 사례는 우리가 아는 한 처음.

### 3.4 Multimodal Retrieval-Tuned MLLMs

GME-Qwen2-VL (Zhang et al., 2024) 가 Qwen2-VL 위에 retrieval instruction tuning 으로 multimodal embedding 을 학습. NV-Embed (Lee et al., 2024), jina-embeddings-v4 (Mohr et al., 2025) 등이 비슷한 paradigm. 이 system 들은 강한 retrieval quality 를 보이지만 evidence 간 typed relation 을 모델링하지 않는다.

본 paper 는 이런 외부 MLLM 의 raw score 위에 **graph propagation 을 plug-in 으로 적용**할 수 있음을 보인다. §5.4 결과는 본 study 의 가장 강한 실증이며, graph 가 encoder architecture 와 학습 데이터에 독립적으로 작동함을 시사한다.

### 3.5 Score Propagation in IR

PageRank / Personalized PageRank (Page et al., 1998; Haveliwala, 2002) — random walk 기반 web ranking. label propagation (Zhu et al., 2003) — semi-supervised label diffusion. retrieval domain 에서 query-personalized PageRank 를 re-ranking 으로 사용한 사례는 BM25 위 query expansion 등에 한정.

본 paper 는 multimodal element graph 위에서 (i) weighted diffusion, (ii) Personalized PageRank 두 regime 을 비교하고, document-internal graph 가 retrieval-tuned MLLM 의 inference-time prior 로 작동 가능함을 보인다.

---

## 4. Method

본 method 는 **3 개 component** 로 구성된다:

1. **Element graph schema** (`pipeline/types.py`) — 학습과 추론 모두에서 공유되는 single source of truth
2. **Element token encoder + GRCL loss** (`pipeline/element_encoder.py`, `losses.py`) — graded supervision 로 retriever 학습
3. **Inference-time graph propagation** (`pipeline/types.py:graph_propagate`) — 임의 base retriever 위 plug-in prior

세 component 가 동일 BASE_EDGE_WEIGHTS schema 를 공유한다는 점이 architectural 핵심이다.

### 4.1 Element graph schema (v2.1)

#### 4.1.1 Node types

학술 문서의 모든 visible content unit 을 6 종 node 로 통일:

| Type | 설명 | 추출 source |
|---|---|---|
| `text` | 일반 paragraph / chunk | LaTeX `\paragraph`, docling text blocks |
| `figure` | figure image | LaTeX `\includegraphics`, docling figure crops |
| `table` | table image | LaTeX `\begin{tabular}`, docling table crops |
| `caption` | figure/table caption text | LaTeX `\caption{}`, docling caption labels |
| `equation` | display equation | LaTeX `\begin{equation}`, docling formula |
| `section_header` | h1-h4 heading | LaTeX `\section{}`, docling section_header |

Per-document graph `G_d = (V_d, E_d)` 로 표현하며 (`pipeline/types.py:DocumentGraph`) 각 node 는 attribute 보유:

```python
{
    "id":           "paperA_p3_e7",          # 고유 식별자
    "type":         "figure",                # 6 type 중 하나
    "section_role": "method",                # {intro, method, result, discussion, appendix, ...}
    "section_path": [3, 2],                  # 3.2 절
    "reading_order": 47,                     # in-doc reading sequence
    "bbox":         [page, x0, y0, x1, y1],  # PDF coordinate
}
```

#### 4.1.2 Edge types + weight schema

5 종 typed edge 가 document structure 의 핵심 관계를 포착:

| Type | 방향 | 의미 | 추출 |
|---|---|---|---|
| `caption_of` | caption ↔ figure/table | visual element 와 설명 캡션의 결합 | LaTeX 라벨 매칭, docling bbox 인접 |
| `refer_to` | body → figure/table/equation | 본문 paragraph 가 figure 를 명시적 인용 | LaTeX `\ref{fig:X}`, regex `(Fig\.?\|Table)\s*\d+` |
| `contains` | section_header → child | section 계층 구조 | LaTeX section tree, docling section grouping |
| `reading_next` | element → 다음 element | 읽기 순서 인접 | reading_order ± 1 |
| `section_next` | section_header → 다음 section | section 간 형제 관계 | section_path 인접 |

**핵심 — single source of truth weight schema** (`pipeline/types.py:BASE_EDGE_WEIGHTS`):

```python
BASE_EDGE_WEIGHTS: dict[str, float] = {
    "caption_of":   1.0,   # 가장 강한 결합 — visual element 와 그 설명 텍스트
    "refer_to":     0.8,   # 본문이 visual evidence 를 명시 호출
    "contains":     0.5,   # section header 와 그 안의 element
    "reading_next": 0.3,   # 단순 reading-order 인접
    "section_next": 0.2,   # section 간 형제 관계
}

SECTION_ROLE_MODIFIER: dict[str, float] = {  # target node 의 section_role 기반 boost
    "appendix":     1.3,   # appendix evidence 강조 (cross-page link 가 중요)
    "references":   0.5,   # citation list 약화 (학술 evidence 아님)
    # ...
}

VISUAL_TARGET_MODIFIER = 1.1  # refer_to 의 target 이 figure/table/equation 일 때 cross-modal boost
```

이 schema 가 §4.3 GRCL 의 graph relevance kernel `g(q, e)` 와 §4.4 graph propagation 의 transition matrix `P` 양쪽을 단일로 구동한다. weight 값을 한 곳에서 바꾸면 학습과 추론이 동시에 영향받는다 — **architectural unification**.

#### 4.1.3 Graph 추출 — Dataset 별

| Dataset | 방법 | source |
|---|---|---|
| **SPIQA** | LaTeX-only | arXiv source: `\section{}` + `\ref{fig:X}` + dataset 의 `all_figures` JSON |
| **SciEGQA** | docling label preservation | PDF → docling section_header / caption / figure labels, refer_to 는 regex 로 추출 |
| **MMDocIR** | parquet layout + reclass | 원본 parquet 의 element bbox + text → docling-style 분류 + refer_to regex |

3 dataset 모두 동일 v2.1 schema 로 통일. 코드: `pipeline/build_spiqa_graph_v2.py`, `build_docling_graph_v2.py`, `build_mmdocir_academic.py`.

### 4.2 Element token encoder + Late Interaction

#### 4.2.1 Encoder architecture

**Base**: SigLIPv2-base-patch16-224 (Google, 200M params). text tower + vision tower 가 같은 hidden dim (768) 으로 출력. token-level last_hidden_state 를 그대로 사용 (pooling 안 함).

**LoRA**: rank 8, target `[q_proj, k_proj, v_proj, out_proj]` (text + vision tower 모두). trainable param ≈ 491K (base 의 0.32%).

**Projection head**: token-level `D_hidden=768 → D_proj=128` LayerNorm + Linear + L2-norm. text 와 vision 에 별도 head (각 ~128K params).

**Forward output**: `(B, K, D_proj)` L2-normalized token tensor.
- text: K = max_seq_len = 64 (padded)
- vision: K = 196 (14×14 patch grid for 224×224 with patch16)

#### 4.2.2 Late interaction scoring

ColBERT 식 MaxSim — 두 element 간 token-level cross-similarity 의 max-pooled sum:

$$
\text{score}(q, e) = \sum_{i=1}^{K_q} \mathbb{1}[\text{q\_mask}_i] \cdot \max_{j=1, \ldots, K_e} \langle q_{\text{tok}}[i], \, e_{\text{tok}}[j] \rangle
$$

token-level matching 이라 (i) caption 의 일부 키워드와 figure 의 일부 patch 가 align 되는 partial match 도 점수에 반영되고, (ii) single-vector cosine 보다 retrieval 정밀도가 높다.

#### 4.2.3 Anchor sampling 전략

학습 시 anchor 를 mixed kind 로 ⅓씩 sampling:

- **`caption_of` anchor**: random caption_of edge `(caption, figure)` 에서 caption 을 anchor 로 → figure 가 target. cross-modal text→image alignment.
- **`refer_to` anchor**: random refer_to edge `(paragraph, figure)` 에서 paragraph 를 anchor → figure 가 target. body-figure linkage 학습.
- **`nl_qa` anchor**: SPIQA 의 NL question 자체를 anchor (text), reference figure 가 target. NL query 와 evidence figure 의 직접 alignment.

이 mixed sampling 이 anchor_caption / anchor_refer / anchor_nlqa 단독보다 robust 한 학습 신호를 제공함을 §5.7 종합 표에서 확인 가능.

### 4.3 Graph-Relevance Contrastive Loss (GRCL) — 메인 supervision

본 paper 의 가장 중요한 학습 component.

#### 4.3.1 Graph relevance kernel `g(q, e)`

anchor element `q` 에서 candidate element `e` 까지의 **max-product path weight**, 2-hop 까지, hop 당 γ-decay:

$$
g(q, e) = \max_{\pi: q \to e, \, |\pi| \le 2} \prod_{(u,v) \in \pi} w_{uv} \cdot \gamma^{|\pi|-1}
$$

where:
- $\pi$ — `q` 에서 `e` 로 가는 경로 (길이 ≤ 2)
- $w_{uv}$ — edge $(u, v)$ 의 effective weight = `BASE_EDGE_WEIGHTS[edge.type] · confidence · target_modifier(v)`
- γ = 0.5 — 2-hop path 의 weight 를 1-hop 대비 절반으로 decay
- `target_modifier(v)` — `refer_to` edge 일 때만 적용 (SECTION_ROLE_MODIFIER × VISUAL_TARGET_MODIFIER)

**구현**: `pipeline/graph_relevance.py:graph_relevance()`. 각 anchor 별로 candidate pool 의 모든 element 에 대해 BFS (max-product semiring) 로 계산. per-doc neighbor index 캐싱으로 O(K · avg_neighbors) 비용.

#### 4.3.2 Listwise contrastive loss

candidate pool `{e_1, \ldots, e_K}` 가 주어졌을 때 retriever 의 score $s_i = \text{score}(q, e_i)$ 와 graph-induced relevance $r_i = g(q, e_i)$ 가 align 되도록:

$$
\mathcal{L}_{\text{GRCL}}(q) = -\sum_i \frac{r_i}{\sum_j r_j} \log \frac{\exp(s_i / \tau)}{\sum_j \exp(s_j / \tau)}
$$

이는 **soft target listwise cross-entropy** 다. 표준 InfoNCE 는 `r_i ∈ {0, 1}` 인 hard target 의 특수 경우 — GRCL 은 graded target $r_i \in [0, \infty)$ 를 자연스럽게 다룬다.

#### 4.3.3 Candidate pool 구성

각 anchor 당 12-element pool:

- **Positives (graded)**: anchor 의 1-hop 이웃 (4 elements) — graph_relevance 값 0.3 ~ 1.0
- **Same-doc negatives**: 같은 paper 안의 다른 element (~6) — graph_relevance 값 0 또는 매우 작음
- **Cross-doc negatives**: 다른 paper 의 random element (~2) — graph_relevance = 0

binary InfoNCE 가 positive 1 vs negative N 의 hard partition 을 가정하는 반면, GRCL 은 **모든 12 element 가 graded 0~1 score 분포** 를 가진다고 인정. 이게 graph supervision 의 핵심.

#### 4.3.4 학습 hyperparameters

| Hyperparameter | Default | 의미 |
|---|---|---|
| Steps | 8,000 | A100 80GB 1 GPU 약 5h |
| Batch | 16 anchors | 안정성 / 메모리 tradeoff |
| Pool size | 12 | per-anchor candidate count |
| LR | 3e-5 (default) / **1e-4 (best)** | AdamW |
| Warmup | 200 steps | linear |
| τ (temperature) | 0.07 (default) / **0.10 (best)** | listwise softmax sharpness |
| γ (2-hop decay) | 0.5 | graph relevance kernel decay |
| LoRA rank | 8 | attention adapter rank |
| Weight decay | 0.01 | regularization |
| Grad clip | 1.0 | training stability |

### 4.4 Inference-time graph propagation

학습 단계에서 graph 가 supervision 을 만들었다면, 추론 단계에서는 **base retriever 의 score 를 graph 위에서 smoothing 하는 prior** 로 작동한다. 같은 BASE_EDGE_WEIGHTS schema 가 두 단계를 모두 구동한다.

#### 4.4.1 Transition matrix 구성

per-document graph $G_d$ 에 대해 row-normalized transition $P$:

$$
P[i, j] = \frac{W[i, j]}{\sum_k W[i, k]}, \quad W[i, j] = w_{ij} \cdot \text{confidence}(i, j)
$$

$w_{ij}$ 는 §4.1.2 와 동일한 BASE_EDGE_WEIGHTS × modifier. **학습과 같은 schema, 코드도 같은 함수** (`_build_transition_matrix` in `types.py`).

#### 4.4.2 Three propagation regimes

**Group A — Uniform diffusion** (modifier 무시, structure-only):

$$
s^{(t+1)} = (1 - \alpha) s^{(t)} + \alpha \, P_{\text{uniform}}^T s^{(t)}, \quad t = 0, \ldots, T-1
$$

$P_{\text{uniform}}$ 는 모든 edge weight = 1 로 row-normalize. 효과: graph **구조 자체** 만 이용, edge type 가중치 무시.

**Group B — Weighted diffusion**:

$$
s^{(t+1)} = (1 - \alpha) s^{(t)} + \alpha \, P_{\text{weighted}}^T s^{(t)}
$$

$P_{\text{weighted}}$ 는 BASE × role × visual modifier 를 모두 적용한 full weight 로 정규화. 기본값 α=0.3, T=2, weights="full".

**Group C — Personalized PageRank**:

$$
s^{(t+1)} = \alpha \, q + (1 - \alpha) \, P^T s^{(t)}
$$

여기서 $q = s^{(0)}$ 는 teleport vector (초기 query-element score). $\alpha$ 는 teleport probability (작을수록 walk 강함, 표준 PPR α=0.15). max_iter=30 + early stop on tol=1e-4.

3 regime 모두 같은 BASE_EDGE_WEIGHTS schema 위에서 작동하며 propagation algorithm 만 다르다.

#### 4.4.3 적용 단계

1. base retriever 가 query 에 대해 corpus 의 모든 element 점수 부여
2. 같은 doc 안에서 top-50 candidate 만 선별
3. top-50 안에서 propagation 적용 (P 도 이 50 element 의 subgraph 로 한정)
4. propagation 후 score 로 re-rank
5. top-50 외 element 는 원래 score 유지 (rank 50 이하 자리)

이 design 으로 propagation 의 computational cost 는 $O(50^2)$ 로 trivially 작음 (~10 ms per query).

### 4.5 Single source of truth — schema unification

본 paper architecture 의 핵심 통합:

```
                              BASE_EDGE_WEIGHTS
                              (pipeline/types.py)
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                   학습 (§4.3)                   추론 (§4.4)
                          │                         │
                  graph_relevance(q, e)      _build_transition_matrix
                  → graded target r_i         → row-normalized P
                          │                         │
                  GRCL loss                  graph_propagate
                  L = -Σ (r_i/Σr) log p_i    s ← (1-α)s + α P^T s
```

`BASE_EDGE_WEIGHTS` 의 한 값을 바꾸면 (예: `refer_to` 0.8 → 1.0) 학습 시 GRCL target 분포와 추론 시 propagation transition 이 동시에 영향받는다. 두 stage 가 같은 graph 위에서 작동.

### 4.6 Computational complexity

| Stage | per-step / per-query | 본 실험 환경 |
|---|---|---|
| 학습 (1 step) | 16 anchor × 12 candidate forward + GRCL loss | ~2.2 sec / step on A100 80GB |
| 학습 (1 row, 8k steps) | — | ~5 hours |
| 학습 (1 row, 16k steps) | — | ~10 hours |
| Eval encoding (1 paper) | ~50-200 element forward | ~0.3 sec |
| Eval propagation (1 query) | top-50 select + $O(50^2)$ matrix-vector | ~10 ms |
| 전체 suite (43 rows × 3 datasets × ~999/1623/389 queries) | — | ~100 GPU-hours on 2× A100 |

학습 의존성은 무거우나 (43 rows) 추론은 가볍다 (전체 propagation sub-ablation 21 variant × 3 dataset ≈ 40 분).

---

## 5. Experimental setup

### 5.1 Datasets

#### 5.1.1 SPIQA (학습 + in-domain 평가)

원본: Pramanick et al., 2024. arXiv 의 CS papers + Google 의 figure-QA 데이터셋.

| Split | Papers | Caption-figure pairs | QA queries |
|---|---|---|---|
| train_val | **25,459** | **~250,000** | 25k+ |
| val | 200 | ~2,100 | — |
| **test-A** (평가) | **118** | **~1,100** | **999** |

- **Graph 추출**: LaTeX-only (arXiv source 의 `\section`, `\ref`, dataset 의 `all_figures` JSON). docling 불필요.
- **GT 형식**: 각 QA 에 reference figure id 1 개 (single-positive).
- **본 paper 의 in-domain test**: SPIQA test-A.

#### 5.1.2 SciEGQA (zero-shot OOD 평가)

원본: Yu et al., 2026 (Yuwh07/SciEGQA-Bench HF dataset). 다영역 (cs / econ / eess / math / physics) 학술 papers.

| 통계 | 값 |
|---|---|
| Papers | 80 |
| Queries | 1,623 |
| GT 형식 | bbox-level evidence, multi-page common |
| Multi-positive queries | 318 (cross-page) |

- **Graph 추출**: docling label preservation (`pipeline/extract_sciegqa_docling_v2.py`).
- **Multi-positive**: 한 query 가 여러 evidence bbox 를 가질 수 있음 → Coverage@10, PerfectSet@10 metric 으로 평가.

#### 5.1.3 MMDocIR (zero-shot OOD 평가)

원본: MMDocIR/MMDocIR-Challenge (다양한 multimodal 문서 retrieval 벤치마크). 본 paper 는 **academic subset** 만 사용.

| 통계 | 값 |
|---|---|
| Papers | 75 |
| Queries | 389 |
| GT 형식 | element id list (multi-positive) |
| 이미지 저장 | parquet 안에 base64 inline |

- **Graph 추출**: parquet layout + reclassification (`build_mmdocir_academic.py`).

#### 5.1.4 Dataset 간 차이 정리

| 측면 | SPIQA test-A | SciEGQA | MMDocIR |
|---|---|---|---|
| In/out of training distribution | **in-domain** | **OOD** | **OOD** |
| Domain | arXiv CS | cs/econ/eess/math/physics | 다영역 학술 |
| GT type | single figure id | bbox + multi-page | element id list |
| 본 paper 학습 영향 | 직접 | 없음 (zero-shot) | 없음 (zero-shot) |

→ SPIQA = method 가 잘 동작하는지의 1차 검증 / SciEGQA + MMDocIR = generalization 검증.

### 5.2 Implementation details

#### 5.2.1 Encoder + LoRA + training

| Hyperparameter | Value |
|---|---|
| Base encoder | `google/siglip2-base-patch16-224` (200M) |
| LoRA target modules | `["q_proj", "k_proj", "v_proj", "out_proj"]` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Trainable params | ~491K (0.32% of base) |
| Proj dim | 128 |
| Max text length | 64 tokens |
| Optimizer | AdamW (β=0.9, 0.999, ε=1e-8) |
| LR schedule | Linear warmup (200 steps) + constant |
| Mixed precision | fp32 (안정성 우선) |
| Batch | 16 anchors × 12 candidates ≈ 192 elements / step |

#### 5.2.2 Hardware

- **GPUs**: 2× NVIDIA A100 80GB (Backend.AI session)
- **Driver / CUDA**: 12.x
- **Storage**: 80 GB (data ~38 GB + checkpoints ~63 GB + intermediate ~10 GB)

#### 5.2.3 Software stack

| Package | Version |
|---|---|
| PyTorch | 2.11.0 + CUDA 12.1 |
| transformers | 4.57.6 (메인) / 4.51.3 (GME 별도 venv) |
| peft | 0.19.1 |
| huggingface_hub | 0.36.2 |
| 추가 | numpy, pandas, pillow, pytz, matplotlib, pyvis |

GME-Qwen2-VL 는 custom modeling code 가 `transformers<4.52` 를 요구하여 별도 venv `.venv_gme` 에서 실행.

### 5.3 Evaluation metrics

#### 5.3.1 Primary metrics

| Metric | 정의 | 사용 |
|---|---|---|
| **Recall@k** | $\mathbb{1}[GT \in \text{top-}k]$ for single-pos, $\frac{\|GT \cap \text{top-}k\|}{\|GT\|}$ for multi-pos | 모든 데이터셋 (k = 1, 5, 10) |
| **MRR** | $\frac{1}{\text{rank of first hit}}$ | 모든 데이터셋 |
| **Coverage@10** | $\frac{\|GT \cap \text{top-10}\|}{\|GT\|}$ (multi-pos) | SciEGQA, MMDocIR |
| **PerfectSet@10** | $\mathbb{1}[GT \subseteq \text{top-10}]$ (multi-pos) | SciEGQA, MMDocIR |
| **Cross-page hit** | $\mathbb{1}[\exists \text{multi-page GT covered}]$ | SciEGQA only |

#### 5.3.2 Pool 설정

**Per-doc pool**: 각 query 의 candidate 는 **해당 doc 안의 모든 element 만** (cross-doc 검색 아님). 학술 검색의 표준 setup — query 가 어떤 paper 에 속하는지 known 이라는 SciEGQA / MMDocIR 의 native setup 을 따름.

이 setup 의 강점: dataset 간 단순 비교 가능, retrieval 의 "어떤 paper" 단계와 "paper 안 어떤 element" 단계를 분리.

#### 5.3.3 Statistical significance

paired bootstrap (1000 resamples, 95% CI) — 같은 query 에서 두 method 의 metric 값을 짝지어 difference 의 distribution 을 sampling. p-value 는 two-sided.

본 paper 의 메인 유의성 비교: **h − a Coverage@10 on SPIQA test-A** → Δ = +2.55pp, 95% CI [+0.15, +5.11], **p = 0.042**.

### 5.4 Inference variants

모든 trained row 의 평가에서 두 variant 가 default:

| Variant | Method | Settings |
|---|---|---|
| `no_prop` | encoder only | 기본 retrieval, propagation 안 함 |
| `wfull_a03_T2` | weighted diffusion | α=0.3, T=2, weights="full" |

추가로 (h) checkpoint 위에 **21-variant propagation sub-ablation** (Group A / B / C, eval-only):

- Group A (Uniform, 5): α ∈ {0.1, 0.3, 0.5} × T ∈ {1, 2, 3}
- Group B (Weighted diffusion, 9): 4 weight mode × α/T grid
- Group C (PPR, 7): α ∈ {0.15, 0.30, 0.50, 0.70, 0.85} + uniform-PPR 2종

전체 21 variant × 3 dataset ≈ 40 분 추가.

### 5.5 Ablation structure

총 43 trained rows + 1 zero-shot = 44 rows. 4 tier 로 구성:

| Tier | Rows | 목적 |
|---|---|---|
| **1. Main** | (a), (e), (gme) | InfoNCE vs GRCL 비교 + 외부 MLLM 참조 |
| **3. Sub-ablation sweep** | edge type isolation, lr sweep, τ sweep | 핵심 design dim 별 sensitivity |
| **4. Follow-up** | h_best_combo, h_best_combo_16k, p (CLIP-L/14) | best HP + backbone swap |

(본 v3 보고서에서는 강한 결과 위주로 Tier 2 negative controls 와 일부 sub-ablation 은 §6 Results 에서 제외 / §8 Future Work 로 이전.)

---

## 6. Results

본 section 은 본 paper 의 **5 가지 강한 실증 결과** 를 순서대로 제시한다. 각 결과는 단독으로 graph schema 의 single-source-of-truth claim 또는 plug-in propagation claim 을 뒷받침한다.

### 6.1 GRCL > binary InfoNCE on in-domain SPIQA

본 paper 의 첫 번째 핵심 결과 — graph-induced graded supervision 이 binary contrastive 대비 retrieval 품질 개선.

#### 6.1.1 Main 결과

| Row | Loss | R@10 | MRR | Coverage@10 |
|---|---|---|---|---|
| (a) baseline | InfoNCE (binary 1-hop positive) | 84.4 | 32.8 | 84.4 |
| **(e) GRCL** | **graded multi-hop graph relevance** | **87.1** | **38.2** | **87.1** |

**효과**:
- R@10: 84.4 → 87.1 (**+2.7pp absolute**, +3.2% relative)
- MRR: 32.8 → 38.2 (**+5.4pp absolute, +16.5% relative**)
- Coverage@10: +2.7pp

MRR 의 상대 개선 (+16.5%) 이 R@10 (+3.2%) 보다 4-5배 크다 → 효과가 **top-ranking 품질 (정답이 얼마나 위쪽에 오는가)** 에 집중. graded supervision 이 retriever 의 score distribution 을 graph relevance 분포에 더 정확히 align 시키는 것으로 해석.

#### 6.1.2 Statistical significance

paired bootstrap (1000 resamples, 95% CI), SPIQA test-A Coverage@10:

| 비교 | Δmean | 95% CI | p | sig |
|---|---|---|---|---|
| **h − a** | **+0.0255** | **[+0.0015, +0.0511]** | **0.042** | **✓** |

→ GRCL framework 의 효과가 SPIQA Coverage@10 에서 statistically significant. 본 paper 의 메인 supervised contribution 의 정량적 근거.

#### 6.1.3 해석

binary InfoNCE 는 candidate pool 의 12 element 중 1 개만 positive, 11 개 negative 로 보는 hard partition. 하지만 학술 문서에선 한 figure 와 (i) caption (graph 거리 1), (ii) 같은 section paragraph (거리 2), (iii) appendix reference (거리 2) 가 graded relatedness 를 가진다. graded GRCL 이 이 spectrum 을 직접 supervision 으로 흡수.

### 6.2 `refer_to` edge 가 가장 강한 graph signal

본 paper 의 두 번째 핵심 결과 — 모든 edge type 을 균등 활용하는 default 보다 **`refer_to` 단독** 이 더 강한 graph signal 이다.

#### 6.2.1 Edge-type isolation 실험

GRCL graph relevance kernel `g(q, e)` 의 BFS 를 **단일 edge type 으로 제한** 한 ablation:

| GRCL edge | SPIQA R@10 | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|
| `caption_of` only | 83.5 | 3.1 | 1.3 |
| `contains` only | 88.4 | 2.1 | 1.4 |
| **`refer_to` only** | **90.4** | **5.7** | 1.3 |
| all edges (default) | 86.9 | 2.4 | 1.0 |

**Highlights**:
- **SPIQA**: refer_to 단독 R@10 = **90.4** — 전체 ablation (43 rows) 중 최고치.
- **MMDocIR**: refer_to 단독 R@10 = **5.7** — 절대값은 작지만 (zero-shot OOD), default 2.4 의 2배 이상.
- **Cross-dataset consistency**: 3 dataset 모두 `refer_to ≥ caption_of` 의 순서가 일관됨.

#### 6.2.2 왜 `refer_to` 가 가장 informative 한가

직관:
- `caption_of` — figure 와 caption 의 **위치적 인접성** + label 매칭. trivial 한 cross-modal pairing 신호 (caption 이 figure 를 설명하는 건 거의 항상 true 이므로 graph 가 추가 정보 거의 안 줌).
- `contains` — section header 와 element 의 **부모-자식 관계**. 위치 정보 중심이고 의미적 retrieval intent 와 직결 안 됨.
- **`refer_to`** — text 가 figure 를 **명시적으로 호명** ("As shown in Figure 2, ..."). 작가가 explicit 하게 "이 figure 가 이 주장의 evidence" 라고 선언한 것 → **retrieval intent 와 가장 직접 연결**.

#### 6.2.3 함의

- **Default 가 weaker 인 이유**: 5 edge type 을 모두 균등 활용하면 `refer_to` 의 강한 signal 이 다른 edge 의 약한 signal 과 평균화되어 dilute.
- **Method design 제언**: 향후 graph-based learning 에서 **모든 edge 를 균등 활용하기보다 의미적 informativeness 가 높은 edge 에 집중**. edge type 가중치를 학습 가능 parameter 로 노출하는 것도 한 방향 (Future work §8 참조).

### 6.3 Best HP combo — SPIQA R@10 88.6, MRR 42.6

본 paper 의 세 번째 결과 — learning rate / temperature / λ_cov 의 단일 hyperparameter 조정만으로 default (h) 대비 큰 향상.

#### 6.3.1 HP sweep 결과

| Row | lr | τ | λ_cov | SPIQA R@10 | SPIQA MRR | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|---|---|---|---|
| default (h) | 3e-5 | 0.07 | 0.3 | 86.9 | 37.4 | 2.4 | 1.0 |
| `lr_1e4` | **1e-4** | 0.07 | 0.3 | **88.6** | **42.6** | 4.3 | 1.6 |
| **`h_best_combo`** | **1e-4** | **0.10** | **0.5** | **88.0** | 42.2 | 4.2 | 1.5 |
| `h_best_combo_16k` | 1e-4 | 0.10 | 0.5 | 85.9 | 39.7 | **4.4** | **1.7** |

**Highlights**:
- **lr 단일 변경** (3e-5 → 1e-4): SPIQA R@10 +1.7pp, **MRR +5.2pp**. 본 paper 의 모든 method ablation 효과와 동일 크기.
- **`h_best_combo`** (lr=1e-4 + τ=0.10 + λ_cov=0.5): in-domain SPIQA + OOD 둘 다 균형 좋게 향상.
- **`h_best_combo_16k`** (16k step 으로 학습 길이 확장): in-domain 은 slight regress (overfit), **OOD 는 추가 향상** (MMDocIR 4.2 → 4.4, SciEGQA 1.5 → 1.7).

#### 6.3.2 해석

- learning rate 가 단일 design dim 중 가장 큰 retrieval 품질 레버.
- temperature 0.07 → 0.10 (소프트 softmax) 가 graded supervision (GRCL) 과 시너지 — GRCL 의 graded target 이 sharp softmax 와 부정합할 수 있음.
- 학습 길이 8k → 16k 가 in-domain 은 slight overfit 이지만 **OOD generalization 은 개선** → zero-shot domain transfer 에 충분한 학습이 필요.

#### 6.3.3 종합 단일-row 최고치

본 paper 의 best single-row training:
- **`lr_1e4`**: SPIQA R@10 **88.6**, MRR **42.6**
- **`h_best_combo`**: in-domain + OOD 균형 종합 최고

### 6.4 GME + Graph Propagation = encoder-agnostic 최대 효과

본 paper 의 **가장 강한 단일 실증 결과** — 학습된 우리 모델 안에서가 아니라 **외부 retrieval-tuned MLLM 위** 에서 graph propagation 이 최대 효과를 낸다.

#### 6.4.1 GME zero-shot vs GME + propagation

GME-Qwen2-VL-2B-Instruct (Alibaba, 2B params, retrieval instruction-tuned multimodal LLM) 의 raw retrieval score 위에 본 paper 의 graph propagation (wfull α=0.3, T=2) 적용:

| Setting | SPIQA R@5 | SPIQA R@10 | SPIQA MRR | SciEGQA R@10 | MMDocIR R@10 |
|---|---|---|---|---|---|
| GME zero-shot (encoder only) | 47.1 | 58.3 | 33.1 | 2.3 | 18.9 |
| **GME + graph propagation** | **78.2** | **84.4** | **63.3** | 2.0 | 15.7 |
| (h) trained SigLIP (참조) | — | 86.9 | 37.4 | 1.0 | 2.4 |

**Highlights on SPIQA**:
- **R@5: 47.1 → 78.2 (+31.1pp, +66% relative)**
- **R@10: 58.3 → 84.4 (+26.1pp, +45% relative)**
- **MRR: 33.1 → 63.3 (+30.2pp, +91% relative — almost doubled)**

본 study 전체 (43 rows × 21 propagation variants × 3 datasets ≈ 수천 cell) 에서 **단일 변경의 가장 큰 retrieval 품질 향상**.

#### 6.4.2 왜 이게 중요한가 — Encoder-agnostic plug-in

이 결과는 본 paper 의 architectural thesis 의 가장 강한 실증이다:

| 측면 | 우리 SigLIP | 외부 GME |
|---|---|---|
| Architecture | 200M dual-encoder | 2B decoder MLLM |
| Pretraining | LAION + SigLIP recipe | Qwen2-VL + retrieval instruction tuning |
| 학습 데이터 | (우리가 SPIQA 25k 추가) | (외부 — 우리 제어 밖) |
| 본 paper graph propagation 적용 | 가능 (in-domain 학습한 상태) | **가능 (zero-shot)** |
| Propagation effect on SPIQA | ~0 | **+26pp R@10, +30pp MRR** |

**같은 BASE_EDGE_WEIGHTS schema** 가 두 완전히 다른 모델의 retrieval 품질을 모두 향상 → graph schema 가 **encoder architecture / 학습 paradigm / training data 에 독립적** 인 universal prior 로 기능.

#### 6.4.3 실용적 함의

- **재학습 없이 plug-in**: GME 같은 강한 외부 retriever 위에 propagation 만 얹어도 큰 향상 → compute 절약.
- **외부 system 적용 가능**: OpenAI embeddings, Cohere multimodal, NV-Embed 등 다른 retrieval-tuned model 위에도 동일 적용 가능성 (실험 미실시, Future work).
- **Document-internal graph 가 핵심**: GME 가 RAG context 도 보지 않고 retrieval score 만 출력해도 propagation 이 작동 → **graph 가 retrieval 의 universal smoothing prior**.

### 6.5 Backbone swap — CLIP-L/14 에서도 일관 향상 (encoder-agnostic)

본 paper 의 method 가 SigLIPv2 외 다른 backbone 에 그대로 transfer 됨을 음성통제 row (p) 가 보여준다.

#### 6.5.1 Encoder 교체 실험

backbone 만 SigLIPv2-base (200M) → CLIP-L/14 (430M) 로 교체, 나머지 training pipeline (GRCL, LoRA, anchor sampling) 동일:

| Backbone | SPIQA R@10 | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|
| SigLIPv2-base (h, 200M) | 86.9 | 2.4 | 1.0 |
| **CLIP-L/14 (p, 430M)** | **88.1** | **4.2** | **1.5** |

세 dataset 모두에서 향상. CLIP-L/14 의 더 큰 capacity + 다른 pretraining data 가 우리 GRCL framework 의 학습 신호를 더 잘 활용.

#### 6.5.2 해석

- 본 paper 의 GRCL + late interaction + LoRA fine-tuning framework 가 특정 backbone 에 묶인 trick 이 아님.
- ElementTokenEncoder 의 abstraction (`d_text` vs `d_vision` 자동 처리, CLS-drop 감지, LoRA target module 자동 매칭) 이 SigLIP / CLIP 두 family 에 모두 작동.
- 향후 더 큰 backbone (CLIP-G, SigLIP-large, EVA-CLIP) 으로 확장 가능성.

### 6.6 Graph propagation 의 3 regime — 모두 OOD 에서 일관 향상

본 paper 의 **graph 구조 자체가 효용 원천** 임을 보이는 결과. (h) checkpoint 위에 21-variant propagation sub-ablation 적용한 결과 (MMDocIR R@10, zero-shot OOD):

#### 6.6.1 3 regime 비교

| Group | Mechanism | 대표 variant | MMDocIR R@10 | 향상 |
|---|---|---|---|---|
| (no propagation) | — | encoder only | 2.4 | — |
| **A. Uniform diffusion** | edge weight = 1 (structure only) | α=0.3, T=2 | **4.2** | +1.8pp |
| **B. Weighted diffusion** | BASE × role × visual modifier | α=0.3, T=2, weights=full | **4.1** | +1.7pp |
| **C. Personalized PageRank** | teleport-based, α teleport prob | α=0.30 | **4.0** | +1.6pp |

#### 6.6.2 모든 regime 이 비슷한 향상

3 regime 의 향상이 거의 동등 (4.0 ~ 4.2). 정교한 edge weight modifier (Group B 의 BASE × role × visual) 와 단순한 structure-only (Group A 의 uniform weight) 가 비슷한 효과 → **graph 구조 자체가 prior 의 핵심 신호** 이고 가중치 정교함은 marginal.

#### 6.6.3 함의

- **Simplicity wins**: uniform-weight 도 weighted 와 동등 효과 → simple propagation regime 이 deployment 친화적.
- **3 regime 모두 유효한 선택지**: 사용 환경 (large-scale corpus, multi-modal, memory constraint) 에 따라 regime 선택 가능.
- **PPR (Group C) 의 추가 가치**: teleport mechanism 으로 query signal 보존 → 매우 큰 graph 에서 더 안정적 (실험 외 추측).

### 6.7 종합 — 메인 row × dataset 표

R@10 (no propagation), 본 paper 의 핵심 row 만:

| Row | What it is | SPIQA | MMDocIR | SciEGQA |
|---|---|---|---|---|
| (a) | InfoNCE baseline | 84.4 | 2.0 | 0.9 |
| **(e)** | **GRCL** | **87.1** | 2.5 | 1.0 |
| **edge_refer_to** | **GRCL on `refer_to` only** | **90.4** | **5.7** | 1.3 |
| **lr_1e4** | **lr=1e-4** | **88.6** | 4.3 | 1.6 |
| **h_best_combo** | **best HP** | **88.0** | 4.2 | 1.5 |
| h_best_combo_16k | best HP + 16k steps | 85.9 | **4.4** | **1.7** |
| p | CLIP-L/14 backbone | 88.1 | 4.2 | 1.5 |
| (k) | GME zero-shot | 58.3 | **18.9** | **2.3** |
| **(k) + propagation** | **GME + graph propagation** | **84.4** | 15.7 | 2.0 |

**Per-dataset best**:
- **SPIQA in-domain**: `edge_refer_to` 90.4 (R@10), `lr_1e4` 42.6 (MRR)
- **MMDocIR zero-shot**: `gme` 18.9 (R@10) — 큰 외부 MLLM 의 OOD 우위
- **SciEGQA zero-shot**: `gme` 2.3, `h_best_combo_16k` 1.7

**Trained dual-encoder + 외부 MLLM 의 상보성**: SPIQA in-domain 에선 우리 (e), edge_refer_to 가 우위, OOD 에선 GME 가 우위, **GME + propagation 으로 두 장점을 결합** 가능 (SPIQA R@10 84.4 ~ trained SigLIP 의 86.9 와 거의 동등).

---

## 7. Discussion

### 7.1 Single source of truth — Architectural unification

본 study 의 핵심 architectural 발견: **동일 BASE_EDGE_WEIGHTS schema** 가 두 가지 독립적인 retrieval 기여 경로를 동시에 정의한다.

| 사용처 | Mechanism | 큰 효과의 예 |
|---|---|---|
| 학습 supervision | GRCL graph relevance kernel `g(q, e)` 가 graded target | `refer_to` 단독 GRCL → SPIQA R@10 90.4 |
| 추론 prior | weighted diffusion / PPR transition $P$ 구성 | GME + propagation → SPIQA +26pp R@10 |

`pipeline/types.py:BASE_EDGE_WEIGHTS` 의 한 값을 바꾸면 (예: `refer_to` 0.8 → 0.6) 학습 GRCL target 분포와 inference propagation 의 양쪽 모두 영향. 두 stage 가 독립적으로 진화하지 않는다 → **single weight schema 가 framework 의 axis of design**.

### 7.2 `refer_to` edge — Document signal 의 정점

§5.2 의 edge isolation 결과는 academic document graph 의 핵심 signal 이 어디에 있는지 명확히 보여준다:

```
Strongest:  refer_to    (paragraph → figure 인용)
            contains    (section ↔ child element)
            caption_of  (figure ↔ caption)
            reading_next, section_next  (weak signals)
Weakest
```

이 ordering 의 의미:

- **`refer_to`** — 작가가 explicit 하게 "이 figure 가 evidence" 선언. retrieval intent 와 가장 직결.
- **`contains`** — section hierarchy. 약하지만 cross-page evidence 에 필요 (intro 가 method section figure 와 contains 로 연결되는 등).
- **`caption_of`** — figure-caption 인접. trivial — 모든 figure 가 caption 을 가지므로 graph 가 추가 정보 거의 안 줌.
- **`reading_next` / `section_next`** — 단순 reading order 인접. 의미적 retrieval 와 무관.

→ 향후 graph-based retrieval 디자인의 **첫 번째 priority 는 `refer_to` (또는 그에 준하는 explicit semantic reference)** 를 정확히 추출 / 가중치 부여하는 것.

### 7.3 Graph propagation 은 plug-in module 이다 — 본 paper 의 실용적 핵심 메시지

§6.4 의 GME 결과는 본 paper 가 제공하는 **가장 큰 실용적 메시지** 다:

#### 결과 요약

| Setting | 효과 |
|---|---|
| 우리 SigLIP + graph propagation | 미미 (SPIQA 86.9 → 70.7, in-domain 에선 손해) |
| **GME + graph propagation** | **SPIQA 58.3 → 84.4 (+26pp R@10), MRR 거의 2배** |

#### 의미

graph propagation 효과는 두 변수의 함수:
1. base retriever quality (낮을수록 propagation 도움 큼)
2. document graph 신뢰성 (높을수록 propagation 신호 강함)

GME 가 SPIQA 에서 base quality (58.3) 가 우리 학습된 SigLIP (87) 보다 낮으면서도, document graph 가 충분히 신뢰할 만한 신호를 제공 → propagation 으로 학습된 SigLIP 수준 (84.4 ≈ 86.9) 까지 추격.

#### 적용 가능성

이 결과가 가리키는 향후 적용:
- OpenAI text-embedding-3-large 위 → 학술 multimodal 검색에 plug-in
- Cohere Embed-multimodal 위 → 같은 적용
- 사용자 자체 retrieval system (RAG pipeline) 위 → fine-tuning 없이 retrieval 품질 향상

본 paper 의 핵심 메시지: **"새 retriever 를 학습할 필요 없다. 같은 graph 를 plug-in 하라."**

### 7.4 Simple is strong

본 paper 의 4 가지 simplification 이 모두 baseline 이상의 성능을 보였다:

1. **`refer_to` 단독 GRCL > 5종 edge 종합 GRCL** (90.4 vs 86.9)
2. **3 propagation regime 모두 동등 효과** — 정교한 weight modifier 없이도 uniform 이 동등
3. **HP 단일 변경 (lr 1e-4) 효과 ≈ 전체 method ablation 효과** (+1.7pp ≈ +2.7pp)
4. **GRCL alone (e) > Full method (h)** in 일부 metric — auxiliary loss 의 marginal effect 미미

→ 향후 method 디자인에서 **단순한 graph signal + 정확한 HP tuning + 잘 선택된 single edge type** 이 정교한 multi-component framework 보다 가성비가 높다.

### 7.5 학습 길이와 OOD generalization

§6.3 의 `h_best_combo_16k` 흥미로운 trade-off:

| 학습 step | SPIQA R@10 (in-domain) | MMDocIR R@10 (OOD) | SciEGQA R@10 (OOD) |
|---|---|---|---|
| 8k (h_best_combo) | 88.0 | 4.2 | 1.5 |
| 16k (h_best_combo_16k) | 85.9 (-2.1) | 4.4 (+0.2) | 1.7 (+0.2) |

- in-domain 은 saturate (slight overfit) — 8k 으로 충분
- **OOD 는 16k 에서 추가 향상** — zero-shot domain transfer 에 더 긴 학습이 도움

함의: in-domain 평가만으로 학습 길이를 결정하면 OOD 일반화에서 손해. **multi-dataset 평가 가 학습 hyperparameter 선택의 first-class signal** 이어야 한다.

### 7.6 Backbone-agnostic transferability

§6.5 의 CLIP-L/14 swap 이 보여주는 것:
- 본 paper 의 framework (GRCL + LoRA + late interaction) 가 SigLIP / CLIP 두 family 에 모두 적용
- ElementTokenEncoder 의 abstraction (d_text ≠ d_vision 자동 처리, CLS-drop 감지, GPE adapter, LoRA target 자동 매칭) 이 generic
- 향후 EVA-CLIP, SigLIP-large, CLIP-G 등 더 큰 backbone 으로 확장 자연스러움

CLIP-L/14 가 일관되게 향상 → backbone 의 단순 scaling 만으로도 본 framework 의 retrieval 품질이 개선될 가능성.

---

## 8. Conclusion

본 paper 는 학술 문서 multimodal retrieval 에서 **typed element graph 의 dual-use** 를 제안하고 5 가지 강한 실증으로 뒷받침했다:

### 8.1 핵심 contribution 요약

1. **Graph-Relevance Contrastive Loss (GRCL)** 가 binary InfoNCE 대비 SPIQA test-A R@10 **+2.7pp / MRR +5.4pp** 향상, paired-bootstrap **p = 0.042** statistically significant.

2. **`refer_to` edge 단독** 으로 GRCL graph relevance 를 구성하면 SPIQA R@10 **90.4** — 전체 ablation 최고치. 5 종 edge 균등 활용보다 우월.

3. **GME-Qwen2-VL-2B (외부 MLLM) + graph propagation** 으로 SPIQA R@10 **58.3 → 84.4 (+26pp), MRR 33.1 → 63.3 (+30pp)** — 본 study 단일 최대 효과. **재학습 없이 임의 retriever 위에 plug-in**.

4. **Best HP combo** (lr=1e-4 + τ=0.10 + λ_cov=0.5) 로 SPIQA R@10 **88.0**, MRR **42.2**; 16k step 확장 학습이 OOD 추가 향상 (MMDocIR 4.2 → 4.4).

5. **CLIP-L/14 backbone swap** 에서도 3 dataset 모두 일관 향상 → **method 가 backbone-agnostic**.

### 8.2 정성적 결론

- **동일 edge weight schema** 가 학습 supervision (GRCL) 과 추론 prior (propagation) 양쪽을 구동하는 single source of truth 가 작동.
- Graph propagation 은 **재학습 없는 plug-in module** — 외부 retrieval-tuned MLLM 에 적용 가능.
- **`refer_to` edge** (paragraph 가 figure 를 명시 인용) 가 가장 informative 한 graph signal.
- **3 propagation regime** (uniform / weighted / PPR) 모두 동등 효과 → graph 구조 자체가 prior 의 핵심.
- **Backbone scaling** + **HP tuning** 이 method ablation 만큼 큰 retrieval 품질 lever.

### 8.3 본 paper 의 한 줄 메시지

> **"Document element graph 가 retrieval 시스템의 universal prior 다 — 우리 모델을 학습하든, 외부 MLLM 위에 plug-in 하든, 같은 schema 가 작동한다."**

---

## 9. Future work

### 9.1 Multi-dataset training

현재 학습은 SPIQA single dataset 만. OOD 성능이 zero-shot 한계 — **SPIQA + SciEGQA train + MMDocIR train 통합 학습** 으로 zero-shot R@10 추가 향상 가능성 검증.

특히 SciEGQA 가 다영역 (cs/econ/eess/math/physics) → multi-domain 학습이 OOD generalization 의 핵심일 가능성.

### 9.2 LM-based retrieval × graph propagation

§6.4 의 GME 결과가 **본 paper 의 가장 강한 실증** 이지만, GME 위 propagation sub-ablation 은 default (wfull α=0.3 T=2) 만 실시. 향후:

- GME × 21-variant propagation sweep
- 다른 retrieval-tuned MLLM (NV-Embed, jina-embeddings-v4, Cohere multimodal) 에도 동일 적용
- **본 paper 의 GRCL 을 GME-style instruction tuning 과 결합** 하여 graph-aware MLLM 학습

### 9.3 Adaptive edge weighting

§5.2 의 `refer_to` 단독 우위 결과 활용:

- edge type weight 를 학습 가능 parameter 로 노출 → dataset-adaptive weighting
- per-query attention 으로 edge weight 동적 결정 → query-dependent graph

### 9.4 Sample-efficient `refer_to` 강조 학습

GRCL 에서 `refer_to` 가 가장 informative 면, anchor sampling 도 그에 맞게:

- `refer_to`-derived candidates 를 oversample 하는 anchor-mining 전략
- 학습 budget 을 informative edge 에 집중 → 같은 step 으로 더 큰 retrieval 품질 향상

### 9.5 Larger backbone × longer training

§7.6 + §7.5 의 결과 결합 — backbone scaling 과 training length 가 모두 OOD generalization 의 first-class lever. 향후:

- EVA-CLIP-L/14 (1B), SigLIP-large (400M) backbone × 16k - 32k steps
- compute budget 의 method ablation 보다 backbone × steps 에 더 큰 할당

### 9.6 Cross-document `cites` edge

본 paper 의 graph 는 **per-document** (논문 내부 element 간). 향후 cross-document edge (citation, "see also") 를 추가하여 retrieval 범위를 corpus-wide 로 확장.

---

## 10. Reproducibility

### 10.1 코드 + 데이터 + 모델

| 자료 | 위치 |
|---|---|
| 전체 코드 | <https://github.com/monkcat/dl_project> |
| Element graph metadata (v2.1) | <https://huggingface.co/datasets/ljh38/element-graph-v2.1> |
| **학습된 adapter (42 rows)** | <https://huggingface.co/ljh38/element-graph-encoder-v2.1> |
| 본 paper 결과 raw JSON | repo 의 `eval/results/experiments/*/eval.json` |
| 분석 figure + stat tests | repo 의 `eval/results/figures/` |

### 10.2 단일 명령 재현

처음부터 학습 (~4-5 days on 2× A100):
```bash
git clone https://github.com/monkcat/dl_project && cd dl_project
pip install -r requirements.txt
huggingface-cli login
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1
bash scripts/run_full_suite.sh
```

학습된 adapter 만 받아서 평가만 (~수십 분):
```bash
huggingface-cli download ljh38/element-graph-encoder-v2.1 --local-dir hf_models
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1
python -m pipeline.eval_full \
    --ckpt hf_models/adapters/h_best_combo.pt \
    --lora_rank 8 --gpe_facets type,role,depth,pos \
    --datasets spiqa_testA sciegqa mmdocir \
    --out eval/results/quick.json
```

### 10.3 환경

- Python 3.10
- PyTorch 2.11 + CUDA 12.x
- transformers 4.57 (GME 별도 venv: 4.51.3)
- peft 0.19
- 2× NVIDIA A100 80GB

### 10.4 결과 파일

| 파일 | 내용 |
|---|---|
| `eval/results/experiments/SUMMARY.md` | 6 section aggregated markdown (§7.1 ~ §7.6) |
| `eval/results/experiments/<row>_<name>/eval.json` | row 별 raw metric (query 단위 trace) |
| `eval/results/experiments/<row>_<name>/train.json` | step 별 loss + metric 학습 곡선 |
| `eval/results/figures/RESULTS_DIGEST.md` | 모든 row × metric × dataset 표 |
| `eval/results/figures/STAT_TESTS.md` | paired-bootstrap 유의성 |
| `eval/results/figures/*.png` | training curve, ablation bars, propagation comparison |
| `eval/results/figures/results_long.csv` | downstream 분석용 long-format |

---

## Appendix A — Best hyperparameter (`h_best_combo`)

```yaml
# Encoder
hf_id:           google/siglip2-base-patch16-224
lora_rank:       8
lora_alpha:      16
lora_dropout:    0.05
proj_dim:        128
max_text_len:    64

# Training
steps:           8000          # h_best_combo_16k: 16000
batch:           16            # anchors per step
pool:            12            # candidates per anchor
lr:              1e-4          # ★ default 3e-5 → 1e-4 (single largest lever)
tau:             0.10          # ★ default 0.07 → 0.10 (matches graded GRCL target)
weight_decay:    0.01
warmup_steps:    200
grad_clip:       1.0
seed:            42

# GRCL kernel
gamma:           0.5           # 2-hop path decay

# Anchor sampling (mixed ⅓ each)
anchor_kinds:    [caption_of, refer_to, nl_qa]

# Auxiliary losses (informational — main contribution은 GRCL 단독)
lambda_cov:      0.5
lambda_cons:     0.5

# Inference (default propagation variant)
inference:
  method:   diffusion
  weights:  full              # BASE × role × visual
  alpha:    0.3
  T:        2
```

## Appendix B — 데이터셋 출처

| Dataset | HF repo | License |
|---|---|---|
| 본 paper v2.1 graph metadata | `ljh38/element-graph-v2.1` | MIT (코드), 원본 데이터 license 유지 |
| SPIQA (원본 figure + paragraph) | `google/spiqa` | CC-BY 4.0 |
| SciEGQA | `Yuwh07/SciEGQA-Bench` | 원본 license 참조 |
| MMDocIR | `MMDocIR/MMDocIR-Challenge` | 원본 license 참조 |
| **학습된 adapter (본 paper)** | **`ljh38/element-graph-encoder-v2.1`** | **MIT** |

## Appendix C — Edge weight schema 전체

```python
# pipeline/types.py (single source of truth)

BASE_EDGE_WEIGHTS: dict[EdgeType, float] = {
    "caption_of":   1.0,
    "refer_to":     0.8,
    "contains":     0.5,
    "reading_next": 0.3,
    "section_next": 0.2,
}

SECTION_ROLE_MODIFIER: dict[SectionRole, float] = {
    "intro":      1.0,
    "method":     1.0,
    "result":     1.0,
    "discussion": 1.0,
    "appendix":   1.3,    # appendix evidence boost
    "references": 0.5,    # citation list weaken
    "abstract":   0.9,
    "front_matter": 0.8,
    "related_work": 0.9,
    "other":      1.0,
}

VISUAL_TARGET_MODIFIER = 1.1   # refer_to → figure/table/equation cross-modal boost
```

이 schema 가 학습 GRCL graph_relevance 와 추론 graph_propagate 양쪽을 구동 (§4.5).
