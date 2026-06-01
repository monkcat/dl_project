# 학술 문서 Multimodal Retrieval을 위한 Element Graph

**팀**: Jaehyeon Lee (20262670) · Seoyeon Lee (20262097) · Suhyeon Jun (20262717) · Minjun Kim (20262242)
**버전**: 2026-05-19 (method revision)

---

## 1. Abstract / 한 줄 thesis

> **학술 문서의 typed element graph는 retriever 학습의 graded relevance distribution을 정의하는 동시에 inference 시점의 propagation prior로 작동한다. 우리는 isolated positive pair가 아닌 graph-induced relevance distribution에 retriever를 fit시키며, 같은 edge-weight schema가 학습과 추론을 모두 구동한다.**

**Main contribution**:
- **Graph-Relevance Contrastive Loss (GRCL)** — typed academic element graph로부터 edge type, confidence, path length, target-attribute modifier에 기반한 graded relevance distribution을 만들고, retriever의 retrieval distribution을 이에 fit시키는 listwise objective. 같은 schema가 inference 시점 score propagation도 구동 (dual-use).

**Supporting mechanisms** — GRCL의 graph signal을 안정적으로 inject·활용하는 보조 장치:
- **Graph Position Embedding (GPE)** — node-level 구조 속성을 gated additive PE로 encoder embedding에 주입 (cross-document-safe relative feature)
- **PE-Dropout Consistency Loss** — graph metadata 과의존 방지 (element-anchor 학습 시 GPE 의존 완화)
- **Coverage Boundary Loss** — evidence set 전체가 top-K 안으로 들어오도록 직접 enforce
- **Inference-time Graph Propagation** — `pipeline/types.py:graph_propagate` 활용, 학습과 동일한 edge schema

핵심 설계 — **동일한 edge weight schema** (`pipeline/types.py:BASE_EDGE_WEIGHTS`)가 학습 (GRCL graph relevance)과 추론 (propagation) 양쪽을 구동. document graph가 retrieval 시스템 전체의 *single source of truth*.

**평가 데이터셋**:
- **SciEGQA** (1,623 query, 318 multi-bbox 모두 cross-page)
- **MMDocIR academic subset** (389 query, 76 multi-element, 78% cross-page)
- **SPIQA test-A** (161 query, single-element GT)

**학습**: SPIQA train (~25k papers, ~250k caption-figure pair — 원본 `all_figures` JSON + LaTeX `\ref{}` 매크로에서 직접 추출, docling 불필요).

---

## 2. 서론

### 2.1 Motivating example

다음 query 고려:

> *"Method A가 baseline 대비 hard split에서 얼마나 향상되었고, 부록은 이 격차를 어떻게 정당화하는가?"*

답하려면 4개 element 필요:

| # | Element | 위치 | 역할 |
|---|---|---|---|
| 1 | Table 1 (figure) | p.5 | 수치 비교 |
| 2 | Table 1 caption | p.5 | "hard split" 정의 |
| 3 | §4.2 마지막 paragraph | p.6 | 결과 해석 |
| 4 | Appendix B | p.12 | 이론적 정당화 |

이들은 페이지와 modality가 흩어져 있지만 문서 구조 안에서 `caption_of(2 → 1)`, `refer_to(3 → 1)`, `refer_to(3 → 4)`로 명시적으로 연결. **답에 필요한 evidence는 본질적으로 graph-structured**.

이 패턴은 실증적: SciEGQA의 1,623 query 중 318개 (20%)가 cross-page 2-bbox GT를 가지며, MMDocIR academic의 multi-element query 중 78%가 cross-page. Single-positive contrastive retriever는 이 구조를 자연스럽게 표현 못 함.

### 2.2 기존 접근의 한계

| 분류 | 대표 | Gap |
|---|---|---|
| Text-only RAG | BM25, SBERT | figure/table 손실 (M3) |
| Page-level visual RAG | ColPali, VisRAG | element granularity 없음, cross-page evidence 누락 (M2, M4) |
| 2-stage page→crop | SciEGQA-style | page retrieval 병목, relation 무시 (M2, M4) |
| Element-level retrieval | MMDocIR, RegionRAG | 각 element 독립 평가, set objective 부재 (M4) |
| Multi-evidence RAG | MMDocRAG, HM-RAG | retriever 평면, set은 사후 처리 (M4) |
| Modality-gap 방법 | AlignCLIP, CLIP-Refine | 일반 distribution 정렬, document structure 무시 (M3 부분) |

### 2.3 실증 — off-the-shelf multimodal embedding의 학술 visual element retrieval 실패

6개 multimodal embedding × 3 dataset = 18 cell zero-shot retrieval (cosine, no FT, per-doc pool):

- **Dual-encoder 계열**: SigLIPv2-base (370M), CLIP-ViT-L/14 (430M), Jina-CLIP-v2 (~400M, retrieval-specialized), BGE-VL-base (570M, retrieval-specialized)
- **Single-decoder MLLM 계열**: Raw Qwen2-VL-2B-Instruct (2B, no retrieval FT, last-token pooling), GME-Qwen2-VL-2B (Alibaba CVPR 2025, retrieval instruction tuning)

**Observation 1 — visual GT retrieval 실패는 compact dual encoder에서 일관됨.**

| Dataset | Encoder | text-only Hit@10 | **visual-only Hit@10** | mixed R@10 |
|---|---|---:|---:|---:|
| MMDocIR | SigLIPv2 | 0.20 | **0.000** | 0.08 |
| MMDocIR | CLIP-L/14 | 0.36 | **0.000** | 0.18 |
| MMDocIR | BGE-VL-base | 0.53 | **0.000** | 0.07 |
| MMDocIR | Jina-CLIP-v2 | 0.70 | **0.022** | 0.17 |
| MMDocIR | Raw Qwen2-VL-2B | 0.07 | **0.000** | 0.05 |
| MMDocIR | **GME-Qwen2-VL-2B** | **0.74** | **0.685** | **0.32** |
| SciEGQA | SigLIPv2 | 0.35 | **0.000** | 0.08 |
| SciEGQA | BGE-VL-base | 0.75 | **0.000** | 0.16 |
| SciEGQA | Jina-CLIP-v2 | 0.73 | **0.000** | 0.18 |
| SciEGQA | **GME-Qwen2-VL-2B** | **0.84** | **1.000** | **0.34** |
| SPIQA test-A | SigLIPv2 | — | **0.000** | — |
| SPIQA test-A | BGE-VL-base | — | **0.000** | — |
| SPIQA test-A | Jina-CLIP-v2 | — | **0.014** | — |
| SPIQA test-A | **GME-Qwen2-VL-2B** | — | **0.636** | — |

**Robust observation**: 4개 dual-encoder 모두 visual-only Hit@10 ≤ 2.2%로 실패. retrieval-FT (Jina, BGE)도, 더 큰 BGE-VL-base (570M)도 못 풀음. 큰 retrieval-tuned MLLM (GME, 2B)만 0.64-1.00 달성하지만 4-6× 큰 모델 + 20-40× 느린 inference 비용.

**Cautious observation**: Raw Qwen2-VL (GME backbone, retrieval FT 없이 last-token pooling)도 같이 실패. pooling/prompting 선택이 raw MLLM embedding geometry에 영향 (Ethayarajh 2019, Gao 2021). raw Qwen2-VL의 cosine geometry (모든 pairwise > 0.85)는 anisotropy / representation collapse 패턴과 일치.

**Observation 2 — visual GT rank ratio.**

| Dataset | Encoder | text GT rank | visual GT rank | ratio |
|---|---|---:|---:|---:|
| MMDocIR | SigLIPv2 | 50 | 208 | 4.2× |
| MMDocIR | Jina-CLIP-v2 | 20 | 131 | 6.7× |
| MMDocIR | **GME-Qwen2-VL-2B** | **17** | **19** | **1.1×** |
| SciEGQA | SigLIPv2 | 46 | 299 | 6.5× |
| SciEGQA | BGE-VL-base | 16 | 170 | 11.0× |
| SciEGQA | **GME-Qwen2-VL-2B** | **10** | **21** | **2.1×** |

Dual-encoder는 visual GT가 text GT 대비 4-11× 낮은 순위. GME에서는 1.1-2.1×로 줄어듦.

**Observation 3 — Geometric profile (모달리티 갭).**

| Architecture | Encoder | text-text | visual-visual | text-visual | gap |
|---|---|---:|---:|---:|---:|
| Dual encoder | SigLIPv2 | 0.85 | 0.69 | 0.08 | **0.69** |
| Dual encoder | CLIP-L/14 | 0.55 | 0.59 | 0.17 | **0.41** |
| Dual encoder | BGE-VL-base | 0.23 | 0.59 | 0.10 | **0.32** |
| Dual encoder | Jina-CLIP-v2 | 0.30 | 0.39 | 0.13 | **0.22** |
| Raw MLLM | Raw Qwen2-VL-2B | **0.92** | **0.97** | **0.88** | **0.06** |
| Retrieval-tuned MLLM | **GME-Qwen2-VL-2B** | **0.25** | **0.11** | **0.11** | **0.05** |

Dual encoder는 큰 intra-cross gap (0.22-0.69) — 분리된 modality cone (Liang et al. NeurIPS 2022). Raw Qwen2-VL은 작은 gap이지만 *모든* pairwise cosine이 매우 높음 (anisotropy/collapse). GME-Qwen2-VL은 작은 gap + moderate well-spread cosine (discriminative unified space).

**Motivation 요약**. Compact dual-encoder 영역 — production RAG retriever의 대부분 — 이 학술 visual element retrieval에 실패. 큰 retrieval-tuned MLLM이 해결하지만 비용 큼. **본 연구의 research question은 "MLLM을 cheap하게 대체할 수 있나"가 아니라, *encoder 아키텍처/규모와 독립적인 explicit document-native signal*이 학술 문서 evidence-set retrieval을 개선할 수 있는가**이다. 우리는 학술 element graph를 이 signal로 제안 — 학습 (graph-induced relevance distribution)과 추론 (score propagation) 모두에서 사용. Compact dual encoder와 retrieval-tuned MLLM을 strong orthogonal baseline으로 비교.

Hypersphere visualization (`eval/analysis/hypersphere_viz.py`)으로 gap이 tail 현상이 아니라 모든 embedding에서 일관됨 확인.

### 2.4 우리의 thesis와 contributions

학술 element relation graph는 — 어떤 layout-parsed PDF에서도, SPIQA의 경우 PDF parsing 없이 원본 dataset metadata에서도 — 세 가지 motivation을 모두 푸는 단일 통합 signal:

| Motivation | Element graph가 어떻게 푸나 |
|---|---|
| M2 cross-page evidence | `refer_to` with target `section_role=appendix`는 propagation modifier 1.3을 받아, page 거리와 무관하게 본문과 멀리 떨어진 페이지를 연결 |
| M3 modality gap | `caption_of` edge는 high-confidence cross-modal pair (SPIQA의 경우 author-provided `all_figures` metadata에서 직접). GRCL이 이 edge에 최대 supervision weight (1.0)을 할당 — graph supervision-level modality alignment. *Edge precision은 §A.4에서 dataset별로 측정·보고* |
| M4 cross-element reasoning | retrieval objective 자체가 set-prediction: GRCL이 retriever를 *graph-distance-graded relevance distribution*에 fit, 단일 positive가 아닌 evidence set 학습 |

**Contributions (main + supporting)**:

**Main**:
1. **Graph-Relevance Contrastive Loss (GRCL)** — typed academic element graph로부터 edge type, confidence, path length, target attribute modifier에 기반한 graded relevance distribution을 만들고, retriever를 이에 fit. explicit positive set sampling 불필요. ListNet 계열 listwise learning-to-rank의 *graph-derived target distribution* 변형 — novelty는 listwise form 자체가 아니라 typed document graph로 supervision distribution을 원리적으로 구성하는 것.

**Supporting mechanisms**:
2. **Graph Position Embedding (GPE)** — node-level 구조 속성 (type, section role, depth, intra-section 위치, graph-anchor 거리)을 gated additive PE로 인코딩. cross-document-safe relative feature만 사용 (absolute `section_path`는 cross-doc contamination 방지 위해 excluded)
3. **PE-Dropout Consistency Loss** — element-anchor 학습 시 graph metadata 의존 완화 (element with GPE → element without GPE distillation)
4. **Coverage Boundary Loss** — evidence set 전체가 top-K boundary 안으로 들어오도록 직접 enforce, 평가 metric Coverage@K와 1:1
5. **Edge-typed inference propagation** — GRCL과 **동일한** edge schema (`pipeline/types.py:BASE_EDGE_WEIGHTS`)를 inference 시점 score propagation에도 사용. dual-use of single source of truth.

---

## 3. Related Work

### 3.1 Page-level visual RAG

**ColPali** (Faysse et al., ICLR 2025) — page 단위 patch-token late interaction. **VisRAG** (Yu et al., 2024) — page를 단일 embedding으로. **M3DocRAG** (Cho et al., 2024) — multi-page page retrieval. Page granularity의 두 문제: (i) 답 아닌 element가 page representation 오염, (ii) cross-page evidence 손실.

### 3.2 Element-level retrieval

**MMDocIR** (Dong et al., EMNLP 2025) — "layout-level" task 정의. **RegionRAG** (AAAI 2026) — OCR-free, vision backbone에서 region 직접 추출. **SciEGQA** (arXiv 2511.15090) — Grounding-Crop-then-Answer protocol, multi-region/multi-page GT.

이 work들은 element **relation**을 first-class retrieval signal로 보지 않음 — element가 query에 독립적으로 점수 매겨짐.

### 3.3 Multi-evidence retrieval

**MMDocRAG** (arXiv 2505.16470), **HM-RAG** (2025), **PRISM** (arXiv 2510.14278) 모두 multi-evidence를 retrieval 하류에서 처리. **retriever 학습 objective에 multi-evidence를 통합**한 work 부재.

### 3.4 Document graph / structure-aware retrieval

**gDSA** (ICLR 2025) — 학술 element relation 대규모 dataset (80K docs, 4M relation), 구조 분석 용도. **SuperRAG** (NAACL Industry 2025), **MultiDocFusion** (EMNLP 2025), **BookRAG** (arXiv 2512.03413) 모두 graph 활용하지만 retrieval objective로 미사용.

가장 가까운 gDSA도 relation extraction에서 멈춤. 우리는 graph를 **학습 supervision + inference signal로 이중 활용** — 문헌에 없는 angle.

### 3.5 Position encoding for documents

**LayoutLMv3** (2022) — 1D order + 2D bbox PE 기반의 layout-coordinate PE 흐름. **Tree-PE** (Shiv & Quirk 2019) — tree path sinusoidal PE, 코드/HTML에서 검증. **DocPolarBERT** (2025) — 2D relative polar PE.

이 work들은 모두 *layout-coordinate* (bbox, page position) 또는 *tree path*에 기반함. Unlike layout-coordinate PE, **Graph Position Embedding (GPE)** encodes graph-derived structural facets (element type, section role, relative section position, graph-anchor distances) for retrieval, with each facet gated by a learnable scalar. 즉 "첫 PE"가 아니라 *layout PE가 아닌 graph-faceted PE*.

### 3.6 Modality gap

**Mind the Gap** (Liang et al., NeurIPS 2022), **AlignCLIP** (ICLR 2025), **CLIP-Refine** (2025), **I0T** (ACL 2025), **S1-MMAlign** (arXiv 2601.00264) 모두 *distribution*-level alignment. 우리 방법은 **구조적**: figure-caption이 high-confidence `caption_of` edge (SPIQA의 경우 author metadata에서)로 연결되어 있을 때 GRCL이 이를 최강 supervision (relevance weight 1.0)으로 사용 — post-hoc heuristic이 아닌 supervision-level alignment.

### 3.7 Graph supervision을 이용한 learning to rank

GRCL은 listwise learning-to-rank loss (cf. ListNet, Cao et al. 2007; ApproxNDCG, Bruch et al. 2019)의 *graph-derived target distribution* 변형. Novelty는 listwise form 자체가 아니라 typed document graph로부터 relevance distribution을 원리적으로 구성하는 것.

### 3.8 Gap 요약

To our knowledge, prior work has not used typed academic element graphs as both a source of graded listwise supervision and an inference-time propagation prior for element-level multimodal retrieval. 구체적으로 다음 측면들이 결합되어 다뤄진 work를 우리는 찾지 못함:

1. **G1**: element graph를 graded training supervision 형태로 사용한 element-level retrieval — gDSA는 graph extraction에서 멈추고, MMDocIR는 element evaluation을 제공하지만 둘이 결합되지 않음
2. **G2**: document graph node attribute을 layout-coordinate PE가 아닌 graph-faceted PE로 인코딩
3. **G3**: graph-aware retrieval의 train/inference query mismatch 처리 (학습 anchor는 graph attribute을 가지나 inference NL query는 가지지 않음)
4. **G4**: typed edge에 target-attribute modifier가 있는 inference-time score propagation, 학습 supervision과 동일 schema 공유

이 네 측면을 *jointly* 다루는 것이 우리 contribution. 각각은 부분적으로 인접 work들이 있고, 최근 multimodal document retrieval은 빠르게 변하므로 (예: ColMate 2025 등 document-specific training objective 시도들), 우리 claim은 "각 측면이 first"가 아니라 "이 네 측면의 결합".

---

## 4. Method

### 4.1 Notation

- 전체 element 코퍼스 $D = \{e_1, \ldots, e_N\}$
- Element $e$: type $\tau(e)$, source document $\text{doc}(e)$, multi-token 표현 $T(e) \in \mathbb{R}^{K_e \times D}$
- Per-document element graph $G_d = (V_d, E_d)$
- Query $q$: $T(q) \in \mathbb{R}^{K_q \times D}$
- Cosine similarity 전체 사용, 모든 token embedding L2-normalized

### 4.2 System overview

```
[Offline — corpus-wide, 1회]
─────────────────────────────────
원본 dataset metadata (SPIQA all_figures JSON, LaTeX) 또는 PDF + docling
  → element list + per-doc element graph G_d (schema v2.1)
  └─ Element token encoder φ (SigLIPv2 + LoRA, fp32)
       text/caption → ~50 subword token embedding
       figure/table/equation → 196 patch embedding
  └─ Graph PE gpe(v) 를 content(v)에 추가
  └─ Storage: tokens.pt + element_graph.json + metadata.jsonl

[Training — encoder fine-tuning on SPIQA train]
─────────────────────────────────
Mixed-anchor batch:
  - SPIQA QA (NL query, no GPE) — ~262k
  - caption_of pair (element anchor, GPE dropout p=0.5)
  - refer_to pair (element anchor, GPE dropout p=0.5)

각 (q, batch B):
  graph relevance g(q, e) 계산 (모든 e ∈ B)
  score s(q, e) = late_interaction(T(q), T(e))
  Loss = L_GRCL + λ_cov · L_cov + λ_cons · L_cons

LoRA on SigLIPv2 + GPE parameter (~1.1M trainable)

[Inference — per query]
─────────────────────────────────
NL query → T(q), no GPE → z_q
corpus element → T(e) + GPE → z_e (caching됨)
  ├─ Stage 1: late interaction scoring
  │     score(q, e) = Σ_i max_j ⟨T(q)_i, T(e)_j⟩
  │     → top-50 per query (per-doc pool)
  ├─ Stage 2: graph propagation
  │     s^{(t+1)} = (1-α) s^{(t)} + α · P^T · s^{(t)}
  │     T=2 iteration, P^T = typed W with target modifier
  └─ Final top-k
```

### 4.3 Element graph schema (v2.1)

`pipeline/types.py` 정의. Per-doc graph $G_d = (V_d, E_d)$.

**Node types**: `text`, `figure`, `table`, `equation`, `caption`, `section_header`

**Node attributes**:
- `id`, `type` (required)
- `page`, `section_path` (informational, inference scoring에서만 사용, GPE에는 X)
- `section_role` ∈ {`front_matter`, `intro`, `related_work`, `method`, `result`, `discussion`, `conclusion`, `references`, `appendix`, `other`}
- `label` (e.g. `"figure 2"`)

**Edge types & base weight** (single source of truth, `BASE_EDGE_WEIGHTS`):

| Edge | 방향 | Base weight | Source |
|---|---|---|---|
| `caption_of` | bidirectional | **1.0** | dataset GT (SPIQA `all_figures`) or layout proximity |
| `refer_to` | directional | **0.8** | paragraph text regex |
| `contains` | bidirectional | **0.5** | section_header → child |
| `reading_next` | bidirectional | **0.3** | sequential paragraphs **within same section** |
| `section_next` | directional | **0.2** | section → next section |

**Target-attribute modifier** (`refer_to` edge에만 적용, propagation 시점에서):

$$
w_{\text{eff}}(e_i \rightarrow e_j) = \alpha_{\tau} \cdot \text{conf}(e_i, e_j) \cdot \rho_{\text{section\_role}}(\tau(e_j)) \cdot \mu_{\text{vis}}(\tau(e_j))
$$

$\rho_{\text{appendix}} = 1.3$ (cross-page boost, M2), $\rho_{\text{references}} = 0.5$ (bibliography), $\mu_{\text{vis}} = 1.1$ (target이 figure/table/equation일 때, M3).

Edge extraction 규칙: Appendix A 참조.

### 4.4 Element token encoder + Graph Position Embedding (Supporting Component)

**Base encoder**. SigLIPv2-base-patch16-224 (370M) + LoRA (rank 8, alpha 16, dropout 0.05, ~1M trainable). Element $v$의 multi-token 표현:

$$
T(v) = \text{SigLIP+LoRA}(v.\text{text\_or\_image}) \in \mathbb{R}^{K_v \times D}
$$

text/caption은 ~50 subword token, figure/table/equation은 196 patch token.

**Graph Position Embedding (GPE)** — 5 facet의 gated sum (element-level vector $\mathbb{R}^D$):

$$
\text{gpe}(v) = \alpha_\text{type} \cdot e_\text{type}(\tau(v)) + \alpha_\text{role} \cdot e_\text{role}(\text{role}(v)) + \alpha_\text{depth} \cdot \text{PE}_\text{sin}(\text{depth}(v)) + \alpha_\text{pos} \cdot \text{PE}_\text{sin}(\rho_\text{intra}(v)) + \alpha_\text{anchor} \cdot \text{MLP}(\delta_\text{anchors}(v))
$$

- $e_\text{type} \in \mathbb{R}^{6 \times D}$, $e_\text{role} \in \mathbb{R}^{10 \times D}$ learned embedding
- $\text{depth}(v) \in \{1, 2, 3\}$ subsection nesting depth (sinusoidal)
- $\rho_\text{intra}(v) \in [0, 1]$ paragraph 위치 ratio (sinusoidal)
- $\delta_\text{anchors}(v) \in \mathbb{R}^4$ graph hop distance to nearest [figure, table, section_header, caption] (small MLP)
- $\alpha_\text{facet} = \sigma(\text{raw\_}\alpha)$, init raw $= -3.0$ → $\alpha \approx 0.047$

**Cross-document safety**. 모든 facet은 *relative*: `section_role`은 의미 카테고리 (universal), `depth`는 작은 정수 (universal), `intra_pos_ratio`는 [0, 1] (per-doc 정규화), `anchor_distance`는 graph hop count (universal). 절대 `section_path` (예: `[3, 2]`)는 cross-doc encoder embedding에 굽지 않음 — Paper A의 §3.2 = Method이고 Paper B의 §3.2 = Experiment일 수 있어 false similarity 생성.

**GPE × late interaction 결합** (명시). $T(v)$는 multi-token이고 $\text{gpe}(v)$는 single vector. GPE는 모든 token에 broadcast로 추가 (작은 learnable scalar $\beta$로 gated):

$$
T'_j(v) = \text{F.normalize}\bigl(\text{LN}(T_j(v) + \beta \cdot \text{gpe}(v))\bigr), \quad j = 1, \ldots, K_v
$$

- $\beta = \sigma(\text{raw\_}\beta)$, init raw $= -3.0$ → $\beta \approx 0.05$ — content embedding이 dominate하도록 PE 기여를 작게 시작
- 모든 token에 같은 PE 추가는 token-level granularity를 PE에서는 가지지 않음을 의미. 의도된 것 — PE는 element-level structural prior이지 token-level signal이 아님
- 다만 token-level matching에 PE shortcut 위험 (같은 role/type이면 MaxSim 자동 상승) — §6.5에 ablation으로 PE injection variant 측정 (all tokens / pooled / removed at inference)

Late interaction score는 $T'(q)$, $T'(e)$를 사용:

$$
s(q, e) = \sum_{i=1}^{K_q} \max_{j=1}^{K_e} \langle T'_i(q), T'_j(e) \rangle
$$

### 4.5 Graph-Relevance Contrastive Loss (Component 2) — 메인 loss

**Motivation**. 표준 InfoNCE는 binary positive/negative. 학술 evidence는 graded이다 — `caption_of` neighbor는 매우 강하게 관련, `refer_to`는 강하게, same-section sibling은 약하게. **Graph를 graded relevance의 teacher**로 사용.

**Graph relevance**:

$$
g(q, e) = \max_{\pi: q \rightsquigarrow e, |\pi| \le 2} \prod_{r \in \pi} \alpha_{\tau(r)} \cdot \text{conf}(r) \cdot \gamma^{|\pi|-1}
$$

$\alpha_\tau$ = `BASE_EDGE_WEIGHTS` (single source of truth), `conf(r)` ∈ [0.5, 1.0], $\gamma = 0.5$ (2-hop decay).

**Normalized weight**:

$$
\tilde{w}(q, e) = \frac{\max(0, g(q, e))}{\sum_{x \in B} \max(0, g(q, x))}
$$

**Weighted multi-positive InfoNCE**:

$$
\mathcal{L}_{\text{GRCL}}(q) = -\sum_{e \in B} \tilde{w}(q, e) \cdot \log \frac{\exp(s(q, e) / \tau)}{\sum_{x \in B} \exp(s(q, x) / \tau)}
$$

**핵심 성질**:
- $g = 0$ candidate은 numerator 기여 0, denominator에 살아있음 (implicit hard negative)
- `caption_of` (cross-modal, weight 1.0)이 자동으로 최대 supervision — modality gap 직접 공격
- `refer_to` to appendix는 effective weight `0.8 × 0.9 × 1.3 = 0.936` — cross-page evidence가 supervision-prioritized

**Anchor type 처리**:

| Anchor type | Source | Query GPE | $g(q, e)$ 계산 |
|---|---|---|---|
| NL question | SPIQA QA (262k) | **항상 zero** | $g(q, e) := g(\text{ref}, e)$ (QA의 `reference`로부터) |
| caption element | `caption_of` pair | dropout $p=0.5$ | direct |
| paragraph element | `refer_to` pair | dropout $p=0.5$ | direct |
| figure element | flip | dropout $p=0.5$ | direct |

NL-query는 graph node가 아니라 PE 정의 불가. QA의 `reference` figure가 graph anchor 역할 — "single GT를 graph로 multi-positive로 확장"의 직접 구현.

### 4.6 PE-Dropout Consistency Loss (Supporting Component) — auxiliary

**Motivation — 두 종류 mismatch 분리**. 학습 anchor와 inference query 사이에는 사실 두 가지 다른 mismatch가 있다:

1. **Element-with-GPE vs element-without-GPE** (PE 의존도) — element-anchor 학습에서 모델이 GPE metadata에 과의존하면, GPE가 없는 상황 (예: 새 element에 GPE를 계산 못 한 경우)에서 성능 저하
2. **Element anchor vs NL query anchor** (anchor type) — 학습은 element를 anchor로 쓰지만, 실제 inference query는 자연어 질문

`L_cons`는 **(1)을 직접 해결**한다 — element-anchor 학습 시 GPE-on teacher와 GPE-off student의 retrieval distribution을 align하여 모델이 GPE metadata에 over-rely하지 않도록.

(2)는 별개 메커니즘으로 다룬다: SPIQA QA 262k NL query batch를 학습에 직접 섞고, NL query는 항상 GPE=0으로 두어 자연스럽게 NL-query path를 학습시킨다.

→ **결합 효과**: `L_cons` + NL query mixing + query-side GPE always-zero policy의 조합이 train/inference gap 전체를 처리. `L_cons` 단독으로는 (1)만 해결.

**Formulation** (element-anchor batch에만 활성):

$$
z_q^{\text{full}} = \text{enc}(q, \text{gpe}=\text{True}), \quad z_q^{\text{drop}} = \text{enc}(q, \text{gpe}=\text{False})
$$

$$
p^*(e | q) = \text{softmax}_e\bigl(s(z_q^*, z_e) / \tau\bigr), \quad
\mathcal{L}_{\text{cons}} = \text{KL}\bigl(\text{stopgrad}(p^{\text{full}}) \,\Vert\, p^{\text{drop}}\bigr)
$$

Teacher (GPE on) → student (GPE off) asymmetric distillation. NL-query batch에서는 $L_\text{cons} = 0$ (teacher 없음).

### 4.7 Coverage Boundary Loss (auxiliary)

GRCL은 distribution match이지만 "set boundary" 직접 enforce 안 함. Strong positive $S_+^* = \{e : g(q, e) \ge 0.5\}$에 대해:

$$
\mathcal{L}_{\text{cov}}(q) = \frac{1}{|S_+^*|} \sum_{p \in S_+^*} \log\bigl(1 + \exp(\eta(\theta_K(q) - s(q, p)))\bigr)
$$

$\theta_K(q)$는 negative score 중 $K$-th highest. $K = 10$, $\eta = 5.0$. 모든 strong positive가 K-th negative 위에 있을 때 0.

### 4.8 Total loss

$$
\mathcal{L} = \mathcal{L}_{\text{GRCL}} + \lambda_{\text{cov}} \cdot \mathcal{L}_{\text{cov}} + \lambda_{\text{cons}} \cdot \mathcal{L}_{\text{cons}}
$$

$\lambda_{\text{cov}} = 0.3$, $\lambda_{\text{cons}} = 0.5$ (defaults). 3 loss, 각각 고유 역할:

| Loss | 역할 |
|---|---|
| $\mathcal{L}_{\text{GRCL}}$ | graph 구조를 graded relevance supervision으로 활용 |
| $\mathcal{L}_{\text{cov}}$ | evidence set 전체를 top-K boundary로 끌어들임 |
| $\mathcal{L}_{\text{cons}}$ | Graph PE train/inference mismatch 완화 |

### 4.9 Inference-time graph propagation (Supporting Component)

`pipeline/types.py:graph_propagate`에 이미 구현.

**Adjacency**. Top-50 candidate set의 typed weighted edge matrix $W \in \mathbb{R}^{N \times N}$:

$$
W_{ij} = \alpha_{\tau(r_{ij})} \cdot \text{conf}(r_{ij}) \cdot \rho_{\text{section\_role}}(\tau(e_j)) \cdot \mu_{\text{vis}}(\tau(e_j))
$$

여기서 $W_{ij}$는 노드 $i$에서 $j$로 가는 effective edge weight.

**Bidirectional edge의 modifier 처리**. Bidirectional edge (예: `caption_of`, `contains`)의 경우, traversal 방향에 따라 target attribute modifier가 다를 수 있다 (예: section_header → appendix paragraph는 $\rho_\text{appendix} = 1.3$이지만 reverse는 $\rho$ = 1.0). 두 방향이 같은 edge를 가리키므로 양쪽 $W_{ij}, W_{ji}$에 동일하게 $\max(\rho(\text{src}), \rho(\text{dst}))$를 사용한다. 이는 "edge의 strongest semantic role"을 보존하는 design choice — appendix 같은 high-value role 옆에 있는 element는 어느 방향에서 traverse되든 동일 boost를 받는다. *Direction-specific weighting을 선호하는 경우는 §6.5 ablation에서 비교.*

**Propagation**. Row-stochastic transition matrix $P = D_\text{out}^{-1} W$를 정의 ($P_{ij}$ = "노드 $i$에서 random walk 시 $j$로 갈 확률", 각 row 합 = 1). Update rule:

$$
\mathbf{s}^{(t+1)} = (1 - \alpha) \mathbf{s}^{(t)} + \alpha \cdot P^\top \mathbf{s}^{(t)}
$$

해석: $(P^\top \mathbf{s})_j = \sum_i P_{ij} s_i$ — 각 target 노드 $j$에 대해, 그 노드로 들어오는 (incoming) predecessor $i$들의 score를 transition probability $P_{ij}$로 가중합. 이는 **transposed random walk** (PageRank 계열에서 incoming score aggregation에 표준적으로 사용되는 형태). $P^\top$ 자체는 column-stochastic matrix이지만, 강조해야 할 semantic은 "incoming flow aggregation".

$\alpha = 0.3$ (default), $T = 2$ iteration.

**Single source of truth**. $\alpha_\tau$ 가중치 (`BASE_EDGE_WEIGHTS`), modifier (`SECTION_ROLE_MODIFIER`, `VISUAL_TARGET_MODIFIER`)는 GRCL에서 사용한 것과 동일 (`pipeline/types.py`). 학습 supervision과 추론 signal이 동일 상수에서 derive — 별도 inference-time tuning 불필요.

**Motivation 어떻게 해결되나**:
- **M2 cross-page**: $\rho_\text{appendix} = 1.3$이 appendix element로 score flow 부스트, 페이지 거리 무관
- **M3 modality gap**: $\mu_\text{vis} = 1.1$이 visual element로 score flow 강화. `caption_of` (alpha=1.0)이 figure-caption 양방향 연결
- **M4 cross-element reasoning**: 높은 점수 text anchor가 `refer_to` target (figure, table, appendix)으로 score propagate → coherent evidence set

### 4.10 컴포넌트 통합

4 컴포넌트가 stack:

```
                    Component 1 (GPE)               Component 4 (Propagation)
                          │                                  │
content(v) ────→  z(v) = content(v) + gpe(v)  ─────────→  s^{(t+1)} = (1-α)s + α·P^T·s
                          │                                  │
                          │           training only          │
                          │                                  │
                          ▼                                  ▼
                  L_GRCL + L_cov + L_cons        graph_propagate (T=2)
                  Component 2 + Coverage + Component 3
```

**핵심**: document graph $G_d$가 단일 edge weight schema로 학습과 추론 양쪽을 구동. GRCL이 graph structure를 encoder에 distill, GPE가 node attribute을 embedding에 주입, 추론이 explicit graph propagation으로 추가 smoothing. 셋 다 redundant가 아닌 coordinated.

---

## 5. Datasets

| Dataset | Elements | Queries | GT 단위 | 역할 |
|---|---|---|---|---|
| **SPIQA train** (LaTeX-only) | ~25,459 papers, ~250k figure-caption pair | 262,524 NL QA | single-element | Encoder FT 학습 |
| **SPIQA test-A** | 118 papers, ~28k elements | 161 | single-element | Sanity, M3 figure-centric |
| **SciEGQA** | 20,791 | 1,623 (**318 multi-bbox cross-page**) | multi-element | **Multi-positive eval, M2/M4** |
| **MMDocIR** academic | 14,454 | 389 (**76 multi-element, 78% cross-page**) | multi-element (GT size 1-11) | **Multi-positive eval** |

**왜 PDF parsing 없이 SPIQA-centric 학습**. SPIQA는 원본 `all_figures` JSON (figure-caption pair GT) + `paragraphs/*.txt` (paragraph 분리) + `raw_tex/*.tex` (LaTeX 소스)를 제공. caption-figure pair는 저자 작성 metadata에서 직접, `\ref{fig:foo}` 매크로는 직접 refer_to edge. **PDF parsing 불필요**, 25k paper로 trivially 확장 가능.

**Train/test 분리**: SPIQA train ∩ test-A = 0 (paper ID 기준 확인). SciEGQA / MMDocIR은 학습 노출 0 — zero-shot domain transfer 평가.

**Pool 구조**: per-doc pool (`restrict_to_paper=True`), SciEGQA protocol과 citation-grounded RAG 표준. Cross-doc retrieval은 보조 ablation.

---

## 6. Experimental Setup

### 6.1 Metrics

**Retrieval**: Recall@k (k ∈ {1, 5, 10}), Hit@k, MRR, NDCG@k graded relevance

**Set-prediction (메인 novelty target)**:
- **Coverage@k**: GT의 fraction in top-k
- **PerfectSet@k**: GT 전체가 top-k에 들어옴
- **Cross-page hit rate**: multi-page GT가 모두 hit

**Modality gap (학습 효과 직접 측정)**: caption_of pair 평균 cosine (FT 전 vs 후), cross-modal centroid distance, anisotropy (Ethayarajh 2019)

**Cost**: indexing wall-clock, retrieval latency p50/p95, peak memory

### 6.2 Baselines — tiered comparison

| Tier | ID | Method | Note |
|---|---|---|---|
| 1 (same design space) | B2 | SigLIPv2 zero-shot single-vec | floor (= row (a)) |
| 1 | B3 | SigLIPv2 + standard contrastive FT | basic FT |
| 1 | B5 | ColBERT v2 (text-only) | text late interaction |
| 1 | B6 | AlignCLIP | encoder-sharing modality-gap |
| 2 (sanity) | B1 | BM25 | text-only sparse |
| 3 (context only) | B4 | ColPali v1.2 | page granularity (별도 표) |
| 3 | B7 | VisRAG | page granularity (별도 표) |
| 4 (orthogonal) | B8 | GME-Qwen2-VL-2B | retrieval-tuned 2B MLLM (complementarity test) |

### 6.3 메인 ablation 표

| Row | GPE | Loss | Inference graph | Purpose |
|---|---|---|---|---|
| (a) | — | InfoNCE | — | base + standard FT (floor) |
| (b) | type only | InfoNCE | — | type prior |
| (c) | type + role | InfoNCE | — | role prior |
| (d) | all facets | InfoNCE | — | full GPE (no GRCL) |
| (e) | — | GRCL | — | GRCL (no PE) |
| (f) | all facets | GRCL | — | GPE + GRCL synergy |
| (g) | all facets | GRCL + $\lambda_\text{cov}$ | — | + coverage |
| (h) | all facets | GRCL + $\lambda_\text{cov}$ + $\lambda_\text{cons}$ | — | + consistency (full training) |
| (i) | — | InfoNCE | ✓ prop | inference graph alone |
| (j) | all facets | GRCL + $\lambda_\text{cov}$ + $\lambda_\text{cons}$ | ✓ prop | **Full system** |
| (k) | GME frozen | — | — | strong MLLM reference |
| (l) | GME frozen | — | ✓ prop | **encoder-agnostic test** |

핵심 비교:
- (a) → (d): GPE 단독
- (a) → (e): GRCL 단독
- (a) → (f): GPE + GRCL synergy
- (a) → (j): full system gain
- **(k) → (l)**: graph propagation이 retrieval-tuned MLLM 위에도 추가 향상? → **encoder-agnostic claim 직접 검증**

### 6.4 Negative controls

| Row | 변경 | 측정 의도 |
|---|---|---|
| (m) | Shuffled `section_role` (같은 doc 안 random permutation) | PE가 진짜 structure 학습했나? (m) ≈ (j)면 PE는 regularization뿐 |
| (n) | Random `section_role` (uniform per element) | role signal 실제로 쓰이나? |
| (o) | Query PE dropout off | train/inference mismatch 영향 |
| (p) | Encoder swap (SigLIPv2 → CLIP-L/14) | encoder-specific quirk? |

음성 통제 통과 기준: (m), (n), (o)는 (j) 대비 통계적 유의 저하. (p)는 encoder 변경에도 gain robust.

### 6.5 Sub-ablations (SigLIPv2에서)

- Per-facet α gating
- γ (graph distance decay) ∈ {0.3, 0.5, 0.7}
- λ_cov, λ_cons ∈ {0, 0.1, 0.3, 0.5, 1.0}
- Propagation α ∈ {0.1, 0.3, 0.5, 0.7}, T ∈ {1, 2, 3}
- Edge type isolation (caption_of / refer_to / contains)
- Token count per visual ∈ {16, 64, 196}
- LoRA rank ∈ {4, 8, 16, 32}

### 6.6 Failure analysis

20 query × 3 dataset = 60 정성 분석. 분류: Extraction fail / GPE fail / Encoder fail / Propagation miss / Coverage miss.

### 6.7 통계 유의성

Paired bootstrap (1,000 resample), per-query metric difference의 95% CI. p < 0.05 보고.

---

## 7. Results

*최종 run: 2026-05-30 ~ 06-01. 43 row (42 trained + GME zero-shot) + (h) checkpoint propagation sub-ablation 21 variant. 분석 파이프라인 5단계(aggregate → plot → re-eval → stat_test → diagram) 완료. 집계 원본: `eval/results/experiments/SUMMARY.md`, 통계: `eval/results/figures/STAT_TESTS.md`, digest/CSV: `eval/results/figures/`.*

### 7.0 실험 구성 메모 (row naming, 실제 run)

- §6.3 설계 표의 row (i)/(j)/(l)은 **별도 학습 row가 아니라 추론 시점 variant**로 실현되었다. 모든 trained row에 대해 `enc only`(propagation 없음)와 `full α=0.3 T=2`(weighted-diffusion propagation) 두 variant를 평가한다.
  - 설계상의 **(j) Full system** = `(h) full α=0.3 T=2`, **(i)** = `(a) full α=0.3 T=2`, **(l) encoder-agnostic** = `(k=GME) full α=0.3 T=2`.
- 모든 평가는 per-doc pool. 지표: R@5/R@10, MRR, Coverage@10(Cov), PerfectSet@10(Pf), cross-page hit(Xpg, 멀티페이지 GT에서만).
- **학습은 SPIQA train 단독.** 따라서 SPIQA test-A = in-domain, **SciEGQA·MMDocIR = zero-shot domain transfer**. 이 사실이 결과 전체를 지배한다.

### 7.1 Main ablation (Tier 1, rows a–h + k=GME)

**SPIQA test-A** (in-domain, single-positive)

| Row | enc-only R@10 / MRR | +prop R@10 / MRR |
|---|---|---|
| (a) InfoNCE baseline | 84.4 / 32.8 | 69.8 / 29.3 |
| (b) +GPE type | 82.1 / 31.7 | 69.7 / 28.6 |
| (c) +GPE type+role | 83.8 / 32.3 | 69.2 / 29.1 |
| (d) +GPE full | 79.9 / 29.2 | 66.2 / 27.4 |
| (e) GRCL, no GPE | **87.1 / 38.2** | 70.3 / 30.4 |
| (f) GRCL + GPE | 85.1 / 35.2 | 70.9 / 29.8 |
| (g) GRCL + GPE + L_cov | **87.1 / 39.0** | 70.0 / 30.6 |
| (h) Full method | 86.9 / 37.4 | 70.7 / 30.0 |
| (k) GME zero-shot | 58.3 / 33.1 | **84.4 / 63.3** |

읽히는 패턴: ① GRCL이 유일하게 의미 있는 상승 요인(a→e: R@10 +2.7, MRR +5.4). ② GPE는 손해(a→d로 단조 감소, e→f로도 하락). ③ L_cov/L_cons는 사실상 무변화(e≈g≈h). ④ trained SigLIP에서 propagation은 **일관되게 손해**(86.9→70.7). ⑤ 그러나 **GME에서는 propagation이 극적으로 도움**(58.3→84.4, MRR 33.1→63.3) — §8 H5.

**SciEGQA** (zero-shot, multi-positive)

| Row | enc-only R@10 / Xpg | +prop R@10 / Xpg |
|---|---|---|
| (a) | 0.9 / 24.1 | 1.3 / 16.1 |
| (e) | 1.0 / 26.6 | 1.5 / 17.4 |
| (h) | 1.0 / 26.9 | 1.4 / 16.5 |
| (k) GME | **2.3 / 51.3** | 2.0 / 37.3 |

**MMDocIR** (zero-shot, multi-positive)

| Row | enc-only R@10 | +prop R@10 |
|---|---|---|
| (a) | 2.0 | 3.7 |
| (e) | 2.5 | 3.8 |
| (h) | 2.4 | 4.1 |
| (k) GME | **18.9** | 15.7 |

OOD에서 trained SigLIP은 전부 R@10 ≈ 0.9–1.1%(SciEGQA) / 2.0–2.6%(MMDocIR)로 붕괴하며 row 간 차이는 노이즈 수준. 반면 GME zero-shot은 2.3% / 18.9%로 압도. trained 모델에서는 propagation이 OOD에서 **소폭 도움**(MMDocIR 2.4→4.1), GME에서는 **손해**(18.9→15.7) — 정확히 SPIQA와 반대 방향(§8 통합 해석).

### 7.2 Negative controls (Tier 2, rows m–p) — 가장 깨끗한 결과

SPIQA test-A R@10 (enc-only):

| Row | 변경 | R@10 | MRR | vs (h) |
|---|---|---|---|---|
| (m) shuffled section_role | role 라벨 doc 내 셔플 | 86.3 | 38.7 | ≈ |
| (n) random section_role | role 라벨 uniform 랜덤 | 85.1 | 37.6 | ≈ |
| (o) no query PE dropout | L_cons 입력 제거 | 87.1 | 38.0 | ≈(약간↑) |
| (p) encoder swap CLIP-L/14 | 백본 교체 | **88.1** | 36.2 | ↑ |
| (h) Full method | — | 86.9 | 37.4 | — |

- **(m)/(n)이 (h)와 통계적으로 구분 불가** → `section_role` 신호가 학습에 실질적으로 사용되지 않음. GPE role facet 무용을 직접 증명.
- **(o) ≈ (h)** → query PE dropout(=L_cons의 핵심 입력)을 꺼도 차이 없음 → train/inference mismatch 효과 미측정.
- **(p) CLIP-L/14가 (h) 능가**(SPIQA 88.1, MMDocIR 4.2 vs 2.4) → gain의 출처가 우리 방법이 아니라 **백본**임을 시사. §6.4 음성통제 통과 기준((m),(n),(o) 유의 저하)은 **불충족**.

### 7.3 Sub-ablation sweeps (Tier 3, 25 rows)

전체 8-열 표는 `SUMMARY.md` §7.3, master 표는 아래 §7.7 참조. SPIQA R@10(enc-only) 기준 요약:

| 패밀리 | 결과 (SPIQA R@10) | 판정 |
|---|---|---|
| γ (GRCL 2-hop decay) | 0.3=83.9, **0.5(h)=86.9**, 0.7=85.7 | 0.5 적정, 둔감 |
| λ_cov | 0.0/0.1/0.5/1.0 = 85.4/87.1/87.1/85.7 | **무변화** |
| λ_cons | 0.0~1.0 = 86.0~86.9 | **무변화** |
| LoRA rank | r4=83.0, **r8(h)=86.9**, r16=86.2, r32=84.7 | r8 적정, 키워도 이득 없음 |
| learning rate | 1e-5=79.9, 3e-5(h)=86.9, **1e-4=88.6** | ★lr이 최대 레버 |
| τ | 0.05/0.07/0.10 = 85.6/86.9/86.2 | 둔감 |
| anchor kind | caption=86.0, **refer=74.9(붕괴)**, nlqa=84.8, mixed(h)=86.9 | mixed 적정 |
| edge type isolation | caption_of=83.5, **refer_to=90.4**, contains=88.4 | ★refer_to가 최강 signal |
| visual tokens | 16=83.0, 64=84.7, **196(h)=86.9** | 늘려도 이득 없음 |

전 데이터셋에서 일관 상승한 변경은 **lr_1e4**(SPIQA 88.6 / SciEGQA 1.6 / MMDocIR 4.3)와 **edge_refer_to**(SPIQA **90.4** / MMDocIR **5.7**, 둘 다 단일 최고치) 두 가지뿐. 손실항·구조 HP는 SPIQA에서 ±1–2 진동(=seed 노이즈와 구분 불가). anchor_refer(74.9)·lr_1e5(79.9)는 명확한 실패.

### 7.4 Follow-up extensions (Tier 4, 5 rows)

| Row | SPIQA R@10/MRR | SciEGQA R@10 | MMDocIR R@10 |
|---|---|---|---|
| h + best-HP combo | **88.0 / 42.2** | 1.5 | 4.2 |
| h + best-HP combo, **16k steps** | 85.9 / 39.7 | **1.7** | **4.4** |
| h + stronger GPE init | 85.1 / 34.9 | 1.0 | 2.4 |
| h (seed=43) | 85.6 / 35.9 | 1.2 | 2.2 |
| h (seed=44) | 83.0 / 35.0 | 1.0 | 2.5 |
| (h) seed=42 | 86.9 / 37.4 | 1.0 | 2.4 |

- **best-HP combo가 종합 최고 학습 모델**(SPIQA 88.0, MRR 42.2). 16k step은 in-domain 소폭↓·OOD 소폭↑ (과적합 vs 전이 trade-off).
- **stronger GPE init 무효**(85.1 < h) → GPE 무용 재확인.
- ⚠️ **seed 분산이 큼**: 86.9 / 85.6 / 83.0 → SPIQA R@10 **±~4pt**. 메인 ablation의 유의 효과(§7.6 h−a = +2.5pp)가 이 분산보다 작다 — 해석 시 핵심 경고.

### 7.5 Propagation sub-ablation on (h) — 3 regime × HP sweep (21 variant)

**SPIQA (in-domain)** — 모든 regime에서 손해:

| variant | R@10 |
|---|---|
| enc only | **86.9** |
| weighted-diffusion full α0.3 T1 | 78.7 |
| full α0.1 T2 | 74.3 |
| base / base+role / base+visual / full (α0.3 T2) | **70.7 (4종 모두 동일)** |
| full α0.7 T2 | 65.5 |
| uniform α0.3 T2 | 67.0 |
| PPR α0.15 / α0.85 | 63.5 / 73.4 |

→ **modifier 설계(base/role/visual/full)가 결과에 영향 0** (single source of truth schema의 정교함이 in-domain에서 무의미). α↓·T↓일수록 enc-only에 가까워 덜 나쁨.

**MMDocIR (OOD)** — 도움:

| variant | R@10 |
|---|---|
| enc only | 2.4 |
| uniform / weighted / PPR 대부분 | 3.9 – 4.2 |

→ 약한 인코더 + multi-positive에서는 그래프 구조가 weak prior로 작동. SciEGQA는 1.0→1.3~1.5로 미미한 상승.

### 7.6 Statistical significance (paired bootstrap, 1000 resample, Coverage@10, no-prop)

| 비교 | SPIQA Δ / 95% CI / p | MMDocIR p | SciEGQA p | 유의 |
|---|---|---|---|---|
| **h − a** | +0.0255 / [+0.0015, +0.0511] / **0.042** | 0.514 | 0.538 | SPIQA만 ✓ |
| h − e | −0.0015 / [−0.0165, +0.0135] / 0.97 | 0.918 | 0.984 | ✗ |
| f − e | −0.0195 / [−0.0435, +0.0045] / 0.12 | 0.444 | 0.278 | ✗ |
| g − f | +0.0195 / [−0.0045, +0.0435] / 0.14 | 0.262 | 0.308 | ✗ |

유의한 효과는 **h−a(전체 파이프라인 vs InfoNCE), SPIQA에서만** 하나뿐. GPE 추가(f−e)·coverage 추가(g−f)는 비유의. ⚠️ 유의 효과크기(+2.5pp)가 seed 분산(±4pp, §7.4)보다 작다.

### 7.7 종합 — R@10 (no propagation), 전 row × 전 dataset

| Row | SPIQA | SciEGQA | MMDocIR | | Row | SPIQA | SciEGQA | MMDocIR |
|---|---|---|---|---|---|---|---|---|
| a | 84.4 | 0.9 | 2.0 | | edge_refer_to | **90.4** | 1.3 | **5.7** |
| b | 82.1 | 0.9 | 2.5 | | edge_contains | 88.4 | 1.4 | 2.1 |
| c | 83.8 | 1.1 | 2.3 | | edge_caption_of | 83.5 | 1.3 | 3.1 |
| d | 79.9 | 1.1 | 2.4 | | gamma_03 | 83.9 | 0.9 | 2.5 |
| e | 87.1 | 1.0 | 2.5 | | gamma_07 | 85.7 | 1.2 | 2.4 |
| f | 85.1 | 1.1 | 2.3 | | cov_00/01/05/10 | 85.4–87.1 | ~1.0 | ~2.5 |
| g | 87.1 | 1.0 | 2.6 | | cons_00/01/03/10 | 85.7–86.9 | ~1.0 | ~2.4 |
| h | 86.9 | 1.0 | 2.4 | | lora_r4/16/32 | 83.0/86.2/84.7 | 1.2/1.4/1.3 | 2.1/3.1/2.2 |
| **gme** | 58.3 | **2.3** | **18.9** | | lr_1e5 / lr_1e4 | 79.9 / **88.6** | 1.3 / 1.6 | 1.4 / 4.3 |
| m | 86.3 | 0.9 | 2.4 | | tau_005 / tau_010 | 85.6 / 86.2 | 1.0 | 2.6 / 2.2 |
| n | 85.1 | 1.2 | 3.2 | | anchor_caption/refer/nlqa | 86.0/74.9/84.8 | 1.3/1.4/1.4 | 1.5/1.8/2.2 |
| o | 87.1 | 1.0 | 2.5 | | tokens_016 / tokens_064 | 83.0 / 84.7 | 1.1 / 1.0 | 2.4 / 2.6 |
| p | 88.1 | 1.5 | 4.2 | | h_best_combo / _16k | 88.0 / 85.9 | 1.5 / 1.7 | 4.2 / 4.4 |
| | | | | | h_gpe_strong | 85.1 | 1.0 | 2.4 |
| | | | | | h_seed_43 / _44 | 85.6 / 83.0 | 1.2 / 1.0 | 2.2 / 2.5 |

*(전체 8-열 metric 표 및 propagation variant별 수치는 `eval/results/experiments/SUMMARY.md` §7.1–§7.6, 그림은 `eval/results/figures/`.)*

---

## 8. Discussion

### 8.1 가설 검증 결과

| 가설 | 판정 | 근거 |
|---|---|---|
| **H1** GRCL > InfoNCE (Coverage@10 큰 폭) | **부분 인정** | SPIQA Cov@10 84.4→87.1, MRR 32.8→38.2; h−a p=0.042로 유일하게 유의. 단 "큰 폭"은 아니고(~2.5pp) OOD에선 비유의. MRR 개선(+5.4)이 더 뚜렷 → "상위 랭킹 품질 개선"으로 봐야 정확. |
| **H2** GPE encoder-agnostic moderate gain | **기각** | GPE는 SigLIP에서 음의 효과(a→d 84.4→79.9, e→f 87.1→85.1). m/n 음성통제가 role facet 무용 입증. CLIP-swap (p)의 우위는 GPE가 아닌 백본 효과. |
| **H3** L_cons가 mismatch 대부분 해소 | **기각** | (o) no-PE-dropout = 87.1 ≥ (h) 86.9. λ_cons sweep 평평(86.0~86.9). mismatch 완화 효과 측정 안 됨. |
| **H4** propagation은 GRCL 위에 additive (cross-page에서 최대) | **기각(in-domain)** | SigLIP에서 propagation은 SPIQA 86.9→70.7로 큰 손해, modifier 4종 결과 동일(70.7). additive 아님. OOD에선 소폭 도움(2.4→4.1)이라 **조건부**. |
| **H5** (k)→(l) 양수 Δ — propagation이 compact encoder 너머로 일반화 | **인정(SPIQA)** | GME 58.3→84.4, MRR 33.1→**63.3** — 전 실험 통틀어 가장 큰 양수 효과. 단 OOD(MMDocIR 18.9→15.7)에선 음수. |

### 8.2 통합 해석 — propagation은 "약한 base의 보완재"

두 인코더의 propagation 효과가 정확히 **대칭**으로 나타난다:

| | base가 강한 dataset | base가 약한 dataset |
|---|---|---|
| **SigLIP-h** | SPIQA(86.9) → prop **손해**(70.7) | MMDocIR(2.4) → prop **도움**(4.1) |
| **GME** | MMDocIR(18.9) → prop **손해**(15.7) | SPIQA(58.3) → prop **도움**(84.4) |

→ graph propagation은 **base retriever가 해당 데이터셋에서 약할 때 smoothing prior로 도움, 강할 때는 정답 점수를 이웃으로 흩뜨려 손해**. "encoder-agnostic additive gain"(H4/H5의 원래 강한 형태)은 성립하지 않고, **"base가 약한 경우에 한해 generalize"라는 약화된 형태**로만 성립. modifier 설계(role/visual boost)의 정교함은 in-domain에서 영향 0이었으므로, 효용의 원천은 정교한 가중치가 아니라 **그래프 구조의 존재 자체**다.

### 8.3 가장 큰 실증 메시지 — 도메인 전이 실패와 백본의 지배

- 200M SigLIP은 in-domain(SPIQA 87%)에서는 GME(58%)를 압도하지만, **zero-shot OOD에서 붕괴**(SciEGQA 1%, MMDocIR 2.4%, 일부 chance 이하). 학습이 SPIQA 단독이었던 것이 직접 원인으로 보인다.
- 반대로 2B retrieval-tuned GME는 OOD(MMDocIR 18.9%)를 지배. §2.3 motivation("compact dual-encoder는 학술 visual element retrieval에 실패, 큰 retrieval-tuned MLLM이 해결")이 학습 후에도 **OOD 영역에서 재확인**됐다.
- 음성통제 (p) CLIP-L/14가 본 방법을 능가한 것까지 합치면, 본 실험에서 retrieval 품질을 실제로 움직인 레버는 **그래프 방법론보다 백본·학습률**이었다.

### 8.4 한계

1. **단일 seed 메인 비교**: 유일한 유의 효과(+2.5pp)가 seed 분산(±4pp)보다 작다. 메인 row의 다중 seed(3–5) 재실행이 필요.
2. **OOD가 chance 수준**: SciEGQA recall이 랜덤 이하 구간 존재 → 전이 실패의 정상 동작 한계로 명시.
3. **학습 데이터가 SPIQA 단독**: multi-dataset 학습 미실시가 OOD 붕괴의 직접 원인일 수 있음.

---

## 9. Conclusion

학술 typed element graph를 retriever 학습 supervision(GRCL)과 추론 propagation에 이중 활용하는 가설을 SigLIPv2 백본에서 43-row ablation으로 검증했다. 결과는 **대체로 부정적이되 몇 가지 명확한 시그널**을 남겼다:

1. **GRCL은 in-domain에서 측정 가능한 이득**을 준다(SPIQA R@10 84.4→87.1, MRR 32.8→38.2, h−a p=0.042). 효과는 작고 상위 랭킹(MRR) 개선에 집중된다.
2. **그래프 propagation은 조건부로만 유용**하다 — base retriever가 약할 때 smoothing prior로 작동하며, GME에 얹었을 때 SPIQA에서 R@10 58.3→84.4, MRR 33.1→63.3의 가장 큰 단일 향상을 보였다(encoder-agnostic의 약화된 형태 확인).
3. **GPE·L_cov·L_cons는 효과 없음**, 음성통제(shuffled/random role)가 이를 깨끗하게 반증했다.
4. **실제 성능 레버는 백본·학습률·refer_to edge**였다(lr 1e-4 → SPIQA 88.6, refer_to-only → 90.4, CLIP-L/14 swap → 88.1).
5. **작은 도메인특화 인코더는 OOD 전이에 실패**하고, 큰 retrieval-tuned MLLM(GME)이 OOD를 지배한다 — §2.3 motivation 재확인.

후속 1순위: 메인 row 다중 seed 재현 + `lr=1e-4 + GRCL(refer_to 중심) + GPE off + propagation off` 조합 + multi-dataset(SciEGQA/MMDocIR 포함) 학습으로 전이 실패 직접 공략. 자세한 수치는 §7 및 `eval/results/experiments/SUMMARY.md` 참조.

---

## 10. Future Work

- Edge-type 가중치와 encoder 공동 학습 (learnable $\alpha_\tau$)
- Query type별 adaptive propagation depth $T$
- Cross-document `cites` edge 확장 (citation-grounded retrieval)
- LLM-as-judge로 retrieved evidence set generation 평가
- Academic 외 적용 (legal, medical 강한 structural prior)

---

## 11. Implementation Plan

단독 개발자 구현. Module 경계는 코드 정리 목적:

| Module | 책임 |
|---|---|
| `pipeline/types.py` | Schema + `graph_propagate` + `late_interaction_score` (single source of truth) |
| `pipeline/build_spiqa_graph_v2.py` | LaTeX/JSON-only graph builder (docling 불필요) |
| `pipeline/graph_pe.py` | GraphPositionEmbedding (per-facet gated) |
| `pipeline/graph_relevance.py` | $g(q, e)$ 계산 (2-hop) |
| `pipeline/losses.py` | GRCL, L_cov, L_cons |
| `pipeline/element_encoder.py` | SigLIPv2 + LoRA + GPE forward |
| `pipeline/retrieve.py` | Late interaction + `graph_propagate` orchestrator |
| `eval/runners/` | Training/evaluation orchestrator |
| `eval/analysis/visualize_graph_v2.py` | pyvis interactive HTML viz |

Module 간 interface는 `pipeline/types.py`에 통합 정의.

---

## Appendix A. Element graph extraction (v2.1)

### A.1 Schema (요약)

`pipeline/types.py`:
- `ElementType` = {`text`, `figure`, `table`, `caption`, `equation`, `section_header`}
- `EdgeType` = {`caption_of`, `refer_to`, `contains`, `reading_next`, `section_next`}
- `BASE_EDGE_WEIGHTS` = {1.0, 0.8, 0.5, 0.3, 0.2}
- `SECTION_ROLE_MODIFIER` = {`appendix`: 1.3, `references`: 0.5, `front_matter`: 0.8, others: 1.0}
- `VISUAL_TARGET_MODIFIER` = 1.1

### A.2 Edge extraction (SPIQA, LaTeX-only — `pipeline/build_spiqa_graph_v2.py`)

**`caption_of`** (target precision >0.95):
1. SPIQA `all_figures` JSON이 figure filename → caption text 직접 매핑 (저자 작성 GT)
2. Label은 filename에서 정규식 (`^.*-(Figure|Table)\d+-\d+\.png$`)
3. Confidence: 1.0

**`refer_to`** (target precision >0.85):
1. Paragraph text에서 `(Figure|Fig\.?|Table|Tab\.?|Algorithm)\s*(\d+)` regex
2. `{label → figure_id}` map lookup → src=paragraph, dst=figure
3. False positive 차단: citation `[n]` ±50자 안 패턴은 외부 논문 참조로 보고 제외
4. Confidence: 0.9

**`contains`**:
- Paragraph→section: raw_tex 매칭 (normalize, paragraph 첫 50자 search, 직전 `\section{}` 찾기)
- Figure→section: caption text를 raw_tex에서 찾거나, `refer_to` source section plurality vote fallback
- Confidence: 0.9–1.0

**`reading_next`**: 연속 paragraph + 같은 section. Cross-section은 제외 (`section_next`가 처리)

**`section_next`**: `\section{}` 순서대로 directional

### A.3 Cross-doc fallback (SciEGQA / MMDocIR)

- **SciEGQA**: PDF only → docling 추출. `caption_of`는 bbox 인접 + label match heuristic
- **MMDocIR**: 원본 parquet에 layout 분해 이미 포함 → docling 불필요

### A.4 Validation gate (Week 2)

각 edge type 100 sample 수동 검수. target precision 미달 시 rule 보강 (citation filter window, proximity threshold). 미달 지속 시 LLM-aided extraction (Qwen2.5-7B).

---

## Appendix B. Hyperparameter setup

| Hyperparameter | Default | Sweep range |
|---|---|---|
| LoRA rank | 8 | {4, 8, 16, 32} |
| LoRA alpha | 16 | {8, 16, 32} |
| Batch size | 64 | {32, 64, 128} |
| Learning rate | 1e-5 | {3e-6, 1e-5, 3e-5} |
| Temperature τ (init) | 1/0.07 | learnable |
| GPE facet α (init raw) | -3.0 → σ ≈ 0.047 | learnable |
| γ (graph distance decay) | 0.5 | {0.3, 0.5, 0.7} |
| λ_cov | 0.3 | {0, 0.1, 0.3, 0.5, 1.0} |
| λ_cons | 0.5 | {0, 0.1, 0.3, 0.5, 1.0} |
| Coverage K | 10 | {5, 10, 20} |
| Coverage η | 5.0 | {1, 5, 10} |
| Propagation α | 0.3 | {0.1, 0.3, 0.5, 0.7} |
| Propagation T | 2 | {1, 2, 3} |
| Query PE dropout p (element anchor) | 0.5 | {0, 0.3, 0.5, 0.7, 1.0} |
| Hard negative per anchor | 1 | {0, 1, 3} |
| Training epoch | 5 | early stop on val Coverage@10 |
| Top-N for propagation | 50 | {25, 50, 100} |
| Token per visual element | 196 | {16, 64, 196} |

---

## Appendix C. 이전 method version과의 차이

### 폐기된 SA-AU (Structure-aware Alignment-and-Uniformity)
- Alignment-uniformity framework가 학술 도메인 특별 적합성 없음
- 결과 marginal (SciEGQA Hit@10 +1.7pp), graph alignment term은 negative
- 3 contribution이 thematically isolated, unifying thesis 부재

### 중간 버전 (v2 — graph as binary set-supervision + late interaction)
- Multi-positive set을 explicit sampling으로 정의
- Loss 5개 (multi-positive InfoNCE + $L_\text{cap}$ + $L_\text{ref}$ + $L_\text{cov}$ + no consistency)
- Train/inference query mismatch 미해결
- Section_path absolute encoding contamination 위험

### 현재 v2.1 (Graph PE + GRCL + L_cons + propagation)
- **단일 GRCL이 multi-positive sampler, caption_of supervision, refer_to supervision 흡수** — loss 3개
- **Graph PE는 cross-doc-safe relative feature만** (absolute section_path는 GPE에서 제외, inference within-doc scoring에만)
- **L_cons로 train/inference mismatch 명시 처리**
- **Single source of truth (`pipeline/types.py:BASE_EDGE_WEIGHTS`)** 가 GRCL supervision target과 propagation transition weight 양쪽을 통제

**살아남은 인프라**:
- 평가 framework (`eval/baselines/unified_eval.py`, dataset loader, metric)
- Motivation experiment (6 encoder × 3 dataset, 18 cell)
- 데이터 추출 파이프라인: docling for SciEGQA/MMDocIR, LaTeX-only for SPIQA
- Schema v2.1 (`pipeline/types.py`)
- pyvis 시각화 (`eval/analysis/visualize_graph_v2.py`)
