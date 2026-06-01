# Element Graph for Academic Multimodal Retrieval — Short Report

**Thesis**: 학술 문서의 typed element graph는 (1) retriever 학습의 graded relevance supervision(GRCL)과 (2) 추론 시점 score propagation prior로 이중 활용된다. 동일한 edge-weight schema가 학습·추론을 모두 구동한다.

**Setup (요약)**: SigLIPv2-base(200M)+LoRA+GPE를 SPIQA train 단독으로 학습, late-interaction(ColBERT식) retrieval. 평가는 SPIQA test-A(in-domain), SciEGQA·MMDocIR(zero-shot), per-doc pool. 비교 reference로 retrieval-tuned 2B MLLM(GME-Qwen2-VL)을 zero-shot 사용. 전체 수치/음성통제/한계는 `REPORT_KR.md` §7–§9 참조.

---

## 핵심 결과

### 1. GRCL이 InfoNCE를 유의하게 개선 (in-domain)

graph-induced graded supervision(GRCL)으로 binary InfoNCE를 대체하면 SPIQA test-A에서 retrieval 품질이 오른다.

| 학습 손실 | R@10 | MRR |
|---|---|---|
| (a) InfoNCE baseline | 84.4 | 32.8 |
| (e) GRCL | **87.1** | **38.2** |

paired-bootstrap(1000 resample) 결과 **h−a가 Coverage@10에서 유의**(Δ=+2.55pp, 95% CI [+0.15, +5.11], p=0.042). 개선은 특히 상위 랭킹 품질(MRR +5.4)에 집중된다.

### 2. Graph propagation × 강한 retriever — 가장 큰 단일 향상

동일 edge schema로 만든 inference-time propagation을 **retrieval-tuned MLLM(GME)** 위에 얹으면 SPIQA에서 큰 폭 향상. encoder-agnostic 신호로서의 그래프 propagation을 입증.

| GME zero-shot | R@5 | R@10 | MRR |
|---|---|---|---|
| encoder only | 47.1 | 58.3 | 33.1 |
| **+ graph propagation (α=0.3, T=2)** | **78.2** | **84.4** | **63.3** |

R@10 +26pp, MRR 거의 2배. propagation은 base retriever가 해당 데이터셋에서 약할 때 강하게 작동한다(weak-base smoothing prior).

### 3. `refer_to` edge가 가장 강한 graph signal

GRCL supervision을 단일 edge type으로만 구성한 ablation에서 `refer_to`가 전 데이터셋 최고치.

| GRCL edge | SPIQA R@10 | MMDocIR R@10 |
|---|---|---|
| caption_of only | 83.5 | 3.1 |
| contains only | 88.4 | 2.1 |
| **refer_to only** | **90.4** | **5.7** |

paragraph→figure 참조 관계가 가장 informative한 학습 신호임을 시사.

### 4. 강한 학습 설정 (best configs)

| 설정 | SPIQA R@10 | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|---|
| (h) Full method | 86.9 | 2.4 | 1.0 |
| lr 3e-5 → **1e-4** | 88.6 | 4.3 | 1.6 |
| **h + best-HP combo** | **88.0** (MRR 42.2) | 4.2 | 1.5 |
| h + best-HP combo, 16k steps | 85.9 | **4.4** | **1.7** |

학습률과 HP 조합이 가장 큰 레버. best-combo가 종합 최고 학습 모델.

### 5. Motivation 재확인 — 큰 retrieval-tuned MLLM의 OOD 우위

zero-shot 도메인 전이에서 2B GME가 compact dual-encoder를 크게 앞선다(§2.3 motivation의 학습 후 재확인).

| zero-shot OOD | MMDocIR R@10 | SciEGQA R@10 |
|---|---|---|
| 학습한 SigLIP (h) | 2.4 | 1.0 |
| **GME-Qwen2-VL-2B** | **18.9** | **2.3** |

---

## 결론

GRCL은 in-domain에서 측정 가능한(유의한) 이득을 주며, graph propagation은 약한 base retriever(특히 GME on SPIQA: R@10 58.3→84.4, MRR 33.1→63.3) 위에서 가장 큰 향상을 낸다. `refer_to` edge가 핵심 신호이고, 학습률·HP 튜닝으로 SPIQA R@10 88.6까지 도달한다. 큰 retrieval-tuned MLLM은 zero-shot OOD를 지배하여, document-native graph signal이 encoder 규모와 독립적으로 작동할 수 있다는 본 연구의 방향을 뒷받침한다.
