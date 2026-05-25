# `pipeline/` — Training, evaluation, and graph machinery

Single-source-of-truth code for the element-graph multimodal retrieval study.
Module dependencies are linear: each later module imports from earlier ones,
never the reverse.

```
        types.py
           │
           ├── graph_pe.py
           ├── graph_relevance.py
           ├── losses.py
           │
           └── element_encoder.py
                  │
                  ├── trainer.py
                  ├── eval_full.py
                  │
                  ├── run_experiment.py   (orchestrator)
                  ├── aggregate_results.py (reporter)
                  └── sanity_check.py     (preflight)
```

---

## Schema + math (`types.py`)

The lowest-level module — everything else imports `types`.

- **Node/edge schema** (`ElementType`, `EdgeType`, `DocumentGraph`):
  Six node types (`text`, `figure`, `table`, `caption`, `equation`, `section_header`)
  and five edge types (`caption_of`, `refer_to`, `contains`, `reading_next`,
  `section_next`).

- **Edge weighting**:
  - `BASE_EDGE_WEIGHTS` — per-type baseline (`caption_of=1.0`, `refer_to=0.8`, …)
  - `SECTION_ROLE_MODIFIER` — appendix/references boost (target-attribute)
  - `VISUAL_TARGET_MODIFIER` — cross-modal boost (1.1 for figure/table targets)

- **`graph_propagate`** — three regimes:
  - `method="none"` → identity
  - `method="diffusion"` → `s_{t+1} = (1-α)·s_t + α·P^T·s_t` (T iterations)
  - `method="ppr"` → `s_{t+1} = α·q + (1-α)·P^T·s_t` (PPR, iterate to convergence)
  Plus `weights` axis: `"uniform" | "base" | "base+role" | "base+visual" | "full"`.

- **`_build_transition_matrix`** — builds row-normalized P with the right
  modifier combination based on `weights` mode (factored out so the same
  edge-weight pipeline is used by diffusion and PPR).

- **`late_interaction_score`** — ColBERT-style MaxSim: for token-level
  encoder outputs `(K, D)` per element, score = Σ_i max_j q_tok[i] · e_tok[j].

---

## Graph Position Embedding (`graph_pe.py`)

GPE injects structural prior into the encoder representation as a token-broadcast,
β-gated additive vector:

```
T'_j = LayerNorm(T_j + β · gpe(v))     for all tokens j of element v
```

**Four facets** (each sigmoid-bounded learnable gate, init small ≈0.05):
- `type` — `nn.Embedding(6, D)` over `ElementType`
- `role` — `nn.Embedding(10, D)` over `SectionRole`
- `depth` — sinusoidal PE of `section_path` depth (int)
- `pos` — sinusoidal PE of intra-section position (float ∈ [0,1])

`active_facets` kwarg lets you disable any subset → rows (b), (c), (d) use
type-only / type+role / all four.

`extract_node_features(graph)` precomputes per-node (`type_id`, `role_id`,
`depth`, `intra_pos`) → cheap lookup at training time.

---

## Graph relevance kernel (`graph_relevance.py`)

`g(q, e)` for the GRCL loss: **max-product path weight** from anchor to
candidate, up to 2 hops, with per-step γ decay.

```python
g_vec = graph_relevance(anchor_id, candidate_ids, doc_graph,
                        gamma=0.5, max_hops=2,
                        edge_types_only={'caption_of'})   # optional filter
```

`edge_types_only` (row-level sub-ablation) restricts the BFS to a specific
edge type to isolate which edge family contributes most to retrieval.

`build_neighbor_index(graph)` → `{node_id: [(nbr_id, edge_dict), ...]}`,
amortizes graph traversal across multiple `graph_relevance` calls per doc.

---

## Loss functions (`losses.py`)

Three losses, summed with weights:

```
L = L_GRCL + λ_cov · L_cov + λ_cons · L_cons
```

| Loss | Purpose | Formula |
|---|---|---|
| **`grcl_loss`** | listwise contrastive on graded `g(q,e)` | weighted multi-positive InfoNCE (binary-fallback via `infonce_threshold` for InfoNCE rows) |
| **`coverage_loss`** | enforce positives ≥ top-K negative | softplus boundary margin |
| **`consistency_loss`** | train/inference mismatch fix | KL(scores_PE-on ‖ scores_PE-dropped) with stopgrad teacher |

`total_loss(L_grcl, L_cov, L_cons, lambda_cov, lambda_cons)` is the convenience aggregator.

---

## Token-level encoder (`element_encoder.py`)

`ElementTokenEncoder` wraps any HF AutoModel with text + vision towers
(SigLIPv2 by default; CLIP-L/14 supported via the `gpe_to_vision` adapter when
`d_text ≠ d_vision`).

**Forward flow** (per modality):
```
input → text_model / vision_model        (with LoRA)        → (K, D_hidden)
                            ↓
                      + β · gpe(v).unsqueeze(0)            # GPE addition
                            ↓
                      LayerNorm + Linear(D_hidden → D_proj)
                            ↓
                      F.normalize(dim=-1)                  # L2-norm tokens
                            ↓
                      (K, D_proj)                          # ready for late interaction
```

Key features:
- **LoRA**: applied to `q_proj/k_proj/v_proj/out_proj`; ~0.3% of base params trainable
- **CLS drop**: detected automatically for CLIP-style models (model_type contains "clip")
- **Patch pooling**: `tokens_per_visual=N` does 2D adaptive avg-pool on the patch grid
- **GPE adapter**: `Linear(d_text, d_vision)` zero-init when towers differ (CLIP-L)
- **`batch_encode_elements`**: groups text/vision elements for batched forward — major training speedup

---

## Training loop (`trainer.py`)

End-to-end fine-tuning. Reads config from `--config <row_id>` (preset from
`configs.py`) or accepts overrides via CLI.

Pipeline (per training step):
1. Sample N anchors from `SpiqaSplitData` (mix of caption_of / refer_to / NL-QA)
2. For each anchor build a 12-element candidate pool (positives + same-doc neg + cross-doc neg)
3. Compute graph_relevance `g(anchor, cand)` for each candidate
4. `batch_encode_elements` (one text + one vision forward for all candidates)
5. Late interaction scores → `compute_batch_losses`:
   - GRCL on scores vs graded `g`
   - L_cov on positive coverage at K
   - L_cons on PE-dropped teacher (skipped if `query_pe_dropout=0`)

Eval every 500 steps on held-out SPIQA test-A (caption-figure cosine + Recall@k).

**Negative control hooks** (Tier 2):
- `--section_role_mode shuffled|random` → mutates section_role at data load
- `--query_pe_dropout 0` → disables L_cons teacher pass
- `--hf_id openai/clip-vit-large-patch14` → encoder swap
- `--edge_types_only X` → restricts graph_relevance to one edge type

---

## Multi-dataset evaluation (`eval_full.py`)

For each `EVAL_DATASETS` entry:
1. Encode the document corpus once (`encode_corpus_for_doc`, dedup + batched)
2. For each query: encode → late interaction → top-50 candidates
3. Apply every requested `INFERENCE_VARIANTS` (none / diffusion / PPR)
4. Compute Recall@k, MRR, Coverage@K, PerfectSet@K, cross-page hit rate

Metrics handle the multi-positive case (SciEGQA/MMDocIR have multiple GT element
ids per query) and the cross-page case (GT spans multiple pages).

---

## Experiment orchestration

| Module | Role |
|---|---|
| `configs.py` | All row presets, inference variants, dataset paths. Single source of truth — both trainer and launcher import from here. |
| `run_experiment.py` | One row end-to-end: train (skips on `skip_training`) → eval → write `summary.json`. Tolerates Backend.AI nvmlShutdown segfault when train.json shows `"final"` key. |
| `aggregate_results.py` | Walks `eval/results/experiments/*/eval.json`, builds the 6-section markdown summary (§7.1 - §7.6). |
| `sanity_check.py` | Preflight: imports / configs / data paths / graph schema / GPU visibility. Run before launching the full suite. |

---

## Graph builders (not on the runtime path)

These build the v2.1 element graphs from raw datasets — run **once** to produce
`element_graph_v2.json` + `elements_v2.jsonl`. The runtime pipeline only
consumes these outputs.

| Builder | Source | Notes |
|---|---|---|
| `build_spiqa_graph_v2.py` | SPIQA arXiv LaTeX | `\section{}` + `\ref{}` + `all_figures` JSON |
| `build_docling_graph_v2.py` | SciEGQA + MMDocIR PDFs | Uses docling labels (section_header, caption, formula) |
| `extract_sciegqa_docling_v2.py` | SciEGQA PDFs | Re-extract preserving docling labels |
| `prepare_hf_export.py` | local v2.1 | Stage for HF upload |
| `upload_to_hf.py` | local hf_export | Push to `ljh38/element-graph-v2.1` |
| `upload_full_mirror.py` | local data + models | Push to private mirror repo (used when GPU server's HF access is broken) |

---

## Quick smoke tests

```bash
# 1. Show all rows and inference variants
python -m pipeline.configs

# 2. Preflight check before launching the full suite
python -m pipeline.sanity_check

# 3. Run a single row end-to-end (uses (h) preset)
python -m pipeline.run_experiment --config h

# 4. Re-evaluate an existing checkpoint with specific variants
python -m pipeline.eval_full --ckpt ckpt/h_full_method.pt \
    --variants no_prop wfull_a03_T2 ppr_a030 \
    --out eval/results/quick.json
```
