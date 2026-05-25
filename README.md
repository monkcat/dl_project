# Element Graph for Academic Multimodal Retrieval

Typed document element graphs for retrieval research on academic papers
(**SPIQA / SciEGQA / MMDocIR**). The graph plays two roles:

1. **Training supervision** via *Graph-Relevance Contrastive Loss* (GRCL) —
   a graph-induced graded relevance distribution replaces binary positives.
2. **Inference-time score propagation** via typed edges with target-attribute
   modifiers (appendix boost, cross-modal boost) and an alternative
   *Personalized PageRank* (PPR) regime.

The same edge-weight schema (`pipeline/types.py:BASE_EDGE_WEIGHTS`) drives both
training and inference (single source of truth).

> **Full design + results**: see [REPORT_KR.md](REPORT_KR.md) for methodology,
> related work, dataset stats, experimental setup, and (post-run) results.

---

## What's in this repo

| Path | Contents |
|---|---|
| [`pipeline/`](pipeline/) | Core training + eval code ([details](pipeline/README.md)) |
| [`eval/`](eval/) | Evaluation pipeline + baselines ([details](eval/README.md)) |
| [`scripts/`](scripts/) | Launcher scripts + new-env playbook ([details](scripts/README_new_env.md)) |
| [`REPORT_KR.md`](REPORT_KR.md) | Full report (Korean) — motivation, schema, method, ablation design |
| `data/benchmarks/` | (gitignored) populated by setup scripts |
| `ckpt/` | (gitignored) trained checkpoints |
| `eval/results/` | (gitignored) runtime outputs |

---

## Quick start

### 1. Set up the environment

```bash
git clone https://github.com/monkcat/dl_project.git
cd dl_project

# Python — venv or conda either works
python3.10 -m venv .venv && source .venv/bin/activate
# (or)  conda create -n dl_hw2 python=3.10 -y && conda activate dl_hw2

# PyTorch (A100 → CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# HuggingFace login (for model downloads + private repos)
huggingface-cli login
```

### 2. Pull data and models

Choose ONE of these depending on your environment.

**(A) Direct download from HF on a connected machine** (recommended):
```bash
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1
```
Populates `data/benchmarks/{spiqa,sciegqa,mmdocir}/` from:
- `ljh38/element-graph-v2.1`     (our v2.1 graphs, ~2.4 GB)
- `google/spiqa`                 (~33 GB)
- `Yuwh07/SciEGQA-Bench`         (~1.3 GB)
- `MMDocIR/MMDocIR-Challenge`    (~2.5 GB)

**(B) GPU server can't reach HF** — download on laptop first, then copy:
```bash
# On laptop:
python scripts/download_all_to_local.py
# (creates ./dl_pack/ with data + models, ~48 GB)

# Transfer ./dl_pack/ to server (Backend.AI file browser, scp, rsync, ...)

# On server:
mv ~/dl_pack/data/benchmarks ~/dl_project/data/
echo 'export HF_HOME=~/dl_pack/hf_home'   >> ~/.bashrc
echo 'export HF_HUB_OFFLINE=1'            >> ~/.bashrc
echo 'export TRANSFORMERS_OFFLINE=1'      >> ~/.bashrc
source ~/.bashrc
```

### 3. Run the full experiment suite

```bash
bash scripts/run_full_suite.sh
```

This single command:
1. Runs preflight `pipeline.sanity_check` (validates configs, paths, schema, GPUs)
2. Launches **43 rows** across **4 tiers** on 2× A100 in pair-waves
3. Runs the **propagation sub-ablation** (21 inference variants) on the (h) checkpoint
4. Aggregates everything into `eval/results/experiments/SUMMARY.md`

Subset / resume options:
```bash
bash scripts/run_full_suite.sh --tier 1                 # just Tier 1 (~24h)
bash scripts/run_full_suite.sh --tier 1,2               # Tier 1 + 2
bash scripts/run_full_suite.sh --rows a,h,gme           # specific rows only
bash scripts/run_full_suite.sh --dry_run                # show schedule, don't run
bash scripts/run_full_suite.sh --no_prop_sweep          # skip the 21-variant propagation sub-ablation
```

The launcher **is resume-safe**: every row writes `summary.json` on completion,
and the next invocation skips already-done rows. Safe to Ctrl-C and re-run.

Total estimated time: **~4-5 days on 2× A100 80GB** (8k steps × 42 trained rows
+ eval). With `--tier 1`, ~24 hours.

---

## Experiment design overview

**43 rows total** = 42 trained + 1 GME zero-shot, organized into 4 tiers
(REPORT_KR §6.3 - §6.5):

| Tier | Rows | Purpose |
|---|---|---|
| **1 — Main ablation** | a, b, c, d, e, f, g, h, gme | InfoNCE/GRCL × GPE facet inclusion × L_cov/L_cons presence + GME reference |
| **2 — Negative controls** | m, n, o, p | Shuffled/random `section_role`, no query PE dropout, encoder swap (CLIP-L/14) |
| **3 — Sub-ablations** | 25 rows | Single-hyperparameter sweeps on (h): γ, λ_cov, λ_cons, LoRA rank, lr, τ, anchor kind, edge type, visual tokens |
| **4 — Follow-up** | 5 rows | h_best_combo, h_best_combo_16k, h_gpe_strong, h_seed_43/44 |

**Inference variants** applied per row (no retraining required):
- `no_prop` — encoder only (baseline)
- `wfull_a03_T2` — default weighted-diffusion propagation

Plus a **propagation sub-ablation** on the (h) checkpoint covering **21 variants**
across three regimes:
- **Group A — Uniform weights** (5): all edges weight=1, isolates the modifier design
- **Group B — Weighted diffusion** (9): modifier-component ablation + α/T sweep
- **Group C — Personalized PageRank** (7): teleport-based α sweep + uniform-PPR comparison

### Three propagation regimes (math)

| Regime | Formula | Hyperparams |
|---|---|---|
| **none** | `s` (identity) | — |
| **diffusion** | `s_{t+1} = (1-α)·s_t + α·P^T·s_t`, iterate T steps | α, T, weights mode |
| **PPR** | `s_{t+1} = α·q + (1-α)·P^T·s_t`, iterate to convergence | α (teleport prob), max_iter, weights mode |

Where `q = initial scores`, `P` is built from `BASE_EDGE_WEIGHTS` ×
`SECTION_ROLE_MODIFIER` × `VISUAL_TARGET_MODIFIER` per the
`weights ∈ {uniform, base, base+role, base+visual, full}` selection.

---

## Architecture summary

**Encoder**: SigLIPv2-base (200M) + LoRA (rank 8) + Graph Position Embedding (GPE)
- GPE: 4 sigmoid-gated facets (`type`, `role`, `depth`, `pos`) added token-broadcast
- Late interaction (ColBERT-style MaxSim) for retrieval scoring
- For row (p): CLIP-L/14 swap path with `d_text` ≠ `d_vision` handled via adapter

**Training losses**:
- `L_GRCL` — graded contrastive (graph-relevance kernel `g(q, e)`)
- `L_cov` — coverage boundary (positives above K-th negative)
- `L_cons` — PE-dropout consistency (KL between full-PE and PE-dropped query)

Total: `L = L_GRCL + λ_cov · L_cov + λ_cons · L_cons`

For full math, see [REPORT_KR.md](REPORT_KR.md) §4 Method.

---

## Available models

The encoder backbones used in this study:

| HF id | Used by | Size |
|---|---|---|
| `google/siglip2-base-patch16-224` | All Tier 1-4 except (gme), (p) | 200M |
| `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` | (gme) — zero-shot reference | 2B |
| `openai/clip-vit-large-patch14` | (p) — encoder swap negative control | 430M |

`pipeline.sanity_check --check_models` probes the HF cache and warns if any are missing.

---

## Reading results

After the suite finishes:

```bash
cat eval/results/experiments/SUMMARY.md
```

The summary has six sections matching REPORT_KR §7:

- §7.1 Tier 1 main ablation (per dataset × variant)
- §7.2 Tier 2 negative controls
- §7.3 Tier 3 sub-ablation sweeps (grouped by HP family)
- §7.4 Tier 4 follow-up extensions
- §7.5 Propagation sub-ablation (Group A / B / C tables)
- §7.6 Quick comparison: R@10 (no-prop) on every dataset × every row

Per-row raw eval JSON:
```bash
ls eval/results/experiments/h_full_method/
# → train.json   eval.json   summary.json
```

---

## License & attribution

Code: MIT (default for class projects).
Data: original dataset licenses (CC-BY 4.0 for SPIQA, etc.) apply where redistributed.
Citation: TBD (DL project, KAIST 2026).
