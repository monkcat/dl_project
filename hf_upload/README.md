---
license: mit
language:
  - en
library_name: peft
tags:
  - retrieval
  - multimodal
  - document-retrieval
  - late-interaction
  - lora
  - siglip
base_model: google/siglip2-base-patch16-224
pipeline_tag: feature-extraction
---

# Element Graph Encoders (v2.1) — academic multimodal retrieval

Trained encoder **adapters** for the *Element Graph for Academic Multimodal
Retrieval* project. Each adapter is a SigLIPv2-base (200M) backbone fine-tuned
with **LoRA + Graph Position Embedding (GPE)**, trained on SPIQA with a
**Graph-Relevance Contrastive Loss (GRCL)** and late-interaction (ColBERT-style
MaxSim) scoring.

- **Code / method / full report**: https://github.com/monkcat/dl_project
- **Dataset + graphs**: [`ljh38/element-graph-v2.1`](https://huggingface.co/datasets/ljh38/element-graph-v2.1)
- **Backbone**: `google/siglip2-base-patch16-224` (loaded from the hub; not redistributed here)

## Why adapters only

Each full checkpoint is ~1.5 GB but only **~5 MB is trained** (LoRA + GPE);
the rest is the frozen pretrained SigLIP backbone, identical across all 42 rows.
This repo ships only the trained delta (≈263 MB for all 42 rows) and rebuilds
the backbone from the hub at load time.

## Usage

```bash
git clone https://github.com/monkcat/dl_project && cd dl_project
pip install -r requirements.txt
huggingface-cli download ljh38/element-graph-encoder-v2.1 --local-dir hf_models
```

```python
import sys; sys.path.append("hf_models")   # so load_adapter.py is importable
from load_adapter import load_row
enc = load_row("h_best_combo")   # builds SigLIP+LoRA+GPE, loads the adapter
```

Or evaluate directly with the project's eval harness (it loads with
`strict=False`, so the frozen backbone stays pretrained):

```bash
python -m pipeline.eval_full --ckpt hf_models/adapters/h_best_combo.pt \
    --lora_rank 8 --gpe_facets type,role,depth,pos \
    --datasets spiqa_testA sciegqa mmdocir
```

> Match `--lora_rank` / `--gpe_facets` / `--hf_id` to each row's entry in
> `manifest.json` (e.g. `lora_r32` needs `--lora_rank 32`, `p_encoder_clip_l14`
> needs `--hf_id openai/clip-vit-large-patch14`).

## Repository layout

```
adapters/<row>.pt     trained LoRA + GPE state (load with strict=False)
manifest.json         per-row build config: hf_id, lora_rank, lora_alpha, gpe_facets
load_adapter.py       helper: load_row(name) -> ready ElementTokenEncoder
```

## Selected results (SPIQA test-A / SciEGQA / MMDocIR, Recall@10, no propagation)

| Row | what it is | SPIQA | SciEGQA | MMDocIR |
|---|---|---|---|---|
| `a_baseline_infonce` | InfoNCE baseline | 84.4 | 0.9 | 2.0 |
| `e_grcl_no_gpe` | GRCL, no GPE | 87.1 | 1.0 | 2.5 |
| `h_full_method` | full method (GRCL+GPE+L_cov+L_cons) | 86.9 | 1.0 | 2.4 |
| `edge_refer_to` | GRCL on `refer_to` edges only | **90.4** | 1.3 | **5.7** |
| `lr_1e4` | lr 1e-4 | 88.6 | 1.6 | 4.3 |
| **`h_best_combo`** | best HP combo | 88.0 | 1.5 | 4.2 |
| `h_best_combo_16k` | best combo, 16k steps | 85.9 | 1.7 | 4.4 |

Key findings: GRCL gives a small but significant in-domain gain over InfoNCE
(SPIQA Coverage@10, paired-bootstrap p=0.042); `refer_to` edges are the strongest
graph signal; GPE / coverage / consistency losses show no measurable effect;
compact encoders fail to transfer zero-shot (SciEGQA/MMDocIR), where a large
retrieval-tuned MLLM (GME) leads. Full tables: see the GitHub report (§7).

## Available rows (42)

Tier 1 `a_baseline_infonce` `b_gpe_type_infonce` `c_gpe_type_role_infonce`
`d_gpe_full_infonce` `e_grcl_no_gpe` `f_grcl_gpe` `g_grcl_gpe_cov`
`h_full_method` · Tier 2 `m_shuffled_role` `n_random_role`
`o_no_query_pe_dropout` `p_encoder_clip_l14` · Tier 3 `gamma_03` `gamma_07`
`cov_00/01/05/10` `cons_00/01/03/10` `lora_r4/r16/r32` `lr_1e4` `lr_1e5`
`tau_005` `tau_010` `anchor_caption/refer/nlqa`
`edge_caption_of/refer_to/contains` `tokens_016/064` · Tier 4 `h_best_combo`
`h_best_combo_16k` `h_gpe_strong` `h_seed_43` `h_seed_44`

## License

MIT (code). Backbone and datasets retain their original licenses.
