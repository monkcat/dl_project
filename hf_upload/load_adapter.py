#!/usr/bin/env python
"""Load an element-graph encoder adapter on top of its pretrained backbone.

Adapters store only the trained delta (LoRA + Graph Position Embedding, ~5 MB);
the frozen backbone is pulled from the HF hub at load time. Build config per
row (backbone id, LoRA rank, GPE facets) lives in manifest.json.

Requires the project code from https://github.com/monkcat/dl_project
(`pipeline/element_encoder.py`).

    from load_adapter import load_row
    enc = load_row("h_best_combo")          # ready-to-encode ElementTokenEncoder
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())


def load_row(row: str, device: str = "cuda"):
    from pipeline.element_encoder import ElementTokenEncoder  # project code

    if row not in MANIFEST:
        raise KeyError(f"{row!r} not in manifest. options: {sorted(MANIFEST)}")
    cfg = MANIFEST[row]
    enc = ElementTokenEncoder(
        hf_id=cfg["hf_id"],
        lora_rank=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        gpe_facets=tuple(cfg["gpe_facets"].split(",")),
        device=device,
    )
    state = torch.load(HERE / cfg["adapter"], map_location=device)
    missing, unexpected = enc.load_state_dict(state, strict=False)
    # 'missing' = frozen backbone keys (expected, stay pretrained); unexpected → 0
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    return enc.eval()


if __name__ == "__main__":
    import sys
    print(f"{len(MANIFEST)} rows available:")
    for r, c in sorted(MANIFEST.items()):
        print(f"  {r:28s} rank={c['lora_rank']:2d}  {c['hf_id'].split('/')[-1]}")
