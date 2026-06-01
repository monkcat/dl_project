#!/usr/bin/env python
"""Extract trainable-only (LoRA + GPE + heads) adapters from full checkpoints.

Each ckpt/*.pt stores the full SigLIP backbone (~1.5 GB) but only ~4.8 MB is
trainable (LoRA adapters + GPE params). We drop the frozen pretrained backbone
and keep only the trained delta, so all 42 rows fit in ~200 MB. The adapter
loads via the existing path: build ElementTokenEncoder(...), then
load_state_dict(adapter, strict=False) — the frozen base stays pretrained.

Per-row build config (hf_id, lora_rank, lora_alpha, gpe_facets) is read from
each run's train.json and written to manifest.json so every adapter is loadable.

Usage: python scripts/export_adapters.py --out hf_upload
"""
import argparse
import json
import re
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "eval/results/experiments"


def is_trainable(k: str) -> bool:
    # frozen backbone lives under base.base_model.model.* (minus LoRA / logits)
    if "lora_" in k:
        return True
    if k.endswith("logit_scale") or k.endswith("logit_bias"):
        return True
    if not k.startswith("base.base_model.model."):
        return True  # raw_beta, gpe.*, any top-level adapter/proj
    return False


def clean_name(stem: str) -> str:
    m = re.match(r"^(.+)_\1$", stem)  # collapse doubled suffix (gamma_07_gamma_07)
    return m.group(1) if m else stem


def find_train_json(stem: str) -> Path | None:
    # checkpoint stem ~ experiment dir name; find best match
    cands = sorted(EXP.glob("*/train.json"))
    for tj in cands:
        if tj.parent.name == stem or tj.parent.name.startswith(stem):
            return tj
    # fallback: dir name contained in stem
    for tj in cands:
        if stem.startswith(tj.parent.name):
            return tj
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=str(REPO / "ckpt"))
    ap.add_argument("--out", default=str(REPO / "hf_upload"))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "adapters").mkdir(parents=True, exist_ok=True)
    manifest = {}
    total = 0

    for pt in sorted(Path(args.ckpt_dir).glob("*.pt")):
        name = clean_name(pt.stem)
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        adapter = {k: v for k, v in ck.items() if is_trainable(k)}
        dst = out / "adapters" / f"{name}.pt"
        torch.save(adapter, dst)
        sz = dst.stat().st_size
        total += sz

        cfg = {"hf_id": "google/siglip2-base-patch16-224",
               "lora_rank": 8, "lora_alpha": 16,
               "gpe_facets": "type,role,depth,pos"}
        tj = find_train_json(pt.stem)
        if tj:
            a = json.loads(tj.read_text()).get("args", {})
            for key in ("hf_id", "lora_rank", "lora_alpha", "gpe_facets"):
                if a.get(key) is not None:
                    cfg[key] = a[key]
        manifest[name] = {**cfg, "adapter": f"adapters/{name}.pt",
                          "n_trainable_keys": len(adapter),
                          "size_bytes": sz}
        print(f"  {name:28s} {len(adapter):3d} keys  {sz/1e6:5.1f} MB  "
              f"(rank={cfg['lora_rank']}, {cfg['hf_id'].split('/')[-1]})")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} adapters, total {total/1e6:.1f} MB → {out}")


if __name__ == "__main__":
    main()
