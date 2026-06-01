"""One-stop runner for a single experiment row.

Steps:
    1. Train the encoder using config preset (skip if --skip_training).
    2. Evaluate the resulting checkpoint on all 3 datasets × inference variants.
    3. Save trainer history + eval results into a single JSON.

Usage:
    # Full pipeline (recommended)
    python -m pipeline.run_experiment --config h

    # Only re-evaluate an existing checkpoint
    python -m pipeline.run_experiment --config h --skip_training \\
        --ckpt ckpt/h_full_method.pt

    # GME reference (no training, just evaluation)
    python -m pipeline.run_experiment --config gme
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.configs import EVAL_DATASETS, INFERENCE_VARIANTS, get_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="row id from configs.ROW_CONFIGS (a, e, f, h, gme)")
    ap.add_argument("--skip_training", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="override checkpoint path (default: ckpt/<row_id>_<name>.pt)")
    ap.add_argument("--out_root", type=str, default="eval/results/experiments",
                    help="directory for run-level output")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_queries", type=int, default=None,
                    help="cap on eval queries (debug)")
    ap.add_argument("--dump_per_query", action="store_true",
                    help="forward to eval_full: also write eval.perquery.json (for stat tests)")
    args = ap.parse_args()

    cfg = get_config(args.config)
    run_name = cfg["name"]
    out_dir = REPO / args.out_root / f"{cfg['row_id']}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.ckpt or f"ckpt/{cfg['row_id']}_{run_name}.pt"
    train_log = out_dir / "train.json"
    eval_log = out_dir / "eval.json"

    print(f"=== Experiment row {cfg['row_id']}: {cfg['name']} ===")
    print(f"  description: {cfg.get('description', '')}")
    print(f"  output dir:  {out_dir}")
    print(f"  checkpoint:  {ckpt_path}")

    # ── Training ──
    if not args.skip_training and not cfg.get("skip_training", False):
        print("\n--- Training ---")
        cmd = [
            sys.executable, "-m", "pipeline.trainer",
            "--config", args.config,
            "--out", str(train_log),
            "--save_ckpt", ckpt_path,
        ]
        print(f"$ {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=REPO)
        if result.returncode != 0:
            # Backend.AI's CUDA hook segfaults on nvmlShutdown during Python
            # exit even after training finished cleanly. Treat as success if
            # train.json was fully written (has the "final" key) and ckpt exists.
            ok = False
            try:
                if train_log.exists() and Path(ckpt_path).exists():
                    with open(train_log) as f:
                        d = json.load(f)
                    if "final" in d:
                        ok = True
            except Exception:
                ok = False
            if not ok:
                print(f"training failed (exit code {result.returncode}); aborting")
                return 1
            print(f"  training exited with code {result.returncode} but "
                  f"train.json + ckpt look complete — treating as success")
        print(f"  training done in {(time.time()-t0)/60:.1f} min")
    else:
        print(f"\n(skipping training for {cfg['row_id']})")

    # ── Evaluation ──
    if args.skip_eval:
        print("(skipping evaluation)")
        return 0

    print(f"\n--- Evaluation ---")
    cmd = [
        sys.executable, "-m", "pipeline.eval_full",
        "--out", str(eval_log),
        "--device", args.device,
    ]
    if not cfg.get("skip_training", False) and Path(ckpt_path).exists():
        cmd += ["--ckpt", ckpt_path]
    cmd += ["--lora_rank", str(cfg.get("lora_rank", 8))]
    cmd += ["--lora_alpha", str(cfg.get("lora_alpha", 16))]
    cmd += ["--hf_id", cfg.get("hf_id", "google/siglip2-base-patch16-224")]
    if cfg.get("encoder_kind"):
        cmd += ["--encoder_kind", cfg["encoder_kind"]]
    if cfg.get("gpe_facets"):
        cmd += ["--gpe_facets", cfg["gpe_facets"]]
    if cfg.get("tokens_per_visual"):
        cmd += ["--tokens_per_visual", str(cfg["tokens_per_visual"])]
    if args.max_queries:
        cmd += ["--max_queries", str(args.max_queries)]
    if args.dump_per_query:
        cmd += ["--dump_per_query"]

    available = [
        ds_id for ds_id, dc in EVAL_DATASETS.items()
        if (REPO / dc["elements"]).exists()
    ]
    missing = [ds for ds in EVAL_DATASETS if ds not in available]
    if missing:
        print(f"  WARNING: skipping eval datasets with missing elements file: {missing}")
    if not available:
        print("  ERROR: no eval datasets available")
        return 3
    cmd += ["--datasets"] + available

    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        ok = False
        try:
            if eval_log.exists():
                with open(eval_log) as f:
                    d = json.load(f)
                # eval_full writes per-dataset results; non-empty top-level dict ⇒ done
                if isinstance(d, dict) and len(d) > 0:
                    ok = True
        except Exception:
            ok = False
        if not ok:
            print(f"eval failed (exit code {result.returncode})")
            return 2
        print(f"  eval exited with code {result.returncode} but "
              f"eval.json looks complete — treating as success")
    print(f"  eval done in {(time.time()-t0)/60:.1f} min")

    # ── Summary ──
    summary = {
        "config_id": args.config,
        "config": cfg,
        "ckpt": ckpt_path,
        "train_log": str(train_log) if (not args.skip_training and not cfg.get("skip_training", False)) else None,
        "eval_log": str(eval_log),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Experiment {cfg['row_id']} complete → {out_dir}/summary.json ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
