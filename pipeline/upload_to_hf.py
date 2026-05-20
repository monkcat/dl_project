"""Upload v2.1 element-graph metadata to a HuggingFace dataset repo.

This uploads ONLY graph metadata (graphs, elements, queries, schema) — NOT
images. Images come from the original SPIQA / SciEGQA / MMDocIR repos.

Total upload size: ~2.4 GB.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login

Usage:
    python -m pipeline.upload_to_hf --repo <username>/element-graph-v2.1
    # Examples:
    python -m pipeline.upload_to_hf --repo jaehyeon-lee/element-graph-v2.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

EXPORT_DIR = REPO / "data/hf_export/element-graph-v2.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="HF repo id, e.g. <username>/element-graph-v2.1")
    ap.add_argument("--private", action="store_true",
                    help="create private repo (default: public)")
    ap.add_argument("--commit_message", default="upload v2.1 element graph metadata")
    args = ap.parse_args()

    if not EXPORT_DIR.exists():
        print(f"ERROR: export directory not found: {EXPORT_DIR}")
        print("Run `python -m pipeline.prepare_hf_export` first.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    # ── Create repo if not exists ──
    print(f"Creating dataset repo: {args.repo}")
    try:
        create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"  ✓ repo ready: https://huggingface.co/datasets/{args.repo}")
    except Exception as e:
        print(f"ERROR creating repo: {e}")
        sys.exit(2)

    # ── Show what we're about to upload ──
    total = 0
    for p in sorted(EXPORT_DIR.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            total += sz
    print(f"\nUploading {EXPORT_DIR} ({total/1e9:.2f} GB)")
    print("Top-level structure:")
    for p in sorted(EXPORT_DIR.iterdir()):
        if p.is_dir():
            sub_total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            print(f"  {p.name}/  ({sub_total/1e6:.0f} MB)")
        else:
            print(f"  {p.name}  ({p.stat().st_size/1e3:.0f} KB)")
    print()

    # ── Upload ──
    print("Uploading...")
    api = HfApi()
    api.upload_folder(
        folder_path=str(EXPORT_DIR),
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=args.commit_message,
        ignore_patterns=["*.pyc", "__pycache__/*", ".DS_Store", "*.log"],
    )
    print(f"\n✓ Upload complete")
    print(f"  → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
