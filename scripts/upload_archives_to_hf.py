"""Upload data as ARCHIVES (not extracted) to a HF dataset repo.

Problem this solves:
    SPIQA train images extract to ~270k PNG files. Uploading that many small
    files to HuggingFace is painfully slow (each file = a separate LFS object).
    Instead we upload the original .zip / .tar archives (≈15 files total) plus
    the graph metadata, then extract on the server.

What gets uploaded (file count ≈ 17, total ≈ 35 GB):
    data/spiqa/train_val/SPIQA_train_val_Images.zip      (30 GB, 1 file)
    data/spiqa/train_val/{element_graph_v2.json, elements_v2.jsonl, SPIQA_train.json}
    data/spiqa/test-A/SPIQA_testA_Images_224px.zip
    data/spiqa/test-A/{element_graph_v2.json, elements_v2.jsonl, SPIQA_testA.json}
    data/sciegqa/{PDF.tar, Images.tar, element_graph_v2.json, elements_v2.jsonl, queries_with_gt_docling.jsonl}
    data/mmdocir/{MMDocIR_layouts.parquet, element_graph_v2.json, elements_v2.jsonl, academic_queries.jsonl}
    hf_home/hub/models--...   (optional, --with_models)

Counterpart: scripts/setup_from_archives.sh downloads + extracts on the server.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   # write access to the target repo

Usage:
    # data archives only (private repo, recommended for unclear licenses)
    python scripts/upload_archives_to_hf.py --repo ljh38/dl-project-archives

    # also bundle the 3 encoder models
    python scripts/upload_archives_to_hf.py --repo ljh38/dl-project-archives --with_models

    # skip the 30GB SPIQA train zip (eval-only mirror)
    python scripts/upload_archives_to_hf.py --repo ljh38/dl-project-archives --skip_spiqa_train

    # just print what would upload
    python scripts/upload_archives_to_hf.py --repo ljh38/dl-project-archives --dry_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "data/benchmarks"

# (local relative path under data/benchmarks, path inside the repo under data/)
ARCHIVE_FILES = [
    # SPIQA train
    ("spiqa/train_val/SPIQA_train_val_Images.zip", "spiqa/train_val/SPIQA_train_val_Images.zip"),
    ("spiqa/train_val/element_graph_v2.json",      "spiqa/train_val/element_graph_v2.json"),
    ("spiqa/train_val/elements_v2.jsonl",          "spiqa/train_val/elements_v2.jsonl"),
    ("spiqa/train_val/SPIQA_train.json",           "spiqa/train_val/SPIQA_train.json"),
    # SPIQA test-A
    ("spiqa/test-A/SPIQA_testA_Images_224px.zip",  "spiqa/test-A/SPIQA_testA_Images_224px.zip"),
    ("spiqa/test-A/element_graph_v2.json",         "spiqa/test-A/element_graph_v2.json"),
    ("spiqa/test-A/elements_v2.jsonl",             "spiqa/test-A/elements_v2.jsonl"),
    ("spiqa/test-A/SPIQA_testA.json",              "spiqa/test-A/SPIQA_testA.json"),
    # SciEGQA
    ("sciegqa/PDF.tar",                            "sciegqa/PDF.tar"),
    ("sciegqa/Images.tar",                         "sciegqa/Images.tar"),
    ("sciegqa/element_graph_v2.json",              "sciegqa/element_graph_v2.json"),
    ("sciegqa/elements_v2.jsonl",                  "sciegqa/elements_v2.jsonl"),
    ("sciegqa/queries_with_gt_docling.jsonl",      "sciegqa/queries_with_gt_docling.jsonl"),
    # MMDocIR
    ("mmdocir/MMDocIR_layouts.parquet",            "mmdocir/MMDocIR_layouts.parquet"),
    ("mmdocir/element_graph_v2.json",              "mmdocir/element_graph_v2.json"),
    ("mmdocir/elements_v2.jsonl",                  "mmdocir/elements_v2.jsonl"),
    ("mmdocir/academic_queries.jsonl",             "mmdocir/academic_queries.jsonl"),
]

# Files to drop when --skip_spiqa_train is set (the 30 GB train zip)
SPIQA_TRAIN_ZIP = "spiqa/train_val/SPIQA_train_val_Images.zip"

MODELS_NEEDED = [
    "google/siglip2-base-patch16-224",
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    "openai/clip-vit-large-patch14",
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ljh38/dl-project-archives",
                    help="target HF dataset repo")
    ap.add_argument("--public", action="store_true",
                    help="make repo public (default: private)")
    ap.add_argument("--with_models", action="store_true",
                    help="also stage + upload the 3 encoder models under hf_home/")
    ap.add_argument("--skip_spiqa_train", action="store_true",
                    help="exclude the 30GB SPIQA train zip (eval-only mirror)")
    ap.add_argument("--dry_run", action="store_true",
                    help="list files + sizes, don't upload")
    ap.add_argument("--commit_message", default="upload data archives + metadata")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo, snapshot_download
    except ImportError:
        print("ERROR: pip install huggingface_hub")
        return 1

    # ── Resolve the file list ──
    files = list(ARCHIVE_FILES)
    if args.skip_spiqa_train:
        files = [(l, r) for (l, r) in files if l != SPIQA_TRAIN_ZIP]

    print(f"Target repo: {args.repo}  (private={not args.public})")
    print(f"\nData files to upload ({len(files)}):")
    total = 0
    missing = []
    for local_rel, repo_rel in files:
        p = BENCH / local_rel
        if not p.exists():
            missing.append(local_rel)
            print(f"  ✗ MISSING  data/{repo_rel}")
            continue
        sz = p.stat().st_size
        total += sz
        print(f"  ✓ {human(sz):>9}  data/{repo_rel}")
    print(f"  ── total: {human(total)} across {len(files) - len(missing)} files")

    if missing:
        print(f"\nWARNING: {len(missing)} files missing — they will be skipped.")

    # ── Optional: stage models into a temp hf_home ──
    staged_hf_home = None
    if args.with_models:
        staged_hf_home = REPO_ROOT / "data/_mirror_hf_home"
        print(f"\nStaging {len(MODELS_NEEDED)} models into {staged_hf_home}/ ...")
        if not args.dry_run:
            staged_hf_home.mkdir(parents=True, exist_ok=True)
            for mid in MODELS_NEEDED:
                print(f"  ↪ {mid}")
                snapshot_download(repo_id=mid, cache_dir=str(staged_hf_home))

    if args.dry_run:
        print("\n[dry_run] nothing uploaded.")
        return 0

    # ── Create repo ──
    print(f"\nCreating repo {args.repo} ...")
    create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    api = HfApi()

    # ── Upload each data file individually (resume-friendly — failed files
    #    can be re-run without re-uploading the rest) ──
    for local_rel, repo_rel in files:
        p = BENCH / local_rel
        if not p.exists():
            continue
        print(f"  ↑ data/{repo_rel}  ({human(p.stat().st_size)})")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f"data/{repo_rel}",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"{args.commit_message}: {repo_rel}",
        )

    # ── Upload models (folder upload handles the HF cache layout) ──
    if args.with_models and staged_hf_home is not None:
        print(f"\n  ↑ hf_home/ (models)")
        api.upload_folder(
            folder_path=str(staged_hf_home),
            path_in_repo="hf_home",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"{args.commit_message}: models",
            ignore_patterns=["*.lock", "*.metadata", ".no_exist/*"],
        )

    print(f"\n✓ done → https://huggingface.co/datasets/{args.repo}")
    print("\nOn the server, run:")
    print(f"  bash scripts/setup_from_archives.sh --repo {args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
