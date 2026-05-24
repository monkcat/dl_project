"""Upload data + models to a single HF dataset repo for one-shot download on the GPU server.

Layout of the mirror repo:
    ljh38/dl-project-mirror/
    ├── data/
    │   └── benchmarks/
    │       ├── spiqa/   (test-A + train_val graph/elements/qa + train_val/images)
    │       ├── sciegqa/ (graph + elements + queries + PDF/ + Images/)
    │       └── mmdocir/ (graph + elements + queries + parquet)
    └── hf_home/
        └── hub/
            ├── models--google--siglip2-base-patch16-224/...
            ├── models--Alibaba-NLP--gme-Qwen2-VL-2B-Instruct/...
            └── models--openai--clip-vit-large-patch14/...

On GPU server, a single command pulls everything:
    huggingface-cli download ljh38/dl-project-mirror --repo-type dataset --local-dir ~/dl_mirror
    export HF_HOME=~/dl_mirror/hf_home
    ln -sfn ~/dl_mirror/data/benchmarks ~/dl_project/data/benchmarks

Prerequisites:
    pip install huggingface_hub rsync
    huggingface-cli login

Usage:
    # Stage + upload everything (default: private repo)
    python -m pipeline.upload_full_mirror --repo ljh38/dl-project-mirror

    # Skip 31GB SPIQA train images for first test
    python -m pipeline.upload_full_mirror --repo ljh38/dl-project-mirror --skip_spiqa_train_images

    # Stage only (don't upload yet — inspect /tmp/dl_mirror before pushing)
    python -m pipeline.upload_full_mirror --repo ljh38/dl-project-mirror --stage_only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS_NEEDED = [
    "google/siglip2-base-patch16-224",
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    "openai/clip-vit-large-patch14",
]

# Files/dirs to skip when mirroring data/benchmarks/ (intermediate artifacts, zip+extracted dup, docling-only obsolete files)
DATA_EXCLUDE = [
    "spiqa/train_val/SPIQA_train_val_Images.zip",     # 30GB zip — already extracted in images/
    "spiqa/test-A/SPIQA_testA_Images_224px.zip",      # 45MB zip — already extracted
    "spiqa/test-A/mineru_out/",                       # 1.2GB docling artifact
    "spiqa/test-A/paragraphs/",                       # parsing artifact
    "spiqa/test-A/pdfs/",                             # 246MB original PDFs (not needed)
    "spiqa/test-A/raw_tex/",                          # LaTeX source (not needed at runtime)
    "spiqa/test-A/elements_docling.jsonl",            # superseded by elements_v2.jsonl
    "spiqa/test-A/mineru_vs_spiqa_comparison.json",
    "spiqa/test-A/section_paths_docling.json",
    "spiqa/train_val/elements_docling.jsonl",
    "sciegqa/PDF.tar",                                # already extracted to PDF/
    "sciegqa/Images.tar",                             # already extracted to Images/
    "sciegqa/element_graph.json",                     # superseded by _v2.json
    "sciegqa/element_graph_docling.json",
    "sciegqa/elements.jsonl",
    "sciegqa/elements_docling.jsonl",
    "sciegqa/elements_docling_v2.jsonl",
    "sciegqa/queries_with_gt.jsonl",                  # superseded by _docling.jsonl
    "sciegqa/section_paths.json",
    "sciegqa/section_paths_docling.json",
    "sciegqa/section_paths_docling_v2.json",
]


def stage_models(staging: Path) -> None:
    """snapshot_download each model into staging/hf_home/ in HF cache format.

    Result lives at staging/hf_home/hub/models--<org>--<name>/...
    matching the ~/.cache/huggingface/hub layout, so on the server we just set
    HF_HOME=<extracted dir> and AutoModel.from_pretrained finds them.
    """
    from huggingface_hub import snapshot_download

    hf_home = staging / "hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    print(f"[stage models] target: {hf_home}/")

    for mid in MODELS_NEEDED:
        print(f"  ↪ snapshot_download {mid}")
        path = snapshot_download(repo_id=mid, cache_dir=str(hf_home))
        print(f"    saved → {path}")


def stage_data(staging: Path, skip_spiqa_train_images: bool) -> None:
    """rsync data/benchmarks/ → staging/data/benchmarks/ with exclude list."""
    src = REPO_ROOT / "data/benchmarks"
    dst = staging / "data/benchmarks"
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[stage data] {src} → {dst}/")

    excludes = list(DATA_EXCLUDE)
    if skip_spiqa_train_images:
        excludes.append("spiqa/train_val/images/")
        print("  (skipping 31GB SPIQA train images)")

    exclude_args: list[str] = []
    for e in excludes:
        exclude_args += ["--exclude", e]

    cmd = ["rsync", "-a", "--info=progress2"] + exclude_args + [
        str(src) + "/", str(dst) + "/"
    ]
    subprocess.run(cmd, check=True)


def show_staging(staging: Path) -> None:
    print("\n[staged contents]")
    for p in sorted(staging.iterdir()):
        if p.is_dir():
            size = subprocess.check_output(["du", "-sh", str(p)]).decode().split()[0]
            print(f"  {p.name}/  ({size})")
    total = subprocess.check_output(["du", "-sh", str(staging)]).decode().split()[0]
    print(f"  ──────")
    print(f"  total: {total}")


def upload_mirror(staging: Path, repo_id: str, private: bool, commit_msg: str) -> None:
    from huggingface_hub import HfApi, create_repo

    print(f"\n[upload] creating repo {repo_id} (private={private})")
    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"  https://huggingface.co/datasets/{repo_id}")

    api = HfApi()
    print(f"\n[upload] uploading {staging}/ → {repo_id}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_msg,
        ignore_patterns=["*.pyc", "__pycache__/*", ".DS_Store", "*.log"],
    )
    print(f"\n✓ upload complete")
    print(f"  → https://huggingface.co/datasets/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="ljh38/dl-project-mirror",
                    help="HF repo id for the mirror")
    ap.add_argument("--staging", default="/tmp/dl_mirror",
                    help="local staging dir (default: /tmp/dl_mirror)")
    ap.add_argument("--public", action="store_true",
                    help="make repo public (default: private — recommended for unclear licenses)")
    ap.add_argument("--no_models", action="store_true",
                    help="skip model staging+upload")
    ap.add_argument("--no_data", action="store_true",
                    help="skip data staging+upload")
    ap.add_argument("--skip_spiqa_train_images", action="store_true",
                    help="exclude the 31GB SPIQA train_val/images dir (eval-only setup)")
    ap.add_argument("--stage_only", action="store_true",
                    help="stage to local dir, don't upload (inspect first)")
    ap.add_argument("--upload_only", action="store_true",
                    help="skip staging, upload existing /tmp/dl_mirror")
    ap.add_argument("--commit_message", default="upload full mirror (data + models)")
    args = ap.parse_args()

    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)

    if not args.upload_only:
        if not args.no_models:
            stage_models(staging)
        if not args.no_data:
            stage_data(staging, args.skip_spiqa_train_images)
        show_staging(staging)

    if args.stage_only:
        print(f"\n[stage_only] not uploading. Inspect {staging}/ then re-run with --upload_only")
        return 0

    upload_mirror(staging, args.repo, private=not args.public, commit_msg=args.commit_message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
