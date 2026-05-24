"""Download everything (data + models) into a clean folder layout for transfer.

Run this on your LAPTOP. The resulting `./dl_pack/` directory can then be
uploaded to the GPU server (Backend.AI file browser, scp, rsync, whatever).

Resulting layout (everything self-contained, no symlinks):

    ./dl_pack/
    ├── data/
    │   └── benchmarks/
    │       ├── spiqa/{train_val, test-A}/{element_graph_v2.json, elements_v2.jsonl, SPIQA_*.json, images*/}
    │       ├── sciegqa/{element_graph_v2.json, elements_v2.jsonl, queries_with_gt_docling.jsonl, PDF/, Images/}
    │       └── mmdocir/{element_graph_v2.json, elements_v2.jsonl, academic_queries.jsonl, MMDocIR_*.parquet}
    └── hf_home/
        └── hub/
            ├── models--google--siglip2-base-patch16-224/...   (1.5 GB)
            ├── models--Alibaba-NLP--gme-Qwen2-VL-2B-Instruct/...  (8.3 GB)
            └── models--openai--clip-vit-large-patch14/...     (1.6 GB)

After upload to the GPU server, on the server:
    1. mv ~/dl_pack/data/benchmarks ~/dl_project/data/
    2. export HF_HOME=~/dl_pack/hf_home
       export HF_HUB_OFFLINE=1
       export TRANSFORMERS_OFFLINE=1
    3. bash scripts/run_all_experiments.sh

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login  # for any private/gated repos

Total download size:
    - data with SPIQA train images: ~37 GB
    - data without SPIQA train images (--skip_spiqa_train): ~6 GB
    - models: 11.4 GB
    Default (full): ~48 GB

Usage:
    python scripts/download_all_to_local.py
    python scripts/download_all_to_local.py --dest ~/dl_pack
    python scripts/download_all_to_local.py --skip_spiqa_train   # eval-only setup
    python scripts/download_all_to_local.py --only_models        # just the 3 models
    python scripts/download_all_to_local.py --only_data          # just the data
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

MODELS_NEEDED = [
    "google/siglip2-base-patch16-224",
    "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    "openai/clip-vit-large-patch14",
]


def have_hf_hub() -> bool:
    try:
        from huggingface_hub import snapshot_download, hf_hub_download  # noqa
        return True
    except ImportError:
        return False


# ───────────────────────────────────────────────────────────────────────────
# Models — populate dest/hf_home/hub/ in HF cache layout
# ───────────────────────────────────────────────────────────────────────────

def download_models(dest: Path) -> None:
    from huggingface_hub import snapshot_download

    hf_home = dest / "hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    print(f"\n[models] target: {hf_home}/")

    for mid in MODELS_NEEDED:
        print(f"  ↓ {mid}")
        path = snapshot_download(repo_id=mid, cache_dir=str(hf_home))
        # Show size
        sz = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        print(f"    → {sz/1e9:.2f} GB at {path}")


# ───────────────────────────────────────────────────────────────────────────
# Data — populate dest/data/benchmarks/ matching what the trainer expects
# ───────────────────────────────────────────────────────────────────────────

def link_or_copy(src: Path, dst: Path) -> None:
    """Copy src → dst (no symlinks since this is meant to be moved as a unit)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy(src, dst)


def download_our_graphs(dest_data: Path, graph_repo: str) -> None:
    """Download ljh38/element-graph-v2.1 and lay out under data/benchmarks/."""
    from huggingface_hub import snapshot_download

    print(f"\n[1/4] our v2.1 graph metadata: {graph_repo}")
    cache_dir = dest_data.parent / "_hf_export"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_export = cache_dir / graph_repo.split("/")[-1]
    snapshot_download(
        repo_id=graph_repo, repo_type="dataset",
        local_dir=str(hf_export), local_dir_use_symlinks=False,
    )

    benchmarks = dest_data / "benchmarks"

    # SPIQA: spiqa-v2.1/{train, val, test-A}
    spiqa_src = hf_export / "spiqa-v2.1"
    if spiqa_src.exists():
        for split, target_dir in [
            ("train",  benchmarks / "spiqa/train_val"),
            ("val",    benchmarks / "spiqa/train_val"),
            ("test-A", benchmarks / "spiqa/test-A"),
        ]:
            src = spiqa_src / split
            if not src.exists():
                continue
            if split == "train":
                link_or_copy(src / "element_graph.json", target_dir / "element_graph_v2.json")
                link_or_copy(src / "elements.jsonl",     target_dir / "elements_v2.jsonl")
                link_or_copy(src / "qa.json",            target_dir / "SPIQA_train.json")
            elif split == "val":
                link_or_copy(src / "element_graph.json", target_dir / "element_graph_v2_val.json")
                link_or_copy(src / "elements.jsonl",     target_dir / "elements_v2_val.jsonl")
                link_or_copy(src / "qa.json",            target_dir / "SPIQA_val.json")
            else:  # test-A
                link_or_copy(src / "element_graph.json", target_dir / "element_graph_v2.json")
                link_or_copy(src / "elements.jsonl",     target_dir / "elements_v2.jsonl")
                link_or_copy(src / "qa.json",            target_dir / "SPIQA_testA.json")
        print(f"  ✓ spiqa-v2.1 → {benchmarks}/spiqa/")

    # SciEGQA
    sciegqa_src = hf_export / "sciegqa-v2.1"
    if sciegqa_src.exists():
        target_dir = benchmarks / "sciegqa"
        link_or_copy(sciegqa_src / "element_graph.json", target_dir / "element_graph_v2.json")
        link_or_copy(sciegqa_src / "elements.jsonl",     target_dir / "elements_v2.jsonl")
        link_or_copy(sciegqa_src / "queries.jsonl",      target_dir / "queries_with_gt_docling.jsonl")
        print(f"  ✓ sciegqa-v2.1 → {target_dir}/")

    # MMDocIR
    mmdocir_src = hf_export / "mmdocir-v2.1"
    if mmdocir_src.exists():
        target_dir = benchmarks / "mmdocir"
        link_or_copy(mmdocir_src / "element_graph.json", target_dir / "element_graph_v2.json")
        link_or_copy(mmdocir_src / "elements.jsonl",     target_dir / "elements_v2.jsonl")
        link_or_copy(mmdocir_src / "queries.jsonl",      target_dir / "academic_queries.jsonl")
        print(f"  ✓ mmdocir-v2.1 → {target_dir}/")


def download_spiqa_originals(dest_data: Path, skip_train: bool) -> None:
    """SPIQA original from google/spiqa: figure images (+ optional train images)."""
    from huggingface_hub import hf_hub_download

    target = dest_data / "benchmarks/spiqa"
    print(f"\n[2/4] SPIQA originals → {target}/")

    files = [
        ("test-A/SPIQA_testA_Images_224px.zip",
            target / "test-A/SPIQA_testA_Images_224px.zip"),
    ]
    if not skip_train:
        files.append((
            "train_val/SPIQA_train_val_Images.zip",
            target / "train_val/SPIQA_train_val_Images.zip",
        ))
    else:
        print("  (skipping 31 GB train images — eval-only setup)")

    for hf_path, local in files:
        if local.exists():
            print(f"  (already have {hf_path})")
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  ↓ {hf_path}")
        hf_hub_download(
            repo_id="google/spiqa", repo_type="dataset", filename=hf_path,
            local_dir=str(target), local_dir_use_symlinks=False,
        )

    # Extract zips into the destination tree (no wrapper dir — matches trainer paths)
    for zp in target.rglob("*.zip"):
        marker = zp.parent / zp.stem
        if marker.exists() and any(marker.iterdir()):
            print(f"  (already extracted: {zp.name})")
            continue
        print(f"  unzip {zp.name}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(zp.parent)
        print(f"    ✓")
    # Keep zips? Delete to save space.
    for zp in target.rglob("*.zip"):
        zp.unlink()


def download_sciegqa_originals(dest_data: Path, repo_id: str) -> None:
    from huggingface_hub import snapshot_download

    target = dest_data / "benchmarks/sciegqa"
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/4] SciEGQA originals: {repo_id} → {target}/")

    snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        local_dir=str(target), local_dir_use_symlinks=False,
    )
    # Extract PDF.tar / Images.tar
    for tar_name in ("PDF.tar", "Images.tar"):
        tp = target / tar_name
        if not tp.exists():
            continue
        out_dir = target / tp.stem
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  (already extracted: {tar_name})")
            continue
        print(f"  extract {tar_name}")
        with tarfile.open(tp) as t:
            t.extractall(target)
    # Delete tars to save space
    for tar_name in ("PDF.tar", "Images.tar"):
        tp = target / tar_name
        if tp.exists():
            tp.unlink()


def download_mmdocir_originals(dest_data: Path) -> None:
    from huggingface_hub import hf_hub_download

    target = dest_data / "benchmarks/mmdocir"
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n[4/4] MMDocIR originals → {target}/")

    for hf_path in ("MMDocIR_layouts.parquet", "MMDocIR_annotations.jsonl"):
        local = target / hf_path
        if local.exists():
            print(f"  (already have {hf_path})")
            continue
        print(f"  ↓ {hf_path}")
        hf_hub_download(
            repo_id="MMDocIR/MMDocIR-Challenge", repo_type="dataset",
            filename=hf_path,
            local_dir=str(target), local_dir_use_symlinks=False,
        )


# ───────────────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="./dl_pack",
                    help="output directory (default: ./dl_pack)")
    ap.add_argument("--graph_repo", default="ljh38/element-graph-v2.1",
                    help="our v2.1 graph repo")
    ap.add_argument("--sciegqa_repo", default="Yuwh07/SciEGQA-Bench")
    ap.add_argument("--skip_spiqa_train", action="store_true",
                    help="skip 31 GB SPIQA train images (eval-only setup)")
    ap.add_argument("--only_models", action="store_true")
    ap.add_argument("--only_data", action="store_true")
    args = ap.parse_args()

    if not have_hf_hub():
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {dest}")

    # ── Models ──
    if not args.only_data:
        download_models(dest)

    # ── Data ──
    if not args.only_models:
        dest_data = dest / "data"
        dest_data.mkdir(exist_ok=True)
        download_our_graphs(dest_data, args.graph_repo)
        download_spiqa_originals(dest_data, args.skip_spiqa_train)
        try:
            download_sciegqa_originals(dest_data, args.sciegqa_repo)
        except Exception as e:
            print(f"  WARNING: sciegqa download failed: {e}")
        try:
            download_mmdocir_originals(dest_data)
        except Exception as e:
            print(f"  WARNING: mmdocir download failed: {e}")

        # Clean up the temp HF export cache
        export_cache = dest_data / "_hf_export"
        if export_cache.exists():
            shutil.rmtree(export_cache)

    # ── Verification + size summary ──
    print("\n" + "=" * 60)
    print(f"DONE — staged at: {dest}")
    print("=" * 60)

    def show(label: str, p: Path) -> None:
        if not p.exists():
            print(f"  ✗ {label}: missing ({p})")
            return
        sz = subprocess.check_output(["du", "-sh", str(p)]).decode().split()[0]
        print(f"  ✓ {label}: {sz}  ({p.relative_to(dest)})")

    show("models",           dest / "hf_home")
    show("data/benchmarks",  dest / "data/benchmarks")
    show("  spiqa",          dest / "data/benchmarks/spiqa")
    show("  sciegqa",        dest / "data/benchmarks/sciegqa")
    show("  mmdocir",        dest / "data/benchmarks/mmdocir")

    total = subprocess.check_output(["du", "-sh", str(dest)]).decode().split()[0]
    print(f"\n  TOTAL: {total}")

    print(f"""
Next steps:
  1. Upload all of {dest}/ to the GPU server (Backend.AI file browser, scp, ...)
  2. On the server, place it somewhere (e.g. ~/dl_pack/)
  3. Wire up:
       mv ~/dl_pack/data/benchmarks/* ~/dl_project/data/benchmarks/   # or symlink
       export HF_HOME=~/dl_pack/hf_home
       export HF_HUB_OFFLINE=1
       export TRANSFORMERS_OFFLINE=1
  4. bash scripts/run_all_experiments.sh
""")


if __name__ == "__main__":
    main()
