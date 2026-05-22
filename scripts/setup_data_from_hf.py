"""Setup script for a new environment — downloads all needed data.

Run this AFTER cloning the code repo into the new environment. It pulls:

  1. Our v2.1 element graphs from HF (~2.4 GB; from --graph_repo)
  2. Original SPIQA dataset (~32 GB train + ~1 GB test-A; from google/spiqa)
  3. Original SciEGQA dataset (PDFs + Images.tar; from --sciegqa_repo)
  4. Original MMDocIR dataset (~2.5 GB parquet; from MMDocIR/MMDocIR-Challenge)

Skip flags let you download only what's needed:
    --skip_spiqa_train   (saves 32 GB)
    --skip_sciegqa
    --skip_mmdocir

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   # if private repo / rate limits

Usage:
    python scripts/setup_data_from_hf.py --graph_repo jaehyeon-lee/element-graph-v2.1

    # Quick setup for inference-only (no train images):
    python scripts/setup_data_from_hf.py \\
        --graph_repo jaehyeon-lee/element-graph-v2.1 \\
        --skip_spiqa_train
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def have_hf_hub():
    try:
        from huggingface_hub import snapshot_download  # noqa: F401
        return True
    except ImportError:
        return False


def download_graph_metadata(graph_repo: str, target: Path):
    """Download our v2.1 graph metadata (graphs/elements/queries/schema)."""
    from huggingface_hub import snapshot_download
    print(f"[1/4] Graph metadata from {graph_repo} → {target}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=graph_repo, repo_type="dataset",
        local_dir=str(target), local_dir_use_symlinks=False,
    )
    print(f"  ✓ done")


def materialize_graph_into_benchmarks(hf_export: Path, benchmarks: Path):
    """Map our HF export layout into data/benchmarks/ layout expected by code.

    HF layout:
        element-graph-v2.1/
            spiqa-v2.1/{train,val,test-A}/{element_graph.json, elements.jsonl, qa.json}
            sciegqa-v2.1/{element_graph.json, elements.jsonl, queries.jsonl}
            mmdocir-v2.1/{element_graph.json, elements.jsonl, queries.jsonl}

    Local layout (expected by trainer/eval):
        data/benchmarks/
            spiqa/train_val/{element_graph_v2.json, elements_v2.jsonl, SPIQA_train.json, ...}
            spiqa/test-A/{element_graph_v2.json, elements_v2.jsonl, SPIQA_testA.json}
            sciegqa/{element_graph_v2.json, elements_v2.jsonl, queries_with_gt_docling.jsonl}
            mmdocir/{element_graph_v2.json, elements_v2.jsonl, academic_queries.jsonl}
    """
    def link_or_copy(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy(src, dst)

    # SPIQA
    spiqa_src = hf_export / "spiqa-v2.1"
    if spiqa_src.exists():
        for split, target_dir in [
            ("train",  benchmarks / "spiqa/train_val"),
            ("val",    benchmarks / "spiqa/train_val"),
            ("test-A", benchmarks / "spiqa/test-A"),
        ]:
            src = spiqa_src / split
            if not src.exists(): continue
            # graphs / elements: rename to expected suffix
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
        print(f"  → linked spiqa-v2.1 into {benchmarks}/spiqa/")

    # SciEGQA
    sciegqa_src = hf_export / "sciegqa-v2.1"
    if sciegqa_src.exists():
        target_dir = benchmarks / "sciegqa"
        link_or_copy(sciegqa_src / "element_graph.json", target_dir / "element_graph_v2.json")
        link_or_copy(sciegqa_src / "elements.jsonl",     target_dir / "elements_v2.jsonl")
        link_or_copy(sciegqa_src / "queries.jsonl",      target_dir / "queries_with_gt_docling.jsonl")
        print(f"  → linked sciegqa-v2.1 into {target_dir}")

    # MMDocIR
    mmdocir_src = hf_export / "mmdocir-v2.1"
    if mmdocir_src.exists():
        target_dir = benchmarks / "mmdocir"
        link_or_copy(mmdocir_src / "element_graph.json", target_dir / "element_graph_v2.json")
        link_or_copy(mmdocir_src / "elements.jsonl",     target_dir / "elements_v2.jsonl")
        link_or_copy(mmdocir_src / "queries.jsonl",      target_dir / "academic_queries.jsonl")
        print(f"  → linked mmdocir-v2.1 into {target_dir}")


def download_spiqa(target: Path, skip_train: bool = False):
    """Download original SPIQA from google/spiqa.

    Files needed:
        test-A/SPIQA_testA_Images_224px.zip        ← test eval images
        train_val/SPIQA_train_val_Images.zip       ← training images (32 GB)
    """
    from huggingface_hub import hf_hub_download
    print(f"[2/4] SPIQA dataset → {target}")
    target.mkdir(parents=True, exist_ok=True)

    files_to_get = [
        # test-A
        ("test-A/SPIQA_testA_Images_224px.zip", target / "test-A/SPIQA_testA_Images_224px.zip"),
        # paragraph + raw_tex zips (used for graph build, but we have v2.1 already — get for completeness)
        ("SPIQA_train_val_test-A_extracted_paragraphs.zip",
         target / "SPIQA_train_val_test-A_extracted_paragraphs.zip"),
    ]
    if not skip_train:
        files_to_get.append((
            "train_val/SPIQA_train_val_Images.zip",
            target / "train_val/SPIQA_train_val_Images.zip",
        ))
    else:
        print(f"  (skipping 32 GB train images — set --skip_spiqa_train=False to include)")

    for hf_path, local in files_to_get:
        print(f"  downloading {hf_path}...")
        local.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id="google/spiqa", repo_type="dataset", filename=hf_path,
            local_dir=str(target), local_dir_use_symlinks=False,
        )
        print(f"    ✓ saved → {path}")


def extract_zips(target: Path):
    """Unzip downloaded SPIQA archives into the layout the trainer/eval expect.

    The zip top-level is `SPIQA_*_Images/`, but the trainer expects a wrapper
    directory above that (`images/` for train, `images_224px/` for test-A).
    We hardcode the destinations so the resulting layout matches:

        test-A/images_224px/SPIQA_testA_Images_224px/<paper_id>/*.png
        train_val/images/SPIQA_train_val_Images/<paper_id>/*.png

    Other zips (paragraphs etc.) are extracted in-place next to the archive.
    """
    import zipfile

    # SPIQA images: explicit destinations
    spiqa_image_zips = {
        target / "test-A/SPIQA_testA_Images_224px.zip":
            target / "test-A/images_224px",
        target / "train_val/SPIQA_train_val_Images.zip":
            target / "train_val/images",
    }
    for zp, dest in spiqa_image_zips.items():
        if not zp.exists():
            continue
        # If wrapper/<top> already populated, skip
        marker = dest / zp.stem  # e.g. images_224px/SPIQA_testA_Images_224px
        if marker.exists() and any(marker.iterdir()):
            print(f"  (already extracted: {marker.relative_to(target)})")
            continue
        print(f"  unzip {zp.name} → {dest.relative_to(target)}/")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(dest)
        print(f"    ✓")

    # Anything else: extract next to the archive (skip if already done)
    for zp in target.rglob("*.zip"):
        if zp in spiqa_image_zips:
            continue
        out_dir = zp.parent / zp.stem
        if out_dir.exists():
            print(f"  (already extracted: {out_dir.name})")
            continue
        print(f"  unzip {zp.name}...")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(zp.parent)
        print(f"    ✓")


def download_sciegqa(target: Path, repo_id: str | None):
    """Download SciEGQA from HF and extract PDF.tar / Images.tar.

    Default repo: Yuwh07/SciEGQA-Bench (contains SciEGQA_Bench.jsonl,
    PDF.tar ~132 MB, Images.tar ~1.13 GB).
    """
    if repo_id is None:
        repo_id = "Yuwh07/SciEGQA-Bench"
    from huggingface_hub import snapshot_download
    print(f"[3/4] SciEGQA from {repo_id} → {target}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        local_dir=str(target), local_dir_use_symlinks=False,
    )
    # extract PDF.tar / Images.tar in place
    import tarfile
    for tar_name in ("PDF.tar", "Images.tar"):
        tp = target / tar_name
        if not tp.exists():
            continue
        out_dir = target / tp.stem  # PDF/ or Images/
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  (already extracted: {tar_name})")
            continue
        print(f"  extracting {tar_name}...")
        with tarfile.open(tp) as t:
            t.extractall(target)
        print(f"    ✓")
    print(f"  ✓ done")


def download_mmdocir(target: Path):
    """Download MMDocIR_layouts.parquet from MMDocIR/MMDocIR-Challenge."""
    from huggingface_hub import hf_hub_download
    print(f"[4/4] MMDocIR → {target}")
    target.mkdir(parents=True, exist_ok=True)
    try:
        for hf_path in ("MMDocIR_layouts.parquet", "MMDocIR_annotations.jsonl"):
            print(f"  downloading {hf_path}...")
            path = hf_hub_download(
                repo_id="MMDocIR/MMDocIR-Challenge", repo_type="dataset",
                filename=hf_path,
                local_dir=str(target), local_dir_use_symlinks=False,
            )
            print(f"    ✓ saved → {path}")
    except Exception as e:
        print(f"  WARNING: MMDocIR download failed: {e}")
        print(f"  Manual download: https://huggingface.co/datasets/MMDocIR/MMDocIR-Challenge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_repo", required=True,
                    help="our HF dataset repo id, e.g. jaehyeon-lee/element-graph-v2.1")
    ap.add_argument("--data_root", default="data", help="local data directory")
    ap.add_argument("--skip_graph_metadata", action="store_true")
    ap.add_argument("--skip_spiqa", action="store_true")
    ap.add_argument("--skip_spiqa_train", action="store_true",
                    help="don't download the 32GB train_val images")
    ap.add_argument("--skip_sciegqa", action="store_true")
    ap.add_argument("--skip_mmdocir", action="store_true")
    ap.add_argument("--sciegqa_repo", default=None,
                    help="HF repo for SciEGQA (provider-specific)")
    ap.add_argument("--no_extract", action="store_true", help="skip unzip step")
    args = ap.parse_args()

    if not have_hf_hub():
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    data_root = REPO / args.data_root
    benchmarks = data_root / "benchmarks"
    benchmarks.mkdir(parents=True, exist_ok=True)

    # 1. v2.1 graph metadata
    if not args.skip_graph_metadata:
        hf_export = data_root / "hf_export/element-graph-v2.1"
        download_graph_metadata(args.graph_repo, hf_export)
        materialize_graph_into_benchmarks(hf_export, benchmarks)

    # 2. SPIQA original
    if not args.skip_spiqa:
        download_spiqa(benchmarks / "spiqa", skip_train=args.skip_spiqa_train)
        if not args.no_extract:
            print("  extracting SPIQA zips...")
            extract_zips(benchmarks / "spiqa")

    # 3. SciEGQA original
    if not args.skip_sciegqa:
        download_sciegqa(benchmarks / "sciegqa", args.sciegqa_repo)

    # 4. MMDocIR original
    if not args.skip_mmdocir:
        download_mmdocir(benchmarks / "mmdocir")

    print("\n" + "=" * 60)
    print("Setup complete. Verify:")
    for p in [
        "data/benchmarks/spiqa/train_val/element_graph_v2.json",
        "data/benchmarks/spiqa/test-A/element_graph_v2.json",
        "data/benchmarks/sciegqa/element_graph_v2.json",
        "data/benchmarks/mmdocir/element_graph_v2.json",
    ]:
        path = REPO / p
        ok = "✓" if path.exists() else "✗"
        print(f"  {ok} {p}")
    print("\nNext step: bash scripts/run_all_experiments.sh")


if __name__ == "__main__":
    main()
