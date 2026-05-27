#!/usr/bin/env bash
# Server-side counterpart to scripts/upload_archives_to_hf.py.
#
# Downloads the archive-form mirror (≈17 files, ~35 GB) from your HF repo and
# extracts the zips/tars into the layout the trainer expects.
#
# Why archives: SPIQA train extracts to ~270k PNG files. Downloading 270k
# individual LFS objects is slow; downloading one 30GB zip then extracting
# locally is far faster.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login   # token that can read the (private) repo
#
# Usage:
#   bash scripts/setup_from_archives.sh
#   bash scripts/setup_from_archives.sh --repo ljh38/dl-project-archives
#   bash scripts/setup_from_archives.sh --dest ~/dl_dl --keep_archives
#   bash scripts/setup_from_archives.sh --with_models     # also pull hf_home/ + wire HF_HOME

set -euo pipefail

REPO_ID="${REPO_ID:-ljh38/dl-project-archives}"
DEST="${DEST:-$HOME/dl_archives}"
KEEP_ARCHIVES=0
WITH_MODELS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --repo)           REPO_ID="$2"; shift 2 ;;
        --dest)           DEST="$2"; shift 2 ;;
        --keep_archives)  KEEP_ARCHIVES=1; shift ;;
        --with_models)    WITH_MODELS=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$PROJECT_ROOT/data/benchmarks"

echo "[setup_from_archives] repo:    $REPO_ID"
echo "[setup_from_archives] dest:    $DEST"
echo "[setup_from_archives] project: $PROJECT_ROOT"

# ── 1. Download ──
mkdir -p "$DEST"
INCLUDE=(--include "data/**")
if [ "$WITH_MODELS" = "1" ]; then INCLUDE+=(--include "hf_home/**"); fi

echo ""
echo "[1/4] downloading archives from $REPO_ID → $DEST"
huggingface-cli download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$DEST" \
    "${INCLUDE[@]}"

# ── 2. Move metadata + archives into project's data/benchmarks/ ──
echo ""
echo "[2/4] placing files into $BENCH"
mkdir -p "$BENCH"
if [ -d "$DEST/data" ]; then
    # Copy preserves the spiqa/sciegqa/mmdocir subtree structure
    cp -rn "$DEST/data/." "$BENCH/" 2>/dev/null || cp -r "$DEST/data/." "$BENCH/"
fi

# ── 3. Extract archives in place (into the trainer-expected layout) ──
echo ""
echo "[3/4] extracting archives"

extract_zip () {
    local zip_path="$1" out_marker="$2"
    [ -f "$zip_path" ] || { echo "  (no $zip_path)"; return; }
    if [ -d "$out_marker" ] && [ -n "$(ls -A "$out_marker" 2>/dev/null)" ]; then
        echo "  (already extracted: $(basename "$out_marker"))"
        return
    fi
    echo "  unzip $(basename "$zip_path") ..."
    ( cd "$(dirname "$zip_path")" && python -c "
import zipfile, sys
with zipfile.ZipFile('$(basename "$zip_path")') as z:
    z.extractall('.')
" )
    echo "    ✓"
}

extract_tar () {
    local tar_path="$1" out_marker="$2"
    [ -f "$tar_path" ] || { echo "  (no $tar_path)"; return; }
    if [ -d "$out_marker" ] && [ -n "$(ls -A "$out_marker" 2>/dev/null)" ]; then
        echo "  (already extracted: $(basename "$out_marker"))"
        return
    fi
    echo "  untar $(basename "$tar_path") ..."
    ( cd "$(dirname "$tar_path")" && tar xf "$(basename "$tar_path")" )
    echo "    ✓"
}

# SPIQA images (zip top-level = SPIQA_*_Images → extract in place, no wrapper)
extract_zip "$BENCH/spiqa/train_val/SPIQA_train_val_Images.zip" \
            "$BENCH/spiqa/train_val/SPIQA_train_val_Images"
extract_zip "$BENCH/spiqa/test-A/SPIQA_testA_Images_224px.zip" \
            "$BENCH/spiqa/test-A/SPIQA_testA_Images_224px"
# SciEGQA (tar top-level = PDF/ , Images/)
extract_tar "$BENCH/sciegqa/PDF.tar"    "$BENCH/sciegqa/PDF"
extract_tar "$BENCH/sciegqa/Images.tar" "$BENCH/sciegqa/Images"

# Optionally reclaim disk by deleting archives after extraction
if [ "$KEEP_ARCHIVES" = "0" ]; then
    echo ""
    echo "  removing archives to reclaim disk (pass --keep_archives to retain)"
    rm -f "$BENCH/spiqa/train_val/SPIQA_train_val_Images.zip" \
          "$BENCH/spiqa/test-A/SPIQA_testA_Images_224px.zip" \
          "$BENCH/sciegqa/PDF.tar" "$BENCH/sciegqa/Images.tar"
fi

# ── 4. Models (optional) ──
if [ "$WITH_MODELS" = "1" ] && [ -d "$DEST/hf_home" ]; then
    echo ""
    echo "[4/4] HF_HOME setup"
    BASHRC="$HOME/.bashrc"
    if [ -f "$BASHRC" ] && ! grep -q "HF_HOME=$DEST/hf_home" "$BASHRC"; then
        {
            echo ""
            echo "# Added by dl_project setup_from_archives.sh"
            echo "export HF_HOME=$DEST/hf_home"
            echo "export HF_HUB_OFFLINE=1"
            echo "export TRANSFORMERS_OFFLINE=1"
        } >> "$BASHRC"
        echo "  ✓ wrote HF_HOME / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE to $BASHRC"
        echo "  run 'source ~/.bashrc' or open a new shell"
    else
        echo "  HF_HOME already configured in $BASHRC (or no .bashrc)"
    fi
fi

echo ""
echo "════════════════════════════════════════"
echo "[setup_from_archives] DONE"
echo "════════════════════════════════════════"
echo ""
echo "Verify:"
echo "  python -m pipeline.sanity_check"
echo ""
echo "Then launch:"
echo "  bash scripts/run_full_suite.sh"
