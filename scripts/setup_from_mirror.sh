#!/usr/bin/env bash
# One-shot setup on the GPU server, pulling from ljh38/dl-project-mirror.
#
# This bypasses the original HF dataset repos (google/spiqa,
# Yuwh07/SciEGQA-Bench, MMDocIR/MMDocIR-Challenge) and downloads everything
# — data + models — from a single mirror repo.
#
# Use this when:
#   - GPU server can't reach google/spiqa or other public HF repos (DPI block)
#   - But can reach your own private HF repo
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login   (with token that can read the private mirror repo)
#
# Usage:
#   bash scripts/setup_from_mirror.sh
#   bash scripts/setup_from_mirror.sh --repo ljh38/dl-project-mirror --dest ~/dl_mirror
#   bash scripts/setup_from_mirror.sh --skip_models     # data only

set -euo pipefail

MIRROR_REPO="${MIRROR_REPO:-ljh38/dl-project-mirror}"
DEST="${DEST:-$HOME/dl_mirror}"
SKIP_MODELS=0
SKIP_DATA=0

while [ $# -gt 0 ]; do
    case "$1" in
        --repo)         MIRROR_REPO="$2"; shift 2 ;;
        --dest)         DEST="$2"; shift 2 ;;
        --skip_models)  SKIP_MODELS=1; shift ;;
        --skip_data)    SKIP_DATA=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "[setup_from_mirror] repo:    $MIRROR_REPO"
echo "[setup_from_mirror] dest:    $DEST"
echo "[setup_from_mirror] project: $REPO_ROOT"

# ── 1. Download from mirror repo ──
mkdir -p "$DEST"

if [ "$SKIP_MODELS" = "1" ] && [ "$SKIP_DATA" = "1" ]; then
    echo "ERROR: --skip_models and --skip_data both set, nothing to do"
    exit 1
fi

ALLOW_PATTERNS=()
if [ "$SKIP_MODELS" = "0" ]; then ALLOW_PATTERNS+=(--include "hf_home/**"); fi
if [ "$SKIP_DATA"   = "0" ]; then ALLOW_PATTERNS+=(--include "data/**"); fi

echo ""
echo "[1/3] downloading from $MIRROR_REPO → $DEST"
huggingface-cli download "$MIRROR_REPO" \
    --repo-type dataset \
    --local-dir "$DEST" \
    "${ALLOW_PATTERNS[@]}"

# ── 2. Wire up data: symlink dl_project/data/benchmarks → mirror/data/benchmarks ──
if [ "$SKIP_DATA" = "0" ] && [ -d "$DEST/data/benchmarks" ]; then
    echo ""
    echo "[2/3] linking data into project: $REPO_ROOT/data/benchmarks"
    mkdir -p "$REPO_ROOT/data"
    if [ -e "$REPO_ROOT/data/benchmarks" ] && [ ! -L "$REPO_ROOT/data/benchmarks" ]; then
        echo "  WARNING: $REPO_ROOT/data/benchmarks exists and is not a symlink"
        echo "  Move it aside or remove it manually if you want to replace it."
    else
        ln -sfn "$DEST/data/benchmarks" "$REPO_ROOT/data/benchmarks"
        echo "  → $REPO_ROOT/data/benchmarks -> $DEST/data/benchmarks"
    fi
fi

# ── 3. Set HF_HOME so models are picked up from the mirror cache ──
if [ "$SKIP_MODELS" = "0" ] && [ -d "$DEST/hf_home" ]; then
    echo ""
    echo "[3/3] HF_HOME setup"
    echo "  Add the following to ~/.bashrc (or your shell rc):"
    echo ""
    echo "    export HF_HOME=$DEST/hf_home"
    echo "    export HF_HUB_OFFLINE=1        # block any re-download attempt"
    echo "    export TRANSFORMERS_OFFLINE=1  # same for transformers lib"
    echo ""

    # Offer to add automatically (idempotent)
    BASHRC="$HOME/.bashrc"
    if [ -f "$BASHRC" ] && ! grep -q "HF_HOME=$DEST/hf_home" "$BASHRC"; then
        echo "  Adding to $BASHRC..."
        {
            echo ""
            echo "# Added by dl_project setup_from_mirror.sh"
            echo "export HF_HOME=$DEST/hf_home"
            echo "export HF_HUB_OFFLINE=1"
            echo "export TRANSFORMERS_OFFLINE=1"
        } >> "$BASHRC"
        echo "  ✓ updated $BASHRC (run 'source ~/.bashrc' or open a new shell)"
    fi
fi

echo ""
echo "════════════════════════════════════════"
echo "[setup_from_mirror] DONE"
echo "════════════════════════════════════════"
echo ""
echo "Verify with:"
echo "  ls $REPO_ROOT/data/benchmarks/"
echo "  ls $DEST/hf_home/hub/"
echo ""
echo "Then start experiments:"
echo "  source ~/.bashrc      # pick up HF_HOME"
echo "  bash scripts/run_all_experiments.sh"
