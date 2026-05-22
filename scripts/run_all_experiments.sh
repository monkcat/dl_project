#!/usr/bin/env bash
# Launch all ablation experiments on 2× A100.
#
# Strategy: 4 FT rows + 1 GME reference. Each FT run = 1 GPU.
# Run 2 rows in parallel per wave (GPU 0 + GPU 1).
#
# Wave 1:  (a) on GPU 0   +   (e) on GPU 1
# Wave 2:  (f) on GPU 0   +   (h) on GPU 1
# Wave 3:  (gme reference, no training, single GPU)
#
# Each row trains for ~5-6 hours on A100 (8k steps, batch 16).
# Each row's evaluation runs immediately after training on same GPU (~1 hour).
# Total wall time: ~12-14 hours.
#
# Usage:
#   bash scripts/run_all_experiments.sh
#
# Logs go to logs/<row_id>_<name>.log.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

mkdir -p logs ckpt eval/results/experiments

# Optional: activate conda env if conda is available and not already in dl_hw2.
# Skip silently if conda is not installed (venv/system python is fine).
if command -v conda >/dev/null 2>&1; then
    if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "dl_hw2" ]; then
        if conda env list | grep -q "^dl_hw2 "; then
            echo "[INFO] Activating conda env: dl_hw2"
            # shellcheck disable=SC1091
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate dl_hw2
        fi
    fi
fi
echo "[INFO] Using python: $(which python)"

# Sanity: required files exist
for f in \
    data/benchmarks/spiqa/train_val/element_graph_v2.json \
    data/benchmarks/spiqa/test-A/element_graph_v2.json \
    data/benchmarks/sciegqa/element_graph_v2.json \
    data/benchmarks/mmdocir/element_graph_v2.json; do
    if [ ! -f "$f" ]; then
        echo "MISSING data file: $f" >&2
        exit 1
    fi
done
echo "[INFO] All data files present."

run_one () {
    local row="$1"
    local gpu="$2"
    local logf="logs/${row}.log"
    echo "[GPU $gpu] launching row $row → $logf"
    CUDA_VISIBLE_DEVICES="$gpu" \
        python -m pipeline.run_experiment --config "$row" \
        > "$logf" 2>&1 &
    echo $!
}

wait_pids () {
    local pids=("$@")
    local fail=0
    for p in "${pids[@]}"; do
        if ! wait "$p"; then
            echo "[ERROR] process $p exited non-zero" >&2
            fail=$((fail + 1))
        fi
    done
    if [ "$fail" -gt 0 ]; then
        echo "[ERROR] $fail process(es) failed in this wave" >&2
        exit 1
    fi
}

# ── Wave 1: (a) and (e) ──
echo ""
echo "================================================================"
echo "= Wave 1: (a) baseline_infonce + (e) grcl_no_gpe"
echo "================================================================"
P1=$(run_one a 0)
P2=$(run_one e 1)
wait_pids "$P1" "$P2"
echo "[Wave 1] complete"

# ── Wave 2: (f) and (h) ──
echo ""
echo "================================================================"
echo "= Wave 2: (f) grcl_gpe + (h) full_method"
echo "================================================================"
P3=$(run_one f 0)
P4=$(run_one h 1)
wait_pids "$P3" "$P4"
echo "[Wave 2] complete"

# ── Wave 3: GME zero-shot reference (no training, eval only) ──
echo ""
echo "================================================================"
echo "= Wave 3: (gme) zero-shot reference + propagation"
echo "================================================================"
P5=$(run_one gme 0)
wait_pids "$P5"
echo "[Wave 3] complete"

# ── Aggregate ──
echo ""
echo "================================================================"
echo "= Aggregating results"
echo "================================================================"
python -m pipeline.aggregate_results \
    --out eval/results/experiments/SUMMARY.md
echo ""
echo "ALL EXPERIMENTS DONE. See eval/results/experiments/SUMMARY.md"
