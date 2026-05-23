#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ckpt eval/results/experiments

launch() {
    # Usage: launch <row_id> <gpu_id>
    local r=$1 g=$2
    echo "[GPU $g] launching $r" >&2
    CUDA_VISIBLE_DEVICES=$g python -m pipeline.run_experiment --config "$r" \
        > "logs/${r}.log" 2>&1 &
}

echo "=== Wave 1: a + e ===" >&2
launch a 0; A=$!
launch e 1; E=$!
wait $A && echo "a done" >&2 || { echo "a FAILED" >&2; exit 1; }
wait $E && echo "e done" >&2 || { echo "e FAILED" >&2; exit 1; }

echo "=== Wave 2: f + h ===" >&2
launch f 0; F=$!
launch h 1; H=$!
wait $F && echo "f done" >&2 || { echo "f FAILED" >&2; exit 1; }
wait $H && echo "h done" >&2 || { echo "h FAILED" >&2; exit 1; }

echo "=== Wave 3: gme ===" >&2
launch gme 0; G=$!
wait $G && echo "gme done" >&2 || { echo "gme FAILED" >&2; exit 1; }

echo "=== HP Wave 4: h_lr_low + h_lr_high ===" >&2
launch h_lr_low 0;  P4A=$!
launch h_lr_high 1; P4B=$!
wait $P4A && echo "h_lr_low done" >&2 || { echo "h_lr_low FAILED" >&2; exit 1; }
wait $P4B && echo "h_lr_high done" >&2 || { echo "h_lr_high FAILED" >&2; exit 1; }

echo "=== HP Wave 5: h_tau_low + h_tau_high ===" >&2
launch h_tau_low 0;  P5A=$!
launch h_tau_high 1; P5B=$!
wait $P5A && echo "h_tau_low done" >&2 || { echo "h_tau_low FAILED" >&2; exit 1; }
wait $P5B && echo "h_tau_high done" >&2 || { echo "h_tau_high FAILED" >&2; exit 1; }

echo "=== HP Wave 6: h_rank_low + h_rank_high ===" >&2
launch h_rank_low 0;  P6A=$!
launch h_rank_high 1; P6B=$!
wait $P6A && echo "h_rank_low done" >&2 || { echo "h_rank_low FAILED" >&2; exit 1; }
wait $P6B && echo "h_rank_high done" >&2 || { echo "h_rank_high FAILED" >&2; exit 1; }

echo "=== HP Wave 7: h_lcov_low + h_lcov_high ===" >&2
launch h_lcov_low 0;  P7A=$!
launch h_lcov_high 1; P7B=$!
wait $P7A && echo "h_lcov_low done" >&2 || { echo "h_lcov_low FAILED" >&2; exit 1; }
wait $P7B && echo "h_lcov_high done" >&2 || { echo "h_lcov_high FAILED" >&2; exit 1; }

echo "=== HP Wave 8: h_lcons_low + h_lcons_high ===" >&2
launch h_lcons_low 0;  P8A=$!
launch h_lcons_high 1; P8B=$!
wait $P8A && echo "h_lcons_low done" >&2 || { echo "h_lcons_low FAILED" >&2; exit 1; }
wait $P8B && echo "h_lcons_high done" >&2 || { echo "h_lcons_high FAILED" >&2; exit 1; }

echo "=== Aggregating ===" >&2
python -m pipeline.aggregate_results --out eval/results/experiments/SUMMARY.md
echo "ALL DONE"
