#!/usr/bin/env bash
# Phase 3: post-HP-sweep follow-ups.
#   Wave 9 : h_best_combo + h_gpe_strong (parallel, ~1.5h each)
#   Wave 10: h_best_combo_16k (GPU0, ~3h) || prop-sweep re-eval on h_lr_high & h_tau_high (GPU1, ~30m)
#   Aggregate.
# Waits for any existing pipeline.trainer / run_experiment processes to finish first.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs ckpt eval/results/experiments

echo "[$(date)] Phase 3: waiting for Phase 2 to finish..." >&2
while pgrep -f "pipeline\.(trainer|run_experiment|eval_full)" >/dev/null 2>&1; do
    sleep 30
done
echo "[$(date)] Phase 2 clear, starting Phase 3" >&2

launch() {
    local r=$1 g=$2; shift 2
    echo "[GPU $g] launching $r $*" >&2
    CUDA_VISIBLE_DEVICES=$g python -m pipeline.run_experiment --config "$r" "$@" \
        > "logs/${r}.log" 2>&1 &
}

# ── Wave 9: best-combo retrain + GPE strong init ──
echo "=== Wave 9: h_best_combo + h_gpe_strong ===" >&2
launch h_best_combo 0; A=$!
launch h_gpe_strong 1; B=$!
wait $A && echo "h_best_combo done" >&2 || { echo "h_best_combo FAILED" >&2; exit 1; }
wait $B && echo "h_gpe_strong done"  >&2 || { echo "h_gpe_strong FAILED" >&2; exit 1; }

# ── Wave 10: 16k long training (GPU 0) || prop sweep eval on existing best ckpts (GPU 1) ──
echo "=== Wave 10: h_best_combo_16k + prop-sweep re-eval ===" >&2
launch h_best_combo_16k 0; C=$!

# Re-eval h_lr_high and h_tau_high on GPU 1 with extended INFERENCE_VARIANTS
(
    set -e
    echo "[GPU 1] prop-sweep re-eval: h_lr_high" >&2
    CUDA_VISIBLE_DEVICES=1 python -m pipeline.run_experiment --config h_lr_high \
        --skip_training --ckpt ckpt/h_lr_high_hp_lr_1e4.pt \
        > logs/h_lr_high_propsweep.log 2>&1
    echo "[GPU 1] prop-sweep re-eval: h_tau_high" >&2
    CUDA_VISIBLE_DEVICES=1 python -m pipeline.run_experiment --config h_tau_high \
        --skip_training --ckpt ckpt/h_tau_high_hp_tau_010.pt \
        > logs/h_tau_high_propsweep.log 2>&1
    echo "[GPU 1] prop-sweep re-eval: h_best_combo" >&2
    CUDA_VISIBLE_DEVICES=1 python -m pipeline.run_experiment --config h_best_combo \
        --skip_training --ckpt ckpt/h_best_combo_best_combo.pt \
        > logs/h_best_combo_propsweep.log 2>&1
) &
D=$!

wait $C && echo "h_best_combo_16k done" >&2 || { echo "h_best_combo_16k FAILED" >&2; exit 1; }
wait $D && echo "prop-sweep re-eval done" >&2 || { echo "prop-sweep FAILED" >&2; exit 1; }

echo "=== Aggregating ===" >&2
python -m pipeline.aggregate_results --out eval/results/experiments/SUMMARY.md \
    || echo "(aggregate failed, but results are saved)" >&2
echo "ALL PHASE 3 DONE"
