#!/usr/bin/env bash
# =============================================================================
# Deterministic post-experiment analysis. Runs WITHOUT an interactive agent —
# safe to chain after run_full_suite.sh (see scripts/watch_and_analyze.sh).
#
# Produces, under eval/results/:
#   experiments/SUMMARY.md          §7.1-§7.6 result tables (aggregate_results)
#   figures/ablation_*.png          tier-1 ablation bars per metric
#   figures/propagation_*.png       propagation-variant sweep curves
#   figures/training_curves.png     loss vs step
#   figures/RESULTS_DIGEST.md       compact no_prop table
#   figures/results_long.csv        every (row,dataset,variant,metric)
#   figures/STAT_TESTS.md           paired-bootstrap CIs for key comparisons
#   figures/system_diagram.png      architecture diagram (no data needed)
#
# Usage: bash scripts/run_analysis.sh
# Env:   HF_HUB_CACHE / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE inherited.
# =============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${ANALYSIS_PY:-python}"
EXP="eval/results/experiments"
FIG="eval/results/figures"
mkdir -p "$FIG"

echo "[analysis] 1/5 aggregate_results → SUMMARY.md"
$PY -m pipeline.aggregate_results --out "$EXP/SUMMARY.md" \
    || echo "[analysis] WARN: aggregate_results failed"

echo "[analysis] 2/5 plot_results → figures + digest + csv"
$PY -m eval.analysis.plot_results --exp_root "$EXP" --out_dir "$FIG" \
    || echo "[analysis] WARN: plot_results failed"

echo "[analysis] 3/5 re-eval key rows with --dump_per_query (for stat tests)"
# Re-evaluate the main-ablation chain into eval/results/stat/ (does NOT touch
# the canonical eval.json) so paired-bootstrap has per-query data.
STAT_ROWS="${STAT_ROWS:-a e f g h}"
for row in $STAT_ROWS; do
    if ls "$EXP/${row}_"*/summary.json >/dev/null 2>&1; then
        if ! ls eval/results/stat/${row}_*/eval.perquery.json >/dev/null 2>&1; then
            echo "  re-eval $row ..."
            CUDA_VISIBLE_DEVICES="${ANALYSIS_GPU:-0}" $PY -m pipeline.run_experiment \
                --config "$row" --skip_training --dump_per_query \
                --out_root eval/results/stat \
                > "logs/reeval_${row}.log" 2>&1 \
                || echo "  WARN: re-eval $row failed (see logs/reeval_${row}.log)"
        fi
    fi
done

echo "[analysis] 4/5 stat_test → STAT_TESTS.md"
$PY -m eval.analysis.stat_test --exp_root eval/results/stat \
    --metric coverage@10 --variant no_prop \
    --compare h:a h:e f:e g:f --out "$FIG/STAT_TESTS.md" \
    || echo "[analysis] WARN: stat_test failed"

echo "[analysis] 5/5 system_diagram"
$PY -m eval.analysis.system_diagram --out "$FIG/system_diagram.png" 2>/dev/null \
    || echo "[analysis] (system_diagram skipped)"

echo "[analysis] DONE → $FIG  (+ $EXP/SUMMARY.md)"
