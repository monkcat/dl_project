#!/usr/bin/env bash
# Launch full ablation suite on 2× A100.
#
# Total training runs: 30 trained rows + 1 GME zero-shot = 31 rows.
# Tier 1 — Main ablation:     a, b, c, d, e, f, g, h, gme       (REPORT_KR §6.3)
# Tier 2 — Negative controls: m, n, o, p                        (REPORT_KR §6.4)
# Tier 3 — Sub-ablations:     γ×2, λ_cov×4, λ_cons×4, lora×3,
#                             edge×3                             (REPORT_KR §6.5)
#
# Inference variants applied to every row:
#   no_prop        encoder only
#   prop_a03_T2    + graph propagation α=0.3, T=2
# Propagation α/T sweep (prop_a01_T2, prop_a05_T2, prop_a07_T2, prop_a03_T1,
# prop_a03_T3) is applied ONLY on (h) after training, via a separate eval pass.
#
# Strategy: pair rows two-at-a-time on GPU 0 + GPU 1.
#   Per row: ~5h train + ~1h eval = ~6h. 28 rows / 2 GPUs = 14 waves ≈ 84h ≈ 3.5 days.
# Resume-friendly: each row writes summary.json on completion; the launcher skips
# rows that already have summary.json (so you can re-run after interruption).
#
# Usage:
#   bash scripts/run_all_experiments.sh                  # full suite
#   bash scripts/run_all_experiments.sh --tier 1         # just Tier 1
#   bash scripts/run_all_experiments.sh --rows a,b,h     # custom list
#   bash scripts/run_all_experiments.sh --dry_run        # show what would run

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p logs ckpt eval/results/experiments

# Optional: activate conda env if conda is available and not already in dl_hw2.
if command -v conda >/dev/null 2>&1; then
    if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "dl_hw2" ]; then
        if conda env list | grep -q "^dl_hw2 "; then
            echo "[INFO] Activating conda env: dl_hw2"
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate dl_hw2
        fi
    fi
fi
echo "[INFO] Using python: $(which python)"

# Sanity: required data files
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

# Parse args
TIER=""
ROWS=""
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --tier)    TIER="$2"; shift 2 ;;
        --rows)    ROWS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Resolve row list
if [ -n "$ROWS" ]; then
    IFS=',' read -ra ROW_LIST <<< "$ROWS"
elif [ "$TIER" = "1" ]; then
    ROW_LIST=(a b c d e f g h gme)
elif [ "$TIER" = "2" ]; then
    ROW_LIST=(m n o p)
elif [ "$TIER" = "3" ]; then
    ROW_LIST=(gamma_03 gamma_07
              cov_00 cov_01 cov_05 cov_10
              cons_00 cons_01 cons_03 cons_10
              lora_r4 lora_r16 lora_r32
              edge_caption_of edge_refer_to edge_contains
              tokens_016 tokens_064)
else
    # Full suite (default)
    ROW_LIST=(a b c d e f g h gme
              m n o p
              gamma_03 gamma_07
              cov_00 cov_01 cov_05 cov_10
              cons_00 cons_01 cons_03 cons_10
              lora_r4 lora_r16 lora_r32
              edge_caption_of edge_refer_to edge_contains
              tokens_016 tokens_064)
fi

# Skip rows already complete (summary.json exists). Use real config name (row_id_name).
already_done () {
    local row="$1"
    # Find any dir matching <row>_*/summary.json
    for d in eval/results/experiments/${row}_*/; do
        if [ -f "${d}summary.json" ]; then
            return 0  # done
        fi
    done
    return 1
}

# Pair up: launch two rows in parallel (GPU 0 + GPU 1)
run_one () {
    # Echo informational lines to stderr; only the final PID goes to stdout.
    local row="$1" gpu="$2"
    local logf="logs/${row}.log"
    if already_done "$row"; then
        echo "[GPU $gpu] SKIP $row (already complete)" >&2
        return 0
    fi
    echo "[GPU $gpu] launching $row → $logf" >&2
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
        python -m pipeline.run_experiment --config "$row" > "$logf" 2>&1 &
    echo $!
}

wait_pids () {
    local pids=("$@")
    local fail=0
    for p in "${pids[@]}"; do
        # Skip empty (from already-done rows)
        [ -z "$p" ] && continue
        if ! wait "$p"; then
            echo "[ERROR] process $p exited non-zero" >&2
            fail=$((fail + 1))
        fi
    done
    if [ "$fail" -gt 0 ]; then
        echo "[ERROR] $fail process(es) failed" >&2
        return 1
    fi
}

# Launch in pairs (GPU 0, GPU 1)
total=${#ROW_LIST[@]}
echo "[INFO] Total rows to run: $total"
i=0
wave=1
while [ $i -lt $total ]; do
    row0="${ROW_LIST[$i]}"
    row1=""
    [ $((i + 1)) -lt $total ] && row1="${ROW_LIST[$((i + 1))]}"

    echo ""
    echo "================================================================"
    echo "= Wave $wave: $row0 (GPU 0) + ${row1:-<none>} (GPU 1)"
    echo "================================================================"

    pid0=$(run_one "$row0" 0)
    pid1=""
    if [ -n "$row1" ]; then
        pid1=$(run_one "$row1" 1)
    fi
    wait_pids "$pid0" "$pid1" || echo "[WARN] wave $wave had failures (continuing)"
    echo "[Wave $wave] complete"

    i=$((i + 2))
    wave=$((wave + 1))
done

# ── Propagation α/T sub-ablation on (h) checkpoint only ──
if [ -f "ckpt/h_full_method.pt" ] && [ "$DRY_RUN" != "1" ]; then
    echo ""
    echo "================================================================"
    echo "= Propagation α/T sub-ablation on (h)"
    echo "================================================================"
    CUDA_VISIBLE_DEVICES=0 python -m pipeline.eval_full \
        --ckpt ckpt/h_full_method.pt \
        --variants prop_a01_T2 prop_a05_T2 prop_a07_T2 prop_a03_T1 prop_a03_T3 \
        --out eval/results/experiments/h_prop_sweep/eval.json \
        > logs/h_prop_sweep.log 2>&1 &
    wait $!
    echo "[Prop sweep] complete"
fi

# ── HP Sweep Wave 4: lr variants ──
echo ""
echo "================================================================"
echo "= HP Wave 4: (h_lr_low) lr=1e-5 + (h_lr_high) lr=1e-4"
echo "================================================================"
P6=$(run_one h_lr_low 0)
P7=$(run_one h_lr_high 1)
wait_pids "$P6" "$P7"
echo "[HP Wave 4] complete"

# ── HP Sweep Wave 5: tau variants ──
echo ""
echo "================================================================"
echo "= HP Wave 5: (h_tau_low) tau=0.05 + (h_tau_high) tau=0.10"
echo "================================================================"
P8=$(run_one h_tau_low 0)
P9=$(run_one h_tau_high 1)
wait_pids "$P8" "$P9"
echo "[HP Wave 5] complete"

# ── HP Sweep Wave 6: LoRA rank variants ──
echo ""
echo "================================================================"
echo "= HP Wave 6: (h_rank_low) rank=4 + (h_rank_high) rank=16"
echo "================================================================"
P10=$(run_one h_rank_low 0)
P11=$(run_one h_rank_high 1)
wait_pids "$P10" "$P11"
echo "[HP Wave 6] complete"

# ── HP Sweep Wave 7: lambda_cov variants ──
echo ""
echo "================================================================"
echo "= HP Wave 7: (h_lcov_low) lcov=0.1 + (h_lcov_high) lcov=0.5"
echo "================================================================"
P12=$(run_one h_lcov_low 0)
P13=$(run_one h_lcov_high 1)
wait_pids "$P12" "$P13"
echo "[HP Wave 7] complete"

# ── HP Sweep Wave 8: lambda_cons variants ──
echo ""
echo "================================================================"
echo "= HP Wave 8: (h_lcons_low) lcons=0.1 + (h_lcons_high) lcons=1.0"
echo "================================================================"
P14=$(run_one h_lcons_low 0)
P15=$(run_one h_lcons_high 1)
wait_pids "$P14" "$P15"
echo "[HP Wave 8] complete"

# ── Aggregate ──
if [ "$DRY_RUN" != "1" ]; then
    echo ""
    echo "================================================================"
    echo "= Aggregating results"
    echo "================================================================"
    python -m pipeline.aggregate_results \
        --out eval/results/experiments/SUMMARY.md
    echo ""
    echo "ALL EXPERIMENTS DONE. See eval/results/experiments/SUMMARY.md"
fi
