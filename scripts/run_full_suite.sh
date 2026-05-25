#!/usr/bin/env bash
# =============================================================================
# Full experiment suite — one-shot launcher for the entire study.
#
# Runs every row in pipeline/configs.py (Tier 1 + Tier 2 + Tier 3 + Tier 4)
# on 2× A100 80GB by pairing two rows per wave. Total trained rows: 42 + 1
# zero-shot (gme). After all rows finish, runs the 21-variant propagation
# sub-ablation on the (h) checkpoint.
#
# Resume-safe:
#   - Each row writes eval/results/experiments/<rid>_<name>/summary.json
#     on completion. The launcher skips rows that already have this file.
#   - Safe to Ctrl-C and re-run. Done rows are skipped.
#
# GPU usage:
#   - Wave = 2 rows in parallel, one per GPU.
#   - Backend.AI segfault on shutdown is treated as success when
#     train.json has "final" key and ckpt file exists (handled in
#     pipeline/run_experiment.py).
#
# Usage:
#   bash scripts/run_full_suite.sh                 # full suite (~4-5 days)
#   bash scripts/run_full_suite.sh --tier 1        # Tier 1 only (~24h)
#   bash scripts/run_full_suite.sh --tier 1,2      # Tier 1+2 only
#   bash scripts/run_full_suite.sh --rows a,h,gme  # specific rows only
#   bash scripts/run_full_suite.sh --dry_run       # show schedule without running
#   bash scripts/run_full_suite.sh --no_prop_sweep # skip the propagation sub-ablation
#
# Output:
#   ckpt/<rid>_<name>.pt                                trained checkpoint
#   eval/results/experiments/<rid>_<name>/eval.json     evaluation results
#   eval/results/experiments/<rid>_<name>/summary.json  done marker
#   eval/results/experiments/SUMMARY.md                 aggregated final report
#   logs/<rid>.log                                       full training/eval log
# =============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p logs ckpt eval/results/experiments

# ── Activate conda env if available; otherwise rely on caller's python ──
if command -v conda >/dev/null 2>&1; then
    if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "dl_hw2" ]; then
        if conda env list 2>/dev/null | grep -q "^dl_hw2 "; then
            echo "[INFO] activating conda env: dl_hw2" >&2
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate dl_hw2
        fi
    fi
fi
echo "[INFO] python: $(which python)" >&2

# ── Sanity-check required data files (fail fast, don't waste hours) ──
# Exit codes from pipeline.sanity_check:
#   0 = all green, 1 = warnings (still launchable), 2 = fatal
echo "[INFO] running preflight sanity check..." >&2
set +e
python -m pipeline.sanity_check --quiet
SC_RC=$?
set -e
if [ "$SC_RC" -ge 2 ]; then
    echo "[ERROR] sanity check FAILED (rc=$SC_RC) — see output above" >&2
    exit 1
fi
if [ "$SC_RC" -eq 1 ]; then
    echo "[INFO] sanity check passed with warnings (rc=1) — continuing" >&2
fi

# ============================================================================
# Argument parsing
# ============================================================================
TIER=""
ROWS=""
DRY_RUN=0
NO_PROP_SWEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --tier)          TIER="$2"; shift 2 ;;
        --rows)          ROWS="$2"; shift 2 ;;
        --dry_run)       DRY_RUN=1; shift ;;
        --no_prop_sweep) NO_PROP_SWEEP=1; shift ;;
        -h|--help)
            head -40 "$0" | sed 's/^# *//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ============================================================================
# Row schedule — explicit ordering for predictable resume behaviour
# ============================================================================
# Each tier is its own block; the launcher pairs rows 2-at-a-time.
# Long rows (h_best_combo_16k = 16k steps ≈ 2× normal) get a quiet pair.

TIER1=(a b c d e f g h gme)
TIER2=(m n o p)
TIER3=(gamma_03 gamma_07
       cov_00 cov_01 cov_05 cov_10
       cons_00 cons_01 cons_03 cons_10
       lora_r4 lora_r16 lora_r32
       lr_1e5 lr_1e4
       tau_005 tau_010
       anchor_caption anchor_refer anchor_nlqa
       edge_caption_of edge_refer_to edge_contains
       tokens_016 tokens_064)
TIER4=(h_best_combo h_gpe_strong h_seed_43 h_seed_44 h_best_combo_16k)

# Resolve which rows actually run, based on --tier / --rows
if [ -n "$ROWS" ]; then
    IFS=',' read -ra ROW_LIST <<< "$ROWS"
elif [ -n "$TIER" ]; then
    ROW_LIST=()
    IFS=',' read -ra T_LIST <<< "$TIER"
    for t in "${T_LIST[@]}"; do
        case "$t" in
            1) ROW_LIST+=("${TIER1[@]}") ;;
            2) ROW_LIST+=("${TIER2[@]}") ;;
            3) ROW_LIST+=("${TIER3[@]}") ;;
            4) ROW_LIST+=("${TIER4[@]}") ;;
            *) echo "unknown tier: $t (allowed 1, 2, 3, 4)" >&2; exit 1 ;;
        esac
    done
else
    # Default: full suite
    ROW_LIST=("${TIER1[@]}" "${TIER2[@]}" "${TIER3[@]}" "${TIER4[@]}")
fi

# ============================================================================
# Per-row helpers
# ============================================================================
already_done () {
    local row="$1"
    for d in eval/results/experiments/${row}_*/; do
        if [ -f "${d}summary.json" ]; then return 0; fi
    done
    return 1
}

run_one () {
    # stdout: the PID of the launched background process (empty if skipped)
    # stderr: human-readable status line
    local row="$1" gpu="$2"
    local logf="logs/${row}.log"
    if already_done "$row"; then
        echo "[GPU $gpu] SKIP $row (already done)" >&2
        return 0
    fi
    echo "[GPU $gpu] LAUNCH $row → $logf" >&2
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
        python -m pipeline.run_experiment --config "$row" \
        > "$logf" 2>&1 &
    echo $!
}

wait_pids () {
    local fail=0
    for p in "$@"; do
        [ -z "$p" ] && continue
        if ! wait "$p"; then
            echo "[WARN] PID $p exited non-zero — check log" >&2
            fail=$((fail + 1))
        fi
    done
    if [ "$fail" -gt 0 ]; then
        echo "[WARN] $fail process(es) failed in this wave (continuing — resume-safe)" >&2
    fi
}

# ============================================================================
# Schedule + run
# ============================================================================
total=${#ROW_LIST[@]}
echo "[INFO] schedule: $total rows" >&2

wave=1
i=0
while [ $i -lt $total ]; do
    row0="${ROW_LIST[$i]}"
    row1=""
    [ $((i + 1)) -lt $total ] && row1="${ROW_LIST[$((i + 1))]}"

    echo "" >&2
    echo "════════════════════════════════════════════════════════════════" >&2
    echo "  Wave $wave   GPU 0: $row0   GPU 1: ${row1:-<idle>}" >&2
    echo "════════════════════════════════════════════════════════════════" >&2

    pid0=$(run_one "$row0" 0)
    pid1=""
    if [ -n "$row1" ]; then
        pid1=$(run_one "$row1" 1)
    fi
    wait_pids "$pid0" "$pid1"
    echo "[Wave $wave] complete" >&2

    i=$((i + 2))
    wave=$((wave + 1))
done

# ============================================================================
# Propagation sub-ablation on (h) — eval only, runs on a single GPU
# (Skipped if --no_prop_sweep, or if (h) checkpoint is not available)
# ============================================================================
H_CKPT="ckpt/h_full_method.pt"
if [ "$NO_PROP_SWEEP" = "0" ] && [ "$DRY_RUN" = "0" ] && [ -f "$H_CKPT" ]; then
    PROP_OUT_DIR="eval/results/experiments/h_prop_sweep"
    if [ -f "$PROP_OUT_DIR/eval.json" ]; then
        echo "" >&2
        echo "[Prop sweep] already done — skip (delete $PROP_OUT_DIR/eval.json to re-run)" >&2
    else
        echo "" >&2
        echo "════════════════════════════════════════════════════════════════" >&2
        echo "  Prop sub-ablation on (h)   21 variants × 3 datasets ≈ 40 min" >&2
        echo "════════════════════════════════════════════════════════════════" >&2
        mkdir -p "$PROP_OUT_DIR"
        CUDA_VISIBLE_DEVICES=0 python -m pipeline.eval_full \
            --ckpt "$H_CKPT" \
            --variants \
                no_prop \
                uniform_a01_T2 uniform_a03_T1 uniform_a03_T2 uniform_a03_T3 uniform_a05_T2 \
                wbase_a03_T2 wbase_role_a03_T2 wbase_vis_a03_T2 \
                wfull_a01_T2 wfull_a03_T1 wfull_a03_T2 wfull_a03_T3 wfull_a05_T2 wfull_a07_T2 \
                ppr_a015 ppr_a030 ppr_a050 ppr_a070 ppr_a085 ppr_unif_a030 ppr_unif_a050 \
            --out "$PROP_OUT_DIR/eval.json" \
            > logs/h_prop_sweep.log 2>&1
        echo "[Prop sweep] complete" >&2
    fi
fi

# ============================================================================
# Aggregate results
# ============================================================================
if [ "$DRY_RUN" = "0" ]; then
    echo "" >&2
    echo "════════════════════════════════════════════════════════════════" >&2
    echo "  Aggregating results" >&2
    echo "════════════════════════════════════════════════════════════════" >&2
    python -m pipeline.aggregate_results \
        --out eval/results/experiments/SUMMARY.md \
        || echo "[WARN] aggregate failed; per-row results still safe in eval/results/experiments/" >&2
    echo "" >&2
    echo "DONE — see eval/results/experiments/SUMMARY.md" >&2
fi
