#!/usr/bin/env bash
# =============================================================================
# Waits for run_full_suite.sh to finish, then runs the deterministic analysis
# (scripts/run_analysis.sh). Meant to be launched DETACHED so it survives the
# IDE/SSH session closing:
#
#   nohup setsid bash scripts/watch_and_analyze.sh >> analysis_watch.log 2>&1 &
#
# Completion = the suite prints its "DONE" line, OR the launcher process has
# been gone for ~15 min (crash fallback → analyze whatever finished).
# =============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/dl/hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SUITE_LOG="full_suite_run.log"
POLL="${WATCH_POLL_SECONDS:-300}"

echo "[watch] $(date) — watching for suite completion (poll ${POLL}s)"
gone=0
while true; do
    if grep -q "^DONE — see" "$SUITE_LOG" 2>/dev/null; then
        echo "[watch] $(date) — suite signalled DONE"
        break
    fi
    if pgrep -f 'scripts/run_full_suite.sh' >/dev/null 2>&1; then
        gone=0
    else
        gone=$((gone + 1))
        echo "[watch] $(date) — launcher not running (consecutive=$gone)"
        if [ "$gone" -ge 3 ]; then
            echo "[watch] $(date) — launcher gone ~$((gone * POLL / 60))min, no DONE — analyzing partial results"
            break
        fi
    fi
    sleep "$POLL"
done

echo "[watch] $(date) — launching analysis"
bash scripts/run_analysis.sh
echo "[watch] $(date) — analysis complete. Figures in eval/results/figures/"
