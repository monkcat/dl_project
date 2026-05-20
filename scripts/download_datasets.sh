#!/usr/bin/env bash
# Download benchmark datasets for the project.
#
# Usage:
#   bash scripts/download_datasets.sh [spiqa|mmdocir|sciegqa|all]
#
# Requirements: huggingface_hub (pip install huggingface_hub)
# For private/gated datasets, run `huggingface-cli login` first.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/data/benchmarks"
TARGET="${1:-all}"

download_spiqa() {
  echo "==> SPIQA (google/spiqa)"
  mkdir -p "$BENCH/spiqa"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='google/spiqa',
    repo_type='dataset',
    local_dir='$BENCH/spiqa',
    local_dir_use_symlinks=False,
)
"
}

download_mmdocir() {
  echo "==> MMDocIR (MMDocIR/MMDocRAG)"
  mkdir -p "$BENCH/mmdocir"
  # Org has multiple repos; download the main MMDocRAG dataset
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='MMDocIR/MMDocRAG',
    repo_type='dataset',
    local_dir='$BENCH/mmdocir',
    local_dir_use_symlinks=False,
)
"
  echo "    추가 repos: https://huggingface.co/MMDocIR (필요 시 수동 다운로드)"
}

download_sciegqa() {
  echo "==> SciEGQA"
  echo "    SciEGQA는 HF 공개 미러가 없을 수 있음."
  echo "    Project page: https://yuwenhan07.github.io/SciEGQA-project/"
  echo "    arxiv: https://arxiv.org/abs/2511.15090"
  echo "    수동으로 $BENCH/sciegqa 에 다운로드 후 재시도 필요"
}

case "$TARGET" in
  spiqa)   download_spiqa ;;
  mmdocir) download_mmdocir ;;
  sciegqa) download_sciegqa ;;
  all)
    download_spiqa
    download_mmdocir
    download_sciegqa
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Usage: $0 [spiqa|mmdocir|sciegqa|all]"
    exit 1
    ;;
esac

echo "==> Done. Outputs under $BENCH"
