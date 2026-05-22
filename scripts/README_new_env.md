# 새 환경에서 실험 실행 (A100 × 2)

## 단계

### 1. 코드 클론 + Python 환경

Conda나 venv 둘 다 OK — 격리만 되면 충분.

```bash
git clone https://github.com/monkcat/dl_project.git
cd dl_project
```

**옵션 A — venv (가벼움, conda 안 깔린 환경 추천)**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

**옵션 B — conda**
```bash
conda create -n dl_hw2 python=3.10 -y
conda activate dl_hw2
```

### 2. 의존성 설치

```bash
# PyTorch (CUDA 12.1 - A100용)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 나머지는 requirements.txt 로 한 번에
pip install -r requirements.txt
```

### 3. HuggingFace 로그인

```bash
huggingface-cli login
# 토큰 입력 (read 권한이면 충분)
```

> SPIQA / SciEGQA / MMDocIR 모두 공개 dataset이지만 rate limit 회피 + private repo 대비 위해 로그인 권장.

### 4. 데이터 다운로드

```bash
# 전체 다운로드 (~38 GB: 우리 graph 2.4GB + SPIQA train 32GB + SciEGQA 1.3GB + MMDocIR 2.5GB)
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1

# 학습용 SPIQA train images (32GB) 제외하고 평가만 (~6 GB)
python scripts/setup_data_from_hf.py \
    --graph_repo ljh38/element-graph-v2.1 \
    --skip_spiqa_train
```

스크립트가 자동으로 처리:
- v2.1 graph metadata (`ljh38/element-graph-v2.1`) → `data/hf_export/element-graph-v2.1/`
- `data/benchmarks/{spiqa,sciegqa,mmdocir}/` 로 symlink 정리 (파일명도 trainer가 기대하는 형태로)
- SPIQA 원본 (`google/spiqa`) 다운로드 + zip 자동 추출
- SciEGQA 원본 (`Yuwh07/SciEGQA-Bench`) 다운로드 + `PDF.tar` / `Images.tar` 자동 추출
- MMDocIR 원본 (`MMDocIR/MMDocIR-Challenge`) 다운로드

다른 SciEGQA mirror를 쓰고 싶으면 `--sciegqa_repo <repo_id>` 로 override.

### 5. 데이터 검증

```bash
python -c "
import json
for p in [
    'data/benchmarks/spiqa/train_val/element_graph_v2.json',
    'data/benchmarks/spiqa/test-A/element_graph_v2.json',
    'data/benchmarks/sciegqa/element_graph_v2.json',
    'data/benchmarks/mmdocir/element_graph_v2.json',
]:
    g = json.load(open(p))
    print(f'{p}: {len(g)} docs')
"
```

기대 출력:
```
spiqa/train_val: 25459 docs
spiqa/test-A:    118 docs
sciegqa:         80 docs
mmdocir:         75 docs
```

### 6. 실험 실행

```bash
# 전체 (Tier 1+2+3, 31 rows ≈ 90h on 2× A100)
bash scripts/run_all_experiments.sh

# 또는 단계별
bash scripts/run_all_experiments.sh --tier 1     # Tier 1만 (9 rows, ~27h)
bash scripts/run_all_experiments.sh --tier 2     # Negative controls (4 rows, ~12h)
bash scripts/run_all_experiments.sh --tier 3     # Sub-ablations (18 rows, ~54h)

# 또는 특정 rows만
bash scripts/run_all_experiments.sh --rows a,h,gme

# 무엇이 돌아갈지만 확인 (실제 학습 X)
bash scripts/run_all_experiments.sh --dry_run
```

**실험 행 구성** (REPORT_KR §6.3–6.5):

| Tier | Rows | 설명 |
|---|---|---|
| **1 — Main** | a, b, c, d, e, f, g, h, gme | InfoNCE / GRCL × GPE facet 조합 + GME reference |
| **2 — Negative controls** | m, n, o, p | section_role shuffle/random, no query PE dropout, encoder swap CLIP-L/14 |
| **3 — Sub-ablations** | γ×2, λ_cov×4, λ_cons×4, lora×3, edge×3, tokens×2 | (h)를 base로 한 hyperparameter sweep |

**자동 처리**:
- Wave마다 GPU 0 + GPU 1 병렬 (총 15 wave)
- 각 row 끝나면 `eval/results/experiments/<row>_<name>/summary.json` 생성
- 이미 완료된 row는 자동 skip (중단 후 재실행 안전)
- 끝나면 (h) 체크포인트로 propagation α/T sweep eval-only 5 variant 추가 실행
- 자동 aggregate → `eval/results/experiments/SUMMARY.md`

> conda가 없으면 그냥 현재 활성화된 python으로 실행 (스크립트가 conda 존재 여부 자동 감지).

### 7. 결과 확인

```bash
cat eval/results/experiments/SUMMARY.md

# 개별 row 결과 (예: full method)
ls eval/results/experiments/h_full_method/
# → train.json, eval.json, summary.json

# 학습 곡선 / metric history
python -m json.tool eval/results/experiments/h_full_method/train.json | head -50
```

---

## 트러블슈팅

### GPU 메모리 부족
`pipeline/configs.py` 의 `COMMON["batch"]` 를 16 → 8 또는 4 로 감소.

### HF rate limit / 401
`huggingface-cli login` 재실행. 토큰은 read 권한이면 충분.

### SciEGQA tar 추출 실패
디스크 공간 부족 가능 (PDF.tar 132MB → ~200MB, Images.tar 1.1GB → ~1.5GB 추출됨). `df -h .` 확인.

### 기존 환경에서 SciEGQA 복사하고 싶을 때
HF 다운로드 대신 직접 복사 가능:
```bash
# 기존 환경
tar czf sciegqa_orig.tar.gz data/benchmarks/sciegqa/{PDF,Images,SciEGQA_Bench.jsonl}
# 새 환경 (scp 후)
tar xzf sciegqa_orig.tar.gz
# 그 뒤 --sciegqa_repo 없이 setup 스크립트 다시 돌리면 됨 (이미 있는 파일은 스킵)
```

### `ModuleNotFoundError: pipeline`
프로젝트 루트(`dl_project/`)에서 실행해야 함. `cd /path/to/dl_project` 먼저.

### 학습 중단 후 재개
체크포인트는 `ckpt/<row>_<name>.pt` 에 저장됨. resume은 현재 미구현이라 처음부터 다시 돌려야 함.

---

## 데이터 흐름 요약

```
HF: ljh38/element-graph-v2.1            (v2.1 그래프 metadata, ~2.4 GB)
HF: google/spiqa                         (SPIQA 원본 images, ~33 GB)
HF: Yuwh07/SciEGQA-Bench                 (SciEGQA 원본 PDF + page images, ~1.3 GB)
HF: MMDocIR/MMDocIR-Challenge            (MMDocIR layout parquet, ~2.5 GB)
       │
       ▼
  scripts/setup_data_from_hf.py
       │  (snapshot_download + zip/tar 추출 + symlink 정리)
       ▼
data/benchmarks/
  ├─ spiqa/{train_val,test-A}/{element_graph_v2.json, elements_v2.jsonl, SPIQA_train.json, SPIQA_testA.json, images/}
  ├─ sciegqa/{element_graph_v2.json, elements_v2.jsonl, queries_with_gt_docling.jsonl, PDF/, Images/}
  └─ mmdocir/{element_graph_v2.json, elements_v2.jsonl, academic_queries.jsonl, MMDocIR_layouts.parquet}
       │
       ▼
  bash scripts/run_all_experiments.sh
       │  (4 rows × train+eval, 2 GPU 병렬, ~12–14h)
       ▼
ckpt/                       모델 체크포인트
eval/results/experiments/   row별 train.json / eval.json
eval/results/experiments/SUMMARY.md     ← 최종 비교 표
```
