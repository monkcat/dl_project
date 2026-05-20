# 새 환경에서 실험 실행 (A100 × 2)

## 단계

### 1. 코드 클론 + 의존성 설치

```bash
git clone <your-repo-url> dl_project
cd dl_project

# Conda 환경 생성
conda create -n dl_hw2 python=3.10 -y
conda activate dl_hw2

# PyTorch (A100 → CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성
pip install transformers==4.57 peft pillow numpy pandas huggingface_hub pyvis matplotlib pytz
```

### 2. HuggingFace 로그인 (private repo이거나 rate limit 대비)

```bash
huggingface-cli login
# 토큰 입력
```

### 3. 데이터 다운로드 (HF에서)

```bash
# 옵션 A: 전체 다운로드 (SPIQA train images 32GB 포함)
python scripts/setup_data_from_hf.py \
    --graph_repo <YOUR_HF_USERNAME>/element-graph-v2.1

# 옵션 B: SPIQA train images 제외 (eval만 가능, 학습용 이미지 없음)
python scripts/setup_data_from_hf.py \
    --graph_repo <YOUR_HF_USERNAME>/element-graph-v2.1 \
    --skip_spiqa_train

# SciEGQA는 원본 dataset repo 알 경우 --sciegqa_repo 지정
# MMDocIR는 자동으로 MMDocIR/MMDocIR-Challenge 에서 다운로드
```

이게 자동으로:
- v2.1 graph metadata 다운로드 (`data/hf_export/element-graph-v2.1/`)
- 적절한 경로로 symlink 생성 (`data/benchmarks/{spiqa,sciegqa,mmdocir}/`)
- SPIQA / MMDocIR 원본 다운로드
- zip 자동 풀기

### 4. 데이터 검증

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
spiqa/test-A: 118 docs
sciegqa: 80 docs
mmdocir: 75 docs
```

### 5. 실험 실행

```bash
bash scripts/run_all_experiments.sh
```

이게 자동으로:
- Wave 1: GPU 0에 (a) baseline + GPU 1에 (e) GRCL → ~6시간
- Wave 2: GPU 0에 (f) GPE + GPU 1에 (h) full → ~6시간
- Wave 3: (gme) zero-shot reference → ~1시간
- 자동 aggregate → `eval/results/experiments/SUMMARY.md`

총 **12-14시간**.

### 6. 결과 확인

```bash
cat eval/results/experiments/SUMMARY.md

# 개별 row 결과
ls eval/results/experiments/h_full_method/
# train.json, eval.json, summary.json

# 학습 곡선 / metric history
cat eval/results/experiments/h_full_method/train.json | python -m json.tool | head -50
```

## Troubleshooting

### GPU 메모리 부족
`pipeline/configs.py` 의 `COMMON["batch"]` 16 → 8 또는 4로 감소.

### SciEGQA repo 미지정
`--sciegqa_repo` 옵션 생략 시 SciEGQA 다운로드 스킵. 학습 단계엔 영향 없음 (SPIQA만 학습). Eval에만 필요.

기존 환경에 SciEGQA가 있다면 `rsync`/`scp`로 복사:
```bash
# 기존 환경에서:
tar czf sciegqa_orig.tar.gz data/benchmarks/sciegqa/{PDF,Images.tar,SciEGQA_Bench.jsonl}
# 새 환경으로 전송 후:
tar xzf sciegqa_orig.tar.gz
```

### HF private repo
업로드 시 `--private` 플래그 사용했으면 다운로드 환경에서도 `huggingface-cli login` 필수.

### 학습 중단 후 재개
checkpoint가 `ckpt/<row>_<name>.pt`에 저장됨. 학습 재개는 trainer에 `--resume` 옵션 추가 필요 (현재 미구현 — 시작부터 다시).

## 데이터 흐름 요약

```
HF: <YOUR_HF>/element-graph-v2.1                 (우리 metadata, 2.4GB)
HF: google/spiqa                                  (원본 SPIQA, 32GB+)
HF: MMDocIR/MMDocIR-Challenge                     (원본 MMDocIR, 2.5GB)
HF: <SciEGQA-repo>                                (원본 SciEGQA, 1GB)
       │
       ▼
  setup_data_from_hf.py
       │
       ▼
data/benchmarks/
  ├─ spiqa/{train_val,test-A}/{element_graph_v2.json, elements_v2.jsonl, *.json, images/}
  ├─ sciegqa/{element_graph_v2.json, elements_v2.jsonl, queries.jsonl, PDF/, Images/}
  └─ mmdocir/{element_graph_v2.json, elements_v2.jsonl, queries.jsonl, MMDocIR_layouts.parquet}
       │
       ▼
  bash scripts/run_all_experiments.sh
       │
       ▼
  ckpt/, eval/results/experiments/, SUMMARY.md
```
