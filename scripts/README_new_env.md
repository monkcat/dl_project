# 새 환경에서 실험 실행 (A100 × 2)

전체 실험을 한 줄 명령으로 돌리기 위한 셋업 가이드.

## 0. 사전 확인

GPU 서버:
- A100 80GB × 2장
- Python 3.10+ + CUDA 12.x 또는 11.8
- 디스크: 데이터 ~38GB + 모델 ~12GB + 결과/체크포인트 ~30GB = **약 80GB 권장**

## 1. 코드 클론 + 환경 구축

```bash
git clone https://github.com/monkcat/dl_project.git
cd dl_project

# 옵션 A: venv (가벼움, conda 없는 환경 추천)
python3.10 -m venv .venv && source .venv/bin/activate

# 옵션 B: conda
conda create -n dl_hw2 python=3.10 -y && conda activate dl_hw2
```

PyTorch + 나머지 의존성:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. HuggingFace 로그인

```bash
huggingface-cli login    # read-only 토큰이면 충분
```

> 공개 dataset도 rate limit 회피 + 모델 캐시 인증 일관성 차원에서 로그인 권장.

## 3. 데이터 + 모델 받기

**시나리오 A — GPU 서버가 HF를 정상 접근할 수 있는 경우** (대부분):

```bash
python scripts/setup_data_from_hf.py --graph_repo ljh38/element-graph-v2.1
```

자동으로:
- `ljh38/element-graph-v2.1` (v2.1 그래프 metadata, ~2.4 GB)
- `google/spiqa` (원본 SPIQA train + test-A images, ~33 GB)
- `Yuwh07/SciEGQA-Bench` (PDF.tar + Images.tar, ~1.3 GB)
- `MMDocIR/MMDocIR-Challenge` (parquet + annotations, ~2.5 GB)

→ `data/benchmarks/{spiqa,sciegqa,mmdocir}/` 에 trainer가 기대하는 layout으로 정리됨.

3개 모델은 첫 실행 시 자동 캐시됨:
- `google/siglip2-base-patch16-224`
- `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`
- `openai/clip-vit-large-patch14`

**시나리오 B — GPU 서버 HF 접근 차단** (DPI / firewall):

본인 노트북에서 받아서 옮기는 방식:

```bash
# 노트북에서 (전체 패키지 ~48GB 생성)
python scripts/download_all_to_local.py
# → ./dl_pack/data/benchmarks/ + ./dl_pack/hf_home/ 생성

# Backend.AI 파일 브라우저 / scp / rsync로 ./dl_pack/ 통째로 서버에 업로드

# 서버에서
mv ~/dl_pack/data/benchmarks ~/dl_project/data/
cat >> ~/.bashrc <<'EOF'
export HF_HOME=~/dl_pack/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EOF
source ~/.bashrc
```

학습 이미지(31GB) 빼고 평가만 하려면:
```bash
python scripts/download_all_to_local.py --skip_spiqa_train   # ~17GB
```

## 4. 데이터 검증 (preflight)

```bash
python -m pipeline.sanity_check
```

이게 통과해야 학습 시작 가능. 검증 항목:
1. 모든 모듈 import OK
2. 모든 row config 필수 필드 존재 (43 rows)
3. `EVAL_DATASETS` 의 모든 file path 존재
4. SPIQA train/test-A 코퍼스 존재 + 이미지 디렉토리 도달 가능
5. 그래프 schema sample 통과 (node type / section role mapping)
6. `graph_propagate` 3 method × 5 weight mode 동작
7. GPU 2장 가시화

예상 출력:
```
───── summary ─────
  PASS: 36   WARN: 0   FAIL: 0
[OK] all checks passed — suite is ready to launch
```

`FAIL`이 있으면 그 항목 먼저 해결. `WARN`만 있으면 진행 가능.

## 5. 전체 실험 실행 — 한 줄

```bash
bash scripts/run_full_suite.sh
```

이게 자동으로:
- preflight `sanity_check` 실행
- **43 rows** (42 학습 + 1 zero-shot) × **22 wave** 에 걸쳐 2 GPU 병렬 학습
- (h) 체크포인트 위에 **21-variant 전파 sub-ablation** (Group A uniform / B weighted / C PPR)
- 모든 결과 → `eval/results/experiments/SUMMARY.md`

총 wall time ~**4-5일** (2× A100 기준).

### 단계별 실행

```bash
bash scripts/run_full_suite.sh --tier 1                  # 9 rows  (~24h)
bash scripts/run_full_suite.sh --tier 2                  # 4 rows  (~12h)
bash scripts/run_full_suite.sh --tier 3                  # 25 rows (~60h)
bash scripts/run_full_suite.sh --tier 4                  # 5 rows  (~14h, h_best_combo_16k 1배 추가)
bash scripts/run_full_suite.sh --tier 1,2                # Tier 1+2 만

bash scripts/run_full_suite.sh --rows a,h,gme            # 특정 rows 만
bash scripts/run_full_suite.sh --dry_run                 # 스케줄 미리보기
bash scripts/run_full_suite.sh --no_prop_sweep           # 전파 sub-ablation 스킵
```

### Resume 동작

각 row 완료 시 `eval/results/experiments/<rid>_<name>/summary.json` 작성됨.
launcher 재실행하면 **이미 끝난 row는 자동 skip**. Ctrl-C 후 재실행 안전.

## 6. 결과 확인

```bash
# 전체 요약
cat eval/results/experiments/SUMMARY.md

# 개별 row 상세
ls eval/results/experiments/h_full_method/
# → train.json (학습 곡선), eval.json (metric), summary.json (config 메타)

python -m json.tool eval/results/experiments/h_full_method/train.json | head -50
```

`SUMMARY.md`의 6 섹션:
- §7.1 Tier 1 main ablation
- §7.2 Tier 2 negative controls
- §7.3 Tier 3 sub-ablation sweeps (γ / λ_cov / λ_cons / LoRA / lr / τ / anchor / edge / tokens)
- §7.4 Tier 4 follow-up extensions
- §7.5 Propagation sub-ablation (Group A/B/C tables)
- §7.6 Quick comparison (R@10 no-prop 모든 row × dataset)

---

## Troubleshooting

### GPU OOM
`pipeline/configs.py`의 `COMMON["batch"]` 16 → 8 또는 4로 줄이기.

### HF rate limit / 401
`huggingface-cli login` 재실행. 또는 시나리오 B로 전환.

### 학습 도중 nvmlShutdown segfault (Backend.AI)
`run_experiment.py`가 자동으로 처리함 — `train.json`에 `"final"` 키 있고
ckpt 존재하면 성공으로 처리. 무시해도 됨.

### 한 row 실패 → 전체 abort?
**No.** launcher가 wave 단위로 wait + 실패 카운트만 로깅 후 계속. 실패한 row만 재실행:
```bash
bash scripts/run_full_suite.sh --rows <failed_row>
```

### `ModuleNotFoundError: pipeline`
프로젝트 루트(`dl_project/`)에서 실행해야 함:
```bash
cd /path/to/dl_project
bash scripts/run_full_suite.sh
```

### conda 없는 환경에서 `python` 명령이 없음
- venv 활성화: `source .venv/bin/activate`
- 또는 `python3 -m pipeline.sanity_check` 처럼 명시
- launcher는 conda 있으면 자동 활성화, 없으면 활성화된 python 그대로 사용

### Mirror repo로 한 번에 받기
```bash
# 노트북에서 (한 번만)
python -m pipeline.upload_full_mirror --repo ljh38/dl-project-mirror

# 서버에서 한 줄로 받기
bash scripts/setup_from_mirror.sh
```

---

## 데이터 흐름 요약

```
HF: ljh38/element-graph-v2.1            (v2.1 그래프 metadata,  ~2.4 GB)
HF: google/spiqa                         (SPIQA 원본 이미지,     ~33 GB)
HF: Yuwh07/SciEGQA-Bench                 (SciEGQA 원본,         ~1.3 GB)
HF: MMDocIR/MMDocIR-Challenge            (MMDocIR layout,       ~2.5 GB)
HF: google/siglip2-base-patch16-224      (메인 encoder,         ~1.5 GB)
HF: Alibaba-NLP/gme-Qwen2-VL-2B-Instruct (zero-shot reference,  ~8.3 GB)
HF: openai/clip-vit-large-patch14        ((p) encoder swap,     ~1.6 GB)
       │
       ▼
  scripts/setup_data_from_hf.py    (시나리오 A — 서버에서 직접 받기)
       또는
  scripts/download_all_to_local.py + Backend.AI 파일 브라우저  (시나리오 B)
       │
       ▼
data/benchmarks/                    ← trainer가 읽는 경로
  ├─ spiqa/{train_val, test-A}/
  ├─ sciegqa/
  └─ mmdocir/
       │
       ▼
  bash scripts/run_full_suite.sh    (43 rows × 22 waves, ~4-5일)
       │
       ▼
ckpt/                                ← 체크포인트
eval/results/experiments/<rid>/      ← row별 train/eval/summary.json
eval/results/experiments/SUMMARY.md  ← 최종 비교 표 (§7.1-§7.6)
```
