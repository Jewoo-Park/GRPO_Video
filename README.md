# GRPO_Video_2

`Qwen2.5-VL-7B-Instruct`를 백본으로, **SFT(LoRA) → SFT merge → GRPO(LoRA) → GRPO merge → 벤치마크 추론** 순서의 연구 파이프라인을 담은 레포입니다.  
GRPO는 `Video-R1` 계열 데이터를 공통 JSONL 포맷으로 두고, `open_r1` 기반 비디오 GRPO(`grpo_video`)와 vLLM 롤아웃을 사용합니다.

YAML·스크립트 파일명에 남아 있는 `3b` 표기는 과거 명명이며, **실제 기본 설정은 7B 모델**입니다.

---

## 1. 전체 구조와 데이터 흐름

### 1-1. 단계별 파이프라인

| 단계 | 입력 | 출력 | 비고 |
|------|------|------|------|
| **SFT** | Dataset 1 (length) 또는 Dataset 2 (perspective) JSONL + 프레임 | LoRA 어댑터 (`sft/outputs/qwen25vl7b_lora_sft_*`) | `sft/scripts/run_train.sh` 등 |
| **SFT merge** | 백본 HF 가중치 + SFT LoRA | merge된 전체 가중치 (`…/qwen25vl7b_lora_merged_*`, 대용량은 scratch 권장) | `remap_adapter_keys: true` |
| **GRPO** | merge 모델 + Video-R1 GRPO train JSONL | GRPO LoRA (`src/r1-v/outputs/…`) | vLLM + DeepSpeed, LoRA만 저장 |
| **GRPO merge** | SFT merge 모델 + GRPO LoRA | 최종 추론용 merge 모델 | `merge_lora_grpo_*.yaml` |
| **테스트** | 최종(또는 중간) 모델 + 벤치 JSONL | `test_predictions.jsonl` 등 | UVB / VideoMMMU / MMVU |

### 1-2. 디렉터리 개요

```text
GRPO_Video_2/
├── data/                    # 보통 scratch로 심볼릭 링크 (아래 §2)
├── sft/                     # LoRA SFT, merge, SFT 데이터·YAML
├── src/
│   ├── r1-v/                # GRPO 학습 패키지 (open_r1)
│   ├── eval/                # 데이터 준비·평가 보조 스크립트
│   └── scripts/             # run_grpo_answer_only_lora.sh, check_environment.sh 등
├── scripts/                 # HPC용 venv 설치·활성화 (run_setup_sft.sh, run_setup_grpo.sh …)
├── setup.sh                 # GRPO 전용 의존성 설치 (SFT와 분리)
├── merge_readme.md, QUICKSTART.md, sft/README.md
└── README.md                # 본 문서
```

### 1-3. 공통 GRPO JSONL 스키마

학습·평가용 데이터는 아래 키를 중심으로 맞춥니다.

- `video_id`, `question_id`, `question_category`
- `problem` (질문·선지 텍스트)
- `frames` (프레임 이미지 경로 리스트)
- `solution` (정답·태그, 예: `<ANSWER>B</ANSWER>`)

동일 스키마로 **Video-R1 train**, **Urban Video Bench / VideoMMMU / MMVU** 테스트를 처리합니다.

---

## 2. 데이터 위치 (scratch · `data/` 링크)

용량이 큰 JSONL·프레임·비디오는 **home/workspace 쿼터를 피하기 위해 scratch**에 두는 구성을 권장합니다.

이 체크아웃에서는 `GRPO_Video_2/data`가 예시로 다음에 연결될 수 있습니다.

- `…/GRPO_Video_2/data` → `…/scratch/GRPO_Video_2_data/data`

실제 경로는 `readlink -f data`로 확인하세요.

### 2-1. SFT용 (Dataset 1 · Dataset 2)

| 목적 | 내용 | 비고 |
|------|------|------|
| **Dataset 1 (length)** | `<ANSWER>`, `<COT>`, `<LONG_COT>` 등 길이·형식 supervision | raw 예: `question`, `options`, `gold_answer`, `frame_subdir`, `answer_raw`, `cot_raw`, `long_cot_raw` |
| **Dataset 2 (perspective)** | `<REASONING_TYPE>`, `<REASONING>`, `<ANSWER>` (추론 관점) | raw 예: `granularity_type`, `granularity_thinking_raw` 등 |

raw JSONL을 학습용으로 바꿀 때:

```bash
cd sft
python scripts/prepare_sft_dataset.py --mode length   --input data/generated_length.jsonl   --output data/video_r1_length_sft.jsonl
python scripts/prepare_sft_dataset.py --mode perspective --input data/generated_granulity.jsonl --output data/video_r1_perspective_sft.jsonl
```

기본 학습 YAML의 `train_files`는 예를 들어 `video_r1_length_sft_from_filtered.jsonl`, `video_r1_perspective_sft.jsonl` 등을 가리킵니다. **scratch에만 두었다면** 해당 YAML의 경로를 맞추거나 심볼릭 링크를 사용합니다.

### 2-2. GRPO 학습용

- **Train**: `data/video_r1/grpo/video_r1_grpo_train.jsonl` (Video-R1 전처리 결과)

### 2-3. 테스트 세트 (세 종)

레포 기준 상대 경로( scratch `data` 루트와 동일):

- `data/urban_video_bench/grpo/uvb_grpo_test.jsonl`
- `data/video_mmmu/grpo/videommmu_grpo_test.jsonl`
- `data/mmvu/grpo/mmvu_grpo_test.jsonl`

데이터 준비 스크립트는 `sft/data/prepare_/` 등에 복사본이 있을 수 있으며, 원본 파이프라인은 `src/eval/` 및 레포 내 `prepare_*` 스크립트를 참고하면 됩니다.

---

## 3. 환경 분리 원칙

**SFT와 GRPO는 서로 다른 venv를 씁니다.** 한 venv에 `pip`로 섞이면 의존성이 깨지기 쉽습니다.

| 목적 | 최초 1회 설치 | 매 세션 활성화 | 기본 경로 |
|------|----------------|----------------|-----------|
| **SFT** | `bash scripts/run_setup_sft.sh` | `source scripts/hpc_activate_sft.sh` | `$HOME/scratch/.venv_sft` |
| **GRPO** | `bash scripts/run_setup_grpo.sh` | `source scripts/hpc_activate_grpo.sh` | `$HOME/scratch/.venv_grpo` |

- `setup.sh`는 **GRPO용**으로 `run_setup_grpo.sh`에서 호출됩니다. **SFT 설치에 직접 쓰지 않습니다.**
- (구버전) `run_setup_realign.sh` / `hpc_activate_realign.sh`는 동일 계열 실험용 별칭으로 남아 있을 수 있으며, **현재 권장 GRPO 경로는 `_grpo` 스크립트**입니다.
- HPC에서는 **모듈 로드가 포함된** `hpc_activate_*.sh`로 활성화하는 것이 안전합니다. 모듈 없이 `~/.venv_sft/bin/activate`만 하면 `libpython` 오류가 날 수 있습니다.

`pip` 캐시는 기본적으로 `$HOME/scratch/pip_cache`를 쓰도록 맞춰 두었습니다.

---

## 4. SFT 재현

작업 디렉터리는 **`sft/`** 기준입니다.

### 4-1. 가상환경 (venv_sft)

레포 루트에서:

```bash
cd /path/to/GRPO_Video_2
bash scripts/run_setup_sft.sh
```

GPU 노드에서 flash-attn까지 쓰려면 (예):

```bash
module load cuda/12.2.2   # 사이트에 맞게
INSTALL_FLASH_ATTN=true bash scripts/run_setup_sft.sh
```

### 4-2. 활성화

```bash
module load cuda/12.2.2    # 학습 시 권장
source /path/to/GRPO_Video_2/scripts/hpc_activate_sft.sh
cd /path/to/GRPO_Video_2/sft
```

### 4-3. 학습 (Dataset 1 · Dataset 2)

```bash
# Length
SFT_MODE=length USE_VISION=true bash scripts/run_train.sh

# Perspective
SFT_MODE=perspective USE_VISION=true bash scripts/run_train.sh
```

산출물 예:

- `outputs/qwen25vl7b_lora_sft_length` / `outputs/qwen25vl7b_lora_sft_perspective`

### 4-4. 백본 로드 후 SFT LoRA merge

merge는 **다시 백본 `Qwen/Qwen2.5-VL-7B-Instruct`를 로드**하고 어댑터 가중치를 합칩니다.

```bash
cd sft
SFT_MODE=length bash scripts/run_merge.sh
# 또는
python scripts/merge_lora.py --config configs/merge_lora_qwen25vl3b_length.yaml
```

`merge_lora_qwen25vl3b_*.yaml`에서 `export_dir`를 **scratch**로 두면 전체 가중치(~15GB+) 저장 시 쿼터 문제를 줄일 수 있습니다. 예:

- `export_dir: /scratch/users/<USER>/models/qwen25vl7b_lora_merged_length`

**`remap_adapter_keys: true`**는 Qwen2.5-VL에서 저장 키와 백본 키 불일치(`language_model.layers` vs `model.layers` 등)를 줄이기 위해 유지합니다.

---

## 5. GRPO 재현

### 5-1. 시작 모델

- **`QWEN_PATH`**: SFT merge가 끝난 디렉터리 (전체 가중치).
- **`QWEN_BASE_PATH` / `PROCESSOR_PATH`**: merged 트리만으로는 Qwen2.5-VL용 `AutoProcessor` JSON이 깨지는 경우가 있어, **깨끗한 HF 베이스**에서 프로세서만 읽도록 분리합니다.

`src/scripts/run_grpo_answer_only_lora.sh` 기본값 예(계정별로 수정):

```bash
export QWEN_PATH="/scratch/users/<USER>/models/qwen25vl7b_lora_merged_length"
export QWEN_BASE_PATH="/scratch/users/<USER>/models/Qwen2.5-VL-7B-Instruct"
# PROCESSOR_PATH는 기본으로 QWEN_BASE_PATH를 따름
```

플레이스홀더 경로(`/path/to/...`, `...`)는 스크립트가 **실행 전에 거부**합니다.

### 5-2. 가상환경 (venv_grpo)

```bash
cd /path/to/GRPO_Video_2
bash scripts/run_setup_grpo.sh
```

flash-attn은 GPU 아키텍처에 맞게 빌드하려면 `run_setup_grpo.sh` 상단 주석의 `GRPO_CUDA_MODULE`, `TORCH_CUDA_ARCH_LIST` 등을 참고합니다.

### 5-3. 활성화

```bash
export GRPO_CUDA_MODULE="cuda/12.2.2"   # DeepSpeed import용 nvcc 등
source /path/to/GRPO_Video_2/scripts/hpc_activate_grpo.sh
```

`hpc_activate_grpo.sh`는 **venv의 `nvidia-nvjitlink` 휠 경로를 `LD_LIBRARY_PATH` 앞에 붙여** `__nvJitLinkComplete_12_4` 류 오류를 완화합니다.

### 5-4. 데이터 확인

- 학습 소스: 환경 변수 **`TRAIN_FILE`** (기본값은 `data/video_r1/grpo/video_r1_grpo_train.jsonl`을 `TRAIN_SOURCE`로 사용)
- 스크립트는 `split_jsonl_train_eval.py`로 **train/eval 분할 JSONL**을 만들고, **기본**으로는 그 **eval 분할**을 테스트 JSONL로 넘깁니다.
- **UVB / VideoMMMU / MMVU** 등 고정 벤치마크를 쓰려면 **`GRPO_TEST_FILE`**에 해당 JSONL 경로를 지정합니다. (미설정 시 train에서 뽑은 eval 분할을 사용)

```bash
export GRPO_TEST_FILE="$(pwd)/data/urban_video_bench/grpo/uvb_grpo_test.jsonl"
```

### 5-5. GRPO 실행

레포 루트에서:

```bash
cd /path/to/GRPO_Video_2
source scripts/hpc_activate_grpo.sh

export QWEN_PATH="/scratch/users/<USER>/models/qwen25vl7b_lora_merged_length"
export QWEN_BASE_PATH="/scratch/users/<USER>/models/Qwen2.5-VL-7B-Instruct"
export TRAIN_FILE="$(pwd)/data/video_r1/grpo/video_r1_grpo_train.jsonl"
export OUTPUT_DIR="$(pwd)/src/r1-v/outputs/video_r1_uvb_grpo_answer_only_lora"
export NUM_GPUS=2
export TRAIN_NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0,1

bash src/scripts/run_grpo_answer_only_lora.sh
```

로그는 기본으로 `OUTPUT_DIR/training_log.txt`에 `tee`됩니다.

필요 시 이어하기:

```bash
export RESUME_FROM_CHECKPOINT="/path/to/.../checkpoint-500"
bash src/scripts/run_grpo_answer_only_lora.sh
```

### 5-6. 오류 방지용으로 반영해 둔 사항 (요약)

실행 전 점검:

```bash
bash src/scripts/check_environment.sh
# 로그인 노드만: GRPO_CHECK_NO_GPU=1 bash src/scripts/check_environment.sh
```

학습 안정화·환경 이슈 대응:

| 이슈 | 대응 |
|------|------|
| Qwen2.5-VL rotary에서 `float32` vs `bfloat16` assert | 스크립트가 기동 시 `apply_rotary_dtype_hotfix.sh` 실행 (`GRPO_APPLY_ROTARY_DTYPE_HOTFIX=false`로 끔) |
| `DynamicCache` + Flash Attention padding 오류 | trainer 쪽에서 `use_cache=False` 경로 사용 |
| GRPO 메모리 | LoRA 타깃을 `q_proj,k_proj,v_proj,o_proj`로 제한, `flash_attention_2` 기본, `MAX_PIXELS`/`MIN_PIXELS` 조정 |
| 8bit 로드 후 가중치 초기화 오류 | `LOAD_IN_8BIT` 기본 `false` (필요 시에만 켬) |
| NCCL/통신 불안정 | `NCCL_SAFE_MODE=true` 등 환경 변수 (스크립트 주석 참고) |
| DeepSpeed `CUDA_HOME` | `GRPO_CUDA_MODULE` 또는 `nvcc`가 잡히도록 모듈 로드 |

### 5-7. GRPO LoRA를 SFT merge 모델에 merge

`SFT merge 모델`이 베이스이고, 그 위에 학습된 **GRPO LoRA**를 다시 얹습니다.

```bash
cd sft
# 예: length SFT merge → GRPO 출력
python scripts/merge_lora.py --config configs/merge_lora_grpo_length.yaml
```

`merge_lora_grpo_length.yaml` / `merge_lora_grpo_perspective.yaml`에서 `adapter_name_or_path`, `export_dir`를 실제 출력 디렉터리에 맞게 수정합니다.

---

## 6. 테스트 (세 벤치마크 추론)

동일한 GRPO 런처에서 **`GRPO_TEST_FILE`만 바꿔** 세 번 실행하거나, 학습이 끝난 뒤 평가 전용 스크립트(`src/eval/uvb_eval_only.py` 등)로 공통 JSONL을 넣습니다.

예 (Urban Video Bench):

```bash
export GRPO_TEST_FILE="$(pwd)/data/urban_video_bench/grpo/uvb_grpo_test.jsonl"
# QWEN_PATH, OUTPUT_DIR 등은 위와 동일하게
bash src/scripts/run_grpo_answer_only_lora.sh
```

VideoMMMU / MMVU도 각각:

- `data/video_mmmu/grpo/videommmu_grpo_test.jsonl`
- `data/mmvu/grpo/mmvu_grpo_test.jsonl`

학습이 끝나면 `output_dir` 아래에 **`test_predictions.jsonl`**이 생성됩니다(`grpo.py`의 `write_test_predictions_jsonl`).

---

## 7. 권장 실행 순서 (체크리스트)

1. `data`가 scratch를 가리키는지, SFT·GRPO JSONL·프레임이 있는지 확인  
2. **SFT**: `run_setup_sft.sh` → `hpc_activate_sft.sh` → `run_train.sh` (length / perspective)  
3. **SFT merge**: `run_merge.sh` 또는 `merge_lora.py` (필요 시 scratch `export_dir`)  
4. **GRPO**: `run_setup_grpo.sh` → `hpc_activate_grpo.sh` → `run_grpo_answer_only_lora.sh` (`QWEN_PATH`, `QWEN_BASE_PATH`)  
5. **GRPO merge**: `merge_lora_grpo_*.yaml`  
6. **평가**: 세 `*_grpo_test.jsonl`에 대해 추론 실행  

---

## 8. 관련 문서

- [`sft/README.md`](sft/README.md) — SFT·merge 상세  
- [`merge_readme.md`](merge_readme.md) — merge 절차 보조  
- [`QUICKSTART.md`](QUICKSTART.md) — 빠른 참고  
- [`src/eval/README.md`](src/eval/README.md) — eval 데이터 준비(있는 경우)
