# Qwen2.5-VL LoRA SFT Pipeline

이 폴더는 **Qwen2.5-VL** 계열을 LoRA SFT 하기 위한 파이프라인입니다. 기본 학습 YAML(`configs/train_lora_qwen25vl3b_*.yaml`)은 **`Qwen/Qwen2.5-VL-7B-Instruct`** 와 `outputs/qwen25vl7b_*` 경로를 쓰며, 파일명의 `3b`는 과거 명명입니다. 기존 `instruction/input/output` 텍스트 JSON뿐 아니라, 이미지/프레임이 포함된 JSON/JSONL도 처리할 수 있습니다.

현재 SFT 목표는 두 가지를 지원합니다.

- `length`: Direct Answer / CoT / Long CoT 길이 supervision
- `perspective`: Abstract / Temporal / Spatio-temporal 추론 관점 선택 + reasoning supervision

## Folder structure

- `data/`: SFT 데이터셋(JSON/JSONL)
- `requirements.txt` / `install_requirements.sh`: SFT 전용 파이썬 의존성 (GRPO `setup.sh`와 별도)
- `configs/train_lora_qwen25vl3b_length.yaml`: length 학습 설정 (`train_files`, `output_dir` 등)
- `configs/train_lora_qwen25vl3b_perspective.yaml`: perspective 학습 설정
- `configs/merge_lora_qwen25vl3b_*.yaml`: LoRA 병합 설정
- `scripts/train_sft.py`: 학습 본체
- `scripts/run_train.sh`: GPU 개수 자동 감지, DDP, `SFT_MODE`, `RESUME_FROM_CHECKPOINT` 처리
- `scripts/merge_lora.py`, `scripts/run_merge.sh`, `scripts/run_pipeline.sh`: 병합 / 파이프라인

## 1) Environment

**GRPO와 SFT는 서로 다른 venv를 씁니다.** 같은 디렉터리에 `pip`로 섞이면 안 됩니다.

| 목적 | 한 번 설치 | 매 세션 활성화 | 기본 venv 경로 |
|------|------------|----------------|----------------|
| **SFT** | 레포 루트에서 `bash scripts/run_setup_sft.sh` | `source scripts/hpc_activate_sft.sh` | `~/scratch/.venv_sft` |
| **GRPO** | `bash scripts/run_setup_realign.sh` | `source scripts/hpc_activate_realign.sh` | `~/scratch/.venv_realign` |

- `setup.sh`는 **GRPO 전용**입니다. SFT 설치에는 **쓰지 않습니다.**
- `run_setup_sft.sh`: 모듈 로드 → venv 생성/활성화 → torch cu124 → `sft/install_requirements.sh` (requirements + 혹시 남은 `deepspeed`/`r1-v` 제거) → (선택) flash-attn.
- 로그인 노드에는 보통 `nvcc`가 없어 **flash-attn 기본 생략**. GPU 노드에서만: `module load cuda/12.2.2 && INSTALL_FLASH_ATTN=true bash scripts/run_setup_sft.sh`
- HPC에서는 **`hpc_activate_sft.sh`로만** venv를 켜세요. 모듈이 없는 셸에서 `~/.venv_sft/bin/activate`만 하면 `libpython` 오류가 날 수 있습니다.

로컬:

```bash
cd sft
python -m venv .venv
source .venv/bin/activate
bash install_requirements.sh   # torch는 별도 설치 후 이 스크립트만 써도 됨
pip check
```

HPC — 최초 1회 (레포 **루트**에서):

```bash
cd /path/to/GRPO_Video_2
bash scripts/run_setup_sft.sh
```

이미 GRPO용 패키지가 섞였다면, 활성화한 뒤:

```bash
cd sft && bash install_requirements.sh
pip check
```

학습/병합 **매 세션** (계산 노드 권장):

```bash
module load cuda/12.2.2
source /path/to/GRPO_Video_2/scripts/hpc_activate_sft.sh
cd /path/to/GRPO_Video_2/sft
```

선택: Hub 속도·제한 — `export HF_TOKEN=...` 또는 `huggingface-cli login`

## 2) Train (LoRA SFT)

작업 디렉터리는 항상 **`sft/`** 기준입니다.

### 기본 데이터·산출물 (YAML 그대로)

| `SFT_MODE` | 기본 config | `train_files` (상대 경로) | `output_dir` |
|------------|-------------|---------------------------|--------------|
| `length` | `configs/train_lora_qwen25vl3b_length.yaml` | `./data/video_r1_length_sft_from_filtered.jsonl` | `./outputs/qwen25vl7b_lora_sft_length` |
| `perspective` | `configs/train_lora_qwen25vl3b_perspective.yaml` | `./data/video_r1_perspective_sft.jsonl` | `./outputs/qwen25vl7b_lora_sft_perspective` |

다른 데이터나 모델을 쓰려면 해당 yaml의 `train_files` / `model_name_or_path` 를 바꾸거나 `CONFIG_PATH=configs/...` 로 지정하세요.

### `run_train.sh` 동작

- `nvidia-smi`로 **보이는 GPU 개수**를 세어 `torch.distributed.run` (다중 GPU) 또는 단일 프로세스로 실행합니다.
- `CUDA_VISIBLE_DEVICES`를 안 주면 `0,1,...,N-1` 을 씁니다.
- 로그를 파일로 볼 때는 스크립트 안의 **`PYTHONUNBUFFERED=1`** 덕분에 버퍼링이 줄어듭니다.
- PEFT unwrap 시 **DeepSpeed가 설치돼 있으면** import 단계에서 `CUDA_HOME`이 필요할 수 있습니다. SFT venv에서는 `install_requirements.sh`로 `deepspeed`를 제거하는 것을 권장합니다. 그래도 필요하면 `module load cuda/12.2.2` 후 `nvcc` 기준으로 `run_train.sh`가 `CUDA_HOME`을 잡습니다.

### 포그라운드 예시

```bash
cd sft
SFT_MODE=length USE_VISION=true bash scripts/run_train.sh
```

```bash
# perspective
SFT_MODE=perspective USE_VISION=true bash scripts/run_train.sh
```

직접 `train_sft.py`만 호출할 때:

```bash
cd sft
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_sft.py \
  --config configs/train_lora_qwen25vl3b_length.yaml --use-vision true
```

### HPC: 처음부터 새로 학습 (nohup)

기존 `output_dir`에 `checkpoint-*`가 있으면 꼬일 수 있으니 **백업 후** 시작합니다.

```bash
module load cuda/12.2.2
source /path/to/GRPO_Video_2/scripts/hpc_activate_sft.sh
cd /path/to/GRPO_Video_2/sft
unset RESUME_FROM_CHECKPOINT

OUT=./outputs/qwen25vl7b_lora_sft_length   # perspective면 qwen25vl7b_lora_sft_perspective
[[ -d "$OUT" ]] && mv "$OUT" "${OUT}.bak.$(date +%Y%m%d_%H%M%S)"

nohup env SFT_MODE=length USE_VISION=true bash scripts/run_train.sh > sft_train_nohup.log 2>&1 &
echo "PID=$!  log=$(pwd)/sft_train_nohup.log"
tail -f sft_train_nohup.log
```

PBS 대화형 잡(`qsub -I`)이 살아 있는 동안은 SSH를 끊어도 `nohup` 프로세스는 보통 계속됩니다.

### 체크포인트에서 이어서

```bash
export RESUME_FROM_CHECKPOINT=/path/to/GRPO_Video_2/sft/outputs/qwen25vl7b_lora_sft_length/checkpoint-NNN
SFT_MODE=length USE_VISION=true bash scripts/run_train.sh
```

raw annotation JSONL에서 최종 SFT JSONL을 만드는 변환기도 포함되어 있습니다. 현재 생성 중인 raw 샘플 형식이 아래와 같다면:

- Dataset 1 raw: `question / options / gold_answer / frame_subdir / answer_raw / cot_raw / long_cot_raw`
- Dataset 2 raw: `question / options / gold_answer / frame_subdir / granularity_type / granularity_thinking_raw`

다음처럼 바로 최종 학습 파일로 바꿀 수 있습니다.

```bash
cd sft
python scripts/prepare_sft_dataset.py \
  --mode length \
  --input data/generated_length_0_500.jsonl \
  --output data/video_r1_length_sft.jsonl

python scripts/prepare_sft_dataset.py \
  --mode perspective \
  --input data/generated_granulity_0_1000.jsonl \
  --output data/video_r1_perspective_sft.jsonl
```

기본적으로 `--input` 파일이 있는 폴더의 `frames/`를 찾아서 `frame_subdir`를 실제 `frames/train/...` 경로로 풀어줍니다. 따라서 `sft/data/frames/` 아래에 이미지가 정리돼 있으면 그대로 사용할 수 있습니다.

체크포인트/어댑터는 YAML의 `output_dir`에 저장됩니다 (현재 기본값):

- `outputs/qwen25vl7b_lora_sft_length`
- `outputs/qwen25vl7b_lora_sft_perspective`

`run_train.sh` / `run_pipeline.sh` 기본은 `USE_VISION=true` 입니다.

## 3) Merge LoRA adapter

```bash
cd sft
python scripts/merge_lora.py --config configs/merge_lora_qwen25vl3b_length.yaml
# or
SFT_MODE=length bash scripts/run_merge.sh
```

병합 모델도 yaml 기준 (현재 기본값):

- `outputs/qwen25vl7b_lora_merged_length`
- `outputs/qwen25vl7b_lora_merged_perspective`

`run_merge.sh`는 기본적으로:

- `MERGE_STAGE=sft` + `SFT_MODE=length` -> `configs/merge_lora_qwen25vl3b_length.yaml`
- `MERGE_STAGE=sft` + `SFT_MODE=perspective` -> `configs/merge_lora_qwen25vl3b_perspective.yaml`
- `MERGE_STAGE=grpo` + `SFT_MODE=length` -> `configs/merge_lora_grpo_length.yaml`
- `MERGE_STAGE=grpo` + `SFT_MODE=perspective` -> `configs/merge_lora_grpo_perspective.yaml`

## 4) Train + Merge 한 번에 실행

```bash
cd sft
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_pipeline.sh
```

기본적으로 SFT 이후 바로 merge까지 이어집니다.

## Notes

- 데이터 포맷은 아래 둘을 모두 지원합니다.
  - 텍스트 SFT: `instruction/input/output` JSON list
  - 멀티모달 SFT: `problem/solution` + `frames` 또는 `image`/`images` 필드가 있는 JSON/JSONL
- 기본 스택은 `sft/requirements.txt` (`transformers>=4.51,<6` 등) 기준입니다.
- `use_vision: true` 또는 `--use-vision true`일 때는 이미지/프레임을 실제로 로드해서 SFT에 사용합니다.
- `lora_target_modules: auto`일 때는:
  - 텍스트 모드: 언어 모듈만 LoRA 적용
  - 비전 모드: 언어 + 비전 선형 모듈 전체에 LoRA 적용
- 출력 태그 기반 학습을 지원합니다: `answer`, `cot`, `long_cot`.
- `CODE` CoT는 기본값으로 학습에서 제외됩니다 (`drop_code_cot: true`).
- length 모드에서는 `configs/train_lora_qwen25vl3b_length.yaml`에서 아래를 조정해 형식을 선택할 수 있습니다.
- 기본 LoRA target은 언어 모듈(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)로 고정되어 있습니다.

```yaml
reasoning_formats: [answer, cot, long_cot]
format_mix_strategy: expand
append_format_instruction: true
drop_code_cot: true
```

- 형식별 동작
  - `answer`: `<ANSWER>...</ANSWER>`만 학습
  - `cot`: `<COT>...</COT> + <ANSWER>...</ANSWER>` 학습
  - `long_cot`: `<LONG_COT>...</LONG_COT> + <ANSWER>...</ANSWER>` 학습

- `sft_mode: perspective`일 때는 아래 형식을 학습합니다.

```xml
<REASONING_TYPE>
TEMPORAL
</REASONING_TYPE>
<REASONING>
...
</REASONING>
<ANSWER>
A
</ANSWER>
```

- perspective 모드에서는 모델이 입력 `(V, q, O)`만 보고 `REASONING_TYPE -> REASONING -> ANSWER`를 순서대로 생성하도록 학습합니다.

## VL 모델에 텍스트 SFT를 해도 되는가?

가능합니다. Qwen2.5-VL도 언어 디코더를 포함하므로, 텍스트 샘플만으로 SFT를 진행할 수 있습니다.

다만 이 경우 모델은 주로 텍스트 추론 스타일에 맞춰지고, 시각 태스크 성능은 별도로 좋아지지 않습니다.  
시각 성능까지 유지/개선하려면 이후에 이미지 포함 샘플을 섞어서 추가 SFT를 진행하는 것이 안전합니다.
