# Qwen2.5-VL-3B SFT Pipeline

이 폴더는 `Qwen/Qwen2.5-VL-3B-Instruct`를 LoRA SFT 하기 위한 파이프라인입니다. 기존 `instruction/input/output` 텍스트 JSON뿐 아니라, 이미지/프레임이 포함된 JSON/JSONL도 처리할 수 있습니다.

현재 SFT 목표는 두 가지를 지원합니다.

- `length`: Direct Answer / CoT / Long CoT 길이 supervision
- `perspective`: Abstract / Temporal / Spatio-temporal 추론 관점 선택 + reasoning supervision

## Folder structure

- `data/`: SFT 데이터셋(JSON)
- `configs/train_lora_qwen25vl3b_length.yaml`: Dataset 1 학습 설정
- `configs/train_lora_qwen25vl3b_perspective.yaml`: Dataset 2 학습 설정
- `configs/merge_lora_qwen25vl3b.yaml`: LoRA 병합 설정
- `scripts/train_sft.py`: 학습 스크립트
- `scripts/merge_lora.py`: LoRA 병합 스크립트
- `scripts/run_pipeline.sh`: SFT -> merge 연속 실행 스크립트

## 1) Environment

```bash
cd sft
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Train (LoRA SFT)

```bash
cd sft
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_sft.py --config configs/train_lora_qwen25vl3b_length.yaml --use-vision true
# or
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh
```

`run_train.sh` / `run_pipeline.sh`는 `SFT_MODE`로 기본 config를 고를 수 있습니다.

```bash
# Dataset 1: reasoning length
cd sft
SFT_MODE=length USE_VISION=true CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh

# Dataset 2: reasoning perspective
cd sft
SFT_MODE=perspective USE_VISION=true CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh
```

체크포인트/어댑터는 모드에 따라 아래에 저장됩니다.

- `outputs/qwen25vl3b_lora_sft_length`
- `outputs/qwen25vl3b_lora_sft_perspective`

`bash scripts/run_train.sh`는 실행 시 이미지/프레임 입력 사용 여부를 직접 묻습니다.

## 3) Merge LoRA adapter

```bash
cd sft
python scripts/merge_lora.py --config configs/merge_lora_qwen25vl3b.yaml
# or
bash scripts/run_merge.sh
```

병합 모델도 모드에 따라 아래에 저장됩니다.

- `outputs/qwen25vl3b_lora_merged_length`
- `outputs/qwen25vl3b_lora_merged_perspective`

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
- Qwen2.5-VL-3B 로딩을 위해 `transformers>=4.51` 기준으로 작성했습니다.
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

가능합니다. `Qwen2.5-VL-3B`도 언어 디코더를 포함하므로, 텍스트 샘플만으로 SFT를 진행할 수 있습니다.

다만 이 경우 모델은 주로 텍스트 추론 스타일에 맞춰지고, 시각 태스크 성능은 별도로 좋아지지 않습니다.  
시각 성능까지 유지/개선하려면 이후에 이미지 포함 샘플을 섞어서 추가 SFT를 진행하는 것이 안전합니다.
