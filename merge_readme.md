# SFT LoRA Merge 실행 가이드

이 문서는 환경 세팅부터 SFT LoRA 어댑터를 백본 모델에 merge하는 전 과정을 기록합니다.

현재 SFT는 두 가지 모드를 지원합니다.

- `length`: Direct Answer / CoT / Long CoT supervision
- `perspective`: Abstract / Temporal / Spatio-temporal reasoning perspective supervision

둘 다 merge 절차는 동일하고, 단지 adapter/output 경로만 다릅니다.

---

## 전제 조건

| 항목 | 경로 |
|------|------|
| SFT LoRA 어댑터 | `sft/outputs/qwen25vl7b_lora_sft_length/` 또는 `sft/outputs/qwen25vl7b_lora_sft_perspective/` |
| 백본 모델 | `Qwen/Qwen2.5-VL-7B-Instruct` (HuggingFace에서 자동 다운로드) |
| merge 결과 | `sft/outputs/qwen25vl7b_lora_merged_length/` 또는 `sft/outputs/qwen25vl7b_lora_merged_perspective/` |

---

## 1단계: 가상환경 생성 및 패키지 설치

```bash
cd /workspace/GRPO_Video

# 기존 venv가 있으면 재생성 (충돌 방지)
rm -rf .venv_realign
python3.11 -m venv .venv_realign
source .venv_realign/bin/activate

# pip 업그레이드
python -m pip install -U pip setuptools wheel

# 레포 전체 패키지 설치 (torch, vllm, deepspeed, flash-attn 등)
bash setup.sh
```

> `setup.sh`는 `src/r1-v`를 editable로 설치하고 `vllm==0.7.2`, `deepspeed==0.15.4`, `trl==0.14.0` 등을 설치합니다.
> `/workspace` 디렉터리에 venv를 만들어야 디스크 공간 부족(`/usr/local` 파티션) 문제를 피할 수 있습니다.

---

## 2단계: 환경 체크

```bash
cd /workspace/GRPO_Video
bash src/scripts/check_environment.sh
```

모든 항목이 ✅이어야 합니다. 특히:
- `torch (2.5.1+cu124)` ✅
- `vllm (0.7.2)` ✅
- `deepspeed (0.15.4)` ✅

---

## 3단계: flash-attn 재설치 (중요)

`setup.sh`가 `flash-attn`을 설치한 뒤 `vllm==0.7.2`가 torch를 2.4.x → 2.5.1로 업그레이드하면서 **flash-attn .so 파일과 torch 버전 불일치**가 발생합니다.

```
ImportError: flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol
```

이 에러가 발생하면 아래로 해결합니다:

```bash
# flash-attn 제거 (merge 스크립트는 flash-attn 불필요)
pip uninstall flash-attn -y

# merge 완료 후 GRPO 훈련 전에 현재 torch(2.5.1) 기준으로 재설치
pip install flash-attn --no-build-isolation --no-binary :all:
```

> merge 스크립트 자체는 flash-attn이 없어도 동작합니다.
> flash-attn은 GRPO 훈련 전에 재설치하면 됩니다.

---

## 4단계: merge 설정 파일 확인

예를 들어 length SFT adapter를 merge하려면 `sft/configs/merge_lora_qwen25vl7b_length.yaml` 또는 그에 준하는 설정에서 아래 세 값을 맞춘다.

```yaml
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
adapter_name_or_path: ./outputs/qwen25vl7b_lora_sft_length
export_dir: ./outputs/qwen25vl7b_lora_merged_length
remap_adapter_keys: true
```

perspective SFT를 merge할 때는 대응 경로만 바꾸면 된다.

```yaml
adapter_name_or_path: ./outputs/qwen25vl7b_lora_sft_perspective
export_dir: ./outputs/qwen25vl7b_lora_merged_perspective
```

> **`remap_adapter_keys: true` 필수.**
> `false`로 설정하면 LoRA 가중치 키 이름 불일치(`language_model.layers` vs `layers`)로 인해 LoRA가 적용되지 않고 base 모델과 동일한 결과가 나옵니다.

---

## 5단계: merge 실행

```bash
cd /workspace/GRPO_Video/sft
SFT_MODE=length bash scripts/run_merge.sh
```

원하는 adapter가 perspective라면 `SFT_MODE=perspective`로 바꾸면 된다.

정상 진행 시 출력:
```
[SFT-MERGE] config: configs/merge_lora_qwen25vl7b_length.yaml
Loading checkpoint shards: 100%|███████| 2/2 [...]
# (경고/로그 없이) 프롬프트 복귀
```

> `Found missing adapter keys` UserWarning이 뜨면 `remap_adapter_keys: false` 상태입니다.
> 반드시 `true`로 설정 후 재실행하세요.

---

## 6단계: merge 결과 검증

```bash
ls -lh /workspace/GRPO_Video/sft/outputs/qwen25vl7b_lora_merged_length/
```

정상 결과:

| 파일 | 크기 |
|------|------|
| `model-00001-of-00002.safetensors` | ~4.7G |
| `model-00002-of-00002.safetensors` | ~2.4G |
| `model.safetensors.index.json` | 64K |
| `tokenizer.json` | 11M |
| `preprocessor_config.json` | 존재 |
| 총합 | **~7.1G** |

총합이 약 7GB이고 tokenizer/processor 파일이 모두 있으면 정상입니다.

---

## 다음 단계: GRPO 학습

merge 완료 후 GRPO 학습으로 이어집니다. 상세 내용은 `src/scripts/RUN_GRPO.md` 참조.

```bash
# flash-attn 재설치 (GRPO 훈련에 필요)
pip install flash-attn --no-build-isolation --no-binary :all:

# GRPO 실행 (레포 루트에서)
cd /home/users/ntu/n2500182/workspace_JWP/repos/GRPO_Video_2
source /home/users/ntu/n2500182/scratch/.venv_realign/bin/activate
export QWEN_PATH="/scratch/users/ntu/n2500182/models/qwen25vl7b_lora_merged_length"
export QWEN_BASE_PATH="/scratch/users/ntu/n2500182/models/Qwen2.5-VL-7B-Instruct"
export NUM_GPUS=3
export TRAIN_NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1,2
bash src/scripts/run_grpo_answer_only_lora.sh
```

`QWEN_PATH`는 병합 산출물 디렉터리. 이 계정에서는 위 scratch 경로 또는 `sft/outputs/qwen25vl7b_lora_merged_length` / `..._perspective` (레포 상대).

---

## 트러블슈팅 요약

| 에러 | 원인 | 해결 |
|------|------|------|
| `No space left on device` | venv 없이 `/usr/local`에 설치 시도 | `/workspace` 아래 venv 생성 후 재설치 |
| `flash_attn undefined symbol` | torch 버전 업그레이드 후 flash-attn 불일치 | `pip uninstall flash-attn -y` 후 재컴파일 |
| `Can't find adapter_config.json` | merge config의 `adapter_name_or_path` 경로 오타 | yaml에서 실제 어댑터 경로로 수정 |
| `Found missing adapter keys` | `remap_adapter_keys: false` | yaml에서 `true`로 변경 후 재실행 |
| merge 결과 ~2.9G (파일 1개) | 이전 merge가 중간에 중단된 잔여물 | `rm -rf` 후 재실행 |
