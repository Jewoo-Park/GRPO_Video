#!/bin/bash
# Canonical GRPO launcher (LoRA / PEFT always on). Configure via env vars; see RUN_GRPO.md.
# RECOVERED COPY — copy to repos/GRPO_Video_2/src/scripts/run_grpo_answer_only_lora.sh when workspace quota allows.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/src/r1-v"

export DEBUG_MODE="true"
export LOG_PATH="./logs/video_r1_uvb_grpo_answer_only_lora.log"
mkdir -p ./logs

# Qwen2.5-VL vision + flash-attn rotary: assert fp32 q vs bf16 cos/sin. One-time patch of transformers in the active venv.
# Set GRPO_APPLY_ROTARY_DTYPE_HOTFIX=false to skip (e.g. read-only site-packages).
GRPO_APPLY_ROTARY_DTYPE_HOTFIX="${GRPO_APPLY_ROTARY_DTYPE_HOTFIX:-true}"
GRPO_ROTARY_HOTFIX_LC="$(printf '%s' "${GRPO_APPLY_ROTARY_DTYPE_HOTFIX}" | tr '[:upper:]' '[:lower:]')"
if [[ "${GRPO_ROTARY_HOTFIX_LC}" == "true" || "${GRPO_ROTARY_HOTFIX_LC}" == "1" || "${GRPO_ROTARY_HOTFIX_LC}" == "yes" ]]; then
  PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)"
  echo "[VIDEO-GRPO-LORA] Qwen2.5-VL rotary dtype hotfix (active Python: ${PYTHON_BIN})"
  PYTHON_BIN="${PYTHON_BIN}" bash "${REPO_ROOT}/src/scripts/apply_rotary_dtype_hotfix.sh" || \
    echo "[VIDEO-GRPO-LORA] WARNING: rotary hotfix failed or patterns differ; see src/scripts/apply_rotary_dtype_hotfix.sh" >&2
fi

# Merged weights + clean HF tree for AutoProcessor (merged dirs often break Qwen2.5-VL processor JSON).
QWEN_PATH="${QWEN_PATH:-/scratch/users/ntu/n2500182/models/qwen25vl7b_lora_merged_length}"
QWEN_BASE_PATH="${QWEN_BASE_PATH:-/scratch/users/ntu/n2500182/models/Qwen2.5-VL-7B-Instruct}"
export PROCESSOR_PATH="${PROCESSOR_PATH:-${QWEN_BASE_PATH}}"

_grpo_bad_path_placeholder() {
  local v="$1"
  [[ "${v}" == *"/.../"* ]] || [[ "${v}" == "/..."* ]] || [[ "${v}" == *"your_actual_subdir"* ]] || \
    [[ "${v}" == *"/path/to/"* ]] || [[ "${v}" == "/path/to"* ]]
}

if _grpo_bad_path_placeholder "${QWEN_PATH}"; then
  echo "[VIDEO-GRPO-LORA] ERROR: QWEN_PATH is not a real path (remove ... , your_actual_subdir, /path/to/ placeholders)." >&2
  exit 1
fi
if [[ -n "${PROCESSOR_PATH}" ]] && _grpo_bad_path_placeholder "${PROCESSOR_PATH}"; then
  echo "[VIDEO-GRPO-LORA] ERROR: PROCESSOR_PATH / QWEN_BASE_PATH must be real. Unset them or use /scratch/users/ntu/n2500182/models/Qwen2.5-VL-7B-Instruct (got PROCESSOR_PATH=${PROCESSOR_PATH})" >&2
  exit 1
fi
if [[ ! -d "${QWEN_PATH}" ]]; then
  echo "[VIDEO-GRPO-LORA] ERROR: QWEN_PATH is not a directory: ${QWEN_PATH}" >&2
  exit 1
fi
if [[ -n "${PROCESSOR_PATH}" ]] && [[ ! -d "${PROCESSOR_PATH}" ]]; then
  echo "[VIDEO-GRPO-LORA] ERROR: PROCESSOR_PATH is not a directory: ${PROCESSOR_PATH}" >&2
  exit 1
fi
TRAIN_SOURCE="${TRAIN_FILE:-../../data/video_r1/grpo/video_r1_grpo_train.jsonl}"
GRPO_EVAL_FRACTION="${GRPO_EVAL_FRACTION:-0.05}"
GRPO_EVAL_SPLIT_SEED="${GRPO_EVAL_SPLIT_SEED:-42}"
GRPO_EVAL_VIDEO_ONLY="${GRPO_EVAL_VIDEO_ONLY:-true}"

SPLIT_DIR="${SPLIT_DIR:-../../data/video_r1/grpo/splits}"
_EVAL_FRAC_TAG="${GRPO_EVAL_FRACTION/./p}"
SPLIT_TAG="${SPLIT_TAG:-eval${_EVAL_FRAC_TAG}_seed${GRPO_EVAL_SPLIT_SEED}}"
TRAIN_SPLIT_FILE="${TRAIN_SPLIT_FILE:-${SPLIT_DIR}/video_r1_grpo_train__train_${SPLIT_TAG}.jsonl}"
EVAL_SPLIT_FILE="${EVAL_SPLIT_FILE:-${SPLIT_DIR}/video_r1_grpo_train__eval_${SPLIT_TAG}.jsonl}"
# Optional: set GRPO_TEST_FILE to a fixed benchmark JSONL (UVB / VideoMMMU / MMVU). If unset, use eval split.
TEST_FILE="${GRPO_TEST_FILE:-${EVAL_SPLIT_FILE}}"

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/video_r1_uvb_grpo_answer_only_lora}"
RUN_NAME="${RUN_NAME:-video_r1_uvb_grpo_answer_only_lora_$(date +%Y%m%d_%H%M%S)}"
DS_CONFIG="${DS_CONFIG:-./configs/zero1_no_optimizer.json}"
MASTER_PORT="${MASTER_PORT:-12346}"
REPORT_TO="${REPORT_TO:-none}"
DATASET_NAME="${DATASET_NAME:-video_r1_train_uvb_eval}"
TEE_TRAINING_LOG="${TEE_TRAINING_LOG:-true}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.4}"
VLLM_DEVICE="${VLLM_DEVICE:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NCCL_SAFE_MODE="${NCCL_SAFE_MODE:-false}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
# ~128 * 28 * 28; tighter vision bounds to reduce VRAM (override with MAX_PIXELS / MIN_PIXELS).
MAX_PIXELS="${MAX_PIXELS:-100352}"
MIN_PIXELS="${MIN_PIXELS:-100352}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
# Default 1500 steps for manageable wall time; full epoch training: export MAX_STEPS=-1 before launch.
MAX_STEPS="${MAX_STEPS:-1500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-200}"
# 8-bit loads int8 weights; Transformers+Qwen2.5-VL then runs _initialize_weights(normal_) which fails on Char.
# Default off: use bf16 (TORCH_DTYPE). Set LOAD_IN_8BIT=true only if you need VRAM and your stack supports it.
LOAD_IN_8BIT="${LOAD_IN_8BIT:-false}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-}"
ANSWER_ACCURACY_WEIGHT="${ANSWER_ACCURACY_WEIGHT:-0.8}"
ANSWER_FORMAT_WEIGHT="${ANSWER_FORMAT_WEIGHT:-0.2}"
GRPO_TRAIN_VIDEO_ONLY="${GRPO_TRAIN_VIDEO_ONLY:-true}"
GRPO_TRAIN_VIDEO_ONLY_LC="$(printf '%s' "${GRPO_TRAIN_VIDEO_ONLY}" | tr '[:upper:]' '[:lower:]')"

mkdir -p "${OUTPUT_DIR}"

mkdir -p "${SPLIT_DIR}"
if [[ ! -s "${TRAIN_SPLIT_FILE}" || ! -s "${EVAL_SPLIT_FILE}" || "${TRAIN_SOURCE}" -nt "${TRAIN_SPLIT_FILE}" || "${TRAIN_SOURCE}" -nt "${EVAL_SPLIT_FILE}" ]]; then
  echo "[VIDEO-GRPO-LORA] creating train/eval split from TRAIN_SOURCE (fraction=${GRPO_EVAL_FRACTION}, seed=${GRPO_EVAL_SPLIT_SEED}, video_only=${GRPO_EVAL_VIDEO_ONLY})"
  python "${REPO_ROOT}/src/scripts/split_jsonl_train_eval.py" \
    --input "${TRAIN_SOURCE}" \
    --train-out "${TRAIN_SPLIT_FILE}" \
    --eval-out "${EVAL_SPLIT_FILE}" \
    --eval-fraction "${GRPO_EVAL_FRACTION}" \
    --seed "${GRPO_EVAL_SPLIT_SEED}" \
    --video-only "${GRPO_EVAL_VIDEO_ONLY}"
fi

TRAIN_FILE="${TRAIN_SPLIT_FILE}"
if [[ -n "${GRPO_TEST_FILE:-}" ]]; then
  TEST_FILE="${GRPO_TEST_FILE}"
else
  TEST_FILE="${EVAL_SPLIT_FILE}"
fi

if [[ -n "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="${NUM_GPUS}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  else
    NUM_GPUS="1"
  fi
fi

if [[ -n "${TRAIN_NUM_GPUS:-}" ]]; then
  TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS}"
elif [[ "${NUM_GPUS}" -gt 1 ]]; then
  TRAIN_NUM_GPUS="$((NUM_GPUS - 1))"
else
  TRAIN_NUM_GPUS="${NUM_GPUS}"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi

NCCL_SAFE_MODE_LC="$(printf '%s' "${NCCL_SAFE_MODE}" | tr '[:upper:]' '[:lower:]')"
LOAD_IN_8BIT_LC="$(printf '%s' "${LOAD_IN_8BIT}" | tr '[:upper:]' '[:lower:]')"

if [[ "${NCCL_SAFE_MODE_LC}" == "true" ]]; then
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
fi

echo "[VIDEO-GRPO-LORA] QWEN_PATH=${QWEN_PATH}"
if [[ -n "${PROCESSOR_PATH}" ]]; then
  echo "[VIDEO-GRPO-LORA] PROCESSOR_PATH=${PROCESSOR_PATH} (AutoProcessor; weights still from QWEN_PATH)"
else
  echo "[VIDEO-GRPO-LORA] PROCESSOR_PATH=<unset> (AutoProcessor from model path; set QWEN_BASE_PATH if processor load fails)"
fi
echo "[VIDEO-GRPO-LORA] TRAIN_SOURCE=${TRAIN_SOURCE}"
echo "[VIDEO-GRPO-LORA] TRAIN_FILE=${TRAIN_FILE}"
echo "[VIDEO-GRPO-LORA] TEST_FILE=${TEST_FILE}"
echo "[VIDEO-GRPO-LORA] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[VIDEO-GRPO-LORA] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-<none>}"
echo "[VIDEO-GRPO-LORA] NUM_GPUS=${NUM_GPUS} (train processes: ${TRAIN_NUM_GPUS}, 1 GPU reserved for vLLM when NUM_GPUS>1)"
echo "[VIDEO-GRPO-LORA] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[VIDEO-GRPO-LORA] VLLM_GPU_UTIL=${VLLM_GPU_UTIL}"
echo "[VIDEO-GRPO-LORA] TORCH_DTYPE=${TORCH_DTYPE}"
echo "[VIDEO-GRPO-LORA] NCCL_SAFE_MODE=${NCCL_SAFE_MODE}"
echo "[VIDEO-GRPO-LORA] NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-<unset>}"
echo "[VIDEO-GRPO-LORA] NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-<unset>}"
echo "[VIDEO-GRPO-LORA] TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-<unset>}"
echo "[VIDEO-GRPO-LORA] MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}"
echo "[VIDEO-GRPO-LORA] MIN_PIXELS=${MIN_PIXELS} MAX_PIXELS=${MAX_PIXELS}"
echo "[VIDEO-GRPO-LORA] NUM_GENERATIONS=${NUM_GENERATIONS}"
echo "[VIDEO-GRPO-LORA] NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "[VIDEO-GRPO-LORA] LEARNING_RATE=${LEARNING_RATE}"
echo "[VIDEO-GRPO-LORA] MAX_STEPS=${MAX_STEPS} (use MAX_STEPS=-1 for epoch-only, no cap)"
echo "[VIDEO-GRPO-LORA] LOGGING_STEPS=${LOGGING_STEPS}"
echo "[VIDEO-GRPO-LORA] SAVE_STEPS=${SAVE_STEPS}"
echo "[VIDEO-GRPO-LORA] VLLM_MAX_FRAMES=${VLLM_MAX_FRAMES:-8}"
echo "[VIDEO-GRPO-LORA] LOAD_IN_8BIT=${LOAD_IN_8BIT}"
echo "[VIDEO-GRPO-LORA] LORA_R=${LORA_R} LORA_ALPHA=${LORA_ALPHA} LORA_DROPOUT=${LORA_DROPOUT}"
echo "[VIDEO-GRPO-LORA] GRPO_TRAIN_VIDEO_ONLY=${GRPO_TRAIN_VIDEO_ONLY}"
echo "[VIDEO-GRPO-LORA] REWARD_WEIGHTS=${REWARD_WEIGHTS:-<default>}"
echo "[VIDEO-GRPO-LORA] ANSWER_ACCURACY_WEIGHT=${ANSWER_ACCURACY_WEIGHT:-<unset>}"
echo "[VIDEO-GRPO-LORA] ANSWER_FORMAT_WEIGHT=${ANSWER_FORMAT_WEIGHT:-<unset>}"

REWARD_ARGS=()
if [[ -n "${REWARD_WEIGHTS}" ]]; then
  REWARD_ARGS+=(--reward_weights "${REWARD_WEIGHTS}")
fi
if [[ -n "${ANSWER_ACCURACY_WEIGHT}" ]]; then
  REWARD_ARGS+=(--answer_accuracy_weight "${ANSWER_ACCURACY_WEIGHT}")
fi
if [[ -n "${ANSWER_FORMAT_WEIGHT}" ]]; then
  REWARD_ARGS+=(--answer_format_weight "${ANSWER_FORMAT_WEIGHT}")
fi

TRAIN_VIDEO_ONLY_ARGS=()
if [[ "${GRPO_TRAIN_VIDEO_ONLY_LC}" == "true" || "${GRPO_TRAIN_VIDEO_ONLY_LC}" == "1" || "${GRPO_TRAIN_VIDEO_ONLY_LC}" == "yes" ]]; then
  TRAIN_VIDEO_ONLY_ARGS+=(--train_video_only true)
fi

MODEL_QUANT_ARGS=()
if [[ "${LOAD_IN_8BIT_LC}" == "true" ]]; then
  MODEL_QUANT_ARGS+=(--load_in_8bit true)
fi

TEE_TRAINING_LOG_LC="$(printf '%s' "${TEE_TRAINING_LOG}" | tr '[:upper:]' '[:lower:]')"

MAX_STEPS_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
  MAX_STEPS_ARGS+=(--max_steps "${MAX_STEPS}")
fi

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

LORA_ARGS=(
  --use_peft true
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_target_modules q_proj k_proj v_proj o_proj
)

_run_training() {
PYTHONPATH="./src" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m torch.distributed.run --nproc_per_node="${TRAIN_NUM_GPUS}" \
  --nnodes="1" \
  --node_rank="0" \
  --master_addr="127.0.0.1" \
  --master_port="${MASTER_PORT}" \
  -m open_r1.grpo_video \
  --use_vllm "${USE_VLLM}" \
  --dataset_name "${DATASET_NAME}" \
  --vllm_device "${VLLM_DEVICE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_UTIL}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${QWEN_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --test_file "${TEST_FILE}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type "constant" \
  --logging_steps "${LOGGING_STEPS}" \
  --bf16 true \
  --torch_dtype "${TORCH_DTYPE}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --min_pixels "${MIN_PIXELS}" \
  --max_pixels "${MAX_PIXELS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --run_name "${RUN_NAME}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --save_only_model true \
  --report_to "${REPORT_TO}" \
  --temperature 1.0 \
  --num_generations "${NUM_GENERATIONS}" \
  --deepspeed "${DS_CONFIG}" \
  "${RESUME_ARGS[@]}" \
  "${MAX_STEPS_ARGS[@]}" \
  "${MODEL_QUANT_ARGS[@]}" \
  "${REWARD_ARGS[@]}" \
  "${TRAIN_VIDEO_ONLY_ARGS[@]}" \
  "${LORA_ARGS[@]}" \
  2>&1
}

if [[ "${TEE_TRAINING_LOG_LC}" == "true" || "${TEE_TRAINING_LOG_LC}" == "1" || "${TEE_TRAINING_LOG_LC}" == "yes" ]]; then
  _run_training | tee "${OUTPUT_DIR}/training_log.txt"
else
  _run_training
fi
