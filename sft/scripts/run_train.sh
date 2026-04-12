#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SFT_DIR}"

# Line-buffered stdout when logging to a file (nohup, PBS) — avoids empty .log until buffer fills.
export PYTHONUNBUFFERED=1

# accelerate imports DeepSpeed when unwrapping PEFT models; DeepSpeed's import checks CUDA_HOME even
# though this job uses HF Trainer + torch DDP only. If CUDA_HOME is unset, set it from nvcc (after e.g. module load cuda).
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  _NVCC="$(command -v nvcc)"
  export CUDA_HOME="$(cd "$(dirname "${_NVCC}")/../.." && pwd)"
  echo "[SFT] CUDA_HOME was unset; set from nvcc -> ${CUDA_HOME}"
fi

SFT_MODE="${SFT_MODE:-length}"
CONFIG_PATH="${CONFIG_PATH:-}"
MASTER_PORT="${MASTER_PORT:-12355}"
USE_VISION="${USE_VISION:-true}"

if [[ -z "${CONFIG_PATH}" ]]; then
  case "${SFT_MODE}" in
    length)
      CONFIG_PATH="configs/train_lora_qwen25vl3b_length.yaml"
      ;;
    perspective)
      CONFIG_PATH="configs/train_lora_qwen25vl3b_perspective.yaml"
      ;;
    *)
      echo "[SFT] Unsupported SFT_MODE: ${SFT_MODE}" >&2
      exit 1
      ;;
  esac
fi

if [[ -n "${NUM_GPUS:-}" ]]; then
  NUM_GPUS="${NUM_GPUS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
else
  NUM_GPUS="1"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_GPUS - 1)))"
fi

echo "[SFT] config: ${CONFIG_PATH}"
echo "[SFT] mode: ${SFT_MODE}"
echo "[SFT] NUM_GPUS=${NUM_GPUS}"
echo "[SFT] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[SFT] USE_VISION=${USE_VISION}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  echo "[SFT] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    scripts/train_sft.py --config "${CONFIG_PATH}" --use-vision "${USE_VISION}" "${RESUME_ARGS[@]}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/train_sft.py --config "${CONFIG_PATH}" --use-vision "${USE_VISION}" "${RESUME_ARGS[@]}"
fi
