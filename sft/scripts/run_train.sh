#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SFT_DIR}"

SFT_MODE="${SFT_MODE:-length}"
CONFIG_PATH="${CONFIG_PATH:-}"
MASTER_PORT="${MASTER_PORT:-12355}"
USE_VISION="${USE_VISION:-}"

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

if [[ -z "${USE_VISION}" ]]; then
  read -r -p "[SFT] 이미지/프레임 입력을 사용할까요? [y/N]: " USE_VISION_REPLY
  if [[ "${USE_VISION_REPLY,,}" =~ ^(y|yes)$ ]]; then
    USE_VISION="true"
  else
    USE_VISION="false"
  fi
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

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    scripts/train_sft.py --config "${CONFIG_PATH}" --use-vision "${USE_VISION}"
else
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python scripts/train_sft.py --config "${CONFIG_PATH}" --use-vision "${USE_VISION}"
fi
