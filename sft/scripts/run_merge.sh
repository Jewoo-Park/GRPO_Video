#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SFT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/merge_lora_qwen25vl3b.yaml}"
SFT_MODE="${SFT_MODE:-length}"
MERGE_STAGE="${MERGE_STAGE:-sft}"

if [[ -z "${CONFIG_PATH:-}" || "${CONFIG_PATH}" == "configs/merge_lora_qwen25vl3b.yaml" ]]; then
  case "${MERGE_STAGE}" in
    sft)
      case "${SFT_MODE}" in
        length)
          CONFIG_PATH="configs/merge_lora_qwen25vl3b_length.yaml"
          ;;
        perspective)
          CONFIG_PATH="configs/merge_lora_qwen25vl3b_perspective.yaml"
          ;;
        *)
          echo "[SFT-MERGE] Unsupported SFT_MODE: ${SFT_MODE}" >&2
          exit 1
          ;;
      esac
      ;;
    grpo)
      case "${SFT_MODE}" in
        length)
          CONFIG_PATH="configs/merge_lora_grpo_length.yaml"
          ;;
        perspective)
          CONFIG_PATH="configs/merge_lora_grpo_perspective.yaml"
          ;;
        *)
          echo "[SFT-MERGE] Unsupported SFT_MODE: ${SFT_MODE}" >&2
          exit 1
          ;;
      esac
      ;;
    *)
      echo "[SFT-MERGE] Unsupported MERGE_STAGE: ${MERGE_STAGE}" >&2
      exit 1
      ;;
  esac
fi

echo "[SFT-MERGE] config: ${CONFIG_PATH}"
echo "[SFT-MERGE] stage: ${MERGE_STAGE}"
echo "[SFT-MERGE] mode: ${SFT_MODE}"
python scripts/merge_lora.py --config "${CONFIG_PATH}"
