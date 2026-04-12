#!/usr/bin/env bash
# SFT 전용 일회 설치 (기본: ~/scratch/.venv_sft). setup.sh(GRPO)는 호출하지 않는다.
#
#   bash /path/to/GRPO_Video_2/scripts/run_setup_sft.sh
#
# Overrides: VENV_PATH, INSTALL_TORCH_CU124 (default true), INSTALL_FLASH_ATTN (default false)
# flash-attn: GPU 노드에서 module load cuda/12.2.2 후 INSTALL_FLASH_ATTN=true ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
export VENV_PATH="${VENV_PATH:-${HOME}/scratch/.venv_sft}"
INSTALL_TORCH_CU124="${INSTALL_TORCH_CU124:-true}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-false}"

module purge 2>/dev/null || true
module load gcc/11.4.0-nscc
module load python/3.11.7-gcc11

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/scratch/pip_cache}"
mkdir -p "${PIP_CACHE_DIR}"

deactivate 2>/dev/null || true
unset VIRTUAL_ENV

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[run_setup_sft] Creating venv: ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

VENV_PYTHON="${VENV_PATH}/bin/python"
echo "[run_setup_sft] Using python: ${VENV_PYTHON}"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

INSTALL_TORCH_CU124_LC="$(printf '%s' "${INSTALL_TORCH_CU124}" | tr '[:upper:]' '[:lower:]')"
INSTALL_FLASH_ATTN_LC="$(printf '%s' "${INSTALL_FLASH_ATTN}" | tr '[:upper:]' '[:lower:]')"

if [[ "${INSTALL_TORCH_CU124_LC}" == "true" ]]; then
  echo "[run_setup_sft] Installing torch/cu124"
  "${VENV_PYTHON}" -m pip install \
    torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
else
  echo "[run_setup_sft] Skipping torch/cu124 (set INSTALL_TORCH_CU124=true to enable)"
fi

bash "${REPO_ROOT}/sft/install_requirements.sh"

if [[ "${INSTALL_FLASH_ATTN_LC}" == "true" ]]; then
  "${VENV_PYTHON}" -m pip install "flash-attn==2.6.3" --no-build-isolation || true
fi

echo "[run_setup_sft] Done. Next shells: source ${REPO_ROOT}/scripts/hpc_activate_sft.sh"
