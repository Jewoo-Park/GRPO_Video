#!/bin/bash
# GRPO / r1-v 환경 전용. SFT는 scripts/run_setup_sft.sh + sft/requirements.txt 를 쓴다 (이 파일을 쓰지 않음).
#
# venv 위치는 VENV_PATH 로만 정한다:
#   scripts/run_setup_grpo.sh   → 기본 $HOME/scratch/.venv_grpo
#   scripts/run_setup_realign.sh → 기본 $HOME/scratch/.venv_realign
#   수동: VENV_PATH=/path/to/venv bash setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv_realign}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
INSTALL_TORCH_CU124="${INSTALL_TORCH_CU124:-false}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-true}"
VENV_SYSTEM_SITE_PACKAGES="${VENV_SYSTEM_SITE_PACKAGES:-false}"

# Avoid accidentally mixing in ~/.local packages (can happen after running pip without a venv).
export PYTHONNOUSERSITE=1

# Allow redirecting pip cache away from home quota (recommended on HPC).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-}"

# Do not trust a pre-set VIRTUAL_ENV from another shell; always use VENV_PATH.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV

_venv_usable() {
  [[ -f "${VENV_PATH}/bin/activate" ]] && [[ -x "${VENV_PATH}/bin/python" ]]
}

if ! _venv_usable; then
  if [[ -e "${VENV_PATH}" ]]; then
    echo "[setup] Removing incomplete venv (interrupted create or broken tree): ${VENV_PATH}"
    rm -rf "${VENV_PATH}"
  fi
  echo "[setup] Creating venv: ${VENV_PATH}"
  VENV_SYSTEM_SITE_PACKAGES_LC="$(printf '%s' "${VENV_SYSTEM_SITE_PACKAGES}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${VENV_SYSTEM_SITE_PACKAGES_LC}" == "true" ]]; then
    # Useful on clusters where torch/CUDA are provided via module load.
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_PATH}"
  else
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
  fi
fi
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

VENV_PYTHON="${VENV_PATH}/bin/python"
echo "[setup] Using python: ${VENV_PYTHON} ($("${VENV_PYTHON}" -c "import sys; print(sys.prefix)") )"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

INSTALL_TORCH_CU124_LC="$(printf '%s' "${INSTALL_TORCH_CU124}" | tr '[:upper:]' '[:lower:]')"
INSTALL_FLASH_ATTN_LC="$(printf '%s' "${INSTALL_FLASH_ATTN}" | tr '[:upper:]' '[:lower:]')"

if [[ "${INSTALL_TORCH_CU124_LC}" == "true" ]]; then
  echo "[setup] Installing torch/cu124 pinned stack"
  "${VENV_PYTHON}" -m pip install \
    torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
else
  echo "[setup] Skipping explicit torch/cu124 install (set INSTALL_TORCH_CU124=true to enable)"
fi

cd "${REPO_ROOT}/src/r1-v"

# Install project code without pulling the huge dependency closure from setup.py.
# We'll install only what GRPO actually needs below.
"${VENV_PYTHON}" -m pip install -e "." --no-deps

# Runtime dependencies aligned with GRPO runs
# liger_kernel: declared in r1-v setup.py install_requires; satisfies pip metadata when using pip install -e . --no-deps
"${VENV_PYTHON}" -m pip install \
  "trl==0.14.0" \
  "peft==0.14.0" \
  "liger_kernel==0.5.2" \
  "accelerate" \
  "deepspeed==0.15.4" \
  "datasets" \
  "bitsandbytes>=0.43.0" \
  "sentencepiece>=0.1.99" \
  "huggingface-hub[cli]>=0.19.2,<1.0" \
  "hf_transfer>=0.1.4" \
  "vllm==0.7.2" \
  "wandb>=0.19.1" \
  "tensorboardx" \
  "qwen_vl_utils" \
  "torchvision" \
  "pillow" \
  "einops>=0.8.0" \
  "packaging>=23.0"

# Keep transformers at a known-good revision used by this repo.
"${VENV_PYTHON}" -m pip install "git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef"

if [[ "${INSTALL_FLASH_ATTN_LC}" == "true" ]]; then
  # flash-attn's setup imports torch. Do NOT `pip install --upgrade nvidia-nvjitlink-cu12`:
  # torch 2.5.1+cu124 requires nvidia-nvjitlink-cu12==12.4.127 exactly; upgrading breaks imports.
  # If you accidentally upgraded: pip install "nvidia-nvjitlink-cu12==12.4.127" --force-reinstall
  # flash-attn build can fail on unsupported environments (e.g., non-GPU login node, wrong arch).
  # On failure: INSTALL_FLASH_ATTN=false here, then on a GPU node: pip install flash-attn==2.6.3 --no-build-isolation
  if "${VENV_PYTHON}" -m pip install "flash-attn==2.6.3" --no-build-isolation; then
    echo "[setup] flash-attn installed OK"
  else
    echo "[setup] WARNING: flash-attn install failed (common on login nodes). Set INSTALL_FLASH_ATTN=false and install on a GPU node, or fix CUDA/nvjitlink. Continuing." >&2
  fi
fi

echo "[setup] Done. Active venv: ${VIRTUAL_ENV:-<none>}"
