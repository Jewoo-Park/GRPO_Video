#!/usr/bin/env bash
# GRPO 전용 venv 활성화 (scratch/.venv_grpo).
#   source /path/to/GRPO_Video_2/scripts/hpc_activate_grpo.sh
#
# 다른 경로에 만들었다면: VENV_GRPO_PATH=/path/to/.venv_grpo source hpc_activate_grpo.sh
#
# GPU/DeepSpeed: TRL → deepspeed import 시 CUDA_HOME 이 필요할 수 있음.
# 컴퓨트 노드 예:  export GRPO_CUDA_MODULE="cuda/12.2.2"   # module avail cuda

module purge 2>/dev/null || true
module load gcc/11.4.0-nscc
module load python/3.11.7-gcc11
if [[ -n "${GRPO_CUDA_MODULE:-}" ]]; then
  # shellcheck disable=SC1091
  module load "${GRPO_CUDA_MODULE}" 2>/dev/null || echo "[hpc_activate_grpo] WARNING: module load ${GRPO_CUDA_MODULE} failed" >&2
fi

# DeepSpeed: installed_cuda_version() requires CUDA_HOME → existing toolkit with bin/nvcc
_grpo_ensure_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" ]] && [[ -d "${CUDA_HOME}" ]] && [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    local _nv
    _nv="$(command -v nvcc)"
    export CUDA_HOME
    CUDA_HOME="$(cd "$(dirname "${_nv}")/.." && pwd)"
    return 0
  fi
  local d
  for d in \
    /app/apps/cuda/12.2.2 \
    /app/apps/cuda/12.4 \
    /usr/local/cuda-12.4 \
    /usr/local/cuda-12 \
    /usr/local/cuda; do
    if [[ -d "${d}" && -x "${d}/bin/nvcc" ]]; then
      export CUDA_HOME="${d}"
      return 0
    fi
  done
  return 1
}
_grpo_ensure_cuda_home || echo "[hpc_activate_grpo] WARNING: CUDA_HOME unset (DeepSpeed import may fail). Set GRPO_CUDA_MODULE or export CUDA_HOME." >&2

if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/bin" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/scratch/pip_cache}"
mkdir -p "${PIP_CACHE_DIR}"

VENV_GRPO="${VENV_GRPO_PATH:-${HOME}/scratch/.venv_grpo}"
if [[ ! -f "${VENV_GRPO}/bin/activate" ]]; then
  echo "[hpc_activate_grpo] ERROR: venv not found: ${VENV_GRPO}" >&2
  echo "[hpc_activate_grpo] Create it with: bash <REPO>/GRPO_Video_2/scripts/run_setup_grpo.sh" >&2
  echo "[hpc_activate_grpo] Or set VENV_GRPO_PATH to an existing .venv_grpo" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${VENV_GRPO}/bin/activate"

# torch+cu124 + pip nvidia-nvjitlink-cu12: CUDA module's lib64 can supply an older libnvJitLink.so.12
# first in LD_LIBRARY_PATH → ImportError __nvJitLinkComplete_12_4. Prefer venv wheel.
_grpo_prepend_nvjitlink_ld() {
  [[ -n "${VIRTUAL_ENV:-}" ]] || return 0
  local _lib
  _lib="$(python -c 'import os, sysconfig; p=os.path.join(sysconfig.get_path("platlib"), "nvidia", "nvjitlink", "lib"); print(p if os.path.isdir(p) else "")' 2>/dev/null || true)"
  if [[ -n "${_lib}" ]]; then
    export LD_LIBRARY_PATH="${_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}
_grpo_prepend_nvjitlink_ld
