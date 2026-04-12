#!/usr/bin/env bash
# GRPO 전용 venv ($HOME/scratch/.venv_grpo). setup.sh 한 번이면 torch·trl·vllm·transformers·flash-attn까지 같이 설치.
#
# 반드시 PyPI에 닿는 노드에서 실행 (로그인 노드 등). 컴퓨트 노드만 오프라인이면 여기서 받아 두고 복사.
#
#   bash scripts/run_setup_grpo.sh
#
# flash-attn을 그 GPU 아키텍처용으로 소스 빌드하려면 (권장, illegal instruction 방지):
#   module avail cuda   # 사이트에 맞는 버전 확인 후
#   export GRPO_CUDA_MODULE="cuda/12.2.2"    # 예시
#   export TORCH_CUDA_ARCH_LIST="90"         # H100=90, A100=80 등
#   bash scripts/run_setup_grpo.sh
#
# Env overrides (setup.sh와 동일):
#   INSTALL_TORCH_CU124=true|false   (default: true)
#   INSTALL_FLASH_ATTN=true|false    (default: true)
#   VENV_PATH=...                    (default: $HOME/scratch/.venv_grpo)
#   PIP_CACHE_DIR=...                (default: $HOME/scratch/pip_cache)
#   GRPO_CUDA_MODULE=...             (optional) e.g. cuda/12.x for nvcc during flash-attn build
#
# flash-attn만 GPU 노드에서 추가 설치할 때 (torch는 nvjitlink 12.4.127 고정 — 업그레이드 금지):
#   source scripts/hpc_activate_grpo.sh
#   pip install "nvidia-nvjitlink-cu12==12.4.127" --force-reinstall   # 잘못 올렸을 때만
#   pip install flash-attn==2.6.3 --no-build-isolation

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module purge 2>/dev/null || true
module load gcc/11.4.0-nscc
module load python/3.11.7-gcc11
if [[ -n "${GRPO_CUDA_MODULE:-}" ]]; then
  # shellcheck disable=SC1091
  module load "${GRPO_CUDA_MODULE}"
  echo "[run_setup_grpo] Loaded CUDA module: ${GRPO_CUDA_MODULE}"
fi

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/scratch/pip_cache}"
mkdir -p "${PIP_CACHE_DIR}"

export VENV_PATH="${VENV_PATH:-${HOME}/scratch/.venv_grpo}"
export INSTALL_TORCH_CU124="${INSTALL_TORCH_CU124:-true}"
export INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-true}"

cd "${REPO_ROOT}"
bash setup.sh

echo "[run_setup_grpo] Done. Next shells: source ${REPO_ROOT}/scripts/hpc_activate_grpo.sh"
