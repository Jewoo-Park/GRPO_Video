#!/usr/bin/env bash
# One-shot GRPO env install matching repo setup.sh (torch cu124, r1-v deps, git transformers).
# Run on a node/login with module system, from anywhere:
#   bash /path/to/GRPO_Video_2/scripts/run_setup_realign.sh
#
# Env overrides:
#   INSTALL_TORCH_CU124=true|false   (default: true)
#   INSTALL_FLASH_ATTN=true|false    (default: true; may fail on some hosts)
#   VENV_PATH=...                    (default: $HOME/scratch/.venv_realign)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module purge 2>/dev/null || true
module load gcc/11.4.0-nscc
module load python/3.11.7-gcc11

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/scratch/pip_cache}"
mkdir -p "${PIP_CACHE_DIR}"

export VENV_PATH="${VENV_PATH:-${HOME}/scratch/.venv_realign}"
export INSTALL_TORCH_CU124="${INSTALL_TORCH_CU124:-true}"
export INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-true}"

cd "${REPO_ROOT}"
bash setup.sh

echo "[run_setup_realign] Done. Next shells: source ${REPO_ROOT}/scripts/hpc_activate_realign.sh"
