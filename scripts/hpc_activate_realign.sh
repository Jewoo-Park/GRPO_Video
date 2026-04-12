#!/usr/bin/env bash
# Source this file so every new shell gets Python + venv without manual module lines:
#   source /path/to/GRPO_Video_2/scripts/hpc_activate_realign.sh
#
# Adjust module names if your site differs: module avail gcc python

module purge 2>/dev/null || true
module load gcc/11.4.0-nscc
module load python/3.11.7-gcc11

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/scratch/pip_cache}"
mkdir -p "${PIP_CACHE_DIR}"

# shellcheck disable=SC1091
source "${HOME}/scratch/.venv_realign/bin/activate"
