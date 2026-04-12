#!/usr/bin/env bash
# SFT-only venv (separate from GRPO: scripts/hpc_activate_realign.sh uses .venv_realign).
#   source /path/to/GRPO_Video_2/scripts/hpc_activate_sft.sh
#
# Prereq: create the env once with: bash scripts/run_setup_sft.sh

module purge 2>/dev/null || true
module load gcc/11.4.0-nscc
module load python/3.11.7-gcc11

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/scratch/pip_cache}"
mkdir -p "${PIP_CACHE_DIR}"

# shellcheck disable=SC1091
source "${HOME}/scratch/.venv_sft/bin/activate"
