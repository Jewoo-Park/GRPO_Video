#!/usr/bin/env bash
# Apply sft/requirements.txt in the *currently active* venv.
# Torch cu124: use scripts/run_setup_sft.sh first, or install torch separately.
# Removes GRPO leftovers that break SFT (DeepSpeed import / r1-v metadata).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${HERE}/requirements.txt"
python -m pip uninstall -y deepspeed 2>/dev/null || true
python -m pip uninstall -y r1-v 2>/dev/null || true
echo "[install_requirements] Done. Run: pip check"
