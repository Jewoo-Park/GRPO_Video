#!/bin/bash

set -euo pipefail

PYTHONPATH="./src" python -m video_r1_sft_annotator.export_sft_dataset \
  --config "configs/export_sft.yaml" \
  "$@"
