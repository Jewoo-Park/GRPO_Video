#!/bin/bash

set -euo pipefail

PYTHONPATH="./src" python -m video_r1_sft_annotator.annotate \
  --config "configs/generate_granularity.yaml" \
  "$@"
