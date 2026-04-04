#!/bin/bash

set -euo pipefail

PYTHONPATH="./src" python -m video_r1_sft_annotator.sample_processed \
  --input-dir "data/video_r1_cot/processed" \
  --output-dir "data/video_r1_cot/processed_4k" \
  --preset "balanced_4k" \
  "$@"
