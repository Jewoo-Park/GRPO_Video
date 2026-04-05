#!/bin/bash

set -euo pipefail

PYTHONPATH="./src" python -m video_r1_sft_annotator.sample_processed \
  --input-dir  "data/video_r1_cot/processed" \
  --output-dir "data/video_r1_cot/sampled_6k" \
  --preset     "additional_6k" \
  --exclude-dir "data/video_r1_cot/sampled_4k" \
  "$@"
