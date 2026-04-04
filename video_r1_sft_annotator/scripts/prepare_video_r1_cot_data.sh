#!/bin/bash

set -euo pipefail

PYTHONPATH="./src" python -m video_r1_sft_annotator.prepare_video_r1_cot \
  --dataset-dir "data/video_r1_cot/raw" \
  --processed-dir "data/video_r1_cot/processed" \
  "$@"
