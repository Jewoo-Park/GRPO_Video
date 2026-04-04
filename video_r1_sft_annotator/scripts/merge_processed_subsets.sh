#!/bin/bash

set -euo pipefail

PYTHONPATH="./src" python -m video_r1_sft_annotator.merge_processed "$@"
