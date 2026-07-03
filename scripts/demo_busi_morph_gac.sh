#!/usr/bin/env bash
set -euo pipefail

python code/build_manifests.py --sample-size 10 --brats-limit 10
python code/phase1_runner.py \
  --manifest manifests/busi_sample.csv \
  --dataset busi \
  --method morph_gac \
  --experiment-id demo \
  --write-polygons
