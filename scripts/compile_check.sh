#!/usr/bin/env bash
set -euo pipefail

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-.pycache}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m py_compile \
  code/adaptive.py \
  code/build_manifests.py \
  code/chan_vese.py \
  code/common.py \
  code/connected_components.py \
  code/guided_unet.py \
  code/morph_gac.py \
  code/multi_otsu.py \
  code/otsu.py \
  code/otsu_watershed.py \
  code/phase1_runner.py \
  code/random_walker.py \
  code/sam.py \
  code/train_guided_unet.py \
  code/train_unet.py \
  code/train_yolo_phase2.py \
  code/unet.py \
  code/watershed.py
