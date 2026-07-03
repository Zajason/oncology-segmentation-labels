#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-busi}"
MANIFEST="${2:-manifests/${DATASET}.csv}"
OUT_DIR="${3:-results/phase1_masks}"
POLY_DIR="${4:-results/polygon_labels}"

case "$DATASET" in
  busi)
    declare -A IDS=(
      [otsu]=E040
      [multi_otsu]=E041
      [adaptive]=E042
      [watershed]=E043
      [otsu_watershed]=E044
      [connected]=E045
      [random_walker]=E046
      [chan_vese]=E047
      [morph_gac]=E048
    )
    ;;
  brats)
    declare -A IDS=(
      [otsu]=E052
      [multi_otsu]=E053
      [adaptive]=E054
      [watershed]=E055
      [otsu_watershed]=E056
      [connected]=E057
      [random_walker]=E058
      [chan_vese]=E059
      [morph_gac]=E060
    )
    ;;
  *)
    echo "Unsupported dataset for this classical runner: $DATASET" >&2
    exit 1
    ;;
esac

for METHOD in otsu multi_otsu adaptive watershed otsu_watershed connected random_walker chan_vese morph_gac; do
  echo "Running ${IDS[$METHOD]} $DATASET $METHOD"
  python code/phase1_runner.py \
    --manifest "$MANIFEST" \
    --dataset "$DATASET" \
    --method "$METHOD" \
    --experiment-id "${IDS[$METHOD]}" \
    --output-dir "$OUT_DIR" \
    --polygon-dir "$POLY_DIR" \
    --write-polygons
done
