import argparse
import csv
import importlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common import boxes_from_binary_mask, dice_iou, load_yolo_boxes, masks_to_polygons, save_yolo_polygons


METHOD_MODULES = {
    "otsu": "otsu",
    "multi_otsu": "multi_otsu",
    "adaptive": "adaptive",
    "watershed": "watershed",
    "otsu_watershed": "otsu_watershed",
    "connected": "connected_components",
    "connected_components": "connected_components",
    "random_walker": "random_walker",
    "chan_vese": "chan_vese",
    "morph_gac": "morph_gac",
    "sam": "sam",
    "unet": "unet",
    "guided_unet": "guided_unet",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 1 mask generation for one dataset/method.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHOD_MODULES))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase1_masks"))
    parser.add_argument("--polygon-dir", type=Path, default=Path("results/polygon_labels"))
    parser.add_argument("--write-polygons", action="store_true")
    return parser.parse_args()


def read_config(path):
    if path is None:
        return {}

    with path.open() as f:
        return json.load(f)


def read_manifest(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_gt_mask(path):
    if not path:
        return None

    mask_path = Path(path)
    if not mask_path.exists():
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    return (mask > 0).astype(np.uint8)


def row_boxes(row, image, gt_mask):
    height, width = image.shape[:2]

    if row.get("boxes_path"):
        return load_yolo_boxes(row["boxes_path"], width, height)

    if gt_mask is not None:
        return boxes_from_binary_mask(gt_mask)

    if all(row.get(k) for k in ["x1", "y1", "x2", "y2"]):
        return [[float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]]

    return []


def combine_instance_masks(masks, shape):
    combined = np.zeros(shape, dtype=np.uint8)

    for mask in masks:
        combined[mask > 0] = 1

    return combined


def safe_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{value:.6f}"


def main():
    args = parse_args()
    config = read_config(args.config)
    rows = read_manifest(args.manifest)
    module_name = METHOD_MODULES[args.method]
    method_module = importlib.import_module(module_name)

    percase_rows = []
    runtime_total = 0.0

    for row in rows:
        case_id = row.get("case_id") or Path(row["image_path"]).stem
        image = load_image(row["image_path"])
        gt_mask = load_gt_mask(row.get("mask_path", ""))
        boxes = row_boxes(row, image, gt_mask)

        start = time.perf_counter()
        masks = method_module.generate_masks(image, boxes, config)
        runtime_s = time.perf_counter() - start
        runtime_total += runtime_s

        pred_mask = combine_instance_masks(masks, image.shape[:2])

        if gt_mask is not None:
            dice, iou = dice_iou(pred_mask, gt_mask)
        else:
            dice, iou = math.nan, math.nan

        percase_rows.append(
            {
                "case_id": case_id,
                "n_boxes": len(boxes),
                "dice": safe_float(dice),
                "iou": safe_float(iou),
                "runtime_s": f"{runtime_s:.6f}",
            }
        )

        if args.write_polygons:
            polygons = masks_to_polygons(masks)
            label_path = args.polygon_dir / args.dataset / args.method / f"{case_id}.txt"
            save_yolo_polygons(label_path, polygons, image.shape[1], image.shape[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    percase_path = args.output_dir / f"{args.experiment_id}_{args.dataset}_{args.method}_percase.csv"
    summary_path = args.output_dir / f"{args.experiment_id}_{args.dataset}_{args.method}_summary.csv"

    with percase_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "n_boxes", "dice", "iou", "runtime_s"])
        writer.writeheader()
        writer.writerows(percase_rows)

    dice_values = [float(row["dice"]) for row in percase_rows if row["dice"] != "NaN"]
    iou_values = [float(row["iou"]) for row in percase_rows if row["iou"] != "NaN"]

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "n_cases": len(percase_rows),
        "dice_mean": safe_float(float(np.mean(dice_values)) if dice_values else math.nan),
        "dice_std": safe_float(float(np.std(dice_values)) if dice_values else math.nan),
        "iou_mean": safe_float(float(np.mean(iou_values)) if iou_values else math.nan),
        "iou_std": safe_float(float(np.std(iou_values)) if iou_values else math.nan),
        "runtime_total_s": f"{runtime_total:.6f}",
    }

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "n_cases",
                "dice_mean",
                "dice_std",
                "iou_mean",
                "iou_std",
                "runtime_total_s",
            ],
        )
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved per-case CSV: {percase_path}")
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent))
    main()
