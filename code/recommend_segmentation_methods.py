from pathlib import Path
import argparse
import csv
import random

import numpy as np

from segmentation_methods import (
    BUSI_DATASET_DIR,
    METHODS,
    evaluate_method,
    find_pairs,
    load_gray,
    summarize,
)


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation methods and recommend the best one by IoU."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BUSI_DATASET_DIR,
        help="Dataset root containing images and *_mask.png files.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of positive image-mask pairs to sample. Use 0 for all pairs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "method_iou_recommendations.csv",
        help="Detailed per-image CSV output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=OUTPUT_DIR / "method_iou_summary.csv",
        help="Summary CSV output path.",
    )
    return parser.parse_args()


def positive_pairs(dataset_dir):
    pairs = []

    for img_path, mask_path in find_pairs(dataset_dir):
        gt_mask = load_gray(mask_path)
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        if np.any(gt_mask > 0):
            pairs.append((img_path, mask_path))

    return pairs


def choose_pairs(pairs, num_images, seed):
    if num_images == 0 or num_images >= len(pairs):
        return pairs

    rng = random.Random(seed)
    return rng.sample(pairs, num_images)


def write_details(csv_path, rows, methods):
    fieldnames = ["image", "mask"]

    for method in methods:
        fieldnames.append(f"{method}_iou")
        fieldnames.append(f"{method}_choice")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(csv_path, summary_rows):
    fieldnames = [
        "rank",
        "method",
        "mean_iou",
        "median_iou",
        "std_iou",
        "min_iou",
        "max_iou",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    args = parse_args()
    random.seed(args.seed)

    pairs = positive_pairs(args.dataset)
    print(f"Positive image-mask pairs found: {len(pairs)}", flush=True)

    if not pairs:
        raise ValueError(f"No non-empty image-mask pairs found in {args.dataset}")

    sample_pairs = choose_pairs(pairs, args.num_images, args.seed)
    print(f"Evaluating image-mask pairs: {len(sample_pairs)}", flush=True)
    print(f"Methods: {', '.join(METHODS)}", flush=True)

    details = []
    method_ious = {method: [] for method in METHODS}

    for idx, (img_path, mask_path) in enumerate(sample_pairs, start=1):
        print(f"[{idx}/{len(sample_pairs)}] {img_path.name}", flush=True)

        image = load_gray(img_path)
        gt_mask = load_gray(mask_path)
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        row = {
            "image": str(img_path.relative_to(args.dataset)),
            "mask": str(mask_path.relative_to(args.dataset)),
        }

        for method in METHODS:
            try:
                iou, choice = evaluate_method(method, image, gt_mask)
            except Exception as exc:
                print(f"  [ERROR] {method}: {exc}", flush=True)
                iou = 0.0
                choice = "error"

            row[f"{method}_iou"] = f"{iou:.6f}"
            row[f"{method}_choice"] = choice
            method_ious[method].append(iou)

        details.append(row)

    ranked = []
    for method in METHODS:
        stats = summarize(method_ious[method])
        ranked.append((method, stats))

    ranked.sort(key=lambda item: (item[1]["mean"], item[1]["median"]), reverse=True)

    summary_rows = []
    for rank, (method, stats) in enumerate(ranked, start=1):
        summary_rows.append(
            {
                "rank": rank,
                "method": method,
                "mean_iou": f"{stats['mean']:.6f}",
                "median_iou": f"{stats['median']:.6f}",
                "std_iou": f"{stats['std']:.6f}",
                "min_iou": f"{stats['min']:.6f}",
                "max_iou": f"{stats['max']:.6f}",
            }
        )

    write_details(args.output, details, METHODS)
    write_summary(args.summary_output, summary_rows)

    print("\n=== METHOD IoU RANKING ===")
    for row in summary_rows:
        print(
            f"{row['rank']:>2}. {row['method']:22s} "
            f"mean={row['mean_iou']} "
            f"median={row['median_iou']} "
            f"std={row['std_iou']}"
        )

    best = summary_rows[0]
    print(
        "\nRecommended method: "
        f"{best['method']} "
        f"(mean IoU={best['mean_iou']}, median IoU={best['median_iou']})"
    )
    print(f"Detailed IoUs saved to: {args.output}")
    print(f"Summary saved to: {args.summary_output}")


if __name__ == "__main__":
    main()
