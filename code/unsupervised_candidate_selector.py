from pathlib import Path
import argparse
import csv
import math
import random
import warnings

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

from segmentation_methods import (
    BUSI_DATASET_DIR,
    adaptive_raw,
    chanvese_raw,
    clean_mask,
    compute_iou,
    find_pairs,
    load_gray,
    morphological_snakes_raw,
    otsu_raw,
    random_walker_raw,
    summarize,
    watershed_raw,
)


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)
MIN_COMPONENT_AREA_RATIO = 0.001
MAX_COMPONENTS_PER_CANDIDATE = 5
MIN_PLAUSIBLE_AREA_RATIO = 0.003
MAX_PLAUSIBLE_AREA_RATIO = 0.25

warnings.filterwarnings(
    "ignore",
    message='The probability range is outside \\[0, 1\\].*',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select synthetic segmentation masks without ground-truth labels, "
            "then report IoU afterward for evaluation."
        )
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
        default=OUTPUT_DIR / "unsupervised_selector_results.csv",
        help="Detailed per-image CSV output path.",
    )
    parser.add_argument(
        "--polarity",
        choices=["dark", "bright", "either"],
        default="dark",
        help=(
            "Expected lesion intensity relative to local background. "
            "BUSI lesions are usually darker, so the default is dark."
        ),
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


def normalize_mask(mask):
    return (mask > 0).astype(np.uint8) * 255


def multi_otsu_candidates(image, num_classes=3):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresholds = threshold_multiotsu(blurred, classes=num_classes)
    regions = np.digitize(blurred, bins=thresholds)

    candidates = []
    for class_idx in range(num_classes):
        mask = (regions == class_idx).astype(np.uint8) * 255
        candidates.append((f"multi_otsu_class_{class_idx}", mask))

    return candidates


def raw_candidates(image):
    candidates = [
        ("otsu", otsu_raw(image)),
        ("adaptive_21_7", adaptive_raw(image, block_size=21, c_value=7)),
        ("adaptive_31_5", adaptive_raw(image, block_size=31, c_value=5)),
        ("adaptive_41_3", adaptive_raw(image, block_size=41, c_value=3)),
        ("watershed", watershed_raw(image)),
        ("chan_vese", chanvese_raw(image)),
        ("morphological_snakes", morphological_snakes_raw(image)),
    ]

    candidates.extend(multi_otsu_candidates(image))

    dark_mask, bright_mask = random_walker_raw(image)
    candidates.append(("random_walker_dark", dark_mask))
    candidates.append(("random_walker_bright", bright_mask))

    oriented = []
    for name, mask in candidates:
        mask = normalize_mask(mask)
        oriented.append((f"{name}_normal", mask))
        oriented.append((f"{name}_inverted", cv2.bitwise_not(mask)))

    return oriented


def components_from_candidate(name, mask):
    mask = clean_mask(mask)
    labels_count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    components = []
    min_area = max(int(mask.size * MIN_COMPONENT_AREA_RATIO), 10)

    for label in range(1, labels_count):
        component = np.zeros_like(mask, dtype=np.uint8)
        component[labels == label] = 255
        area = int((component > 0).sum())

        if area >= min_area:
            components.append((area, f"{name}_component_{label}", component))

    components.sort(key=lambda item: item[0], reverse=True)
    return [(name, component) for _, name, component in components[:MAX_COMPONENTS_PER_CANDIDATE]]


def candidate_components(image):
    components = []

    for name, mask in raw_candidates(image):
        components.extend(components_from_candidate(name, mask))

    return components


def range_score(value, low, high, hard_low=0.0, hard_high=1.0):
    if low <= value <= high:
        return 1.0
    if value < low:
        if value <= hard_low:
            return 0.0
        return (value - hard_low) / (low - hard_low)
    if value >= hard_high:
        return 0.0
    return (hard_high - value) / (hard_high - high)


def boundary_mask(mask):
    binary = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel)
    eroded = cv2.erode(binary, kernel)
    return (dilated - eroded) > 0


def shape_features(mask):
    binary = (mask > 0).astype(np.uint8)
    area = int(binary.sum())

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area": 0,
            "perimeter": 0.0,
            "solidity": 0.0,
            "compactness": 0.0,
            "border_fraction": 1.0,
        }

    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, closed=True))
    contour_area = float(cv2.contourArea(contour))

    if len(contour) >= 3:
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = min(contour_area / hull_area, 1.0)
    else:
        solidity = 0.0

    compactness = 0.0
    if perimeter > 0:
        compactness = min((4.0 * math.pi * contour_area) / (perimeter * perimeter), 1.0)

    border = np.zeros_like(binary)
    border[:3, :] = 1
    border[-3:, :] = 1
    border[:, :3] = 1
    border[:, -3:] = 1
    border_fraction = float(np.logical_and(binary > 0, border > 0).sum() / max(area, 1))

    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": solidity,
        "compactness": compactness,
        "border_fraction": border_fraction,
    }


def score_candidate(image, mask, all_masks, polarity):
    mask = normalize_mask(mask)
    binary = mask > 0
    total_pixels = mask.size
    area_ratio = float(binary.sum() / total_pixels)

    if not binary.any():
        return None

    if area_ratio < MIN_PLAUSIBLE_AREA_RATIO or area_ratio > MAX_PLAUSIBLE_AREA_RATIO:
        return None

    features = shape_features(mask)

    inside = image[binary].astype(np.float32)
    outside = image[~binary].astype(np.float32)
    image_std = max(float(np.std(image)), 1.0)
    inside_mean = float(inside.mean())
    outside_mean = float(outside.mean())
    signed_contrast = (outside_mean - inside_mean) / image_std
    abs_contrast = min(abs(signed_contrast), 1.0)

    if polarity == "dark":
        polarity_contrast = min(max(signed_contrast, 0.0), 1.0)
    elif polarity == "bright":
        polarity_contrast = min(max(-signed_contrast, 0.0), 1.0)
    else:
        polarity_contrast = abs_contrast

    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    boundary = boundary_mask(mask)
    edge_scale = max(float(np.percentile(gradient, 95)), 1.0)
    edge_alignment = 0.0
    if boundary.any():
        edge_alignment = min(float(gradient[boundary].mean()) / edge_scale, 1.0)

    agreement_scores = []
    for other in all_masks:
        if other is mask:
            continue
        other_bin = other > 0
        union = np.logical_or(binary, other_bin).sum()
        if union == 0:
            continue
        overlap = np.logical_and(binary, other_bin).sum() / union
        agreement_scores.append(float(overlap))

    agreement = float(np.mean(agreement_scores)) if agreement_scores else 0.0

    area_score = range_score(area_ratio, low=0.01, high=0.30, hard_low=0.001, hard_high=0.65)
    compactness_score = range_score(
        features["compactness"],
        low=0.12,
        high=0.85,
        hard_low=0.02,
        hard_high=1.0,
    )
    border_score = 1.0 - min(features["border_fraction"] * 4.0, 1.0)

    score = (
        0.25 * polarity_contrast
        + 0.10 * abs_contrast
        + 0.15 * edge_alignment
        + 0.20 * area_score
        + 0.15 * features["solidity"]
        + 0.10 * compactness_score
        + 0.05 * agreement
        + 0.05 * border_score
    )

    return {
        "score": float(score),
        "area_ratio": area_ratio,
        "contrast": abs_contrast,
        "polarity_contrast": polarity_contrast,
        "edge_alignment": edge_alignment,
        "solidity": features["solidity"],
        "compactness": features["compactness"],
        "agreement": agreement,
        "border_fraction": features["border_fraction"],
    }


def select_unsupervised_mask(image, polarity):
    components = candidate_components(image)

    if not components:
        return "none", np.zeros_like(image, dtype=np.uint8), {}

    all_masks = [mask for _, mask in components]
    scored = []

    for name, mask in components:
        metrics = score_candidate(image, mask, all_masks, polarity)
        if metrics is None:
            continue
        scored.append((metrics["score"], name, mask, metrics))

    if not scored:
        return "none", np.zeros_like(image, dtype=np.uint8), {}

    _, name, mask, metrics = max(scored, key=lambda item: item[0])
    return name, mask, metrics


def write_results(csv_path, rows):
    fieldnames = [
        "image",
        "mask",
        "selected_candidate",
        "unsupervised_score",
        "iou_after_selection",
        "area_ratio",
        "contrast",
        "polarity_contrast",
        "edge_alignment",
        "solidity",
        "compactness",
        "agreement",
        "border_fraction",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    pairs = positive_pairs(args.dataset)
    print(f"Positive image-mask pairs found: {len(pairs)}", flush=True)

    if not pairs:
        raise ValueError(f"No non-empty image-mask pairs found in {args.dataset}")

    sample_pairs = choose_pairs(pairs, args.num_images, args.seed)
    print(f"Evaluating image-mask pairs: {len(sample_pairs)}", flush=True)

    rows = []
    ious = []

    for idx, (img_path, mask_path) in enumerate(sample_pairs, start=1):
        print(f"[{idx}/{len(sample_pairs)}] {img_path.name}", flush=True)

        image = load_gray(img_path)
        gt_mask = load_gray(mask_path)
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        selected_name, selected_mask, metrics = select_unsupervised_mask(
            image,
            polarity=args.polarity,
        )
        iou = compute_iou(selected_mask, gt_mask)
        ious.append(iou)

        rows.append(
            {
                "image": str(img_path.relative_to(args.dataset)),
                "mask": str(mask_path.relative_to(args.dataset)),
                "selected_candidate": selected_name,
                "unsupervised_score": f"{metrics.get('score', 0.0):.6f}",
                "iou_after_selection": f"{iou:.6f}",
                "area_ratio": f"{metrics.get('area_ratio', 0.0):.6f}",
                "contrast": f"{metrics.get('contrast', 0.0):.6f}",
                "polarity_contrast": f"{metrics.get('polarity_contrast', 0.0):.6f}",
                "edge_alignment": f"{metrics.get('edge_alignment', 0.0):.6f}",
                "solidity": f"{metrics.get('solidity', 0.0):.6f}",
                "compactness": f"{metrics.get('compactness', 0.0):.6f}",
                "agreement": f"{metrics.get('agreement', 0.0):.6f}",
                "border_fraction": f"{metrics.get('border_fraction', 0.0):.6f}",
            }
        )

    write_results(args.output, rows)
    stats = summarize(ious)

    print("\n=== UNSUPERVISED SELECTOR EVALUATION ===")
    print(f"Mean IoU:   {stats['mean']:.6f}")
    print(f"Median IoU: {stats['median']:.6f}")
    print(f"Std IoU:    {stats['std']:.6f}")
    print(f"Min IoU:    {stats['min']:.6f}")
    print(f"Max IoU:    {stats['max']:.6f}")
    print(f"Results saved to: {args.output}")
    print("\nNote: IoU is computed only after selection, not used to choose the mask.")


if __name__ == "__main__":
    main()
