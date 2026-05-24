from pathlib import Path
import argparse
import csv
import random

import h5py
import numpy as np

from segmentation_methods import METHODS, compute_iou, evaluate_method, summarize
from unsupervised_candidate_selector import select_unsupervised_mask


BASE_DIR = Path(__file__).resolve().parent.parent
BRATS_DATA_DIR = (
    BASE_DIR
    / "data"
    / "archive (1)"
    / "BraTS2020_training_data"
    / "content"
    / "data"
)
BRATS_METADATA_PATH = BASE_DIR / "data" / "archive (1)" / "BraTS20 Training Metadata.csv"
OUTPUT_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

MODALITY_NAMES = {
    0: "modality_0",
    1: "modality_1",
    2: "modality_2",
    3: "modality_3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation methods on BraTS2020 H5 slices."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BRATS_DATA_DIR,
        help="Directory containing BraTS2020 .h5 slice files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=BRATS_METADATA_PATH,
        help="BraTS metadata CSV used to find positive tumor slices quickly.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=25,
        help="Number of positive tumor slices to sample. Use 0 for all positive slices.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling slices.",
    )
    parser.add_argument(
        "--modality",
        choices=["0", "1", "2", "3", "all"],
        default="all",
        help="MRI modality channel to evaluate, or all channels separately.",
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "unsupervised", "both"],
        default="both",
        help="Run supervised IoU benchmark, unsupervised selector evaluation, or both.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=OUTPUT_DIR / "brats2020_method_iou_summary.csv",
        help="Summary CSV for the method benchmark.",
    )
    parser.add_argument(
        "--unsupervised-output",
        type=Path,
        default=OUTPUT_DIR / "brats2020_unsupervised_selector_results.csv",
        help="Detailed CSV for the unsupervised selector.",
    )
    return parser.parse_args()


def normalize_to_uint8(image):
    image = np.asarray(image, dtype=np.float32)
    foreground = image[np.isfinite(image)]

    if foreground.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low, high = np.percentile(foreground, [1, 99])
    if high <= low:
        high = float(np.max(foreground))
        low = float(np.min(foreground))

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def load_brats_slice(path):
    with h5py.File(path, "r") as h5_file:
        image = h5_file["image"][()]
        mask = h5_file["mask"][()]

    gt_mask = (np.any(mask > 0, axis=-1)).astype(np.uint8) * 255
    return image, gt_mask


def positive_slices(data_dir, metadata_path=None):
    if metadata_path and metadata_path.exists():
        paths = []

        with metadata_path.open() as f:
            reader = csv.DictReader(f)

            for row in reader:
                tumor_pixels = (
                    int(row["label0_pxl_cnt"])
                    + int(row["label1_pxl_cnt"])
                    + int(row["label2_pxl_cnt"])
                )

                if tumor_pixels > 0:
                    paths.append(data_dir / Path(row["slice_path"]).name)

        return [path for path in paths if path.exists()]

    paths = []

    for path in sorted(data_dir.glob("*.h5")):
        with h5py.File(path, "r") as h5_file:
            mask = h5_file["mask"][()]

        if np.any(mask > 0):
            paths.append(path)

    return paths


def choose_paths(paths, num_slices, seed):
    if num_slices == 0 or num_slices >= len(paths):
        return paths

    rng = random.Random(seed)
    return rng.sample(paths, num_slices)


def modality_indices(modality_arg):
    if modality_arg == "all":
        return [0, 1, 2, 3]
    return [int(modality_arg)]


def run_benchmark(paths, modalities, output_path):
    method_ious = {
        (modality, method): []
        for modality in modalities
        for method in METHODS
    }

    for idx, path in enumerate(paths, start=1):
        print(f"[benchmark {idx}/{len(paths)}] {path.name}", flush=True)
        image_4ch, gt_mask = load_brats_slice(path)

        for modality in modalities:
            image = normalize_to_uint8(image_4ch[:, :, modality])

            for method in METHODS:
                try:
                    iou, _ = evaluate_method(method, image, gt_mask)
                except Exception as exc:
                    print(f"  [ERROR] {MODALITY_NAMES[modality]} {method}: {exc}", flush=True)
                    iou = 0.0

                method_ious[(modality, method)].append(iou)

    rows = []
    for (modality, method), values in method_ious.items():
        stats = summarize(values)
        rows.append(
            {
                "modality": MODALITY_NAMES[modality],
                "method": method,
                "mean_iou": f"{stats['mean']:.6f}",
                "median_iou": f"{stats['median']:.6f}",
                "std_iou": f"{stats['std']:.6f}",
                "min_iou": f"{stats['min']:.6f}",
                "max_iou": f"{stats['max']:.6f}",
            }
        )

    rows.sort(key=lambda row: (float(row["mean_iou"]), float(row["median_iou"])), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    fieldnames = [
        "rank",
        "modality",
        "method",
        "mean_iou",
        "median_iou",
        "std_iou",
        "min_iou",
        "max_iou",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== BraTS Method Benchmark ===")
    for row in rows[:10]:
        print(
            f"{row['rank']:>2}. {row['modality']:10s} {row['method']:22s} "
            f"mean={row['mean_iou']} median={row['median_iou']}"
        )
    print(f"Benchmark summary saved to: {output_path}")


def run_unsupervised(paths, modalities, output_path):
    rows = []
    ious = []

    for idx, path in enumerate(paths, start=1):
        print(f"[unsupervised {idx}/{len(paths)}] {path.name}", flush=True)
        image_4ch, gt_mask = load_brats_slice(path)

        best = None
        for modality in modalities:
            image = normalize_to_uint8(image_4ch[:, :, modality])
            selected_name, selected_mask, metrics = select_unsupervised_mask(
                image,
                polarity="either",
            )

            # This score is unsupervised. IoU is computed only after selection.
            candidate = (
                metrics.get("score", 0.0),
                modality,
                selected_name,
                selected_mask,
                metrics,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate

        _, modality, selected_name, selected_mask, metrics = best
        iou = compute_iou(selected_mask, gt_mask)
        ious.append(iou)

        rows.append(
            {
                "slice": path.name,
                "selected_modality": MODALITY_NAMES[modality],
                "selected_candidate": selected_name,
                "unsupervised_score": f"{metrics.get('score', 0.0):.6f}",
                "iou_after_selection": f"{iou:.6f}",
                "area_ratio": f"{metrics.get('area_ratio', 0.0):.6f}",
                "contrast": f"{metrics.get('contrast', 0.0):.6f}",
                "edge_alignment": f"{metrics.get('edge_alignment', 0.0):.6f}",
                "solidity": f"{metrics.get('solidity', 0.0):.6f}",
                "compactness": f"{metrics.get('compactness', 0.0):.6f}",
                "agreement": f"{metrics.get('agreement', 0.0):.6f}",
                "border_fraction": f"{metrics.get('border_fraction', 0.0):.6f}",
            }
        )

    fieldnames = [
        "slice",
        "selected_modality",
        "selected_candidate",
        "unsupervised_score",
        "iou_after_selection",
        "area_ratio",
        "contrast",
        "edge_alignment",
        "solidity",
        "compactness",
        "agreement",
        "border_fraction",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats = summarize(ious)
    print("\n=== BraTS Unsupervised Selector ===")
    print(f"Mean IoU:   {stats['mean']:.6f}")
    print(f"Median IoU: {stats['median']:.6f}")
    print(f"Std IoU:    {stats['std']:.6f}")
    print(f"Min IoU:    {stats['min']:.6f}")
    print(f"Max IoU:    {stats['max']:.6f}")
    print(f"Selector details saved to: {output_path}")
    print("Note: IoU is computed only after unsupervised selection.")


def main():
    args = parse_args()

    paths = positive_slices(args.data_dir, args.metadata)
    print(f"Positive BraTS slices found: {len(paths)}", flush=True)

    if not paths:
        raise ValueError(f"No positive BraTS H5 slices found in {args.data_dir}")

    selected_paths = choose_paths(paths, args.num_slices, args.seed)
    modalities = modality_indices(args.modality)

    print(f"Evaluating slices: {len(selected_paths)}", flush=True)
    print(f"Modalities: {', '.join(MODALITY_NAMES[m] for m in modalities)}", flush=True)

    if args.mode in {"benchmark", "both"}:
        run_benchmark(selected_paths, modalities, args.benchmark_output)

    if args.mode in {"unsupervised", "both"}:
        run_unsupervised(selected_paths, modalities, args.unsupervised_output)


if __name__ == "__main__":
    main()
