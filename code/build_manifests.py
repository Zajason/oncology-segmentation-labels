import argparse
import csv
from pathlib import Path

import cv2
import h5py
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BUSI_DIR = BASE_DIR / "data" / "Dataset_BUSI_with_GT"
DEFAULT_BRATS_DIR = (
    BASE_DIR
    / "data"
    / "archive (1)"
    / "BraTS2020_training_data"
    / "content"
    / "data"
)
DEFAULT_BRATS_METADATA = BASE_DIR / "data" / "archive (1)" / "BraTS20 Training Metadata.csv"
DEFAULT_BRAIN_TUMOR_DIR = BASE_DIR / "data" / "brain_tumor_mri_dataset"
DEFAULT_OUTPUT_DIR = BASE_DIR / "manifests"


def parse_args():
    parser = argparse.ArgumentParser(description="Build Phase 1 dataset manifests.")
    parser.add_argument("--busi-dir", type=Path, default=DEFAULT_BUSI_DIR)
    parser.add_argument("--brats-dir", type=Path, default=DEFAULT_BRATS_DIR)
    parser.add_argument("--brats-metadata", type=Path, default=DEFAULT_BRATS_METADATA)
    parser.add_argument("--brain-tumor-dir", type=Path, default=DEFAULT_BRAIN_TUMOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--brats-limit",
        type=int,
        default=200,
        help="Number of positive BraTS slices to export. Use 0 for all positive slices.",
    )
    parser.add_argument(
        "--brats-modality",
        type=int,
        default=3,
        help="MRI channel to export as PNG. Brief says use FLAIR primary; this dataset is treated as channel 3 by default.",
    )
    return parser.parse_args()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "image_path", "mask_path", "boxes_path"]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sample(path, rows, sample_size):
    if sample_size <= 0:
        return
    sample_path = path.with_name(f"{path.stem}_sample.csv")
    write_csv(sample_path, rows[: min(sample_size, len(rows))])


def normalize_to_uint8(image):
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]

    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def matching_busi_masks(image_path):
    candidates = []
    exact = image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}")

    if exact.exists():
        candidates.append(exact)

    candidates.extend(sorted(image_path.parent.glob(f"{image_path.stem}_mask_*{image_path.suffix}")))
    return candidates


def build_busi_manifest(busi_dir, output_dir, sample_size):
    prepared_mask_dir = output_dir / "prepared" / "busi_masks"
    rows = []

    for image_path in sorted(busi_dir.rglob("*.png")):
        if "_mask" in image_path.stem:
            continue

        mask_paths = matching_busi_masks(image_path)
        if not mask_paths:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        combined = np.zeros(image.shape, dtype=np.uint8)
        for mask_path in mask_paths:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            combined[mask > 0] = 255

        if not np.any(combined > 0):
            continue

        class_name = image_path.parent.name
        case_id = f"{class_name}_{image_path.stem}".replace(" ", "_").replace("(", "").replace(")", "")
        prepared_mask_path = prepared_mask_dir / class_name / f"{image_path.stem}_combined_mask.png"
        prepared_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(prepared_mask_path), combined)

        rows.append(
            {
                "case_id": case_id,
                "image_path": str(image_path.resolve()),
                "mask_path": str(prepared_mask_path.resolve()),
                "boxes_path": "",
            }
        )

    manifest_path = output_dir / "busi.csv"
    write_csv(manifest_path, rows)
    write_sample(manifest_path, rows, sample_size)
    return manifest_path, len(rows)


def positive_brats_rows(metadata_path):
    rows = []

    with metadata_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            tumor_pixels = (
                int(row["label0_pxl_cnt"])
                + int(row["label1_pxl_cnt"])
                + int(row["label2_pxl_cnt"])
            )
            if tumor_pixels > 0:
                rows.append(row)

    return rows


def build_brats_manifest(brats_dir, metadata_path, output_dir, sample_size, limit, modality):
    prepared_dir = output_dir / "prepared" / f"brats_modality_{modality}"
    image_dir = prepared_dir / "images"
    mask_dir = prepared_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = positive_brats_rows(metadata_path)
    if limit > 0:
        metadata_rows = metadata_rows[:limit]

    rows = []

    for row in metadata_rows:
        h5_path = brats_dir / Path(row["slice_path"]).name
        if not h5_path.exists():
            continue

        with h5py.File(h5_path, "r") as h5_file:
            image = h5_file["image"][()]
            mask = h5_file["mask"][()]

        if image.ndim != 3 or modality >= image.shape[-1]:
            continue

        image_uint8 = normalize_to_uint8(image[:, :, modality])
        mask_uint8 = (np.any(mask > 0, axis=-1).astype(np.uint8) * 255)

        case_id = h5_path.stem
        image_path = image_dir / f"{case_id}.png"
        mask_path = mask_dir / f"{case_id}_mask.png"
        cv2.imwrite(str(image_path), image_uint8)
        cv2.imwrite(str(mask_path), mask_uint8)

        rows.append(
            {
                "case_id": case_id,
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "boxes_path": "",
            }
        )

    manifest_path = output_dir / "brats.csv"
    write_csv(manifest_path, rows)
    write_sample(manifest_path, rows, sample_size)
    return manifest_path, len(rows)


def build_brain_tumor_manifest(brain_tumor_dir, output_dir, sample_size):
    rows = []
    extensions = {".jpg", ".jpeg", ".png"}

    for split in ["Training", "Testing"]:
        split_dir = brain_tumor_dir / split
        if not split_dir.exists():
            continue

        for image_path in sorted(split_dir.rglob("*")):
            if image_path.suffix.lower() not in extensions:
                continue

            class_name = image_path.parent.name
            case_id = f"{split.lower()}_{class_name}_{image_path.stem}"
            rows.append(
                {
                    "case_id": case_id,
                    "image_path": str(image_path.resolve()),
                    "mask_path": "",
                    "boxes_path": "",
                }
            )

    manifest_path = output_dir / "brain_tumor.csv"
    write_csv(manifest_path, rows)
    write_sample(manifest_path, rows, sample_size)
    return manifest_path, len(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    created = []

    if args.busi_dir.exists():
        created.append(("BUSI",) + build_busi_manifest(args.busi_dir, args.output_dir, args.sample_size))
    else:
        print(f"[WARN] BUSI directory not found: {args.busi_dir}")

    if args.brats_dir.exists() and args.brats_metadata.exists():
        created.append(
            ("BraTS2020",)
            + build_brats_manifest(
                args.brats_dir,
                args.brats_metadata,
                args.output_dir,
                args.sample_size,
                args.brats_limit,
                args.brats_modality,
            )
        )
    else:
        print(f"[WARN] BraTS directory or metadata not found: {args.brats_dir}, {args.brats_metadata}")

    if args.brain_tumor_dir.exists():
        created.append(
            ("Brain Tumor",)
            + build_brain_tumor_manifest(args.brain_tumor_dir, args.output_dir, args.sample_size)
        )
    else:
        print(f"[WARN] Brain Tumor directory not found: {args.brain_tumor_dir}")

    print("\n=== Manifests ===")
    for dataset, path, count in created:
        print(f"{dataset:12s} {count:5d} cases -> {path}")

    print(
        "\nNote: brain_tumor.csv has no mask_path/boxes_path because this local Kaggle copy "
        "is class-folder organized. Box labels are needed before Phase 1 box-prompted methods can run on it."
    )


if __name__ == "__main__":
    main()
