from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "Dataset_BUSI_with_GT"


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def find_pairs(root):
    pairs = []
    for img_path in root.rglob("*.png"):
        if "_mask" in img_path.stem:
            continue

        mask_path = img_path.with_name(f"{img_path.stem}_mask{img_path.suffix}")
        if mask_path.exists():
            pairs.append((img_path, mask_path))

    return pairs


def adaptive_segmentation(image, block_size=21, c_value=7):
    """
    Adaptive thresholding using a local Gaussian-weighted threshold.

    block_size:
        Size of local neighborhood. Must be odd and > 1.
    c_value:
        Constant subtracted from local threshold.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value
    )

    return mask


def clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


def keep_largest_component(mask):
    num_labels, labels = cv2.connectedComponents(mask)

    if num_labels <= 1:
        return mask

    largest_label = 1
    largest_size = 0

    for label in range(1, num_labels):
        size = np.sum(labels == label)
        if size > largest_size:
            largest_size = size
            largest_label = label

    output = np.zeros_like(mask)
    output[labels == largest_label] = 255
    return output


def compute_iou(pred, gt):
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum((pred_bin + gt_bin) > 0)

    if union == 0:
        return 0.0

    return intersection / union


def visualize(image, gt, pred, iou, title=""):
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    overlay[gt > 0] = [0, 255, 0]
    overlay[pred > 0] = [255, 0, 0]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Image")

    axs[1].imshow(gt, cmap="gray")
    axs[1].set_title("Ground Truth")

    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Adaptive Mask")

    axs[3].imshow(overlay)
    axs[3].set_title(f"Overlay (IoU={iou:.2f})")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    pairs = find_pairs(DATASET_DIR)
    print(f"Found {len(pairs)} pairs")

    sample_pairs = random.sample(pairs, min(5, len(pairs)))
    ious = []

    for img_path, mask_path in sample_pairs:
        image = load_gray(img_path)
        gt_mask = load_gray(mask_path)
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        pred_mask = adaptive_segmentation(
            image,
            block_size=21,
            c_value=7
        )

        pred_mask = clean_mask(pred_mask)
        pred_mask = keep_largest_component(pred_mask)

        iou = compute_iou(pred_mask, gt_mask)
        ious.append(iou)

        visualize(
            image=image,
            gt=gt_mask,
            pred=pred_mask,
            iou=iou,
            title=img_path.name
        )

    print("\n=== RESULTS ===")
    print(f"Mean IoU (sample): {np.mean(ious):.3f}")


if __name__ == "__main__":
    main()