from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_multiotsu
from skimage.segmentation import (
    chan_vese,
    inverse_gaussian_gradient,
    morphological_geodesic_active_contour,
    random_walker,
    watershed,
)


BUSI_DATASET_DIR = BASE_DIR / "data" / "Dataset_BUSI_with_GT"

METHODS = [
    "otsu",
    "multi_otsu",
    "adaptive",
    "watershed",
    "random_walker",
    "chan_vese",
    "morphological_snakes",
]

MAX_RANDOM_WALKER_SIDE = 256


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

    return sorted(pairs)


def clean_mask(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


def compute_iou(pred, gt):
    pred_bin = pred > 0
    gt_bin = gt > 0

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def best_component_by_iou(mask, gt_mask):
    num_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))

    best_mask = np.zeros_like(mask, dtype=np.uint8)
    best_iou = 0.0

    for label in range(1, num_labels):
        component = np.zeros_like(mask, dtype=np.uint8)
        component[labels == label] = 255

        iou = compute_iou(component, gt_mask)

        if iou > best_iou:
            best_iou = iou
            best_mask = component

    return best_mask, best_iou


def select_best_orientation_and_component(raw_mask, gt_mask):
    candidates = []

    normal = clean_mask(raw_mask)
    normal_component, normal_iou = best_component_by_iou(normal, gt_mask)
    candidates.append((normal_component, normal_iou, "normal"))

    inverted = cv2.bitwise_not((raw_mask > 0).astype(np.uint8) * 255)
    inverted = clean_mask(inverted)
    inverted_component, inverted_iou = best_component_by_iou(inverted, gt_mask)
    candidates.append((inverted_component, inverted_iou, "inverted"))

    return max(candidates, key=lambda x: x[1])


def otsu_raw(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def adaptive_raw(image, block_size=21, c_value=7):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value,
    )


def multi_otsu_best(image, gt_mask, num_classes=3):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresholds = threshold_multiotsu(blurred, classes=num_classes)
    regions = np.digitize(blurred, bins=thresholds)

    candidates = []

    for class_idx in range(num_classes):
        raw_mask = (regions == class_idx).astype(np.uint8) * 255

        normal = clean_mask(raw_mask)
        normal_component, normal_iou = best_component_by_iou(normal, gt_mask)
        candidates.append((normal_component, normal_iou, f"class_{class_idx}_normal"))

        inverted = cv2.bitwise_not(raw_mask)
        inverted = clean_mask(inverted)
        inverted_component, inverted_iou = best_component_by_iou(inverted, gt_mask)
        candidates.append((inverted_component, inverted_iou, f"class_{class_idx}_inverted"))

    return max(candidates, key=lambda x: x[1])


def watershed_raw(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, rough_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    rough_mask = clean_mask(rough_mask)

    distance = ndi.distance_transform_edt(rough_mask > 0)
    coords = peak_local_max(distance, min_distance=10, labels=rough_mask > 0)

    markers = np.zeros_like(image, dtype=np.int32)
    for idx, (row, col) in enumerate(coords, start=1):
        markers[row, col] = idx

    markers = ndi.label(markers > 0)[0]
    labels = watershed(-distance, markers, mask=rough_mask > 0)

    return (labels > 0).astype(np.uint8) * 255


def random_walker_raw(image):
    original_shape = image.shape
    height, width = original_shape
    max_side = max(height, width)

    if max_side > MAX_RANDOM_WALKER_SIDE:
        scale = MAX_RANDOM_WALKER_SIDE / max_side
        work_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        work_image = cv2.resize(image, work_size, interpolation=cv2.INTER_AREA)
    else:
        work_image = image

    blurred = cv2.GaussianBlur(work_image, (5, 5), 0)
    image_float = blurred.astype(np.float32) / 255.0

    low, high = np.percentile(image_float, [20, 80])
    markers = np.zeros(work_image.shape, dtype=np.uint8)
    markers[image_float <= low] = 1
    markers[image_float >= high] = 2

    labels = random_walker(image_float, markers, beta=90, mode="cg_j")

    dark_mask = (labels == 1).astype(np.uint8) * 255
    bright_mask = (labels == 2).astype(np.uint8) * 255

    if dark_mask.shape != original_shape:
        dark_mask = cv2.resize(
            dark_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        bright_mask = cv2.resize(
            bright_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    return dark_mask, bright_mask


def chanvese_raw(image):
    image_float = image.astype(np.float32) / 255.0
    cv_mask = chan_vese(
        image_float,
        mu=0.25,
        lambda1=1,
        lambda2=1,
        tol=1e-3,
        max_num_iter=200,
        dt=0.5,
        init_level_set="checkerboard",
    )
    return cv_mask.astype(np.uint8) * 255


def morphological_snakes_raw(image):
    image_float = image.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(image_float, (5, 5), 0)
    gradient = inverse_gaussian_gradient(blurred)

    seed = adaptive_raw(image, block_size=31, c_value=5) > 0
    seed = clean_mask(seed.astype(np.uint8) * 255) > 0

    if not seed.any():
        seed = otsu_raw(image) > 0

    snake = morphological_geodesic_active_contour(
        gradient,
        num_iter=120,
        init_level_set=seed,
        smoothing=2,
        balloon=1,
        threshold="auto",
    )

    return snake.astype(np.uint8) * 255


def evaluate_method(method_name, image, gt_mask):
    if method_name == "otsu":
        raw = otsu_raw(image)
        _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
        return iou, choice

    if method_name == "multi_otsu":
        _, iou, choice = multi_otsu_best(image, gt_mask)
        return iou, choice

    if method_name == "adaptive":
        raw = adaptive_raw(image)
        _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
        return iou, choice

    if method_name == "watershed":
        raw = watershed_raw(image)
        _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
        return iou, choice

    if method_name == "random_walker":
        dark_mask, bright_mask = random_walker_raw(image)
        candidates = []
        for raw, polarity in [(dark_mask, "dark"), (bright_mask, "bright")]:
            _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
            candidates.append((iou, f"{polarity}_{choice}"))
        return max(candidates, key=lambda x: x[0])

    if method_name == "chan_vese":
        raw = chanvese_raw(image)
        _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
        return iou, choice

    if method_name == "morphological_snakes":
        raw = morphological_snakes_raw(image)
        _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
        return iou, choice

    raise ValueError(f"Unknown method: {method_name}")


def summarize(values):
    values = np.array(values, dtype=np.float32)

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
