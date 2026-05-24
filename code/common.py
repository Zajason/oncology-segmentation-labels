import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


def to_uint8_gray(image):
    if image.ndim == 3:
        if image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image

    gray = np.asarray(gray, dtype=np.float32)
    finite = gray[np.isfinite(gray)]

    if finite.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)

    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())

    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)

    gray = np.clip((gray - low) / (high - low), 0.0, 1.0)
    return (gray * 255).astype(np.uint8)


def enhance_nuclei_contrast(image_crop):
    if image_crop.ndim == 2:
        contrast = image_crop
    else:
        green = image_crop[:, :, 1]
        contrast = 255 - green

    contrast = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX)
    contrast = exposure.adjust_gamma(contrast, gamma=1.2)
    return contrast.astype(np.uint8)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return [x1, y1, x2, y2]


def crop_box(image, box, pad=0):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None, [x1, y1, x2, y2]

    return image[y1:y2, x1:x2], [x1, y1, x2, y2]


def clean_binary(mask, min_size=16):
    mask = mask.astype(bool)
    mask = ndi.binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size)
    return mask.astype(np.uint8)


def select_instance_component(mask, prefer_center=True):
    mask = clean_binary(mask)
    labels = label(mask)
    props = regionprops(labels)

    if not props:
        return np.zeros(mask.shape, dtype=np.uint8)

    crop_h, crop_w = mask.shape
    center = np.array([crop_h / 2.0, crop_w / 2.0])

    best = None
    best_score = -np.inf

    for region in props:
        area_score = float(region.area)
        if prefer_center:
            dist = np.linalg.norm(np.array(region.centroid) - center)
            dist_score = 1.0 - min(dist / max(crop_h, crop_w), 1.0)
            score = area_score * (0.25 + dist_score)
        else:
            score = area_score

        if score > best_score:
            best_score = score
            best = region.label

    return (labels == best).astype(np.uint8)


def paste_crop_mask(mask_full_shape, crop_mask, crop_box_coords):
    full = np.zeros(mask_full_shape, dtype=np.uint8)
    x1, y1, x2, y2 = crop_box_coords

    if x2 <= x1 or y2 <= y1:
        return full

    crop_mask = cv2.resize(
        crop_mask.astype(np.uint8),
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_NEAREST,
    )
    full[y1:y2, x1:x2] = (crop_mask > 0).astype(np.uint8)
    return full


def generate_by_crop(image, boxes, config, crop_segmenter):
    image = np.asarray(image)
    height, width = image.shape[:2]
    pad = int(config.get("pad", 0))
    masks = []

    for box in boxes:
        crop, crop_coords = crop_box(image, box, pad=pad)

        if crop is None or crop.size == 0:
            masks.append(np.zeros((height, width), dtype=np.uint8))
            continue

        crop_mask = crop_segmenter(crop, config)
        crop_mask = select_instance_component(
            crop_mask,
            prefer_center=bool(config.get("prefer_center", True)),
        )
        masks.append(paste_crop_mask((height, width), crop_mask, crop_coords))

    return masks


def load_yolo_boxes(label_path, image_width, image_height):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                continue

            x1 = (cx - w / 2.0) * image_width
            y1 = (cy - h / 2.0) * image_height
            x2 = (cx + w / 2.0) * image_width
            y2 = (cy + h / 2.0) * image_height
            boxes.append(clamp_box([x1, y1, x2, y2], image_width, image_height))

    return boxes


def boxes_from_binary_mask(mask):
    labels = label(mask > 0)
    boxes = []

    for region in regionprops(labels):
        y1, x1, y2, x2 = region.bbox
        boxes.append([x1, y1, x2, y2])

    return boxes


def dice_iou(pred, gt):
    pred = pred > 0
    gt = gt > 0
    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    union = np.logical_or(pred, gt).sum()

    dice = 1.0 if pred_sum + gt_sum == 0 else (2.0 * intersection) / (pred_sum + gt_sum)
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def masks_to_polygons(masks, min_area=5, class_id=0):
    labels = []

    for mask in masks:
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            contour = contour.reshape(-1, 2)
            if len(contour) < 3:
                continue

            labels.append((class_id, contour.astype(float)))

    return labels


def save_yolo_polygons(path, polygons, image_width, image_height):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for class_id, contour in polygons:
            coords = []
            for x, y in contour:
                coords.append(f"{x / image_width:.6f}")
                coords.append(f"{y / image_height:.6f}")
            f.write(f"{class_id} {' '.join(coords)}\n")
