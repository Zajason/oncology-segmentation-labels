import cv2
import numpy as np
from skimage.segmentation import random_walker

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop)
    blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255.0
    low_q = float(config.get("low_percentile", 20))
    high_q = float(config.get("high_percentile", 80))
    low, high = np.percentile(blur, [low_q, high_q])

    markers = np.zeros(blur.shape, dtype=np.uint8)
    markers[blur <= low] = 1
    markers[blur >= high] = 2

    if np.all(markers > 0):
        return markers == int(config.get("target_label", 2))

    labels = random_walker(
        blur,
        markers,
        beta=float(config.get("beta", 130)),
        mode=config.get("mode", "cg_j"),
    )
    return labels == int(config.get("target_label", 2))


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
