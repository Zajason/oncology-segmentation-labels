import cv2
import numpy as np
from skimage.filters import threshold_multiotsu, threshold_otsu

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    classes = int(config.get("classes", 3))
    target_class = int(config.get("target_class", classes - 1))

    if np.unique(blur).size < classes:
        threshold = threshold_otsu(blur)
        return blur > threshold

    try:
        thresholds = threshold_multiotsu(blur, classes=classes)
    except ValueError:
        threshold = threshold_otsu(blur)
        return blur > threshold

    regions = np.digitize(blur, bins=thresholds)
    return regions == min(target_class, classes - 1)


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
