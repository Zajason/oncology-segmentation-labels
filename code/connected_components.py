import cv2
from skimage.filters import threshold_otsu

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = threshold_otsu(blur)
    binary = (blur > threshold).astype("uint8")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    if num_labels <= 1:
        return binary

    largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
    return labels == largest


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
