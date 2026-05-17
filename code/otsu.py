import cv2
from skimage.filters import threshold_otsu

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = threshold_otsu(blur)
    return blur > threshold


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
