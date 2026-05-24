import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    h, w = gray.shape
    markers = np.zeros((h, w), dtype=np.int32)
    markers[0, :] = 1
    markers[-1, :] = 1
    markers[:, 0] = 1
    markers[:, -1] = 1

    radius = max(2, int(min(h, w) * float(config.get("center_seed_radius", 0.18))))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    markers[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 2

    labels = watershed(gradient, markers)
    mask = labels == 2
    return ndi.binary_fill_holes(mask)


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
