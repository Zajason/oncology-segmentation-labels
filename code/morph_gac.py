import cv2
import numpy as np
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop)
    if min(gray.shape) < 3:
        return gray > 0

    image_float = gray.astype(np.float32) / 255.0
    gradient = inverse_gaussian_gradient(cv2.GaussianBlur(image_float, (5, 5), 0))

    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = max(2, int(min(h, w) * float(config.get("init_radius", 0.35))))
    init = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2)

    mask = morphological_geodesic_active_contour(
        gradient,
        num_iter=int(config.get("iterations", 200)),
        init_level_set=init,
        smoothing=int(config.get("smoothing", 2)),
        threshold=config.get("threshold", "auto"),
        balloon=float(config.get("balloon", 1)),
    )
    return mask


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
