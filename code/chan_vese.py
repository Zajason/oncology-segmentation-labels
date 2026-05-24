import numpy as np
from skimage.segmentation import chan_vese

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop).astype(np.float32) / 255.0
    mask = chan_vese(
        gray,
        mu=float(config.get("mu", 0.25)),
        lambda1=float(config.get("lambda1", 1.0)),
        lambda2=float(config.get("lambda2", 1.0)),
        tol=float(config.get("tol", 1e-3)),
        max_num_iter=int(config.get("iterations", 200)),
        dt=float(config.get("dt", 0.5)),
        init_level_set=config.get("init_level_set", "checkerboard"),
    )
    return mask


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
