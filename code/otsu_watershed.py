import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_multiotsu, threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, disk, erosion
from skimage.segmentation import watershed

from common import enhance_nuclei_contrast, generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    if crop.ndim == 3 and bool(config.get("use_inverted_green", False)):
        gray = enhance_nuclei_contrast(crop)
    else:
        gray = to_uint8_gray(crop)

    enhanced = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = threshold_otsu(enhanced)
    mask_loose = enhanced > threshold
    mask_loose = ndi.binary_fill_holes(mask_loose)
    mask_loose = closing(mask_loose, disk(3))

    distance = ndi.distance_transform_edt(mask_loose)
    distance = gaussian(distance, sigma=1)

    try:
        thresholds = threshold_multiotsu(enhanced, classes=3)
        mask_strict = enhanced > thresholds[1]
        mask_strict = ndi.binary_fill_holes(mask_strict)
        if np.sum(mask_strict) == 0:
            mask_strict = erosion(mask_loose, disk(5))
    except ValueError:
        mask_strict = erosion(mask_loose, disk(5))

    markers = label(mask_strict)
    labels = watershed(-distance, markers, mask=mask_loose)
    return labels > 0


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
