import cv2

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block_size = int(config.get("block_size", 35))
    c_value = int(config.get("c", 5))

    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 3)

    mask = cv2.adaptiveThreshold(
        blur,
        1,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )
    return mask > 0


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
