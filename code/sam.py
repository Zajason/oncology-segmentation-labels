import numpy as np

_PREDICTOR_CACHE = {}


def _load_predictor(config):
    checkpoint = config.get("checkpoint")
    model_type = config.get("model_type", "vit_b")
    device = config.get("device", "cpu")
    cache_key = (str(checkpoint), str(model_type), str(device))

    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    if not checkpoint:
        raise ValueError("SAM requires config['checkpoint'] pointing to a SAM checkpoint.")

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError("Install Meta segment-anything to use sam.py.") from exc

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    _PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def generate_masks(image, boxes, config):
    predictor = _load_predictor(config)
    predictor.set_image(image)
    masks = []

    for box in boxes:
        box_np = np.asarray(box, dtype=np.float32)
        pred_masks, scores, _ = predictor.predict(
            box=box_np,
            multimask_output=bool(config.get("multimask_output", True)),
        )
        best_idx = int(np.argmax(scores))
        masks.append(pred_masks[best_idx].astype(np.uint8))

    return masks
