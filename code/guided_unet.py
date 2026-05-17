import cv2
import numpy as np

from common import crop_box, paste_crop_mask


def _load_model(config):
    checkpoint = config.get("checkpoint")
    device = config.get("device", "cpu")

    if not checkpoint:
        raise ValueError("Guided U-Net requires config['checkpoint'].")

    try:
        import torch
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError("Install torch and segmentation-models-pytorch to use guided_unet.py.") from exc

    model = smp.Unet(
        encoder_name=config.get("encoder", "resnet34"),
        encoder_weights=None,
        in_channels=4,
        classes=1,
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.to(device)
    model.eval()
    return model, torch, device


def _prepare_guided_crop(crop, size, torch, device):
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

    rgb = cv2.resize(crop[:, :, :3], (size, size), interpolation=cv2.INTER_LINEAR)
    guide = np.ones((size, size, 1), dtype=np.uint8) * 255
    stacked = np.concatenate([rgb, guide], axis=2)
    tensor = torch.from_numpy(stacked.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return tensor.to(device)


def generate_masks(image, boxes, config):
    model, torch, device = _load_model(config)
    height, width = image.shape[:2]
    size = int(config.get("size", 256))
    threshold = float(config.get("threshold", 0.5))
    masks = []

    with torch.no_grad():
        for box in boxes:
            crop, coords = crop_box(image, box, pad=int(config.get("pad", 0)))
            if crop is None:
                masks.append(np.zeros((height, width), dtype=np.uint8))
                continue

            tensor = _prepare_guided_crop(crop, size, torch, device)
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            crop_mask = (prob >= threshold).astype(np.uint8)
            masks.append(paste_crop_mask((height, width), crop_mask, coords))

    return masks
