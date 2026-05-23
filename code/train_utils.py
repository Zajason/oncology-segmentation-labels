import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import boxes_from_binary_mask, crop_box


def read_manifest(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def normalize_image(image):
    return image.astype(np.float32) / 255.0


def resize_image_and_mask(image, mask, size):
    resized_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def build_crop_samples(manifest_rows, pad=0):
    samples = []

    for row in manifest_rows:
        if not row.get("mask_path"):
            continue

        image = load_rgb(row["image_path"])
        mask = load_mask(row["mask_path"])
        boxes = boxes_from_binary_mask(mask)

        if not boxes:
            continue

        for box_idx, box in enumerate(boxes):
            _, coords = crop_box(image, box, pad=pad)
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                continue

            samples.append(
                {
                    "case_id": row.get("case_id") or Path(row["image_path"]).stem,
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "coords": coords,
                    "box_idx": box_idx,
                }
            )

    return samples


def split_case_ids(rows, val_fraction, seed):
    case_ids = sorted({row.get("case_id") or Path(row["image_path"]).stem for row in rows})
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    if not case_ids:
        return set(), set()

    val_size = max(1, int(round(len(case_ids) * val_fraction)))
    val_ids = set(case_ids[:val_size])
    train_ids = set(case_ids[val_size:]) or set(case_ids)
    if not train_ids:
        train_ids = set(case_ids[val_size:])
    return train_ids, val_ids


class CropSegmentationDataset(Dataset):
    def __init__(self, samples, size, guided=False):
        self.samples = samples
        self.size = size
        self.guided = guided

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = load_rgb(sample["image_path"])
        mask = load_mask(sample["mask_path"])
        x1, y1, x2, y2 = sample["coords"]

        crop = image[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        crop, crop_mask = resize_image_and_mask(crop, crop_mask, self.size)
        crop = normalize_image(crop)

        if self.guided:
            guide = np.ones((self.size, self.size, 1), dtype=np.float32)
            crop = np.concatenate([crop, guide], axis=2)

        tensor_image = torch.from_numpy(crop.transpose(2, 0, 1)).float()
        tensor_mask = torch.from_numpy(crop_mask[None, :, :].astype(np.float32))
        return tensor_image, tensor_mask


def make_dataloaders(manifest_path, size, batch_size, val_fraction, seed, pad, guided):
    rows = read_manifest(manifest_path)
    train_ids, val_ids = split_case_ids(rows, val_fraction, seed)
    samples = build_crop_samples(rows, pad=pad)

    train_samples = [sample for sample in samples if sample["case_id"] in train_ids]
    val_samples = [sample for sample in samples if sample["case_id"] in val_ids]

    if not train_samples:
        raise ValueError("No training crop samples were built from the manifest.")
    if not val_samples:
        val_samples = train_samples[: min(len(train_samples), max(1, batch_size))]

    train_dataset = CropSegmentationDataset(train_samples, size=size, guided=guided)
    val_dataset = CropSegmentationDataset(val_samples, size=size, guided=guided)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def dice_iou_from_logits(logits, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    masks = (masks > 0.5).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    mask_sum = masks.sum(dim=(1, 2, 3))
    union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))

    dice = torch.where(pred_sum + mask_sum > 0, (2.0 * intersection + eps) / (pred_sum + mask_sum + eps), torch.ones_like(intersection))
    iou = torch.where(union > 0, (intersection + eps) / (union + eps), torch.ones_like(intersection))
    return float(dice.mean().item()), float(iou.mean().item())


@dataclass
class EpochResult:
    train_loss: float
    val_loss: float
    val_dice: float
    val_iou: float


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(mode=train)
    loss_total = 0.0
    dice_values = []
    iou_values = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            loss_total += float(loss.item()) * images.size(0)
            dice, iou = dice_iou_from_logits(logits.detach(), masks)
            dice_values.append(dice)
            iou_values.append(iou)

    mean_loss = loss_total / max(1, len(loader.dataset))
    mean_dice = float(np.mean(dice_values)) if dice_values else 0.0
    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0
    return mean_loss, mean_dice, mean_iou


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    log_path,
    checkpoint_path,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_iou = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_dice": f"{val_dice:.6f}",
                "val_iou": f"{val_iou:.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
            }
        )

        if val_iou >= best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_iou": val_iou}, checkpoint_path)

        with log_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr"],
            )
            writer.writeheader()
            writer.writerows(history)

    return history


def default_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_unet_model(in_channels):
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    )


def train_entrypoint(args, guided):
    import torch

    device = default_device(args.device)
    train_loader, val_loader = make_dataloaders(
        manifest_path=args.manifest,
        size=args.size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        pad=args.pad,
        guided=guided,
    )

    model = build_unet_model(in_channels=4 if guided else 3).to(device)
    log_path = args.output / f"{args.experiment_id}_{args.dataset}_{'guided_unet' if guided else 'unet'}.csv"
    checkpoint_path = args.checkpoint_dir / f"{args.experiment_id}_{args.dataset}_{'guided_unet' if guided else 'unet'}.pt"

    start = time.perf_counter()
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
    )
    runtime_min = (time.perf_counter() - start) / 60.0
    print(f"Saved training log: {log_path}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Runtime minutes: {runtime_min:.2f}")
