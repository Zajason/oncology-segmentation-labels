import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train vanilla U-Net and write the required U0XX CSV log.")
    parser.add_argument("--experiment-id", required=True, help="Example: U040 or U041")
    parser.add_argument("--dataset", required=True, help="Example: busi or brats")
    parser.add_argument("--output", type=Path, default=Path("results/phase1d_training"))
    parser.add_argument("--epochs", type=int, default=0)
    return parser.parse_args()


def write_empty_log(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr"],
        )
        writer.writeheader()


def main():
    args = parse_args()
    log_path = args.output / f"{args.experiment_id}_{args.dataset}_unet.csv"
    write_empty_log(log_path)
    raise NotImplementedError(
        "Vanilla U-Net training must be run in Colab with GPU after dataset manifests are finalized. "
        f"Created required log shell: {log_path}"
    )


if __name__ == "__main__":
    main()
