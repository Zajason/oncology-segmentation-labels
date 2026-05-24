import argparse
from pathlib import Path

from train_utils import train_entrypoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train vanilla U-Net and write the required U0XX CSV log.")
    parser.add_argument("--experiment-id", required=True, help="Example: U040 or U041")
    parser.add_argument("--dataset", required=True, help="Example: busi or brats")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/phase1d_training"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    train_entrypoint(args, guided=False)


if __name__ == "__main__":
    main()
