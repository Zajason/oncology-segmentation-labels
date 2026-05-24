import argparse
import csv
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from train_utils import read_manifest


EXPERIMENT_IDS = {
    ("busi", 1, "yolo11x-seg"): "T040",
    ("busi", 1, "yolo26x-seg"): "T041",
    ("busi", 2, "yolo11x-seg"): "T042",
    ("busi", 2, "yolo26x-seg"): "T043",
    ("brats", 1, "yolo11x-seg"): "T044",
    ("brats", 1, "yolo26x-seg"): "T045",
    ("brats", 2, "yolo11x-seg"): "T046",
    ("brats", 2, "yolo26x-seg"): "T047",
    ("brain_tumor", 1, "yolo11x-seg"): "T048",
    ("brain_tumor", 1, "yolo26x-seg"): "T049",
    ("brain_tumor", 2, "yolo11x-seg"): "T050",
    ("brain_tumor", 2, "yolo26x-seg"): "T051",
}

MODEL_WEIGHTS = {
    "yolo11x-seg": "yolo11x-seg.pt",
    "yolo26x-seg": "yolo26x-seg.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Phase 2 YOLO segmentation models.")
    parser.add_argument("--dataset", required=True, choices=["busi", "brats", "brain_tumor"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--rank", type=int, required=True, choices=[1, 2])
    parser.add_argument("--model", required=True, choices=sorted(MODEL_WEIGHTS))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--polygon-dir", type=Path, default=Path("results/polygon_labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2_yolo"))
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum epoch cap. Training normally stops earlier via --patience.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stop after this many epochs without validation metric improvement.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume this run from its last.pt checkpoint.")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path to resume from. Defaults to the run's weights/last.pt.",
    )
    return parser.parse_args()


def split_rows(rows, val_fraction=0.2):
    n_val = max(1, int(round(len(rows) * val_fraction)))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:] or rows
    return train_rows, val_rows


def ensure_label(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("")


def copy_split(rows, split, dataset_root, labels_root, method):
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        image_path = Path(row["image_path"])
        target_image = image_dir / image_path.name
        shutil.copy2(image_path, target_image)

        label_path = labels_root / method / f"{row['case_id']}.txt"
        target_label = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, target_label)
        else:
            ensure_label(target_label)


def write_dataset_yaml(path, dataset_root):
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: tumor",
                "",
            ]
        )
    )


def metric_or_zero(metric_obj, attr):
    value = getattr(metric_obj, attr, None)
    return float(value) if value is not None else 0.0


def find_resume_checkpoint(args, runs_dir, run_name):
    if args.resume_checkpoint is not None:
        checkpoint = args.resume_checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        return checkpoint

    candidates = [
        runs_dir / run_name / "weights" / "last.pt",
        Path("runs") / "segment" / runs_dir / run_name / "weights" / "last.pt",
    ]
    candidates.extend(Path("runs").glob(f"**/{run_name}/weights/last.pt"))

    existing = [path for path in candidates if path.exists()]
    if not existing:
        searched = "\n".join(str(path) for path in candidates[:2])
        raise FileNotFoundError(
            f"No last.pt checkpoint found for {run_name}. Searched:\n{searched}\n"
            "Pass --resume-checkpoint if the checkpoint is somewhere else."
        )

    return max(existing, key=lambda path: path.stat().st_mtime)


def completed_epochs(runs_dir, run_name, fallback):
    results_path = runs_dir / run_name / "results.csv"
    if not results_path.exists():
        results_path = Path("runs") / "segment" / runs_dir / run_name / "results.csv"
    if not results_path.exists():
        return fallback

    with results_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return fallback
    return len(rows)


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)
    rows = sorted(rows, key=lambda row: row["case_id"])
    train_rows, val_rows = split_rows(rows)

    experiment_id = EXPERIMENT_IDS[(args.dataset, args.rank, args.model)]
    dataset_root = args.output_dir / "datasets" / args.dataset / args.method
    labels_root = args.polygon_dir / args.dataset
    summary_path = args.output_dir / f"{experiment_id}_{args.dataset}_{args.method}_{args.model}.csv"
    if summary_path.exists() and not args.force:
        print(f"Skipping existing Phase 2 summary: {summary_path}")
        return

    dataset_root.mkdir(parents=True, exist_ok=True)

    copy_split(train_rows, "train", dataset_root, labels_root, args.method)
    copy_split(val_rows, "val", dataset_root, labels_root, args.method)

    data_yaml = dataset_root / "dataset.yaml"
    write_dataset_yaml(data_yaml, dataset_root)

    runs_dir = args.output_dir / "runs"
    run_name = f"{experiment_id}_{args.dataset}_{args.method}_{args.model}"

    start = time.perf_counter()
    if args.resume:
        checkpoint_path = find_resume_checkpoint(args, runs_dir, run_name)
        print(f"Resuming {run_name} from {checkpoint_path}", flush=True)
        model = YOLO(str(checkpoint_path))
        model.train(resume=True)
    else:
        model = YOLO(MODEL_WEIGHTS[args.model])
        model.train(
            data=str(data_yaml),
            task="segment",
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            seed=args.seed,
            project=str(runs_dir),
            name=run_name,
            exist_ok=True,
            verbose=True,
            workers=args.workers,
        )
    metrics = model.val(
        data=str(data_yaml),
        task="segment",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="val",
        project=str(runs_dir),
        name=f"{run_name}_val",
        exist_ok=True,
        workers=args.workers,
    )
    runtime_min = (time.perf_counter() - start) / 60.0
    actual_epochs = completed_epochs(runs_dir, run_name, args.epochs)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "model",
                "epochs",
                "B_mAP50",
                "B_mAP50_95",
                "M_mAP50",
                "M_mAP50_95",
                "runtime_min",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": args.dataset,
                "method": args.method,
                "model": args.model,
                "epochs": actual_epochs,
                "B_mAP50": f"{metric_or_zero(metrics.box, 'map50'):.6f}",
                "B_mAP50_95": f"{metric_or_zero(metrics.box, 'map'):.6f}",
                "M_mAP50": f"{metric_or_zero(metrics.seg, 'map50'):.6f}",
                "M_mAP50_95": f"{metric_or_zero(metrics.seg, 'map'):.6f}",
                "runtime_min": f"{runtime_min:.6f}",
            }
        )

    print(f"Saved Phase 2 summary: {summary_path}")


if __name__ == "__main__":
    main()
