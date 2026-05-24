import argparse
import csv
import json
import math
import subprocess
import sys
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "busi": {
        "manifest": BASE_DIR / "manifests" / "busi.csv",
        "phase1": [
            ("E040", "otsu"),
            ("E041", "multi_otsu"),
            ("E042", "adaptive"),
            ("E043", "watershed"),
            ("E044", "otsu_watershed"),
            ("E045", "connected"),
            ("E046", "random_walker"),
            ("E047", "chan_vese"),
            ("E048", "morph_gac"),
            ("E049", "sam"),
            ("E050", "unet"),
            ("E051", "guided_unet"),
        ],
        "train_ids": {
            "unet": "U040",
            "guided_unet": "U042",
        },
    },
    "brats": {
        "manifest": BASE_DIR / "manifests" / "brats.csv",
        "phase1": [
            ("E052", "otsu"),
            ("E053", "multi_otsu"),
            ("E054", "adaptive"),
            ("E055", "watershed"),
            ("E056", "otsu_watershed"),
            ("E057", "connected"),
            ("E058", "random_walker"),
            ("E059", "chan_vese"),
            ("E060", "morph_gac"),
            ("E061", "sam"),
            ("E062", "unet"),
            ("E063", "guided_unet"),
        ],
        "train_ids": {
            "unet": "U041",
            "guided_unet": "U043",
        },
    },
    "brain_tumor": {
        "manifest": BASE_DIR / "manifests" / "brain_tumor.csv",
        "phase1": [
            ("E064", "otsu"),
            ("E065", "multi_otsu"),
            ("E066", "adaptive"),
            ("E067", "watershed"),
            ("E068", "otsu_watershed"),
            ("E069", "connected"),
            ("E070", "random_walker"),
            ("E071", "chan_vese"),
            ("E072", "morph_gac"),
            ("E073", "sam"),
            ("E074", "unet"),
            ("E075", "guided_unet"),
        ],
        "train_ids": {
            "unet": "U041",
            "guided_unet": "U043",
        },
    },
}

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the oncology brief scope.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--unet-epochs", type=int, default=20)
    parser.add_argument(
        "--yolo-epochs",
        type=int,
        default=300,
        help="Maximum YOLO epoch cap. Early stopping normally ends runs sooner.",
    )
    parser.add_argument("--yolo-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--yolo-batch-size", type=int, default=4)
    parser.add_argument("--yolo-workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--brats-limit", type=int, default=0)
    parser.add_argument("--sam-checkpoint", type=Path, default=BASE_DIR / "checkpoints" / "sam_vit_b_01ec64.pth")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_cmd(cmd):
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def run_cmd_with_retries(command_factory, batches):
    last_error = None
    for batch in batches:
        try:
            run_cmd(command_factory(batch))
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"[WARN] Command failed with batch={batch}; retrying with a smaller YOLO batch.", flush=True)
    raise last_error


def ensure_sam_checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"Downloading SAM checkpoint to {path}", flush=True)
    urllib.request.urlretrieve(SAM_URL, path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def rebuild_manifests(brats_limit):
    run_cmd(
        [
            sys.executable,
            str(CODE_DIR / "build_manifests.py"),
            "--sample-size",
            "20",
            "--brats-limit",
            str(brats_limit),
        ]
    )


def unet_outputs_exist(experiment_id, dataset, method):
    suffix = "guided_unet" if method == "guided_unet" else "unet"
    log_path = BASE_DIR / "results" / "phase1d_training" / f"{experiment_id}_{dataset}_{suffix}.csv"
    checkpoint_path = BASE_DIR / "checkpoints" / f"{experiment_id}_{dataset}_{suffix}.pt"
    return log_path.exists() and checkpoint_path.exists()


def train_unets(args):
    for dataset_name, dataset_cfg in DATASETS.items():
        if dataset_name == "brain_tumor":
            continue
        for method, script_name in [("unet", "train_unet.py"), ("guided_unet", "train_guided_unet.py")]:
            exp_id = dataset_cfg["train_ids"][method]
            if unet_outputs_exist(exp_id, dataset_name, method) and not args.force:
                print(f"Skipping existing {exp_id}_{dataset_name}_{method} training outputs.", flush=True)
                continue

            run_cmd(
                [
                    sys.executable,
                    str(CODE_DIR / script_name),
                    "--experiment-id",
                    exp_id,
                    "--dataset",
                    dataset_name,
                    "--manifest",
                    str(dataset_cfg["manifest"]),
                    "--epochs",
                    str(args.unet_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                ]
            )


def phase1_config(dataset, method, args):
    if method == "sam":
        return {
            "checkpoint": str(args.sam_checkpoint.resolve()),
            "model_type": "vit_b",
            "device": args.device,
            "multimask_output": True,
        }

    if method == "unet":
        train_dataset = "brats" if dataset == "brain_tumor" else dataset
        return {
            "checkpoint": str(
                (
                    BASE_DIR
                    / "checkpoints"
                    / f"{DATASETS[dataset]['train_ids']['unet']}_{train_dataset}_unet.pt"
                ).resolve()
            ),
            "device": args.device,
            "size": 256,
        }

    if method == "guided_unet":
        train_dataset = "brats" if dataset == "brain_tumor" else dataset
        return {
            "checkpoint": str(
                (
                    BASE_DIR
                    / "checkpoints"
                    / f"{DATASETS[dataset]['train_ids']['guided_unet']}_{train_dataset}_guided_unet.pt"
                ).resolve()
            ),
            "device": args.device,
            "size": 256,
        }

    return {}


def phase1_outputs_exist(experiment_id, dataset, method):
    stem = BASE_DIR / "results" / "phase1_masks" / f"{experiment_id}_{dataset}_{method}"
    return stem.with_name(f"{stem.name}_summary.csv").exists() and stem.with_name(f"{stem.name}_percase.csv").exists()


def run_phase1(args):
    config_dir = BASE_DIR / "results" / "configs"

    for dataset_name, dataset_cfg in DATASETS.items():
        for experiment_id, method in dataset_cfg["phase1"]:
            if phase1_outputs_exist(experiment_id, dataset_name, method) and not args.force:
                continue

            config_path = config_dir / f"{experiment_id}_{dataset_name}_{method}.json"
            write_json(config_path, phase1_config(dataset_name, method, args))
            run_cmd(
                [
                    sys.executable,
                    str(CODE_DIR / "phase1_runner.py"),
                    "--manifest",
                    str(dataset_cfg["manifest"]),
                    "--dataset",
                    dataset_name,
                    "--method",
                    method,
                    "--experiment-id",
                    experiment_id,
                    "--config",
                    str(config_path),
                    "--write-polygons",
                ]
            )


def read_summary(dataset_name, method):
    for experiment_id, summary_method in DATASETS[dataset_name]["phase1"]:
        if summary_method != method:
            continue
        path = BASE_DIR / "results" / "phase1_masks" / f"{experiment_id}_{dataset_name}_{method}_summary.csv"
        with path.open() as f:
            row = next(csv.DictReader(f))
        return float(row["iou_mean"]), path
    raise KeyError(method)


def top_two_methods(dataset_name):
    if dataset_name == "brain_tumor":
        return top_two_transfer_methods()

    candidates = []
    for _, method in DATASETS[dataset_name]["phase1"]:
        iou_mean, _ = read_summary(dataset_name, method)
        if math.isnan(iou_mean):
            continue
        candidates.append((iou_mean, method))
    candidates.sort(reverse=True)
    return [method for _, method in candidates[:2]]


def top_two_transfer_methods():
    scores_by_method = {}
    for source_dataset in ["busi", "brats"]:
        for _, method in DATASETS[source_dataset]["phase1"]:
            iou_mean, _ = read_summary(source_dataset, method)
            if math.isnan(iou_mean):
                continue
            scores_by_method.setdefault(method, []).append(iou_mean)

    candidates = [
        (sum(scores) / len(scores), method)
        for method, scores in scores_by_method.items()
        if scores
    ]
    candidates.sort(reverse=True)
    return [method for _, method in candidates[:2]]


def yolo_batch_attempts(initial_batch):
    batches = []
    batch = max(1, initial_batch)
    while batch >= 1:
        batches.append(batch)
        if batch == 1:
            break
        batch = max(1, batch // 2)
    return batches


def run_phase2(args):
    for dataset_name, dataset_cfg in DATASETS.items():
        top_methods = top_two_methods(dataset_name)

        for rank, method in enumerate(top_methods, start=1):
            for model in ["yolo11x-seg", "yolo26x-seg"]:
                def make_cmd(batch):
                    cmd = [
                        sys.executable,
                        str(CODE_DIR / "train_yolo_phase2.py"),
                        "--dataset",
                        dataset_name,
                        "--method",
                        method,
                        "--rank",
                        str(rank),
                        "--model",
                        model,
                        "--manifest",
                        str(dataset_cfg["manifest"]),
                        "--epochs",
                        str(args.yolo_epochs),
                        "--patience",
                        str(args.yolo_patience),
                        "--imgsz",
                        str(args.imgsz),
                        "--batch",
                        str(batch),
                        "--device",
                        args.yolo_device,
                        "--workers",
                        str(args.yolo_workers),
                    ]
                    if not args.force:
                        return cmd
                    return [*cmd, "--force"]

                run_cmd_with_retries(make_cmd, yolo_batch_attempts(args.yolo_batch_size))


def main():
    args = parse_args()
    ensure_sam_checkpoint(args.sam_checkpoint)
    rebuild_manifests(args.brats_limit)
    train_unets(args)
    run_phase1(args)
    run_phase2(args)
    print("Completed oncology brief run plan.")


if __name__ == "__main__":
    main()
