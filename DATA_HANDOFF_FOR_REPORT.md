# Data Handoff for Report Writing

This note summarizes what data is currently available in the repository, where it lives, and what is still being generated locally. It is based on the project scope in `ONCOLOGY_TEAM_BRIEF_v2.md`.

## Project Context

The oncology project builds a pipeline for oncology image segmentation experiments:

- Datasets: BUSI breast ultrasound, BraTS 2020 brain tumor MRI, and Brain Tumor MRI from Ultralytics.
- Phase 1: generate masks with classical methods, SAM, U-Net, and Guided U-Net.
- Phase 1D: train U-Net and Guided U-Net models.
- Phase 2: train YOLO segmentation models using the top mask-generation methods.

The report should mainly use the CSV summaries in `results/`, not the raw data, model weights, or generated training folders.

## Data That Is Ready to Use

### Dataset manifests

These files describe the cases/images used by the experiments:

| File | Meaning |
|---|---|
| `manifests/busi.csv` | Full BUSI manifest used for breast ultrasound experiments. |
| `manifests/busi_sample.csv` | Small BUSI sample manifest. |
| `manifests/brats.csv` | Full BraTS manifest used for MRI experiments. |
| `manifests/brats_sample.csv` | Small BraTS sample manifest. |
| `manifests/brain_tumor.csv` | Brain Tumor manifest. |
| `manifests/brain_tumor_sample.csv` | Small Brain Tumor sample manifest. |

Important: the prepared images/masks under `manifests/prepared/` are generated data and are intentionally not meant to be pushed to GitHub.

### Phase 1 mask results

Main folder:

```text
results/phase1_masks/
```

This contains per-case and summary CSVs for BUSI, BraTS, and the completed Brain Tumor runs:

- BUSI runs: `E040` to `E051`
- BraTS runs: `E052` to `E063`
- Brain Tumor runs: `E064` to `E069`, `E074`, and `E075`

Each method has two files:

```text
E0XX_<dataset>_<method>_percase.csv
E0XX_<dataset>_<method>_summary.csv
```

Use the summary files for report tables. Use the per-case files if you need detailed distributions, examples, or case-level analysis.

The methods covered are:

```text
otsu
multi_otsu
adaptive
watershed
otsu_watershed
connected
random_walker
chan_vese
morph_gac
sam
unet
guided_unet
```

There is also a smaller earlier/sample result folder:

```text
results/phase1_masks_100/
```

Use `results/phase1_masks/` for the main report unless there is a reason to discuss the earlier 100-case run separately.

Brain Tumor note: the local Brain Tumor dataset does not contain expert segmentation masks. Brain Tumor Phase 1 runs therefore report `dice=NaN` and `iou=NaN` by design; use them as generated-label runs, not as GT mask validation.

Currently available Brain Tumor Phase 1 results:

| Experiment | Method | Cases | Runtime (s) | Dice/IoU status |
|---|---|---:|---:|---|
| `E064` | Otsu | 7200 | 59.405423 | `NaN` because no GT masks exist |
| `E065` | Multi-Otsu | 7200 | 88.625815 | `NaN` because no GT masks exist |
| `E066` | Adaptive thresholding | 7200 | 65.190279 | `NaN` because no GT masks exist |
| `E067` | Watershed | 7200 | 256.730677 | `NaN` because no GT masks exist |
| `E068` | Otsu + Watershed | 7200 | 320.374104 | `NaN` because no GT masks exist |
| `E069` | Connected components | 7200 | 60.084592 | `NaN` because no GT masks exist |
| `E074` | U-Net transferred from BraTS | 7200 | 18.482218 | `NaN` because no GT masks exist |
| `E075` | Guided U-Net transferred from BraTS | 7200 | 20.840634 | `NaN` because no GT masks exist |

### Phase 1 configs

Folder:

```text
results/configs/
```

These JSON files record the method/run configuration for the Phase 1 experiments:

```text
E040_busi_otsu.json
...
E063_brats_guided_unet.json
```

They are useful for the experimental setup section of the report.

### Phase 1D U-Net training logs

Folder:

```text
results/phase1d_training/
```

Available files:

| File | Meaning |
|---|---|
| `U040_busi_unet.csv` | Vanilla U-Net trained on BUSI. |
| `U041_brats_unet.csv` | Vanilla U-Net trained on BraTS. |
| `U042_busi_guided_unet.csv` | Guided U-Net trained on BUSI. |
| `U043_brats_guided_unet.csv` | Guided U-Net trained on BraTS. |

These CSVs have the schema:

```text
epoch, train_loss, val_loss, val_dice, val_iou, lr
```

Use them for the U-Net training curves and to report final validation Dice/IoU.

### Phase 2 YOLO summary results

Folder:

```text
results/phase2_yolo/
```

These are compact report-ready CSV summaries. They are safe to push and should be used for the Phase 2 table.

Currently available:

| File | Dataset | Method | Model | Status |
|---|---|---|---|---|
| `T040_busi_guided_unet_yolo11x-seg.csv` | BUSI | Guided U-Net | YOLOv11x-seg | Ready |
| `T040_busi_unet_yolo11x-seg.csv` | BUSI | U-Net | YOLOv11x-seg | Ready |
| `T041_busi_guided_unet_yolo26x-seg.csv` | BUSI | Guided U-Net | YOLOv26x-seg | Ready |
| `T041_busi_unet_yolo26x-seg.csv` | BUSI | U-Net | YOLOv26x-seg | Ready |
| `T042_busi_guided_unet_yolo11x-seg.csv` | BUSI | Guided U-Net | YOLOv11x-seg | Ready |
| `T042_busi_unet_yolo11x-seg.csv` | BUSI | U-Net | YOLOv11x-seg | Ready |
| `T043_busi_guided_unet_yolo26x-seg.csv` | BUSI | Guided U-Net | YOLOv26x-seg | Ready |
| `T043_busi_unet_yolo26x-seg.csv` | BUSI | U-Net | YOLOv26x-seg | Ready |
| `T044_brats_guided_unet_yolo11x-seg.csv` | BraTS | Guided U-Net | YOLOv11x-seg | Ready |
| `T045_brats_guided_unet_yolo26x-seg.csv` | BraTS | Guided U-Net | YOLOv26x-seg | Ready |
| `T046_brats_unet_yolo11x-seg.csv` | BraTS | U-Net | YOLOv11x-seg | Ready |
| `T047_brats_unet_yolo26x-seg.csv` | BraTS | U-Net | YOLOv26x-seg | Ready |
| `T048_brain_tumor_unet_yolo11x-seg.csv` | Brain Tumor | U-Net | YOLOv11x-seg | Ready |
| `T049_brain_tumor_unet_yolo26x-seg.csv` | Brain Tumor | U-Net | YOLOv26x-seg | Ready |
| `T050_brain_tumor_guided_unet_yolo11x-seg.csv` | Brain Tumor | Guided U-Net | YOLOv11x-seg | Ready |
| `T051_brain_tumor_guided_unet_yolo26x-seg.csv` | Brain Tumor | Guided U-Net | YOLOv26x-seg | Ready |

Each file has:

```text
dataset, method, model, epochs, B_mAP50, B_mAP50_95, M_mAP50, M_mAP50_95, runtime_min
```

For the report:

- `B_mAP50` and `B_mAP50_95` are box metrics.
- `M_mAP50` and `M_mAP50_95` are mask metrics.
- `runtime_min` is the training runtime in minutes.

## Data That Is Not Yet Ready

### Local YOLO run folders

Compact report-ready CSVs exist for BUSI, BraTS, and the Brain Tumor U-Net / Guided U-Net runs in `results/phase2_yolo/`. The heavier Ultralytics run folders are local-only and should not be pushed.

Known local training folders include:

```text
runs/segment/results/phase2_yolo/runs/T045_brats_guided_unet_yolo26x-seg
runs/segment/results/phase2_yolo/runs/T046_brats_unet_yolo11x-seg
runs/segment/results/phase2_yolo/runs/T047_brats_unet_yolo26x-seg
runs/segment/results/phase2_yolo/runs/T048_brain_tumor_unet_yolo11x-seg
runs/segment/results/phase2_yolo/runs/T049_brain_tumor_unet_yolo26x-seg
runs/segment/results/phase2_yolo/runs/T050_brain_tumor_guided_unet_yolo11x-seg
runs/segment/results/phase2_yolo/runs/T051_brain_tumor_guided_unet_yolo26x-seg
```

The generated YOLO datasets used by these runs are local-only and live at:

```text
results/phase2_yolo/datasets/brats/guided_unet/
results/phase2_yolo/datasets/brats/unet/
```

Their dataset YAML files are:

```text
results/phase2_yolo/datasets/brats/guided_unet/dataset.yaml
results/phase2_yolo/datasets/brats/unet/dataset.yaml
results/phase2_yolo/datasets/brain_tumor/unet/dataset.yaml
results/phase2_yolo/datasets/brain_tumor/guided_unet/dataset.yaml
```

Use the following commands from the repository root if these results need to be regenerated or finalized locally:

```bash
cd /home/zajason/dev/ehealth/eHealth
source .venv/bin/activate
python code/run_two_dataset_brief.py --device cuda --yolo-device 0 --force
```

Run without `--force` to skip work that is already present:

```bash
cd /home/zajason/dev/ehealth/eHealth
source .venv/bin/activate
python code/run_two_dataset_brief.py --device cuda --yolo-device 0
```

Manual BraTS Guided U-Net YOLOv26x training command:

```bash
.venv/bin/python code/train_yolo_phase2.py \
    --dataset brats \
    --method guided_unet \
    --rank 1 \
    --model yolo26x-seg \
    --manifest manifests/brats.csv \
    --epochs 50 \
    --imgsz 640 \
    --batch 1 \
    --device 0 \
    --workers 0
```

Resume the BraTS Guided U-Net YOLOv26x run with:

```bash
.venv/bin/python code/train_yolo_phase2.py \
    --dataset brats \
    --method guided_unet \
    --rank 1 \
    --model yolo26x-seg \
    --manifest manifests/brats.csv \
    --epochs 50 \
    --imgsz 640 \
    --batch 1 \
    --device 0 \
    --workers 0 \
    --resume
```

Manual BraTS U-Net YOLOv11x training command:

```bash
.venv/bin/python code/train_yolo_phase2.py \
    --dataset brats \
    --method unet \
    --rank 2 \
    --model yolo11x-seg \
    --manifest manifests/brats.csv \
    --epochs 50 \
    --imgsz 640 \
    --batch 1 \
    --device 0 \
    --workers 0
```

Manual BraTS U-Net YOLOv26x training command:

```bash
.venv/bin/python code/train_yolo_phase2.py \
    --dataset brats \
    --method unet \
    --rank 2 \
    --model yolo26x-seg \
    --manifest manifests/brats.csv \
    --epochs 50 \
    --imgsz 640 \
    --batch 1 \
    --device 0 \
    --workers 0
```

Resume the BraTS U-Net YOLOv26x run with:

```bash
cd /home/zajason/dev/ehealth/eHealth
source .venv/bin/activate

.venv/bin/python code/train_yolo_phase2.py \
    --dataset brats \
    --method unet \
    --rank 2 \
    --model yolo26x-seg \
    --manifest manifests/brats.csv \
    --epochs 50 \
    --imgsz 640 \
    --batch 1 \
    --device 0 \
    --workers 0 \
    --resume
```

To watch progress:

```bash
tail -f runs/segment/results/phase2_yolo/runs/T045_brats_guided_unet_yolo26x-seg/results.csv
```

### Brain Tumor Remaining Phase 1-only Method Runs

The full oncology brief includes all 12 Brain Tumor Phase 1 methods (`E064-E075`). The fast classical methods plus U-Net and Guided U-Net are ready. These slower/optional Brain Tumor method outputs are not ready yet:

| Expected experiment | Method | Current status |
|---|---|---|
| `E070_brain_tumor_random_walker_*` | Random Walker | Not generated yet |
| `E071_brain_tumor_chan_vese_*` | Chan-Vese | Not generated yet |
| `E072_brain_tumor_morph_gac_*` | Morphological GAC | Not generated yet |
| `E073_brain_tumor_sam_*` | SAM | Not generated yet |

## Local-Only Data Not Intended for GitHub

The following data exists or may exist locally, but should not be pushed because it is large/generated:

| Path | Why it is excluded |
|---|---|
| `data/` | Raw/local datasets. Too large for GitHub. |
| `manifests/prepared/` | Generated prepared images and masks. Rebuildable. |
| `results/polygon_labels/` | Generated YOLO polygon labels. Large and rebuildable. |
| `results/phase2_yolo/datasets/` | Generated YOLO datasets. Large and rebuildable. |
| `runs/` | Ultralytics training runs, logs, plots, and weights. Large. |
| `checkpoints/` | Model checkpoint files. Large. |
| `logs/` | Local run logs. Not report source data. |
| `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.engine` | Model weights/export files. Too large for normal Git. |

If a figure or result from one of these local-only folders is needed in the report, export only the final small artifact, such as a CSV, PNG plot, or table, into a report-safe folder.

## Suggested Report Tables

Use these sources:

| Report section | Source files |
|---|---|
| Dataset description | `manifests/*.csv` |
| Phase 1 mask-generation comparison | `results/phase1_masks/*_summary.csv` |
| Per-case failure/error analysis | `results/phase1_masks/*_percase.csv` |
| Method configs | `results/configs/*.json` |
| U-Net training curves | `results/phase1d_training/*.csv` |
| YOLO Phase 2 comparison | `results/phase2_yolo/*.csv` |

Recommended wording for the current state:

```text
BUSI and BraTS Phase 1 mask-generation results are available. U-Net and Guided U-Net training logs are available. Phase 2 YOLO summaries are available for BUSI and BraTS. For Brain Tumor, Otsu, Multi-Otsu, Adaptive thresholding, Watershed, Otsu + Watershed, Connected components, U-Net, and Guided U-Net Phase 1 results are available; U-Net and Guided U-Net also have YOLOv11/YOLOv26 Phase 2 summaries. Brain Tumor has no GT masks, so Dice/IoU are intentionally reported as NaN. Brain Tumor Random Walker, Chan-Vese, Morphological GAC, and SAM Phase 1 outputs are not ready.
```
