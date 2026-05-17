# Oncology Team — Semester Project Brief (v2)
**Official topic (NTUA):** *Αυτοματοποιημένη παραγωγή συνθετικών συνόλων δεδομένων κατάτμησης στιγμιοτύπων σε ογκολογικές ιατρικές εικόνες από δημόσια διαθέσιμα σύνολα δεδομένων*

**Lead:** Athanasios Delis · **Duration:** 12 weeks · **Platform:** Colab Pro (required for SAM/U-Net/YOLO) · **Co-authorship on IEEE Access paper conditional on full delivery**

---

## 1. WHAT YOU BUILD

A pipeline that takes 3 public oncology imaging datasets, applies **12 mask-generation methods** spanning classical computer vision, foundation models, and learned segmentation networks, converts the resulting masks into YOLO polygon labels (via `cv2.findContours`), and trains both YOLOv11 and YOLOv26 segmentation models on the generated datasets.

**This is the experimental matrix MICCAI reviewers explicitly demanded** (R1: ConnectedComponents, pure Watershed, Guided U-Net; R2: SAM, findContours). Your deliverable becomes the central experimental contribution of the IEEE Access resubmission.

---

## 2. DATASETS (3 — all oncology)

| Dataset | Modality | Pathology | Annotations | Your job |
|---|---|---|---|---|
| **BUSI** | Ultrasound | Breast lesions (normal/benign/malignant) | GT masks | Derive boxes from masks → 12 methods → compare to GT |
| **BraTS 2020** | MRI (T1/T1ce/T2/FLAIR) | Glioma | Expert voxel masks | Same trick. Use FLAIR primary. 2D axial slices. |
| **Brain Tumor (Ultralytics)** | MRI | Tumor | **Box-only** | Run methods on given boxes. No direct GT. Cross-validate by YOLO training. For U-Net family: use BraTS-trained models (domain transfer). |

---

## 3. METHODS (12 — covers MICCAI reviewer demands + formal NTUA topic)

### Classical (9 — no training needed)

| # | Method | Library | Notes |
|---|---|---|---|
| 1 | Otsu thresholding | `skimage.filters.threshold_otsu` | Single global threshold |
| 2 | Multi-Otsu thresholding | `skimage.filters.threshold_multiotsu` | 3-class default |
| 3 | Adaptive / local thresholding | `cv2.adaptiveThreshold` | Block size = 35 default |
| 4 | Pure Watershed | `skimage.segmentation.watershed` | Markers from distance transform, **no Otsu init** |
| 5 | Otsu + Watershed | combined | **Reference baseline — code provided Wednesday** |
| 6 | ConnectedComponents | `cv2.connectedComponents` | After simple binarization |
| 7 | Random Walker | `skimage.segmentation.random_walker` | β=130 default |
| 8 | Chan-Vese | `skimage.segmentation.chan_vese` | Level-set, no edges |
| 9 | Morphological GAC | `skimage.segmentation.morphological_geodesic_active_contour` | Iterations=200 |

### Foundation models (2 — pretrained, inference only)

| # | Method | Library | Notes |
|---|---|---|---|
| 10 | SAM (bbox-prompted) | `segment-anything` from Meta / HuggingFace | ViT-B checkpoint; bbox → mask |
| 10b | SAM-Med2D *(bonus)* | HuggingFace `uni-medical/SAM-Med2D` | Medical-tuned, optional extra |

### Learned (2 — require training step)

| # | Method | Library | Notes |
|---|---|---|---|
| 11 | U-Net (vanilla) | `segmentation-models-pytorch` | **Oracle baseline.** Trained on GT masks (full supervision). Represents upper bound if perfect GT existed. Brain Tumor uses BraTS-trained weights (domain transfer). |
| 12 | Guided U-Net | per Bilic & Egger 2023, IEEE 10324276 | **Box-supervised.** Takes image + bbox encoding as input. Trained on (image, bbox, GT mask) triples. |

---

## 4. REQUIRED FUNCTION SIGNATURE (STRICT — DO NOT IMPROVISE)

All 12 methods must implement this signature. Same interface = clean comparison.

```python
def generate_masks(
    image: np.ndarray,      # HxWx3 RGB or HxW grayscale
    boxes: list,            # [[x1,y1,x2,y2], ...] pixel coords, top-left + bottom-right
    config: dict            # method hyperparameters + paths to pretrained weights when needed
) -> list:
    """Return list of binary masks (HxW uint8, values 0 or 1), one per input box, same order."""
```

One file per method: `otsu.py`, `multi_otsu.py`, `adaptive.py`, `watershed.py`, `otsu_watershed.py`, `connected_components.py`, `random_walker.py`, `chan_vese.py`, `morph_gac.py`, `sam.py`, `unet.py`, `guided_unet.py`.

For U-Net family: include a separate `train_unet.py` / `train_guided_unet.py` that produces the checkpoint, and the `generate_masks` function loads the checkpoint and runs inference.

Polygon conversion is **infrastructure, not a method to compare** — handled centrally by `cv2.findContours` per the MICCAI reviewer #2 recommendation. You do not implement this.

---

## 5. EXPERIMENT IDS (STRICT — paper depends on this)

### Phase 1A — Mask generation on BUSI (E040–E051, 12 runs)

| ID | Method | | ID | Method |
|---|---|---|---|---|
| E040 | busi_otsu | | E046 | busi_random_walker |
| E041 | busi_multi_otsu | | E047 | busi_chan_vese |
| E042 | busi_adaptive | | E048 | busi_morph_gac |
| E043 | busi_watershed | | E049 | busi_sam |
| E044 | busi_otsu_watershed | | E050 | busi_unet |
| E045 | busi_connected | | E051 | busi_guided_unet |

### Phase 1B — Mask generation on BraTS 2020 (E052–E063, 12 runs)
Same 12 methods, prefix `brats_`.

### Phase 1C — Mask generation on Brain Tumor (E064–E075, 12 runs)
Same 12 methods, prefix `braintumor_`. No direct GT comparison; validation comes from Phase 2 YOLO training.

### Phase 1D — U-Net training (U040–U043, 4 runs)

| ID | Description |
|---|---|
| U040 | Train vanilla U-Net on BUSI GT masks → checkpoint used by E050 |
| U041 | Train vanilla U-Net on BraTS GT masks → checkpoint used by E062 + E074 (Brain Tumor uses this) |
| U042 | Train Guided U-Net on BUSI (image, box, GT) → checkpoint used by E051 |
| U043 | Train Guided U-Net on BraTS (image, box, GT) → checkpoint used by E063 + E075 |

### Phase 2 — YOLO training on top-2 methods per dataset (T040–T051, 12 runs)

After Phase 1, pick top-2 methods per dataset by mean IoU, then train YOLOv11x-seg + YOLOv26x-seg on each.

| ID | Dataset | Method-rank | Model |
|---|---|---|---|
| T040 | BUSI | #1 | YOLOv11x-seg |
| T041 | BUSI | #1 | YOLOv26x-seg |
| T042 | BUSI | #2 | YOLOv11x-seg |
| T043 | BUSI | #2 | YOLOv26x-seg |
| T044–T047 | BraTS | (same pattern) | |
| T048–T051 | Brain Tumor | (same pattern, test on BraTS held-out for cross-domain eval) | |

---

## 6. SHARED GOOGLE DRIVE STRUCTURE

```
ieee-access-undergrads/team_oncology/
├── code/
│   ├── otsu.py, multi_otsu.py, adaptive.py, watershed.py,
│   ├── otsu_watershed.py, connected_components.py,
│   ├── random_walker.py, chan_vese.py, morph_gac.py,
│   ├── sam.py, unet.py, guided_unet.py
│   ├── train_unet.py, train_guided_unet.py
│   └── notebook_phase1.ipynb, notebook_phase2.ipynb
├── checkpoints/             ← .pt files for U-Net / Guided U-Net
├── results/
│   ├── phase1_masks/        ← 72 CSVs (36 per-case + 36 summary)
│   ├── phase1d_training/    ← U-Net training logs
│   ├── phase2_yolo/         ← 12 YOLO training CSVs
│   └── braintumor_masks/    ← PNG masks per method
├── figures/
│   ├── busi_qualitative.png
│   ├── brats_qualitative.png
│   └── brain_tumor_qualitative.png
└── report/
    └── report.pdf
```

---

## 7. OUTPUT SCHEMAS (STRICT)

**Phase 1 — per-case CSV** (`E0XX_<dataset>_<method>_percase.csv`):
```
case_id, n_boxes, dice, iou, runtime_s
```

**Phase 1 — summary CSV** (`E0XX_<dataset>_<method>_summary.csv`):
```
dataset, method, n_cases, dice_mean, dice_std, iou_mean, iou_std, runtime_total_s
```

**Phase 1D — U-Net training log** (`U0XX_<dataset>_<arch>.csv`):
```
epoch, train_loss, val_loss, val_dice, val_iou, lr
```

**Phase 2 — YOLO training summary** (`T0XX_<dataset>_<method>_<model>.csv`):
```
dataset, method, model, epochs, B_mAP50, B_mAP50_95, M_mAP50, M_mAP50_95, runtime_min
```

**Qualitative figures** — PNG per dataset, grid: 4 example cases × (input, GT, then 12 methods). For Brain Tumor: omit GT column.

For Brain Tumor (no GT): per-case CSV has `dice=NaN, iou=NaN`, only `n_boxes, runtime_s`.

---

## 8. WEEKLY MILESTONES

| Week | Deliverable |
|---|---|
| **1 (this week)** | Read brief. Read paper + 8 method abstracts. Set up Colab Pro + shared Drive folder. Verify YOLOv11/v26 + SAM install. |
| 2 | Reference baseline (Otsu+Watershed) runs on BUSI. Numbers match ours. Implement methods 1–3. |
| 3 | Methods 4–6 (Watershed, ConnectedComponents). Run on BUSI. E040–E045 done. |
| 4 | Methods 7–9 (Random Walker, Chan-Vese, MorphGAC). Run on BUSI. E046–E048 done. **All 9 classical on BUSI complete.** |
| 5 | SAM (method 10) on BUSI. E049 done. Run all 10 methods on BraTS + Brain Tumor (20 runs, batch overnight). |
| 6 | Train vanilla U-Net on BUSI + BraTS (U040, U041). Inference (E050, E062, E074). |
| 7 | Train Guided U-Net on BUSI + BraTS (U042, U043). Inference (E051, E063, E075). **End of Phase 1 — all 36 mask runs + 4 U-Net trainings complete.** |
| 8 | Aggregate Phase 1. Pick top-2 methods per dataset by mean IoU. Generate qualitative figures. |
| 9 | Phase 2 starts: train YOLOv11 + YOLOv26 on BUSI top-2 methods. T040–T043 done. |
| 10 | Train on BraTS + Brain Tumor top-2 methods. T044–T051 done. |
| 11 | Report writing. |
| 12 | Polish + present. |

**Weekly 30-min check-in.** Day/time TBD this meeting.

---

## 9. COMPUTE NOTES (READ CAREFULLY)

- **Classical methods (1–9):** CPU-only, Colab free tier sufficient.
- **SAM (10):** GPU inference. ViT-B fits free Colab T4. ViT-L needs Pro.
- **U-Net + Guided U-Net training (U040–U043):** ~2–4 hours each on T4. **Colab Pro required** for the 24-hour sessions and faster GPUs. Lab will fund accounts.
- **YOLO Phase 2:** 12 trainings × ~45 min each on T4 = ~9 GPU-hours. Tight on free tier. Colab Pro recommended.
- BraTS is the heaviest dataset. Start its trainings overnight.
- Use `nvidia-smi` to confirm GPU at start of each training. If you get CPU instance, restart runtime.

---

## 10. DELIVERABLE CHECKLIST (end of 12 weeks)

- [ ] 12 method `.py` files implementing the required signature
- [ ] 2 training scripts (`train_unet.py`, `train_guided_unet.py`)
- [ ] 4 checkpoint files in `checkpoints/`
- [ ] 2 main Colab notebooks (Phase 1 + Phase 2)
- [ ] 72 CSVs for Phase 1 mask runs (36 per-case + 36 summary)
- [ ] 4 CSVs for Phase 1D U-Net trainings
- [ ] 12 CSVs for Phase 2 YOLO trainings
- [ ] 3 qualitative figures
- [ ] Report PDF (8–12 pages allowed for this team given scope): goal · datasets · all 12 methods (1 paragraph each) · experimental setup · Phase 1 IoU table (12×3) · Phase 1D U-Net training curves · Phase 2 mAP table · qualitative figure · discussion including: (a) classical vs foundation vs learned tradeoff, (b) where each method fails, (c) YOLOv11 vs YOLOv26 comparison, (d) cross-domain transfer for Brain Tumor · references
- [ ] All filenames exactly as specified above

---

## 11. CO-AUTHORSHIP RULES

Your name goes on the IEEE Access submission **if and only if**:
1. All Phase 1 deliverables (36 mask runs + qualitative figures + Phase 1 IoU table) complete with correct schema and names
2. Code reproduces when we run it cold from your Drive folder
3. Report quality is at the level of a research paper section

All three required. **Phase 1 alone qualifies for co-authorship** given the scope. Phase 1D (U-Net trainings) and Phase 2 (YOLO trainings) are the bonus — they make the paper much stronger and move authorship position favorably.

---

## 12. READING — week 1

Mandatory (abstracts at minimum, **methods sections for ★ items**):
1. Original MICCAI paper PDF — *provided today*
2. Otsu (1979) — Threshold selection
3. Beucher (1994) — Watershed
4. Liao & Chung (2001) — Multi-Otsu
5. Grady (2006) — Random Walker
6. Chan & Vese (2001) — Active contours
7. Marquez-Neila et al. (2014) — MorphGAC
8. ★ Kirillov et al. (2023) — *Segment Anything* (SAM)
9. ★ Ronneberger et al. (2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation*
10. ★ Bilic & Egger (2023) — *Transforming Semantic Segmentation into Instance Segmentation with a Guided U-Net* (DOI 10.1109/SMC53992.2023.10324276) — **read carefully, this is the Guided U-Net reference**
11. Bakas et al. (2017) — BraTS dataset
12. Al-Dhabyani et al. (2020) — BUSI dataset

---

## 13. WHY THIS SCOPE — MICCAI REVIEWER CONTEXT

The IEEE Access paper resubmission must address two specific MICCAI rejection reasons:

> **W1 (Reviewer #2):** "For step 1 [bbox→mask], the authors propose Otsu's thresholding together with watershed. Employing these methods in their pure form is not entirely state-of-the-art and authors should consider employing a Segment Anything (SAM) version (which also works on bounding boxes) and compare the results."

> **W2 (Reviewer #2):** "Section 2.3 describes their proposed, handcrafted, algorithm to convert pixel-level segmentations to polygon labels. Why not simply use OpenCV's findContours or other existing approaches?"

> **Reviewer #1:** "The authors should rewrite the results and compare their proposed algorithm with very popular algorithms such as ConnectedComponents (OpenCV), Watershed algorithm, Guided U-Net (https://ieeexplore.ieee.org/document/10324276)."

Your work directly answers all three demands. Methods 6 (ConnectedComponents), 4 (pure Watershed), 10 (SAM), and 12 (Guided U-Net) exist in the brief because of these specific reviewer requests. The polygon-conversion answer (W2) is handled by infrastructure (`cv2.findContours`), not by you.

---

## 14. THINGS NOT TO DO

- Do not invent your own function signature. §4 is final.
- Do not rename files or experiments. §5–§7 are final.
- Do not skip the GT-comparison validation step on BUSI and BraTS.
- Do not train U-Net on the Brain Tumor dataset (no GT). Use BraTS-trained weights instead — clearly mark as "domain transfer" in your report.
- Do not train YOLO (Phase 2) before completing Phase 1 — you don't yet know which methods are worth training on.
- Do not push to any GitHub repo other than the one shared with you.
- Do not work in silence for more than 3 days.

---

**Questions in the next 24h: directly to Thanos. After that: via the team channel.**
