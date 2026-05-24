# Exam Study Guide: Oncology Medical Image Segmentation App

This guide is for preparing for an exam where you may be asked to explain, extend, debug, or modify this project. It covers the theory, the app architecture, what the code does, what worked well, what did not, and the most likely questions the professor could ask.

The most important mental model:

```text
dataset manifest
-> load image and optional ground-truth mask
-> derive bounding boxes
-> call one segmentation method
-> produce binary masks
-> compare prediction against ground truth with Dice and IoU
-> optionally convert masks to YOLO polygon labels
-> write CSV results
```

This is not a web app and it does not use a backend framework. It is a plain Python experimental pipeline built from small scripts and method modules.

---

## 1. Project Goal

The assignment is about automated generation of synthetic instance segmentation datasets for oncology medical images.

In simpler words:

- We have medical images from public oncology datasets.
- Some datasets already have expert masks.
- Some datasets only have image-level classes or bounding boxes.
- We want to automatically generate segmentation masks.
- Then we convert those masks to polygon labels.
- Those polygon labels can train YOLO segmentation models.

The project tries to reduce the manual effort needed to label medical images. Manual segmentation by experts is slow, expensive, and hard to scale. The project asks whether classical image processing methods, foundation models like SAM, and learned models like U-Net can generate useful synthetic labels.

The official topic is:

```text
Automated production of synthetic instance segmentation datasets in oncology
medical images from publicly available datasets.
```

The key research question:

```text
Can we automatically produce masks and polygon labels good enough to train
segmentation models for oncology images?
```

---

## 2. Datasets

The project uses three oncology datasets.

### BUSI

BUSI is a breast ultrasound dataset.

Characteristics:

- Modality: ultrasound.
- Pathology: breast lesions.
- Classes: normal, benign, malignant.
- Has expert ground-truth masks.
- Good for direct evaluation using Dice and IoU.

In the repository:

```text
data/Dataset_BUSI_with_GT/
manifests/busi.csv
manifests/busi_sample.csv
```

The BUSI images often have noisy ultrasound texture, low contrast, shadows, and variable lesion appearance. This makes simple thresholding difficult.

### BraTS 2020

BraTS is a brain tumor MRI dataset.

Characteristics:

- Modality: MRI.
- Pathology: glioma.
- Has expert voxel masks.
- Stored as 3D data, but this project exports 2D axial slices.
- Multiple MRI modalities/channels exist.

In the repository:

```text
data/archive (1)/BraTS2020_training_data/
manifests/brats.csv
manifests/brats_sample.csv
manifests/prepared/brats_modality_3/
```

The manifest builder exports positive slices and chooses a modality. In this app, modality 3 is treated as the default primary channel.

BraTS worked better than expected for many methods because tumors can often be intensity-distinct in the selected MRI modality.

### Brain Tumor MRI Dataset

This dataset is organized by class folders.

Characteristics:

- Modality: MRI.
- Contains brain tumor images.
- In this local copy, it does not have expert segmentation masks.
- Therefore direct Dice/IoU evaluation is not possible.

In the repository:

```text
data/brain_tumor_mri_dataset/
manifests/brain_tumor.csv
manifests/brain_tumor_sample.csv
```

Important exam point:

```text
Brain Tumor Phase 1 results have Dice=NaN and IoU=NaN because there are no
ground-truth segmentation masks in the local dataset.
```

This is not a bug. It is a dataset limitation.

---

## 3. Main Code Structure

Important files:

```text
code/common.py
code/build_manifests.py
code/phase1_runner.py
code/otsu.py
code/multi_otsu.py
code/adaptive.py
code/watershed.py
code/otsu_watershed.py
code/connected_components.py
code/random_walker.py
code/chan_vese.py
code/morph_gac.py
code/sam.py
code/unet.py
code/guided_unet.py
code/train_unet.py
code/train_guided_unet.py
```

### `build_manifests.py`

This script creates CSV manifests for the datasets.

A manifest row contains:

```text
case_id,image_path,mask_path,boxes_path
```

The manifest tells the runner:

- which image to load,
- where the ground-truth mask is, if available,
- where YOLO boxes are, if available.

For BUSI, masks are combined if an image has multiple mask files.

For BraTS, the H5 data is converted into PNG images and PNG masks.

For Brain Tumor, the manifest has image paths but no mask paths, because the local dataset has no segmentation masks.

### `phase1_runner.py`

This is the central orchestration script.

It does the following:

1. Parses command-line arguments.
2. Reads optional JSON config.
3. Reads a manifest CSV.
4. Dynamically imports the requested method module.
5. Loads each image.
6. Loads the ground-truth mask if available.
7. Determines boxes from YOLO labels, ground-truth masks, or explicit row columns.
8. Calls:

```python
masks = method_module.generate_masks(image, boxes, config)
```

9. Combines all instance masks into one foreground mask.
10. Computes Dice and IoU if a ground-truth mask exists.
11. Writes a per-case CSV.
12. Writes a summary CSV.
13. Optionally converts masks to YOLO polygons.

This file is the best place to modify if the professor asks:

- add a new command-line option,
- add a new metric,
- change the CSV output schema,
- add a new method name,
- change how methods are selected,
- change where output files are saved.

### `common.py`

This contains shared helper functions used by many methods.

Important helpers:

```python
to_uint8_gray(image)
crop_box(image, box, pad=0)
generate_by_crop(image, boxes, config, crop_segmenter)
clean_binary(mask, min_size=16)
select_instance_component(mask, prefer_center=True)
paste_crop_mask(mask_full_shape, crop_mask, crop_box_coords)
load_yolo_boxes(label_path, image_width, image_height)
boxes_from_binary_mask(mask)
dice_iou(pred, gt)
masks_to_polygons(masks, min_area=5)
save_yolo_polygons(path, polygons, image_width, image_height)
```

This file is the best place to modify if the professor asks:

- add shared preprocessing,
- change binary mask cleaning,
- add a new metric,
- change polygon conversion,
- change how components are selected,
- support another annotation format.

### Method files

Each method file implements the same required interface:

```python
def generate_masks(image, boxes, config):
    ...
```

This design is extremely important. It lets the runner compare many segmentation methods without caring how each method works internally.

The method files are the best place to modify if the professor asks:

- add a new segmentation algorithm,
- change Otsu behavior,
- tune watershed markers,
- change random walker seeds,
- change active contour parameters,
- change U-Net inference threshold.

---

## 4. The Required Method Interface

Every method must implement:

```python
def generate_masks(
    image: np.ndarray,
    boxes: list,
    config: dict
) -> list:
    """Return one binary mask per input box."""
```

Input:

- `image`: full image, usually RGB or grayscale.
- `boxes`: list of boxes, each `[x1, y1, x2, y2]`.
- `config`: dictionary of hyperparameters.

Output:

- A list of masks.
- One mask per input box.
- Each mask has shape `(H, W)`, same height and width as the original image.
- Masks should be binary.
- Values are usually `0` and `1`, though some older functions may use `0` and `255`.

Why this matters:

```text
The runner only knows how to call generate_masks. If a method breaks this
contract, the whole pipeline breaks.
```

If the professor asks how to add a new method, this is the main answer.

---

## 5. Typical Classical Method Flow

Most classical methods use the same pattern:

```text
for each bounding box:
    crop the image around the box
    convert crop to grayscale
    normalize intensities
    apply segmentation algorithm
    clean mask
    select one instance component
    paste crop mask back into the full image
return all full-size masks
```

This pattern is implemented through:

```python
generate_by_crop(image, boxes, config, _segment_crop)
```

Example from `otsu.py`:

```python
import cv2
from skimage.filters import threshold_otsu

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = threshold_otsu(blur)
    return blur > threshold


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
```

This is a clean plug-in structure:

- `_segment_crop` only knows how to segment a cropped region.
- `generate_by_crop` handles the repeated box logic.

---

## 6. Important Theory

### Medical image segmentation

Segmentation means assigning a label to each pixel.

For this project, segmentation usually means:

```text
background pixel -> 0
tumor or lesion pixel -> 1
```

In medical imaging, segmentation is important because it can measure:

- tumor size,
- tumor volume,
- lesion boundary,
- disease progression,
- treatment response.

### Semantic segmentation vs instance segmentation

Semantic segmentation asks:

```text
Which pixels belong to the tumor class?
```

Instance segmentation asks:

```text
Which pixels belong to each separate tumor or lesion instance?
```

If there are multiple lesions, instance segmentation should separate them.

This project produces one mask per bounding box, so it approximates instance segmentation.

### Bounding boxes

A bounding box is:

```text
[x1, y1, x2, y2]
```

Where:

- `x1`, `y1`: top-left corner,
- `x2`, `y2`: bottom-right corner.

The box restricts the segmentation method to a smaller region. This helps because the method does not need to search the whole image.

### Binary masks

A binary mask is a 2D array:

```text
0 = background
1 = foreground
```

Sometimes masks use:

```text
0 = background
255 = foreground
```

Both are common. The code usually handles both by checking:

```python
mask > 0
```

### Ground truth

Ground truth is the expert annotation. In this project, BUSI and BraTS have ground-truth masks.

Brain Tumor does not have segmentation masks in the local copy, so it cannot be evaluated with Dice or IoU in Phase 1.

### Dice score

Dice measures overlap:

```text
Dice = 2 * intersection / (prediction area + ground truth area)
```

Dice ranges from 0 to 1.

- `1.0` means perfect overlap.
- `0.0` means no overlap.

Dice is popular in medical segmentation because it handles class imbalance better than pixel accuracy.

### IoU

IoU means Intersection over Union:

```text
IoU = intersection / union
```

IoU also ranges from 0 to 1.

- `1.0` means perfect overlap.
- `0.0` means no overlap.

IoU is stricter than Dice. For the same prediction, Dice is usually higher than IoU.

Relationship:

```text
Dice = 2 * IoU / (1 + IoU)
IoU = Dice / (2 - Dice)
```

### Pixel accuracy is not enough

Medical images often have huge background regions. If a tumor is small, a model can get high pixel accuracy by predicting mostly background.

That is why Dice and IoU are better for this project.

### Preprocessing

Preprocessing prepares the image before segmentation.

Common preprocessing in this app:

- convert RGB to grayscale,
- normalize intensity to uint8,
- blur with Gaussian blur,
- enhance contrast,
- crop around bounding box.

The app uses:

```python
to_uint8_gray(image)
cv2.GaussianBlur(...)
enhance_nuclei_contrast(...)
```

### Postprocessing

Postprocessing cleans the raw mask after segmentation.

Common postprocessing:

- fill holes,
- remove small objects,
- keep largest component,
- keep central component,
- paste crop mask back into full image.

The app uses:

```python
clean_binary(mask)
select_instance_component(mask)
paste_crop_mask(...)
```

### Polygon conversion

YOLO segmentation labels are polygons, not raw masks.

The app converts masks to polygons using:

```python
cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

Then points are normalized:

```text
x_normalized = x / image_width
y_normalized = y / image_height
```

YOLO segmentation label format:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Each coordinate is normalized to the range `[0, 1]`.

---

## 7. Segmentation Methods

### 7.1 Otsu thresholding

File:

```text
code/otsu.py
```

Theory:

Otsu chooses a global threshold from the grayscale histogram. It assumes the image has two main intensity groups:

- background,
- foreground.

It chooses the threshold that best separates those groups.

Code idea:

```python
gray = to_uint8_gray(crop)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
threshold = threshold_otsu(blur)
mask = blur > threshold
```

Strengths:

- Very simple.
- Very fast.
- No training required.
- Good baseline.

Weaknesses:

- Fails when foreground and background overlap in intensity.
- Fails with uneven illumination.
- Sensitive to noise.
- Assumes one global threshold works for the crop.

Results:

- BUSI: weak, IoU around `0.130247`.
- BraTS: much better, IoU around `0.614170`.

Exam explanation:

```text
Otsu is fast and useful as a baseline, but it is too simple for noisy ultrasound.
It worked better on BraTS because the selected MRI modality often gives stronger
tumor-background intensity contrast.
```

### 7.2 Multi-Otsu thresholding

File:

```text
code/multi_otsu.py
```

Theory:

Multi-Otsu generalizes Otsu to more than two intensity classes. Instead of one threshold, it computes multiple thresholds.

Example with 3 classes:

```text
dark pixels
medium pixels
bright pixels
```

Code idea:

```python
thresholds = threshold_multiotsu(blur, classes=classes)
regions = np.digitize(blur, bins=thresholds)
mask = regions == target_class
```

Strengths:

- More flexible than simple Otsu.
- Can separate several tissue intensity classes.
- Still fast and training-free.

Weaknesses:

- Need to choose target class.
- Tumor may not always be the brightest or darkest class.
- Can fail badly if the selected class is wrong.

Results:

- Earlier BUSI exploratory summary: best classical among initial experiments, mean IoU `0.407311`.
- Main Phase 1 BUSI run: poor, IoU around `0.052201`, likely due to target class/config mismatch.
- BraTS main Phase 1: IoU around `0.480651`.
- Focused modality experiment on BraTS found Multi-Otsu strong on modality 3, IoU `0.526456`.

Exam explanation:

```text
Multi-Otsu can work very well when the target class matches the lesion intensity,
but it is sensitive to which class is selected. A wrong target_class can make it
perform worse than simple Otsu.
```

### 7.3 Adaptive thresholding

File:

```text
code/adaptive.py
```

Theory:

Adaptive thresholding computes a local threshold for each region instead of using one threshold for the whole image.

It is useful when illumination or intensity changes across the image.

Code idea:

```python
mask = cv2.adaptiveThreshold(
    blur,
    1,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    block_size,
    c_value,
)
```

Important parameters:

- `block_size`: neighborhood size; must be odd and at least 3.
- `c`: constant subtracted from local mean/weighted mean.

Strengths:

- Handles local intensity variation.
- Very fast.
- No training required.

Weaknesses:

- Can create noisy fragmented masks.
- Sensitive to `block_size` and `c`.
- May oversegment texture.

Results:

- BUSI: moderate, IoU around `0.457421`.
- BraTS: strong classical result, IoU around `0.637753`.

Exam explanation:

```text
Adaptive thresholding worked better than global thresholding when local contrast
varied. It was especially competitive on BraTS because it was fast and produced
reasonable masks without training.
```

### 7.4 Watershed

File:

```text
code/watershed.py
```

Theory:

Watershed treats the image as a topographic surface. Bright/dark regions are interpreted like hills and valleys. Markers are placed, and regions grow from those markers until they meet at boundaries.

In this implementation:

- image gradient is computed,
- border marker represents background,
- center circular marker represents foreground,
- watershed grows the center region.

Code idea:

```python
gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
markers[border] = 1
markers[center_circle] = 2
labels = watershed(gradient, markers)
mask = labels == 2
```

Strengths:

- Good for separating regions with boundaries.
- Marker-based version can be controlled.
- No training required.

Weaknesses:

- Sensitive to marker placement.
- Can undersegment or oversegment.
- If lesion is not central in the crop, center seed can be wrong.

Results:

- BUSI: good classical result, IoU around `0.625035`.
- BraTS: weaker than many alternatives, IoU around `0.498786`.

Exam explanation:

```text
Watershed depends heavily on marker design. In this app, the foreground marker is
central, so it assumes the object is near the center of the crop.
```

### 7.5 Otsu + Watershed

File:

```text
code/otsu_watershed.py
```

Theory:

This combines thresholding and watershed:

- Otsu creates a loose foreground region.
- Multi-Otsu or erosion creates stricter markers.
- Distance transform helps split the object.
- Watershed refines the mask.

Code idea:

```python
threshold = threshold_otsu(enhanced)
mask_loose = enhanced > threshold
distance = ndi.distance_transform_edt(mask_loose)
markers = label(mask_strict)
labels = watershed(-distance, markers, mask=mask_loose)
```

Strengths:

- More structured than raw thresholding.
- Can refine connected blobs.
- Common classical segmentation baseline.

Weaknesses:

- Depends on the quality of the initial threshold.
- Can fail if thresholding is wrong.
- More parameters and assumptions.

Results:

- BUSI: poor in main run, IoU around `0.150619`.
- BraTS: good, IoU around `0.632185`.

Exam explanation:

```text
The method is only as good as the initial threshold and markers. It can improve
thresholding, but it cannot rescue a bad foreground estimate.
```

### 7.6 Connected components

File:

```text
code/connected_components.py
```

Theory:

Connected components labels separate blobs in a binary image. Pixels connected to each other form one component.

In this app:

- threshold the crop,
- find connected components,
- keep the largest component.

Code idea:

```python
binary = (blur > threshold).astype("uint8")
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
largest = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
mask = labels == largest
```

Strengths:

- Simple and fast.
- Removes small noisy objects.
- Good postprocessing step.

Weaknesses:

- Needs a good binary mask first.
- Largest component might not be the lesion.
- If lesion is split into pieces, it may keep only part of it.

Results:

- BUSI: weak, IoU around `0.129881`.
- BraTS: similar to Otsu, IoU around `0.614009`.

Exam explanation:

```text
Connected components is not really a full segmentation method by itself here.
It is thresholding plus blob selection.
```

### 7.7 Random Walker

File:

```text
code/random_walker.py
```

Theory:

Random Walker is graph-based segmentation. Pixels are graph nodes. Edges connect neighboring pixels. Seeds are assigned labels, and unlabeled pixels are assigned based on probabilities of reaching each seed.

In this app:

- low-intensity pixels are seed label 1,
- high-intensity pixels are seed label 2,
- random walker assigns the rest.

Code idea:

```python
low, high = np.percentile(blur, [low_q, high_q])
markers[blur <= low] = 1
markers[blur >= high] = 2
labels = random_walker(blur, markers, beta=130)
mask = labels == target_label
```

Important parameters:

- `low_percentile`
- `high_percentile`
- `beta`
- `target_label`

Strengths:

- More spatially coherent than raw thresholding.
- Can work when seeds are reliable.

Weaknesses:

- Slower than thresholding.
- Seed choice matters a lot.
- May fail if lesion intensity is not consistently low or high.

Results:

- BUSI: weak/moderate, IoU around `0.281784`.
- BraTS: good, IoU around `0.614816`.

Exam explanation:

```text
Random Walker depends on seed quality. Percentile seeds are automatic but not
always medically meaningful.
```

### 7.8 Chan-Vese

File:

```text
code/chan_vese.py
```

Theory:

Chan-Vese is an active contour method. It evolves a curve to split the image into inside and outside regions based on intensity homogeneity.

Unlike edge-based methods, it does not require strong edges.

Code idea:

```python
mask = chan_vese(
    gray,
    mu=0.25,
    lambda1=1.0,
    lambda2=1.0,
    max_num_iter=200,
)
```

Important parameters:

- `mu`: smoothness penalty.
- `lambda1`, `lambda2`: inside/outside fitting weights.
- `iterations`: number of contour evolution steps.
- `init_level_set`: initial contour.

Strengths:

- Can segment weak boundaries.
- Good for region-based objects.

Weaknesses:

- Slower than thresholding.
- Sensitive to initialization and parameters.
- Can leak into wrong homogeneous regions.

Results:

- BUSI: good, IoU around `0.701530`.
- BraTS: moderate, IoU around `0.599382`.

Exam explanation:

```text
Chan-Vese is useful when edges are weak, but it is slower and parameter-sensitive.
```

### 7.9 Morphological Geodesic Active Contour / Morphological Snakes

File:

```text
code/morph_gac.py
```

Theory:

Morphological GAC evolves a contour toward image boundaries using morphological operations. It is related to active contours but implemented with morphological updates.

In this app:

- image is normalized,
- inverse Gaussian gradient is computed,
- an initial circular mask is placed in the crop center,
- contour evolves for a fixed number of iterations.

Code idea:

```python
gradient = inverse_gaussian_gradient(...)
init = center_circle
mask = morphological_geodesic_active_contour(
    gradient,
    num_iter=iterations,
    init_level_set=init,
    smoothing=smoothing,
    balloon=balloon,
)
```

Important parameters:

- `init_radius`
- `iterations`
- `smoothing`
- `threshold`
- `balloon`

Strengths:

- Can capture smooth object boundaries.
- Useful when there are meaningful edges.

Weaknesses:

- Slow.
- Sensitive to initialization.
- Can fail when boundaries are weak or noisy.

Results:

- BUSI: strong classical result, IoU around `0.746727`.
- BraTS: moderate, IoU around `0.604391`.

Exam explanation:

```text
MorphGAC worked well on BUSI but was expensive. It is a good example of the
accuracy/runtime tradeoff.
```

### 7.10 SAM

File:

```text
code/sam.py
```

Theory:

SAM, Segment Anything Model, is a foundation model for segmentation. It can use prompts such as points or boxes.

In this app:

- the full image is sent to the SAM predictor,
- each bounding box is passed as a prompt,
- SAM returns multiple masks if `multimask_output=True`,
- the mask with the best score is selected.

Code idea:

```python
predictor.set_image(image)
pred_masks, scores, _ = predictor.predict(
    box=box_np,
    multimask_output=True,
)
best_idx = np.argmax(scores)
mask = pred_masks[best_idx]
```

Strengths:

- Strong general segmentation model.
- Uses box prompts naturally.
- No task-specific training required.

Weaknesses:

- Requires checkpoint and heavier dependencies.
- Slower than classical methods.
- General SAM is not always optimal for medical images.
- Can segment visually salient regions that are not the true lesion.

Results:

- BUSI: strong, IoU around `0.776696`.
- BraTS: good, IoU around `0.657569`.

Exam explanation:

```text
SAM was strong without training, but learned medical-specific models still beat
it on BUSI and BraTS.
```

### 7.11 U-Net

Files:

```text
code/unet.py
code/train_unet.py
```

Theory:

U-Net is a convolutional neural network architecture designed for biomedical image segmentation.

It has:

- encoder path: extracts features while reducing spatial resolution,
- decoder path: upsamples features back to image resolution,
- skip connections: combine low-level spatial detail with high-level semantic features.

In this app:

- U-Net is trained with ground-truth masks.
- During inference, each crop is resized.
- Model predicts logits.
- Sigmoid converts logits to probabilities.
- Threshold converts probabilities to binary mask.

Code idea:

```python
logits = model(tensor)
prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
crop_mask = prob >= threshold
```

Strengths:

- Best or near-best performance.
- Learns dataset-specific patterns.
- Strong biomedical segmentation baseline.

Weaknesses:

- Requires training data.
- Requires GPU for practical training.
- May not transfer perfectly across datasets.
- More complex than classical methods.

Results:

- BUSI Phase 1: best method, IoU around `0.908938`.
- BraTS Phase 1: second best, IoU around `0.850813`.
- Brain Tumor Phase 1: no direct IoU because no ground truth.

Training logs:

- BUSI U-Net validation Dice `0.926030`, validation IoU `0.865527`.
- BraTS U-Net validation Dice `0.884428`, validation IoU `0.812515`.

Exam explanation:

```text
U-Net is the supervised upper-bound style baseline. It performed best because it
learned from expert masks, but it requires training and labeled data.
```

### 7.12 Guided U-Net

Files:

```text
code/guided_unet.py
code/train_guided_unet.py
```

Theory:

Guided U-Net is a U-Net variant that receives an additional guide channel. The guide encodes the bounding box or prompt region.

In this implementation, the crop itself already comes from the box, and the fourth channel is a guide channel filled with ones.

The model has:

```python
in_channels=4
```

instead of the usual 3 RGB channels.

Strengths:

- Uses box information.
- Strong learned method.
- Helps connect detection/bounding-box supervision with segmentation.

Weaknesses:

- Requires training.
- Current implementation's guide channel is simple.
- Can be similar to vanilla U-Net when the crop already encodes the box.

Results:

- BUSI Phase 1: second best, IoU around `0.895743`.
- BraTS Phase 1: best, IoU around `0.855520`.
- Brain Tumor Phase 1: no direct IoU because no ground truth.

Training logs:

- BUSI Guided U-Net validation Dice `0.926498`, validation IoU `0.866979`.
- BraTS Guided U-Net validation Dice `0.882895`, validation IoU `0.810606`.

Exam explanation:

```text
Guided U-Net uses an additional input channel to encode guidance. It performed
very well, especially on BraTS, because it combines learned segmentation with
box-guided localization.
```

---

## 8. What Worked Well

### BUSI

Main Phase 1 BUSI ranking by IoU:

| Rank | Method | Dice | IoU | Runtime seconds |
|---:|---|---:|---:|---:|
| 1 | U-Net | 0.951680 | 0.908938 | 121.140454 |
| 2 | Guided U-Net | 0.944251 | 0.895743 | 108.087600 |
| 3 | SAM | 0.867883 | 0.776696 | 241.023725 |
| 4 | MorphGAC | 0.849558 | 0.746727 | 329.372099 |
| 5 | Chan-Vese | 0.814392 | 0.701530 | 86.734579 |
| 6 | Watershed | 0.753785 | 0.625035 | 4.838637 |
| 7 | Adaptive | 0.597573 | 0.457421 | 1.598693 |
| 8 | Random Walker | 0.426522 | 0.281784 | 37.171441 |
| 9 | Otsu + Watershed | 0.244835 | 0.150619 | 5.212907 |
| 10 | Otsu | 0.216465 | 0.130247 | 1.435005 |
| 11 | Connected | 0.215793 | 0.129881 | 1.332782 |
| 12 | Multi-Otsu | 0.094297 | 0.052201 | 2.078644 |

What worked:

- U-Net and Guided U-Net worked best because they learned from the dataset.
- SAM worked well without training.
- MorphGAC and Chan-Vese were strong classical methods.
- Watershed was a surprisingly efficient classical method.

What did not work:

- Simple Otsu was weak on BUSI.
- Connected components was weak because it depends on Otsu first.
- Multi-Otsu performed poorly in the main run, probably because the chosen target class did not consistently match lesions.

Why BUSI is hard:

- Ultrasound has speckle noise.
- Lesions can have fuzzy boundaries.
- Intensity varies across images.
- Shadows and artifacts confuse threshold methods.

### BraTS

Main Phase 1 BraTS ranking by IoU:

| Rank | Method | Dice | IoU | Runtime seconds |
|---:|---|---:|---:|---:|
| 1 | Guided U-Net | 0.919473 | 0.855520 | 4299.700942 |
| 2 | U-Net | 0.916652 | 0.850813 | 4317.352481 |
| 3 | SAM | 0.783616 | 0.657569 | 9249.151177 |
| 4 | Adaptive | 0.756547 | 0.637753 | 19.029927 |
| 5 | Otsu + Watershed | 0.750059 | 0.632185 | 67.206084 |
| 6 | Random Walker | 0.740793 | 0.614816 | 111.357892 |
| 7 | Otsu | 0.731411 | 0.614170 | 18.758561 |
| 8 | Connected | 0.731185 | 0.614009 | 19.023352 |
| 9 | MorphGAC | 0.726790 | 0.604391 | 2092.406312 |
| 10 | Chan-Vese | 0.711733 | 0.599382 | 870.887754 |
| 11 | Watershed | 0.627422 | 0.498786 | 34.558551 |
| 12 | Multi-Otsu | 0.607739 | 0.480651 | 28.113604 |

What worked:

- Guided U-Net and U-Net were clearly strongest.
- SAM worked well but was slower.
- Adaptive, Otsu + Watershed, Random Walker, and Otsu were strong classical baselines.
- Classical methods were much better on BraTS than on BUSI.

What did not work as well:

- Pure Watershed was weaker than expected.
- Multi-Otsu was not top in the main run, even though modality-focused experiments showed it can be strong.
- MorphGAC was slow relative to its performance.

Why BraTS worked better:

- MRI modalities can provide clearer intensity contrast.
- Tumors in selected channels may be more separable.
- The exported positive slices contain actual tumor pixels.

### Brain Tumor

Brain Tumor Phase 1 cannot be ranked by IoU because the local dataset does not have expert segmentation masks.

Available generated-label runs:

| Method | Cases | Dice/IoU | Runtime seconds |
|---|---:|---|---:|
| Otsu | 7200 | NaN | 59.405423 |
| Multi-Otsu | 7200 | NaN | 88.625815 |
| Adaptive | 7200 | NaN | 65.190279 |
| Watershed | 7200 | NaN | 256.730677 |
| Otsu + Watershed | 7200 | NaN | 320.374104 |
| Connected | 7200 | NaN | 60.084592 |
| SAM | 7200 | NaN | 586.923909 |
| U-Net | 7200 | NaN | 18.482218 |
| Guided U-Net | 7200 | NaN | 20.840634 |

Exam explanation:

```text
Brain Tumor results are useful for generated synthetic labels and downstream
YOLO training, but not for direct Phase 1 Dice/IoU evaluation.
```

---

## 9. Phase 2 YOLO Results

Phase 2 trains YOLO segmentation models on generated polygon labels.

Metrics:

- `B_mAP50`: bounding-box mAP at IoU 0.50.
- `B_mAP50_95`: bounding-box mAP averaged from IoU 0.50 to 0.95.
- `M_mAP50`: mask mAP at IoU 0.50.
- `M_mAP50_95`: mask mAP averaged from IoU 0.50 to 0.95.

Important idea:

```text
Phase 1 evaluates mask generation against ground truth.
Phase 2 evaluates whether generated labels are useful for training YOLO.
```

Notable Phase 2 observations:

- BUSI YOLO results were strong for both U-Net and Guided U-Net synthetic labels.
- BraTS YOLO results were lower than BUSI, likely because the dataset is larger/more complex and MRI slices vary.
- Brain Tumor U-Net-generated labels trained YOLO much better than Guided U-Net-generated labels in the available runs.

Examples:

| Dataset | Method | Model | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---:|---:|
| BUSI | Guided U-Net | YOLOv26x-seg | 0.887873 | 0.660577 |
| BUSI | U-Net | YOLOv26x-seg | 0.883224 | 0.676730 |
| BraTS | Guided U-Net | YOLOv11x-seg | 0.617187 | 0.386274 |
| BraTS | U-Net | YOLOv11x-seg | 0.601240 | 0.387790 |
| Brain Tumor | U-Net | YOLOv11x-seg | 0.757092 | 0.580918 |
| Brain Tumor | Guided U-Net | YOLOv11x-seg | 0.434062 | 0.274108 |

Possible explanation:

```text
Good Phase 1 masks usually help Phase 2, but downstream YOLO performance also
depends on dataset size, label quality, object consistency, model capacity,
training schedule, and domain transfer.
```

---

## 10. How To Run The App

### Build manifests

```bash
.venv/bin/python code/build_manifests.py
```

This creates:

```text
manifests/busi.csv
manifests/brats.csv
manifests/brain_tumor.csv
```

### Run one method on a sample

```bash
.venv/bin/python code/phase1_runner.py \
  --manifest manifests/busi_sample.csv \
  --dataset busi \
  --method otsu \
  --experiment-id E040
```

### Run one method and write YOLO polygons

```bash
.venv/bin/python code/phase1_runner.py \
  --manifest manifests/busi_sample.csv \
  --dataset busi \
  --method otsu \
  --experiment-id E040 \
  --write-polygons
```

### Run with config

```bash
.venv/bin/python code/phase1_runner.py \
  --manifest manifests/busi_sample.csv \
  --dataset busi \
  --method adaptive \
  --experiment-id E999 \
  --config results/configs/E042_busi_adaptive.json
```

### Output files

Per-case CSV:

```text
results/phase1_masks/E040_busi_otsu_percase.csv
```

Summary CSV:

```text
results/phase1_masks/E040_busi_otsu_summary.csv
```

Polygon labels:

```text
results/polygon_labels/<dataset>/<method>/<case_id>.txt
```

---

## 11. How To Add A New Segmentation Method

This is one of the most likely exam tasks.

Suppose the professor asks:

```text
Add a fixed threshold segmentation method.
```

Step 1: create a new file:

```text
code/fixed_threshold.py
```

Step 2: implement the method:

```python
import cv2

from common import generate_by_crop, to_uint8_gray


def _segment_crop(crop, config):
    gray = to_uint8_gray(crop)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = int(config.get("threshold", 128))
    return blur > threshold


def generate_masks(image, boxes, config):
    return generate_by_crop(image, boxes, config, _segment_crop)
```

Step 3: register it in `phase1_runner.py`:

```python
METHOD_MODULES = {
    ...
    "fixed_threshold": "fixed_threshold",
}
```

Step 4: run it:

```bash
.venv/bin/python code/phase1_runner.py \
  --manifest manifests/busi_sample.csv \
  --dataset busi \
  --method fixed_threshold \
  --experiment-id E999
```

What to say in the exam:

```text
I keep the same generate_masks interface so the central runner does not need to
know the internal algorithm. I use generate_by_crop to reuse the existing crop,
component selection, and paste-back logic.
```

---

## 12. How To Add A New Metric

Another likely exam task:

```text
Add precision or recall to the per-case CSV and summary CSV.
```

Step 1: add function to `common.py`.

Example precision:

```python
def precision_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return 1.0 if tp + fp == 0 else float(tp / (tp + fp))
```

Example recall:

```python
def recall_score(pred, gt):
    pred = pred > 0
    gt = gt > 0
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return 1.0 if tp + fn == 0 else float(tp / (tp + fn))
```

Step 2: import it in `phase1_runner.py`.

```python
from common import dice_iou, precision_score, recall_score
```

Step 3: compute it near Dice/IoU:

```python
if gt_mask is not None:
    dice, iou = dice_iou(pred_mask, gt_mask)
    precision = precision_score(pred_mask, gt_mask)
    recall = recall_score(pred_mask, gt_mask)
else:
    dice, iou = math.nan, math.nan
    precision, recall = math.nan, math.nan
```

Step 4: add to `percase_rows`.

Step 5: add to CSV fieldnames.

Step 6: compute mean/std for the summary.

What to say in the exam:

```text
Metrics belong in the central runner/common layer, not inside individual methods,
because all methods should be evaluated consistently.
```

---

## 13. How To Add A Config Parameter

If asked to make something configurable, use the existing `config` dictionary.

Example:

```python
block_size = int(config.get("block_size", 35))
```

Then a JSON config can override it:

```json
{
  "block_size": 51,
  "c": 7,
  "pad": 8,
  "prefer_center": true
}
```

Good exam answer:

```text
I would not hardcode the parameter. I would read it from config with a safe
default, so existing runs continue to work.
```

---

## 14. How To Debug A Bad Method

If a method returns bad masks, check these in order:

1. Is the image loaded correctly?
2. Is the image RGB or grayscale?
3. Are boxes valid?
4. Are boxes clamped to image boundaries?
5. Does the crop contain the lesion?
6. Is the threshold selecting foreground or background?
7. Is the mask inverted?
8. Are small objects removed too aggressively?
9. Is the selected component really the lesion?
10. Is the mask pasted back into the right coordinates?
11. Is the output mask full image size?
12. Are mask values binary?
13. Is ground truth loaded correctly?
14. Is IoU low because prediction is wrong or because GT/mask alignment is wrong?

Common debugging print:

```python
print(image.shape, len(boxes), masks[0].shape, masks[0].dtype, masks[0].sum())
```

Common visual debugging:

- Save input crop.
- Save raw mask.
- Save cleaned mask.
- Overlay mask on image.

Good exam answer:

```text
I would debug the pipeline stage by stage: image loading, box generation, crop,
raw segmentation, postprocessing, paste-back, and metric calculation.
```

---

## 15. Likely Professor Questions And Answers

### Q1. What is the purpose of this project?

Answer:

```text
The purpose is to automatically generate segmentation masks and polygon labels
for oncology medical images, reducing the need for manual expert annotation.
The generated labels can then be used to train YOLO segmentation models.
```

### Q2. What is the central interface every method must implement?

Answer:

```python
def generate_masks(image, boxes, config):
    return masks
```

It returns one binary full-size mask per input box.

### Q3. Why is the common interface useful?

Answer:

```text
It allows the central runner to compare many methods fairly. The runner does not
need to know whether a method uses Otsu, Watershed, SAM, or U-Net. It only calls
generate_masks and evaluates the returned masks.
```

### Q4. What does `phase1_runner.py` do?

Answer:

```text
It reads a manifest, loads images and masks, derives boxes, imports the selected
method, calls generate_masks, combines masks, computes Dice and IoU, writes
per-case and summary CSVs, and optionally writes YOLO polygon labels.
```

### Q5. What does `common.py` do?

Answer:

```text
It contains shared utilities for image conversion, cropping boxes, cleaning
masks, selecting components, computing metrics, converting masks to polygons,
and saving YOLO labels.
```

### Q6. What is Dice?

Answer:

```text
Dice is 2 times the intersection divided by the sum of prediction area and
ground-truth area. It measures overlap and is common in medical segmentation.
```

Formula:

```text
Dice = 2TP / (2TP + FP + FN)
```

### Q7. What is IoU?

Answer:

```text
IoU is intersection over union. It measures how much the predicted mask overlaps
with the ground-truth mask compared to their combined area.
```

Formula:

```text
IoU = TP / (TP + FP + FN)
```

### Q8. Why are Dice and IoU better than accuracy here?

Answer:

```text
Because medical images often have huge background areas. A model can get high
pixel accuracy by predicting background everywhere, but Dice and IoU focus on
foreground overlap.
```

### Q9. Why does Brain Tumor have NaN Dice and IoU?

Answer:

```text
The local Brain Tumor dataset does not include expert segmentation masks, so
there is no ground truth to compare against. The generated masks can still be
used as synthetic labels, but Phase 1 direct evaluation is not possible.
```

### Q10. Which method worked best on BUSI?

Answer:

```text
U-Net worked best on BUSI, with Dice about 0.951680 and IoU about 0.908938.
Guided U-Net was second, and SAM was third.
```

### Q11. Which method worked best on BraTS?

Answer:

```text
Guided U-Net worked best on BraTS, with Dice about 0.919473 and IoU about
0.855520. U-Net was very close behind.
```

### Q12. Why did U-Net and Guided U-Net perform best?

Answer:

```text
They are supervised learned methods trained on ground-truth masks, so they learn
dataset-specific visual patterns. Classical methods use fixed assumptions about
intensity, shape, or boundaries.
```

### Q13. Why keep classical methods if U-Net is better?

Answer:

```text
Classical methods are fast, interpretable, require no training data, and provide
baselines. They are useful when labeled data or GPU resources are limited.
```

### Q14. Why did Otsu perform poorly on BUSI?

Answer:

```text
BUSI ultrasound images are noisy and lesions do not always separate cleanly by
global intensity. Otsu assumes a single global threshold can separate foreground
and background, which is often false for ultrasound.
```

### Q15. Why did Otsu work better on BraTS?

Answer:

```text
The selected MRI modality often provides better intensity contrast between tumor
and surrounding tissue, so a global threshold can be more meaningful.
```

### Q16. What is the difference between Otsu and adaptive thresholding?

Answer:

```text
Otsu uses one global threshold for the whole crop. Adaptive thresholding computes
local thresholds in neighborhoods, so it can handle varying intensity or
illumination.
```

### Q17. What is the difference between Otsu and Multi-Otsu?

Answer:

```text
Otsu separates the image into two classes using one threshold. Multi-Otsu uses
multiple thresholds to divide the image into more than two intensity classes.
```

### Q18. Why can Multi-Otsu fail?

Answer:

```text
It requires choosing which intensity class corresponds to the lesion. If the
lesion is not consistently in that class, the method selects the wrong pixels.
```

### Q19. What does connected components do?

Answer:

```text
It labels separate blobs in a binary mask. This app uses it to keep the largest
blob after thresholding.
```

### Q20. Why is connected components not enough by itself?

Answer:

```text
It needs a binary mask first. If thresholding is bad, connected components only
selects the largest wrong blob.
```

### Q21. How does watershed work?

Answer:

```text
Watershed treats the image like a topographic surface and grows regions from
markers. Boundaries form where growing regions meet.
```

### Q22. What is the weakness of watershed?

Answer:

```text
It depends heavily on marker placement and gradient quality. Bad markers produce
bad segmentation.
```

### Q23. How does Random Walker work?

Answer:

```text
It treats pixels as graph nodes and assigns unlabeled pixels based on their
probability of reaching foreground or background seeds.
```

### Q24. How does Chan-Vese work?

Answer:

```text
Chan-Vese evolves an active contour to separate image regions based on intensity
homogeneity, not necessarily strong edges.
```

### Q25. What is MorphGAC?

Answer:

```text
Morphological Geodesic Active Contour evolves a contour toward object boundaries
using morphological operations and image gradients.
```

### Q26. What is SAM?

Answer:

```text
SAM is a foundation segmentation model. In this app it receives a box prompt and
returns candidate masks. The best scoring mask is selected.
```

### Q27. Why might SAM not beat U-Net?

Answer:

```text
SAM is general-purpose. U-Net is trained directly on the medical dataset, so it
can learn dataset-specific lesion appearance.
```

### Q28. What is U-Net?

Answer:

```text
U-Net is an encoder-decoder convolutional network with skip connections,
designed for biomedical segmentation.
```

### Q29. What are skip connections in U-Net?

Answer:

```text
Skip connections copy high-resolution features from the encoder to the decoder,
helping recover precise spatial boundaries.
```

### Q30. What is Guided U-Net?

Answer:

```text
Guided U-Net is a U-Net variant that receives an additional guide channel,
usually encoding bounding-box information, to help localize the target object.
```

### Q31. How are masks converted to YOLO labels?

Answer:

```text
The app uses cv2.findContours to extract polygon contours from binary masks.
Then it normalizes contour coordinates by image width and height and writes them
in YOLO segmentation format.
```

### Q32. Why use `cv2.findContours`?

Answer:

```text
It is a standard, tested OpenCV method for extracting polygon boundaries from
binary masks. It avoids custom, fragile polygon conversion code.
```

### Q33. What does `masks_to_polygons` skip?

Answer:

```text
It skips tiny contours below `min_area` and contours with fewer than three
points, because those cannot form useful polygons.
```

### Q34. How are boxes created when only a ground-truth mask exists?

Answer:

```text
The app labels connected components in the ground-truth mask and uses each
component's bounding box.
```

Function:

```python
boxes_from_binary_mask(mask)
```

### Q35. How are YOLO boxes loaded?

Answer:

```text
The app reads normalized YOLO box labels and converts center-width-height format
to pixel coordinates [x1, y1, x2, y2].
```

Function:

```python
load_yolo_boxes(label_path, image_width, image_height)
```

### Q36. What happens if a box is outside image boundaries?

Answer:

```text
`clamp_box` limits the coordinates to the image width and height.
```

### Q37. What happens if a crop is invalid?

Answer:

```text
The method returns an empty full-size mask for that box.
```

### Q38. Why does the app crop around boxes?

Answer:

```text
Cropping reduces background distraction, focuses the segmentation method on one
possible lesion, and helps produce one instance mask per box.
```

### Q39. What is `prefer_center`?

Answer:

```text
It biases component selection toward components near the crop center. This is
useful because the box usually centers the target object.
```

### Q40. Where would you add a new method?

Answer:

```text
Create a new file in `code/`, implement `generate_masks`, then register it in
`METHOD_MODULES` inside `phase1_runner.py`.
```

### Q41. Where would you add a new metric?

Answer:

```text
Add the metric function to `common.py`, then compute and write it in
`phase1_runner.py` so all methods are evaluated consistently.
```

### Q42. Where would you change polygon conversion?

Answer:

```text
In `masks_to_polygons` and `save_yolo_polygons` inside `common.py`.
```

### Q43. Where would you change dataset loading?

Answer:

```text
In `build_manifests.py` if creating manifests, or in `phase1_runner.py` if
changing how manifest rows are interpreted.
```

### Q44. Where would you add a new command-line argument?

Answer:

```text
In `parse_args()` inside `phase1_runner.py` or the relevant script.
```

### Q45. If the professor asks to invert a method's mask, what do you do?

Answer:

```python
return ~(blur > threshold)
```

or:

```python
return blur <= threshold
```

If using uint8:

```python
mask = cv2.bitwise_not(mask)
```

### Q46. If the professor asks to remove small objects, what do you do?

Answer:

Use existing helper:

```python
from common import clean_binary

mask = clean_binary(mask, min_size=32)
```

### Q47. If the professor asks to keep the largest component, what do you do?

Answer:

Use `select_instance_component` or OpenCV connected components.

```python
from common import select_instance_component

mask = select_instance_component(mask, prefer_center=False)
```

### Q48. If the professor asks to make adaptive threshold block size configurable, what do you do?

Answer:

```python
block_size = int(config.get("block_size", 35))
if block_size % 2 == 0:
    block_size += 1
block_size = max(block_size, 3)
```

### Q49. If the professor asks why `block_size` must be odd, what do you say?

Answer:

```text
OpenCV adaptive thresholding requires an odd neighborhood size so there is a
well-defined center pixel.
```

### Q50. If a method returns masks with shape `(crop_h, crop_w)`, why is that wrong?

Answer:

```text
The required output is full-size masks with shape `(image_h, image_w)`, one per
box. Crop masks must be pasted back into the original image coordinates.
```

### Q51. If the professor asks to add a new dataset, what are the steps?

Answer:

1. Add dataset paths/defaults in `build_manifests.py`.
2. Write a function that creates rows with `case_id`, `image_path`, `mask_path`, `boxes_path`.
3. Save a manifest CSV.
4. If masks exist, store their paths.
5. If boxes exist, store YOLO box label paths.
6. Run `phase1_runner.py` with the new manifest and dataset name.

### Q52. What is the difference between Phase 1 and Phase 2?

Answer:

```text
Phase 1 generates and evaluates masks. Phase 2 trains YOLO segmentation models
using the generated polygon labels.
```

### Q53. Why convert masks to polygons?

Answer:

```text
YOLO segmentation training expects polygon label files, not raw binary masks.
```

### Q54. What is the risk of synthetic labels?

Answer:

```text
If generated masks are wrong, YOLO learns noisy labels. Synthetic labels can
scale annotation, but label quality controls downstream model quality.
```

### Q55. Why might a method have high Dice but still be clinically imperfect?

Answer:

```text
Dice measures overlap but not all clinical boundary errors equally. Small
boundary mistakes near important anatomy may matter clinically even if Dice is
high.
```

### Q56. What are false positives and false negatives in segmentation?

Answer:

```text
False positives are pixels predicted as lesion but not in ground truth. False
negatives are lesion pixels missed by the prediction.
```

### Q57. What does high precision but low recall mean?

Answer:

```text
The prediction is conservative. Most predicted pixels are correct, but many true
lesion pixels are missed.
```

### Q58. What does low precision but high recall mean?

Answer:

```text
The prediction captures most lesion pixels but includes too much background.
```

### Q59. If a model oversegments, how can you improve it?

Answer:

Possible fixes:

- increase threshold,
- remove small objects,
- use stronger component selection,
- reduce balloon parameter,
- tune active contour smoothness,
- use better seeds,
- use bounding boxes with less padding.

### Q60. If a model undersegments, how can you improve it?

Answer:

Possible fixes:

- lower threshold,
- increase padding,
- choose different Multi-Otsu class,
- increase active contour balloon,
- improve contrast,
- use a learned method.

---

## 16. Common Exam Coding Tasks

### Task A: Add fixed threshold method

Use the template in Section 11.

Key points:

- new file in `code/`,
- implement `generate_masks`,
- register in `METHOD_MODULES`,
- run sample manifest.

### Task B: Add median blur before Otsu

Modify `otsu.py`:

```python
kernel = int(config.get("median_kernel", 5))
if kernel % 2 == 0:
    kernel += 1
gray = cv2.medianBlur(gray, kernel)
```

Explain:

```text
Median blur can reduce salt-and-pepper noise while preserving edges better than
Gaussian blur.
```

### Task C: Change component selection from central to largest

Pass config:

```json
{
  "prefer_center": false
}
```

Or modify the call:

```python
select_instance_component(mask, prefer_center=False)
```

Explain:

```text
Center preference assumes the target is near the center of the crop. Largest
component assumes the lesion occupies the largest connected foreground area.
```

### Task D: Add recall to CSV

Add `recall_score` to `common.py`, import it in `phase1_runner.py`, compute it, add it to per-case and summary outputs.

### Task E: Add method runtime per image

Already exists:

```text
runtime_s
```

If asked to add average runtime:

```python
runtime_mean_s = runtime_total / len(percase_rows)
```

Add to summary CSV.

### Task F: Add output overlay images

Best place:

- inside `phase1_runner.py`,
- after `pred_mask` is computed.

Idea:

```python
overlay = image.copy()
overlay[pred_mask > 0] = [255, 0, 0]
```

Save with OpenCV after converting RGB to BGR:

```python
cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

### Task G: Add a new config option for polygon minimum area

Currently:

```python
polygons = masks_to_polygons(masks)
```

Change to:

```python
min_area = float(config.get("polygon_min_area", 5))
polygons = masks_to_polygons(masks, min_area=min_area)
```

### Task H: Add support for explicit boxes in manifest

Already partly supported:

```python
if all(row.get(k) for k in ["x1", "y1", "x2", "y2"]):
    return [[float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]]
```

But `build_manifests.py` currently writes only:

```text
case_id,image_path,mask_path,boxes_path
```

So you would also need to update manifest fieldnames.

---

## 17. Common Bugs And Fixes

### Bug: method not accepted by CLI

Symptom:

```text
invalid choice: 'new_method'
```

Cause:

Method not registered in `METHOD_MODULES`.

Fix:

```python
"new_method": "new_method",
```

### Bug: import error for method

Symptom:

```text
ModuleNotFoundError
```

Causes:

- file name does not match module name,
- method file not in `code/`,
- typo in `METHOD_MODULES`.

### Bug: output mask has wrong shape

Cause:

Returning crop-size masks instead of full-size masks.

Fix:

Use:

```python
paste_crop_mask((height, width), crop_mask, coords)
```

or use:

```python
generate_by_crop(...)
```

### Bug: Dice/IoU are NaN

Possible causes:

- no ground-truth mask exists,
- mask path is empty,
- mask path does not exist.

For Brain Tumor, this is expected.

### Bug: all masks are empty

Possible causes:

- boxes are empty,
- threshold too high,
- image normalization failed,
- method selected wrong intensity class,
- postprocessing removed everything.

### Bug: masks cover almost whole crop

Possible causes:

- threshold direction inverted,
- target class wrong,
- active contour balloon too high,
- weak postprocessing.

### Bug: YOLO labels are empty

Possible causes:

- masks are empty,
- contours are smaller than `min_area`,
- contour has fewer than three points.

### Bug: U-Net requires checkpoint

Symptom:

```text
U-Net requires config['checkpoint'].
```

Fix:

Pass a JSON config containing:

```json
{
  "checkpoint": "path/to/checkpoint.pt",
  "device": "cpu"
}
```

### Bug: SAM requires checkpoint

Symptom:

```text
SAM requires config['checkpoint'] pointing to a SAM checkpoint.
```

Fix:

Pass SAM checkpoint path and install `segment-anything`.

---

## 18. How To Explain Results

A strong exam explanation:

```text
The learned methods performed best because they were trained from expert masks.
U-Net achieved the best BUSI IoU, and Guided U-Net achieved the best BraTS IoU.
SAM performed well without task-specific training, which makes it useful as a
foundation-model baseline. Classical methods were faster and easier to run, but
their performance depended strongly on image modality and intensity assumptions.
BUSI ultrasound was harder for simple thresholding because of speckle noise and
weak boundaries. BraTS MRI was more favorable to threshold-based methods because
the selected modality often separated tumor from background more clearly.
```

Another strong explanation:

```text
There is a clear tradeoff between speed, supervision, and accuracy. Otsu and
adaptive thresholding are very fast but less robust. Active contour methods can
be accurate but slower and parameter-sensitive. SAM is strong without training
but computationally heavier. U-Net and Guided U-Net are best when labeled data
and training resources are available.
```

---

## 19. What To Study Before The Exam

High priority:

1. The `generate_masks(image, boxes, config)` interface.
2. The role of `phase1_runner.py`.
3. The helper functions in `common.py`.
4. Dice and IoU formulas.
5. How masks become polygons.
6. How to add a new method.
7. How to add a new metric.
8. The differences between Otsu, adaptive thresholding, watershed, active contours, SAM, and U-Net.
9. Which methods worked best and why.
10. Why Brain Tumor has NaN Phase 1 Dice/IoU.

Medium priority:

1. Exact command-line usage.
2. CSV schemas.
3. YOLO polygon format.
4. U-Net training logs.
5. Phase 2 mAP metrics.

Lower priority:

1. Every exact hyperparameter value.
2. Internal details of SAM.
3. Exact YOLO training commands.

---

## 20. One-Page Oral Exam Summary

If you need to explain the whole app quickly:

```text
This project builds a modular Python pipeline for generating synthetic
segmentation labels in oncology images. The main runner reads a manifest, loads
images and masks, derives bounding boxes, calls a selected segmentation method
through a shared generate_masks interface, evaluates the resulting masks using
Dice and IoU, and optionally converts masks to YOLO polygon labels using
cv2.findContours.

The methods include classical thresholding, watershed, connected components,
random walker, active contours, SAM, U-Net, and Guided U-Net. Classical methods
are fast and interpretable but depend on assumptions about intensity and
boundaries. SAM is a strong zero-shot/foundation baseline. U-Net and Guided
U-Net are supervised learned methods and performed best overall.

On BUSI, U-Net was best with IoU about 0.909, followed by Guided U-Net and SAM.
On BraTS, Guided U-Net was best with IoU about 0.856, followed closely by U-Net.
Brain Tumor has no ground-truth masks in the local dataset, so Phase 1 Dice/IoU
are NaN by design.

To extend the app, I would add a new method file implementing generate_masks,
register it in METHOD_MODULES, and run phase1_runner on a sample manifest. To add
a metric, I would add the metric to common.py and compute/write it centrally in
phase1_runner.py so all methods are evaluated consistently.
```

---

## 21. Final Exam Checklist

Before the exam, make sure you can do these without help:

- Explain the project goal in two sentences.
- Draw the pipeline from manifest to CSV.
- Explain `generate_masks`.
- Add a new method file.
- Register a new method.
- Run `phase1_runner.py`.
- Explain Dice and IoU.
- Explain why Brain Tumor has NaN metrics.
- Explain mask-to-polygon conversion.
- Explain why U-Net beat classical methods.
- Explain at least three classical methods.
- Explain where to edit code for a metric.
- Explain where to edit code for a dataset.
- Explain where to edit code for polygon output.
- Debug an empty mask.
- Debug a wrong-shape mask.

If you can do those, you are in good shape for the exam.

