# === CELL 0 ===
# Define the paths where the data folders are located (specifically for AttriDet labels)
import os
from glob import glob

# Define paths
base_path = r"D:\datasets\dataset\LeukemiaAttri_Dataset"
image_pattern_train = os.path.join(base_path, "**", "images", "train", "*.png")
image_pattern_test = os.path.join(base_path, "**", "images", "test", "*.png")
label_pattern_train = os.path.join(base_path, "**", "txt_labels", "AttriDet", "train", "*.txt")
label_pattern_test = os.path.join(base_path, "**", "txt_labels", "AttriDet", "test", "*.txt")

# glob with recursive=True to find all files
train_images = glob(image_pattern_train, recursive=True)
test_images = glob(image_pattern_test, recursive=True)
train_labels = glob(label_pattern_train, recursive=True)
test_labels = glob(label_pattern_test, recursive=True)

# Number of files of each type
len(train_images), len(train_labels), len(test_images), len(test_labels)

# === CELL 1 ===
# Trial

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, threshold_multiotsu, gaussian
from skimage.morphology import disk, closing, erosion
from skimage.segmentation import watershed
from skimage import exposure
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import os
import random

def load_bounding_boxes_yolo(filepath, img_w, img_h):
    boxes = []
    if not os.path.exists(filepath): return boxes
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) < 5: continue
                try:
                    center_x_norm, center_y_norm = float(parts[1]), float(parts[2])
                    width_norm, height_norm = float(parts[3]), float(parts[4])
                    center_x_pix = center_x_norm * img_w
                    center_y_pix = center_y_norm * img_h
                    width_pix = width_norm * img_w
                    height_pix = height_norm * img_h
                    x_min = int(center_x_pix - (width_pix / 2))
                    y_min = int(center_y_pix - (height_pix / 2))
                    x_max = int(center_x_pix + (width_pix / 2))
                    y_max = int(center_y_pix + (height_pix / 2))
                    boxes.append([max(0, x_min), max(0, y_min), min(img_w, x_max), min(img_h, y_max)])
                except ValueError: continue
    return boxes

def resolve_overlaps(masks, img_h, img_w):
    owner_map = np.zeros((img_h, img_w), dtype=int) - 1
    min_dist_map = np.full((img_h, img_w), np.inf)
    centers = []
    for i, (mask, _) in enumerate(masks):
        coords = np.argwhere(mask)
        if len(coords) > 0: centers.append(coords.mean(axis=0))
        else: centers.append((0, 0))
    for i, (mask, _) in enumerate(masks):
        y_idxs, x_idxs = np.where(mask)
        c_y, c_x = centers[i]
        dists = np.sqrt((y_idxs - c_y)**2 + (x_idxs - c_x)**2)
        for j in range(len(y_idxs)):
            y, x, d = y_idxs[j], x_idxs[j], dists[j]
            if d < min_dist_map[y, x]:
                min_dist_map[y, x] = d
                owner_map[y, x] = i
    clean_masks = []
    for i, (_, color) in enumerate(masks):
        new_mask = (owner_map == i)
        clean_masks.append((new_mask, color))
    return clean_masks

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = int(179 * i / n)
        saturation = 255
        value = 255
        hsv_pixel = np.uint8([[[hue, saturation, value]]])
        rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0]
        colors.append((int(rgb_pixel[0]), int(rgb_pixel[1]), int(rgb_pixel[2])))
    random.shuffle(colors)
    return colors

# ΝΕΑ ΣΥΝΑΡΤΗΣΗ: UNIVERSAL CONTRAST 
def enhance_nuclei_contrast(image_crop):
    """
    Universal method: Inverted Green Channel + Gamma + Normalize.
    Works on: 100X (Purple cells), 10X (Black/Gray cells), and Faded images.
    """
    #   1. Gamma Correction (Darkens faint borders/edges to make them visible)
    image_crop = exposure.adjust_gamma(image_crop, gamma=1.2)

    # 2. Inverted Green (The nucleus always has less green color than the background)
    # By inverting it, the dark nucleus becomes bright white.
    G = image_crop[:, :, 1]
    contrast_img = 255 - G
    
    # 3. Normalization (Fixes low contrast in blurry or faded images)
    # Stretches the brightness to cover the full range (0 to 255).
    contrast_img = cv2.normalize(contrast_img, None, 0, 255, cv2.NORM_MINMAX)
    
    return contrast_img.astype(np.uint8)

def segment_debug_view(image_path, boxes_path):
    image = cv2.imread(image_path)
    if image is None: return None, None, None
    
    img_h, img_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = load_bounding_boxes_yolo(boxes_path, img_w, img_h)
    
    distinct_colors = generate_distinct_colors(len(boxes))
    
    image_with_boxes = image_rgb.copy()
    collected_masks = []

    debug_heatmap = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        current_color = distinct_colors[i]
        
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
        cropped_image = image[y_min:y_max, x_min:x_max]
        if cropped_image.size == 0: continue

        #  1. UNIVERSAL PRE-PROCESSING 
        proc_channel = enhance_nuclei_contrast(cropped_image)

        # Soft CLAHE & Blur
        proc_norm = proc_channel.astype(float) / 255.0
        enhanced = exposure.equalize_adapthist(proc_norm, clip_limit=0.01)
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(enhanced_uint8, (5, 5), 0)
        
        debug_heatmap[y_min:y_max, x_min:x_max] = blurred

        # 2. OTSU & FILL HOLES 
        try:
            thresh = threshold_otsu(blurred)
            mask_loose = blurred > thresh
            mask_loose = ndi.binary_fill_holes(mask_loose)
            mask_loose = closing(mask_loose, disk(3))
        except: continue

        # Markers & Watershed
        distance = ndi.distance_transform_edt(mask_loose)
        distance = gaussian(distance, sigma=1)
        try:
            thresholds = threshold_multiotsu(blurred, classes=3)
            mask_strict = blurred > thresholds[1]
            mask_strict = ndi.binary_fill_holes(mask_strict) # Fill holes here too
            if np.sum(mask_strict) == 0: mask_strict = erosion(mask_loose, disk(5))
        except:
             mask_strict = erosion(mask_loose, disk(5))
             
        markers = label(mask_strict)
        labels = watershed(-distance, markers, mask=mask_loose)

        # Selection Logic
        crop_h, crop_w = labels.shape
        center = (crop_h // 2, crop_w // 2)
        best_lbl = 0
        max_area = 0
        for region in regionprops(labels):
            if region.label == 0: continue
            rc, cc = region.centroid
            d = np.sqrt((rc - center[0])**2 + (cc - center[1])**2)
            is_dominant = region.area > max_area
            is_reasonable = d < (max(crop_h, crop_w) * 0.65)
            if is_dominant and is_reasonable:
                max_area = region.area
                best_lbl = region.label

        if best_lbl == 0: continue
        final_mask = (labels == best_lbl)
        final_mask = erosion(final_mask, disk(1))

        global_m = np.zeros((img_h, img_w), dtype=bool)
        global_m[y_min:y_max, x_min:x_max] = final_mask
        
        collected_masks.append((global_m, current_color))

    # Draw
    clean_masks = resolve_overlaps(collected_masks, img_h, img_w)
    final_mask_img = np.zeros_like(image_rgb, dtype=np.uint8)
    
    for mask, color in clean_masks:
        temp = np.zeros_like(image_rgb, dtype=np.uint8)
        temp[mask] = color
        final_mask_img = np.where(temp > 0, temp, final_mask_img)

    return image_with_boxes, debug_heatmap, final_mask_img

if __name__ == "__main__":

    IMAGE_PATH = r"D:\datasets\dataset\LeukemiaAttri_Dataset\H_100X_C2\images\test\4_56_1000_ALL.png"
    BOXES_PATH = r"D:\datasets\dataset\LeukemiaAttri_Dataset\H_100X_C2\txt_labels\AttriDet\test\4_56_1000_ALL.txt"

    img_res, heatmap_res, mask_res = segment_debug_view(IMAGE_PATH, BOXES_PATH)

    if img_res is not None:
        plt.figure(figsize=(20, 7))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img_res)
        plt.title("1. Original Image with Boxes")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_res, cmap='gray')
        plt.title("2. Universal Contrast Heatmap\n(255 - G) + Gamma")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(mask_res)
        plt.title("3. Final Instance Masks")
        plt.axis('off')
        
        plt.show()

# === CELL 2 ===
# Mask Generator Final Version
import os
import cv2
import numpy as np
import random
import time
from tqdm import tqdm
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, threshold_multiotsu, gaussian
from skimage.morphology import disk, closing, erosion
from skimage.segmentation import watershed
from skimage import exposure
from scipy import ndimage as ndi
import concurrent.futures

# Define where your dataset is located
BASE_PATH = r"D:\datasets\dataset\LeukemiaAttri_Dataset"

#  1. Box reader
def load_bounding_boxes_yolo(filepath, img_w, img_h):
    """
    EXPLANATION:
    Reads the text files (.txt) where YOLO stored the cell locations.
    It converts the weird YOLO numbers (0.5, 0.5) into actual pixels (Row 100, Col 200),
    so we know exactly where to cut the image to find the cell.
    """
    boxes = []
    if not os.path.exists(filepath): return boxes
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) < 5: continue
                try:
                    center_x_norm, center_y_norm = float(parts[1]), float(parts[2])
                    width_norm, height_norm = float(parts[3]), float(parts[4])
                    center_x_pix = center_x_norm * img_w
                    center_y_pix = center_y_norm * img_h
                    width_pix = width_norm * img_w
                    height_pix = height_norm * img_h
                    x_min = int(center_x_pix - (width_pix / 2))
                    y_min = int(center_y_pix - (height_pix / 2))
                    x_max = int(center_x_pix + (width_pix / 2))
                    y_max = int(center_y_pix + (height_pix / 2))
                    boxes.append([max(0, x_min), max(0, y_min), min(img_w, x_max), min(img_h, y_max)])
                except ValueError: continue
    return boxes

# 2. Overlap solver
def resolve_overlaps(masks, img_h, img_w):
    """
    EXPLANATION:
    When two cells are very close, they might fight over the same pixels.
    This function acts as a Referee. It assigns each disputed pixel to the 
    NEAREST cell center. This ensures clean borders between neighbors 
    and prevents colors from mixing.
    """
    owner_map = np.zeros((img_h, img_w), dtype=int) - 1
    min_dist_map = np.full((img_h, img_w), np.inf)
    centers = []
    
    # Find centers of all masks
    for i, (mask, _) in enumerate(masks):
        coords = np.argwhere(mask)
        if len(coords) > 0: centers.append(coords.mean(axis=0))
        else: centers.append((0, 0))
    
    # Assign pixels
    for i, (mask, _) in enumerate(masks):
        y_idxs, x_idxs = np.where(mask)
        c_y, c_x = centers[i]
        dists = np.sqrt((y_idxs - c_y)**2 + (x_idxs - c_x)**2)
        
        # Only update if this cell is closer than the previous owner
        current_mins = min_dist_map[y_idxs, x_idxs]
        mask_closer = dists < current_mins
        min_dist_map[y_idxs[mask_closer], x_idxs[mask_closer]] = dists[mask_closer]
        owner_map[y_idxs[mask_closer], x_idxs[mask_closer]] = i
        
    clean_masks = []
    for i, (_, color) in enumerate(masks):
        new_mask = (owner_map == i)
        clean_masks.append((new_mask, color))
    return clean_masks

# 3. Color generator
def generate_distinct_colors(n):
    """
    Creates a list of unique, bright colors.
    It uses a mathematical trick (HSV color wheel) to ensure that 
    every cell gets a color that is very different from its neighbors.
    """
    colors = []
    for i in range(n):
        hue = int(179 * i / n)
        saturation = 255
        value = 255
        hsv_pixel = np.uint8([[[hue, saturation, value]]])
        rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0]
        colors.append((int(rgb_pixel[0]), int(rgb_pixel[1]), int(rgb_pixel[2])))
    random.shuffle(colors)
    return colors

#  4. Image enchancer
def enhance_nuclei_contrast(image_crop):
    """
    This function fixes issues with lighting and color.
    1. Inverted Green: It takes the Green channel and flips it. Dark nuclei become bright white.
       This works for both Purple cells (100X) and Black cells (10X).
    2. Normalization: It stretches the brightness range so even faded images look sharp.
    3. Gamma: It darkens the background noise to make the cell stand out more.
    """
    # Inverted Green
    G = image_crop[:, :, 1]
    contrast_img = 255 - G
    
    # Normalization (Fixes faded images)
    contrast_img = cv2.normalize(contrast_img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Gamma Correction (Darkens noise)
    contrast_img = exposure.adjust_gamma(contrast_img, gamma=1.2)
    
    return contrast_img.astype(np.uint8)

# 5. Image processor
def generate_mask_batch(image, boxes):
    """
    This is the core logic that processes one image at a time.
    1. Cuts out each cell based on the boxes.
    2. Enhances the image using the universal contrast function.
    3. Blurs it slightly to remove grain.
    4. Calculates the shape using 'Otsu' thresholding.
    5. Fills holes inside the cell (donut fix).
    6. Uses Watershed to separate touching cells.
    7. Assigns a unique color to each cell.
    """
    img_h, img_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    collected_masks = []
    distinct_colors = generate_distinct_colors(len(boxes))

    for i, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        current_color = distinct_colors[i]
        cropped_image = image[y_min:y_max, x_min:x_max]
        if cropped_image.size == 0: continue

        # Universal Pre-processing
        proc_channel = enhance_nuclei_contrast(cropped_image)

        # CLAHE (Enhance local contrast)
        proc_norm = proc_channel.astype(float) / 255.0
        enhanced = exposure.equalize_adapthist(proc_norm, clip_limit=0.01)
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        
        # Blur (Remove noise)
        blurred = cv2.GaussianBlur(enhanced_uint8, (5, 5), 0)
        
        try:
            # Otsu Thresholding (Find the shape)
            thresh = threshold_otsu(blurred)
            mask_loose = blurred > thresh
            
            # FILL HOLES: Crucial for fixing black cells
            mask_loose = ndi.binary_fill_holes(mask_loose)
            
            mask_loose = closing(mask_loose, disk(3))
        except: continue

        # Markers & Watershed (Separate touching cells)
        distance = ndi.distance_transform_edt(mask_loose)
        distance = gaussian(distance, sigma=1)
        try:
            thresholds = threshold_multiotsu(blurred, classes=3)
            mask_strict = blurred > thresholds[1]
            mask_strict = ndi.binary_fill_holes(mask_strict) # Fill holes here too
            if np.sum(mask_strict) == 0: mask_strict = erosion(mask_loose, disk(5))
        except:
             mask_strict = erosion(mask_loose, disk(5))
             
        markers = label(mask_strict)
        labels = watershed(-distance, markers, mask=mask_loose)

        # Selection Logic (Pick the best object)
        crop_h, crop_w = labels.shape
        center = (crop_h // 2, crop_w // 2)
        best_lbl = 0
        max_area = 0
        for region in regionprops(labels):
            if region.label == 0: continue
            rc, cc = region.centroid
            d = np.sqrt((rc - center[0])**2 + (cc - center[1])**2)
            is_dominant = region.area > max_area
            is_reasonable_location = d < (max(crop_h, crop_w) * 0.6)
            if is_dominant and is_reasonable_location:
                max_area = region.area
                best_lbl = region.label

        if best_lbl == 0: continue
        final_mask = (labels == best_lbl)
        final_mask = erosion(final_mask, disk(1))

        global_m = np.zeros((img_h, img_w), dtype=bool)
        global_m[y_min:y_max, x_min:x_max] = final_mask
        collected_masks.append((global_m, current_color))

    # Resolve Overlaps and Draw
    clean_masks = resolve_overlaps(collected_masks, img_h, img_w)
    final_output = np.zeros_like(image_rgb, dtype=np.uint8)
    for mask, color in clean_masks:
        temp = np.zeros_like(image_rgb, dtype=np.uint8)
        temp[mask] = color
        final_output = np.maximum(final_output, temp)
        
    return cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)

# 6. Prallel processing
def process_single_file(args):
    """
    This function handles one image file from start to finish.
    It's designed to run in parallel (multiple images at the same time).
    1. Loads the image and the box file.
    2. Checks if boxes exist.
    3. Calls 'generate_mask_batch' to do the heavy lifting.
    4. Saves the result to the disk.
    """
    img_full_path, txt_full_path, output_path = args
    image = cv2.imread(img_full_path)
    if image is None: return False
    img_h, img_w = image.shape[:2]
    
    # Load boxes
    boxes = load_bounding_boxes_yolo(txt_full_path, img_w, img_h)
    
    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # If no boxes, save a blank image
    if not boxes:
        blank = np.zeros_like(image)
        cv2.imwrite(output_path, blank)
        return True

    # Generate the mask
    final_mask_img = generate_mask_batch(image, boxes)
    
    # Save the result
    cv2.imwrite(output_path, final_mask_img)
    return True

# Main execution
if __name__ == "__main__":
    tasks = []
    
    # Find all data folders
    data_folders = [
        name for name in os.listdir(BASE_PATH)
        if os.path.isdir(os.path.join(BASE_PATH, name)) and (name.startswith("L_") or name.startswith("H_"))
    ]

    # Prepare the list of tasks (images to process)
    for folder_name in data_folders:
        folder_path = os.path.join(BASE_PATH, folder_name)
        for mode in ['train', 'test']:
            images_dir = os.path.join(folder_path, 'images', mode)
            labels_dir = os.path.join(folder_path, 'txt_labels', 'AttriDet', mode)
            
            # Define output folder
            output_dir = os.path.join(folder_path, 'masks_miccai_instance', mode) 
            
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir): continue

            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                img_full_path = os.path.join(images_dir, image_file)
                txt_filename = os.path.splitext(image_file)[0] + ".txt"
                txt_full_path = os.path.join(labels_dir, txt_filename)
                output_path = os.path.join(output_dir, image_file)
                tasks.append((img_full_path, txt_full_path, output_path))

    print(f" Starting UNIVERSAL Processing for {len(tasks)} images...")
    start_time = time.time()
    
    # Run the tasks in parallel (using 4 CPU cores to make it faster)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks), unit="img"))

    end_time = time.time()
    print(f"\n Done. Total Time: {int((end_time - start_time)//60)} min.")

# === CELL 3 ===
# Overlays Generator
import os
import cv2
import numpy as np
import time
from tqdm import tqdm
import concurrent.futures

BASE_PATH = r"D:\datasets\dataset\LeukemiaAttri_Dataset"


def create_visualization(image_path, mask_path, boxes_path):
    """
    This function creates the final 'Quality Control' image.
    1. Loads the original photo.
    2. Loads the colored mask we created.
    3. Reads the box coordinates from the text file.
    4. Blends the mask over the photo (making it look transparent).
    5. Draws white outlines (boxes) around the cells.
    """
    # Load Image and Mask
    image = cv2.imread(image_path)
    if image is None: return None

    if not os.path.exists(mask_path):
        return None
    
    mask = cv2.imread(mask_path)

    # Safety Check: Resize mask if it doesn't match image size
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Load Bounding Boxes (YOLO format -> Pixels)
    img_h, img_w = image.shape[:2]
    boxes = []
    if os.path.exists(boxes_path):
        with open(boxes_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        # YOLO stores data as: Class, Center_X, Center_Y, Width, Height
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert to Pixels (Top-Left and Bottom-Right coordinates)
                        x_min = int((cx - w/2) * img_w)
                        y_min = int((cy - h/2) * img_h)
                        x_max = int((cx + w/2) * img_w)
                        y_max = int((cy + h/2) * img_h)
                        
                        boxes.append([x_min, y_min, x_max, y_max])
                    except: pass

    # Create overlay image
    final_overlay = image.copy()

    # Alpha Blending (Making the mask semi-transparent)
    mask_indices = np.any(mask > 0, axis=-1)
    
    if np.any(mask_indices):
        # 60% Image + 40% Mask
        blended_region = cv2.addWeighted(image, 0.6, mask, 0.4, 0)
        final_overlay[mask_indices] = blended_region[mask_indices]

    # Draw Bounding Boxes (White outlines)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(final_overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return final_overlay

# Ο Wrapper για το Multiprocessing
def process_single_overlay(args):
    """
    Handles a single file.
    1. Receives paths for image, mask, and box.
    2. Calls 'create_visualization' to make the picture.
    3. Saves the result to the disk.
    """
    img_path, mask_path, box_path, save_path = args
    
    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    result_img = create_visualization(img_path, mask_path, box_path)
    
    if result_img is not None:
        cv2.imwrite(save_path, result_img)
        return True
    return False

# Main execution
if __name__ == "__main__":
    
    tasks = []

    # Find folders starting with H_... and L_...
    data_folders = [
        name for name in os.listdir(BASE_PATH)
        if os.path.isdir(os.path.join(BASE_PATH, name)) and (name.startswith("L_") or name.startswith("H_"))
    ]
    
    print(f" Scanning {len(data_folders)} folders to build task list...")

    for folder_name in data_folders:
        folder_path = os.path.join(BASE_PATH, folder_name)
        
        for mode in ['train', 'test']:
            images_dir = os.path.join(folder_path, 'images', mode)
            
            masks_dir = os.path.join(folder_path, 'masks_miccai_instance_smart', mode) 
            labels_dir = os.path.join(folder_path, 'txt_labels', 'AttriDet', mode)
            
            # Output directory
            output_dir = os.path.join(folder_path, 'masks_overlay_miccai_instance_smart', mode)

            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                continue

            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for image_file in image_files:
                # Paths
                img_path = os.path.join(images_dir, image_file)
                
                # Mask Logic (Check for png, then fallback to jpg)
                mask_filename = os.path.splitext(image_file)[0] + ".png" 
                mask_path = os.path.join(masks_dir, mask_filename)
                
                if not os.path.exists(mask_path):
                      mask_path = os.path.join(masks_dir, image_file) # Try original extension

                txt_name = os.path.splitext(image_file)[0] + ".txt"
                box_path = os.path.join(labels_dir, txt_name)
                
                save_path = os.path.join(output_dir, image_file) 

                # Add to tasks
                tasks.append((img_path, mask_path, box_path, save_path))

    print(f" Starting Parallel Overlay Generation for {len(tasks)} images...")
    
    start_time = time.time()
    # Use 4 CPU cores to process images simultaneously
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_single_overlay, tasks), total=len(tasks), unit="img"))

    end_time = time.time()
    total_min = int((end_time - start_time) // 60)
    total_sec = int((end_time - start_time) % 60)

    print(f"\n Done! Overlays saved in 'masks_overlay_miccai_instance'.")
    print(f" Total Time: {total_min} min {total_sec} sec.")
