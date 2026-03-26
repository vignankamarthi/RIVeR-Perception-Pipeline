# YOLO OBB Training Pipeline

> End-to-end: capture images, auto-label, convert, split, train, evaluate.
> Reproducible for any object. For example, swap "banana" for "pear" and re-run.

---

## Prerequisites

**Hardware:**
- Mac with Apple Silicon (MPS acceleration)
- Intel RealSense D455 (USB, workspace view)
- Azure Kinect DK (USB, top-down view, connected to Ubuntu machine)
- Lab network: Mac and Ubuntu on same router

**Software (Mac):**
```bash
pip install ultralytics labelme pytest opencv-python numpy
# pyrealsense2-macosx for RealSense capture (Mac-specific wheel)
pip install pyrealsense2-macosx
```

**Software (Ubuntu -- Kinect only):**
```bash
pip install pyk4a
# Note: Kinect capture requires a physical display (Terminator, not SSH)
```

All auto-labeling models (YOLO-World, SAM2) are bundled in `ultralytics`. Zero additional installs.

---

## Pipeline Overview

```
Step 1: Capture images (RealSense + Kinect)
Step 2: Auto-label with YOLO-World + SAM2
Step 3: Audit labels in LabelMe
Step 4: Convert LabelMe JSON to YOLO OBB format
Step 5: Split into train/val (80/20)
Step 6: Train YOLOv8-OBB
Step 7: Evaluate
```

---

## Step 1: Capture Images

### RealSense (Mac native)

```bash
python3 scripts/capture_realsense.py --camera 0
```

- Press **SPACE** to capture, **Q** to quit
- Saves to `images/realsense/rs_001.png`, `rs_002.png`, ...
- RGB only (640x480), OpenCV UVC backend

### Kinect (Ubuntu, physical display required)

```bash
# On Ubuntu machine (Terminator, NOT SSH -- depth engine needs a display)
python3 scripts/capture_kinect.py --gui
```

- Press **SPACE** to capture, **Q** to quit
- Saves to `images/kinect/kt_001.png`, `kt_002.png`, ...
- RGB (1920x1080)

### Transfer Kinect images to Mac

```bash
scp river@192.168.0.150:~/captures/kinect/*.png images/kinect/
```

### Capture protocol

- Target: ~50 arrangements x 2 cameras = ~100 images
- Both cameras stay fixed. Move the object between captures.
- Vary: position, rotation, distance (close/far), clutter level, partial occlusion, near robot arm
- Include easy cases (object alone, centered) AND hard cases (edge of frame, behind gripper, with other objects)

---

## Step 2: Auto-Label with YOLO-World + SAM2

### What it does

Processes saved images **offline** (not live). For each image:

1. **YOLO-World** (`yolov8x-worldv2.pt`, 140MB) -- open-vocabulary detection. Given a text prompt like "banana", detects the object in any image. No retraining needed to switch objects.
2. **SAM2** (`sam2_b.pt`, 154MB) -- Segment Anything Model 2. Takes detection bounding boxes and produces pixel-perfect segmentation masks.
3. **mask_to_obb** -- our custom conversion. Runs `cv2.findContours` on the mask, then `cv2.minAreaRect` to get a tight oriented bounding box (4 corner points).

Both models auto-download on first use and are included in `ultralytics`.

### How to run

```bash
# Single class (default: banana)
python3 scripts/auto_label.py

# Multiple classes
python3 scripts/auto_label.py --classes banana pear can

# Custom confidence threshold
python3 scripts/auto_label.py --conf 0.3
```

### Output

- **LabelMe JSON** in `labels/` -- one `.json` per image, for human review
- **YOLO OBB txt** in `dataset/labels/` -- one `.txt` per image, for training

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--images-dir` | `../images` | Directory with image subdirectories |
| `--labels-dir` | `../labels` | Output for LabelMe JSON |
| `--yolo-dir` | `../dataset/labels` | Output for YOLO OBB txt |
| `--classes` | `banana` | Space-separated class names |
| `--conf` | `0.25` | Detection confidence threshold |
| `--det-weights` | `yolov8x-worldv2.pt` | YOLO-World model |
| `--sam-weights` | `sam2_b.pt` | SAM2 model |

### Expected output

```
Images processed: 104
Total detections: 99
Images with 0 detections: 11
Confidence range: 0.257 - 0.972
Confidence mean: 0.816
```

Typical results: 85-90% of images get correct auto-labels. The rest need manual attention (see Step 3).

---

## Step 3: Audit Labels in LabelMe

### Why

Auto-labeling is not perfect. Common issues:
- **Missed detections** (0 labels) -- happens with heavy occlusion, clutter, unusual angles
- **False positives** (2+ labels) -- similar-colored objects detected as target (e.g., pear detected as banana)
- **Duplicate detections** -- same object detected twice with identical OBB

### Setup

The auto-labeler copies JSON files next to the images. Open LabelMe pointed at an image directory:

```bash
# Review kinect images (check these first -- more misses from top-down angle)
labelme images/kinect/

# Review realsense images
labelme images/realsense/
```

LabelMe will show each image with its auto-generated polygon overlaid. Use **Next Image** / **Prev Image** to navigate.

### What to check

| Issue | Action |
|-------|--------|
| No polygon on image, banana clearly visible | Draw a 4-point polygon, label as the class name (e.g., `banana`) |
| No polygon, banana heavily occluded (<50% visible) | **Skip it** -- bad training data, move to next |
| Two polygons, one is wrong | Delete the false positive (click shape, press Delete) |
| Polygon is loose but covers the banana | **Leave it** -- rough rectangles are fine for OBB training |
| Polygon is tight and accurate | **Leave it** -- auto-label worked correctly |

### Label conventions

- **Lowercase class names**: `banana`, `pear`, `can` (must match `CLASS_MAP` in `labelme_to_yolo_obb.py`)
- **4-point polygons**: either tight polygons or rectangles (both work)
- **One label per object instance**

### After review

Sync labels back to the `labels/` directory if you edited them in the image directories:

```bash
python3 -c "
import json, shutil
from pathlib import Path
labels_dir = Path('labels')
for subdir in ['kinect', 'realsense']:
    for jf in Path(f'images/{subdir}').glob('*.json'):
        shutil.copy2(jf, labels_dir / jf.name)
print('Synced')
"
```

---

## Step 4: Convert to YOLO OBB Format

```bash
python3 scripts/labelme_to_yolo_obb.py
```

Converts LabelMe JSON annotations to YOLO OBB format:
- Input: `labels/*.json`
- Output: `dataset/labels/*.txt`
- Format: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (normalized to [0, 1])
- Handles both 4-point polygons and LabelMe rectangles (auto-expanded to 4 corners)

### Adding new classes

Edit `CLASS_MAP` in `scripts/labelme_to_yolo_obb.py`:

```python
CLASS_MAP = {"banana": 0}                    # current
CLASS_MAP = {"banana": 0, "pear": 1}         # adding pear
CLASS_MAP = {"banana": 0, "pear": 1, "can": 2}  # adding can
```

Also update `dataset/data.yaml` to match:

```yaml
nc: 2
names:
  0: banana
  1: pear
```

---

## Step 5: Train/Val Split

```bash
python3 scripts/split_dataset.py
```

- 80% train, 20% val (deterministic, seed=42)
- Input: images from `images/realsense/` and `images/kinect/`, labels from `dataset/labels/`
- Output:
  ```
  dataset/
    images/train/   (84 images)
    images/val/     (20 images)
    labels/train/   (84 labels)
    labels/val/     (20 labels)
    data.yaml
  ```

---

## Step 6: Train YOLOv8-OBB

```bash
python3 scripts/train.py
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8n-obb.pt` (nano, pretrained on DOTA) |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Device | Apple Silicon CPU (MPS) |

### What to monitor

- **box_loss** and **cls_loss** should decrease over epochs
- **mAP50** should exceed 0.8 (our target)
- Training time: ~20-25 min on Apple Silicon for ~100 images

### Results (banana, 104 images, 2026-03-25)

| Metric | Value |
|--------|-------|
| Precision | 1.000 |
| Recall | 0.889 |
| mAP50 | 0.921 |
| mAP50-95 | 0.778 |
| Model size | 6.5 MB |
| Training time | 22 min |

### Output

- Best weights: `models/banana_obb/weights/best.pt`
- Training curves: `models/banana_obb/results.png`

### Save the model

```bash
cp models/banana_obb/weights/best.pt models/banana_obb_v1.pt
```

---

## Step 7: Evaluate

```bash
python3 scripts/evaluate.py
```

Runs inference on the validation set and saves images with drawn OBBs to `runs/obb/predict/`.

### What to check

- OBBs should be tight around the object
- Rotation should match the object's orientation
- No false positives (boxes on non-target objects)
- Missed detections are acceptable on heavily occluded images

### Custom evaluation

```bash
# Different weights
python3 scripts/evaluate.py --weights models/banana_obb_v1.pt

# Different images
python3 scripts/evaluate.py --source path/to/test/images/

# Different confidence threshold
python3 scripts/evaluate.py --conf 0.7
```

---

## Reproducing for a New Object (e.g., Pear)

1. **Capture** ~50+ images of the new object with both cameras (Step 1)
2. **Auto-label**: `python3 scripts/auto_label.py --classes pear`
3. **Audit** in LabelMe (Step 3)
4. **Update class map** in `labelme_to_yolo_obb.py` and `data.yaml`
5. **Convert**: `python3 scripts/labelme_to_yolo_obb.py`
6. **Split**: `python3 scripts/split_dataset.py`
7. **Train**: `python3 scripts/train.py`
8. **Evaluate**: `python3 scripts/evaluate.py`

For multi-class (banana + pear together): capture images with both objects, auto-label with `--classes banana pear`, and update the class map accordingly.