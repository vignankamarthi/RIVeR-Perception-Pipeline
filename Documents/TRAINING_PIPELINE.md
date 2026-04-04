# YOLO OBB Training Pipeline

> End-to-end: capture images, auto-label, convert, split, train, evaluate.
> Reproducible for any object set. Swap the class names and re-run.

---

## Prerequisites

**Hardware:**
- Mac with Apple Silicon (CPU training, ~12 min for 60 images)
- Azure Kinect DK (USB, top-down view, connected to Ubuntu machine)
- Intel RealSense D455 (USB, side view -- optional, Kinect is primary)
- Lab network: Mac and Ubuntu on same router

**Software (Mac):**
```bash
pip install ultralytics labelme pytest opencv-python numpy
```

**Software (Ubuntu -- Kinect only):**
```bash
pip3 install pyk4a
# Note: Kinect capture requires a physical display (Terminator, not SSH)
```

All auto-labeling models (YOLO-World, SAM2) are bundled in `ultralytics`. Zero additional installs.

---

## Pipeline Overview

```
Step 1: Capture images (Kinect top-down)
Step 2: Transfer to Mac (scp)
Step 3: Auto-label with YOLO-World + SAM2
Step 4: Audit labels in LabelMe
Step 5: Convert LabelMe JSON to YOLO OBB format
Step 6: Split into train/val (80/20)
Step 7: Train YOLOv8-OBB
Step 8: Evaluate
```

---

## Step 1: Capture Images

### Kinect (Ubuntu, physical display required)

```bash
# On Ubuntu machine (Terminator, NOT SSH -- depth engine needs OpenGL)
cd ~/RIVeR-Perception-Pipeline
python3 scripts/capture_kinect.py --gui
```

- Press **SPACE** to capture, **Q** to quit
- Saves to `images/kinect/kt_001.png`, `kt_002.png`, ...
- RGB (1920x1080), top-down view

### RealSense (Mac, optional)

```bash
python3 scripts/capture_realsense.py --camera 0
# Use --list to find the correct camera index (RealSense exposes multiple UVC streams)
```

### Transfer Kinect images to Mac

```bash
# From Mac -- quote the remote path (zsh glob expansion)
scp "river@192.168.0.150:~/RIVeR-Perception-Pipeline/images/kinect/kt_*.png" images/kinect/
```

### Capture protocol

- Target: ~60+ arrangements with all objects in every scene
- Kinect stays fixed (top-down). Move objects between captures.
- Vary: position, rotation, spacing, partial occlusion, near gripper, touching/overlapping
- Include easy cases (objects spread apart) AND hard cases (clustered, near arm, partially hidden)
- Skip arrangements where any object is fully occluded (<50% visible)

---

## Step 2: Auto-Label with YOLO-World + SAM2

### What it does

Processes saved images **offline** (not live). For each image:

1. **YOLO-World** (`yolov8x-worldv2.pt`, 140MB) -- open-vocabulary detection. Given text prompts like "banana", "lime", "can", detects objects in any image.
2. **SAM2** (`sam2_b.pt`, 154MB) -- Segment Anything Model 2. Takes detection bounding boxes and produces pixel-perfect segmentation masks.
3. **mask_to_obb** -- runs `cv2.findContours` on the mask, then `cv2.minAreaRect` to get a tight oriented bounding box (4 corner points).

Both models auto-download on first use and are included in `ultralytics`.

### How to run

```bash
# Multi-class (current setup)
python3 scripts/auto_label.py --classes banana lime can

# Custom confidence threshold
python3 scripts/auto_label.py --classes banana lime can --conf 0.3
```

### Top-down detection notes

YOLO-World may struggle with objects viewed top-down. Custom text prompts can help:
- Lime from above looks like a small green ball -- "small green ball" detects better than "lime"
- Cans from above show only the circular top -- very low detection rates (~0.12 confidence)
- Use a custom labeling script with prompt mapping (see auto-label session notes)

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

### Expected output (multi-class, 60 images)

```
Images processed: 60
Banana: 56/60 auto-detected
Lime:   58/60 auto-detected
Can:    19/60 auto-detected (low confidence from top-down)
```

Typical results: 85-95% for distinctive objects (banana, lime), much lower for objects with minimal visual signal from the capture angle (can top-down).

---

## Step 3: Audit Labels in LabelMe

### Why

Auto-labeling is not perfect. Common issues:
- **Missed detections** (0 labels) -- happens with top-down views, heavy occlusion, similar colors
- **False positives** -- similar objects detected as wrong class
- **Duplicate detections** -- same object detected twice

### Setup

Copy JSON labels next to images (if not already there), fix `imagePath` to be relative:

```bash
labelme images/kinect/ --output images/kinect/
```

LabelMe will show each image with its auto-generated polygon overlaid. Use **Next Image** / **Prev Image** to navigate.

### What to check

| Issue | Action |
|-------|--------|
| Object visible, no polygon | Draw a 4-point polygon, label with exact class name |
| Object <50% visible | **Skip it** -- bad training data |
| Two polygons on same object | Delete the false positive (click shape, press Delete) |
| Polygon is loose but covers the object | **Leave it** -- rough rectangles are fine for OBB training |
| Polygon is tight and accurate | **Leave it** -- auto-label worked correctly |

### Label conventions

- **Exact lowercase class names**: `banana`, `lime`, `can` (must match `CLASS_MAP` in `labelme_to_yolo_obb.py`)
- **4-point polygons**: rotated rectangles that fit tightly around the object at any angle
- **One label per object instance**

### After review

Sync labels back to the `labels/` directory:

```bash
cp images/kinect/*.json labels/
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
CLASS_MAP = {"banana": 0, "lime": 1, "can": 2}              # current
CLASS_MAP = {"banana": 0, "lime": 1, "can": 2, "cup": 3}    # adding cup
```

Also update `dataset/data.yaml` to match:

```yaml
nc: 4
names:
  0: banana
  1: lime
  2: can
  3: cup
```

---

## Step 5: Train/Val Split

```bash
python3 scripts/split_dataset.py
```

- 80% train, 20% val (deterministic, seed=42)
- Input: images from `images/kinect/`, labels from `dataset/labels/`
- Output:
  ```
  dataset/
    images/train/   (48 images)
    images/val/     (12 images)
    labels/train/   (48 labels)
    labels/val/     (12 labels)
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
| Device | Apple Silicon CPU |

### Results (multi-class, 60 images, 2026-04-04)

| Metric | All | Banana | Lime | Can |
|--------|-----|--------|------|-----|
| Precision | 0.982 | 0.978 | 0.975 | 0.993 |
| Recall | 1.000 | 1.000 | 1.000 | 1.000 |
| mAP50 | 0.995 | 0.995 | 0.995 | 0.995 |
| mAP50-95 | 0.892 | 0.908 | 0.932 | 0.837 |

| Spec | Value |
|------|-------|
| Model size | 6.6 MB |
| Training time | 12 min |
| Inference | 17ms/image |

### Output

- Best weights: `models/multi_class_obb/weights/best.pt`
- Training curves: `models/multi_class_obb/results.png`
- Eval images: `models/multi_class_obb/eval/`

---

## Step 7: Evaluate

```bash
python3 scripts/evaluate.py
```

Runs inference on the validation set and saves images with drawn OBBs.

### What to check

- OBBs should be tight around each object
- Rotation should match the object's orientation
- No false positives (boxes on wrong objects)
- Correct class labels on each detection
- Missed detections are acceptable on heavily occluded images

### Custom evaluation

```bash
# Different weights
python3 scripts/evaluate.py --weights models/multi_class_obb/weights/best.pt

# Different images
python3 scripts/evaluate.py --source path/to/test/images/

# Different confidence threshold
python3 scripts/evaluate.py --conf 0.7
```

---

## Reproducing for New Objects

1. **Capture** ~60+ images with all objects in scene using Kinect (Step 1)
2. **Transfer** to Mac via scp
3. **Auto-label**: `python3 scripts/auto_label.py --classes obj1 obj2 obj3`
4. **Audit** in LabelMe (Step 3) -- especially for objects hard to detect from top-down
5. **Update class map** in `labelme_to_yolo_obb.py` and `data.yaml`
6. **Convert**: `python3 scripts/labelme_to_yolo_obb.py`
7. **Split**: `python3 scripts/split_dataset.py`
8. **Train**: `python3 scripts/train.py`
9. **Evaluate**: `python3 scripts/evaluate.py`
