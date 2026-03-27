# RIVeR Perception Pipeline

**Oriented object detection for robotic manipulation via YOLO OBB. Auto-labeling with YOLO-World + SAM2, dual-camera capture, training, evaluation, ROS2 live inference. TDD-enforced (91 tests).**

---

## Overview

End-to-end perception pipeline: detect and track objects using oriented bounding boxes (OBB), which capture position and rotation -- critical for manipulation where object orientation matters.

- **Auto-labeling**: YOLO-World (open-vocabulary detection) + SAM2 (segmentation) -- text-prompted, zero extra dependencies
- **YOLO OBB**: Oriented bounding box detection via YOLOv8-OBB (Ultralytics)
- **Dual-camera**: Intel RealSense D455 (workspace) + Azure Kinect DK (top-down)
- **ROS2 inference**: Live dual-camera detection with depth-based 3D localization
- **Reproducible**: Swap the class name and re-run for any object

---

## Quick Start

```bash
# Install (macOS Apple Silicon)
pip install pyrealsense2-macosx labelme ultralytics pytest

# Run tests
pytest tests/ -v  # 91 passing

# Auto-label images (offline batch)
python3 scripts/auto_label.py --classes banana

# Train
python3 scripts/train.py
```

---

## Pipeline

| Step | Script | What |
|------|--------|------|
| Capture (RealSense) | `scripts/capture_realsense.py` | RGB frames from RealSense D455 |
| Capture (Kinect) | `scripts/capture_kinect.py` | RGB frames from Azure Kinect DK |
| Auto-label | `scripts/auto_label.py` | YOLO-World + SAM2 auto-labeling |
| Review | LabelMe (external) | Audit + correct auto-labels |
| Convert | `scripts/labelme_to_yolo_obb.py` | LabelMe JSON to YOLO OBB format |
| Split | `scripts/split_dataset.py` | 80/20 train/val split |
| Train | `scripts/train.py` | Fine-tune YOLOv8n-OBB |
| Evaluate | `scripts/evaluate.py` | Inference on val set |
| Visualize | `scripts/visualize_labels.py` | Overlay labels for spot-checking |
| **ROS2 Detect** | **`scripts/ros_detect.py`** | **Live dual-camera OBB detection** |

Full documentation: [`Documents/TRAINING_PIPELINE.md`](Documents/TRAINING_PIPELINE.md)

---

## Trained Model (Banana)

| Metric | Value |
|--------|-------|
| Precision | 1.000 |
| Recall | 0.889 |
| mAP50 | 0.921 |
| mAP50-95 | 0.778 |
| Model size | 6.5 MB |
| Inference | 25ms/image |
| Dataset | 104 images (52 RealSense + 52 Kinect) |

Weights: `models/banana_obb/weights/best.pt`

---

## Auto-Labeling System

The auto-labeler replaces manual LabelMe annotation with a text-prompted pipeline:

1. **YOLO-World** (`yolov8x-worldv2.pt`) -- detects objects from text prompts (e.g., "banana")
2. **SAM2** (`sam2_b.pt`) -- segments detected regions into pixel-perfect masks
3. **mask_to_obb** -- converts masks to oriented bounding boxes via `cv2.minAreaRect`

Outputs both LabelMe JSON (for review) and YOLO OBB txt (for training). Zero additional pip installs -- both models are bundled in `ultralytics`.

```bash
# Single class
python3 scripts/auto_label.py --classes banana

# Multiple classes
python3 scripts/auto_label.py --classes banana pear can
```

---

## ROS2 Live Inference

Dual-camera detection node for ROS2 Humble. Subscribes to RealSense + Kinect RGB/depth streams, runs YOLO OBB inference, computes 3D object positions via depth + camera intrinsics, and publishes detections as JSON.

```bash
# On Ubuntu lab machine (3 terminals):

# Terminal 1 (physical): Kinect driver
source /opt/ros/humble/setup.zsh && source ~/ros2_ws/install/setup.zsh
ros2 launch azure_kinect_ros_driver driver.launch.py

# Terminal 2: RealSense driver
source /opt/ros/humble/setup.zsh
ros2 launch realsense2_camera rs_launch.py depth_module.enable:=true

# Terminal 3: Detection node (with live visualization)
source /opt/ros/humble/setup.zsh && source ~/ros2_ws/install/setup.zsh
python3 scripts/ros_detect.py --visualize
```

Publishes to: `/detections/banana_obb` (JSON with class, confidence, OBB, 3D position)

---

## Project Structure

```
scripts/
  capture_utils.py          Shared utilities (file naming, indexing)
  capture_realsense.py      RealSense D455 capture (Mac native)
  capture_kinect.py         Azure Kinect DK capture (Ubuntu via SSH)
  auto_label.py             YOLO-World + SAM2 auto-labeling
  labelme_to_yolo_obb.py    LabelMe JSON -> YOLO OBB format
  split_dataset.py          Train/val split with reproducible seed
  train.py                  YOLOv8-OBB fine-tuning
  evaluate.py               Inference + visual confirmation
  visualize_labels.py       OBB label overlay for spot-checking
  detect_utils.py           Detection logic (parsing, depth, 3D conversion)
  ros_detect.py             ROS2 dual-camera live detection node
tests/
  test_auto_label.py        20 tests (conversion + orchestration)
  test_capture_utils.py     15 tests
  test_detect_utils.py      21 tests (detection logic)
  test_labelme_to_yolo_obb.py  14 tests
  test_split_dataset.py     11 tests
  test_visualize_labels.py  10 tests
images/
  realsense/                52 RealSense captures (rs_001.png ...)
  kinect/                   52 Kinect captures (kt_001.png ...)
labels/                     LabelMe JSON annotations (hand-reviewed)
dataset/
  data.yaml                 Class config
  labels/                   YOLO OBB txt annotations
models/
  banana_obb/weights/best.pt  Trained model (6.5 MB)
Documents/
  TRAINING_PIPELINE.md      Full pipeline documentation
```
