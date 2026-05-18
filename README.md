# RIVeR Perception Pipeline

> **ARCHIVED 2026-05-18.** Complete and frozen -- foundational entry work in Northeastern's RIVeR Lab. Not under active development.

**Multi-class oriented object detection + 6DOF pose estimation for robotic manipulation via YOLO OBB. Auto-labeling with YOLO-World + SAM2, Kinect top-down capture, training, ROS2 live inference with PoseStamped output. TDD-enforced (125 tests).**

---

## Overview

End-to-end perception pipeline: detect and track objects using oriented bounding boxes (OBB), estimate full 6DOF pose via PnP, and publish standard ROS2 PoseStamped messages for robot control.

- **Multi-class**: Banana, lime, can (3 classes, extensible)
- **Auto-labeling**: YOLO-World + SAM2 -- text-prompted, zero extra dependencies
- **YOLO OBB**: Oriented bounding box detection via YOLOv8-OBB (Ultralytics)
- **6DOF Pose**: solvePnP with IPPE solver, depth disambiguation, per-class PoseStamped publishing
- **Camera**: Azure Kinect DK (top-down view, primary deployment camera)
- **Reproducible**: Swap the class names and re-run for any object

---

## Quick Start

```bash
# Install (macOS Apple Silicon)
pip install ultralytics labelme pytest

# Run tests
pytest tests/ -v  # 125 passing

# Auto-label images (offline batch)
python3 scripts/auto_label.py --classes banana lime can

# Train
python3 scripts/train.py
```

---

## Pipeline

| Step | Script | What |
|------|--------|------|
| Capture (Kinect) | `scripts/capture_kinect.py` | RGB frames from Azure Kinect DK (top-down) |
| Capture (RealSense) | `scripts/capture_realsense.py` | RGB frames from RealSense D455 (side view) |
| Auto-label | `scripts/auto_label.py` | YOLO-World + SAM2 auto-labeling |
| Review | LabelMe (external) | Audit + correct auto-labels |
| Convert | `scripts/labelme_to_yolo_obb.py` | LabelMe JSON to YOLO OBB format |
| Split | `scripts/split_dataset.py` | 80/20 train/val split |
| Train | `scripts/train.py` | Fine-tune YOLOv8n-OBB |
| Evaluate | `scripts/evaluate.py` | Inference on val set |
| Visualize | `scripts/visualize_labels.py` | Overlay labels for spot-checking |
| **ROS2 Detect** | **`scripts/ros_detect.py`** | **Live detection + 6DOF pose estimation** |

Full documentation: [`Documents/TRAINING_PIPELINE.md`](Documents/TRAINING_PIPELINE.md) | [`Documents/LAUNCH_DETECTION.md`](Documents/LAUNCH_DETECTION.md)

---

## Trained Model (Multi-Class)

| Metric | All | Banana | Lime | Can |
|--------|-----|--------|------|-----|
| Precision | 0.982 | 0.978 | 0.975 | 0.993 |
| Recall | 1.000 | 1.000 | 1.000 | 1.000 |
| mAP50 | 0.995 | 0.995 | 0.995 | 0.995 |
| mAP50-95 | 0.892 | 0.908 | 0.932 | 0.837 |

| Spec | Value |
|------|-------|
| Model size | 6.6 MB |
| Inference | 17ms/image |
| Dataset | 60 images (Kinect top-down) |
| Training | 50 epochs, 12 min (Apple M2 Pro CPU) |

Weights: `models/multi_class_obb/weights/best.pt`

---

## Auto-Labeling System

The auto-labeler replaces manual LabelMe annotation with a text-prompted pipeline:

1. **YOLO-World** (`yolov8x-worldv2.pt`) -- detects objects from text prompts (e.g., "banana")
2. **SAM2** (`sam2_b.pt`) -- segments detected regions into pixel-perfect masks
3. **mask_to_obb** -- converts masks to oriented bounding boxes via `cv2.minAreaRect`

Outputs both LabelMe JSON (for review) and YOLO OBB txt (for training). Zero additional pip installs -- both models are bundled in `ultralytics`.

```bash
python3 scripts/auto_label.py --classes banana lime can
```

Note: top-down views may require custom text prompts for best results (e.g., "small green ball" for lime from above).

---

## ROS2 Live Inference + 6DOF Pose

Detection node for ROS2 Humble. Subscribes to camera RGB + depth streams, runs YOLO OBB inference, estimates 6DOF pose via solvePnP (IPPE solver for coplanar points, depth disambiguation), and publishes `geometry_msgs/PoseStamped` per detected class.

**Published topics:** `/detections/<class>/pose` (e.g., `/detections/banana/pose`, `/detections/lime/pose`, `/detections/can/pose`)

Compatible with RVIZ, MoveIt, and tf2. Topics created dynamically per class.

**Verified performance:**
- Reprojection error: ~1.3px
- Yaw tracking: confirmed (90-degree rotation on table = ~90-degree yaw shift)
- All 6DOF logged: yaw, pitch, roll

---

## Project Structure

```
scripts/
  capture_utils.py          Shared utilities (file naming, indexing)
  capture_realsense.py      RealSense D455 capture (Mac native via UVC)
  capture_kinect.py         Azure Kinect DK capture (Ubuntu, physical terminal)
  auto_label.py             YOLO-World + SAM2 auto-labeling
  labelme_to_yolo_obb.py    LabelMe JSON -> YOLO OBB format
  split_dataset.py          Train/val split with reproducible seed
  train.py                  YOLOv8-OBB fine-tuning
  evaluate.py               Inference + visual confirmation
  visualize_labels.py       OBB label overlay for spot-checking
  detect_utils.py           Detection + 6DOF pose logic (PnP, depth, fusion)
  ros_detect.py             ROS2 live detection + PoseStamped publisher
tests/
  test_auto_label.py        20 tests (conversion + orchestration)
  test_capture_utils.py     15 tests
  test_detect_utils.py      55 tests (detection + 6DOF pose)
  test_labelme_to_yolo_obb.py  14 tests
  test_split_dataset.py     11 tests
  test_visualize_labels.py  10 tests
images/
  kinect/                   60 Kinect captures (top-down, multi-class)
labels/                     LabelMe JSON annotations (hand-reviewed)
dataset/
  data.yaml                 Class config (3 classes: banana, lime, can)
  labels/                   YOLO OBB txt annotations
  images/train/             48 training images
  images/val/               12 validation images
models/
  multi_class_obb/          Trained multi-class model (banana + lime + can)
  banana_obb/               Legacy banana-only model
Documents/
  TRAINING_PIPELINE.md      Training pipeline documentation
  LAUNCH_DETECTION.md       Launch instructions for lab machine
```
