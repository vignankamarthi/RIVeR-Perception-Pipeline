# RIVeR Perception Pipeline

**Oriented object detection + 6DOF pose estimation for robotic manipulation via YOLO OBB. Auto-labeling with YOLO-World + SAM2, dual-camera capture, training, ROS2 live inference with PoseStamped output. TDD-enforced (125 tests).**

---

## Overview

End-to-end perception pipeline: detect and track objects using oriented bounding boxes (OBB), estimate full 6DOF pose via PnP, and publish standard ROS2 PoseStamped messages for robot control.

- **Auto-labeling**: YOLO-World + SAM2 -- text-prompted, zero extra dependencies
- **YOLO OBB**: Oriented bounding box detection via YOLOv8-OBB (Ultralytics)
- **6DOF Pose**: solvePnP with IPPE solver, depth disambiguation, per-class PoseStamped publishing
- **Dual-camera**: Intel RealSense D455 (workspace) + Azure Kinect DK (top-down)
- **Reproducible**: Swap the class name and re-run for any object

---

## Quick Start

```bash
# Install (macOS Apple Silicon)
pip install pyrealsense2-macosx labelme ultralytics pytest

# Run tests
pytest tests/ -v  # 125 passing

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
| **ROS2 Detect** | **`scripts/ros_detect.py`** | **Live detection + 6DOF pose estimation** |

Full documentation: [`Documents/TRAINING_PIPELINE.md`](Documents/TRAINING_PIPELINE.md) | [`Documents/LAUNCH_DETECTION.md`](Documents/LAUNCH_DETECTION.md)

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
python3 scripts/auto_label.py --classes banana pear can
```

---

## ROS2 Live Inference + 6DOF Pose

Detection node for ROS2 Humble. Subscribes to camera RGB + depth streams, runs YOLO OBB inference, estimates 6DOF pose via solvePnP (IPPE solver for coplanar points, depth disambiguation), and publishes `geometry_msgs/PoseStamped` per detected class.

**Published topics:** `/detections/<class>/pose` (e.g., `/detections/banana/pose`)

Compatible with RVIZ, MoveIt, and tf2. Topics created dynamically per class -- multi-object ready.

**Verified performance:**
- Reprojection error: ~1.3px
- Yaw tracking: confirmed (90-degree rotation on table = ~90-degree yaw shift)
- All 6DOF logged: yaw, pitch, roll

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
  realsense/                52 RealSense captures
  kinect/                   52 Kinect captures
labels/                     LabelMe JSON annotations (hand-reviewed)
dataset/
  data.yaml                 Class config
  labels/                   YOLO OBB txt annotations
models/
  banana_obb/weights/best.pt  Trained model (6.5 MB)
Documents/
  TRAINING_PIPELINE.md      Training pipeline documentation
  LAUNCH_DETECTION.md       Launch instructions for lab machine
REFERENCE.md                Conceptual reference (ROS2, PnP, euler angles)
```
