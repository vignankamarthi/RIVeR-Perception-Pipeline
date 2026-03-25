# RIVeR Perception Pipeline

**Vision-only object detection for robotic manipulation using YOLO OBB (Oriented Bounding Boxes).**

End-to-end pipeline: camera capture, labeling, format conversion, training, and evaluation. Designed for reproducibility -- repeat the pipeline for any new object by following the documented workflow.

---

## Overview

Perception module for the RIVeR Lab's vision-only failure detection research. Detects and tracks objects in a robot's workspace using oriented bounding boxes, which capture both position and rotation -- critical for manipulation tasks where object orientation matters.

- **YOLO OBB**: Oriented bounding box detection via YOLOv8-OBB (Ultralytics)
- **Dual-camera capture**: Intel RealSense D455 (workspace view) + Azure Kinect DK (top-down view)
- **TDD-enforced**: 50 tests across all pipeline logic. No untested code.
- **Reproducible**: Documented end-to-end for adding new objects

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd RIVeR-Perception-Pipeline

# Install dependencies (macOS Apple Silicon)
pip install pyrealsense2-macosx labelme ultralytics pytest

# Verify
python3 -c "import pyrealsense2; import labelme; import ultralytics; print('all OK')"
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Pipeline

| Step | Script | What |
|------|--------|------|
| Capture (RealSense) | `scripts/capture_realsense.py` | Save RGB frames from RealSense D455 |
| Capture (Kinect) | `scripts/capture_kinect.py` | Save RGB frames from Azure Kinect DK |
| Label | LabelMe (external tool) | Draw oriented bounding boxes |
| Convert | `scripts/labelme_to_yolo_obb.py` | LabelMe JSON to YOLO OBB format |
| Split | `scripts/split_dataset.py` | 80/20 train/val split |
| Train | `scripts/train.py` | Fine-tune YOLOv8n-OBB |
| Evaluate | `scripts/evaluate.py` | Inference on val images |
| Visualize | `scripts/visualize_labels.py` | Overlay OBB labels on images |

---

## Project Structure

```
scripts/
  capture_utils.py          Shared utilities (file naming, indexing)
  capture_realsense.py      RealSense D455 capture (Mac native)
  capture_kinect.py         Azure Kinect DK capture (Ubuntu via SSH)
  labelme_to_yolo_obb.py    LabelMe JSON -> YOLO OBB format
  split_dataset.py          Train/val split with reproducible seed
  train.py                  YOLOv8-OBB fine-tuning
  evaluate.py               Inference + visual confirmation
  visualize_labels.py       OBB label overlay for spot-checking
tests/
  test_capture_utils.py     15 tests -- file naming, indexing
  test_labelme_to_yolo_obb.py  14 tests -- normalization, conversion, error handling
  test_split_dataset.py     11 tests -- split logic, no data leakage
  test_visualize_labels.py  10 tests -- label parsing, denormalization
dataset/
  data.yaml                 Class config (tracked)
  images/train/ val/        Training data (gitignored)
  labels/train/ val/        YOLO OBB labels (gitignored)
images/                     Raw captures (gitignored)
labels/                     LabelMe annotations (gitignored)
models/                     Trained weights (gitignored)
```

---

## Hardware

| Component | Role |
|-----------|------|
| Intel RealSense D455 | RGB-D, workspace view, plugged into Mac |
| Azure Kinect DK | RGB-D, top-down view, on Ubuntu lab machine |
| UR3e | Robot arm (Lorena's domain) |
| Anker USB-C Hub | Ethernet + USB 3.0 for lab network |

---

## Context

Part of the RIVeR Lab's research on vision-only failure detection in robotic manipulation. This pipeline builds the perception layer -- downstream, YOLO OBB detections feed into an unsupervised anomaly detection model that learns "normal" manipulation and flags deviations.

**Team:** Vignan Kamarthi (perception) + Lorena Genua (control)
**Lab:** RIVeR Lab, Northeastern University
