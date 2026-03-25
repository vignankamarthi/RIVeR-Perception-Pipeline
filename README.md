# RIVeR Perception Pipeline

**Oriented object detection for robotic manipulation via YOLO OBB. Dual-camera capture, LabelMe labeling, format conversion, fine-tuning, evaluation. TDD-enforced (50 tests).**

---

## Overview

End-to-end perception pipeline: detect and track objects using oriented bounding boxes (OBB), which capture position and rotation -- critical for manipulation where object orientation matters.

- **YOLO OBB**: Oriented bounding box detection via YOLOv8-OBB (Ultralytics)
- **Dual-camera**: Intel RealSense D455 (workspace) + Azure Kinect DK (top-down)
- **Reproducible**: Documented for repeating with new objects

---

## Quick Start

```bash
# Install (macOS Apple Silicon)
pip install pyrealsense2-macosx labelme ultralytics pytest

# Verify
python3 -c "import pyrealsense2; import labelme; import ultralytics; print('OK')"

# Run tests
pytest tests/ -v
```

---

## Pipeline

| Step | Script | What |
|------|--------|------|
| Capture (RealSense) | `scripts/capture_realsense.py` | RGB frames from RealSense D455 |
| Capture (Kinect) | `scripts/capture_kinect.py` | RGB frames from Azure Kinect DK |
| Label | LabelMe (external) | Oriented bounding boxes |
| Convert | `scripts/labelme_to_yolo_obb.py` | LabelMe JSON to YOLO OBB format |
| Split | `scripts/split_dataset.py` | 80/20 train/val split |
| Train | `scripts/train.py` | Fine-tune YOLOv8n-OBB |
| Evaluate | `scripts/evaluate.py` | Inference on val set |
| Visualize | `scripts/visualize_labels.py` | Overlay labels for spot-checking |

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
  test_capture_utils.py     15 tests
  test_labelme_to_yolo_obb.py  14 tests
  test_split_dataset.py     11 tests
  test_visualize_labels.py  10 tests
dataset/
  data.yaml                 Class config (tracked)
  images/ labels/           Data (gitignored)
```
