"""
Tests for auto_label.py -- YOLO-World + SAM2 auto-labeling pipeline.

Tests the pure logic: mask-to-OBB conversion, output formatting, file generation,
and orchestration (process_detections, save_labels).
Model inference is hardware-dependent (tested manually by running the script).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from auto_label import (
    mask_to_obb,
    obb_to_yolo_line,
    obb_to_labelme_shape,
    make_labelme_json,
    process_detections,
    save_labels,
)


class TestMaskToObb:
    """Tests for converting binary masks to oriented bounding boxes."""

    def test_rectangular_mask(self):
        """Axis-aligned rectangle mask should produce 4 corner points."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:80] = 1
        box = mask_to_obb(mask)
        assert box is not None
        assert box.shape == (4, 2)

    def test_rotated_mask(self):
        """Diagonal mask should produce a rotated bounding box."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        pts = np.array([[50, 100], [100, 50], [150, 100], [100, 150]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        box = mask_to_obb(mask)
        assert box is not None
        assert box.shape == (4, 2)

    def test_empty_mask_returns_none(self):
        """Empty mask (no object) should return None."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        box = mask_to_obb(mask)
        assert box is None

    def test_tiny_mask_returns_none(self):
        """Mask smaller than noise threshold should return None."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1  # single pixel
        box = mask_to_obb(mask)
        assert box is None

    def test_box_points_are_float(self):
        """cv2.boxPoints returns float coordinates."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:50, 10:90] = 1
        box = mask_to_obb(mask)
        assert box.dtype == np.float32 or box.dtype == np.float64

    def test_box_within_image_bounds(self):
        """OBB points should be within image dimensions (approximately)."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:90] = 1
        box = mask_to_obb(mask)
        assert box[:, 0].min() >= -1  # small tolerance for float rounding
        assert box[:, 0].max() <= 101
        assert box[:, 1].min() >= -1
        assert box[:, 1].max() <= 101


class TestObbToYoloLine:
    """Tests for converting OBB to YOLO OBB format string."""

    def test_format(self):
        """Should produce 'class_id x1 y1 x2 y2 x3 y3 x4 y4'."""
        box = np.array([[10, 20], [90, 20], [90, 80], [10, 80]], dtype=np.float32)
        line = obb_to_yolo_line(box, class_id=0, img_w=100, img_h=100)
        parts = line.split()
        assert len(parts) == 9
        assert parts[0] == "0"

    def test_normalized_coords(self):
        """Coordinates should be normalized to [0, 1]."""
        box = np.array([[50, 50], [100, 50], [100, 100], [50, 100]], dtype=np.float32)
        line = obb_to_yolo_line(box, class_id=0, img_w=200, img_h=200)
        parts = line.split()
        for coord_str in parts[1:]:
            coord = float(coord_str)
            assert 0.0 <= coord <= 1.0

    def test_class_id_preserved(self):
        """Class ID should match input."""
        box = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        line = obb_to_yolo_line(box, class_id=3, img_w=100, img_h=100)
        assert line.startswith("3 ")


class TestObbToLabelmeShape:
    """Tests for converting OBB to LabelMe shape dict."""

    def test_shape_structure(self):
        """Should have required LabelMe fields."""
        box = np.array([[10, 20], [90, 20], [90, 80], [10, 80]], dtype=np.float32)
        shape = obb_to_labelme_shape(box, "banana")
        assert shape["label"] == "banana"
        assert shape["shape_type"] == "polygon"
        assert len(shape["points"]) == 4

    def test_points_are_lists(self):
        """Points should be list of lists (JSON-serializable)."""
        box = np.array([[10, 20], [90, 20], [90, 80], [10, 80]], dtype=np.float32)
        shape = obb_to_labelme_shape(box, "banana")
        for point in shape["points"]:
            assert isinstance(point, list)
            assert len(point) == 2


class TestMakeLabelmeJson:
    """Tests for building complete LabelMe JSON."""

    def test_required_fields(self):
        """Should have all LabelMe required fields."""
        shapes = [{"label": "banana", "points": [[0, 0], [1, 0], [1, 1], [0, 1]], "shape_type": "polygon", "group_id": None, "description": "", "flags": {}}]
        data = make_labelme_json(Path("test.png"), shapes, img_h=480, img_w=640)
        assert "shapes" in data
        assert "imageHeight" in data
        assert "imageWidth" in data
        assert "imagePath" in data
        assert data["imageHeight"] == 480
        assert data["imageWidth"] == 640

    def test_json_serializable(self):
        """Output should be JSON-serializable."""
        shapes = [{"label": "banana", "points": [[0, 0], [1, 0], [1, 1], [0, 1]], "shape_type": "polygon", "group_id": None, "description": "", "flags": {}}]
        data = make_labelme_json(Path("test.png"), shapes, img_h=480, img_w=640)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# Orchestration tests -- process_detections + save_labels
# No model inference, just masks → output formats → files.
# ---------------------------------------------------------------------------

def _make_mask(h, w, y1, y2, x1, x2):
    """Helper: create a binary mask with a filled rectangle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


class TestProcessDetections:
    """Tests for process_detections: masks + metadata → (shapes, yolo_lines)."""

    def test_single_detection(self):
        """One mask should produce one LabelMe shape and one YOLO line."""
        masks = [_make_mask(100, 100, 20, 60, 30, 80)]
        class_names = ["banana"]
        class_map = {"banana": 0}
        shapes, yolo_lines = process_detections(masks, class_names, class_map, img_w=100, img_h=100)
        assert len(shapes) == 1
        assert len(yolo_lines) == 1
        assert shapes[0]["label"] == "banana"
        assert yolo_lines[0].startswith("0 ")

    def test_multiple_detections(self):
        """Two masks should produce two shapes and two YOLO lines."""
        masks = [
            _make_mask(200, 200, 10, 50, 10, 90),
            _make_mask(200, 200, 100, 180, 50, 150),
        ]
        class_names = ["banana", "banana"]
        class_map = {"banana": 0}
        shapes, yolo_lines = process_detections(masks, class_names, class_map, img_w=200, img_h=200)
        assert len(shapes) == 2
        assert len(yolo_lines) == 2

    def test_zero_detections(self):
        """Empty mask list should produce empty outputs."""
        shapes, yolo_lines = process_detections([], [], {"banana": 0}, img_w=100, img_h=100)
        assert shapes == []
        assert yolo_lines == []

    def test_tiny_mask_filtered_out(self):
        """Mask too small for mask_to_obb should be skipped, not crash."""
        tiny_mask = np.zeros((100, 100), dtype=np.uint8)
        tiny_mask[50, 50] = 1  # single pixel -- below min_area
        masks = [tiny_mask]
        class_names = ["banana"]
        class_map = {"banana": 0}
        shapes, yolo_lines = process_detections(masks, class_names, class_map, img_w=100, img_h=100)
        assert len(shapes) == 0
        assert len(yolo_lines) == 0


class TestSaveLabels:
    """Tests for save_labels: write LabelMe JSON + YOLO OBB txt to disk."""

    def test_labelme_json_written(self, tmp_path):
        """Should write a valid LabelMe JSON file."""
        labels_dir = tmp_path / "labels"
        yolo_dir = tmp_path / "yolo"
        img_path = Path("test_img.png")
        shapes = [obb_to_labelme_shape(
            np.array([[10, 20], [90, 20], [90, 80], [10, 80]], dtype=np.float32),
            "banana",
        )]
        yolo_lines = ["0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8"]

        save_labels(img_path, shapes, yolo_lines, labels_dir, yolo_dir, img_h=100, img_w=100)

        json_path = labels_dir / "test_img.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["imageWidth"] == 100
        assert len(data["shapes"]) == 1

    def test_yolo_txt_written(self, tmp_path):
        """Should write a YOLO OBB text file with correct content."""
        labels_dir = tmp_path / "labels"
        yolo_dir = tmp_path / "yolo"
        img_path = Path("test_img.png")
        shapes = []
        yolo_lines = ["0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8"]

        save_labels(img_path, shapes, yolo_lines, labels_dir, yolo_dir, img_h=100, img_w=100)

        txt_path = yolo_dir / "test_img.txt"
        assert txt_path.exists()
        content = txt_path.read_text().strip()
        assert content == "0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8"

    def test_empty_detection_still_writes_files(self, tmp_path):
        """Even with no detections, should write empty files (so downstream knows it was processed)."""
        labels_dir = tmp_path / "labels"
        yolo_dir = tmp_path / "yolo"
        img_path = Path("empty_img.png")

        save_labels(img_path, [], [], labels_dir, yolo_dir, img_h=100, img_w=100)

        assert (labels_dir / "empty_img.json").exists()
        assert (yolo_dir / "empty_img.txt").exists()
