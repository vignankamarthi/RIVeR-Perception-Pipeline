"""
Tests for LabelMe JSON -> YOLO OBB format conversion.

This is the highest-value test suite in the pipeline.
Coordinate normalization bugs here silently corrupt the entire dataset.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from labelme_to_yolo_obb import (
    normalize_points,
    labelme_shape_to_yolo_obb_line,
    convert_single_file,
    CLASS_MAP,
)


class TestNormalizePoints:
    """Tests for coordinate normalization to [0, 1]."""

    def test_basic_normalization(self):
        """Points should be divided by image dimensions."""
        points = [[100, 200], [300, 200], [300, 400], [100, 400]]
        result = normalize_points(points, img_width=1000, img_height=1000)
        assert result == [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]]

    def test_full_image_bbox(self):
        """Bounding box covering full image should normalize to [0,0] to [1,1]."""
        points = [[0, 0], [640, 0], [640, 480], [0, 480]]
        result = normalize_points(points, img_width=640, img_height=480)
        assert result == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    def test_non_square_image(self):
        """Should normalize x by width and y by height independently."""
        points = [[320, 240]]
        result = normalize_points(points, img_width=1280, img_height=720)
        assert result[0][0] == pytest.approx(0.25)
        assert result[0][1] == pytest.approx(1 / 3)

    def test_zero_origin(self):
        """Point at origin should normalize to [0, 0]."""
        points = [[0, 0]]
        result = normalize_points(points, img_width=100, img_height=100)
        assert result == [[0.0, 0.0]]

    def test_preserves_four_points(self):
        """Should return exactly 4 points for a 4-point input."""
        points = [[10, 20], [30, 20], [30, 40], [10, 40]]
        result = normalize_points(points, img_width=100, img_height=100)
        assert len(result) == 4


class TestLabelmeShapeToYoloObbLine:
    """Tests for converting a single LabelMe shape to a YOLO OBB line."""

    def test_basic_conversion(self):
        """Should produce 'class_id x1 y1 x2 y2 x3 y3 x4 y4' format."""
        shape = {
            "label": "banana",
            "points": [[100, 200], [300, 200], [300, 400], [100, 400]],
            "shape_type": "polygon",
        }
        line = labelme_shape_to_yolo_obb_line(shape, img_width=1000, img_height=1000)
        parts = line.split()
        assert len(parts) == 9  # class_id + 4 x,y pairs
        assert parts[0] == "0"  # banana = class 0

    def test_coordinates_are_normalized(self):
        """All coordinates should be in [0, 1]."""
        shape = {
            "label": "banana",
            "points": [[100, 200], [300, 200], [300, 400], [100, 400]],
            "shape_type": "polygon",
        }
        line = labelme_shape_to_yolo_obb_line(shape, img_width=1000, img_height=1000)
        parts = line.split()
        for coord_str in parts[1:]:
            coord = float(coord_str)
            assert 0.0 <= coord <= 1.0, f"Coordinate {coord} out of [0,1] range"

    def test_unknown_class_raises(self):
        """Unknown class label should raise ValueError."""
        shape = {
            "label": "unknown_object",
            "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "shape_type": "polygon",
        }
        with pytest.raises(ValueError, match="unknown_object"):
            labelme_shape_to_yolo_obb_line(shape, img_width=100, img_height=100)

    def test_wrong_point_count_raises(self):
        """Shapes with != 4 points should raise ValueError."""
        shape = {
            "label": "banana",
            "points": [[0, 0], [1, 0], [1, 1]],  # only 3 points
            "shape_type": "polygon",
        }
        with pytest.raises(ValueError, match="4 points"):
            labelme_shape_to_yolo_obb_line(shape, img_width=100, img_height=100)

    def test_class_id_mapping(self):
        """Class IDs should match CLASS_MAP."""
        for label, expected_id in CLASS_MAP.items():
            shape = {
                "label": label,
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "shape_type": "polygon",
            }
            line = labelme_shape_to_yolo_obb_line(shape, img_width=100, img_height=100)
            assert line.split()[0] == str(expected_id)


class TestConvertSingleFile:
    """Tests for converting a full LabelMe JSON file to YOLO OBB format."""

    def _write_labelme_json(self, path: Path, shapes: list, img_w: int, img_h: int):
        """Helper: write a minimal LabelMe JSON file."""
        data = {
            "version": "5.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": "test.png",
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w,
        }
        path.write_text(json.dumps(data))

    def test_single_object(self, tmp_path):
        """Single banana annotation should produce one line."""
        json_path = tmp_path / "test.json"
        self._write_labelme_json(
            json_path,
            shapes=[
                {
                    "label": "banana",
                    "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
                    "shape_type": "polygon",
                    "flags": {},
                }
            ],
            img_w=640,
            img_h=480,
        )

        lines = convert_single_file(json_path)
        assert len(lines) == 1
        assert lines[0].startswith("0 ")

    def test_multiple_objects(self, tmp_path):
        """Multiple annotations should produce multiple lines."""
        json_path = tmp_path / "test.json"
        self._write_labelme_json(
            json_path,
            shapes=[
                {
                    "label": "banana",
                    "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                    "shape_type": "polygon",
                    "flags": {},
                },
                {
                    "label": "banana",
                    "points": [[100, 100], [150, 100], [150, 150], [100, 150]],
                    "shape_type": "polygon",
                    "flags": {},
                },
            ],
            img_w=640,
            img_h=480,
        )

        lines = convert_single_file(json_path)
        assert len(lines) == 2

    def test_empty_annotations(self, tmp_path):
        """No shapes should produce empty output."""
        json_path = tmp_path / "test.json"
        self._write_labelme_json(json_path, shapes=[], img_w=640, img_h=480)

        lines = convert_single_file(json_path)
        assert len(lines) == 0

    def test_output_format_parseable(self, tmp_path):
        """Output lines should be parseable as: int float float float float float float float float."""
        json_path = tmp_path / "test.json"
        self._write_labelme_json(
            json_path,
            shapes=[
                {
                    "label": "banana",
                    "points": [[50, 50], [200, 50], [200, 150], [50, 150]],
                    "shape_type": "polygon",
                    "flags": {},
                }
            ],
            img_w=640,
            img_h=480,
        )

        lines = convert_single_file(json_path)
        parts = lines[0].split()
        assert len(parts) == 9
        int(parts[0])  # class_id should be int
        for p in parts[1:]:
            float(p)  # coordinates should be float
