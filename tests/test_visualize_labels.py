"""
Tests for visualize_labels.py -- label parsing and denormalization logic.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from visualize_labels import read_yolo_obb_label, denormalize_points


class TestReadYoloObbLabel:
    """Tests for parsing YOLO OBB label files."""

    def test_single_label(self, tmp_path):
        """Single annotation should return one dict."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")
        result = read_yolo_obb_label(label_file)
        assert len(result) == 1
        assert result[0]["class_id"] == 0
        assert result[0]["class_name"] == "banana"
        assert len(result[0]["points"]) == 4

    def test_multiple_labels(self, tmp_path):
        """Multiple lines should return multiple dicts."""
        label_file = tmp_path / "test.txt"
        label_file.write_text(
            "0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n"
            "0 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n"
        )
        result = read_yolo_obb_label(label_file)
        assert len(result) == 2

    def test_empty_file(self, tmp_path):
        """Empty file should return empty list."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("")
        result = read_yolo_obb_label(label_file)
        assert result == []

    def test_malformed_line_skipped(self, tmp_path):
        """Lines with wrong number of fields should be skipped."""
        label_file = tmp_path / "test.txt"
        label_file.write_text(
            "0 0.1 0.2 0.3\n"  # too few fields
            "0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n"  # valid
        )
        result = read_yolo_obb_label(label_file)
        assert len(result) == 1

    def test_points_are_floats(self, tmp_path):
        """Parsed points should be float values."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.15 0.25 0.35 0.25 0.35 0.45 0.15 0.45\n")
        result = read_yolo_obb_label(label_file)
        for point in result[0]["points"]:
            assert isinstance(point[0], float)
            assert isinstance(point[1], float)

    def test_unknown_class_id(self, tmp_path):
        """Unknown class ID should still parse with fallback name."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("99 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")
        result = read_yolo_obb_label(label_file)
        assert result[0]["class_id"] == 99
        assert result[0]["class_name"] == "class_99"


class TestDenormalizePoints:
    """Tests for converting normalized coords back to pixels."""

    def test_basic_denormalization(self):
        """Should multiply by image dimensions."""
        points = [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]
        result = denormalize_points(points, img_width=640, img_height=480)
        assert result == [[320, 240], [640, 240], [640, 480], [320, 480]]

    def test_origin(self):
        """[0, 0] should map to pixel [0, 0]."""
        result = denormalize_points([[0.0, 0.0]], img_width=1280, img_height=720)
        assert result == [[0, 0]]

    def test_returns_integers(self):
        """Pixel coordinates should be integers."""
        result = denormalize_points([[0.333, 0.666]], img_width=100, img_height=100)
        assert all(isinstance(v, int) for point in result for v in point)

    def test_roundtrip_consistency(self):
        """Normalize then denormalize should approximately recover original."""
        from labelme_to_yolo_obb import normalize_points

        original = [[100, 200], [300, 200], [300, 400], [100, 400]]
        normalized = normalize_points(original, img_width=640, img_height=480)
        recovered = denormalize_points(normalized, img_width=640, img_height=480)
        assert recovered == original
