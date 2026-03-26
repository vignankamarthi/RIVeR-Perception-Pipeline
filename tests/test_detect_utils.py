"""
Tests for detect_utils.py -- pure logic for ROS2 detection node.

Tests parsing YOLO OBB results, depth lookup, pixel-to-3D conversion,
detection formatting, and confidence filtering.
No ROS2 dependency -- runs on Mac.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from detect_utils import (
    parse_obb_results,
    get_depth_at_pixel,
    pixel_to_3d,
    format_detections_json,
    filter_detections,
)


# ---------------------------------------------------------------------------
# parse_obb_results: YOLO OBB output -> structured dicts
# ---------------------------------------------------------------------------

class TestParseObbResults:
    """Tests for parsing YOLO OBB inference output into structured dicts."""

    def test_single_detection(self):
        """One detection should produce one dict with required fields."""
        obb_points = [np.array([[10, 20], [90, 20], [90, 80], [10, 80]], dtype=np.float32)]
        class_ids = [0]
        confidences = [0.95]
        class_names = {0: "banana"}

        results = parse_obb_results(obb_points, class_ids, confidences, class_names)

        assert len(results) == 1
        det = results[0]
        assert det["class_name"] == "banana"
        assert det["class_id"] == 0
        assert det["confidence"] == 0.95
        assert "obb_points" in det
        assert "center_pixel" in det
        assert len(det["obb_points"]) == 4

    def test_multiple_detections(self):
        """Multiple detections should all be parsed."""
        obb_points = [
            np.array([[10, 20], [90, 20], [90, 80], [10, 80]], dtype=np.float32),
            np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32),
        ]
        class_ids = [0, 0]
        confidences = [0.95, 0.80]
        class_names = {0: "banana"}

        results = parse_obb_results(obb_points, class_ids, confidences, class_names)
        assert len(results) == 2

    def test_empty_detections(self):
        """No detections should return empty list."""
        results = parse_obb_results([], [], [], {0: "banana"})
        assert results == []

    def test_center_pixel_calculated(self):
        """Center pixel should be mean of OBB corners."""
        obb_points = [np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)]
        class_ids = [0]
        confidences = [0.9]
        class_names = {0: "banana"}

        results = parse_obb_results(obb_points, class_ids, confidences, class_names)
        cx, cy = results[0]["center_pixel"]
        assert abs(cx - 50.0) < 1.0
        assert abs(cy - 50.0) < 1.0

    def test_obb_points_are_lists(self):
        """OBB points should be JSON-serializable lists, not numpy arrays."""
        obb_points = [np.array([[10, 20], [90, 20], [90, 80], [10, 80]], dtype=np.float32)]
        class_ids = [0]
        confidences = [0.95]
        class_names = {0: "banana"}

        results = parse_obb_results(obb_points, class_ids, confidences, class_names)
        for point in results[0]["obb_points"]:
            assert isinstance(point, list)


# ---------------------------------------------------------------------------
# get_depth_at_pixel: depth image + pixel coords -> depth value
# ---------------------------------------------------------------------------

class TestGetDepthAtPixel:
    """Tests for depth lookup from a depth image."""

    def test_basic_lookup(self):
        """Should return depth value at the given pixel."""
        depth = np.full((100, 100), 1500, dtype=np.uint16)  # 1500mm everywhere
        val = get_depth_at_pixel(depth, 50, 50)
        assert val == 1500

    def test_windowed_median(self):
        """Should use a window and take median to reduce noise."""
        depth = np.full((100, 100), 1000, dtype=np.uint16)
        depth[50, 50] = 0  # noisy pixel
        val = get_depth_at_pixel(depth, 50, 50, window_size=5)
        assert val > 0  # median should ignore the single zero

    def test_out_of_bounds_returns_zero(self):
        """Pixel outside image should return 0."""
        depth = np.full((100, 100), 1500, dtype=np.uint16)
        val = get_depth_at_pixel(depth, 200, 200)
        assert val == 0

    def test_all_zero_window_returns_zero(self):
        """If entire window is zero (no depth), return 0."""
        depth = np.zeros((100, 100), dtype=np.uint16)
        val = get_depth_at_pixel(depth, 50, 50, window_size=5)
        assert val == 0


# ---------------------------------------------------------------------------
# pixel_to_3d: pixel coords + depth + intrinsics -> 3D point
# ---------------------------------------------------------------------------

class TestPixelTo3d:
    """Tests for converting pixel + depth to 3D coordinates."""

    def test_center_pixel(self):
        """Principal point with known depth should give (0, 0, depth)."""
        # Camera intrinsics: fx=500, fy=500, cx=320, cy=240
        result = pixel_to_3d(320, 240, 1.0, fx=500, fy=500, cx=320, cy=240)
        assert abs(result["x"]) < 0.001
        assert abs(result["y"]) < 0.001
        assert abs(result["z"] - 1.0) < 0.001

    def test_offset_pixel(self):
        """Pixel offset from center should give proportional 3D offset."""
        result = pixel_to_3d(420, 240, 2.0, fx=500, fy=500, cx=320, cy=240)
        # x = (420 - 320) * 2.0 / 500 = 0.4
        assert abs(result["x"] - 0.4) < 0.001
        assert abs(result["y"]) < 0.001
        assert abs(result["z"] - 2.0) < 0.001

    def test_zero_depth_returns_none(self):
        """Zero depth (no measurement) should return None."""
        result = pixel_to_3d(320, 240, 0.0, fx=500, fy=500, cx=320, cy=240)
        assert result is None

    def test_result_has_xyz(self):
        """Result should have x, y, z keys."""
        result = pixel_to_3d(100, 100, 1.5, fx=500, fy=500, cx=320, cy=240)
        assert "x" in result
        assert "y" in result
        assert "z" in result


# ---------------------------------------------------------------------------
# filter_detections: filter by confidence threshold
# ---------------------------------------------------------------------------

class TestFilterDetections:
    """Tests for confidence-based filtering."""

    def test_filter_low_confidence(self):
        """Detections below threshold should be removed."""
        detections = [
            {"class_name": "banana", "confidence": 0.9},
            {"class_name": "banana", "confidence": 0.3},
            {"class_name": "banana", "confidence": 0.7},
        ]
        filtered = filter_detections(detections, min_confidence=0.5)
        assert len(filtered) == 2
        assert all(d["confidence"] >= 0.5 for d in filtered)

    def test_all_pass(self):
        """All detections above threshold should remain."""
        detections = [
            {"class_name": "banana", "confidence": 0.9},
            {"class_name": "banana", "confidence": 0.8},
        ]
        filtered = filter_detections(detections, min_confidence=0.5)
        assert len(filtered) == 2

    def test_all_filtered(self):
        """All below threshold returns empty list."""
        detections = [
            {"class_name": "banana", "confidence": 0.1},
            {"class_name": "banana", "confidence": 0.2},
        ]
        filtered = filter_detections(detections, min_confidence=0.5)
        assert filtered == []

    def test_empty_input(self):
        """Empty list in, empty list out."""
        filtered = filter_detections([], min_confidence=0.5)
        assert filtered == []


# ---------------------------------------------------------------------------
# format_detections_json: detections -> JSON string for ROS publishing
# ---------------------------------------------------------------------------

class TestFormatDetectionsJson:
    """Tests for JSON formatting of detections."""

    def test_valid_json(self):
        """Output should be valid JSON."""
        detections = [
            {
                "class_name": "banana",
                "class_id": 0,
                "confidence": 0.95,
                "obb_points": [[10, 20], [90, 20], [90, 80], [10, 80]],
                "center_pixel": [50, 50],
                "position_3d": {"x": 0.1, "y": 0.2, "z": 0.5},
            }
        ]
        json_str = format_detections_json(detections, "realsense", 1234567890.0)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_has_required_fields(self):
        """JSON should have camera, timestamp, detections fields."""
        detections = []
        json_str = format_detections_json(detections, "kinect", 100.0)
        parsed = json.loads(json_str)
        assert "camera" in parsed
        assert "timestamp" in parsed
        assert "detections" in parsed
        assert parsed["camera"] == "kinect"

    def test_detections_preserved(self):
        """Detection data should survive JSON roundtrip."""
        detections = [
            {
                "class_name": "banana",
                "class_id": 0,
                "confidence": 0.95,
                "obb_points": [[10, 20], [90, 20], [90, 80], [10, 80]],
                "center_pixel": [50, 50],
                "position_3d": {"x": 0.1, "y": 0.2, "z": 0.5},
            }
        ]
        json_str = format_detections_json(detections, "realsense", 100.0)
        parsed = json.loads(json_str)
        assert len(parsed["detections"]) == 1
        assert parsed["detections"][0]["class_name"] == "banana"
        assert parsed["detections"][0]["confidence"] == 0.95

    def test_empty_detections(self):
        """No detections should produce valid JSON with empty list."""
        json_str = format_detections_json([], "kinect", 100.0)
        parsed = json.loads(json_str)
        assert parsed["detections"] == []
