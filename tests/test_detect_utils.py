"""
Tests for detect_utils.py -- pure logic for ROS2 detection node.

Tests parsing YOLO OBB results, depth lookup, pixel-to-3D conversion,
detection formatting, and confidence filtering.
No ROS2 dependency -- runs on Mac.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from detect_utils import (
    parse_obb_results,
    get_depth_at_pixel,
    pixel_to_3d,
    format_detections_json,
    filter_detections,
    make_object_model,
    obb_corners_from_xywhr,
    make_camera_matrix,
    compute_reprojection_error,
    solve_pose_pnp,
    pose_to_position_and_euler,
    fuse_poses,
    estimate_banana_pose,
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


# ---------------------------------------------------------------------------
# 6DOF Pose Estimation via PnP
# ---------------------------------------------------------------------------

# Shared synthetic test fixtures
_CAM_MATRIX = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)


def _project_points(model_pts, rvec, tvec, cam_matrix):
    """Helper: project 3D points to 2D using a known pose."""
    pts_2d, _ = cv2.projectPoints(model_pts, rvec, tvec, cam_matrix, None)
    return pts_2d.reshape(-1, 2)


class TestMakeObjectModel:
    """Tests for creating 3D model points from object dimensions."""

    def test_correct_shape(self):
        model = make_object_model(0.18, 0.04, 0.035)
        assert model.shape == (4, 3)

    def test_symmetric_about_origin_xy(self):
        """Mean of X and Y should be 0 (centered at origin)."""
        model = make_object_model(0.18, 0.04, 0.035)
        assert abs(model[:, 0].mean()) < 1e-6
        assert abs(model[:, 1].mean()) < 1e-6

    def test_all_same_z(self):
        """All points should be at z = H/2 (top face)."""
        model = make_object_model(0.18, 0.04, 0.035)
        assert np.allclose(model[:, 2], 0.035 / 2)

    def test_dimensions_match(self):
        """Distance between adjacent corners should match L and W."""
        model = make_object_model(0.20, 0.05, 0.03)
        # Adjacent edges: 0→1 should be length, 1→2 should be width
        edge_01 = np.linalg.norm(model[1] - model[0])
        edge_12 = np.linalg.norm(model[2] - model[1])
        assert abs(edge_01 - 0.20) < 1e-6
        assert abs(edge_12 - 0.05) < 1e-6


class TestObbCornersFromXywhr:
    """Tests for reconstructing OBB corners from xywhr in canonical order."""

    def test_correct_shape(self):
        corners = obb_corners_from_xywhr(320, 240, 100, 50, 0.0)
        assert corners.shape == (4, 2)

    def test_center_preserved(self):
        """Mean of corners should equal (cx, cy)."""
        corners = obb_corners_from_xywhr(320, 240, 100, 50, 0.0)
        assert abs(corners[:, 0].mean() - 320) < 1e-4
        assert abs(corners[:, 1].mean() - 240) < 1e-4

    def test_no_rotation_axis_aligned(self):
        """With r=0, corners should form an axis-aligned rectangle."""
        corners = obb_corners_from_xywhr(100, 100, 80, 40, 0.0)
        # Width along X, height along Y
        xs = sorted(corners[:, 0])
        ys = sorted(corners[:, 1])
        assert abs(xs[2] - xs[0] - 80) < 1e-4  # width span
        assert abs(ys[2] - ys[0] - 40) < 1e-4  # height span

    def test_dimensions_preserved_with_rotation(self):
        """Edge lengths should match w and h regardless of rotation."""
        corners = obb_corners_from_xywhr(200, 200, 120, 60, np.pi / 4)
        edge_01 = np.linalg.norm(corners[1] - corners[0])
        edge_12 = np.linalg.norm(corners[2] - corners[1])
        assert abs(edge_01 - 120) < 1e-3
        assert abs(edge_12 - 60) < 1e-3

    def test_known_rotation(self):
        """90-degree rotation should swap X and Y spans."""
        corners_0 = obb_corners_from_xywhr(100, 100, 80, 40, 0.0)
        corners_90 = obb_corners_from_xywhr(100, 100, 80, 40, np.pi / 2)
        # X-span and Y-span should swap
        xspan_0 = corners_0[:, 0].max() - corners_0[:, 0].min()
        yspan_0 = corners_0[:, 1].max() - corners_0[:, 1].min()
        xspan_90 = corners_90[:, 0].max() - corners_90[:, 0].min()
        yspan_90 = corners_90[:, 1].max() - corners_90[:, 1].min()
        assert abs(xspan_0 - yspan_90) < 1e-3
        assert abs(yspan_0 - xspan_90) < 1e-3


class TestMakeCameraMatrix:
    """Tests for building camera intrinsic matrix."""

    def test_correct_shape(self):
        K = make_camera_matrix(500, 500, 320, 240)
        assert K.shape == (3, 3)

    def test_values(self):
        K = make_camera_matrix(607, 607, 330.7, 250.8)
        assert K[0, 0] == 607
        assert K[1, 1] == 607
        assert K[0, 2] == 330.7
        assert K[1, 2] == 250.8
        assert K[2, 2] == 1.0
        assert K[0, 1] == 0.0
        assert K[1, 0] == 0.0


class TestComputeReprojectionError:
    """Tests for reprojection error calculation."""

    def test_perfect_pose_zero_error(self):
        """Projecting model points with correct pose should give ~0 error."""
        model = make_object_model(0.18, 0.04, 0.035)
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.0, 0.0, 1.0])
        image_pts = _project_points(model, rvec, tvec, _CAM_MATRIX)
        err = compute_reprojection_error(model, image_pts, rvec, tvec, _CAM_MATRIX)
        assert err < 0.01

    def test_perturbed_pose_nonzero_error(self):
        """Slightly wrong pose should give nonzero error."""
        model = make_object_model(0.18, 0.04, 0.035)
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.0, 0.0, 1.0])
        image_pts = _project_points(model, rvec, tvec, _CAM_MATRIX)
        wrong_tvec = np.array([0.05, 0.0, 1.0])
        err = compute_reprojection_error(model, image_pts, rvec, wrong_tvec, _CAM_MATRIX)
        assert err > 1.0


class TestSolvePosePnp:
    """Tests for PnP pose estimation."""

    def test_synthetic_frontal(self):
        """Camera looking straight at object should recover correct pose."""
        model = make_object_model(0.18, 0.04, 0.035)
        true_rvec = np.array([0.0, 0.0, 0.0])
        true_tvec = np.array([0.0, 0.0, 0.8])
        image_pts = _project_points(model, true_rvec, true_tvec, _CAM_MATRIX)
        result = solve_pose_pnp(image_pts, model, _CAM_MATRIX, measured_depth=0.8)
        assert result is not None
        assert abs(result["tvec"][2] - 0.8) < 0.05

    def test_synthetic_angled(self):
        """Camera at angle should still recover reasonable pose."""
        model = make_object_model(0.18, 0.04, 0.035)
        true_rvec = np.array([0.3, 0.2, 0.1])
        true_tvec = np.array([0.05, -0.02, 0.6])
        image_pts = _project_points(model, true_rvec, true_tvec, _CAM_MATRIX)
        result = solve_pose_pnp(image_pts, model, _CAM_MATRIX, measured_depth=0.6)
        assert result is not None
        assert abs(result["tvec"][2] - 0.6) < 0.1

    def test_has_required_keys(self):
        """Result should have rvec, tvec, method, reprojection_error."""
        model = make_object_model(0.18, 0.04, 0.035)
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.0, 0.0, 1.0])
        image_pts = _project_points(model, rvec, tvec, _CAM_MATRIX)
        result = solve_pose_pnp(image_pts, model, _CAM_MATRIX, measured_depth=1.0)
        assert "rvec" in result
        assert "tvec" in result
        assert "method" in result
        assert "reprojection_error" in result

    def test_degenerate_returns_none(self):
        """Degenerate (collinear) image points should return None."""
        model = make_object_model(0.18, 0.04, 0.035)
        # All points on a line
        image_pts = np.array([[100, 200], [110, 200], [120, 200], [130, 200]], dtype=np.float64)
        result = solve_pose_pnp(image_pts, model, _CAM_MATRIX)
        assert result is None

    def test_reprojection_error_low(self):
        """Well-posed synthetic data should give low reprojection error."""
        model = make_object_model(0.18, 0.04, 0.035)
        rvec = np.array([0.1, 0.1, 0.5])
        tvec = np.array([0.0, 0.0, 0.7])
        image_pts = _project_points(model, rvec, tvec, _CAM_MATRIX)
        result = solve_pose_pnp(image_pts, model, _CAM_MATRIX, measured_depth=0.7)
        assert result is not None
        assert result["reprojection_error"] < 2.0


class TestPoseToPositionAndEuler:
    """Tests for converting PnP output to readable position + orientation."""

    def test_identity_rotation(self):
        """Zero rotation should give roll=pitch=yaw=0."""
        result = pose_to_position_and_euler(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]))
        assert abs(result["orientation_euler"]["roll"]) < 1.0
        assert abs(result["orientation_euler"]["pitch"]) < 1.0
        assert abs(result["orientation_euler"]["yaw"]) < 1.0

    def test_translation_passthrough(self):
        """Translation values should appear in position dict."""
        result = pose_to_position_and_euler(np.array([0.0, 0.0, 0.0]), np.array([0.5, -0.3, 1.2]))
        assert abs(result["position"]["x"] - 0.5) < 1e-6
        assert abs(result["position"]["y"] - (-0.3)) < 1e-6
        assert abs(result["position"]["z"] - 1.2) < 1e-6

    def test_quaternion_unit_norm(self):
        """Output quaternion should have unit magnitude."""
        result = pose_to_position_and_euler(np.array([0.3, 0.5, 0.7]), np.array([0.0, 0.0, 1.0]))
        q = result["orientation_quat"]
        mag = np.sqrt(q["x"]**2 + q["y"]**2 + q["z"]**2 + q["w"]**2)
        assert abs(mag - 1.0) < 1e-4

    def test_has_all_keys(self):
        """Result should have position, orientation_euler, orientation_quat, rotation_matrix."""
        result = pose_to_position_and_euler(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        assert "position" in result
        assert "orientation_euler" in result
        assert "orientation_quat" in result
        assert "rotation_matrix" in result

    def test_known_z_rotation(self):
        """90-degree rotation about Z axis should show ~90 in yaw."""
        # Rodrigues: rotation of pi/2 about Z = [0, 0, pi/2]
        result = pose_to_position_and_euler(np.array([0.0, 0.0, np.pi / 2]), np.array([0.0, 0.0, 1.0]))
        assert abs(abs(result["orientation_euler"]["yaw"]) - 90.0) < 2.0


class TestFusePoses:
    """Tests for fusing multiple pose estimates."""

    def test_single_pose_passthrough(self):
        """One pose in should be the same pose out."""
        pose = {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation_quat": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "reprojection_error": 1.0,
        }
        result = fuse_poses([pose])
        assert result is not None
        assert abs(result["position"]["x"] - 1.0) < 1e-6

    def test_equal_weight_midpoint(self):
        """Two poses with equal weight should give midpoint position."""
        p1 = {
            "position": {"x": 0.0, "y": 0.0, "z": 1.0},
            "orientation_quat": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "reprojection_error": 1.0,
        }
        p2 = {
            "position": {"x": 2.0, "y": 0.0, "z": 1.0},
            "orientation_quat": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "reprojection_error": 1.0,
        }
        result = fuse_poses([p1, p2], weights=[0.5, 0.5])
        assert abs(result["position"]["x"] - 1.0) < 1e-4

    def test_empty_returns_none(self):
        result = fuse_poses([])
        assert result is None

    def test_quaternion_normalized(self):
        """Fused quaternion should have unit norm."""
        p1 = {
            "position": {"x": 0.0, "y": 0.0, "z": 1.0},
            "orientation_quat": {"x": 0.0, "y": 0.0, "z": 0.38, "w": 0.92},
            "reprojection_error": 2.0,
        }
        p2 = {
            "position": {"x": 0.1, "y": 0.0, "z": 1.1},
            "orientation_quat": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "reprojection_error": 1.0,
        }
        result = fuse_poses([p1, p2])
        q = result["orientation_quat"]
        mag = np.sqrt(q["x"]**2 + q["y"]**2 + q["z"]**2 + q["w"]**2)
        assert abs(mag - 1.0) < 1e-3


class TestEstimateBananaPose:
    """Tests for the top-level pose estimation orchestrator."""

    def test_integration_synthetic(self):
        """Full pipeline with synthetic data should produce valid pose."""
        dims = {"length": 0.18, "width": 0.04, "height": 0.035}
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
        # Create a synthetic OBB: object at (0, 0, 0.8), no rotation
        model = make_object_model(0.18, 0.04, 0.035)
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([0.0, 0.0, 0.8])
        pts_2d = _project_points(model, rvec, tvec, _CAM_MATRIX)
        # Compute xywhr from the projected points
        rect = cv2.minAreaRect(pts_2d.astype(np.float32))
        cx, cy = rect[0]
        w, h = max(rect[1]), min(rect[1])
        r = np.deg2rad(rect[2]) if rect[1][0] < rect[1][1] else np.deg2rad(rect[2] + 90)
        result = estimate_banana_pose((cx, cy, w, h, r), intrinsics, dims, measured_depth=0.8)
        assert result is not None
        assert "position" in result
        assert "orientation_euler" in result
        assert abs(result["position"]["z"] - 0.8) < 0.15

    def test_returns_none_on_failure(self):
        """Bad input should return None, not crash."""
        dims = {"length": 0.18, "width": 0.04, "height": 0.035}
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
        # Degenerate OBB with zero area
        result = estimate_banana_pose((320, 240, 0, 0, 0), intrinsics, dims, measured_depth=1.0)
        assert result is None

    def test_output_has_all_fields(self):
        """Output should have position, orientation, reprojection_error, method."""
        dims = {"length": 0.18, "width": 0.04, "height": 0.035}
        intrinsics = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
        model = make_object_model(0.18, 0.04, 0.035)
        rvec = np.array([0.0, 0.0, 0.3])
        tvec = np.array([0.0, 0.0, 0.7])
        pts_2d = _project_points(model, rvec, tvec, _CAM_MATRIX)
        rect = cv2.minAreaRect(pts_2d.astype(np.float32))
        cx, cy = rect[0]
        w, h = max(rect[1]), min(rect[1])
        r = np.deg2rad(rect[2]) if rect[1][0] < rect[1][1] else np.deg2rad(rect[2] + 90)
        result = estimate_banana_pose((cx, cy, w, h, r), intrinsics, dims, measured_depth=0.7)
        assert result is not None
        assert "position" in result
        assert "orientation_euler" in result
        assert "orientation_quat" in result
        assert "reprojection_error" in result
        assert "method" in result
