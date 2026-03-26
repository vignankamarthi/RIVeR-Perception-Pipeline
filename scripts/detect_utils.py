"""
Pure logic for the ROS2 detection node.

No ROS2 dependency -- testable on Mac.
Handles: YOLO OBB result parsing, depth lookup, pixel-to-3D conversion,
detection formatting, and confidence filtering.
"""

import json

import numpy as np


def parse_obb_results(
    obb_points: list[np.ndarray],
    class_ids: list[int],
    confidences: list[float],
    class_names: dict[int, str],
) -> list[dict]:
    """Parse YOLO OBB inference output into structured dicts.

    Args:
        obb_points: List of (4, 2) arrays -- OBB corner points per detection.
        class_ids: Integer class ID per detection.
        confidences: Confidence score per detection.
        class_names: Mapping from class ID to name.

    Returns:
        List of detection dicts with class_name, class_id, confidence,
        obb_points (as lists), and center_pixel.
    """
    results = []
    for points, cid, conf in zip(obb_points, class_ids, confidences):
        center = points.mean(axis=0)
        results.append({
            "class_name": class_names[cid],
            "class_id": int(cid),
            "confidence": float(conf),
            "obb_points": [[float(x), float(y)] for x, y in points],
            "center_pixel": [float(center[0]), float(center[1])],
        })
    return results


def get_depth_at_pixel(
    depth_image: np.ndarray, x: int, y: int, window_size: int = 5
) -> float:
    """Look up depth value at a pixel, using a median window to reduce noise.

    Args:
        depth_image: (H, W) depth image in millimeters (uint16).
        x: Pixel x coordinate (column).
        y: Pixel y coordinate (row).
        window_size: Size of the square window for median filtering.

    Returns:
        Depth value in original units (mm). Returns 0 if out of bounds or no valid depth.
    """
    h, w = depth_image.shape[:2]
    half = window_size // 2

    # Bounds check
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0

    # Extract window, clipping to image bounds
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)

    window = depth_image[y1:y2, x1:x2]

    # Filter out zero (no depth) values
    valid = window[window > 0]
    if len(valid) == 0:
        return 0

    return float(np.median(valid))


def pixel_to_3d(
    u: float, v: float, depth: float,
    fx: float, fy: float, cx: float, cy: float,
) -> dict | None:
    """Convert pixel coordinates + depth to 3D point using camera intrinsics.

    Uses the pinhole camera model:
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

    Args:
        u: Pixel x coordinate.
        v: Pixel y coordinate.
        depth: Depth in meters.
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.

    Returns:
        Dict with x, y, z in meters, or None if depth is 0.
    """
    if depth <= 0:
        return None

    return {
        "x": float((u - cx) * depth / fx),
        "y": float((v - cy) * depth / fy),
        "z": float(depth),
    }


def filter_detections(
    detections: list[dict], min_confidence: float
) -> list[dict]:
    """Filter detections by minimum confidence threshold.

    Args:
        detections: List of detection dicts (must have 'confidence' key).
        min_confidence: Minimum confidence to keep.

    Returns:
        Filtered list of detections.
    """
    return [d for d in detections if d["confidence"] >= min_confidence]


def format_detections_json(
    detections: list[dict], camera_name: str, timestamp: float
) -> str:
    """Format detections as a JSON string for ROS publishing.

    Args:
        detections: List of detection dicts.
        camera_name: Camera identifier ("realsense" or "kinect").
        timestamp: ROS timestamp as float (seconds).

    Returns:
        JSON string with camera, timestamp, and detections fields.
    """
    msg = {
        "camera": camera_name,
        "timestamp": timestamp,
        "detections": detections,
    }
    return json.dumps(msg)
