"""
Pure logic for the ROS2 detection node.

No ROS2 dependency -- testable on Mac.
Handles: YOLO OBB result parsing, depth lookup, pixel-to-3D conversion,
detection formatting, confidence filtering, and 6DOF pose estimation via PnP.
"""

import json
import math

import cv2
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


# ---------------------------------------------------------------------------
# 6DOF Pose Estimation via PnP
# ---------------------------------------------------------------------------

# Measured at lab 2026-03-30 with cm tape measure
BANANA_DIMS_M = {"length": 0.196, "width": 0.067, "height": 0.036}


def make_object_model(length: float, width: float, height: float) -> np.ndarray:
    """Create 3D model points for the top face of a rectangular bounding box.

    Object frame centered at box centroid. X=length, Y=width, Z=height.
    Returns 4 top-face corners in canonical order: TL, TR, BR, BL.

    Args:
        length: Object length in meters (long dimension).
        width: Object width in meters (short dimension).
        height: Object height/thickness in meters.

    Returns:
        (4, 3) float64 array of 3D points in object coordinates.
    """
    hl, hw, hh = length / 2, width / 2, height / 2
    return np.array([
        [-hl, -hw, hh],  # TL
        [+hl, -hw, hh],  # TR
        [+hl, +hw, hh],  # BR
        [-hl, +hw, hh],  # BL
    ], dtype=np.float64)


def obb_corners_from_xywhr(
    cx: float, cy: float, w: float, h: float, r: float,
) -> np.ndarray:
    """Reconstruct OBB corners from (cx, cy, w, h, r) in canonical order.

    Canonical order: TL, TR, BR, BL in the OBB's local frame,
    rotated by r and translated to (cx, cy).

    Args:
        cx, cy: OBB center in pixels.
        w: OBB width (longer dimension) in pixels.
        h: OBB height (shorter dimension) in pixels.
        r: Rotation angle in radians.

    Returns:
        (4, 2) float64 array of 2D corner points in pixel coordinates.
    """
    hw, hh = w / 2, h / 2
    # Local corners before rotation (TL, TR, BR, BL)
    local = np.array([
        [-hw, -hh],
        [+hw, -hh],
        [+hw, +hh],
        [-hw, +hh],
    ], dtype=np.float64)

    # Rotation matrix
    cos_r, sin_r = np.cos(r), np.sin(r)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)

    # Rotate and translate
    rotated = local @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy

    return rotated


def make_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix.

    Args:
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.

    Returns:
        (3, 3) float64 camera matrix.
    """
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def compute_reprojection_error(
    model_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> float:
    """Compute mean reprojection error for a PnP solution.

    Args:
        model_points: (N, 3) 3D object points.
        image_points: (N, 2) corresponding 2D image points.
        rvec: (3,) Rodrigues rotation vector.
        tvec: (3,) translation vector.
        camera_matrix: (3, 3) camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.

    Returns:
        Mean Euclidean distance (pixels) between projected and actual points.
    """
    projected, _ = cv2.projectPoints(
        model_points, rvec, tvec, camera_matrix,
        dist_coeffs if dist_coeffs is not None else np.zeros(4),
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    return float(errors.mean())


def solve_pose_pnp(
    image_points: np.ndarray,
    model_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    measured_depth: float | None = None,
) -> dict | None:
    """Estimate 6DOF pose using solvePnP. Tries IPPE first, falls back to EPNP.

    Args:
        image_points: (4, 2) float array of 2D pixel coordinates.
        model_points: (4, 3) float array of 3D object model points.
        camera_matrix: (3, 3) camera intrinsic matrix.
        dist_coeffs: Distortion coefficients (None = zero distortion).
        measured_depth: Depth in meters at object center for disambiguating IPPE.

    Returns:
        Dict with rvec, tvec, method, reprojection_error. None if all fail.
    """
    image_pts = image_points.astype(np.float64).reshape(-1, 1, 2)
    model_pts = model_points.astype(np.float64).reshape(-1, 1, 3)
    dist = dist_coeffs if dist_coeffs is not None else np.zeros(4)

    # Validate: check for degenerate input
    area = cv2.contourArea(image_points.astype(np.float32))
    if area < 10:
        return None

    best_result = None

    # Try IPPE (coplanar solver) -- returns up to 2 solutions
    for method, name in [
        (cv2.SOLVEPNP_IPPE, "IPPE"),
        (cv2.SOLVEPNP_EPNP, "EPNP"),
    ]:
        try:
            success, rvec, tvec = cv2.solvePnP(
                model_pts, image_pts, camera_matrix, dist, flags=method,
            )
        except cv2.error:
            continue

        if not success:
            continue

        rvec = rvec.flatten()
        tvec = tvec.flatten()

        # Sanity checks
        if np.any(np.isnan(rvec)) or np.any(np.isnan(tvec)):
            continue
        if np.any(np.isinf(rvec)) or np.any(np.isinf(tvec)):
            continue
        if tvec[2] <= 0:
            continue  # object must be in front of camera

        reproj_err = compute_reprojection_error(
            model_points, image_points, rvec, tvec, camera_matrix, dist,
        )

        # Reject if reprojection error is too high
        if reproj_err > 10.0:
            continue

        # Depth disambiguation for IPPE
        if measured_depth is not None and measured_depth > 0:
            depth_diff = abs(tvec[2] - measured_depth)
            if best_result is not None:
                best_depth_diff = abs(best_result["tvec"][2] - measured_depth)
                if depth_diff >= best_depth_diff:
                    continue  # existing solution is closer to measured depth

        candidate = {
            "rvec": rvec,
            "tvec": tvec,
            "method": name,
            "reprojection_error": reproj_err,
        }

        if best_result is None or reproj_err < best_result["reprojection_error"]:
            best_result = candidate

        # If IPPE succeeded, don't try EPNP
        if name == "IPPE":
            break

    return best_result


def pose_to_position_and_euler(
    rvec: np.ndarray, tvec: np.ndarray,
) -> dict:
    """Convert solvePnP output to position + euler angles + quaternion.

    Args:
        rvec: (3,) Rodrigues rotation vector.
        tvec: (3,) translation vector.

    Returns:
        Dict with position, orientation_euler (degrees), orientation_quat,
        rotation_matrix.
    """
    R, _ = cv2.Rodrigues(rvec)

    # Euler angles from rotation matrix (ZYX convention)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    # Quaternion from rotation matrix
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    # Normalize quaternion
    qnorm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if qnorm > 0:
        qx, qy, qz, qw = qx / qnorm, qy / qnorm, qz / qnorm, qw / qnorm

    return {
        "position": {
            "x": float(tvec[0]),
            "y": float(tvec[1]),
            "z": float(tvec[2]),
        },
        "orientation_euler": {
            "roll": float(math.degrees(roll)),
            "pitch": float(math.degrees(pitch)),
            "yaw": float(math.degrees(yaw)),
        },
        "orientation_quat": {
            "x": float(qx),
            "y": float(qy),
            "z": float(qz),
            "w": float(qw),
        },
        "rotation_matrix": R.tolist(),
    }


def fuse_poses(
    poses: list[dict],
    weights: list[float] | None = None,
) -> dict | None:
    """Fuse multiple 6DOF pose estimates into a single estimate.

    Weighted average for position, SLERP for quaternion orientation.

    Args:
        poses: List of pose dicts (each with position, orientation_quat,
            reprojection_error).
        weights: Explicit weights (must sum to 1). If None, weights are
            computed as 1/reprojection_error (normalized).

    Returns:
        Fused pose dict, or None if poses is empty.
    """
    if not poses:
        return None

    if len(poses) == 1:
        return poses[0]

    # Compute weights from reprojection error if not provided
    if weights is None:
        errors = [max(p.get("reprojection_error", 1.0), 0.01) for p in poses]
        inv_errors = [1.0 / e for e in errors]
        total = sum(inv_errors)
        weights = [w / total for w in inv_errors]

    # Weighted average position
    fused_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
    for pose, w in zip(poses, weights):
        for axis in ("x", "y", "z"):
            fused_pos[axis] += pose["position"][axis] * w

    # Weighted average quaternion (simple linear blend + normalize)
    # For small angular differences, this approximates SLERP well
    q_sum = np.zeros(4)
    for pose, w in zip(poses, weights):
        q = pose["orientation_quat"]
        q_vec = np.array([q["x"], q["y"], q["z"], q["w"]])
        # Ensure consistent hemisphere (flip if dot product with first is negative)
        if len(q_sum) > 0 and np.dot(q_vec, q_sum) < 0:
            q_vec = -q_vec
        q_sum += q_vec * w

    # Normalize
    q_norm = np.linalg.norm(q_sum)
    if q_norm > 0:
        q_sum /= q_norm

    return {
        "position": fused_pos,
        "orientation_quat": {
            "x": float(q_sum[0]),
            "y": float(q_sum[1]),
            "z": float(q_sum[2]),
            "w": float(q_sum[3]),
        },
    }


def estimate_banana_pose(
    obb_xywhr: tuple[float, float, float, float, float],
    camera_intrinsics: dict,
    object_dims: dict,
    measured_depth: float | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> dict | None:
    """Top-level: OBB detection -> 6DOF banana pose.

    Args:
        obb_xywhr: (cx, cy, w, h, r) from YOLO OBB.
        camera_intrinsics: Dict with fx, fy, cx, cy.
        object_dims: Dict with length, width, height in meters.
        measured_depth: Depth at object center in meters.
        dist_coeffs: Lens distortion coefficients.

    Returns:
        Full pose dict (position, orientation_euler, orientation_quat,
        reprojection_error, method). None if PnP fails.
    """
    cx, cy, w, h, r = obb_xywhr

    # Guard against degenerate OBBs
    if w * h < 1.0:
        return None

    # Build 2D image points from xywhr
    image_points = obb_corners_from_xywhr(cx, cy, w, h, r)

    # Build 3D model points from known dimensions
    model_points = make_object_model(
        object_dims["length"], object_dims["width"], object_dims["height"],
    )

    # Build camera matrix
    cam_matrix = make_camera_matrix(
        camera_intrinsics["fx"], camera_intrinsics["fy"],
        camera_intrinsics["cx"], camera_intrinsics["cy"],
    )

    # Solve PnP
    pnp_result = solve_pose_pnp(
        image_points, model_points, cam_matrix, dist_coeffs, measured_depth,
    )

    if pnp_result is None:
        return None

    # Convert to readable format
    pose = pose_to_position_and_euler(pnp_result["rvec"], pnp_result["tvec"])
    pose["reprojection_error"] = pnp_result["reprojection_error"]
    pose["method"] = pnp_result["method"]

    return pose
