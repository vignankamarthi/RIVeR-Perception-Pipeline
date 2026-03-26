"""
ROS2 dual-camera YOLO OBB detection node.

Subscribes to RealSense and Kinect RGB + depth topics, runs YOLO OBB
inference, computes 3D positions, and publishes detections as JSON.

Usage (on Ubuntu lab machine):
    # Terminal 1: Kinect driver (physical terminal only)
    source /opt/ros/humble/setup.zsh && source ~/ros2_ws/install/setup.zsh
    ros2 launch azure_kinect_ros_driver driver.launch.py

    # Terminal 2: RealSense driver
    source /opt/ros/humble/setup.zsh
    ros2 launch realsense2_camera rs_launch.py depth_module.enable:=true

    # Terminal 3: Detection node
    source /opt/ros/humble/setup.zsh && source ~/ros2_ws/install/setup.zsh
    python3 scripts/ros_detect.py

    # Optional args:
    python3 scripts/ros_detect.py --weights path/to/best.pt --conf 0.5
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO

# Import pure logic (tested separately in test_detect_utils.py)
sys.path.insert(0, str(Path(__file__).parent))
from detect_utils import (
    parse_obb_results,
    get_depth_at_pixel,
    pixel_to_3d,
    format_detections_json,
    filter_detections,
)

# Default model path (relative to scripts/)
DEFAULT_WEIGHTS = Path(__file__).parent.parent / "models" / "banana_obb" / "weights" / "best.pt"

# Camera topic configuration
CAMERA_CONFIG = {
    "realsense": {
        "rgb_topic": "/camera/camera/color/image_raw",
        "depth_topic": "/camera/camera/depth/image_rect_raw",
        "camera_info_topic": "/camera/camera/color/camera_info",
    },
    "kinect": {
        "rgb_topic": "/rgb/image_raw",
        "depth_topic": "/depth_to_rgb/image_raw",
        "camera_info_topic": "/rgb/camera_info",
    },
}


class DualCameraDetector(Node):
    """ROS2 node that runs YOLO OBB detection on two camera feeds."""

    def __init__(self, weights: str, conf: float = 0.5):
        super().__init__("dual_camera_detector")

        self.bridge = CvBridge()
        self.conf = conf

        # Load YOLO model
        self.get_logger().info(f"Loading YOLO model: {weights}")
        self.model = YOLO(weights)
        self.get_logger().info("Model loaded.")

        # Class names from the model
        self.class_names = self.model.names  # {0: "banana", ...}

        # Storage for latest depth images and camera intrinsics
        self.depth_images = {"realsense": None, "kinect": None}
        self.camera_intrinsics = {"realsense": None, "kinect": None}

        # Publisher for detections (JSON on string topic)
        self.detection_pub = self.create_publisher(String, "/detections/banana_obb", 10)

        # Subscribe to both cameras
        for cam_name, config in CAMERA_CONFIG.items():
            # RGB subscription -- triggers detection
            self.create_subscription(
                Image,
                config["rgb_topic"],
                lambda msg, name=cam_name: self.rgb_callback(msg, name),
                10,
            )

            # Depth subscription -- stored for lookup
            self.create_subscription(
                Image,
                config["depth_topic"],
                lambda msg, name=cam_name: self.depth_callback(msg, name),
                10,
            )

            # Camera info subscription -- for intrinsics
            self.create_subscription(
                CameraInfo,
                config["camera_info_topic"],
                lambda msg, name=cam_name: self.camera_info_callback(msg, name),
                10,
            )

            self.get_logger().info(f"Subscribed to {cam_name}: {config['rgb_topic']}")

        self.get_logger().info("Dual camera detector ready. Waiting for images...")

    def depth_callback(self, msg: Image, camera_name: str):
        """Store the latest depth image for a camera."""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_images[camera_name] = depth_image
        except Exception as e:
            self.get_logger().warn(f"[{camera_name}] Depth conversion failed: {e}")

    def camera_info_callback(self, msg: CameraInfo, camera_name: str):
        """Store camera intrinsics (focal length, principal point)."""
        if self.camera_intrinsics[camera_name] is None:
            self.camera_intrinsics[camera_name] = {
                "fx": msg.k[0],
                "fy": msg.k[4],
                "cx": msg.k[2],
                "cy": msg.k[5],
            }
            self.get_logger().info(
                f"[{camera_name}] Intrinsics: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} "
                f"cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}"
            )

    def rgb_callback(self, msg: Image, camera_name: str):
        """Run YOLO detection on an RGB frame and publish results."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"[{camera_name}] RGB conversion failed: {e}")
            return

        # Run YOLO OBB inference
        results = self.model(frame, conf=self.conf, verbose=False)
        det = results[0]

        if det.obb is None or len(det.obb) == 0:
            return

        # Extract OBB results
        obb_xyxyxyxy = det.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        class_ids = det.obb.cls.cpu().numpy().astype(int).tolist()
        confidences = det.obb.conf.cpu().numpy().tolist()

        # Convert to list of (4, 2) arrays
        obb_points = [obb_xyxyxyxy[i] for i in range(len(obb_xyxyxyxy))]

        # Parse into structured dicts
        detections = parse_obb_results(obb_points, class_ids, confidences, self.class_names)
        detections = filter_detections(detections, self.conf)

        # Add 3D position if depth is available
        depth_image = self.depth_images.get(camera_name)
        intrinsics = self.camera_intrinsics.get(camera_name)

        for det_dict in detections:
            det_dict["position_3d"] = None

            if depth_image is not None and intrinsics is not None:
                cx, cy = det_dict["center_pixel"]
                depth_mm = get_depth_at_pixel(depth_image, int(cx), int(cy))

                if depth_mm > 0:
                    depth_m = depth_mm / 1000.0  # mm -> meters
                    pos = pixel_to_3d(
                        cx, cy, depth_m,
                        intrinsics["fx"], intrinsics["fy"],
                        intrinsics["cx"], intrinsics["cy"],
                    )
                    det_dict["position_3d"] = pos

        # Publish as JSON
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        json_str = format_detections_json(detections, camera_name, timestamp)

        out_msg = String()
        out_msg.data = json_str
        self.detection_pub.publish(out_msg)

        # Log summary
        for d in detections:
            pos_str = ""
            if d["position_3d"]:
                p = d["position_3d"]
                pos_str = f" @ ({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f})m"
            self.get_logger().info(
                f"[{camera_name}] {d['class_name']} conf={d['confidence']:.2f}"
                f" center=({d['center_pixel'][0]:.0f}, {d['center_pixel'][1]:.0f})"
                f"{pos_str}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="ROS2 dual-camera YOLO OBB detector")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to YOLO OBB weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Error: weights not found at {weights}")
        print("Run train.py first, or specify --weights path/to/best.pt")
        sys.exit(1)

    rclpy.init()
    node = DualCameraDetector(str(weights), conf=args.conf)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
