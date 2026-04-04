# Launching the Detection Node

> Multi-class YOLO OBB detection + 6DOF pose estimation on the lab machine.
> **Must run on physical terminal** (Kinect depth engine needs OpenGL).

---

## Launch

**Terminal 1 -- Kinect driver:**
```bash
source /opt/ros/humble/setup.zsh && source ~/ros2_ws/install/setup.zsh
ros2 launch azure_kinect_ros_driver driver.launch.py
```

**Terminal 2 -- Detection node:**
```bash
source /opt/ros/humble/setup.zsh && source ~/ros2_ws/install/setup.zsh
cd ~/RIVeR-Perception-Pipeline && python3 scripts/ros_detect.py --visualize
```

---

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/detections/<class>/pose` | `geometry_msgs/PoseStamped` | Per-class 6DOF pose (position + quaternion) |

Topics created dynamically per detected class:

```bash
ros2 topic echo /detections/banana/pose
ros2 topic echo /detections/lime/pose
ros2 topic echo /detections/can/pose
```

Compatible with RVIZ, MoveIt, and tf2.

---

## Classes

| ID | Class | Confidence (typical) |
|----|-------|---------------------|
| 0 | banana | 0.90+ |
| 1 | lime | 0.85+ |
| 2 | can | 0.45-0.85 |

Model weights: `models/multi_class_obb/weights/best.pt`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Waiting for images..." | Kinect not publishing -- check USB, restart driver |
| `_ARRAY_API not found` | `pip3 install numpy==1.26.4` |
| No internet for pip | `echo "nameserver 8.8.8.8" \| sudo tee /etc/resolv.conf` |
| Kinect driver fails over SSH | Must be physical terminal (OpenGL) |
| Wrong model loaded | Check `--weights` flag points to `multi_class_obb` not `banana_obb` |
