# Launching the Detection Node

> YOLO OBB detection + 6DOF pose estimation on the lab machine.
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
| `/detections/<class>/pose` | `geometry_msgs/PoseStamped` | Per-class 6DOF pose (position + quaternion). RVIZ/MoveIt/tf2 compatible. |

Topics created dynamically per class (e.g., `/detections/banana/pose`, `/detections/pear/pose`).

```bash
ros2 topic echo /detections/banana/pose
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Waiting for images..." | Kinect not publishing -- check USB, restart driver |
| `_ARRAY_API not found` | `pip3 install numpy==1.26.4` |
| No internet for pip | `echo "nameserver 8.8.8.8" \| sudo tee /etc/resolv.conf` |
| Kinect driver fails over SSH | Must be physical terminal (OpenGL) |
