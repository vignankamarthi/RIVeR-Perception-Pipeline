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

Wait for "Detector ready. Waiting for images..." in Terminal 2. If detections never appear after that, the Kinect driver in Terminal 1 likely crashed -- restart it.

---

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/detections/banana/pose` | `geometry_msgs/PoseStamped` | Banana 6DOF pose (position + quaternion) |
| `/detections/lime/pose` | `geometry_msgs/PoseStamped` | Lime 6DOF pose |
| `/detections/can/pose` | `geometry_msgs/PoseStamped` | Can 6DOF pose |

Topics are created dynamically on first detection of each class. Compatible with RVIZ, MoveIt, and tf2.

**Subscribe to a topic:**
```bash
ros2 topic echo /detections/banana/pose
```

**List active detection topics:**
```bash
ros2 topic list | grep detections
```

---

## PoseStamped Message Format

Each message contains position (x, y, z in meters) and orientation (quaternion x, y, z, w):

```
header:
  stamp: <timestamp>
  frame_id: <camera_frame>
pose:
  position:
    x: 0.15    # meters from camera
    y: -0.08
    z: 0.72
  orientation:
    x: 0.0
    y: 0.0
    z: 0.38    # encodes yaw (rotation on table)
    w: 0.92
```

Position is in the camera's coordinate frame. With `--visualize`, the ROS2 console also logs euler angles for each detection:

```
6DOF: yaw=45.2 pitch=1.3 roll=-0.8
```

Yaw = rotation on the table (most meaningful from top-down). Pitch and roll are near zero for flat objects.

---

## Classes and Object Dimensions

| ID | Class | Dimensions (L x W x H cm) | Confidence (typical) |
|----|-------|---------------------------|---------------------|
| 0 | banana | 19.6 x 6.7 x 3.6 | 0.90+ |
| 1 | lime | 6.4 x 5.8 x 4.4 | 0.85+ |
| 2 | can | 6.65 x 6.65 x 10.12 | 0.45-0.85 |

Model weights: `models/multi_class_obb/weights/best.pt`

Can confidence is lower because the circular top-down view provides less visual signal than elongated objects.

---

## CLI Options

```bash
python3 scripts/ros_detect.py --visualize          # default model, show OBB + pose axes
python3 scripts/ros_detect.py --conf 0.3           # lower confidence threshold
python3 scripts/ros_detect.py --weights path/to.pt # custom model weights
```

---

## Updating the Software

All changes flow through git. On the Ubuntu machine:

```bash
cd ~/RIVeR-Perception-Pipeline && git pull
```

Then restart the detection node (Terminal 2). No SCP needed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Waiting for images..." but no detections appear | Kinect driver crashed -- restart Terminal 1 |
| "Waiting for images..." | Kinect not publishing -- check USB, restart driver |
| Only banana detected (no lime/can) | Wrong model -- check weights path points to `multi_class_obb` |
| `_ARRAY_API not found` | `pip3 install numpy==1.26.4` |
| No internet for pip | `echo "nameserver 8.8.8.8" \| sudo tee /etc/resolv.conf` |
| Kinect driver fails over SSH | Must be physical terminal (OpenGL 4.4 required) |
| "No dimensions for class X" warning | New class detected without PnP measurements -- add to `OBJECT_DIMS` in `detect_utils.py` |
| Driver version error | Restart the Kinect driver, sometimes needs a second launch |
