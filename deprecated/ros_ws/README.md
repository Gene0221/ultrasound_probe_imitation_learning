# Two-Camera RGB Extrinsic Calibration Workspace

This ROS Noetic workspace is used to calibrate the relative pose between two independent `Intel RealSense D435i` devices using only their `RGB` images.

The goal is to estimate camera `B` in camera `A` coordinates:

- `R_A_B`
- `t_A_B`
- `T_A_B` as a `4x4` homogeneous transform

The workflow is separated from the single-camera intrinsic calibration pipeline in `camera_calibration/`.

## Workspace Layout

```text
ros_ws/
  README.md
  src/
    CMakeLists.txt
    two_camera_rgb_extrinsic/
      CMakeLists.txt
      package.xml
      config/
        default.yaml
        dual_d435i.yaml
        dual_camera_preview.rviz
      launch/
        capture_pairs.launch
        dual_d435i_rgb.launch
        dual_d435i_capture.launch
      scripts/
        capture_rgb_pairs.py
        estimate_rgb_extrinsics.py
        launch_dual_d435i_rgb.py
        print_realsense_devices.py
```

## What This Workspace Does

1. Subscribe to the RGB image topics of camera `A` and camera `B`.
2. Synchronize the two streams using ROS timestamps.
3. Save paired RGB frames with matching indices.
4. Use the known RGB intrinsics of both cameras.
5. Detect the same checkerboard in both images.
6. Estimate the checkerboard pose in each camera independently.
7. Recover `B` in `A` coordinates from the two board poses.
8. Aggregate multiple paired observations into one final extrinsic result.

## Build

From the `ros_ws` directory:

```bash
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Start Two D435i RGB Topics

This workspace includes a sequential launcher that starts `realsense2_camera` for both devices.

List connected RealSense serial numbers first:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun two_camera_rgb_extrinsic print_realsense_devices.py
```

Then launch both RGB streams:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch two_camera_rgb_extrinsic dual_d435i_rgb.launch \
  serial_no_camera_a:=<serial_a> \
  serial_no_camera_b:=<serial_b>
```

This publishes:

- `/camera_a/color/image_raw`
- `/camera_b/color/image_raw`

Only RGB is enabled by default in this launch flow.
Camera `A` is started first. Camera `B` starts only after camera `A` is already publishing its RGB topic.
The launch file also starts `rviz` automatically with a placeholder configuration:

- `src/two_camera_rgb_extrinsic/config/dual_camera_preview.rviz`

You can open that configuration in `rviz`, modify it, and save your own version later.

To keep two-camera ROS preview responsive, the default RGB stream settings are intentionally lighter:

- `1280x720`
- `15 fps`

If your machine and USB setup are stable, you can override them from the launch command line.

## Capture Paired RGB Images

Edit the topic names and output directory in:

[`src/two_camera_rgb_extrinsic/config/default.yaml`](C:/Users/zhj80/OneDrive/Desktop/Master Course Material/research/data_collection/ros_ws/src/two_camera_rgb_extrinsic/config/default.yaml)

The capture node is optimized to keep ROS subscription responsive:

- synchronized callbacks only store the latest ROS messages
- interactive capture now runs in a C++ node instead of the older Python path
- image decoding is rate-limited by the preview loop
- image saving is handled asynchronously in a background worker
- preview defaults are reduced to a lighter refresh rate and width

Then run:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch two_camera_rgb_extrinsic capture_pairs.launch
```

Or launch the two RealSense drivers and the capture node together:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch two_camera_rgb_extrinsic dual_d435i_capture.launch \
  serial_no_camera_a:=<serial_a> \
  serial_no_camera_b:=<serial_b> \
  session_name:=session_01
```

In this combined capture launch, `rviz` is disabled by default so that the capture path does not add another heavy subscriber on top of the two raw RGB streams.

In the preview window:

- `S`: save the current synchronized RGB pair
- `Q`: quit capture

## Run Offline Extrinsic Calibration

Example:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun two_camera_rgb_extrinsic estimate_rgb_extrinsics.py \
  --dataset-dir /path/to/dataset \
  --camera-a-intrinsics /path/to/camera_a_rgb_intrinsics.json \
  --camera-b-intrinsics /path/to/camera_b_rgb_intrinsics.json \
  --rows 6 \
  --cols 9 \
  --square-size 25.0
```

## Outputs

The offline calibration script writes:

- `extrinsics_result.json`
- `extrinsics_result.yaml`

Both files include:

- per-pair board poses
- per-pair `T_A_B`
- averaged `R_A_B`
- averaged `t_A_B`
- final `T_A_B`
- static transform fields suitable for ROS `tf`

## Assumptions

- ROS distribution: `Noetic`
- Two independent `D435i` devices
- External calibration uses only `RGB` images
- Intrinsics calibration has already been completed separately
- Multiple synchronized checkerboard observations are collected first
- Final extrinsics are solved offline after capture

## Why Serial Numbers Matter

It is possible to enumerate two connected cameras without explicitly setting serial numbers, but that does not guarantee a stable mapping between physical devices and logical names such as `camera_a` and `camera_b`.

If you rely only on connection order:

- `camera_a` and `camera_b` may swap after reboot
- the mapping may change after reconnecting USB cables
- the calibration result may become ambiguous because you no longer know which physical device was treated as `A`

For a reproducible extrinsic calibration workflow, fixing camera `A` and camera `B` by serial number is the safer approach.
