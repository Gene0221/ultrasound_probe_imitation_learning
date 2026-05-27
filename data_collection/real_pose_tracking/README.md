# Real Pose Tracking Workspace

This workspace reads end-effector poses directly from a Franka Emika Panda
robot through the official C++ `libfranka` interface and writes pose-delta
records at a user-level rate of 30 Hz.

## Directory Layout

```text
real_pose_track/
  CMakeLists.txt
  build.sh
  config/
    default.yaml
    move_ee_local_linear.yaml
  launch.sh
  launch_move_ee_local_linear.sh
  src/
    read_franka_ee_pose.cpp
    move_ee_local_linear.cpp
  output/
  README.md
```

## Output Protocol

The logger writes one JSON object per line to a JSONL file. Each record follows
the same pose-delta style used by the visual pose tracking workspace and
contains:

- `host_timestamp_s`
- `prev_host_timestamp_s`
- `curr_host_timestamp_s`
- `delta_transform_prev_to_curr`
- `delta_translation_xyz`
- `delta_quaternion_xyzw`

## Configuration

All runtime parameters live in [config/default.yaml](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/tracking/config/default.yaml), including:

- robot IP
- connection timeout
- target sample rate
- pose source field
- output directory and file names
- optional sample limit

## Build

This workspace expects:

- `libfranka`
- `yaml-cpp`
- a CMake toolchain with C++17 support

Example:

```bash
bash data_collection/tracking/build.sh
```

Equivalent manual commands:

```bash
cmake -S data_collection/tracking -B data_collection/tracking/build
cmake --build data_collection/tracking/build -j
```

The build now produces two executables:

- `read_franka_ee_pose`
- `move_ee_local_linear`

## Launch

On Linux:

```bash
bash data_collection/tracking/launch.sh
```

or after making it executable:

```bash
chmod +x data_collection/tracking/launch.sh
./data_collection/tracking/launch.sh
```

The launch script automatically:

- locates the built executable under `build/` or `build/Release/`
- also falls back to `src/` if you compile the binary there manually
- uses `config/default.yaml` by default

## Local Linear Motion

`move_ee_local_linear.cpp` moves the end effector along a direction expressed in
the end effector's own coordinate frame while keeping orientation fixed.

Its config lives in
[config/move_ee_local_linear.yaml](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/real_pose_tracking/config/move_ee_local_linear.yaml)
and includes:

- `direction_ee_xyz`
- `distance_m`
- `speed_mps`
- `accel_time_s`

Example launch:

```bash
bash data_collection/real_pose_tracking/launch_move_ee_local_linear.sh
```

or with an explicit config:

```bash
bash data_collection/real_pose_tracking/launch_move_ee_local_linear.sh \
  data_collection/real_pose_tracking/config/move_ee_local_linear.yaml
```

## Manual Usage

```bash
./data_collection/tracking/build/read_franka_ee_pose \
  data_collection/tracking/config/default.yaml
```

Press `Ctrl+C` to stop logging.
