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
  launch.sh
  src/
    read_franka_ee_pose.cpp
  output/
  README.md
```

## Output Protocol

The logger writes one JSON object per line to a JSONL file. Each record follows
the same pose-delta style used by the visual pose tracking workspace and
contains:

- `prev_host_timestamp_s`
- `curr_host_timestamp_s`
- `delta_transform_prev_to_curr`
- `delta_translation_xyz`
- `delta_quaternion_xyzw`

For debugging and downstream alignment, the logger also stores the current
absolute pose:

- `curr_position_xyz`
- `curr_quaternion_xyzw`

## Configuration

All runtime parameters live in [config/default.yaml](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/tracking/config/default.yaml), including:

- robot IP
- connection timeout
- target sample rate
- pose source field
- output directory and file names
- optional stdout streaming
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

## Manual Usage

```bash
./data_collection/tracking/build/read_franka_ee_pose \
  data_collection/tracking/config/default.yaml
```

Press `Ctrl+C` to stop logging.
