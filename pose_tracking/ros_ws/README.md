# ROS Bridge Workspace

This workspace is part of the tracking workspace itself.

The heavy sensor access and pose estimation stay in:

- `pose_tracking`

The ROS bridge package lives in:

- `pose_tracking/ros_ws/src/APROS`

## Purpose

- Run the local Python tracking pipeline in `pose_tracking`
- Read standardized pose-delta records from stdout
- Convert `prev_host_timestamp_s` into ROS `header.stamp`
- Publish a stable ROS topic for downstream consumers

## Current Source Strategy

- `vision` is implemented
- `fusion` is reserved and can be added later without changing the ROS topic shape

## Build

```bash
cd pose_tracking/ros_ws
catkin_make
source devel/setup.bash
```

## Run

```bash
roslaunch APROS pose_delta_bridge.launch
```

## Topic

Default topic:

- `/probe_pose_delta`

Message type:

- `APROS/ProbePoseDelta`
