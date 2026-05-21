# Pose Tracking Workspace

This workspace is organized by component type first, and by solution path second.

The active solution paths are:

- `vision`
- `imu`
- `fusion`

## Directory Layout

```text
pose_tracking/
  config/
    vision/
      intrinsics/
    imu/
    fusion/

  scripts/
    vision/
    imu/
    fusion/

  utils/
    vision/
    imu/
    fusion/

  data/
    vision/
      output/
    imu/
      calibration/
      logs/
      raw/
    fusion/

  deprecated/

  ros_ws/
    src/
      APROS/
```

## Vision

Vision code and configuration live in:

- `pose_tracking/scripts/vision`
- `pose_tracking/config/vision`

Current vision scripts:

- `pose_tracking/scripts/vision/track_apriltag_pose_deltas.py`
- `pose_tracking/scripts/vision/test_apriltag_pose_single_camera.py`
- `pose_tracking/scripts/vision/test_apriltag_delta_consistency_dual_camera.py`

Vision intrinsics live in:

- `pose_tracking/config/vision/intrinsics`

Vision outputs are written under:

- `data/vision/output/tag_pose_deltas.jsonl`
- `data/vision/output/tracking_summary.json`

## IMU

IMU code and configuration live in:

- `pose_tracking/scripts/imu`
- `pose_tracking/utils/imu`
- `pose_tracking/config/imu`

Current IMU scripts:

- `pose_tracking/scripts/imu/main.py`
- `pose_tracking/scripts/imu/imu_bias_calibration.py`
- `pose_tracking/scripts/imu/rs_imu_calibration.py`

The IMU attitude viewer is also kept under:

- `pose_tracking/utils/imu/attitude_viewer.py`

IMU data lives in:

- `pose_tracking/data/imu/raw`
- `pose_tracking/data/imu/calibration`
- `pose_tracking/data/imu/logs`

## Fusion

Fusion is parallel to vision and IMU within the same component layers.

Current fusion utilities live in:

- `pose_tracking/utils/fusion`
- `pose_tracking/config/fusion`
- `pose_tracking/scripts/fusion`

## ROS Bridge

The ROS workspace is part of tracking and lives in:

- `pose_tracking/ros_ws`
- `pose_tracking/ros_ws/src/APROS`

## Deprecated

Legacy workflows are kept in:

- `pose_tracking/deprecated`

## Dependencies

```bash
python -m pip install -r pose_tracking/requirements.txt
```
