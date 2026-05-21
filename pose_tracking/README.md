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

  visualization/
    imu/

  data/
    vision/
      output/
    imu/
      calibration/
      logs/
      raw/
    fusion/

  deprecated/
```

## Vision

Vision code and configuration live in:

- [scripts/vision](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/scripts/vision)
- [config/vision](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/config/vision)

Current vision scripts:

- [track_apriltag_pose_deltas.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/scripts/vision/track_apriltag_pose_deltas.py)
- [test_apriltag_pose_single_camera.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/scripts/vision/test_apriltag_pose_single_camera.py)
- [test_apriltag_delta_consistency_dual_camera.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/scripts/vision/test_apriltag_delta_consistency_dual_camera.py)

Vision intrinsics live in:

- [config/vision/intrinsics](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/config/vision/intrinsics)

Vision outputs are written under:

- `data/vision/output/<session_name>/...`

## IMU

IMU code and configuration live in:

- [scripts/imu](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/scripts/imu)
- [utils/imu](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/utils/imu)
- [config/imu](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/config/imu)

Current IMU scripts:

- [main.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/scripts/imu/main.py)
- [imu_bias_calibration.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/scripts/imu/imu_bias_calibration.py)
- [rs_imu_calibration.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/scripts/imu/rs_imu_calibration.py)

The IMU attitude viewer is also kept under:

- [utils/imu/attitude_viewer.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/utils/imu/attitude_viewer.py)

IMU data lives in:

- [data/imu/raw](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/data/imu/raw)
- [data/imu/calibration](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/data/imu/calibration)
- [data/imu/logs](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/data/imu/logs)

## Fusion

Fusion is parallel to vision and IMU within the same component layers.

Current fusion utilities live in:

- [utils/fusion](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/utils/fusion)
- [config/fusion](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/config/fusion)
- [scripts/fusion](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/scripts/fusion)

## Deprecated

Legacy workflows are kept in:

- [deprecated](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/pose_tracking/deprecated)

## Dependencies

```bash
python -m pip install -r pose_tracking/requirements.txt
```
