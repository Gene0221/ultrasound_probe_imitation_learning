# Visual Pose Tracking Workspace

This workspace keeps only the visual AprilTag tracking pipeline used for probe
pose-delta estimation.

## Directory Layout

```text
visual_pose_tracking/
  config/
  output/
  scripts/
  utils/
  README.md
  requirements.txt
```

## What Stays Here

- `config`
  - YAML configuration files for AprilTag tracking, consistency tests, and camera intrinsics
- `scripts`
  - executable tracking and validation scripts
- `utils`
  - lightweight helpers such as RGB intrinsics export
- `output`
  - tracking logs such as `tag_pose_deltas.jsonl` and `tracking_summary.json`

## Main Scripts

- `scripts/track_apriltag_pose_deltas.py`
  - dual-camera AprilTag pose-delta tracking
- `scripts/test_apriltag_pose_single_camera.py`
  - single-camera pose sanity check
- `scripts/test_apriltag_delta_consistency_dual_camera.py`
  - dual-camera delta consistency check

## Dependencies

```bash
python -m pip install -r data_collection/visual_pose_tracking/requirements.txt
```
