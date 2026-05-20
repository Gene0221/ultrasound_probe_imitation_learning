# Multi Camera Calibration

This directory contains the non-ROS dual-D435i workflow for:

- synchronized RGB pair capture
- online AprilTag pose-delta tracking with camera-A priority and camera-B fallback
- single-camera AprilTag pose testing
- dual-camera AprilTag delta consistency testing

## Entry Points

```bash
python multi_camera_calibration/capture_two_d435i_rgb_pairs.py --list-devices
python multi_camera_calibration/capture_two_d435i_rgb_pairs.py
python multi_camera_calibration/track_apriltag_pose_deltas.py
python multi_camera_calibration/test_apriltag_pose_single_camera.py
python multi_camera_calibration/test_apriltag_delta_consistency_dual_camera.py
```

## Dependencies

Install Python dependencies with:

```bash
python -m pip install -r multi_camera_calibration/requirements.txt
```

You also need a working `librealsense` / `pyrealsense2` environment.

## Config Files

- [config/capture.yaml](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/multi_camera_calibration/config/capture.yaml)
- [config/apriltag_tracking.yaml](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/multi_camera_calibration/config/apriltag_tracking.yaml)
- [config/apriltag_test.yaml](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/multi_camera_calibration/config/apriltag_test.yaml)
- [config/apriltag_delta_consistency.yaml](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/multi_camera_calibration/config/apriltag_delta_consistency.yaml)

## RGB Capture

`capture_two_d435i_rgb_pairs.py` captures synchronized RGB image pairs from camera A and camera B.

It uses:

- dual frame buffers
- minimum host-timestamp difference matching
- live preview
- image pair saving plus metadata export

The capture config stores:

- camera A serial number
- camera B serial number
- RGB width / height / fps
- preview width
- warmup frames
- startup timeout
- output root
- buffer size
- maximum allowed timestamp delta

## Deprecated

The old dual-camera RGB extrinsics workflow is kept under:

- [deprecated/estimate_rgb_extrinsics.py](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/multi_camera_calibration/deprecated/estimate_rgb_extrinsics.py)
- [deprecated/config/extrinsics.yaml](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/multi_camera_calibration/deprecated/config/extrinsics.yaml)

They are no longer part of the main tracking workflow, but are retained for historical reference and possible later reuse.

## AprilTag Pose Deltas

`track_apriltag_pose_deltas.py` performs real-time tracking for multiple AprilTags.

Current behavior:

- default AprilTag family is `tag36h11`
- two D435i RGB streams run online
- if both cameras see the tag, camera A is preferred
- if camera A misses the tag and camera B sees it, camera B is used directly
- if neither camera sees the tag, that tag is marked missing for the current frame
- each tracked tag is recorded independently
- pose deltas are computed only when adjacent valid frames come from the same camera
- if the preferred source switches between adjacent valid frames, that delta is skipped

For each tracked tag, the output includes:

- current `4x4` pose matrix in the current source-camera coordinates
- adjacent-frame `4x4` delta transform
- adjacent-frame translation delta
- adjacent-frame rotation delta quaternion
- current pose source camera and delta source camera

Tracking outputs:

- `<session_dir>/tag_pose_deltas.jsonl`
- `<session_dir>/tracking_summary.json`

Session selection for tracking:

- the script prompts for `session_name` if `output.session_name` is empty
- it writes tracking logs into `output/<session_name>`
- the current delta logic does not require extrinsics
- the extrinsics workflow is still kept in the workspace for later rigid-body conversion and fusion work

## Single-Camera AprilTag Test

`test_apriltag_pose_single_camera.py` is a quick live test for AprilTag detection and pose estimation with one camera.

It requires:

- one camera serial number
- that camera's RGB intrinsics
- the real AprilTag size

So yes: for this test script, the key calibration input is the single camera intrinsics, plus the physical tag size.

The preview window shows:

- tag outline
- tag ID
- pose axes
- translation `(x, y, z)` in meters
- pose quaternion `(qx, qy, qz, qw)`

## Dual-Camera Delta Consistency Test

`test_apriltag_delta_consistency_dual_camera.py` is an online validation tool.

It uses camera A and camera B to observe the same AprilTag, then:

- computes adjacent-frame pose delta in camera A independently
- computes adjacent-frame pose delta in camera B independently
- compares the two delta transforms
- prints translation and rotation error statistics to the terminal every 3 seconds

This script does not write tracking logs to file.
