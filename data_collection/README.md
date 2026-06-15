# data_collection

This directory contains the current data-collection, calibration, orchestration,
and post-processing workspaces.

## Workspace Roles

- `imu_pose_grasping`: standalone IMU collection
- `paxini_force_grasping`: standalone Paxini force collection
- `visual_pose_tracking`: standalone AprilTag visual pose tracking
- `6D_force_grasping`: standalone 6D force collection
- `real_pose_tracking`: standalone Franka end-effector pose collection
- `ultrasound_image_collection`: standalone ultrasound image collection from a USB capture device
- `collection_runtime`: shared session-based collection runtime used by modules and integrated collectors
- `hospital_data_collection`: integrated hospital collector for visual pose, Paxini, IMU, and ultrasound
- `paxini26D_mapping`: integrated collect-prepare-train workspace for IMU + Paxini + 6D mapping
- `tag2flange_calibration`: one-shot collection plus automatic tag-to-flange solve
- `preprocessing`: offline alignment and dataset-building scripts

## Unified Collection Logic

All continuous collection workspaces now use the same control model:

- start the workspace root `main.py`
- press `Enter` to start a new `session_xxxx`
- press `Enter` again to pause and close the current session
- press `Enter` again to resume into the next session
- press `q` to stop the whole run

Outputs are written inside each workspace's own `output/` or workspace-specific
raw-data root.

## Main Entrypoints

- `hospital_data_collection/main.py`
- `collection_runtime/src/collection_runtime/launcher.py`
- `imu_pose_grasping/main.py`
- `paxini_force_grasping/main.py`
- `visual_pose_tracking/main.py`
- `6D_force_grasping/main.py`
- `real_pose_tracking/main.py`
- `ultrasound_image_collection/main.py`
- `paxini26D_mapping/main.py`
- `tag2flange_calibration/main.py`

## Notes

- Module-specific initialization now lives inside the called submodules. For example,
  Paxini zero calibration and 6D zero calibration run during module startup.
- The shared session controller has been extracted into `collection_runtime`, so
  single-module and integrated workspaces both depend on a neutral common layer.
- `paxini26D_mapping` no longer performs hospital-side inference-input collection.
  Hospital collection is done in `hospital_data_collection`, and the trained mapping
  model is applied later with `paxini26D_mapping/scripts/predict_force.py`.
- `preprocessing` stays outside the interactive session controller.
