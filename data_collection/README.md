# data_collection

This directory contains the data-collection workspaces used for sensing,
calibration, synchronization, and dataset preparation.

## Workspaces

- `imu_pose_grasping`
  - collects IMU pitch/roll data from a D435i
- `paxini_force_grasping`
  - collects left/right Paxini side-force sensor data
- `6D_force_grasping`
  - collects 6D force/torque data and writes timestamped JSONL logs
- `paxini26D_mapping`
  - orchestrates multimodal collection, archives sessions, aligns timestamps,
    builds `.pt` datasets, and trains the mapping model
- `visual_pose_tracking`
  - AprilTag-based visual pose tracking utilities
- `real_pose_tracking`
  - real robot pose tracking workspace
- `tag2flange_calibration`
  - calibration utilities for tag-to-flange transforms
- `preprocessing`
  - shared preprocessing scripts and config
- `ultrasound_image_collection`
  - ultrasound image collection workspace

## Recommended Setup

Create one Python environment for the whole `data_collection` tree and install
the aggregated requirements:

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection
pip install -r requirements.txt
```

This top-level `requirements.txt` merges the dependencies currently declared by
the sub-workspaces. Workspace-local `requirements.txt` files are still kept in
place for local maintenance.

## Multimodal Mapping Flow

The current multimodal force-grasping pipeline is centered on
`paxini26D_mapping`.

### 1. Direct module testing

- IMU:
  - run the logger inside `imu_pose_grasping`
- 6D force:
  - run `python scripts/read_data.py --config config/default.json` inside
    `6D_force_grasping`
- Paxini:
  - run the logger inside `paxini_force_grasping`

### 2. Unified collection

Run this inside `paxini26D_mapping`:

```powershell
python scripts/collect_session.py
```

Each run creates a new `session_xxxx` and archives the outputs from IMU,
Paxini, and 6D force collection.

### 3. Dataset preparation

```powershell
python scripts/prepare_dataset.py
```

This aligns the three modalities by nearest host timestamp with a maximum
allowed delta of `0.05s`, then exports one `.pt` dataset.

### 4. Model training

```powershell
python scripts/train_model.py
```

Model outputs are stored under `paxini26D_mapping/model/`.

## Notes

- The 6D collector currently depends on `pyserial`.
- IMU and visual tracking workspaces depend on `pyrealsense2`.
- Some workspaces also rely on hardware-specific runtime availability beyond
  Python package installation.
