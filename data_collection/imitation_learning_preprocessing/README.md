# imitation_learning_preprocessing

This workspace post-processes collected hospital sessions for downstream
imitation-learning datasets.

It currently provides two batch-friendly steps:

1. transform AprilTag pose deltas into the robot flange frame
2. apply the trained Paxini + IMU force-mapping model to hospital sessions

Both scripts can read sessions from an external drive. Put the external drive
session root in [config/preprocess_dataset.yaml](config/preprocess_dataset.yaml):

```yaml
paths:
  session_root: E:/hospital_collection/output
```

Outputs are written back into each session directory by default, so data on a
mobile drive stays on that same drive:

```text
session_0001/
  visual_pose/tag_pose_deltas.jsonl
  imu/imu_pitch_roll.jsonl
  paxini_force/left_sensor.jsonl
  paxini_force/right_sensor.jsonl
  transformed_pose/flange_pose_deltas.jsonl
  predicted_force/predicted_force.jsonl
```

## Scripts

### Transform Pose To Flange

```powershell
python scripts/transform_pose_to_flange.py --config config/preprocess_dataset.yaml
```

By default this processes all `session_*` folders under `paths.session_root`.
It loads the newest tag-to-flange calibration from `paths.calibration_root`,
supporting current `experiment_*` outputs and older `collection_*` outputs.

Useful overrides:

```powershell
python scripts/transform_pose_to_flange.py --session session_0003
python scripts/transform_pose_to_flange.py --session-root E:/hospital_collection/output
python scripts/transform_pose_to_flange.py --calibration E:/calibration/tag2flange_calibration_data.npz
python scripts/transform_pose_to_flange.py --calibration E:/research_data/experienment/pose_experienment/experiment_full_degree_freedom_20260708_150826
```

### Apply Force Mapping

```powershell
python scripts/apply_force_mapping.py --config config/preprocess_dataset.yaml
```

By default this processes all `session_*` folders under `paths.session_root`.
It loads the newest `model.pt` under `paths.model_root` unless
`model.checkpoint` or `--checkpoint` is provided.

Useful overrides:

```powershell
python scripts/apply_force_mapping.py --session session_0003
python scripts/apply_force_mapping.py --session-root E:/hospital_collection/output
python scripts/apply_force_mapping.py --checkpoint E:/models/my_run/model.pt
python scripts/apply_force_mapping.py --checkpoint E:/models/my_run
```

## Configuration

Important fields in `config/preprocess_dataset.yaml`:

- `paths.session_root`: where collected sessions live; use an absolute external-drive path when needed
- `paths.calibration_root`: directory containing tag-to-flange calibration runs
- `paths.model_root`: directory containing force-mapping training runs
- `session_layout`: subdirectory and file names inside each session
- `pose_transform`: output folder/file names for transformed pose deltas
- `force_mapping`: output folder/file names and timestamp matching threshold

## Shared Module

Common path, config, session-scanning, and JSON writing logic lives in:

```text
module/common.py
```

The scripts keep only their domain-specific work: pose geometry or model
inference.
