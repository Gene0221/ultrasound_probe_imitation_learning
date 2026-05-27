# Visual Real Alignment Workspace

This workspace aligns quaternion pose deltas from the visual and real tracking
pipelines, reports baseline rotation error, and trains a lightweight MLP that
maps visual rotation deltas to real rotation deltas.

## Directory Layout

```text
visual_real_alignment/
  model/
  output/
  scripts/
  README.md
  requirements.txt
```

## Inputs

- visual log:
  `data_collection/visual_pose_tracking/output/tag_pose_deltas.jsonl`
- real log:
  `data_collection/real_pose_tracking/output/franka_ee_pose_deltas.jsonl`

Each record must contain:

- `curr_host_timestamp_s`
- `delta_quaternion_xyzw`

## Step 1: Build Paired Dataset

`scripts/build_paired_quaternion_dataset.py`:

- loads visual and real rotation-delta logs
- matches each visual record to the nearest real record using
  `curr_host_timestamp_s`
- drops pairs whose timestamp gap exceeds `0.05s` by default
- aligns quaternion signs to handle the double-cover ambiguity
- reports rotation-angle error statistics before learning
- writes a training-ready `.pt` file plus a JSON summary

Example:

```bash
python data_collection/visual_real_alignment/scripts/build_paired_quaternion_dataset.py
```

## Synchronized Collection Launch

`scripts/launch_visual_real_tracking.py`:

- launches the real pose tracking launch script and the visual AprilTag tracker together
- reuses the existing `real_pose_tracking/launch.sh` entrypoint instead of locating the binary directly
- runs the visual tracker with the current Python interpreter
- forwards logs from both processes with source prefixes
- shuts both processes down together on `Ctrl+C`

Example:

```bash
python data_collection/visual_real_alignment/scripts/launch_visual_real_tracking.py
```

## Step 2: Train Quaternion Mapper

`scripts/train_quaternion_mapper.py`:

- reads the paired `.pt` dataset
- randomly splits samples into train / val / test subsets by ratio
- trains a small MLP from visual quaternion to real quaternion
- normalizes predictions back onto the unit-quaternion manifold
- evaluates angle error on each split
- saves the trained checkpoint to `model/`

Example:

```bash
python data_collection/visual_real_alignment/scripts/train_quaternion_mapper.py
```

## Notes

- timestamps and time gaps are stored for traceability only and are not used as
  network inputs
- training uses only the quaternion values
- the main evaluation metric is rotation angle error in degrees
