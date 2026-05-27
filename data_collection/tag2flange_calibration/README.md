# Tag To Flange Calibration Workspace

This workspace estimates the fixed rigid transform between the tracked AprilTag
frame and the robot flange / end-effector frame using paired relative motions.

## Directory Layout

```text
tag2flange_calibration/
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
- `delta_transform_prev_to_curr`

## Step 1: Launch Synchronized Collection

`scripts/launch_visual_real_tracking.py` starts:

- the real pose tracking launch script from `real_pose_tracking`
- the visual AprilTag tracking Python script from `visual_pose_tracking`

Example:

```bash
python data_collection/tag2flange_calibration/scripts/launch_visual_real_tracking.py
```

## Step 2: Solve Tag-To-Flange Transform

`scripts/solve_tag2flange_calibration.py`:

- reads visual and real relative-motion logs
- matches samples by nearest-neighbor `curr_host_timestamp_s`
- drops pairs whose time gap exceeds `0.05s` by default
- solves the hand-eye equation `A_i X = X B_i`
- reports both:
  - `T_tag_to_flange`
  - `T_flange_to_tag`
- writes residual statistics for rotation and translation

Example:

```bash
python data_collection/tag2flange_calibration/scripts/solve_tag2flange_calibration.py
```

## Output

The solver writes:

- a JSON report with matched-sample counts, time-alignment stats, estimated
  transforms, and residual metrics
- an `.npz` file with matched `A_i`, `B_i`, timestamps, and solved transform

## Notes

- this workflow does not require a pre-measured ground-truth `tag -> flange`
  transform
- the approach assumes the tag is rigidly attached to the flange during data
  collection
- stable estimation requires diverse motions, especially rotational excitation
