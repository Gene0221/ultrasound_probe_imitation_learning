# tag2flange_calibration

This workspace performs one-shot collection and automatic solving for the
tag-to-flange rigid transform.

## Main Entrypoint

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection\tag2flange_calibration
python main.py
```

## What `main.py` Does

1. starts controlled visual pose tracking
2. starts controlled real pose tracking
3. begins recording immediately
4. waits until you press `q` then `Enter`
5. stops both trackers
6. automatically runs the calibration solver

## Output

Each run writes a bundle under:

```text
output/collection_YYYYMMDD_HHMMSS/
  visual_pose/
  real_pose/
  controls/
  tag2flange_calibration_report.json
  tag2flange_calibration_data.npz
```

## Solver

The actual solve step is implemented in:

- [scripts/solve_tag2flange_calibration.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/tag2flange_calibration/scripts/solve_tag2flange_calibration.py)

It reads paired visual and real relative motions, aligns them by timestamp, and
solves the hand-eye equation.

This workspace is intentionally one-shot. It does not use the multi-session
`Enter` / `Enter` / `q` interaction model from the continuous collection modules.
