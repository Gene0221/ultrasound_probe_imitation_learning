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
4. waits until you press `q`
5. stops both trackers
6. asks for an experiment id at startup and writes it into the output folder name
7. automatically aligns visual and real pose samples by timestamp
8. splits matched samples into train/test sets with a 4:1 ratio
9. solves the calibration from the train split and evaluates the learned transform on the held-out test split

## Output

Each run writes a bundle under:

```text
output/experiment_EXPERIMENT_ID_YYYYMMDD_HHMMSS/
  visual_pose/
  real_pose/
  controls/
  dataset_split/
    train_pairs.npz
    train_pairs.jsonl
    test_pairs.npz
    test_pairs.jsonl
    split_manifest.json
  tag2flange_calibration_report.json
  tag2flange_calibration_data.npz
```

The split is deterministic by default (`--split-seed 42`) and uses
`--train-ratio 0.8`, which corresponds to a 4:1 train/test split. The calibration
transform is solved from `train_pairs`, while `test_pairs` is kept for held-out
evaluation.

The main evaluation fields are written to `tag2flange_calibration_report.json`
under `evaluation`:

```json
{
  "train_translation_mean_mm": 0.0,
  "test_translation_mean_mm": 0.0,
  "translation_mean_gap_mm": 0.0,
  "train_rotation_mean_deg": 0.0,
  "test_rotation_mean_deg": 0.0,
  "rotation_mean_gap_deg": 0.0
}
```

The gap is computed as `test mean - train mean`. Timestamp differences are only
used for sample matching and are not reported as calibration evaluation metrics.

## Solver

The actual solve step is implemented in:

- [scripts/solve_tag2flange_calibration.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/tag2flange_calibration/scripts/solve_tag2flange_calibration.py)

It reads paired visual and real relative motions, aligns them by timestamp, and
solves the hand-eye equation.

This workspace is intentionally one-shot. It does not use the multi-session
`Enter` / `Enter` / `q` interaction model from the continuous collection modules.
