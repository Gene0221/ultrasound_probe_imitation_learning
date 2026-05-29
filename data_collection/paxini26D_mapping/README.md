# paxini26D_mapping

This workspace is the orchestration layer for multimodal force-grasping data.

It does three jobs:

1. start IMU, Paxini, and 6D collection together
2. archive every manual run into a new `session_xxxx`
3. align all sessions into one `.pt` dataset and train a regression model for `Fz`

## Directory Layout

```text
paxini26D_mapping/
  config/
  dataset/
  model/
  scripts/
  sessions/
  src/
```

Each collection run is stored under:

```text
sessions/session_0001/
  imu/
  paxini/
  force6d/
  metadata/
```

## Current Status

- `imu_pose_grasping` is used as the IMU collector.
- `paxini_force_grasping` is wired to the DP-S2015 HAND-board scripts.
- `6D_force_grasping` is wired to the KWR75B acquisition scripts.
- The current supervised target is only the `Fz` axis.

## Commands

Start one manual collection session:

```powershell
python scripts/collect_session.py
```

Press `Enter` to stop all three child collectors.

Run unified calibration first when needed:

```powershell
python scripts/run_calibration.py
```

This launches:

- `paxini_force_grasping/scripts/calibrate_dp_s2015.py`
- `6D_force_grasping/scripts/zero_calibration.py`

and archives the resulting calibration files under `calibrations/`.

For unified collection, `paxini26D_mapping/config/default.json` can override
the child collectors' terminal printing. The default setup keeps Paxini and 6D
live-value printing off during coordinated runs.

Build a unified training dataset:

```powershell
python scripts/prepare_dataset.py
```

Train the first regression model:

```powershell
python scripts/train_model.py
```

## Placeholder Data Contracts

IMU file:

- `imu/imu_pitch_roll.jsonl`
- each row contains `host_timestamp_s`, `pitch_deg`, `roll_deg`

Paxini files:

- `paxini/left_sensor.jsonl`
- `paxini/right_sensor.jsonl`
- each row contains:
  - `host_timestamp_s`
  - `values` where the current layout is `[Fx, Fy, Fz]`
  - plus additional metadata such as `total_force` and `points`

6D file:

- `force6d/force6d.jsonl`
- each row contains:
  - `host_timestamp_s`
  - `Fz`
  - optional diagnostic fields such as `raw_Fz_kg` and `zeroed_Fz_kg`

For direct 6D testing, run the single script below inside the 6D workspace. It
prints live values and writes the same data to JSONL:

```powershell
python scripts/read_data.py --config config/default.json
```

## Alignment Rule

- anchor stream: 6D `Fz`
- matching method: nearest timestamp
- max allowed delta: `0.05s`
- samples beyond the threshold are dropped
