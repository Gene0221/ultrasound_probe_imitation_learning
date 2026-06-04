# paxini_force_grasping

This workspace collects dual DP-S2015 Paxini force data under the unified
session controller.

## Main Entrypoint

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection\paxini_force_grasping
python main.py
```

## Control Logic

- startup runs Paxini zero calibration first
- `Enter` starts a new `session_xxxx`
- `Enter` again pauses and closes the current session
- `Enter` again resumes into the next session
- `q` stops the whole run

## Output

Each session is written under:

```text
output/session_xxxx/
  left_sensor.jsonl
  right_sensor.jsonl
```

Each record includes:

- `host_timestamp_s`
- `sensor_index`
- `label`
- `values`
- `total_force`
- `points`

## Important Files

- [main.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini_force_grasping/main.py)
- [config/default.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini_force_grasping/config/default.yaml)
- [scripts/calibrate_dp_s2015.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini_force_grasping/scripts/calibrate_dp_s2015.py)
- [scripts/log_paxini_force.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini_force_grasping/scripts/log_paxini_force.py)

Use `main.py` for normal collection. The calibration and logging scripts remain
as the low-level implementation used by the controller.
