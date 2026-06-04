# 6D_force_grasping

This workspace collects KWR75B force data under the unified session controller.
Its main payload is a timestamped `Fz` signal plus calibration diagnostics.

## Main Entrypoint

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection\6D_force_grasping
python main.py
```

## Control Logic

- startup runs 6D zero calibration first
- `Enter` starts a new `session_xxxx`
- `Enter` again pauses and closes the current session
- `Enter` again resumes into the next session
- `q` stops the whole run

## Output

Each session is written under:

```text
output/session_xxxx/
  force6d.jsonl
```

The logger writes one JSON object per sample, including:

- `host_timestamp_s`
- `Fz`
- `raw_Fz_kg`
- `zeroed_Fz_kg`

## Important Files

- [main.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/6D_force_grasping/main.py)
- [config/default.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/6D_force_grasping/config/default.yaml)
- [scripts/read_data.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/6D_force_grasping/scripts/read_data.py)
- [scripts/zero_calibration.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/6D_force_grasping/scripts/zero_calibration.py)

Use `main.py` for normal collection. Zeroing and sample logging are handled by
the low-level scripts during module startup and runtime.
