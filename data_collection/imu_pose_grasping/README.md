# imu_pose_grasping

This workspace collects IMU posture from an Intel RealSense D435i under the
unified session controller.

## Main Entrypoint

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection\imu_pose_grasping
python main.py
```

## Control Logic

- startup initializes the IMU logger and its stationary calibration flow
- `Enter` starts a new `session_xxxx`
- `Enter` again pauses and closes the current session
- `Enter` again resumes into the next session
- `q` stops the whole run

## Output

Each session is written under:

```text
output/session_xxxx/
  imu_pitch_roll.jsonl
  summary.json
```

Each IMU record contains at least:

- `host_timestamp_s`
- `pitch_deg`
- `roll_deg`

The default logger also stores fields such as `tilt_deg`,
`device_timestamp_ms`, and `frame_number`.

## Important Files

- [main.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/imu_pose_grasping/main.py)
- [config/session_collection.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/imu_pose_grasping/config/session_collection.yaml)
- [config/default.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/imu_pose_grasping/config/default.yaml)
- [scripts/log_d435i_pitch_roll.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/imu_pose_grasping/scripts/log_d435i_pitch_roll.py)

Use `main.py` for normal collection. `scripts/log_d435i_pitch_roll.py` is the
low-level logger that the session controller calls.
