# real_pose_tracking

This workspace collects Franka end-effector pose deltas under the unified
session controller.

The low-level pose logger is still backed by the C++ `libfranka` executable,
but the user-facing collection interface is now standardized.

## Main Entrypoint

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection\real_pose_tracking
python main.py
```

## Control Logic

- `Enter` starts a new `session_xxxx`
- `Enter` again pauses and closes the current session
- `Enter` again resumes into the next session
- `q` stops the whole run

## Output

Each session is written under:

```text
output/session_xxxx/
  franka_ee_pose_deltas.jsonl
  summary.json
```

## Important Files

- [main.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/real_pose_tracking/main.py)
- [config/session_collection.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/real_pose_tracking/config/session_collection.yaml)
- [config/default.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/real_pose_tracking/config/default.yaml)
- [scripts/controlled_real_pose_logger.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/real_pose_tracking/scripts/controlled_real_pose_logger.py)
- [src/read_franka_ee_pose.cpp](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/real_pose_tracking/src/read_franka_ee_pose.cpp)

Use `main.py` for normal collection. The Python wrapper provides the unified
session interface, while the C++ executable remains the low-level pose source.
