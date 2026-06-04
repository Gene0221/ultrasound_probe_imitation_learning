# hospital_data_collection

This workspace is the hospital-facing integrated collector. It starts the
currently selected hospital collection modules and manages them with the unified
session controller.

The shared controller itself now lives in:

- [collection_runtime/src/collection_runtime](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/collection_runtime/src/collection_runtime)

## Controlled Modules

- `visual_pose`
- `paxini_force`
- `imu`
- `ultrasound`

The current default ultrasound entry is still a placeholder. The module slot
and session layout are already fixed so the real backend can be plugged in later.

## Main Entrypoint

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection\hospital_data_collection
python main.py
```

## Control Logic

- modules initialize once at startup
- initialization is handled by each called submodule
- `Enter` starts a new `session_xxxx`
- `Enter` again pauses and closes the current session
- `Enter` again resumes into the next session
- `q` stops the whole run immediately

Pausing and resuming do not reinitialize devices.

## Output

Each session is written under:

```text
output/session_xxxx/
  visual_pose/
  paxini_force/
  imu/
  ultrasound/
  metadata/
```

Run-level logs are written under `logs/`.

## Important Files

- [main.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/hospital_data_collection/main.py)
- [config/default.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/hospital_data_collection/config/default.yaml)
- [collection_runtime/src/collection_runtime/launcher.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/collection_runtime/src/collection_runtime/launcher.py)
- [collection_runtime/src/collection_runtime/session_manager.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/collection_runtime/src/collection_runtime/session_manager.py)
- [collection_runtime/src/collection_runtime/adapters](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/collection_runtime/src/collection_runtime/adapters)

`main.py` is the only user-facing entrypoint here. The shared implementation is
provided by `collection_runtime`.
