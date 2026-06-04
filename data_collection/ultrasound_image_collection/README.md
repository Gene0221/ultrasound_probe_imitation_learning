# ultrasound_image_collection

This workspace is a placeholder ultrasound collection module under the unified
session controller.

The interactive session behavior is already standardized, but the actual
ultrasound acquisition backend has not been fixed yet.

## Main Entrypoint

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection\ultrasound_image_collection
python main.py
```

## Control Logic

- `Enter` starts a new `session_xxxx`
- `Enter` again pauses and closes the current session
- `Enter` again resumes into the next session
- `q` stops the whole run

## Output

The workspace uses the same session layout as the other collection modules:

```text
output/session_xxxx/
```

At the current stage, this is still a placeholder integration target. The
future ultrasound backend should be attached behind `main.py` so it stays
compatible with the same session controls.
