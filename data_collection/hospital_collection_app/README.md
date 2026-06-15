# hospital_collection_app

This workspace provides a doctor-friendly desktop UI for the integrated
hospital collection workflow.

It wraps the existing
[hospital_data_collection](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/hospital_data_collection/main.py)
backend instead of reimplementing the low-level sensor logic.

## What The First Version Does

- starts the integrated hospital collector backend
- shows backend state and per-module health
- starts, pauses, resumes, and stops collection sessions with buttons
- shows the latest session output directories for each sensor
- streams backend logs into the UI

## Run

```bash
cd /path/to/research/data_collection/hospital_collection_app
pip install -r requirements.txt
python main.py
```

## Notes

- The backend still runs the existing collection modules under
  `hospital_data_collection`.
- The UI controls the backend with a JSON command file and monitors a JSON
  status file, so the command-line workflow remains available too.
- This version focuses only on collection control and status display.
