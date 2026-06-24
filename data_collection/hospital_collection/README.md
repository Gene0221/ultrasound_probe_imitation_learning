# hospital_collection

Merged workspace combining the hospital collection backend and the PySide6
desktop UI into a single importable package.

## Architecture

The backend no longer runs as a separate subprocess.  ``BackendController``
wraps ``CollectionLauncher`` directly as a ``QObject`` and communicates via
Qt signals, eliminating the old JSON-file protocol.

```
hospital_collection/
  ├── main.py                          # Entry point (GUI or --headless)
  ├── config/default.yaml              # Module configuration
  ├── src/hospital_collection/
  │     ├── backend_controller.py      # QObject wrapper for CollectionLauncher
  │     └── app.py                     # PySide6 desktop UI
  └── requirements.txt                 # PySide6>=6.7
```

## Run

```bash
cd /path/to/data_collection/hospital_collection
python main.py          # GUI mode (default)
python main.py --headless              # Interactive console mode
python main.py --headless --command-file /path/to/commands.json --status-file /path/to/status.json
```

## Changes from the old split layout

- ``hospital_data_collection/`` and ``hospital_collection_app/`` are merged here.
- ``collection_runtime/`` remains separate (it is shared by other modules).
- The UI now imports ``BackendController`` instead of spawning a QProcess.
- Status updates are delivered via Qt signals, not JSON file polling.
