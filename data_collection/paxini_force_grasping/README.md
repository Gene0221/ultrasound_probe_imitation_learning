# paxini_force_grasping

This workspace is used to read two DP-S2015 Paxini force sensors through the
Paxini HAND board, run zero calibration, and save timestamped JSONL logs for
later alignment and learning.

## What This Workspace Does

- connects to the Paxini HAND board
- detects two DP-S2015 sensors
- optionally loads a saved zero calibration
- reads left and right force data continuously
- writes one JSONL file per sensor with host timestamps

## Main Scripts

### 1. Zero Calibration

Run calibration first while both sensors are unloaded and still:

```powershell
python scripts/calibrate_dp_s2015.py --config config/default.json
```

This script will:

- open the HAND board serial port
- detect the attached DP-S2015 sensors
- collect zero-force samples
- save the calibration file

Default calibration file:

- `config/dp_s2015_calibration.json`

### 2. Data Collection

Run the acquisition script:

```powershell
python scripts/log_paxini_force.py --config config/default.json
```

This script will:

- open the HAND board serial port
- load the saved calibration if it exists
- stream both sensors continuously
- save left and right sensor logs separately

## Output Files

Default output directory:

- `paxini_force_grasping/output/`

Default output files:

- `output/left_sensor.jsonl`
- `output/right_sensor.jsonl`

Each line is one JSON object with at least:

```json
{
  "host_timestamp_s": 1779951600.123,
  "sensor_index": 0,
  "label": "DP-S2015 #1",
  "point_count": 52,
  "values": [0.1, -0.2, 5.3],
  "total_force": {
    "Fx": 0.1,
    "Fy": -0.2,
    "Fz": 5.3
  },
  "points": []
}
```

Notes:

- `host_timestamp_s` is the host machine timestamp in seconds
- `values` currently stores `[Fx, Fy, Fz]`
- `total_force` is the calibrated total force for that sensor
- `points` contains per-point sensor values from the DP-S2015 grid

## Config

Configuration file:

- [config/default.json](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini_force_grasping/config/default.json:1)

Current config fields:

- `output_root`
  - output directory for JSONL logs
  - relative paths are resolved from the `paxini_force_grasping` workspace root
- `left_file_name`
  - left sensor JSONL file name
- `right_file_name`
  - right sensor JSONL file name
- `print_human_readable`
  - whether to print live force summaries in the terminal
- `stream.sampling_hz`
  - target stream display/update frequency
- `calibration.file`
  - saved calibration file path
- `calibration.tare_samples`
  - number of samples used during zero calibration
- `serial.port`
  - serial port for the Paxini HAND board, for example `/dev/ttyUSB0`
- `serial.serial_number`
  - preferred USB serial identifier for stable port matching when multiple
    serial devices are connected
- `serial.baudrate`
  - serial baudrate

## Files

- [scripts/log_paxini_force.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini_force_grasping/scripts/log_paxini_force.py:1)
  - main acquisition and JSONL logging entry
- [scripts/calibrate_dp_s2015.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini_force_grasping/scripts/calibrate_dp_s2015.py:1)
  - zero-calibration entry
- [scripts/paxini_common.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini_force_grasping/scripts/paxini_common.py:1)
  - shared config/path/module-loading helpers
- [scripts/DP-S2015-Elite.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini_force_grasping/scripts/DP-S2015-Elite.py:1)
  - original Paxini driver and sensor parsing logic

## Integration With paxini26D_mapping

This workspace is called by:

- [paxini26D_mapping/config/default.json](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini26D_mapping/config/default.json:1)

The multimodal collection pipeline uses:

- `scripts/log_paxini_force.py` for data collection

So the same script is used for:

- standalone Paxini testing
- archived multimodal session collection inside `paxini26D_mapping`

## Dependency Notes

This workspace relies on:

- `pyserial`

and on the vendor communication logic contained in `DP-S2015-Elite.py`.

## Troubleshooting

- If the serial port cannot be opened, check `serial.port`.
- Prefer setting `serial.serial_number` when both Paxini and 6D sensors are
  connected at the same time.
- If the second sensor is missing, check the HAND board wiring and module setup.
- If no calibration file exists, the acquisition script will still run, but it
  will use raw values instead of zero-calibrated ones.
- If zero calibration looks unstable, keep both sensors unloaded and still
  during `calibrate_dp_s2015.py`.
