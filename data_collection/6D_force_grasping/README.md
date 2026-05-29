# 6D_force_grasping

This workspace is used to read KWR75B force data, print live values in the
terminal, and save timestamped JSONL logs for later alignment and training.

## What This Workspace Does

- reads raw serial frames from the 6D force sensor
- reads the full sensor frame internally
- saves only the calibrated `Fz` axis to logs
- applies a saved zero-bias calibration
- converts output to SI units
  - force: `N`
  - torque: `N*m`
- writes one JSON object per sample with host timestamps

## Main Script

Use this script for direct data collection and JSONL logging:

```powershell
python scripts/read_data.py --config config/default.json
```

This script will:

- open the serial port
- load the saved zero-bias calibration
- print live force/torque values in the terminal
- save the same data to `6D_force_grasping/output/force6d.jsonl`

## Zero Calibration

Run zero calibration separately before acquisition:

```powershell
python scripts/zero_calibration.py --config config/default.json
```

This script will:

- open the serial port
- collect tare samples while the sensor is unloaded and still
- save the zero bias to `config/zero_bias.json`

## Controls During Running

While the script is running:

- `q`: quit

## Config

Configuration file:

- [config/default.json](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/6D_force_grasping/config/default.json:1)

Current config fields:

- `sampling_hz`
  - target logging frequency in Hz
- `output_root`
  - output directory for saved logs
  - relative paths are resolved from the `6D_force_grasping` workspace root
- `file_name`
  - JSONL file name
- `print_human_readable`
  - whether to print live values in the terminal
- `calibration.bias_file`
  - saved zero-bias file path
- `calibration.tare_samples`
  - number of samples used when running `zero_calibration.py`
- `serial.port`
  - serial port, for example `/dev/ttyUSB0` or `/dev/ttyACM0`
  - can also be `AUTO`, which selects the port automatically when exactly one
    serial port is available
- `serial.serial_number`
  - preferred USB serial identifier for stable port matching when multiple
    serial devices are connected
- `serial.baudrate`
  - serial baudrate
- `serial.request_mode`
  - whether each sample is requested by command
- `serial.debug`
  - whether to print low-level serial debug info

## Output Format

Default output file:

- `6D_force_grasping/output/force6d.jsonl`

Each line is one JSON object:

```json
{
  "host_timestamp_s": 1779951600.123,
  "Fz": 8.901,
  "raw_Fz_kg": 0.9076,
  "zeroed_Fz_kg": 0.9084
}
```

Notes:

- `host_timestamp_s` is the host machine timestamp in seconds
- `Fz` is in `N`
- `raw_Fz_kg` is the raw sensor z-axis reading before calibration
- `zeroed_Fz_kg` is the calibrated z-axis reading in kg-equivalent units

## Files

- [scripts/read_data.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/6D_force_grasping/scripts/read_data.py:1)
  - main entry for reading, displaying, and logging
- [scripts/zero_calibration.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/6D_force_grasping/scripts/zero_calibration.py:1)
  - saves zero-bias calibration values
- [scripts/kwr75b_common.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/6D_force_grasping/scripts/kwr75b_common.py:1)
  - shared serial parsing and calibration helpers
- [requirements.txt](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/6D_force_grasping/requirements.txt:1)
  - Python dependency list for this workspace

## Dependency

Install the required Python package:

```powershell
pip install -r requirements.txt
```

This workspace currently requires:

- `pyserial`

## Integration With paxini26D_mapping

This workspace is called by:

- [paxini26D_mapping/config/default.json](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini26D_mapping/config/default.json:1)

The multimodal collection pipeline uses this script as the 6D force source:

```text
scripts/read_data.py
```

So the same script is used for:

- standalone 6D testing
- archived session collection inside `paxini26D_mapping`

## Troubleshooting

- If you see `pyserial is required`, install `pyserial` first.
- If the serial port cannot be opened, check `serial.port`.
- If `AUTO` reports multiple ports, set the exact port name manually.
- Prefer setting `serial.serial_number` when both Paxini and 6D sensors are
  connected at the same time.
- If frames cannot be parsed, confirm baudrate and sensor protocol settings.
- If zero calibration looks unstable, keep the sensor unloaded and still while
  running `zero_calibration.py`.
