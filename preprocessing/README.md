# Multimodal Preprocessing Workspace

This workspace converts raw pose, force, and ultrasound data into the dataset
layout expected by `imitation_learing_v1`.

## What It Does

- reads pose records from `.json` or `.jsonl`
- reads force records from `.json`, `.jsonl`, or `.csv`
- reads ultrasound image timestamps from `.json`, `.jsonl`, or `.csv`
- aligns force and ultrasound to pose using `curr_host_timestamp_s`
- drops frames when any modality exceeds the configured nearest-neighbor threshold
- writes trajectory folders plus `train_manifest.json` / `val_manifest.json`

## Recommended Raw Data Protocol

### Pose

Pose records should contain at least:

- `curr_host_timestamp_s`
- `delta_translation_xyz`
- `delta_quaternion_xyzw`

Supported file formats:

- `.jsonl`: one JSON object per line
- `.json`: either a top-level array or an object with `records`

### Force

Each force record should contain:

- `host_timestamp_s`
- `force`

Examples:

```json
{"host_timestamp_s": 1712345678.123, "force": [0.1, 0.0, -0.2, 0.01, 0.02, 0.03]}
```

CSV is also supported. Either use a `force` column containing a JSON-style list,
or provide scalar columns such as `fx, fy, fz, mx, my, mz`.

### Ultrasound

Keep image files in one directory and store timestamps in one shared index file.

Recommended `jsonl` index:

```json
{"image": "frame_000001.png", "host_timestamp_s": 1712345678.125}
{"image": "frame_000002.png", "host_timestamp_s": 1712345678.158}
```

The `image` field may be absolute, or relative to `images_dir`.

## Output Layout

```text
dataset/
  manifests/
    train_manifest.json
    val_manifest.json
  trajectories/
    traj_0001/
      images/
      metadata.json
```

`metadata.json` stores one ordered frame list per trajectory. Each frame includes
the training fields:

- `image`
- `pose_delta_7d`
- `force`

It also keeps alignment debug fields such as source timestamps and time deltas.

## Config

See [config/preprocess_dataset.yaml](C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/preprocessing/config/preprocess_dataset.yaml)
for an example configuration.

Each trajectory entry usually represents one continuous collection session.

## Usage

```bash
python data_collection/preprocessing/scripts/build_imitation_dataset.py ^
  --config data_collection/preprocessing/config/preprocess_dataset.yaml
```
