# ultrasound_image_collection

This workspace captures timestamped ultrasound image frames from a USB video
capture device on Ubuntu under the unified session controller.

It is designed for the common chain:

```text
ultrasound machine AV out -> USB capture dongle -> Ubuntu /dev/video*
```

The module saves image files plus a shared timestamp index so downstream pose
alignment can select the nearest frame under a configured threshold.

## Main Entrypoint

```bash
cd /path/to/research/data_collection/ultrasound_image_collection
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
  images/
    frame_000000.png
    frame_000001.png
    ...
  timestamps.jsonl
  summary.json
```

Recommended timestamp index format:

```json
{"image": "images/frame_000000.png", "host_timestamp_s": 1712345678.123456, "frame_index": 0}
```

## Ubuntu Device Identification

The logger supports the same style of stable USB-device identification used by
the force modules:

- explicit device node such as `/dev/video2`
- `/dev/v4l/by-id` substring match
- `/dev/v4l/by-path` substring match
- USB serial number
- vendor/product IDs
- device-name substring

List detected video devices first:

```bash
python scripts/capture_ultrasound_frames.py --config config/default.yaml --list-devices
```

Then fill one or more fields under `device:` in
[config/default.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/ultrasound_image_collection/config/default.yaml).

`device.by_id` is usually the safest option on Ubuntu when the capture card
creates stable `/dev/v4l/by-id/...` symlinks.

## Notes

- The module opens the capture device continuously and only writes images while
  the session controller is in `recording=true`.
- `capture.fps` requests the source frame rate from the driver, while
  `recording.target_hz` controls how often frames are saved to disk.
- If the capture card really outputs `25 fps` rather than `29.97/30 fps`,
  setting `target_hz: 30.0` will not create new information; it will only save
  at most one frame per incoming source frame.
