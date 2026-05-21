# D435i Calibration Toolkit

This workspace is designed for calibrating a single `Intel RealSense D435i` device. It currently supports:

- `RGB` intrinsics calibration
- `IR Left` intrinsics calibration
- `IR Right` intrinsics calibration
- `IR Left <-> IR Right` stereo extrinsics calibration
- `RGB <-> IR Left` extrinsics calibration

The current version focuses on a single D435i and does not yet handle extrinsic calibration between multiple D435i devices.

## Directory Layout

```text
camera_calibration/
  config/
    calibration.yaml
  data/
    raw/
      <session_name>/
        rgb/
        ir_left/
        ir_right/
    output/
      <session_name>/
        rgb/
        ir_left/
        ir_right/
        stereo_ir/
        extrinsics/
        bundle/
  scripts/
    common.py
    capture_realsense_dataset.py
    calibrate_camera.py
    stereo_calibrate.py
    extrinsic_calibrate.py
    run_single_device_calibration.py
    undistort_image.py
  requirements.txt
```

`<session_name>` is the save name you enter in the terminal for one capture-and-calibration run.

If the same session name already exists, the script will ask you to choose:

- `overwrite`: delete the existing session and start over
- `merge`: keep existing files and continue saving with the next frame index
- `cancel`: abort the current operation

## Pairing Rule

This project does not need a separate `pairs/` directory. It uses the numeric suffix in filenames to determine synchronized frames:

- `rgb_0001.png`
- `ir_left_0001.png`
- `ir_right_0001.png`

If the numeric index matches, the images are treated as one synchronized frame set.

## Installation

### Linux

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

If `pyrealsense2` is not available in your environment, make sure Intel RealSense software and `librealsense` are installed first, then install the Python package again.

### Windows

```bash
pip install -r requirements.txt
```

## Capture And Preview

### Interactive capture with automatic calibration

```bash
python scripts/capture_realsense_dataset.py --config config/calibration.yaml
```

The script will:

1. Ask for a session name.
2. Check whether a session with the same name already exists.
3. Start the D435i `RGB + IR Left + IR Right` streams.
4. Open a live preview window showing all three streams.
5. Save synchronized frame sets when you trigger capture.
6. Automatically start calibration after the default target of `30` synchronized frame sets is reached.

Preview window shortcuts:

- `C`: save the current synchronized frame set
- `Space`: save the current synchronized frame set
- `Enter`: save the current synchronized frame set
- `Q`: quit capture

To capture data without starting calibration automatically:

```bash
python scripts/capture_realsense_dataset.py --config config/calibration.yaml --skip-auto-calibration
```

## Camera Connection Notes

In most cases, plugging the D435i into a working `USB 3.x` port is enough for the script to access the camera, but the following conditions should also be true:

- `pyrealsense2` is installed
- the operating system can detect the camera correctly
- no other application is currently using the camera
- the USB cable and port provide enough bandwidth

If these conditions are satisfied, `capture_realsense_dataset.py` should be able to open the `RGB + IR Left + IR Right` streams directly.

If the script cannot open the device, check:

- whether the camera appears in Intel RealSense Viewer
- whether the cable supports high-speed data transfer
- whether the device was connected through `USB 2.0`
- whether another application is occupying the camera

## Calibration Commands

### Run the full calibration pipeline for an existing session

```bash
python scripts/run_single_device_calibration.py --config config/calibration.yaml --session-name your_session_name
```

### Run one intrinsics calibration only

```bash
python scripts/calibrate_camera.py --config config/calibration.yaml --session-name your_session_name --sensor-id rgb
```

Available `sensor-id` values:

- `rgb`
- `ir_left`
- `ir_right`

### Run IR stereo calibration only

```bash
python scripts/stereo_calibrate.py \
  --config config/calibration.yaml \
  --session-name your_session_name \
  --pair-id ir_stereo \
  --left-calibration data/output/your_session_name/ir_left/calibration_result.json \
  --right-calibration data/output/your_session_name/ir_right/calibration_result.json
```

### Run RGB-to-IR-left extrinsics calibration only

```bash
python scripts/extrinsic_calibrate.py \
  --config config/calibration.yaml \
  --session-name your_session_name \
  --pair-id rgb_to_ir_left \
  --source-calibration data/output/your_session_name/rgb/calibration_result.json \
  --target-calibration data/output/your_session_name/ir_left/calibration_result.json
```

## Outputs

### RGB intrinsics

`data/output/<session_name>/rgb/calibration_result.json`

Contains:

- `camera_matrix`
- `dist_coeffs`
- `optimal_camera_matrix`
- `roi`
- `image_size`
- `rms`
- `mean_reprojection_error`
- `per_image_reprojection_error`
- `used_images`

### IR left intrinsics

`data/output/<session_name>/ir_left/calibration_result.json`

### IR right intrinsics

`data/output/<session_name>/ir_right/calibration_result.json`

These files use the same structure as the RGB intrinsics output.

### IR stereo extrinsics

`data/output/<session_name>/stereo_ir/stereo_calibration_result.json`

Contains:

- `left_sensor_id`
- `right_sensor_id`
- `left_camera_matrix`
- `left_dist_coeffs`
- `right_camera_matrix`
- `right_dist_coeffs`
- `rotation_matrix`
- `translation_vector`
- `essential_matrix`
- `fundamental_matrix`
- `stereo_rms`
- `used_frame_indices`
- `rectification`
- `R1`
- `R2`
- `P1`
- `P2`
- `Q`
- `roi1`
- `roi2`

### RGB-to-IR-left extrinsics

`data/output/<session_name>/extrinsics/rgb_to_ir_left.json`

Contains:

- `left_sensor_id`
- `right_sensor_id`
- `rotation_matrix`
- `translation_vector`
- `essential_matrix`
- `fundamental_matrix`
- `stereo_rms`
- `used_frame_indices`

### Bundle file

`data/output/<session_name>/bundle/device_calibration_bundle.json`

This file summarizes:

- chessboard configuration
- reference frame configuration
- intrinsics results for all three sensors
- `ir_stereo` extrinsics results
- `rgb_to_ir_left` extrinsics results

## Capture Recommendations

- Capture `20` to `30` synchronized frame sets.
- Change the chessboard pose and position between captures.
- Cover center, edges, near, and far regions as much as possible.
- Avoid blur, glare, and motion streaking.
- Make sure the chessboard is clearly visible in both RGB and IR images.
