# Force Grasping Workspace

This workspace is for multimodal force-grasping data collection and modeling.
The current goal is to read device posture from an IMU, align it with
side-mounted hand-force sensors and a reference 6D force/torque sensor, and
prepare a clean dataset for supervised learning.

## Problem Setup

The planned system has three signal groups:

- IMU: estimates the device posture relative to gravity
- side force sensors: measure human hand pressing forces on both sides
- reference 6D force sensor: provides the training target label

The intended learning task is:

```text
[IMU posture + left/right side-force signals] -> [reference 6D force/torque]
```

## Initial Assessment

### 1. Can an IMU estimate tilt relative to gravity?

Yes, for tilt-related posture this is feasible.

If the device motion is not too aggressive, the accelerometer can estimate the
gravity direction, and the IMU can recover:

- pitch relative to gravity
- roll relative to gravity

This is the right tool if the main quantity you care about is the device
orientation relative to the vertical direction.

### 2. Can the IMU always know the zero direction of gravity?

Partly yes, partly no.

What the IMU can know robustly:

- the current gravity direction in the IMU frame
- tilt angle away from vertical
- pitch/roll relative to gravity after sensor fusion

What the IMU cannot know from gravity alone:

- absolute heading around the gravity axis
- a permanent global yaw zero, unless you add a magnetometer or define a manual
  initialization pose

So if your "zero start point" means:

- vertical reference: yes, gravity gives this directly
- full 3D absolute orientation: not from accelerometer + gyro alone

### 3. Is gravity zero stable over time?

It is stable for pitch and roll, but practical accuracy depends on motion and
calibration.

Main effects:

- accelerometer bias changes the estimated gravity vector
- gyro drift affects short-term orientation propagation
- rapid motion contaminates acceleration with non-gravity components
- mounting misalignment between IMU and device frame introduces a fixed offset

This means the system should explicitly separate:

- gravity-defined vertical reference
- device-frame zero pose defined during calibration

### 4. Is timestamp alignment feasible?

Yes, and this is one of the most important parts of the whole pipeline.

The cleanest design is to ensure every modality is stamped by the same host
clock at the moment the sample is received:

- IMU record gets `host_timestamp_s`
- left/right side-force records get `host_timestamp_s`
- reference 6D force record gets `host_timestamp_s`

Then training-time alignment can use nearest-neighbor or interpolation on host
time.

## Recommended Measurement Definition

For this project, the IMU output should focus on gravity-driven posture fields.

Recommended IMU features per sample:

- `host_timestamp_s`
- `pitch_deg`
- `roll_deg`
- `tilt_deg`
- optional `gyro_xyz`
- optional `accel_xyz`

This gives you both:

- a physically interpretable posture signal
- a simple signal for synchronization with force data

## Recommended Calibration Strategy

Use a short startup calibration before each collection session.

### A. Static bias calibration

Keep the device still for a few seconds and estimate:

- gyro bias
- accelerometer mean

### B. Mounting offset calibration

Define one device pose as the application zero pose, for example:

- device held upright
- no hand force applied

Record the IMU fused orientation at that instant and save it as:

- `imu_zero_pose`

Later samples can be expressed relative to that zero pose instead of the raw
sensor frame.

### C. Time synchronization check

Verify that:

- IMU timestamps
- side-force sensor timestamps
- reference 6D force timestamps

all come from the same host machine clock domain.

## Data Collection Protocol Suggestion

For each session:

1. Start host logging.
2. Keep the device still for IMU initialization.
3. Capture the application zero pose.
4. Start side-force logging.
5. Start reference 6D force logging.
6. Record synchronized multimodal interaction trials.
7. Save one session folder with raw streams and metadata.

Suggested session metadata:

- subject id
- trial id
- sensor sample rates
- IMU placement description
- zero-pose definition
- calibration values

## Main Technical Risks

### Risk 1. Dynamic acceleration corrupts gravity estimation

If the device moves quickly, raw accelerometer data will not equal gravity.
Mitigation:

- use sensor fusion instead of raw accelerometer-only tilt
- low-pass filter gravity-related signals
- design data collection with moderate motion first

### Risk 2. Side-force sensors may be insufficient for full 6D reconstruction

Two side sensors plus posture may not uniquely determine all 6D force/torque
components in all contact conditions.

This is the biggest modeling risk. The mapping may work only if:

- the interaction setup is constrained
- grasp/contact patterns are limited
- the geometry is consistent

You should expect some output dimensions to be easier than others.

### Risk 3. Sensor placement inconsistency

If the IMU or side sensors shift between sessions, the learned mapping will
degrade.

Mitigation:

- rigid mounting
- repeatable fixture
- explicit per-session calibration

### Risk 4. Time offset between streams

Even small delays can hurt learning if force changes quickly.

Mitigation:

- stamp all samples on one host
- log both sensor-side timestamp and host receive timestamp if available
- estimate residual lag during analysis

## Conclusion

Your overall idea is technically reasonable.

The IMU is suitable for estimating posture relative to gravity, especially the
tilt with respect to the vertical direction. But it should be treated as a
gravity-referenced posture sensor, not as a perfect full-attitude global
tracker. The learning pipeline is also sensible, but its success will depend
more on synchronization quality, mounting consistency, and whether the side
sensors plus posture contain enough information to recover the 6D target.

## Suggested Next Steps

1. Fix the logging schema for all three modalities.
2. Decide the IMU output representation to store.
3. Implement a minimal IMU logger with host timestamps.
4. Run a short pilot collection and inspect synchronization quality.
5. Evaluate which 6D label components are predictable first.

## Current Workspace Scope

This workspace now targets a reduced IMU-only posture logging task:

- input device: Intel RealSense D435i IMU
- software interface: official Python package `pyrealsense2`
- output fields: host timestamp plus gravity-driven `pitch_deg` and `roll_deg`

The first implementation uses:

- `accel` and `gyro` streams from the D435i IMU
- a short stationary initialization period
- gravity-driven `pitch_deg` / `roll_deg` computed directly from the gravity vector
- `pitch_deg` / `roll_deg` expressed relative to the startup zero pose
- JSONL logs for direct timestamp alignment with force sensors

## Logging Output

The default logger writes one JSON object per line to:

```text
output/imu_pitch_roll.jsonl
```

Each record contains at least:

- `host_timestamp_s`
- `pitch_deg`
- `roll_deg`

By default it also includes:

- `tilt_deg`
- `device_timestamp_ms`
- `frame_number`

The script also prints the current record to the terminal at the same rate as
the JSONL logger.

## Launch

List connected RealSense devices:

```bash
python data_collection/force_grasping/scripts/log_d435i_pitch_roll.py --list-devices
```

Run the logger with the default config:

```bash
python data_collection/force_grasping/scripts/log_d435i_pitch_roll.py
```

Use a custom config:

```bash
python data_collection/force_grasping/scripts/log_d435i_pitch_roll.py ^
  --config data_collection/force_grasping/config/default.yaml
```

At startup, keep the device still for a short period so the script can estimate
gyro bias and initialize the gravity-based posture.

## Important Note About Angle Convention

This logger computes `pitch_deg` and `roll_deg` directly from the measured
gravity vector instead of deriving them from a full Euler-angle decomposition.

That means the numerical sign and exact axis interpretation of `pitch_deg` and
`roll_deg` depend on how the D435i is mounted on your device. If your physical
meaning of those axes is different from the logger's default convention, we can
add one small mounting transform later without changing the rest of the data
pipeline.

Most importantly for this project, `pitch_deg` and `roll_deg` are gravity-driven
tilt measurements relative to the startup zero pose.
