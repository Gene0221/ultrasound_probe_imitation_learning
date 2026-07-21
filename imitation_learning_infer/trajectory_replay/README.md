# Franka trajectory replay

This workspace provides the first-stage robot validation controller for the ultrasound data collection project. It replays an offline Cartesian flange trajectory on a Franka Panda with a continuous 1 kHz libfranka control loop.

The intended validation path is:

```text
human-side collection
  -> tag-to-flange transform
  -> flange pose trajectory + optional target Fz
  -> replay_trajectory on Franka Panda
  -> record executed pose and force for validation
```

## Control design

The replay file is discrete, but the robot-side controller is continuous. The program interpolates between trajectory samples inside the libfranka callback and applies per-cycle translation and rotation rate limits.

Default behavior:

- Cartesian pose replay
- relative trajectory mode
- no force correction
- rate-limited 1 kHz command output

Optional behavior:

- absolute trajectory mode
- low-gain force correction along the current tool z axis

Force correction is disabled by default because the correct force sign and tool normal direction must be verified on the real probe mount before contact tests.

## CSV format

The trajectory file must contain a header:

```csv
time_s,x,y,z,qx,qy,qz,qw,target_fz
0.000,0.0000,0.0000,0.0000,0,0,0,1,8.0
0.050,0.0005,0.0000,0.0000,0,0,0,1,8.0
```

Columns:

- `time_s`: trajectory time in seconds, strictly increasing
- `x,y,z`: target translation in meters
- `qx,qy,qz,qw`: target orientation quaternion
- `target_fz`: desired normal contact force in N

In `relative` mode, each pose is interpreted as a transform relative to the robot pose at program start. In `absolute` mode, each pose is interpreted as an absolute base-to-flange pose.

## Build

On the Franka control computer:

```bash
cd /path/to/trajectory_replay
bash ./build.bash
```

## Run

Start with the robot away from contact:

```bash
bash ./launch.bash --robot-ip 172.16.0.2
```

For a slower first test:

```bash
bash ./launch.bash --robot-ip 172.16.0.2 --speed-scale 0.05 --max-translation-speed 0.005 --max-translation-acceleration 0.001 --max-rotation-speed 0.05 --ramp-time 5.0
```

Experimental force correction:

```bash
bash ./launch.bash --robot-ip 172.16.0.2 --enable-force-correction -- --force-gain 0.0002 --max-force-correction 0.003
```

To launch a converted session trajectory:

```bash
bash ./launch.bash --robot-ip 172.16.0.2 --trajectory /path/to/session_0001/franka_replay/replay_trajectory.csv --speed-scale 0.05 --max-translation-speed 0.005 --max-translation-acceleration 0.001 --max-rotation-speed 0.05 --ramp-time 5.0
```

## Safety notes

- Test first in free space with `--speed-scale 0.1`.
- Keep force correction disabled until the sign of measured `Fz` is verified.
- Use small trajectories and low Cartesian speeds for the first contact test.
- Keep the Franka Desk emergency stop and collision thresholds active.

## Convert a processed session to replay CSV

After running the preprocessing scripts that generate:

```text
<session>/transformed_pose/flange_pose_deltas.jsonl
<session>/predicted_force/predicted_force.jsonl
```

convert them into the replay format:

```bash
python scripts/session_to_replay_csv.py --config config/convert_session.yaml
```

The script writes:

```text
<session>/franka_replay/replay_trajectory.csv
```

`flange_pose_deltas.jsonl` stores frame-to-frame deltas, so the converter accumulates them into a start-relative trajectory. This matches `replay_trajectory --mode relative`.

Set the processed session directory in [config/convert_session.yaml](config/convert_session.yaml):

```yaml
paths:
  session: E:/research_data/rotate/output/session_0001
```

By default the converter reads:

```text
<session>/transformed_pose/flange_pose_deltas.jsonl
<session>/predicted_force/predicted_force.jsonl
```

and writes the generated replay files to `replay.output_dir`:

```text
config/replay_trajectory.csv
config/replay_trajectory_raw.csv
```

To write replay files into this workspace instead of the data session, set:

```yaml
replay:
  output_dir: ./config
  output_file: replay_trajectory.csv
```

Relative `output_dir` paths are resolved from `trajectory_replay/`, so this
writes `config/replay_trajectory.csv`.

`replay_trajectory.csv` is smoothed and resampled by default. This is recommended
for Franka replay because raw sampled pose trajectories can contain small
high-frequency direction changes that trigger acceleration discontinuity
reflexes. Tune this in [config/convert_session.yaml](config/convert_session.yaml):

```yaml
smoothing:
  enabled: true
  resample_dt_s: 0.02
  position_window: 9
  orientation_alpha: 0.15
  fixed_orientation: false
```

For the safest first test, set `fixed_orientation: true` to replay translation
only while holding the start orientation.

These directory and file names are configurable under `session_layout` and
`replay`. For one-off runs, command-line arguments still override the config:

```bash
python scripts/session_to_replay_csv.py --session /path/to/session_0001
python scripts/session_to_replay_csv.py --pose-file /path/to/flange_pose_deltas.jsonl --force-file /path/to/predicted_force.jsonl --output-dir ./config
```
