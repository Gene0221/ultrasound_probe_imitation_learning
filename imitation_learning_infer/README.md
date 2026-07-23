# Realtime ultrasound policy deployment

This workspace is for online deployment of the trained ultrasound imitation
policy. It keeps model inference in Python and 1 kHz Franka control in C++.

```text
Python process
  ultrasound image -> ACT/diffusion policy -> 20 x 7 delta action chunk
  -> JSONL over TCP

C++ process
  receive full 20-step chunk -> accumulate SE(3) targets
  -> online causal filtering -> 1 kHz interpolation and motion limits
  -> libfranka Cartesian control
```

## First-version policy

- Default model type: `act`
- Dataset version: `without_force`
- Network action format: `dx, dy, dz, qx, qy, qz, qw`
- Action horizon: 20
- Default action spacing: 0.03 s
- Default speed scale: 0.4, so 20 steps cover about 1.5 s
- The Python sender transmits all 20 steps every inference cycle.
- The C++ controller executes only the first `execute_steps_per_inference`
  step(s), then waits for the next rolling prediction.

## Configuration

Edit [config/default.yaml](config/default.yaml). Important fields:

```yaml
policy:
  type: act
  model_dir: ../imitation_learning_ACT/runs/act_pose_h20
  dataset_version: without_force
  act_model:
    pretrained_resnet18: true
    freeze_encoder: false
    hidden_dim: 512
    nhead: 8
    num_decoder_layers: 4
    dim_feedforward: 2048
    dropout: 0.1

motion:
  action_dt_s: 0.03
  speed_scale: 0.4
  execute_steps_per_inference: 1

force_safety:
  enabled: true
  reader: placeholder  # set to kwr75b_serial for the real 6D sensor
```

Use an image folder for dry runs:

```yaml
ultrasound:
  source: image_folder
  image_dir: C:/path/to/ultrasound/images
```

Use the live camera on the Linux collection/control machine:

```yaml
ultrasound:
  source: live_camera
  live_config: ../data_collection/ultrasound_image_collection/config/default.yaml
```

## Dry-run sender

Until the live ultrasound adapter is wired in, the sender can loop over an
image folder:

```powershell
python python_sender/infer_sender.py --config config/default.yaml --image-dir C:\path\to\ultrasound_images
```

## C++ controller

The C++ controller listens for JSONL chunks from the Python sender and owns the
1 kHz libfranka callback. It currently implements:

- TCP server for Python policy chunks
- `20 x 7` action parsing
- delta-pose accumulation from the current commanded pose
- causal low-pass filtering for translation and orientation
- quintic interpolation over the executed prefix
- Cartesian speed and acceleration limiting
- hold-position behavior on timeout, force-safety violation, or missing policy

For real external force safety, set:

```yaml
force_safety:
  enabled: true
  reader: kwr75b_serial
```

The serial reader reuses the KWR75B protocol from
`../data_collection/6D_force_grasping` and sends `Fx/Fy/Fz/Mx/My/Mz` to the
C++ controller with every policy chunk.

Build on the Franka control computer:

```bash
cd cpp_controller
./build.bash
```

Run:

```bash
./launch.bash --robot-ip 172.16.0.2
```

Build and launch in one command:

```bash
./launch.bash --robot-ip 172.16.0.2 --build
```

Use `--print-only` to inspect the generated command without moving the robot:

```bash
./launch.bash --robot-ip 172.16.0.2 --print-only
```

## Mock Controller

For sender-side validation without the robot, run:

```powershell
python python_sender/mock_controller.py --host 127.0.0.1 --port 50555
```

Then start `infer_sender.py` in another terminal. The mock controller checks
that each JSONL packet contains a valid `20 x 7` action chunk.
