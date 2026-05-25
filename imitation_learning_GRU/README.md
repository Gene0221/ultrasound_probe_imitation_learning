# Imitation Learning V1

This workspace contains the first continuous imitation learning prototype for validating:

"Whether historical ultrasound image sequences, historical pose-delta information, and optional force sequences are more helpful than a single image for predicting the next probe pose delta."

## Current Modeling Setup

- Main task: continuous pose-delta behavior cloning
- Temporal policy: `ResNet18 + pose MLP + optional force MLP + GRU + regression head`
- Baseline: `Single-frame ResNet18 + regression head`
- Image input: grayscale ultrasound copied into 3 channels
- Decision window: `8`
- Pose representation: `[tx, ty, tz, qx, qy, qz, qw]`
- Target: next-step `7D` pose delta
- Force input:
  - enabled with `use_force: true`
  - configured with `force_dim`, e.g. `2` or `6`
- Default loss: `SmoothL1Loss`

## Input Protocol

Before training, the raw data should be preprocessed into a per-timestep package. Each frame is expected to look like:

```json
{
  "image": "images/000000.png",
  "pose_delta_7d": [0.001, -0.002, 0.0, 0.0, 0.0, 0.01, 0.99995],
  "force": [0.3, -0.1, 0.0, 0.02, -0.03, 0.01]
}
```

Where:

- `image`: ultrasound image at the current timestep
- `pose_delta_7d`: pose delta at the current timestep
- `force`: optional force input
  - `force_dim=2` means a 2D force vector
  - `force_dim=6` means standard `fx, fy, fz, mx, my, mz`

## Temporal Alignment

For decision step `t`, the temporal model uses:

- Image sequence: `[I_(t-7), ..., I_t]`
- Historical pose sequence: `[p_(t-7), ..., p_(t-1), PAD_POSE]`
- Force sequence: `[F_(t-7), ..., F_t]`, ignored when `use_force=false`
- Supervision target: `p_t`

Notes:

- `p_t` is the next pose delta, not an absolute pose
- The last timestep corresponds to the current image `I_t`
- The model must not see the target `p_t` in its inputs, so the pose input at the last timestep is a zero-vector `PAD_POSE`

## Recommended Dataset Layout

```text
dataset/
  manifests/
    train_manifest.json
    val_manifest.json
  trajectories/
    traj_0001/
      images/
        000000.png
        000001.png
      metadata.json
```

Recommended `metadata.json` structure:

```json
{
  "trajectory_id": "traj_0001",
  "frames": [
    {
      "image": "images/000000.png",
      "pose_delta_7d": [0.001, 0.0, -0.001, 0.0, 0.0, 0.02, 0.9998],
      "force": [0.1, -0.2, 0.0, 0.01, 0.0, -0.01]
    }
  ]
}
```

Example `train_manifest.json` / `val_manifest.json`:

```json
{
  "use_force": true,
  "force_dim": 6,
  "trajectories": [
    {
      "trajectory_id": "traj_0001",
      "root_dir": "trajectories/traj_0001",
      "metadata_path": "trajectories/traj_0001/metadata.json"
    }
  ]
}
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate a toy dataset:

```bash
python tools/make_toy_dataset.py --output-dir data/toy_dataset --use-force --force-dim 6
```

Train the temporal model:

```bash
python train_temporal.py --config configs/temporal_resnet18_gru.yaml
```

Train the single-frame baseline:

```bash
python train_single_frame.py --config configs/single_frame_resnet18.yaml
```

## Directory Layout

```text
imitation_learing_v1/
  configs/
  data/
  outputs/
  src/ultrasound_il/
  tools/
  train_temporal.py
  train_single_frame.py
  infer_temporal.py
```

## Logged Metrics

Training and validation logs currently report:

- `loss`
- `mae`
- `translation_mae`
- `quaternion_mae`

## Current Default Assumptions

- Pose format is fixed as `[tx, ty, tz, qx, qy, qz, qw]`
- The quaternion is normalized when loading the data
- Force data is already aligned with image timesteps during preprocessing
- The baseline is kept as a single-frame continuous pose-delta regressor
