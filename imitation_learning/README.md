# Unified ultrasound imitation learning

This workspace combines dataset building, ACT training, diffusion training,
realtime inference, Franka control, and offline trajectory replay.

## Layout

```text
imitation_learning/
  config/
    dataset.yaml
    act_train.yaml
    diffusion_train.yaml
    infer.yaml
  scripts/
    build_dataset.py
    train_act.py
    train_diffusion.py
    infer_sender.py
    mock_controller.py
  src/ultrasound_imitation/
    models/
    data/
    inference/
  cpp_controller/
  trajectory_replay/
```

The important change is that training and inference now use the same model
classes from `src/ultrasound_imitation/models`, so the ResNet image encoder and
ACT/diffusion policy bodies are built consistently.

## Dataset

```bash
cd imitation_learning
python scripts/build_dataset.py --config config/dataset.yaml
```

The unified builder writes one dataset layout shared by ACT and diffusion:

```text
dataset_root/
  without_force/
    train/samples.json
    train/images/session_0001/
    val/samples.json
    test/samples.json
  with_force/
    train/samples.json
```

## Training

ACT:

```bash
python scripts/train_act.py --config config/act_train.yaml
```

Diffusion:

```bash
python scripts/train_diffusion.py --config config/diffusion_train.yaml
```

Both checkpoints are saved under `imitation_learning/runs/...` by default and
include `model_state_dict`, `config`, and training history.

## Realtime Inference

One-command full realtime launch:

```bash
cd imitation_learning
./launch_realtime.bash --build
```

This starts the Python policy sender, verifies that the model and ultrasound
video stream are ready, starts the C++ controller, waits for the TCP connection,
and only begins streaming robot motion after you press Enter. The Franka IP
defaults to `franka.robot_ip` in `config/infer.yaml`.

You can still launch either side separately from the root script when debugging:

```bash
cd imitation_learning
./launch_realtime.bash --controller-only --build
```

Python sender only:

```bash
./launch_realtime.bash --sender-only
```

Robot-free realtime inference test:

```bash
python test_realtime_inference.py --image-dir /path/to/images
```

This launches no C++ controller and sends nothing to the robot. It loads the
policy, validates the ultrasound video stream, waits for Enter, runs policy
inference on the current frame, prints the filtered relative pose trajectory,
and writes replay-readable CSV files to `dry_run.output_dir` in `config/infer.yaml`.
By default one image frame produces the full 20-step action chunk as
`replay_trajectory.csv`; `replay_trajectory_raw.csv` keeps the unfiltered
cumulative trajectory.

For a sender-only dry run:

```bash
python scripts/mock_controller.py
python scripts/infer_sender.py --config config/infer.yaml --image-dir /path/to/images
```

## Legacy Workspaces

The old `imitation_learning_ACT`, `imitation_learning_difussion`, and
`imitation_learning_infer` workspaces have been folded into this directory.
