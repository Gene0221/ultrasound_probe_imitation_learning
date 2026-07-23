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

Start the C++ Franka controller first:

```bash
cd imitation_learning/cpp_controller
./launch.bash --robot-ip 172.16.0.2 --build
```

Then start the Python policy sender:

```bash
cd imitation_learning
python scripts/infer_sender.py --config config/infer.yaml
```

For a sender-only dry run:

```bash
python scripts/mock_controller.py
python scripts/infer_sender.py --config config/infer.yaml --image-dir /path/to/images
```

## Legacy Workspaces

The old `imitation_learning_ACT`, `imitation_learning_difussion`, and
`imitation_learning_infer` directories are left in place for compatibility
while this unified workspace is validated.
