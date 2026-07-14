# imitation_learning_ACT

ACT-style action-chunk training for ultrasound-guided flange pose deltas.

## Dataset

Build datasets from processed hospital sessions:

```powershell
python scripts/build_dataset.py --config config/dataset.yaml
```

The builder uses ultrasound frames as the sample timeline. Each sample contains:

- copied ultrasound image
- future `action_horizon` flange pose deltas, default `20 x 7`
- optional 1D normal force in the `with_force` version

Splits are assigned by source session to avoid neighboring frames leaking
between train/val/test.

Output layout:

```text
dataset_root/
  without_force/
    train/session_0001/images/
    train/session_0001/samples.json
    val/
    test/
  with_force/
    train/session_0001/images/
    train/session_0001/samples.json
```

## Training

Train the ACT Transformer chunk policy:

```powershell
python scripts/train.py --config config/train.yaml
```

The model is:

```text
ultrasound image
  -> ResNet18 feature map
  -> visual tokens + positional embedding
  -> Transformer decoder cross-attention from action queries
  -> future pose delta chunk
```

Each future step has one learnable action query, so `action_horizon=20`
produces 20 decoded action tokens. Each token is mapped to one 7D flange
pose delta `[dx, dy, dz, dqx, dqy, dqz, dqw]`.

For `with_force`, the 1D normal force value is projected as an extra memory
token for the Transformer decoder.

The default supervised objective is chunk-level L1 loss:

```text
L = mean(|predicted_pose_delta_chunk - target_pose_delta_chunk|)
```
