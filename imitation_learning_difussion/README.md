# imitation_learning_difussion

Diffusion-policy-style action-chunk training for ultrasound-guided flange pose
deltas.

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
    train/images/session_0001/
    train/samples.json
    val/
    test/
  with_force/
    train/images/session_0001/
    train/samples.json
```

## Training

Train the baseline conditioned diffusion model:

```powershell
python scripts/train.py --config config/train.yaml
```

The baseline is:

```text
ultrasound image -> ResNet18 condition
noisy future pose chunk + timestep + condition -> MLP denoiser
```

For `with_force`, the 1D force value is concatenated to the conditioning vector.

