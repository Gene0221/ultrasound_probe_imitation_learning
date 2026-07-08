# paxini26D_mapping

This workspace collects IMU + Paxini + 6D sessions, aligns them into a mapping
dataset, and trains the force-mapping model.

## Main Entrypoint

```powershell
cd C:\Users\zhj80\OneDrive\Desktop\Master Course Material\research\data_collection\paxini26D_mapping
python main.py
```

## Full Pipeline

`main.py` runs:

1. asks for an experiment name
2. unified collection for `imu`, `paxini`, and `force6d`
3. session recording until you press `q`
4. dataset preparation
5. model training

The collection stage uses the same session controls as the hospital workspace:

- `Enter` to start a new session
- `Enter` again to pause and close the current session
- `Enter` again to resume into the next session
- `q` to stop the whole run

When this integrated workspace launches child modules, the top-level
`modules.*` entries in [config/default.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini26D_mapping/config/default.yaml)
act as the effective runtime config for those children. For example, the
current 26D config disables human-readable terminal streaming for Paxini and
6D force, and disables IMU stdout printing during integrated collection.

## Output Layout

- raw sessions: `sessions/session_xxxx/`
- aligned dataset: `dataset/`
- trained models: `model/`

When an experiment name is entered, the aligned dataset is saved as:

```text
dataset/EXPERIMENT_NAME.pt
```

and the model run directory is saved as:

```text
model/EXPERIMENT_NAME_YYYY-MM-DDTHH-MM-SSZ/
```

Each training run writes a model directory under `model/`:

```text
model/<run_name>/
  model.pt
  summary.json
  dataset_split/
    train.pt
    val.pt
    test.pt
```

The split ratios are kept as `train/val/test = 0.7/0.15/0.15` according to
`config/default.yaml`. The validation and test sets are held out from training.
`summary.json` reports training, validation, and test metrics, plus the
validation/test gaps relative to the training set:

```json
{
  "evaluation": {
    "train": {"mse": 0.0, "mae": 0.0, "rmse": 0.0},
    "validation": {"mse": 0.0, "mae": 0.0, "rmse": 0.0},
    "test": {"mse": 0.0, "mae": 0.0, "rmse": 0.0},
    "validation_gap_from_train": {
      "mse_gap": 0.0,
      "mae_gap": 0.0,
      "rmse_gap": 0.0
    },
    "test_gap_from_train": {
      "mse_gap": 0.0,
      "mae_gap": 0.0,
      "rmse_gap": 0.0
    }
  }
}
```

The JSON report does not print split indices. The saved `.pt` split files are
used for downstream inspection or reproduction.

## Inference Role

This workspace no longer owns hospital-side inference-input collection.
Hospital data collection is responsible for collecting IMU + Paxini inputs.

After a hospital session is collected, use:

```powershell
python scripts/predict_force.py --session <session_dir_or_name>
```

to run the trained mapping model on those collected IMU + Paxini signals.

## Important Files

- [main.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini26D_mapping/main.py)
- [config/default.yaml](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini26D_mapping/config/default.yaml)
- [scripts/prepare_dataset.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini26D_mapping/scripts/prepare_dataset.py)
- [scripts/train_model.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini26D_mapping/scripts/train_model.py)
- [scripts/predict_force.py](/C:/Users/zhj80/OneDrive/Desktop/Master%20Course%20Material/research/data_collection/paxini26D_mapping/scripts/predict_force.py)

This workspace no longer contains separate collection wrappers. `main.py` is the
single user-facing entrypoint for collect-prepare-train.
