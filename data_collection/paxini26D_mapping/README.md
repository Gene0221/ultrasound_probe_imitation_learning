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

1. unified collection for `imu`, `paxini`, and `force6d`
2. session recording until you press `q`
3. dataset preparation
4. model training

The collection stage uses the same session controls as the hospital workspace:

- `Enter` to start a new session
- `Enter` again to pause and close the current session
- `Enter` again to resume into the next session
- `q` to stop the whole run

## Output Layout

- raw sessions: `sessions/session_xxxx/`
- aligned dataset: `dataset/`
- trained models: `model/`

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
