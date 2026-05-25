from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ultrasound_il.config import load_config
from ultrasound_il.inference import TemporalPolicyState, build_temporal_inputs
from ultrasound_il.models.temporal_policy import TemporalPolicyNet
from ultrasound_il.pose import pose_representation_dim
from ultrasound_il.transforms import build_image_transform
from ultrasound_il.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal continuous policy inference on cached observations.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint_path: Path, config: dict, device: torch.device) -> TemporalPolicyNet:
    model = TemporalPolicyNet(
        pose_output_dim=config["model"]["pose_output_dim"],
        pose_input_dim=pose_representation_dim(config["data"]["pose_representation"]),
        use_force=config["data"]["use_force"],
        force_dim=config["data"]["force_dim"],
        pretrained_backbone=False,
        freeze_backbone=False,
        pose_feature_dim=config["model"]["pose_feature_dim"],
        force_feature_dim=config["model"]["force_feature_dim"],
        lstm_hidden_dim=config["model"]["lstm_hidden_dim"],
        lstm_num_layers=config["model"]["lstm_num_layers"],
        lstm_dropout=config["model"]["lstm_dropout"],
        mlp_hidden_dim=config["model"]["mlp_hidden_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_next_pose_delta(
    model: TemporalPolicyNet,
    state: TemporalPolicyState,
    image_transform,
    device: torch.device,
) -> np.ndarray:
    image_tensor, history_pose_tensor, force_tensor = build_temporal_inputs(state, image_transform, device)
    prediction = model(
        image_tensor,
        history_pose_tensor,
        force_tensor if state.use_force else None,
    )
    return prediction.squeeze(0).detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["trainer"]["device"])
    model = build_model_from_checkpoint(Path(args.checkpoint), config, device)
    image_transform = build_image_transform(config["data"]["image_size"], is_train=False)

    state = TemporalPolicyState(
        window_size=config["data"]["window_size"],
        pose_representation=config["data"]["pose_representation"],
        use_force=config["data"]["use_force"],
        force_dim=config["data"]["force_dim"],
    )

    print("Example inference loop:")
    print("1. Maintain an 8-frame grayscale image buffer.")
    print("2. Maintain a 7-step executed pose-delta buffer.")
    print("3. Maintain a force buffer aligned with each image when use_force=true.")
    print("4. After each prediction, execute the predicted pose delta, capture the next image/force, then update buffers.")

    demo_image = np.zeros((config["data"]["image_size"], config["data"]["image_size"]), dtype=np.uint8)
    demo_force = np.zeros(config["data"]["force_dim"], dtype=np.float32)
    for _ in range(config["data"]["window_size"]):
        state.append_observation(demo_image, demo_force if config["data"]["use_force"] else None)
    for _ in range(config["data"]["window_size"] - 1):
        state.append_executed_pose_delta([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    predicted_pose_delta = predict_next_pose_delta(
        model=model,
        state=state,
        image_transform=image_transform,
        device=device,
    )
    print("Predicted next pose delta [tx, ty, tz, qx, qy, qz, qw]:")
    print(predicted_pose_delta)


if __name__ == "__main__":
    main()
