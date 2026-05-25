from __future__ import annotations

from torchvision import transforms


def build_image_transform(image_size: int, is_train: bool) -> transforms.Compose:
    steps = [
        transforms.Resize((image_size, image_size)),
    ]
    if is_train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)
