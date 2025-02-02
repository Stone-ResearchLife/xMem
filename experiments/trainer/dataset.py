import torchvision
import torch
from typing import Tuple
from perf_estimator.config import Config, default_setting


def image_dataset(
    batch: int = 200,
    image_size: Tuple[int, int] = (86, 86),
    config: Config = default_setting,
) -> torch.utils.data.DataLoader:
    transform_option = [
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
    transform = torchvision.transforms.Compose(transform_option)
    train_data = torchvision.datasets.CIFAR10(
        root=config.dataset_dir, train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        train_data, batch_size=batch, shuffle=config.dataset.shuffle
    )
