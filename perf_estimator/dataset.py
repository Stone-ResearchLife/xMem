import torchvision
import os
import torch
from typing import Tuple
from pathlib import Path


def _fetch_cifar10(split, transform) -> torch.utils.data.Dataset:
    temp_dataset_dir = os.path.join(Path().home(), "pytorch_datasets")
    return torchvision.datasets.CIFAR10(
        root=temp_dataset_dir,
        train=(split == "train"),
        download=True,
        transform=transform,
    )


def image_dataset(
    batch: int = 200, image_size: Tuple[int, int] = (86, 86), float16: bool = False
) -> torch.utils.data.DataLoader:

    transform_option = [
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
    if float16:
        transform_option.append(torchvision.transforms.Lambda(lambda x: x.half()))
    transform = torchvision.transforms.Compose(transform_option)
    # temp_dataset_dir = temp_dir_with_specific_path("pytorch", "datasets")
    # temp_dataset_dir = os.path.join(Path().home(), "pytorch_datasets")
    # train_data = torchvision.datasets.CIFAR10(
    #     root=temp_dataset_dir, train=True, download=True, transform=transform
    # )
    train_data = _fetch_cifar10("train", transform=transform)
    # train_data = torchvision.datasets.Food101(root=temp_dataset_dir, download=True, transform=transform)
    return torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)


class HuggingFaceCIFAR10(torch.utils.data.Dataset):
    def __init__(self, spilt, transform=None, image_size: Tuple[int, int] = (86, 86)):
        if transform is None:
            transform = [
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
            transform = torchvision.transforms.Compose(transform)

        self.data = _fetch_cifar10(spilt, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        image, label = self.data[idx]
        return {"pixel_values": image, "labels": label}
