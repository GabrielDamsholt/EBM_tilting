from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from ebm_tilting.types import Image

INTERPOLANTS: Final[list[str]] = [
    "linear",
    "linear_unit_var",
    "quadratic_unit_var",
    "sbdm",
    "polynomial",
    "sigmoidal_trigonometric",
]

MNIST_MEAN_PATH: Final[Path] = Path("mnist_mean.pt")
MNIST_STD: Final[float] = 0.3081


def add_model_init_args(parser: ArgumentParser, suffix: str = "") -> None:
    parser.add_argument(
        f"--init_features{suffix}",
        type=int,
        default=32,
        help="Number of features in the initial layers",
    )
    parser.add_argument(
        f"--time_embedding_dim{suffix}",
        type=int,
        default=128,
        help="Embedding dimension of time",
    )
    parser.add_argument(
        f"--dim_mults{suffix}",
        nargs="+",
        type=int,
        default=[2, 4],
        help="Dimension multipliers for the Unet",
    )
    parser.add_argument(
        f"--interpolation{suffix}",
        type=str,
        default="linear",
        choices=INTERPOLANTS,
        help="Interpolation method",
    )


def transform(
    tensor: torch.Tensor,
    mnist_mean: Image,
) -> torch.Tensor:
    return (tensor - mnist_mean) / MNIST_STD


def detransform(
    tensor: torch.Tensor,
    mnist_mean: Image,
) -> torch.Tensor:
    return tensor * MNIST_STD + mnist_mean


def load_mnist_mean(device: torch.device) -> Image:
    mnist_mean = torch.load(MNIST_MEAN_PATH, weights_only=True).to(device)
    mnist_mean.requires_grad = False
    return mnist_mean


# normalizes MNIST to mean 0 image and average pixel variance 1
class NormalizeMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = load_mnist_mean(torch.device("cpu"))
        self.std = MNIST_STD

    def forward(self, image: Image) -> Image:
        return (image - self.mean) / self.std

class BiasMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_val = 0.8
    
    def forward(self, image: Image) -> Image:
        return image.clamp(0.0, self.max_val)

def create_mnist_dataset(bias: bool, train: bool = True) -> Dataset:
    _transforms = [
        transforms.ToTensor()
    ]
    if bias:
        assert train
        _transforms.append(BiasMNIST())
    _transforms.append(NormalizeMNIST())
    transform = transforms.Compose(_transforms)

    dataset = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )

    return dataset
