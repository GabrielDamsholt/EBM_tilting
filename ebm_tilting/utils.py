from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np

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

def create_mnist_dataset(fraction_ones: float) -> Dataset:
    assert fraction_ones >= 0.0
    assert fraction_ones <= 1.0

    _transforms = [
        transforms.ToTensor(),
        NormalizeMNIST(),
    ]
    transform = transforms.Compose(_transforms)

    mnist_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    labels = mnist_train.targets.numpy()

    indices_0 = np.where(labels == 0)[0]

    indices_1_all = np.where(labels == 1)[0]
    num_1_to_keep = int(len(indices_1_all) * fraction_ones)
    indices_1_subset = np.random.choice(indices_1_all, num_1_to_keep, replace=False)

    final_indices = np.concatenate([indices_0, indices_1_subset])
    np.random.shuffle(final_indices)

    subset_train = Subset(mnist_train, final_indices)

    return subset_train
