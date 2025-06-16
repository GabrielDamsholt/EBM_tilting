import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ebm_tilting.utils import MNIST_MEAN_PATH


def main():
    assert torch.mps.is_available()
    device = "mps"

    mnist = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    dataloader = DataLoader(
        mnist,
        batch_size=1000,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    mnist_mean = torch.zeros(1, 28, 28, device=device)

    num_samples = 0
    for images, _labels in tqdm(dataloader, desc="Batches"):
        images = images.to(device)
        mnist_mean += images.mean(dim=0)
        num_samples += images.size(0)

    mnist_mean /= num_samples
    torch.save(mnist_mean, MNIST_MEAN_PATH)


if __name__ == "__main__":
    main()
