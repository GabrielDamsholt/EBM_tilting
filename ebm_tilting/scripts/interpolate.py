import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from ebm_tilting.interpolation import INTERPOLANTS, Interpolant, string_to_interpolant
from ebm_tilting.scripts.sample import _plot_samples
from ebm_tilting.utils import create_mnist_dataset, detransform, load_mnist_mean


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolate MNIST images using a given interpolant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "interpolation",
        type=str,
        choices=INTERPOLANTS,
        help="Interpolation method",
    )
    parser.add_argument("--seed", type=int, default=154471, help="Random seed for sampling initial noise")

    return parser.parse_args()


def _plot_quantities(interpolant: Interpolant, verbose=True):
    t = torch.linspace(0, 1, 1000)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, interpolant.alpha(t).squeeze(), label="alpha")
    ax[0].plot(t, interpolant.beta(t).squeeze(), label="beta")
    ax[0].legend()
    ax[0].set_ylim(0, 1)
    ax[1].plot(t, interpolant.eps_optimal(t).squeeze(), label="eps_optimal", color="black")
    ax[1].plot(
        t,
        interpolant.eps_optimal(t).squeeze() / interpolant.alpha(t).squeeze(),
        label="eps_optimal_divided_by_alpha",
        color="black",
        linestyle="--",
    )
    ax[1].legend()
    ax[1].set_xlabel("t")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 10)

    if verbose:
        print(f"eps_optimal(0) = {interpolant.eps_optimal(torch.tensor(0)).item()}")
        print(f"eps_optimal(0.00001) = {interpolant.eps_optimal(torch.tensor(0.00001)).item()}")
        print(f"eps_optimal(0.99999) = {interpolant.eps_optimal(torch.tensor(0.99999)).item()}")
        print(f"eps_optimal(1) = {interpolant.eps_optimal(torch.tensor(1)).item()}")

    plt.show()


def main():
    args = parse_args()

    # Set up device
    assert torch.mps.is_available()
    device = torch.device("mps")

    # Set up interpolator
    interpolant = string_to_interpolant(args.interpolation)

    # Generate samples
    num_samples = 4
    num_intermediary_solutions = 9
    num_solutions = num_intermediary_solutions + 2
    sample_generator = torch.Generator(device=device)
    sample_generator_cpu = torch.Generator(device="cpu")
    sample_generator.manual_seed(args.seed)
    sample_generator_cpu.manual_seed(args.seed)

    mnist = create_mnist_dataset(bias=True, train=True)

    dataloader = DataLoader(
        mnist,
        batch_size=num_samples,
        shuffle=True,
        generator=sample_generator_cpu,
    )

    images = next(iter(dataloader))[0].to(device)

    z = torch.randn(images.shape, device=device, generator=sample_generator)

    interpolated_images = torch.empty(num_solutions, num_samples, 1, 28, 28, device=device)
    for i, t in enumerate(torch.linspace(0, 1, num_solutions)):
        interpolated_images[i] = interpolant.interpolate(t.repeat(num_samples).to(device), z, images)

    mnist_mean = load_mnist_mean(device)
    _plot_samples(detransform(interpolated_images, mnist_mean), num_samples, num_solutions)

    _plot_quantities(interpolant)


if __name__ == "__main__":
    main()
