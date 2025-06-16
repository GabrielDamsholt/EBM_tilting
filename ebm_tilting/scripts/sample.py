import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from ebm_tilting.integration import (
    ODE_INTEGRATOR_METHODS,
    SDE_INTEGRATOR_METHODS,
    ODEIntegrator,
    SDEIntegrator,
)
from ebm_tilting.interpolation import string_to_interpolant
from ebm_tilting.model import Model
from ebm_tilting.types import BatchedIntegratorSolution
from ebm_tilting.unet import Unet
from ebm_tilting.utils import (
    add_model_init_args,
    create_mnist_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample MNIST images using a trained stochastic interpolant model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--load_path", type=str, required=True, help="Paths to load model weights from")
    add_model_init_args(parser)
    parser.add_argument("--objective", type=str, default="eta_z", help="Objective to predict")
    parser.add_argument("--seed", type=int, default=154471, help="Random seed for sampling initial noise")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of steps to take in solver if using a fixed-step integrator",
    )
    parser.add_argument("--clip_eta_1", action="store_true", help="Whether to clip eta_1 during sampling")

    subparsers = parser.add_subparsers(title="equation", required=True, dest="equation", help="Equation to solve")

    ode_parser = subparsers.add_parser("ODE")
    sde_parser = subparsers.add_parser("SDE")

    ode_parser.add_argument(
        "--method", type=str, default="euler", choices=ODE_INTEGRATOR_METHODS, help="ODE solver method"
    )

    sde_parser.add_argument(
        "--method", type=str, default="euler", choices=SDE_INTEGRATOR_METHODS, help="SDE solver method"
    )
    sde_parser.add_argument("--eps", type=float, default=1.0, help="Scaling constant for eps_optimal noise amplitude")
    sde_parser.add_argument(
        "--max_eps",
        type=float,
        default=1e6,
        help="Maximum noise amplitude (eps * eps_optimal capped at this value)",
    )

    return parser.parse_args()


def _show_images(ax, images, title=None):
    assert images.ndim == 3
    images = images.unsqueeze(0).squeeze(0)
    ax.imshow(images.cpu().permute(1, 2, 0).numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def _plot_samples(sample, num_samples, num_solutions, true_images=None):
    _fig, axs = plt.subplots(
        num_samples,
        1 + num_solutions + (1 if true_images is not None else 0),
        figsize=(14, 8),
        squeeze=False,
        sharex=True,
    )

    proj_sample = torch.zeros_like(sample[-1])

    for i in range(num_samples):
        _show_images(axs[i, 0], proj_sample[i])
        for j in range(1, num_solutions + 1):
            _show_images(axs[i, j], sample[j - 1, i])
    axs[0, 0].set_title("proj truth")

    for j in range(1, num_solutions + 1):
        t = (1.0 / (num_solutions - 1)) * (j - 1)
        axs[0, j].set_title(f"t = {t:.2f}")

    if true_images is not None:
        mses = ((sample[-1] - true_images) ** 2).mean(dim=(1, 2, 3))
        for i in range(num_samples):
            _show_images(axs[i, -1], true_images[i])
            axs[i, -1].set_xlabel(f"MSE: {mses[i]:.4f}")
        axs[0, -1].set_title("truth")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def main():
    args = parse_args()

    # Set up device
    assert torch.mps.is_available()
    device = torch.device("mps")

    # Set up interpolant
    interpolant = string_to_interpolant(args.interpolation)

    # Set up model
    backbone = Unet(
        time_embedding_dim=args.time_embedding_dim,
        in_channels=1,
        out_channels=1,
        base_dim=args.init_features,
        dim_mults=args.dim_mults,
    ).to(device)
    model = Model(interpolant, args.objective, backbone, clip_eta_1=args.clip_eta_1)

    # Load model weights
    model.backbone.load_state_dict(torch.load(args.load_path, weights_only=True, map_location=device))

    # Set model to evaluation mode
    model.backbone.eval()

    # Generate samples
    num_samples = 4
    num_intermediary_solutions = 9
    num_solutions = num_intermediary_solutions + 2

    sample_generator = torch.Generator(device=device)
    sample_generator_cpu = torch.Generator(device="cpu")
    sample_generator.manual_seed(args.seed)
    sample_generator_cpu.manual_seed(args.seed)

    mnist = create_mnist_dataset(train=True)

    dataloader = DataLoader(
        mnist,
        batch_size=num_samples,
        shuffle=True,
        generator=sample_generator_cpu,
    )

    images = next(iter(dataloader))[0].to(device)

    z = torch.randn(images.shape, device=device, generator=sample_generator)

    sample_kwargs = {
        "z": z,
        "num_solutions": num_solutions,
        "num_steps": args.num_steps,
        "method": args.method,
    }

    if args.equation == "ODE":
        integrator = ODEIntegrator(model, device)
        sample = integrator.integrate(**sample_kwargs)
    elif args.equation == "SDE":
        integrator = SDEIntegrator(model, device)
        sample = integrator.integrate(**sample_kwargs, eps=args.eps, max_eps=args.max_eps)

    assert isinstance(sample, BatchedIntegratorSolution)
    assert sample.size(0) == num_solutions
    assert sample.size(1) == num_samples

    _plot_samples(sample, num_samples, num_solutions)


if __name__ == "__main__":
    main()
