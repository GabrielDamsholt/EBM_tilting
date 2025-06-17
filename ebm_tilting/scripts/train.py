import argparse
import os
from typing import Final, Optional

import matplotlib.pyplot as plt
import torch
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from ebm_tilting.integration import ODEIntegrator, SDEIntegrator
from ebm_tilting.interpolation import Interpolant, string_to_interpolant
from ebm_tilting.model import OBJECTIVES, Model
from ebm_tilting.types import BatchedImage, BatchedScalar
from ebm_tilting.unet import Unet
from ebm_tilting.utils import (
    add_model_init_args,
    create_mnist_dataset,
)

REGRESSORS: Final[list[str]] = ["eta_z", "eta_1", "score", "velocity", "drift"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an unconditional stochastic interpolant on MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory for saving model weights, results and generated images"
    )
    parser.add_argument("--load_path", type=str, default=None, help="Optional checkpoint to resume training from")
    add_model_init_args(parser)
    parser.add_argument("--objective", type=str, default="eta_z", choices=OBJECTIVES, help="Objective to predict")
    parser.add_argument("--regressor", type=str, default="eta_z", choices=REGRESSORS, help="Regressor to optimize for")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Optional max gradient norm after clipping")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--batches_per_epoch", type=int, default=None, help="Optional number of batches per epoch")
    return parser.parse_args()


def plot_image(image):
    plt.imshow(image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


def _plot_metric(metric, name, log_scale=False, y_lim=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metric)
    ax.set_xlabel("step")
    ax.set_ylabel(name)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(name)
    ax.grid(True)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    return fig


def plot_losses(losses, log_scale=True):
    return _plot_metric(losses, "loss", log_scale=log_scale)


def plot_learning_rates(learning_rates):
    return _plot_metric(learning_rates, "learning rate")


def plot_grad_norms(grad_norms):
    return _plot_metric(grad_norms, "gradient L2-norms")


def plot_eta_z_means(eta_z_means):
    return _plot_metric(eta_z_means, "mean of predicted eta_z")


def plot_eta_z_vars(eta_z_vars):
    return _plot_metric(eta_z_vars, "variance of predicted eta_z", y_lim=(0.0, 2.0))


# TODO: solve boundary behaviour at t=1 for drift and velocity
def _get_loss_weight(
    t: BatchedScalar,
    interpolant: Interpolant,
    regressor: str,
    objective: str,
) -> BatchedScalar:
    value = interpolant.beta(t) if objective == "eta_1" else interpolant.alpha(t)
    value += 1e-8

    if regressor == "eta_z":
        return (
            torch.ones_like(value)
            if objective == "eta_z"
            else (interpolant.beta(t) + 1e-8) / (interpolant.alpha(t) + 1e-8)
        )
    elif regressor == "eta_1":
        return (
            torch.ones_like(value)
            if objective == "eta_1"
            else (interpolant.alpha(t) + 1e-8) / (interpolant.beta(t) + 1e-8)
        )
    elif regressor == "score":
        return value
    elif regressor == "velocity":
        return value / (interpolant.eps_optimal(t) + 1e-8)
    elif regressor == "drift":
        return value / (2 * interpolant.eps_optimal(t) + 1e-8)
    else:
        raise ValueError(f"Unsupported regressor: {regressor}")


def train(
    model: Model,
    objective: str,
    regressor: str,
    dataloader: DataLoader,
    ema: EMA,
    device: torch.device,
    num_epochs: int,
    batches_per_epoch: Optional[int],
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: Optional[float],
    save_dir: str,
    z_sample: BatchedImage,
):
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_steps = num_epochs * len(dataloader)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    ema_backbone: torch.nn.Module = ema.ema_model  # type: ignore
    ema_model = Model(model.interpolant, objective, ema_backbone, clip_eta_1=False)

    ode_integrator = ODEIntegrator(ema_model, device)
    sde_integrator = SDEIntegrator(ema_model, device)

    generation_kwargs = {
        "z": z_sample,
        "num_solutions": 2,
        "num_steps": 20,
    }

    num_samples = z_sample.size(0)

    dir_sample_ode = os.path.join(save_dir, "samples_ode")
    os.makedirs(dir_sample_ode, exist_ok=True)
    dir_sample_sde = os.path.join(save_dir, "samples_sde")
    os.makedirs(dir_sample_sde, exist_ok=True)

    save_path = os.path.join(save_dir, "checkpoint.pt")

    model.backbone.train()

    losses = []
    learning_rates = []
    grad_norms = []
    eta_z_means = []
    eta_z_vars = []
    try:
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            progress_bar = tqdm(dataloader, desc="Batches", leave=False, total=batches_per_epoch)
            batch_number = 0

            for images, _labels in progress_bar:
                if batches_per_epoch is not None and batch_number >= batches_per_epoch:
                    break

                batch_size = images.size(0)
                images = images.to(device)
                z = torch.randn_like(images, device=device)
                t = torch.rand((batch_size,), device=device)

                optimizer.zero_grad()

                x_t = model.interpolant.interpolate(t, z, images)

                output = model(t, x_t)

                weight = _get_loss_weight(t, model.interpolant, regressor, objective)
                ground_truth = z if objective == "eta_z" else images
                loss = torch.mean((weight * (output - ground_truth)) ** 2)

                loss.backward()

                grads = [
                    param.grad.detach().flatten() for param in model.backbone.parameters() if param.grad is not None
                ]
                grad_norm = torch.cat(grads).norm()

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                ema.update()

                progress_bar.set_postfix(loss=loss.item())

                # log metrics
                eta_z = model.eta_z.detach()
                losses.append(loss.item())
                learning_rates.append(lr_scheduler.get_last_lr()[0])
                grad_norms.append(grad_norm.item())
                eta_z_var, eta_z_mean = torch.var_mean(eta_z)
                eta_z_means.append(eta_z_mean.item())
                eta_z_vars.append(eta_z_var.item())

                batch_number += 1

            # generate and save samples

            ema_backbone.eval()

            # unclipped ODE generation
            sample_ode = ode_integrator.integrate(
                **generation_kwargs,  # type: ignore
                method="midpoint",
            )[-1]
            fig_path_ode = os.path.join(dir_sample_ode, "epoch_{:0>3}.png".format(epoch + 1))
            save_image(sample_ode, fig_path_ode, nrow=num_samples)

            # unclipped SDE generation
            sample_sde = sde_integrator.integrate(
                **generation_kwargs,  # type: ignore
                method="milstein",
                eps=1.0,
                max_eps=1e3,
            )[-1]
            fig_path_sde = os.path.join(dir_sample_sde, "epoch_{:0>3}.png".format(epoch + 1))
            save_image(sample_sde, fig_path_sde, nrow=num_samples)

            torch.save(ema_backbone.state_dict(), save_path)

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    print(f"Final loss: {loss.item()}")

    torch.save(ema_backbone.state_dict(), save_path)

    return losses, learning_rates, grad_norms, eta_z_means, eta_z_vars


def main():
    args = parse_args()

    # set up device
    assert torch.mps.is_available()
    device = torch.device("mps")

    # load MNIST dataset
    mnist = create_mnist_dataset(fraction_ones=0.2)

    # set up dataloader
    dataloader = DataLoader(mnist, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=4)

    # set up interpolant
    interpolant = string_to_interpolant(args.interpolation)

    # set up model
    backbone = Unet(
        time_embedding_dim=args.time_embedding_dim,
        in_channels=1,
        out_channels=1,
        base_dim=args.init_features,
        dim_mults=args.dim_mults,
    ).to(device)
    model = Model(interpolant, args.objective, backbone, clip_eta_1=False)

    # load model weights if provided
    if args.load_path is not None:
        print(f"Loading model weights from {args.load_path}")
        backbone.load_state_dict(torch.load(args.load_path, weights_only=True, map_location=device))

    # set up exponential moving average
    ema = EMA(
        backbone,
        beta=0.995,
        update_every=10,
    ).to(device)

    # set up noise sample
    generator_sample = torch.Generator(device=device)
    seed_sample = 154471
    generator_sample.manual_seed(seed_sample)
    num_samples = 6
    z_sample = torch.randn(
        (num_samples, *(next(iter(dataloader))[0].shape[1:])), device=device, generator=generator_sample
    )

    # train model
    losses, learning_rates, grad_norms, eta_z_means, eta_z_vars = train(
        model,
        args.objective,
        args.regressor,
        dataloader,
        ema,
        device,
        args.num_epochs,
        args.batches_per_epoch,
        args.learning_rate,
        args.weight_decay,
        args.max_grad_norm,
        args.save_dir,
        z_sample,
    )

    # plot losses
    loss_fig = plot_losses(losses)
    loss_fig.savefig(os.path.join(args.save_dir, "loss.png"))

    # plot learning rates
    learning_rate_fig = plot_learning_rates(learning_rates)
    learning_rate_fig.savefig(os.path.join(args.save_dir, "learning_rate.png"))

    # plot gradient norms
    grad_norm_fig = plot_grad_norms(grad_norms)
    grad_norm_fig.savefig(os.path.join(args.save_dir, "grad_norm.png"))

    # plot predicted noise means and variances
    eta_z_mean_fig, eta_z_var_fig = (
        plot_eta_z_means(eta_z_means),
        plot_eta_z_vars(eta_z_vars),
    )
    eta_z_mean_fig.savefig(os.path.join(args.save_dir, "eta_z_mean.png"))
    eta_z_var_fig.savefig(os.path.join(args.save_dir, "eta_z_var.png"))


if __name__ == "__main__":
    main()
