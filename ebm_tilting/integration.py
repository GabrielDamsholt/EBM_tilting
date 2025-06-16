from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import torch
import torchdiffeq
import torchsde
from torchsde import BrownianInterval

from ebm_tilting.model import Model
from ebm_tilting.types import (
    BatchedImage,
    BatchedIntegratorSolution,
    BatchedScalar,
    Scalar,
)
from ebm_tilting.utils import detransform, load_mnist_mean

ODE_INTEGRATOR_METHODS_ADAPTIVE = ["dopri5", "dopri8", "bosh3", "adaptive_heun"]
ODE_INTEGRATOR_METHODS_FIXED = ["euler", "midpoint", "rk4", "explicit_adams", "implicit_adams"]
ODE_INTEGRATOR_METHODS = ODE_INTEGRATOR_METHODS_ADAPTIVE + ODE_INTEGRATOR_METHODS_FIXED

SDE_INTEGRATOR_METHODS = ["euler", "milstein", "srk"]


class Integrator(ABC):
    def __init__(self, model: Model, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.mnist_mean = load_mnist_mean(device)
        super().__init__()

    def _setup(
        self,
        x: BatchedImage,
        num_solutions: int,
        num_steps: int,
        reverse: bool = False,
    ) -> tuple[float, BatchedScalar]:
        assert num_solutions >= 2

        device = x.device

        step_size = 1.0 / num_steps

        t = torch.linspace(0.0, 1.0, num_solutions).to(device)
        if reverse:
            t = 1.0 - t

        return step_size, t

    @abstractmethod
    def _integrate(
        self,
        t: BatchedScalar,
        x: BatchedImage,
        step_size: float,
        reverse: bool,
        method: str,
    ) -> BatchedIntegratorSolution:
        pass


    @torch.inference_mode()
    def integrate(
        self,
        z: BatchedImage,
        num_solutions: int,
        num_steps: int,
        method: str,
        **kwargs: Any,
    ) -> BatchedIntegratorSolution:
        step_size, t = self._setup(z, num_solutions, num_steps, reverse=False)
        solution = self._integrate(t, z, step_size, reverse=False, method=method, **kwargs)
        return detransform(solution, self.mnist_mean)

    @torch.inference_mode()
    def integrate_reverse(
        self,
        image: BatchedImage,
        num_solutions: int,
        num_steps: int,
        method: str,
        **kwargs: Any,
    ) -> BatchedIntegratorSolution:
        step_size, t = self._setup(image, num_solutions, num_steps, reverse=True)
        solution = self._integrate(t, image, step_size, reverse=True, method=method, **kwargs)
        return detransform(solution, self.mnist_mean)


class ODEIntegrator(Integrator):
    def __init__(self, model: Model, device: torch.device) -> None:
        super().__init__(model, device)

    def _validate_method(self, method: str) -> None:
        assert method in ODE_INTEGRATOR_METHODS

    def _velocity(self, t: Scalar, x: BatchedImage) -> BatchedImage:
        assert t.ndim == 0
        self.model(t.repeat(x.size(0)), x)
        return self.model.velocity

    def _integrate(
        self,
        t: BatchedScalar,
        x_start: BatchedImage,
        step_size: float,
        reverse: bool,
        method: str = "euler",
    ) -> BatchedIntegratorSolution:
        self._validate_method(method)

        options: dict[str, Any] = (
            {"dtype": torch.float32} if method in ODE_INTEGRATOR_METHODS_ADAPTIVE else {"step_size": step_size}
        )

        solver_fn = partial(
            torchdiffeq.odeint,
            t=t,
            method=method,
            rtol=1e-7,
            atol=1e-9,
            options=options,
        )

        solution: BatchedIntegratorSolution = solver_fn(self._velocity, x_start)  # type: ignore

        return solution


class SDE(torch.nn.Module):
    noise_type = "additive"
    sde_type = "ito"

    def __init__(
        self,
        model: Model,
        eps: float,
        max_eps: float,
        image_width: int,
        image_height: int,
        reverse: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.max_eps = max_eps
        self.eps = lambda t: eps * self.model.interpolant.eps_optimal(t, max_value=self.max_eps)
        self.image_width = image_width
        self.image_height = image_height
        self.reverse = reverse

    def _batch_t(self, t: torch.Tensor, batch_size: int) -> BatchedScalar:
        return t.repeat(batch_size)

    def _unsqueeze(self, x: BatchedImage) -> BatchedImage:
        batch_size = x.size(0)
        return x.view(batch_size, 1, self.image_width, self.image_height)

    # drift
    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        _t = self._batch_t(t, batch_size)

        _x = self._unsqueeze(x)

        self.model(_t, _x)

        drift = self.model.drift(self.eps(t), reverse=self.reverse)

        return drift.view(x.shape)

    # action of diffusion
    def g_prod(self, t: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        state_size = x.size(1)
        assert w.shape == (batch_size, state_size)

        _t = self._batch_t(t, batch_size)

        eps_t = self.eps(_t).view(batch_size, 1).expand(batch_size, state_size)

        diffusion = torch.sqrt(2.0 * eps_t) * w
        return diffusion if not self.reverse else -diffusion


class SDEIntegrator(Integrator):
    def __init__(self, model: Model, device: torch.device) -> None:
        super().__init__(model, device)

    def _validate_method(self, method: str) -> None:
        assert method in SDE_INTEGRATOR_METHODS

    def _integrate(
        self,
        t: BatchedScalar,
        x_start: BatchedImage,
        step_size: float,
        reverse: bool,
        method: str = "euler",
        eps: float = 1.0,
        max_eps: float = 1e6,
    ) -> BatchedIntegratorSolution:
        self._validate_method(method)
        assert eps > 0.0

        batch_size = x_start.size(0)
        num_channels = x_start.size(1)
        image_width = x_start.size(3)
        image_height = x_start.size(2)
        state_size = num_channels * image_width * image_height

        sde = SDE(self.model, eps, max_eps, image_width, image_height, reverse)
        bm = BrownianInterval(t0=0.0, t1=1.0, size=(batch_size, state_size), device=x_start.device)

        _solution: torch.Tensor = torchsde.sdeint(
            sde, x_start.flatten(start_dim=1), t, bm=bm, method=method, dt=step_size
        )  # type: ignore
        assert _solution.shape[-1] == x_start.size(1) * x_start.size(2) * x_start.size(3)
        solution: BatchedIntegratorSolution = _solution.view(-1, *x_start.shape)

        return solution
