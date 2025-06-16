from typing import Final, Optional

import torch
from torch import nn

from ebm_tilting.interpolation import Interpolant
from ebm_tilting.types import BatchedImage, BatchedScalar
from ebm_tilting.utils import detransform, load_mnist_mean, transform

OBJECTIVES: Final[list[str]] = ["eta_z", "eta_1"]


class Model(nn.Module):
    def __init__(self, interpolant: Interpolant, objective: str, backbone: nn.Module, clip_eta_1: bool = False, baseline: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.interpolant = interpolant
        self.objective = objective
        self.backbone = backbone
        self.clip_eta_1 = clip_eta_1
        self.mnist_mean = load_mnist_mean(next(backbone.parameters()).device)
        self.baseline = baseline

    def _maybe_clip_eta_1(self, eta_1: BatchedImage) -> BatchedImage:
        if self.clip_eta_1:
            return transform(detransform(eta_1, self.mnist_mean).clamp(min=0.0, max=1.0), self.mnist_mean)
        else:
            return eta_1

    def forward(self, t: BatchedScalar, x: BatchedImage) -> BatchedImage:
        output = self.backbone(t, x)
        if self.baseline is not None:
            output += self.baseline(t, x)

        self._cached_t: BatchedScalar = t
        self._cached_x: BatchedImage = x
        self._cached_output: BatchedImage = output

        t_view = t.view(-1, 1, 1, 1)
        zero = torch.zeros_like(output)
        if self.objective == "eta_z":
            self._cached_output = torch.where(t_view == 1.0, zero, output)
            self._cached_output = torch.where(t_view == 0.0, x, output)
        else:
            assert self.objective == "eta_1"
            self._cached_output = torch.where(t_view == 0.0, zero, output)
            self._cached_output = torch.where(t_view == 1.0, x, output)

        return output

    @property
    def eta_z(self) -> BatchedImage:
        if self.objective == "eta_z":
            if not self.clip_eta_1:
                return self._cached_output
            else:
                eta_1 = self._maybe_clip_eta_1(
                    self.interpolant.eta_1(self._cached_t, self._cached_x, self._cached_output)
                )
                return self.interpolant.eta_z(self._cached_t, self._cached_x, eta_1)
        else:
            assert self.objective == "eta_1"
            return self.interpolant.eta_z(self._cached_t, self._cached_x, self._maybe_clip_eta_1(self._cached_output))

    @property
    def eta_1(self) -> BatchedImage:
        if self.objective == "eta_1":
            eta_1 = self._cached_output
        else:
            assert self.objective == "eta_z"
            eta_1 = self.interpolant.eta_1(self._cached_t, self._cached_x, self._cached_output)
        return self._maybe_clip_eta_1(eta_1)

    @property
    def score(self) -> BatchedImage:
        return self.interpolant.score(self._cached_t, self.eta_z)

    @property
    def velocity(self) -> BatchedImage:
        return self.interpolant.velocity(self._cached_t, self.eta_z, self.eta_1)

    def drift(self, eps: BatchedScalar, reverse: bool = False) -> BatchedImage:
        return self.interpolant.drift(self.velocity, self.score, eps, reverse=reverse)
