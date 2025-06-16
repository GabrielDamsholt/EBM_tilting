from abc import ABC, abstractmethod

import torch

from ebm_tilting.types import (
    BatchedImage,
    BatchedScalar,
    ExpandedScalar,
    MaybeBatchedScalar,
)
from ebm_tilting.utils import INTERPOLANTS


# spatially linear one-sided stochastic interpolant
class Interpolant(ABC):
    def alpha(self, t: MaybeBatchedScalar) -> ExpandedScalar:
        return self._alpha(t).view(-1, 1, 1, 1)

    def alpha_dot(self, t: MaybeBatchedScalar) -> ExpandedScalar:
        return self._alpha_dot(t).view(-1, 1, 1, 1)

    def beta(self, t: MaybeBatchedScalar) -> ExpandedScalar:
        return self._beta(t).view(-1, 1, 1, 1)

    def beta_dot(self, t: MaybeBatchedScalar) -> ExpandedScalar:
        return self._beta_dot(t).view(-1, 1, 1, 1)

    @abstractmethod
    def _alpha(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        pass

    @abstractmethod
    def _alpha_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        pass

    @abstractmethod
    def _beta(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        pass

    @abstractmethod
    def _beta_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        pass

    def eps_optimal(self, t: MaybeBatchedScalar, max_value: float = float("inf")) -> MaybeBatchedScalar:
        t_clamped = t.clamp(min=1e-6, max=1.0 - 1e-6)
        eps_optimal = self.alpha(t_clamped) ** 2 * (self.beta_dot(t_clamped) / (self.beta(t_clamped))) - self.alpha(
            t_clamped
        ) * self.alpha_dot(t_clamped)

        return torch.clamp(eps_optimal, max=max_value)

    def eta_z(self, t: MaybeBatchedScalar, x: BatchedImage, eta_1: BatchedImage) -> BatchedImage:
        return (x - self.beta(t) * eta_1) / (self.alpha(t) + 1e-8)

    def eta_1(self, t: MaybeBatchedScalar, x: BatchedImage, eta_z: BatchedImage) -> BatchedImage:
        return (x - self.alpha(t) * eta_z) / (self.beta(t) + 1e-8)

    def score(self, t: MaybeBatchedScalar, eta_z: BatchedImage) -> BatchedImage:
        return -eta_z / (self.alpha(t) + 1e-8)

    def velocity(
        self,
        t: MaybeBatchedScalar,
        eta_z: BatchedImage,
        eta_1: BatchedImage,
    ) -> BatchedImage:
        return self.alpha_dot(t) * eta_z + self.beta_dot(t) * eta_1

    def drift(
        self,
        velocity: BatchedImage,
        score: BatchedImage,
        eps: BatchedScalar,
        reverse: bool = False,
    ) -> BatchedImage:
        if not reverse:
            return velocity + eps * score
        else:
            return velocity - eps * score

    def interpolate(self, t: BatchedScalar, z: BatchedImage, image: BatchedImage) -> BatchedImage:
        return self.alpha(t) * z + self.beta(t) * image


class LinearInterpolant(Interpolant):
    def _alpha(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return torch.ones_like(t) - t

    def _alpha_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return -torch.ones_like(t)

    def _beta(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return t

    def _beta_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return torch.ones_like(t)


class LinearUnitVarianceInterpolant(Interpolant):
    def _scale(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return torch.sqrt(torch.ones_like(t) + 2 * (t**2 - t))

    def _dot_scale(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return (1 + 2 * (t**2 - t)) ** (3.0 / 2.0)

    def _alpha(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return (torch.ones_like(t) - t) / self._scale(t)

    def _alpha_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return -torch.ones_like(t) / self._dot_scale(t)

    def _beta(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return t / self._scale(t)

    def _beta_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return (torch.ones_like(t) - t) / self._dot_scale(t)


class QuadraticUnitVarianceInterpolant(Interpolant):
    def _scale(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return torch.sqrt((torch.ones_like(t) - t**2) ** 2 + t**4)

    def _dot_scale(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return (1 + 2 * t**4 - 2 * t**2) ** (3.0 / 2.0)

    def _alpha(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return (torch.ones_like(t) - t**2) / self._scale(t)

    def _alpha_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return -(2 * t**3) / self._dot_scale(t)

    def _beta(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return t**2 / self._scale(t)

    def _beta_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return (2 * t * (1 - t**2)) / self._dot_scale(t)


class SBDMInterpolant(Interpolant):
    def _alpha(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return torch.sqrt(torch.ones_like(t) - t**2)

    def _alpha_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return -t / self._alpha(t.clamp(max=1.0 - 1e-6))

    def _beta(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return t

    def _beta_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return torch.ones_like(t)


class PolynomialInterpolant(Interpolant):
    def __init__(self, degree: int = 2):
        if degree <= 1:
            raise ValueError(f"degree must be greater than 1, got {degree}")
        self.degree = degree

    def _alpha(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return torch.ones_like(t) - t**self.degree

    def _alpha_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return -self.degree * t ** (self.degree - 1)

    def _beta(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return t**self.degree

    def _beta_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return self.degree * t ** (self.degree - 1)


class SigmoidalTrigonometricInterpolant(Interpolant):
    def _alpha(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return 0.5 * (1 + torch.cos(torch.pi * t))

    def _alpha_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return -0.5 * torch.pi * torch.sin(torch.pi * t)

    def _beta(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return 0.5 * (1 - torch.cos(torch.pi * t))

    def _beta_dot(self, t: MaybeBatchedScalar) -> MaybeBatchedScalar:
        return 0.5 * torch.pi * torch.sin(torch.pi * t)


def string_to_interpolant(interpolation: str) -> Interpolant:
    if interpolation == "linear":
        return LinearInterpolant()
    elif interpolation == "linear_unit_var":
        return LinearUnitVarianceInterpolant()
    elif interpolation == "quadratic_unit_var":
        return QuadraticUnitVarianceInterpolant()
    elif interpolation == "sbdm":
        return SBDMInterpolant()
    elif interpolation == "polynomial":
        return PolynomialInterpolant()
    elif interpolation == "sigmoidal_trigonometric":
        return SigmoidalTrigonometricInterpolant()
    else:
        raise ValueError(f"`interpolation` must be one of {INTERPOLANTS}")
