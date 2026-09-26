from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PerturbationStats:
    tensors: int
    parameters: int


class GaussianPerturber:
    """Apply seeded Gaussian noise and restore selected tensors bit-for-bit.

    A single CPU snapshot is kept for the life of the object. Restoring by copy,
    rather than subtracting noise in low precision, makes the restoration check exact.
    """

    def __init__(
        self,
        model: nn.Module,
        include: str | None = None,
        exclude: str | None = None,
        scale_mode: str = "absolute",
        verify_restore: bool = True,
    ) -> None:
        if scale_mode not in {"absolute", "rms_relative"}:
            raise ValueError("scale_mode must be 'absolute' or 'rms_relative'")
        include_pattern = re.compile(include) if include else None
        exclude_pattern = re.compile(exclude) if exclude else None
        self.model = model
        self.scale_mode = scale_mode
        self.verify_restore = verify_restore
        self._parameters: list[tuple[str, nn.Parameter]] = []
        self._snapshot: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if not parameter.is_floating_point():
                continue
            if include_pattern and not include_pattern.search(name):
                continue
            if exclude_pattern and exclude_pattern.search(name):
                continue
            self._parameters.append((name, parameter))
            self._snapshot[name] = parameter.detach().to(device="cpu", copy=True)
        if not self._parameters:
            raise ValueError("No parameters matched the perturbation filters")
        self._active = False

    @property
    def stats(self) -> PerturbationStats:
        return PerturbationStats(
            tensors=len(self._parameters),
            parameters=sum(parameter.numel() for _, parameter in self._parameters),
        )

    @torch.no_grad()
    def apply(self, seed: int, sigma: float) -> None:
        if self._active:
            raise RuntimeError("A perturbation is already active")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        generators: dict[torch.device, torch.Generator] = {}
        for _, parameter in self._parameters:
            device = parameter.device
            if device not in generators:
                generators[device] = torch.Generator(device=device).manual_seed(int(seed))
            noise = torch.randn(parameter.shape, dtype=parameter.dtype, device=device, generator=generators[device])
            scale = float(sigma)
            if self.scale_mode == "rms_relative":
                scale *= float(parameter.detach().float().square().mean().sqrt().item())
            parameter.add_(noise, alpha=scale)
        self._active = True

    @torch.no_grad()
    def restore(self) -> None:
        if not self._active:
            raise RuntimeError("No active perturbation to restore")
        for name, parameter in self._parameters:
            parameter.copy_(self._snapshot[name], non_blocking=False)
        if self.verify_restore:
            self.assert_restored()
        self._active = False

    @torch.no_grad()
    def assert_restored(self) -> None:
        mismatches: list[str] = []
        for name, parameter in self._parameters:
            if not torch.equal(parameter.detach().cpu(), self._snapshot[name]):
                mismatches.append(name)
                if len(mismatches) == 5:
                    break
        if mismatches:
            raise AssertionError(f"Exact restoration failed for: {', '.join(mismatches)}")

    @contextlib.contextmanager
    def perturb(self, seed: int, sigma: float) -> Iterator[None]:
        self.apply(seed, sigma)
        try:
            yield
        finally:
            self.restore()
