"""Fixed lognormal-quantile spectral aggregation of directional costs."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch


class LognormalSpectralResult(NamedTuple):
    """Value and detached finite-spectrum diagnostics."""

    value: torch.Tensor
    weights: torch.Tensor
    sigma: float
    entropy: float
    ess: float


def _validate_costs(h: torch.Tensor) -> None:
    if not isinstance(h, torch.Tensor):
        raise TypeError("h must be a torch.Tensor")
    if h.ndim != 1 or h.numel() == 0:
        raise ValueError(f"h must be a non-empty 1-D tensor, got {tuple(h.shape)}")
    if not h.is_floating_point():
        raise TypeError("h must have a floating dtype")
    if not bool(torch.isfinite(h).all()):
        raise ValueError("h contains NaN or Inf")
    if bool((h < 0).any()):
        raise ValueError("directional costs h must be non-negative")


def _validate_sigma(sigma: float) -> float:
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be finite and non-negative")
    return sigma


def lognormal_spectral_weights(
    L: int,
    sigma: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return exact cell-integrated lognormal-quantile spectrum weights.

    The two endpoint CDF values are inserted analytically as zero and one;
    ``ndtri`` is evaluated only at strict interior probabilities. Construction
    is performed in float64 and cast only at the return boundary when another
    floating output dtype was explicitly requested.
    """
    L = int(L)
    sigma = _validate_sigma(sigma)
    if L <= 0:
        raise ValueError("L must be positive")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating")
    target_device = torch.device(device)
    if L == 1:
        return torch.ones(1, device=target_device, dtype=dtype)
    if sigma == 0.0:
        return torch.full((L,), 1.0 / L, device=target_device, dtype=dtype)

    work_dtype = torch.float64
    interior = torch.arange(1, L, device=target_device, dtype=work_dtype) / L
    transformed = torch.special.ndtr(torch.special.ndtri(interior) - sigma)
    edges = torch.cat(
        (
            torch.zeros(1, device=target_device, dtype=work_dtype),
            transformed,
            torch.ones(1, device=target_device, dtype=work_dtype),
        )
    )
    weights = edges[1:] - edges[:-1]
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise RuntimeError("non-finite or negative lognormal spectral cell weight")
    # Telescoping endpoints make this one in exact arithmetic. Preserve the
    # closed form; do not clip or renormalize its cells.
    return weights.to(dtype=dtype)


def spectral_power(h: torch.Tensor, ordered_weights: torch.Tensor) -> torch.Tensor:
    """Apply fixed increasing-rank weights with a detached stable ordering."""
    _validate_costs(h)
    if not isinstance(ordered_weights, torch.Tensor):
        raise TypeError("ordered_weights must be a torch.Tensor")
    if ordered_weights.ndim != 1 or ordered_weights.numel() != h.numel():
        raise ValueError("ordered_weights must be 1-D and match h")
    weights = ordered_weights.detach().to(device=h.device, dtype=h.dtype)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("ordered_weights must be finite and non-negative")
    if bool((weights[1:] < weights[:-1]).any()):
        raise ValueError("ordered_weights must be nondecreasing")
    if not torch.allclose(weights.sum(), weights.new_tensor(1.0), atol=2e-14, rtol=2e-14):
        raise ValueError("ordered_weights must sum to one")
    order = torch.argsort(h.detach(), stable=True)
    assigned = torch.empty_like(weights)
    assigned[order] = weights
    return torch.sum(assigned.detach() * h)


def lognormal_spectral_power(h: torch.Tensor, sigma: float) -> LognormalSpectralResult:
    """Aggregate ``h`` by exact lognormal-quantile spectral cell weights.

    Rank construction and assigned weights are detached. Therefore autograd
    exposes the deterministic rank-assigned spectral subgradient. At sigma=0,
    the value uses ``mean`` directly for exact regression to uniform SPDHSW.
    """
    _validate_costs(h)
    sigma = _validate_sigma(sigma)
    ordered = lognormal_spectral_weights(h.numel(), sigma, h.device, h.dtype)
    order = torch.argsort(h.detach(), stable=True)
    assigned = torch.empty_like(ordered)
    assigned[order] = ordered
    assigned = assigned.detach()
    value = h.mean() if sigma == 0.0 else torch.sum(assigned * h)
    positive = assigned > 0
    entropy = float(-(assigned[positive] * assigned[positive].log()).sum())
    ess = float(1.0 / assigned.square().sum())
    return LognormalSpectralResult(value, assigned, sigma, entropy, ess)

