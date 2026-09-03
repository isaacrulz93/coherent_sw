"""Aggregations of a finite vector of directional :math:`W_p^p` costs.

EVaR and CVaR return detached optimizer weights and expose the optimized value
with the corresponding Danskin subgradient.  The baselines expose their usual
full-gradient form and, where relevant, a detached-weight ablation.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch


class EVaRResult(NamedTuple):
    value: torch.Tensor
    beta: float
    weights: torch.Tensor
    achieved_kl: float
    entropy: float
    status: str


class CVaRResult(NamedTuple):
    value: torch.Tensor
    weights: torch.Tensor


def _validate_h(h: torch.Tensor) -> None:
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


def sw_power(h: torch.Tensor) -> torch.Tensor:
    """Uniform mean of directional p-costs."""
    _validate_h(h)
    return h.mean()


def ebsw_exp_power(
    h: torch.Tensor,
    beta: float,
    full_gradient: bool = True,
) -> torch.Tensor:
    """Self-normalized exponential EBSW baseline.

    ``torch.softmax`` uses the standard max-shift/log-sum-exp stabilization.
    """
    _validate_h(h)
    if not math.isfinite(float(beta)):
        raise ValueError("beta must be finite")
    weights = torch.softmax(h * float(beta), dim=0)
    if not full_gradient:
        weights = weights.detach()
    return torch.sum(weights * h)


def power_ebsw_power(
    h: torch.Tensor,
    gamma: float,
    eps: float = 1e-12,
    full_gradient: bool = True,
) -> torch.Tensor:
    """Pure-power EBSW, ``sum(h**(gamma+1))/sum(h**gamma)``.

    Scaling by ``max(h)`` prevents overflow without changing the quotient.
    The all-zero branch is exactly zero and has an exactly zero gradient.
    """
    _validate_h(h)
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if gamma == 0.0:
        return h.mean()

    scale = h.detach().max()
    if float(scale) == 0.0:
        # h.sum()*0 preserves a usable autograd edge for requires_grad inputs.
        return h.sum() * 0.0
    relative = h / scale.clamp_min(eps)
    weights = relative.pow(gamma)
    denominator = weights.sum()
    if not full_gradient:
        weights = (weights / denominator).detach()
    else:
        weights = weights / denominator
    return torch.sum(weights * h)


def entropic_power(h: torch.Tensor, beta: float) -> torch.Tensor:
    """Fixed-multiplier log-partition baseline.

    A centering identity avoids subtracting two nearly equal large quantities
    when ``beta`` is small.  At beta=0 the continuous limit is the mean.
    """
    _validate_h(h)
    beta = float(beta)
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if beta == 0.0:
        return h.mean()
    center = h.mean()
    log_count = h.new_tensor(math.log(h.numel()))
    return center + (torch.logsumexp(beta * (h - center), dim=0) - log_count) / beta


def _entropy(weights: torch.Tensor) -> float:
    positive = weights > 0
    return float(-(weights[positive] * weights[positive].log()).sum())


def _kl_uniform(weights: torch.Tensor) -> float:
    positive = weights > 0
    return float((weights[positive] * (weights[positive].log() + math.log(weights.numel()))).sum())


def evar_power(
    h: torch.Tensor,
    kappa: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> EVaRResult:
    """Worst directional mean in a discrete KL ball about uniform.

    The scalar multiplier and optimizer weights are solved from ``h.detach()``.
    Consequently the returned value has the exact Danskin gradient ``weights``
    and does not differentiate through the root solve.
    """
    _validate_h(h)
    kappa = float(kappa)
    tol = float(tol)
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and non-negative")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be positive")

    count = h.numel()
    solve_dtype = torch.float64
    # The optimizer is one-dimensional and L is typically small.  Solving on
    # CPU avoids one device synchronization per bisection iteration on CUDA;
    # only the final L-vector is copied back for the Danskin dot product.
    detached = h.detach().to(device="cpu", dtype=solve_dtype)
    uniform = torch.full_like(detached, 1.0 / count)

    if kappa == 0.0:
        weights = uniform.to(device=h.device, dtype=h.dtype).detach()
        value = torch.sum(weights * h)
        return EVaRResult(value, 0.0, weights, 0.0, math.log(count), "mean")

    maximum = detached.max()
    max_mask = detached == maximum
    number_maxima = int(max_mask.sum())
    kappa_max = math.log(count / number_maxima)
    max_weights = max_mask.to(detached.dtype) / number_maxima

    if kappa >= kappa_max:
        weights = max_weights.to(device=h.device, dtype=h.dtype).detach()
        value = torch.sum(weights * h)
        return EVaRResult(
            value,
            math.inf,
            weights,
            _kl_uniform(max_weights),
            _entropy(max_weights),
            "max",
        )

    # Constants have kappa_max=0 and have already taken the max branch.
    shifted = detached - maximum

    def evaluate(beta: float) -> tuple[torch.Tensor, float]:
        logits = shifted * beta
        weights = torch.softmax(logits, dim=0)
        # This form is the requested KL identity, with the max shift cancelling.
        kl_tensor = beta * torch.sum(weights * shifted) - torch.logsumexp(logits, dim=0)
        kl_tensor = kl_tensor + math.log(count)
        kl = max(0.0, float(kl_tensor))
        return weights, kl

    cost_range = float((maximum - detached.min()).abs())
    low = 0.0
    high = 1.0 / max(cost_range, torch.finfo(solve_dtype).tiny)
    high_weights, high_kl = evaluate(high)
    bracket_steps = 0
    while high_kl < kappa:
        low = high
        high *= 2.0
        bracket_steps += 1
        if not math.isfinite(high) or bracket_steps > 1024:
            raise RuntimeError("failed to bracket the EVaR multiplier")
        high_weights, high_kl = evaluate(high)

    best_weights = high_weights
    best_kl = high_kl
    beta = high
    status = "max_iter"
    for _ in range(int(max_iter)):
        beta = 0.5 * (low + high)
        best_weights, best_kl = evaluate(beta)
        if abs(best_kl - kappa) <= tol:
            status = "solved"
            break
        if best_kl < kappa:
            low = beta
        else:
            high = beta
    if status == "max_iter" and abs(best_kl - kappa) <= max(tol, 1e-12):
        status = "solved"

    weights = best_weights.to(device=h.device, dtype=h.dtype).detach()
    value = torch.sum(weights * h)
    return EVaRResult(value, beta, weights, best_kl, _entropy(best_weights), status)


def cvar_power(h: torch.Tensor, alpha: float) -> CVaRResult:
    """Worst directional mean under the cap ``pi_l <= 1/(alpha*L)``."""
    _validate_h(h)
    alpha = float(alpha)
    if not math.isfinite(alpha) or not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")

    count = h.numel()
    mass_count = alpha * count
    whole = int(math.floor(mass_count))
    fraction = mass_count - whole
    # Remove floating artifacts at integral boundaries such as .1 * 10.
    if abs(fraction) <= 32.0 * torch.finfo(torch.float64).eps * max(1.0, mass_count):
        fraction = 0.0
    order = torch.argsort(h.detach(), descending=True)
    weights = torch.zeros_like(h)
    if whole > 0:
        weights[order[:whole]] = 1.0 / mass_count
    if fraction > 0.0 and whole < count:
        weights[order[whole]] = fraction / mass_count
    weights = weights.detach()
    return CVaRResult(torch.sum(weights * h), weights)
