"""Exact empirical one-dimensional Wasserstein costs."""

from __future__ import annotations

import torch


def w_p_power_per_direction(
    source: torch.Tensor,
    target: torch.Tensor,
    p: float = 2.0,
) -> torch.Tensor:
    """Return one exact empirical ``W_p^p`` value per leading direction.

    ``source`` and ``target`` have shapes ``(L, n)`` and ``(L, m)``.  Uniform
    empirical weights and unequal sample counts are supported.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise ValueError("expected source=(L,n), target=(L,m) with common L")
    if source.shape[1] == 0 or target.shape[1] == 0:
        raise ValueError("empirical measures must be non-empty")
    if p < 1:
        raise ValueError("p must be at least one")
    n, m = source.shape[-1], target.shape[-1]
    source, _ = torch.sort(source, dim=-1)
    target, _ = torch.sort(target, dim=-1)
    source_cdf = torch.arange(1, n + 1, device=source.device, dtype=source.dtype) / n
    target_cdf = torch.arange(1, m + 1, device=target.device, dtype=target.dtype) / m
    axis, _ = torch.sort(torch.cat((source_cdf, target_cdf), dim=0))
    source_index = torch.searchsorted(source_cdf, axis).clamp(max=n - 1)
    target_index = torch.searchsorted(target_cdf, axis).clamp(max=m - 1)
    source_icdf = source[..., source_index]
    target_icdf = target[..., target_index]
    widths = torch.diff(torch.nn.functional.pad(axis, (1, 0)))
    return torch.sum(widths * torch.abs(source_icdf - target_icdf).pow(p), dim=-1)


def directional_costs(
    source: torch.Tensor,
    target: torch.Tensor,
    directions: torch.Tensor,
    p: float = 2.0,
) -> torch.Tensor:
    """Project two point clouds with one shared direction matrix."""
    if source.ndim != 2 or target.ndim != 2 or directions.ndim != 2:
        raise ValueError("expected source=(n,q), target=(m,q), directions=(L,q)")
    if source.shape[1] != target.shape[1] or source.shape[1] != directions.shape[1]:
        raise ValueError("ambient dimensions do not match")
    source_projection = source @ directions.T
    target_projection = target @ directions.T
    return w_p_power_per_direction(source_projection.T, target_projection.T, p=p)
