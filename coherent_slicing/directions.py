"""Shared Euclidean direction sampling."""

from __future__ import annotations

import torch


def sample_unit_directions(
    count: int,
    dimension: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if count <= 0 or dimension <= 0:
        raise ValueError("count and dimension must be positive")
    generator = torch.Generator(device=device).manual_seed(int(seed))
    directions = torch.randn(count, dimension, generator=generator, device=device, dtype=dtype)
    return directions / directions.norm(dim=1, keepdim=True)

