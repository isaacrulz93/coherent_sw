from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm

from coherent_slicing import (
    directional_costs,
    lognormal_spectral_power,
    lognormal_spectral_weights,
)


DTYPE = torch.float64
EXTERNAL = Path("/home/pikachu/EBSPDSW")
if str(EXTERNAL) not in sys.path:
    sys.path.insert(0, str(EXTERNAL))

from evobank.bank import Bank  # noqa: E402
from evobank.ot1d import w2_squared_per_direction  # noqa: E402
from evobank.svec import SvecBasis  # noqa: E402


def _symmetric(count: int, d: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(count, d, d, generator=generator, dtype=DTYPE)
    return 0.5 * (raw + raw.transpose(-1, -2))


def _explicit_triple() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]], dtype=DTYPE)
    nu = torch.tensor([[-1.0, -1.0], [0.0, 1.0]], dtype=DTYPE)
    eta = torch.tensor([[1.0, -2.0], [1.0, -2.0]], dtype=DTYPE)
    return mu, nu, eta


def _angular_directions(count: int) -> torch.Tensor:
    angles = (torch.arange(count, dtype=DTYPE) + 0.5) * math.pi / count
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def test_sigma_zero_is_exact_uniform_mean() -> None:
    h = torch.tensor([0.2, 0.7, 2.1, 4.0], dtype=DTYPE)
    result = lognormal_spectral_power(h, 0.0)
    assert torch.equal(result.weights, torch.full_like(h, 0.25))
    assert torch.equal(result.value, h.mean())


@pytest.mark.parametrize("L", [1, 2, 40, 500, 2000])
@pytest.mark.parametrize("sigma", [0.0, 0.5, 1.0, 1.25, 1.5])
def test_weight_shape_monotonicity_and_mass(L: int, sigma: float) -> None:
    weights = lognormal_spectral_weights(L, sigma, "cpu", DTYPE)
    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0).all())
    assert bool((weights[1:] >= weights[:-1]).all())
    assert float(weights.sum()) == pytest.approx(1.0, abs=3e-15)


def test_closed_form_weights_match_high_accuracy_quadrature() -> None:
    L, sigma = 7, 1.25
    weights = lognormal_spectral_weights(L, sigma, "cpu", DTYPE).numpy()
    numerical = []
    for index in range(L):
        lo = norm.ppf(index / L) if index else -np.inf
        hi = norm.ppf((index + 1) / L) if index + 1 < L else np.inf
        value, error = quad(lambda x: norm.pdf(x - sigma), lo, hi, epsabs=2e-14, epsrel=2e-14, limit=300)
        assert error < 2e-12
        numerical.append(value)
    assert np.allclose(weights, numerical, rtol=2e-13, atol=2e-14)


def test_permutation_invariance() -> None:
    h = torch.tensor([0.03, 4.0, 0.4, 1.2, 0.8], dtype=DTYPE)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    assert torch.allclose(
        lognormal_spectral_power(h, 1.25).value,
        lognormal_spectral_power(h[permutation], 1.25).value,
        atol=2e-15,
        rtol=2e-15,
    )


@pytest.mark.parametrize("scale", [0.01, 0.25, 2.0, 17.0])
def test_positive_homogeneity(scale: float) -> None:
    h = torch.tensor([0.02, 0.3, 0.9, 2.7, 8.0], dtype=DTYPE)
    base = lognormal_spectral_power(h, 1.0)
    scaled = lognormal_spectral_power(scale * h, 1.0)
    assert torch.allclose(scaled.value, scale * base.value, atol=2e-14, rtol=2e-14)


def test_common_direction_triangle_explicit_and_random_fields() -> None:
    directions = _angular_directions(1 << 14)
    mu, nu, eta = _explicit_triple()
    costs = [directional_costs(a, b, directions, p=1) for a, b in ((mu, nu), (nu, eta), (mu, eta))]
    distances = [lognormal_spectral_power(h, 1.5).value for h in costs]
    assert float(distances[2] - distances[0] - distances[1]) <= 2e-12

    worst = -math.inf
    for seed in range(20):
        generator = torch.Generator().manual_seed(7000 + seed)
        left = torch.rand(127, generator=generator, dtype=DTYPE)
        right = torch.rand(127, generator=generator, dtype=DTYPE)
        direct = left + right
        d_left = lognormal_spectral_power(left.square(), 1.25).value.sqrt()
        d_right = lognormal_spectral_power(right.square(), 1.25).value.sqrt()
        d_direct = lognormal_spectral_power(direct.square(), 1.25).value.sqrt()
        worst = max(worst, float(d_direct - d_left - d_right))
    assert worst <= 2e-12


def test_gradient_is_rank_assigned_weight_away_from_ties() -> None:
    h = torch.tensor([0.7, 0.02, 4.0, 1.1, 0.3], dtype=DTYPE, requires_grad=True)
    result = lognormal_spectral_power(h, 1.25)
    result.value.backward()
    assert torch.equal(h.grad, result.weights)


def test_tie_is_finite_valid_deterministic_subgradient() -> None:
    h = torch.tensor([1.0, 3.0, 3.0, 0.0], dtype=DTYPE, requires_grad=True)
    result = lognormal_spectral_power(h, 1.5)
    result.value.backward()
    assert bool(torch.isfinite(result.value))
    assert bool(torch.isfinite(h.grad).all())
    assert torch.equal(h.grad, result.weights)
    assert float(result.weights.sum()) == pytest.approx(1.0, abs=2e-15)


def test_sigma_zero_reproduces_existing_normalized_spdhsw_loss() -> None:
    source, target = _symmetric(9, 5, 101), _symmetric(7, 5, 202)
    basis = SvecBasis(5, "cpu", DTYPE)
    bank = Bank(basis, bottleneck=4, final_slices=19, seed=6398, rule="r0", persist=0)
    source_vec, target_vec = basis.forward(source), basis.forward(target)
    z_source, z_target = source_vec @ bank.vec.T, target_vec @ bank.vec.T
    scale = bank.scales()
    projected_source = (z_source @ bank.psi.T) / scale[None, :]
    projected_target = (z_target @ bank.psi.T) / scale[None, :]
    h = w2_squared_per_direction(projected_source.T, projected_target.T)
    existing = h.mean()
    spectral = lognormal_spectral_power(h, 0.0)
    assert torch.equal(spectral.value, existing)


def test_two_stage_projection_matches_materialized_normalized_direction() -> None:
    source = _symmetric(11, 6, 303)
    basis = SvecBasis(6, "cpu", DTYPE)
    source_vec = basis.forward(source)
    bank = Bank(basis, bottleneck=5, final_slices=23, seed=3654, rule="r0", persist=0)
    scale = bank.scales()
    cheap = (source_vec @ bank.vec.T @ bank.psi.T) / scale[None, :]
    effective = (bank.psi @ bank.vec) / scale[:, None]
    direct = source_vec @ effective.T
    assert torch.allclose(cheap, direct, atol=3e-15, rtol=3e-15)
    assert torch.allclose(effective.norm(dim=1), torch.ones(23, dtype=DTYPE), atol=3e-15, rtol=3e-15)


@pytest.mark.parametrize("scale", [0.125, 3.0, 100.0])
def test_assigned_weights_are_scale_invariant(scale: float) -> None:
    h = torch.tensor([0.1, 1.3, 0.5, 4.2, 2.0], dtype=DTYPE)
    original = lognormal_spectral_power(h, 1.25)
    dilated = lognormal_spectral_power(scale * h, 1.25)
    assert torch.equal(original.weights, dilated.weights)


def test_registered_maximum_is_finite_at_L2000() -> None:
    h = torch.linspace(0.0, 1e12, 2000, dtype=DTYPE, requires_grad=True)
    result = lognormal_spectral_power(h, 1.5)
    result.value.backward()
    assert bool(torch.isfinite(result.value))
    assert bool(torch.isfinite(result.weights).all())
    assert bool(torch.isfinite(h.grad).all())
    assert math.isfinite(result.entropy)
    assert math.isfinite(result.ess)


@pytest.mark.skipif(torch.cuda.device_count() <= 3, reason="physical GPU 3 unavailable")
def test_registered_maximum_weights_on_physical_gpu3() -> None:
    weights = lognormal_spectral_weights(2000, 1.5, "cuda:3", DTYPE)
    assert weights.device == torch.device("cuda:3")
    assert bool(torch.isfinite(weights).all())
    assert float(weights.sum()) == pytest.approx(1.0, abs=3e-15)
