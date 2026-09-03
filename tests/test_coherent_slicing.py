from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.optimize import linprog

from coherent_slicing import (
    cvar_power,
    directional_costs,
    ebsw_exp_power,
    entropic_power,
    evar_power,
    power_ebsw_power,
    sample_unit_directions,
    sw_power,
)


DTYPE = torch.float64


def test_evar_endpoints() -> None:
    h = torch.tensor([0.1, 2.0, 2.0, 0.7, 1.2], dtype=DTYPE)
    at_mean = evar_power(h, 0.0)
    assert at_mean.status == "mean"
    assert torch.equal(at_mean.weights, torch.full_like(h, 0.2))
    assert torch.allclose(at_mean.value, h.mean(), atol=0.0, rtol=0.0)

    kappa_max = math.log(h.numel() / 2)
    at_max = evar_power(h, kappa_max)
    expected = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0], dtype=DTYPE)
    assert at_max.status == "max"
    assert torch.equal(at_max.weights, expected)
    assert torch.allclose(at_max.value, h.max(), atol=0.0, rtol=0.0)
    assert at_max.achieved_kl == pytest.approx(kappa_max, abs=2e-15)


def test_evar_scale_equivariance() -> None:
    h = torch.tensor([0.02, 0.15, 0.4, 0.8, 1.7, 3.2], dtype=DTYPE)
    p, c, kappa = 2.0, 3.25, 0.55
    base = evar_power(h, kappa, tol=1e-12, max_iter=160)
    scaled = evar_power((c**p) * h, kappa, tol=1e-12, max_iter=160)
    assert base.status == scaled.status == "solved"
    assert torch.allclose(scaled.value, (c**p) * base.value, rtol=2e-11, atol=2e-12)
    assert torch.allclose(scaled.weights, base.weights, rtol=2e-11, atol=2e-12)
    assert scaled.beta == pytest.approx(base.beta / (c**p), rel=2e-11, abs=2e-12)


@pytest.mark.parametrize("seed", range(8))
def test_cvar_fractional_boundary_matches_linear_program(seed: int) -> None:
    rng = np.random.default_rng(seed)
    count = 7
    alpha = 0.37  # alpha*L = 2.59, deliberately non-integral
    h = torch.tensor(rng.uniform(0.0, 4.0, count), dtype=DTYPE)
    result = cvar_power(h, alpha)
    cap = 1.0 / (alpha * count)
    lp = linprog(
        -h.numpy(),
        A_eq=np.ones((1, count)),
        b_eq=np.ones(1),
        bounds=[(0.0, cap)] * count,
        method="highs",
    )
    assert lp.success
    assert float(result.value) == pytest.approx(-lp.fun, abs=2e-12)
    assert float(result.weights.sum()) == pytest.approx(1.0, abs=2e-15)
    assert float(result.weights.max()) <= cap + 2e-15


def test_evar_danskin_gradient_matches_finite_difference() -> None:
    h = torch.tensor([0.08, 0.31, 0.77, 1.4, 2.9], dtype=DTYPE, requires_grad=True)
    kappa = 0.43
    result = evar_power(h, kappa, tol=1e-13, max_iter=180)
    result.value.backward()
    assert torch.allclose(h.grad, result.weights, atol=0.0, rtol=0.0)

    step = 2e-6
    finite = []
    base = h.detach()
    for index in range(h.numel()):
        direction = torch.zeros_like(base)
        direction[index] = step
        plus = evar_power(base + direction, kappa, tol=1e-13, max_iter=180).value
        minus = evar_power(base - direction, kappa, tol=1e-13, max_iter=180).value
        finite.append(float((plus - minus) / (2 * step)))
    finite_tensor = torch.tensor(finite, dtype=DTYPE)
    assert torch.allclose(finite_tensor, result.weights, rtol=3e-7, atol=3e-8)


def _angular_directions(count: int) -> torch.Tensor:
    # Midpoint rule on projective directions [0, pi); deterministic and antipodal-free.
    angles = (torch.arange(count, dtype=DTYPE) + 0.5) * math.pi / count
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def _exact_counterexample() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Repeated endpoints express Dirac masses while keeping equal empirical counts.
    mu = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]], dtype=DTYPE)
    nu = torch.tensor([[-1.0, -1.0], [0.0, 1.0]], dtype=DTYPE)
    eta = torch.tensor([[1.0, -2.0], [1.0, -2.0]], dtype=DTYPE)
    return mu, nu, eta


def test_exact_p1_gamma1_ebsw_counterexample_in_r2() -> None:
    directions = _angular_directions(1 << 18)
    mu, nu, eta = _exact_counterexample()
    h_mn = directional_costs(mu, nu, directions, p=1)
    h_ne = directional_costs(nu, eta, directions, p=1)
    h_me = directional_costs(mu, eta, directions, p=1)
    d_mn = power_ebsw_power(h_mn, gamma=1)
    d_ne = power_ebsw_power(h_ne, gamma=1)
    d_me = power_ebsw_power(h_me, gamma=1)

    # Analytic angular integrals:
    # direct = 5*pi/4; each leg = 5*(pi+1)/(2*(sqrt(10)+sqrt(5))).
    exact_direct = 5.0 * math.pi / 4.0
    exact_leg = 5.0 * (math.pi + 1.0) / (2.0 * (math.sqrt(10.0) + math.sqrt(5.0)))
    exact_violation = exact_direct - 2.0 * exact_leg
    assert float(d_me) == pytest.approx(exact_direct, abs=8e-11)
    assert float(d_mn) == pytest.approx(exact_leg, abs=8e-11)
    assert float(d_ne) == pytest.approx(exact_leg, abs=8e-11)
    assert float(d_me - d_mn - d_ne) == pytest.approx(exact_violation, abs=2e-10)
    assert exact_violation > 0.09


def test_pure_power_perturbation_triple_four_regime_signs() -> None:
    """Regression signs around the concentration transition of one fixed triple.

    These are the four regimes used by the numerical perturbation audit:
    uniform (gamma=0), weak reweighting, finite violating reweighting, and the
    high-power/max limit.  The same measures and angular grid are used in all
    four cases, so only the pure-power exponent changes.
    """
    directions = _angular_directions(1 << 16)
    mu, nu, eta = _exact_counterexample()
    h = [directional_costs(a, b, directions, p=1) for a, b in ((mu, nu), (nu, eta), (mu, eta))]
    expected_signs = {0.0: -1, 0.25: -1, 1.0: 1, 8.0: 1}
    for gamma, expected_sign in expected_signs.items():
        distances = [power_ebsw_power(cost, gamma=gamma) for cost in h]
        slack = float(distances[2] - distances[0] - distances[1])
        assert math.copysign(1.0, slack) == expected_sign
        assert abs(slack) > 1e-4


@pytest.mark.parametrize("method,parameter", [("evar", 0.7), ("cvar", 0.23)])
def test_common_direction_metric_audit(method: str, parameter: float, record_property) -> None:
    worst_slack = -math.inf
    for seed in range(12):
        generator = torch.Generator().manual_seed(1000 + seed)
        q, count, samples = 5, 127, 9
        directions = sample_unit_directions(count, q, seed=2000 + seed)
        clouds = [torch.randn(samples, q, generator=generator, dtype=DTYPE) for _ in range(3)]
        costs = [
            directional_costs(clouds[a], clouds[b], directions, p=2)
            for a, b in ((0, 1), (1, 2), (0, 2))
        ]
        if method == "evar":
            distances = [evar_power(x, parameter).value.sqrt() for x in costs]
        else:
            distances = [cvar_power(x, parameter).value.sqrt() for x in costs]
        slack = float(distances[2] - distances[0] - distances[1])
        worst_slack = max(worst_slack, slack)
    record_property("worst_triangle_slack", worst_slack)
    assert worst_slack <= 2e-10


def test_independent_direction_control_records_without_theorem_claim(record_property) -> None:
    generator = torch.Generator().manual_seed(9182)
    clouds = [torch.randn(8, 4, generator=generator, dtype=DTYPE) for _ in range(3)]
    rows = []
    for seed in range(20):
        costs = [
            directional_costs(clouds[a], clouds[b], sample_unit_directions(5, 4, seed=seed * 10 + pair), p=2)
            for pair, (a, b) in enumerate(((0, 1), (1, 2), (0, 2)))
        ]
        distances = [evar_power(x, 0.8).value.sqrt() for x in costs]
        rows.append(float(distances[2] - distances[0] - distances[1]))
    # This is deliberately descriptive: independently sampled pairwise banks do
    # not share the realization needed by the metric proof.
    record_property("independent_direction_min_slack", min(rows))
    record_property("independent_direction_max_slack", max(rows))
    assert all(math.isfinite(value) for value in rows)


EDGE_CASES = [
    [0.0, 0.0, 0.0, 0.0],
    [2.0, 2.0, 2.0, 2.0],
    [0.1, 3.0, 3.0, 0.2],
    [0.0, 1e-12, 1.0, 1e6],
]


@pytest.mark.parametrize("values", EDGE_CASES)
def test_all_methods_are_finite_with_finite_gradients(values: list[float]) -> None:
    methods = [
        lambda x: sw_power(x),
        lambda x: ebsw_exp_power(x, beta=20.0, full_gradient=True),
        lambda x: ebsw_exp_power(x, beta=20.0, full_gradient=False),
        lambda x: power_ebsw_power(x, gamma=1.0, full_gradient=True),
        lambda x: power_ebsw_power(x, gamma=1.0, full_gradient=False),
        lambda x: entropic_power(x, beta=20.0),
        lambda x: evar_power(x, kappa=0.5).value,
        lambda x: cvar_power(x, alpha=0.37).value,
    ]
    for method in methods:
        h = torch.tensor(values, dtype=DTYPE, requires_grad=True)
        value = method(h)
        value.backward()
        assert bool(torch.isfinite(value))
        assert h.grad is not None
        assert bool(torch.isfinite(h.grad).all())


def test_entropic_beta_zero_limit() -> None:
    h = torch.tensor([0.2, 0.9, 2.1, 4.0], dtype=DTYPE)
    assert torch.equal(entropic_power(h, 0.0), h.mean())
    assert torch.allclose(entropic_power(h, 1e-7), h.mean(), atol=3e-7, rtol=0.0)
