from __future__ import annotations

import inspect
import math

import pytest
import torch

from coherent_slicing import lognormal_spectral_weights
from experiments import run_ebsw_response_terminal as experiment


DTYPE = torch.float64


def _synthetic_h(count: int = 500) -> torch.Tensor:
    return torch.linspace(0.001, 2.0, count, dtype=DTYPE).square() + 0.03


def _basis_target_parameter() -> tuple[object, torch.Tensor, torch.Tensor]:
    basis = experiment.SvecBasis(3, "cpu", DTYPE)
    generator = torch.Generator().manual_seed(20260903)
    target = torch.randn(7, basis.m, generator=generator, dtype=DTYPE)
    parameter = torch.randn(8, basis.m, generator=generator, dtype=DTYPE, requires_grad=True)
    return basis, target, parameter


def test_registered_dimensions_subjects_and_seeds_are_exact() -> None:
    assert experiment.N_PROJ == 500
    assert experiment.SUBJECTS == (1, 3, 8)
    assert experiment.SEEDS == (6398, 3654, 1788)
    assert experiment.P == 2


def test_exactly_sixteen_scientific_methods_and_288_runs() -> None:
    assert len(experiment.METHODS) == 16
    assert len({method.name for method in experiment.METHODS}) == 16
    assert experiment.registered_config()["trajectory_count"] == 288
    assert experiment.registered_config()["method_names"] == [method.name for method in experiment.METHODS]


def test_resampled_banks_only_and_method_independent_seed_stream() -> None:
    config = experiment.registered_config()
    assert config["sampling"] == "resampled_every_epoch_only"
    for seed in experiment.SEEDS:
        assert experiment.direction_seed(seed, 0) == seed
        assert experiment.direction_seed(seed, 1) != seed
        assert experiment.direction_seed(seed, 25) == experiment.direction_seed(seed, 25)


def test_common_bank_hash_and_epoch0_h_hash_across_methods() -> None:
    basis, target, parameter = _basis_target_parameter()
    seed = experiment.direction_seed(6398, 0)
    hashes, h_hashes = set(), set()
    for _ in experiment.METHODS:
        directions = experiment.sample_frobenius_directions(500, basis, seed)
        h = experiment.w2_squared_per_direction(
            (parameter @ directions.T).T, (target @ directions.T).T
        )
        hashes.add(experiment.tensor_sha256(directions, full=True))
        h_hashes.add(experiment.tensor_sha256(h, full=True))
    assert len(hashes) == len(h_hashes) == 1


def test_full_stop_values_identical_at_same_h_and_beta() -> None:
    h = _synthetic_h().requires_grad_(True)
    beta = 7.25
    full_weights = experiment.stable_exp_weights(h, beta, detach=False)
    stop_weights = full_weights.detach()
    full = torch.sum(full_weights * h)
    stop = torch.sum(stop_weights * h)
    assert torch.equal(full.detach(), stop.detach())


def test_analytic_full_gradient_matches_autograd_strict_float64() -> None:
    h = torch.tensor([0.03, 0.12, 0.4, 0.95, 2.1, 3.7], dtype=DTYPE, requires_grad=True)
    beta = 1.75
    alpha = experiment.stable_exp_weights(h, beta, detach=False)
    value = torch.sum(alpha * h)
    actual = torch.autograd.grad(value, h)[0]
    expected = alpha.detach() * (1.0 + beta * (h.detach() - value.detach()))
    assert torch.allclose(actual, expected, rtol=2e-15, atol=2e-15)
    assert float(expected.sum()) == pytest.approx(1.0, abs=2e-15)


def test_stop_gradient_is_exactly_alpha() -> None:
    h = torch.tensor([0.03, 0.12, 0.4, 0.95, 2.1, 3.7], dtype=DTYPE, requires_grad=True)
    alpha = experiment.stable_exp_weights(h, 2.5, detach=True)
    value = torch.sum(alpha * h)
    actual = torch.autograd.grad(value, h)[0]
    assert torch.equal(actual, alpha)


def test_effective_coefficients_sum_to_one() -> None:
    coefficients = experiment.effective_coefficients(_synthetic_h(), beta=12.0)
    assert float(coefficients.sum()) == pytest.approx(1.0, abs=2e-14)


def test_ess_at_beta_zero_is_N() -> None:
    h = _synthetic_h()
    weights = experiment.stable_exp_weights(h, 0.0, detach=True)
    assert float(1.0 / weights.square().sum()) == pytest.approx(h.numel(), abs=2e-12)
    assert experiment.ess_rho(weights) == pytest.approx(1.0, abs=3e-15)


@pytest.mark.parametrize("target", [0.25, 0.50, 0.75, experiment.RHO_MATCH])
def test_ess_solver_hits_registered_targets(target: float) -> None:
    solve = experiment.solve_ess_beta(_synthetic_h(), target)
    assert solve.status == "solved"
    assert solve.beta >= 0.0 and math.isfinite(solve.beta)
    assert solve.achieved_rho == pytest.approx(target, abs=experiment.ESS_RHO_TOL)


def test_ess_solver_constant_vector_returns_uniform() -> None:
    solve = experiment.solve_ess_beta(torch.ones(500, dtype=DTYPE), 0.25)
    assert solve == experiment.ESSSolveResult(0.0, 1.0, "constant_uniform", 0, 0)


def test_rho_match_is_exact_sigma_one_spectral_ess_fraction() -> None:
    weights = lognormal_spectral_weights(500, 1.0, "cpu", DTYPE)
    expected = float(1.0 / weights.square().sum() / 500)
    assert experiment.RHO_MATCH == expected
    assert experiment.registered_config()["rho_match"] == expected


def test_beta_scale_calibration_is_declared_before_training() -> None:
    source = inspect.getsource(experiment.main)
    assert "prepare()" in source
    assert "execute_grid" in source
    assert experiment.registered_config()["created_before_scientific_runs"] is True
    assert "1/hbar" in experiment.registered_config()["fixed_beta"]


def test_ess_beta_is_solved_from_detached_costs_and_is_not_differentiated() -> None:
    source = inspect.getsource(experiment.solve_ess_beta)
    assert "h.detach()" in source
    h = _synthetic_h().requires_grad_(True)
    method = experiment.METHOD_BY_NAME["ebsw_full_ess050"]
    ordered = lognormal_spectral_weights(500, 1.0, "cpu", DTYPE)
    result = experiment.method_power(h, method, beta_scale=2.0, ordered_spectral=ordered)
    assert isinstance(result.beta, float)
    gradient = torch.autograd.grad(result.value, h)[0]
    assert bool(torch.isfinite(gradient).all())
    assert "danskin" not in inspect.getsource(experiment).lower()


def test_q_one_lpwp_equals_uniform_sw_exactly() -> None:
    h = _synthetic_h()
    assert torch.equal(experiment.lpwp_power(h, 1.0), h.mean())


def test_lpwp_is_described_as_directional_Wp_field_aggregation() -> None:
    description = experiment.registered_config()["lpwp_description"]
    assert description == "L^(p*q)-aggregation of the directional W_p field"
    assert "standard SW" not in description
    assert {method.q for method in experiment.METHODS if method.family == "lpwp"} == {2.0, 4.0}


def test_frozen_update_values_and_no_raw_power_scientific_rerun() -> None:
    assert experiment.ETA_NORM == pytest.approx(2.793683898093503, rel=0, abs=0)
    assert experiment.ETA_ROOT == pytest.approx(589.107249530589, rel=0, abs=0)
    assert experiment.UPDATES == ("normalized", "raw_rooted")
    assert experiment.registered_config()["raw_power_scientific_rerun"] is False
    gradient = torch.tensor([3.0, 4.0], dtype=DTYPE)
    assert torch.allclose(experiment.applied_update(gradient, "normalized"), -experiment.ETA_NORM * gradient / 5.0)
    assert torch.equal(experiment.applied_update(gradient, "raw_rooted"), -experiment.ETA_ROOT * gradient)


def test_independent_lew_function_has_no_training_direction_argument() -> None:
    assert "direction" not in inspect.signature(experiment.evaluate_independent_lew).parameters
    assert experiment.registered_config()["independent_lew"] is True
    assert experiment.registered_config()["evaluation_epochs"] == list(range(0, 501, 25))


def test_mechanism_copied_updates_do_not_mutate_real_state() -> None:
    basis, target_vec, parameter = _basis_target_parameter()
    target_logs = basis.inverse(target_vec)
    evaluator = experiment.LEWEvaluator(target_logs)
    directions = experiment.sample_frobenius_directions(20, basis, 901)
    projected_target = target_vec @ directions.T
    before = parameter.detach().clone()
    method = experiment.METHOD_BY_NAME["ebsw_full_b1"]
    response, coefficients, one_step = experiment.mechanism_diagnostic(
        parameter, directions, projected_target, 1.0, evaluator, basis,
        method=method, update="raw_rooted", subject=1, seed=6398,
        epoch=0, bank_seed=6398,
        bank_hash=experiment.tensor_sha256(directions, full=True),
    )
    assert torch.equal(parameter.detach(), before)
    assert parameter.grad is None
    assert math.isfinite(response["response_ratio"])
    assert coefficients["effective_coeff_sum"] == pytest.approx(1.0, abs=2e-14)
    assert math.isfinite(one_step["LEW_after_stop_normmatched"])


def test_no_hierarchy_import_or_invocation() -> None:
    source = inspect.getsource(experiment)
    assert "from evobank.bank" not in source
    assert "import evobank.bank" not in source
    assert experiment.registered_config()["hierarchy"] is False
    assert experiment.DATASET == "BNCI2014_001"


def test_spectral_rank_weights_are_fixed_and_rho_audited() -> None:
    h = _synthetic_h().requires_grad_(True)
    ordered = lognormal_spectral_weights(500, 1.0, "cpu", DTYPE)
    assigned = experiment.assigned_spectral_weights(h, ordered)
    assert not assigned.requires_grad
    assert float(assigned.sum()) == pytest.approx(1.0, abs=4e-15)
    assert experiment.ess_rho(assigned) == experiment.RHO_MATCH


def test_frozen_branch_heads_are_unchanged() -> None:
    assert {
        branch: experiment.branch_sha(branch)
        for branch in experiment.FROZEN_BRANCHES
    } == experiment.FROZEN_BRANCHES


def test_prior_frozen_result_hashes_and_abandoned_source_are_unchanged() -> None:
    assert {
        relative: experiment.tree_sha256(experiment.PROJECT / relative)
        for relative in experiment.FROZEN_RESULT_HASHES
    } == experiment.FROZEN_RESULT_HASHES
    assert {
        relative: experiment.sha256(experiment.PROJECT / relative)
        for relative in experiment.FROZEN_UNTRACKED_FILES
    } == experiment.FROZEN_UNTRACKED_FILES
