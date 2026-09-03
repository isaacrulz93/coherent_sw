from __future__ import annotations

import inspect
import math

import pytest
import torch

from coherent_slicing import lognormal_spectral_weights
from experiments import run_fixed_vs_resampled_spectral_spdsw as experiment


DTYPE = torch.float64


def _basis_and_target(d: int = 5) -> tuple[object, torch.Tensor]:
    basis = experiment.SvecBasis(d, "cpu", DTYPE)
    generator = torch.Generator().manual_seed(20260903)
    target = torch.randn(7, basis.m, generator=generator, dtype=DTYPE)
    return basis, target


def test_sigma_zero_spectral_equals_uniform_mean() -> None:
    h = torch.tensor([0.2, 1.1, 0.04, 3.0], dtype=DTYPE)
    method = experiment.Method("test", "fixed", "spectral", k=4, sigma=0.0)
    weights = lognormal_spectral_weights(4, 0.0, "cpu", DTYPE)
    value, assigned = experiment.aggregate_directional_costs(h, method, weights)
    assert torch.equal(value, h.mean())
    assert torch.equal(assigned, torch.full_like(h, 0.25))


def test_fixed_spectral_uses_same_bank_at_all_epochs() -> None:
    basis, _ = _basis_and_target()
    method = experiment.PRIMARY_METHODS[2]
    hashes = []
    for epoch in (0, 1, 25, 499):
        bank = experiment.sample_frobenius_directions(
            method.k, basis, experiment.method_bank_seed(method, 6398, epoch)
        )
        hashes.append(experiment.tensor_sha256(bank))
    assert len(set(hashes)) == 1


def test_resampled_spectral_changes_bank_across_epochs() -> None:
    basis, _ = _basis_and_target()
    method = experiment.PRIMARY_METHODS[3]
    hashes = {
        experiment.tensor_sha256(
            experiment.sample_frobenius_directions(
                method.k, basis, experiment.method_bank_seed(method, 6398, epoch)
            )
        )
        for epoch in (0, 1, 2, 25)
    }
    assert len(hashes) == 4


@pytest.mark.parametrize("sampling,indices", [("fixed", (0, 2)), ("resampled", (1, 3))])
def test_uniform_and_spectral_paired_conditions_share_bank(sampling: str, indices: tuple[int, int]) -> None:
    basis, _ = _basis_and_target()
    left, right = (experiment.PRIMARY_METHODS[index] for index in indices)
    assert left.sampling == right.sampling == sampling
    for epoch in (0, 19, 499):
        left_seed = experiment.method_bank_seed(left, 3654, epoch)
        right_seed = experiment.method_bank_seed(right, 3654, epoch)
        assert left_seed == right_seed
        left_bank = experiment.sample_frobenius_directions(left.k, basis, left_seed)
        right_bank = experiment.sample_frobenius_directions(right.k, basis, right_seed)
        assert torch.equal(left_bank, right_bank)
        assert experiment.tensor_sha256(left_bank) == experiment.tensor_sha256(right_bank)


def test_fixed_target_projection_is_bitwise_reused() -> None:
    basis, target = _basis_and_target()
    state = experiment.build_fixed_bank_state(basis, target, 1788)
    method = experiment.PRIMARY_METHODS[0]
    first = experiment.epoch_bank(method, basis, target, 1788, 0, state)
    last = experiment.epoch_bank(method, basis, target, 1788, 499, state)
    assert first[1].data_ptr() == last[1].data_ptr() == state.target_projection.data_ptr()
    assert torch.equal(first[1], last[1])
    assert first[5] == last[5] == state.target_projection_hash
    assert first[6:] == (0.0, 0.0)
    assert last[6:] == (0.0, 0.0)


def test_all_factorial_methods_use_exactly_k40() -> None:
    assert len(experiment.PRIMARY_METHODS) == 4
    assert {method.k for method in experiment.PRIMARY_METHODS} == {40}
    assert {method.sampling for method in experiment.PRIMARY_METHODS} == {"fixed", "resampled"}
    assert {method.aggregation for method in experiment.PRIMARY_METHODS} == {"uniform", "spectral"}


def test_sorting_rank_assignment_is_detached() -> None:
    h = torch.tensor([0.7, 0.02, 4.0, 1.1], dtype=DTYPE, requires_grad=True)
    method = experiment.Method("test", "fixed", "spectral", k=4, sigma=0.5)
    ordered = lognormal_spectral_weights(4, 0.5, "cpu", DTYPE)
    value, assigned = experiment.aggregate_directional_costs(h, method, ordered)
    assert not assigned.requires_grad
    value.backward()
    assert torch.equal(h.grad, assigned)
    assert h.grad_fn is None


def test_spectral_weights_sum_to_one() -> None:
    weights = lognormal_spectral_weights(40, 0.5, "cpu", DTYPE)
    assert float(weights.sum()) == pytest.approx(1.0, abs=3e-15)


def test_spectral_weights_are_monotone_with_rank() -> None:
    weights = lognormal_spectral_weights(40, 0.5, "cpu", DTYPE)
    assert bool((weights[1:] >= weights[:-1]).all())


@pytest.mark.parametrize("scale", [0.01, 0.25, 3.0, 100.0])
def test_positive_cost_scaling_preserves_rank_weights(scale: float) -> None:
    h = torch.tensor([0.7, 0.02, 4.0, 1.1, 0.3], dtype=DTYPE)
    method = experiment.Method("test", "fixed", "spectral", k=5, sigma=0.5)
    ordered = lognormal_spectral_weights(5, 0.5, "cpu", DTYPE)
    _, original = experiment.aggregate_directional_costs(h, method, ordered)
    _, scaled = experiment.aggregate_directional_costs(scale * h, method, ordered)
    assert torch.equal(original, scaled)


@pytest.mark.parametrize("seed", [6398, 3654, 1788])
def test_normalized_applied_update_norm_matches_common_target(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    gradient = torch.randn(11, 17, generator=generator, dtype=DTYPE)
    update = experiment.normalized_update(gradient, experiment.NORMALIZED_STEP_TARGET)
    assert float(update.norm()) == pytest.approx(experiment.NORMALIZED_STEP_TARGET, rel=2e-15)


def test_exact_lew_evaluation_is_independent_of_training_directions() -> None:
    basis = experiment.SvecBasis(3, "cpu", DTYPE)
    generator = torch.Generator().manual_seed(99)
    parameter = torch.randn(5, basis.m, generator=generator, dtype=DTYPE)
    target_vec = torch.randn(4, basis.m, generator=generator, dtype=DTYPE)
    target = basis.inverse(target_vec)
    evaluator = experiment.LEWEvaluator(target)
    first_bank = experiment.sample_frobenius_directions(40, basis, 1)
    second_bank = experiment.sample_frobenius_directions(40, basis, 2)
    assert not torch.equal(first_bank, second_bank)
    first, _ = experiment.evaluate_independent_lew(evaluator, basis, parameter)
    second, _ = experiment.evaluate_independent_lew(evaluator, basis, parameter)
    assert first == second
    assert "direction" not in inspect.signature(experiment.evaluate_independent_lew).parameters


def test_timing_accounting_excludes_lew_evaluation() -> None:
    clocks = experiment.RunClocks()
    stages = experiment.StageTimes(
        direction_sampling_ms=1.0,
        source_projection_ms=2.0,
        target_projection_ms=3.0,
        wasserstein_1d_ms=4.0,
        sorting_aggregation_ms=5.0,
        backward_ms=6.0,
        optimizer_update_ms=7.0,
    )
    clocks.add_optimization(stages.optimization_total_ms())
    before = clocks.optimization_ms
    clocks.add_evaluation(10_000.0)
    assert before == 28.0
    assert clocks.optimization_ms == before
    assert clocks.evaluation_ms == 10_000.0


def test_common_random_number_hashes_reproduce_across_reruns() -> None:
    basis, _ = _basis_and_target()
    seed = experiment.direction_seed(6398, 314)
    first = experiment.sample_frobenius_directions(40, basis, seed)
    second = experiment.sample_frobenius_directions(40, basis, seed)
    assert torch.equal(first, second)
    assert experiment.tensor_sha256(first) == experiment.tensor_sha256(second)


def test_no_hierarchy_code_path_is_present() -> None:
    source = inspect.getsource(experiment)
    assert "from evobank.bank" not in source
    assert "import evobank.bank" not in source
    assert all(method.sampling in {"fixed", "resampled"} for method in experiment.ALL_METHODS)
    assert all(math.isfinite(method.sigma) for method in experiment.ALL_METHODS)
