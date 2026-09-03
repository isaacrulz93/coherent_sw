from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch

from coherent_slicing import lognormal_spectral_weights
from experiments import run_spectral_sampling_update_factorial as experiment


DTYPE = torch.float64


def _basis_and_target(d: int = 5) -> tuple[object, torch.Tensor]:
    basis = experiment.SvecBasis(d, "cpu", DTYPE)
    generator = torch.Generator().manual_seed(20260903)
    target = torch.randn(7, basis.m, generator=generator, dtype=DTYPE)
    return basis, target


def test_exactly_twelve_registered_methods() -> None:
    assert len(experiment.METHODS) == 12
    assert len({method.name for method in experiment.METHODS}) == 12
    assert {
        (method.sampling, method.aggregation, method.update)
        for method in experiment.METHODS
    } == {
        (sampling, aggregation, update)
        for sampling in ("fixed", "resampled")
        for aggregation in ("uniform", "spectral")
        for update in ("normalized_power", "raw_power", "raw_rooted")
    }


def test_N_proj_is_exactly_500_for_every_method() -> None:
    assert experiment.N_PROJ == 500
    assert {method.N_proj for method in experiment.METHODS} == {500}
    assert experiment.CONFIG_TEMPLATE["N_proj"] == 500


def test_sigma_is_exactly_one_without_sweep() -> None:
    assert experiment.SIGMA == 1.0
    assert {method.sigma for method in experiment.METHODS if method.aggregation == "spectral"} == {1.0}
    assert experiment.CONFIG_TEMPLATE["spectral"]["sigma_sweep"] is False


def test_fixed_six_methods_share_one_bank_and_it_never_changes() -> None:
    basis, target = _basis_and_target()
    state = experiment.build_fixed_bank_state(basis, target, 6398)
    fixed = [method for method in experiment.METHODS if method.sampling == "fixed"]
    assert len(fixed) == 6
    banks = [experiment.epoch_bank(method, basis, target, 6398, 499, state) for method in fixed]
    assert {item[0].data_ptr() for item in banks} == {state.directions.data_ptr()}
    assert {item[3] for item in banks} == {state.bank_hash}
    first = experiment.epoch_bank(fixed[0], basis, target, 6398, 0, state)
    assert torch.equal(first[0], banks[0][0])


def test_resampled_six_share_sequence_and_change_after_epoch_zero() -> None:
    basis, _ = _basis_and_target()
    methods = [method for method in experiment.METHODS if method.sampling == "resampled"]
    assert len(methods) == 6
    for epoch in (0, 1, 17, 499):
        assert {experiment.method_bank_seed(method, 3654, epoch) for method in methods} == {
            experiment.direction_seed(3654, epoch)
        }
    first = experiment.sample_frobenius_directions(
        500, basis, experiment.method_bank_seed(methods[0], 3654, 0)
    )
    repeat = experiment.sample_frobenius_directions(
        500, basis, experiment.method_bank_seed(methods[-1], 3654, 0)
    )
    second = experiment.sample_frobenius_directions(
        500, basis, experiment.method_bank_seed(methods[0], 3654, 1)
    )
    assert torch.equal(first, repeat)
    assert not torch.equal(first, second)


def test_fixed_and_resampled_share_epoch_zero_bank() -> None:
    basis, target = _basis_and_target()
    fixed = experiment.METHODS[0]
    resampled = experiment.METHODS[6]
    assert experiment.method_bank_seed(fixed, 1788, 0) == experiment.method_bank_seed(resampled, 1788, 0)
    fixed_bank = experiment.sample_frobenius_directions(500, basis, experiment.method_bank_seed(fixed, 1788, 0))
    resampled_bank = experiment.sample_frobenius_directions(500, basis, experiment.method_bank_seed(resampled, 1788, 0))
    assert torch.equal(fixed_bank, resampled_bank)
    state = experiment.build_fixed_bank_state(basis, target, 1788)
    fixed_epoch = experiment.epoch_bank(fixed, basis, target, 1788, 0, state)
    resampled_epoch = experiment.epoch_bank(resampled, basis, target, 1788, 0, None)
    assert fixed_epoch[3] == resampled_epoch[3]
    assert fixed_epoch[4] == resampled_epoch[4] == "full_tensor_sha256"


def test_uniform_spectral_and_all_update_forms_share_direction_seed() -> None:
    for sampling in ("fixed", "resampled"):
        methods = [method for method in experiment.METHODS if method.sampling == sampling]
        for epoch in (0, 25, 500):
            assert len({experiment.method_bank_seed(method, 6398, epoch) for method in methods}) == 1


def test_normalized_power_uses_grad_F_and_frozen_eta_norm() -> None:
    x = torch.tensor([3.0, 4.0], dtype=DTYPE)
    update = experiment.raw_update(x, "normalized_power", eta_root=123.0)
    assert torch.allclose(update, -experiment.ETA_NORM * x / x.norm())
    assert experiment.objective_for_update(torch.tensor(4.0), "normalized_power").item() == 4.0
    assert experiment.ETA_NORM == pytest.approx(2.793683898093503, rel=0, abs=0)


def test_raw_power_update_is_exactly_minus_3000_grad_F() -> None:
    gradient = torch.tensor([[1.0, -2.0], [0.25, 4.0]], dtype=DTYPE)
    update = experiment.raw_update(gradient, "raw_power", eta_root=77.0)
    assert torch.equal(update, -3000.0 * gradient)


def test_raw_rooted_differentiates_sqrt_F() -> None:
    x = torch.tensor(3.0, dtype=DTYPE, requires_grad=True)
    power = x.square() + 7.0
    objective = experiment.objective_for_update(power, "raw_rooted")
    gradient = torch.autograd.grad(objective, x)[0]
    assert objective.item() == pytest.approx(4.0)
    assert gradient.item() == pytest.approx(0.75)


def test_one_common_root_eta_calibration_formula_uses_three_subject_medians() -> None:
    eta = experiment.rooted_step_from_norms([3.0, 9.0, 6.0], [2.0, 8.0, 4.0])
    assert eta == pytest.approx(1.5)
    assert experiment.CONFIG_TEMPLATE["rooted"]["calibration_aggregation"] == "uniform only"
    assert experiment.CONFIG_TEMPLATE["rooted"]["calibration_seed"] == 6398


def test_eta_root_is_single_frozen_value_for_all_rooted_methods(tmp_path, monkeypatch) -> None:
    rooted = [method for method in experiment.METHODS if method.update == "raw_rooted"]
    assert len(rooted) == 4
    assert experiment.CONFIG_TEMPLATE["updates"]["raw_rooted_eta_source"] == "ROOTED_STEP_CALIBRATION.json"
    gradient = torch.tensor([1.0, 2.0], dtype=DTYPE)
    assert all(torch.equal(experiment.raw_update(gradient, method.update, 17.0), -17.0 * gradient) for method in rooted)


def test_spectral_weights_are_detached_sum_to_one_and_monotone() -> None:
    ordered = lognormal_spectral_weights(500, 1.0, "cpu", DTYPE)
    assert float(ordered.sum()) == pytest.approx(1.0, abs=4e-15)
    assert bool((ordered[1:] >= ordered[:-1]).all())
    h = torch.tensor([0.7, 0.02, 4.0, 1.1], dtype=DTYPE, requires_grad=True)
    small = lognormal_spectral_weights(4, 1.0, "cpu", DTYPE)
    value, assigned = experiment.aggregate_directional_costs(h, "spectral", small)
    assert not assigned.requires_grad
    value.backward()
    assert torch.equal(h.grad, assigned)


def test_independent_lew_api_accepts_no_training_direction_bank() -> None:
    assert "direction" not in inspect.signature(experiment.evaluate_independent_lew).parameters
    assert experiment.CONFIG_TEMPLATE["evaluation"]["independent_of_training_banks"] is True
    assert experiment.CONFIG_TEMPLATE["evaluation"]["epochs"] == list(range(501))


def test_exact_threshold_first_hit_has_no_interpolation() -> None:
    values = [1.0, 0.97, 0.92, 0.79, 0.81]
    assert experiment.exact_first_hit(values, 0.80) == 3
    assert experiment.exact_first_hit(values, 0.70) is None
    assert experiment.THRESHOLDS == (0.95, 0.90, 0.80, 0.70, 0.60)


def test_evaluation_time_is_separate_from_optimization_components() -> None:
    stages = experiment.StageTimes(1, 2, 3, 4, 5, 6, 7)
    assert stages.total_epoch_ms() == 28.0
    assert "evaluation_ms" not in experiment.StageTimes.__dataclass_fields__
    assert experiment.CONFIG_TEMPLATE["evaluation"]["excluded_from_optimization_wall_clock"] is True


def test_fixed_target_projection_cache_is_reused_and_epoch_timing_zero() -> None:
    basis, target = _basis_and_target()
    state = experiment.build_fixed_bank_state(basis, target, 1788)
    method = next(method for method in experiment.METHODS if method.sampling == "fixed")
    first = experiment.epoch_bank(method, basis, target, 1788, 0, state)
    last = experiment.epoch_bank(method, basis, target, 1788, 499, state)
    assert first[1].data_ptr() == last[1].data_ptr() == state.target_projection.data_ptr()
    assert first[5] == last[5] == state.target_projection_hash
    assert first[6:] == (0.0, 0.0) and last[6:] == (0.0, 0.0)


def test_resampled_target_projection_is_recomputed() -> None:
    basis, target = _basis_and_target()
    method = next(method for method in experiment.METHODS if method.sampling == "resampled")
    first = experiment.epoch_bank(method, basis, target, 1788, 0, None)
    second = experiment.epoch_bank(method, basis, target, 1788, 1, None)
    assert first[1].data_ptr() != second[1].data_ptr()
    assert first[5] != second[5]
    assert first[7] > 0.0 and second[7] > 0.0


def test_gradient_diagnostic_does_not_mutate_parameter() -> None:
    basis, target = _basis_and_target(d=3)
    generator = torch.Generator().manual_seed(12)
    parameter = torch.randn(6, basis.m, generator=generator, dtype=DTYPE, requires_grad=True)
    directions = experiment.sample_frobenius_directions(20, basis, 99)
    target_projection = target @ directions.T
    ordered = lognormal_spectral_weights(20, 1.0, "cpu", DTYPE)
    before = parameter.detach().clone()
    result = experiment.paired_gradient_diagnostic(parameter, directions, target_projection, "raw_power", ordered)
    assert torch.equal(parameter.detach(), before)
    assert parameter.grad is None
    assert all(math.isfinite(value) for value in result)


def test_no_hierarchy_import_or_code_path() -> None:
    source = inspect.getsource(experiment)
    assert "from evobank.bank" not in source
    assert "import evobank.bank" not in source
    assert experiment.CONFIG_TEMPLATE["hierarchical_methods"] is False
    assert experiment.DATASET == "BNCI2014_001"


def test_registered_run_count_and_no_early_stopping() -> None:
    assert len(experiment.METHODS) * len(experiment.SUBJECTS) * len(experiment.SEEDS) == 108
    assert experiment.EPOCHS == 500
    assert experiment.CONFIG_TEMPLATE["no_early_stopping"] is True


def test_frozen_branch_heads_are_unchanged() -> None:
    assert {
        branch: experiment.branch_sha(branch)
        for branch in experiment.FROZEN_BRANCHES
    } == experiment.FROZEN_BRANCHES


def test_frozen_previous_result_hashes_are_unchanged() -> None:
    assert {
        relative: experiment.tree_sha256(experiment.PROJECT / relative)
        for relative in experiment.FROZEN_RESULT_HASHES
    } == experiment.FROZEN_RESULT_HASHES
    assert {
        relative: experiment.sha256(experiment.PROJECT / relative)
        for relative in experiment.FROZEN_UNTRACKED_FILES
    } == experiment.FROZEN_UNTRACKED_FILES


def test_direct_projection_terminology_and_no_k_field() -> None:
    fields = set(experiment.Method.__dataclass_fields__)
    assert "N_proj" in fields
    assert "k" not in fields
    assert "N_proj" in experiment.EPOCH_COLUMNS
    assert "k" not in experiment.EPOCH_COLUMNS
