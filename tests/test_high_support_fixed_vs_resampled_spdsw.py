from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest
import torch

from coherent_slicing import lognormal_spectral_weights
from experiments import run_high_support_fixed_vs_resampled_spdsw as experiment


DTYPE = torch.float64


def _basis_and_target(d: int = 5) -> tuple[object, torch.Tensor]:
    basis = experiment.SvecBasis(d, "cpu", DTYPE)
    generator = torch.Generator().manual_seed(20260903)
    target = torch.randn(7, basis.m, generator=generator, dtype=DTYPE)
    return basis, target


def _synthetic_calibration(auc_2000: float, auc_5000: float, final_2000: float, final_5000: float) -> pd.DataFrame:
    rows = []
    for N_proj in experiment.HGD_CALIBRATION_N_PROJ:
        for sampling in ("fixed", "resampled"):
            rows.append({
                "N_proj": N_proj,
                "sampling": sampling,
                "relative_lew_auc": (
                    auc_2000 if N_proj == 2000 else auc_5000 if N_proj == 5000 else 0.9
                ),
                "lew_final": (
                    final_2000 if N_proj == 2000 else final_5000 if N_proj == 5000 else 80.0
                ),
            })
    return pd.DataFrame(rows)


def test_all_four_factorial_methods_use_identical_N_proj() -> None:
    for dataset, count in (("BNCI2014_001", 500), ("Schirrmeister2017", 2000), ("Schirrmeister2017", 5000)):
        methods = experiment.factorial_methods(dataset, count)
        assert len(methods) == 4
        assert {method.N_proj for method in methods} == {count}
        assert {method.sampling for method in methods} == {"fixed", "resampled"}
        assert {method.aggregation for method in methods} == {"uniform", "spectral"}


def test_bnci_N_proj_is_exactly_500() -> None:
    assert experiment.BNCI_N_PROJ == 500
    config = experiment.CONFIG_TEMPLATE["datasets"]["BNCI2014_001"]
    assert config["N_proj"] == 500


def test_hgd_final_N_proj_is_restricted_to_2000_or_5000() -> None:
    assert experiment.HGD_ALLOWED_FINAL_N_PROJ == (2000, 5000)
    passing = _synthetic_calibration(1.01, 1.0, 101.0, 100.0)
    failing_auc = _synthetic_calibration(1.0100001, 1.0, 100.0, 100.0)
    failing_final = _synthetic_calibration(1.0, 1.0, 101.00001, 100.0)
    assert experiment.select_hgd_N_proj(passing)["selected_N_proj_HGD"] == 2000
    assert experiment.select_hgd_N_proj(failing_auc)["selected_N_proj_HGD"] == 5000
    assert experiment.select_hgd_N_proj(failing_final)["selected_N_proj_HGD"] == 5000


def test_hgd_calibration_counts_are_exactly_registered_set() -> None:
    assert experiment.HGD_CALIBRATION_N_PROJ == (500, 1000, 2000, 5000)
    methods = [
        method for count in experiment.HGD_CALIBRATION_N_PROJ
        for method in experiment.calibration_methods(count)
    ]
    assert len(methods) == 8
    assert {method.N_proj for method in methods} == {500, 1000, 2000, 5000}
    assert all(method.aggregation == "uniform" for method in methods)


def test_calibration_selection_rule_is_deterministic() -> None:
    frame = _synthetic_calibration(0.909, 0.90, 70.7, 70.0)
    first = experiment.select_hgd_N_proj(frame)
    second = experiment.select_hgd_N_proj(frame.sample(frac=1.0, random_state=12))
    assert first == second
    assert first["selected_N_proj_HGD"] == 2000


def test_fixed_bank_is_bitwise_identical_across_epochs() -> None:
    basis, target = _basis_and_target()
    state = experiment.build_fixed_bank_state(basis, target, 6398, 500)
    method = experiment.Method("fixed_uniform", "fixed", "uniform", 500)
    first = experiment.epoch_bank(method, basis, target, 6398, 0, state)
    last = experiment.epoch_bank(method, basis, target, 6398, 499, state)
    assert first[0].data_ptr() == last[0].data_ptr() == state.directions.data_ptr()
    assert torch.equal(first[0], last[0])
    assert first[3] == last[3] == state.bank_hash


def test_resampled_bank_changes_deterministically_across_epochs() -> None:
    basis, _ = _basis_and_target()
    method = experiment.Method("resampled_uniform", "resampled", "uniform", 500)
    first = experiment.sample_frobenius_directions(
        method.N_proj, basis, experiment.method_bank_seed(method, 3654, 1)
    )
    repeated = experiment.sample_frobenius_directions(
        method.N_proj, basis, experiment.method_bank_seed(method, 3654, 1)
    )
    next_epoch = experiment.sample_frobenius_directions(
        method.N_proj, basis, experiment.method_bank_seed(method, 3654, 2)
    )
    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_epoch)


@pytest.mark.parametrize("sampling,indices", [("fixed", (0, 2)), ("resampled", (1, 3))])
def test_uniform_and_spectral_share_registered_banks(sampling: str, indices: tuple[int, int]) -> None:
    basis, _ = _basis_and_target()
    methods = experiment.factorial_methods("BNCI2014_001", 500)
    left, right = (methods[index] for index in indices)
    assert left.sampling == right.sampling == sampling
    for epoch in (0, 19, 499):
        left_seed = experiment.method_bank_seed(left, 1788, epoch)
        right_seed = experiment.method_bank_seed(right, 1788, epoch)
        assert left_seed == right_seed
        assert torch.equal(
            experiment.sample_frobenius_directions(500, basis, left_seed),
            experiment.sample_frobenius_directions(500, basis, right_seed),
        )


def test_fixed_target_projection_is_cached_and_reused() -> None:
    basis, target = _basis_and_target()
    state = experiment.build_fixed_bank_state(basis, target, 1788, 500)
    method = experiment.Method("fixed_uniform", "fixed", "uniform", 500)
    first = experiment.epoch_bank(method, basis, target, 1788, 0, state)
    last = experiment.epoch_bank(method, basis, target, 1788, 499, state)
    assert first[1].data_ptr() == last[1].data_ptr() == state.target_projection.data_ptr()
    assert first[5] == last[5] == state.target_projection_hash
    assert first[6:] == (0.0, 0.0)
    assert last[6:] == (0.0, 0.0)


def test_resampled_target_projection_is_recomputed() -> None:
    basis, target = _basis_and_target()
    method = experiment.Method("resampled_uniform", "resampled", "uniform", 500)
    first = experiment.epoch_bank(method, basis, target, 1788, 0, None)
    second = experiment.epoch_bank(method, basis, target, 1788, 1, None)
    assert first[1].data_ptr() != second[1].data_ptr()
    assert first[5] != second[5]
    assert first[7] > 0.0 and second[7] > 0.0


def test_no_hierarchical_code_path_is_invoked() -> None:
    source = inspect.getsource(experiment)
    assert "from evobank.bank" not in source
    assert "import evobank.bank" not in source
    assert experiment.CONFIG_TEMPLATE["hierarchical_methods"] is False
    assert all(
        method.sampling in {"fixed", "resampled"}
        for method in experiment.factorial_methods("Schirrmeister2017", 2000)
    )


def test_sigma_is_exactly_point_five_and_no_search_exists() -> None:
    methods = experiment.factorial_methods("Schirrmeister2017", 2000)
    assert {method.sigma for method in methods if method.aggregation == "spectral"} == {0.5}
    assert experiment.SIGMA == 0.5
    assert experiment.CONFIG_TEMPLATE["factorial"]["sigma_search"] is False


@pytest.mark.parametrize("count", [500, 2000, 5000])
def test_spectral_weights_sum_to_one_and_are_rank_monotone(count: int) -> None:
    weights = lognormal_spectral_weights(count, 0.5, "cpu", DTYPE)
    assert float(weights.sum()) == pytest.approx(1.0, abs=3e-15)
    assert bool((weights[1:] >= weights[:-1]).all())


def test_sorting_and_rank_assignment_are_detached() -> None:
    h = torch.tensor([0.7, 0.02, 4.0, 1.1], dtype=DTYPE, requires_grad=True)
    method = experiment.Method("fixed_spectral", "fixed", "spectral", 4, 0.5)
    ordered = lognormal_spectral_weights(4, 0.5, "cpu", DTYPE)
    value, assigned = experiment.aggregate_directional_costs(h, method, ordered)
    assert not assigned.requires_grad
    value.backward()
    assert torch.equal(h.grad, assigned)


@pytest.mark.parametrize("seed", experiment.SEEDS)
def test_normalized_applied_update_norm_matches_common_target(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    gradient = torch.randn(11, 17, generator=generator, dtype=DTYPE)
    for target in (experiment.HGD_NORMALIZED_STEP, 3.25):
        update = experiment.normalized_update(gradient, target)
        assert float(update.norm()) == pytest.approx(target, rel=2e-15)


def test_independent_lew_evaluator_has_no_training_bank() -> None:
    basis = experiment.SvecBasis(3, "cpu", DTYPE)
    generator = torch.Generator().manual_seed(99)
    parameter = torch.randn(5, basis.m, generator=generator, dtype=DTYPE)
    target_vec = torch.randn(4, basis.m, generator=generator, dtype=DTYPE)
    evaluator = experiment.LEWEvaluator(basis.inverse(target_vec))
    first, _ = experiment.evaluate_independent_lew(evaluator, basis, parameter)
    second, _ = experiment.evaluate_independent_lew(evaluator, basis, parameter)
    assert first == second
    assert "direction" not in inspect.signature(experiment.evaluate_independent_lew).parameters


def test_timing_excludes_independent_lew_evaluation() -> None:
    clocks = experiment.RunClocks()
    stages = experiment.StageTimes(1, 2, 3, 4, 5, 6, 7)
    clocks.add_optimization(stages.total_epoch_ms())
    before = clocks.optimization_ms
    clocks.add_evaluation(10_000)
    assert before == 28.0
    assert clocks.optimization_ms == before
    assert clocks.evaluation_ms == 10_000.0


def test_epoch_csv_reader_uses_explicit_schema_without_inference(tmp_path) -> None:
    path = tmp_path / "epoch.csv"
    path.write_text(
        "dataset,epoch,lew,nan,diverged,status,bank_hash\n"
        "HGD,0,84.5,False,False,initial,abc\n"
        "HGD,1,,False,False,ok,def\n"
    )
    frame = experiment.read_typed_csv(path)
    assert frame.dataset.tolist() == ["HGD", "HGD"]
    assert frame.epoch.tolist() == [0, 1]
    assert frame.lew.iloc[0] == 84.5 and np.isnan(frame.lew.iloc[1])
    assert frame["nan"].tolist() == [False, False]


def test_frozen_branch_heads_are_unchanged() -> None:
    assert {
        branch: experiment.branch_sha(branch)
        for branch in experiment.FROZEN_BRANCHES
    } == experiment.FROZEN_BRANCHES


def test_direct_projection_count_uses_N_proj_terminology() -> None:
    fields = set(experiment.Method.__dataclass_fields__)
    assert "N_proj" in fields
    assert "k" not in fields
    assert "N_proj" in experiment.EPOCH_COLUMNS
    assert "k" not in experiment.EPOCH_COLUMNS


def test_registered_trajectory_counts() -> None:
    counts = experiment.CONFIG_TEMPLATE["registered_trajectory_counts"]
    assert counts["HGD_support_calibration"] == 4 * 2 * 3 * 1
    assert counts["BNCI_per_control"] == 4 * 3 * 3
    assert counts["HGD_development_per_control"] == 4 * 3 * 3


def test_rank_transition_diagnostics_include_top5_and_top10() -> None:
    ranks = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    previous = torch.tensor([2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12])
    ever5 = torch.zeros(12, dtype=torch.bool)
    ever10 = torch.zeros(12, dtype=torch.bool)
    values = experiment.rank_transition_diagnostics(ranks, previous, ever5, ever10)
    assert math.isfinite(values[0])
    assert values[1] == 0.8
    assert values[2] == 1.0
    assert values[3] == 5 / 12
    assert values[4] == 10 / 12
