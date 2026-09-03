#!/usr/bin/env python
"""Terminal EBSW weight-response audit for BNCI SPD/EEG alignment.

Only direct, resampled ambient Frobenius directions are used.  ESS calibration
is a conditional numerical calibration: beta is solved from detached costs and
is held constant while differentiating the EBSW objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from coherent_slicing import lognormal_spectral_weights, spectral_power


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "ebsw_response_terminal_v1"
EXTERNAL = Path("/home/pikachu/EBSPDSW")
sys.path.insert(0, str(EXTERNAL))

from evobank.data import load as load_cached_subject  # noqa: E402
from evobank.lew import LEWEvaluator  # noqa: E402
from evobank.ot1d import w2_squared_per_direction  # noqa: E402
from evobank.svec import SvecBasis  # noqa: E402


BRANCH = "exp/ebsw-response-terminal-v1"
DATASET = "BNCI2014_001"
SUBJECTS = (1, 3, 8)
SEEDS = (6398, 3654, 1788)
EPOCHS = 500
N_PROJ = 500
P = 2
SIGMA = 1.0
ETA_NORM = 2.793683898093503
ETA_ROOT = 589.107249530589
UPDATES = ("normalized", "raw_rooted")
EVAL_EVERY = 25
DIAGNOSTIC_EPOCHS = (0, 25, 50, 100, 200, 300, 400, 500)
ESS_TARGETS = (0.25, 0.50, 0.75)
ESS_RHO_TOL = 1e-10
ESS_MAX_ITER = 120
ESS_MAX_BRACKET_STEPS = 1024
ESS_CONSTANT_RTOL = 1e-14
ESS_CONSTANT_ATOL = 0.0
ROOT_EPSILON = 0.0
DTYPE = torch.float64
DEVICE = torch.device("cuda:2")
PHYSICAL_GPU = 1
ALLOW_COTENANCY = True

FROZEN_BRANCHES = {
    "main": "4edf5dda470c5e525c5feb274462414751348b4b",
    "exp/lognormal-spectral-spdhsw-v1": "b0e5b47e17d45a94f39b2b5ba08fa965a5d3a77c",
    "exp/fixed-vs-resampled-spectral-spdsw-v1": "ec4a68af6107b3522d5e841dd35f708402a6378b",
    "exp/high-support-fixed-vs-resampled-spdsw-v2": "b9f61cc6824e2fc71e925ceb960c241bdaf5dbec",
    "exp/spectral-sampling-update-factorial-v1": "6249fcc883de585b16b1498532a154e8ca468e0c",
    "exp/spectral-raw-optimization-audit-v1": "b9f61cc6824e2fc71e925ceb960c241bdaf5dbec",
    "exp/spectral-update-formulation-audit-v1": "b9f61cc6824e2fc71e925ceb960c241bdaf5dbec",
}
FROZEN_RESULT_HASHES = {
    "results/coherent_sw_overnight": "7d840778efb9f95d6a255961d234a65ce8af55077fa96fa27d8358ce5dd47f48",
    "results/lognormal_spectral_spdhsw_v1": "baa0352a347ca49bc522a6e375755791d464220af5769f7f756d49b2129864e9",
    "results/fixed_vs_resampled_spectral_spdsw_v1": "9eaadb1b5b6136cd418065db22e06cd90387d21ed8529f72ff9a2fbc40e8e451",
    "results/high_support_fixed_vs_resampled_spdsw_v2": "64cc8922267042f98ec4f5ba10344e341a83ffacac6abf4dac45457daa7bdc7f",
    "results/spectral_sampling_update_factorial_v1": "34b124549cdbb0c86c7b3b44b207bdbc180207893616a78ce0330e0338d73e18",
    "results/coherent_sw/euclidean_v1": "73b366fab00db2badf6f3135039cbc51fb60b01b855ddebbe25c92c66a04ed4f",
}
FROZEN_UNTRACKED_FILES = {
    "experiments/run_spectral_raw_optimization_audit.py":
        "f06c9990f6253c019f1dd5112e514e21c0c1f6cfd4d62fdfd0d7d4349b373686",
}


@dataclass(frozen=True)
class Method:
    name: str
    family: str
    response: str = "none"
    calibration: str = "none"
    target_rho: float | None = None
    q: float | None = None


METHODS = (
    Method("sw", "sw"),
    Method("ebsw_full_b1", "ebsw", "full", "b1"),
    Method("ebsw_stop_b1", "ebsw", "stop", "b1"),
    Method("ebsw_full_bscale", "ebsw", "full", "bscale"),
    Method("ebsw_stop_bscale", "ebsw", "stop", "bscale"),
    Method("ebsw_full_ess025", "ebsw", "full", "ess025", 0.25),
    Method("ebsw_stop_ess025", "ebsw", "stop", "ess025", 0.25),
    Method("ebsw_full_ess050", "ebsw", "full", "ess050", 0.50),
    Method("ebsw_stop_ess050", "ebsw", "stop", "ess050", 0.50),
    Method("ebsw_full_ess075", "ebsw", "full", "ess075", 0.75),
    Method("ebsw_stop_ess075", "ebsw", "stop", "ess075", 0.75),
    Method("ebsw_full_essmatch", "ebsw", "full", "essmatch"),
    Method("ebsw_stop_essmatch", "ebsw", "stop", "essmatch"),
    Method("spectral_s1", "spectral"),
    Method("lpwp_q2", "lpwp", q=2.0),
    Method("lpwp_q4", "lpwp", q=4.0),
)
METHOD_BY_NAME = {method.name: method for method in METHODS}
PAIR_KEYS = ("b1", "bscale", "ess025", "ess050", "ess075", "essmatch")
CALIBRATED_PAIR_KEYS = ("bscale", "ess025", "ess050", "ess075", "essmatch")
ESS_PAIR_KEYS = ("ess025", "ess050", "ess075", "essmatch")


class ESSSolveResult(NamedTuple):
    beta: float
    achieved_rho: float
    status: str
    iterations: int
    bracket_steps: int


class PowerResult(NamedTuple):
    value: torch.Tensor
    weights: torch.Tensor
    beta: float
    target_rho: float
    achieved_rho: float
    solver_status: str
    solver_iterations: int
    bracket_steps: int


@dataclass
class StageTimes:
    direction_sampling_ms: float = 0.0
    source_projection_ms: float = 0.0
    target_projection_ms: float = 0.0
    wasserstein_1d_ms: float = 0.0
    aggregation_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_update_ms: float = 0.0

    def total_epoch_ms(self) -> float:
        return float(sum(getattr(self, item.name) for item in fields(self)))


def spectral_rho_match() -> float:
    weights = lognormal_spectral_weights(N_PROJ, SIGMA, "cpu", DTYPE)
    return float((1.0 / weights.square().sum()) / N_PROJ)


RHO_MATCH = spectral_rho_match()


def registered_config() -> dict[str, object]:
    return {
        "version": "ebsw_response_terminal_v1",
        "created_before_scientific_runs": True,
        "terminal_audit": True,
        "branch": BRANCH,
        "dataset": DATASET,
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "N_proj": N_PROJ,
        "p": P,
        "sampling": "resampled_every_epoch_only",
        "method_names": [method.name for method in METHODS],
        "method_count": len(METHODS),
        "update_formulations": list(UPDATES),
        "trajectory_count": len(METHODS) * len(UPDATES) * len(SUBJECTS) * len(SEEDS),
        "eta_norm": ETA_NORM,
        "eta_root": ETA_ROOT,
        "raw_power_scientific_rerun": False,
        "fixed_beta": [1.0, "1/hbar"],
        "ess_targets": list(ESS_TARGETS),
        "rho_match": RHO_MATCH,
        "rho_match_source": "exact sigma-1 lognormal spectral weights at N_proj=500",
        "ess_solver": {
            "dtype": "float64", "lower_bound": 0.0,
            "bracket": "range-scaled initial high then deterministic doubling",
            "rho_tolerance": ESS_RHO_TOL, "max_iterations": ESS_MAX_ITER,
            "max_bracket_steps": ESS_MAX_BRACKET_STEPS,
            "constant_rtol": ESS_CONSTANT_RTOL,
            "constant_atol": ESS_CONSTANT_ATOL,
            "beta_calibration_gradient": "stopped; beta is a Python float solved from h.detach()",
        },
        "spectral_sigma": SIGMA,
        "lpwp_q": [2, 4],
        "lpwp_description": "L^(p*q)-aggregation of the directional W_p field",
        "root_epsilon": ROOT_EPSILON,
        "evaluation_epochs": list(range(0, EPOCHS + 1, EVAL_EVERY)),
        "diagnostic_epochs": list(DIAGNOSTIC_EPOCHS),
        "independent_lew": True,
        "diagnostic_cost_excluded_from_optimization_wall_clock": True,
        "hierarchy": False,
        "early_stopping": False,
        "gate_definitions": {
            "one_step_subject_seed_summary": (
                "mean(delta_LEW_full-delta_LEW_stop_normmatched) over the eight "
                "diagnostic states from each normalized FULL trajectory; negative favors FULL"
            ),
            "material_instability": "method divergence count exceeds paired baseline by >1 or is >=5/9",
            "spectral_beats_sw": "favorable grand mean AUC and >=7/9 paired runs",
        },
        "euclidean_context": {
            "source": "frozen experiments/run_euclidean.py E5",
            "dimension": 2, "samples": 32, "seeds": list(SEEDS), "epochs": 100,
            "train_L": 100, "eval_L": 5000, "eval_every": 10,
            "update": "frozen normalized common-step protocol",
            "terminal_gate": False,
        },
    }


EPOCH_COLUMNS = [
    "dataset", "method", "family", "response", "calibration", "update",
    "subject", "seed", "epoch", "bank_epoch", "epochs", "N_proj", "p",
    "power_objective", "rooted_objective", "lew", "relative_lew",
    "lew_reduction_pct", "raw_gradient_norm", "applied_update_norm",
    "beta_t", "target_rho", "ess", "ess_over_N", "realized_rho_error",
    "weight_entropy", "max_weight", "top5_mass", "top10_mass",
    "solver_status", "solver_iterations", "bracket_steps", "bank_seed",
    "bank_hash", "bank_hash_kind", "h_hash_epoch0", "initial_source_hash",
    "target_hash", "direction_sampling_ms", "source_projection_ms",
    "target_projection_ms", "wasserstein_1d_ms", "aggregation_ms",
    "backward_ms", "optimizer_update_ms", "total_epoch_ms",
    "evaluation_ms", "cumulative_optimization_ms", "cumulative_evaluation_ms",
    "cumulative_diagnostic_ms", "nan", "diverged", "solver_failure", "status",
]

RESPONSE_COLUMNS = [
    "method", "calibration", "update", "subject", "seed", "epoch", "beta_t",
    "target_rho", "ess_over_N", "response_ratio", "cosine_full_stop",
    "cosine_full_vs_sw", "cosine_stop_vs_sw", "full_gradient_norm",
    "stop_gradient_norm", "sw_gradient_norm", "bank_seed", "bank_hash",
    "diagnostic_ms",
]
SIGNED_COLUMNS = [
    "method", "calibration", "update", "subject", "seed", "epoch", "beta_t",
    "effective_coeff_sum", "effective_coeff_l1", "negative_coeff_fraction",
    "negative_coeff_mass", "positive_coeff_mass", "min_effective_coeff",
    "max_effective_coeff", "bank_seed", "bank_hash",
]
ONE_STEP_COLUMNS = [
    "method", "calibration", "update", "subject", "seed", "epoch", "beta_t",
    "LEW_before", "LEW_after_full", "LEW_after_stop",
    "LEW_after_stop_normmatched", "delta_LEW_full", "delta_LEW_stop",
    "delta_LEW_stop_normmatched", "full_minus_stop_delta_LEW",
    "full_minus_normmatched_delta_LEW", "norm_ratio_full_over_stop",
    "bank_seed", "bank_hash", "diagnostic_ms",
]

T = TypeVar("T")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(f"{sha256(path)}  {path.relative_to(root)}\n".encode())
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor, *, full: bool = True) -> str:
    detached = tensor.detach()
    if not full and detached.ndim >= 1 and detached.shape[0] > 3:
        detached = detached[[0, detached.shape[0] // 2, detached.shape[0] - 1]]
    array = detached.to(device="cpu", dtype=torch.float64).contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def to_builtin(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n")


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(to_builtin(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def current_branch() -> str:
    return subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=PROJECT, text=True
    ).strip()


def branch_sha(branch: str) -> str:
    return subprocess.check_output(["git", "rev-parse", branch], cwd=PROJECT, text=True).strip()


def configure_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def check_device() -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= int(DEVICE.index):
        raise RuntimeError("registered CUDA device is unavailable")
    query = subprocess.run(
        ["nvidia-smi", "-i", str(PHYSICAL_GPU),
         "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    processes = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if processes and not ALLOW_COTENANCY:
        raise RuntimeError(f"physical GPU {PHYSICAL_GPU} is contaminated: {processes}")
    physical_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(PHYSICAL_GPU), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip().removeprefix("GPU-")
    torch.cuda.set_device(DEVICE)
    properties = torch.cuda.get_device_properties(DEVICE)
    if str(properties.uuid) != physical_uuid:
        raise RuntimeError("registered physical GPU / torch ordinal mismatch")
    return {
        "physical_gpu": PHYSICAL_GPU, "torch_device": str(DEVICE),
        "name": properties.name, "uuid": physical_uuid,
        "total_memory_bytes": properties.total_memory,
        "compute_processes_before_initialization": processes,
        "cotenant_policy": (
            "recorded and allowed because wall-clock is not a scientific endpoint"
            if ALLOW_COTENANCY else "forbidden"
        ),
    }


def sync(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device | str, operation: Callable[[], T]) -> tuple[T, float]:
    sync(device)
    started = time.perf_counter()
    value = operation()
    sync(device)
    return value, 1000.0 * (time.perf_counter() - started)


def direction_seed(seed: int, epoch_zero_based: int) -> int:
    return int(seed + epoch_zero_based * (epoch_zero_based + 1) // 2)


def sample_frobenius_directions(N_proj: int, basis: SvecBasis, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=basis.device).manual_seed(int(seed))
    gaussian = torch.randn(
        N_proj, basis.d, basis.d, generator=generator,
        device=basis.device, dtype=basis.dtype,
    )
    matrices = gaussian + gaussian.transpose(-1, -2)
    matrices = matrices / matrices.norm(dim=(-1, -2), keepdim=True)
    return basis.forward(matrices)


def stable_exp_weights(h: torch.Tensor, beta: float, *, detach: bool) -> torch.Tensor:
    logits = float(beta) * h
    logits = logits - logits.detach().max()
    weights = torch.softmax(logits, dim=0)
    return weights.detach() if detach else weights


def weight_statistics(weights: torch.Tensor) -> dict[str, float]:
    weights = weights.detach()
    positive = weights > 0
    return {
        "ess": float(1.0 / weights.square().sum()),
        "ess_over_N": float(1.0 / weights.square().sum() / weights.numel()),
        "weight_entropy": float(-(weights[positive] * weights[positive].log()).sum()),
        "max_weight": float(weights.max()),
        "top5_mass": float(torch.topk(weights, min(5, weights.numel())).values.sum()),
        "top10_mass": float(torch.topk(weights, min(10, weights.numel())).values.sum()),
    }


def ess_rho(weights: torch.Tensor) -> float:
    return float(1.0 / weights.square().sum() / weights.numel())


def solve_ess_beta(h: torch.Tensor, target_rho: float) -> ESSSolveResult:
    target_rho = float(target_rho)
    if not (0.0 < target_rho <= 1.0) or not math.isfinite(target_rho):
        return ESSSolveResult(math.nan, math.nan, "FAILED", 0, 0)
    detached = h.detach().to(device="cpu", dtype=torch.float64)
    if detached.ndim != 1 or detached.numel() == 0 or not bool(torch.isfinite(detached).all()):
        return ESSSolveResult(math.nan, math.nan, "FAILED", 0, 0)
    data_range = float(detached.max() - detached.min())
    scale = max(float(detached.abs().max()), torch.finfo(torch.float64).tiny)
    if data_range <= ESS_CONSTANT_ATOL + ESS_CONSTANT_RTOL * scale:
        return ESSSolveResult(0.0, 1.0, "constant_uniform", 0, 0)

    shifted = detached - detached.max()

    def rho_at(beta: float) -> float:
        return ess_rho(torch.softmax(float(beta) * shifted, dim=0))

    if target_rho == 1.0:
        return ESSSolveResult(0.0, 1.0, "solved", 0, 0)
    low = 0.0
    high = 1.0 / max(data_range, torch.finfo(torch.float64).tiny)
    high_rho = rho_at(high)
    bracket_steps = 0
    while high_rho > target_rho:
        low = high
        high *= 2.0
        bracket_steps += 1
        if not math.isfinite(high) or bracket_steps > ESS_MAX_BRACKET_STEPS:
            return ESSSolveResult(math.nan, high_rho, "FAILED", 0, bracket_steps)
        high_rho = rho_at(high)
    beta = high
    achieved = high_rho
    for iteration in range(1, ESS_MAX_ITER + 1):
        beta = 0.5 * (low + high)
        achieved = rho_at(beta)
        if abs(achieved - target_rho) <= ESS_RHO_TOL:
            return ESSSolveResult(beta, achieved, "solved", iteration, bracket_steps)
        if achieved > target_rho:
            low = beta
        else:
            high = beta
    if abs(achieved - target_rho) <= ESS_RHO_TOL:
        return ESSSolveResult(beta, achieved, "solved", ESS_MAX_ITER, bracket_steps)
    return ESSSolveResult(math.nan, achieved, "FAILED", ESS_MAX_ITER, bracket_steps)


def lpwp_power(h: torch.Tensor, q: float) -> torch.Tensor:
    q = float(q)
    if not math.isfinite(q) or q < 1.0:
        raise ValueError("q must be finite and at least one")
    return h.pow(q).mean().pow(1.0 / q)


def lpwp_gradient_weights(h: torch.Tensor, q: float) -> torch.Tensor:
    raw = h.detach().clamp_min(0.0).pow(float(q) - 1.0)
    if float(raw.sum()) == 0.0:
        return torch.full_like(raw, 1.0 / raw.numel())
    return raw / raw.sum()


def assigned_spectral_weights(h: torch.Tensor, ordered: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(h.detach(), stable=True)
    assigned = torch.empty_like(ordered)
    assigned[order] = ordered.detach()
    return assigned.detach()


def resolved_target_rho(method: Method) -> float:
    if method.calibration == "essmatch":
        return RHO_MATCH
    return float(method.target_rho) if method.target_rho is not None else math.nan


def method_power(
    h: torch.Tensor, method: Method, *, beta_scale: float,
    ordered_spectral: torch.Tensor,
) -> PowerResult:
    count = h.numel()
    uniform = torch.full_like(h, 1.0 / count).detach()
    if method.family == "sw":
        return PowerResult(h.mean(), uniform, math.nan, math.nan, 1.0, "not_applicable", 0, 0)
    if method.family == "spectral":
        weights = assigned_spectral_weights(h, ordered_spectral)
        return PowerResult(spectral_power(h, ordered_spectral), weights, math.nan, RHO_MATCH, RHO_MATCH, "not_applicable", 0, 0)
    if method.family == "lpwp":
        assert method.q is not None
        weights = lpwp_gradient_weights(h, method.q)
        return PowerResult(lpwp_power(h, method.q), weights, math.nan, math.nan, ess_rho(weights), "not_applicable", 0, 0)
    if method.family != "ebsw":
        raise ValueError(method.family)
    target_rho = resolved_target_rho(method)
    if method.calibration == "b1":
        beta = 1.0
        status, iterations, brackets = "fixed_beta", 0, 0
    elif method.calibration == "bscale":
        beta = float(beta_scale)
        status, iterations, brackets = "fixed_beta", 0, 0
    else:
        solve = solve_ess_beta(h, target_rho)
        beta, status = solve.beta, solve.status
        iterations, brackets = solve.iterations, solve.bracket_steps
        if status == "FAILED":
            nan = h.sum() * math.nan
            return PowerResult(nan, uniform, math.nan, target_rho, solve.achieved_rho, status, iterations, brackets)
    weights = stable_exp_weights(h, beta, detach=(method.response == "stop"))
    value = torch.sum(weights * h)
    achieved = ess_rho(weights.detach())
    return PowerResult(value, weights.detach(), beta, target_rho, achieved, status, iterations, brackets)


def rooted(power: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(power) if ROOT_EPSILON == 0.0 else torch.sqrt(power + ROOT_EPSILON)


def training_objective(power: torch.Tensor, update: str) -> torch.Tensor:
    if update == "normalized":
        return power
    if update == "raw_rooted":
        return rooted(power)
    raise ValueError(update)


def applied_update(gradient: torch.Tensor, update: str) -> torch.Tensor:
    if update == "normalized":
        norm = gradient.norm()
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            return torch.full_like(gradient, math.nan)
        return -ETA_NORM * gradient / norm
    if update == "raw_rooted":
        return -ETA_ROOT * gradient
    raise ValueError(update)


def evaluate_independent_lew(
    evaluator: LEWEvaluator, basis: SvecBasis, parameter: torch.Tensor
) -> tuple[float, float]:
    started = time.perf_counter()
    value = evaluator(basis.inverse(parameter.detach()))
    return value, 1000.0 * (time.perf_counter() - started)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm()) * float(right.norm())
    return float(torch.sum(left * right)) / denominator if denominator > 0.0 else math.nan


def effective_coefficients(h: torch.Tensor, beta: float) -> torch.Tensor:
    detached = h.detach()
    alpha = stable_exp_weights(detached, beta, detach=True)
    value = torch.sum(alpha * detached)
    return alpha * (1.0 + float(beta) * (detached - value))


def mechanism_diagnostic(
    parameter: torch.Tensor, directions: torch.Tensor, projected_target: torch.Tensor,
    beta: float, evaluator: LEWEvaluator, basis: SvecBasis, *, method: Method,
    update: str, subject: int, seed: int, epoch: int, bank_seed: int,
    bank_hash: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    before_parameter = parameter.detach().clone()
    started = time.perf_counter()
    projected_source = parameter @ directions.T
    h = w2_squared_per_direction(projected_source.T, projected_target.T)
    alpha_full = stable_exp_weights(h, beta, detach=False)
    alpha_stop = alpha_full.detach()
    full_power = torch.sum(alpha_full * h)
    stop_power = torch.sum(alpha_stop * h)
    sw_power = h.mean()
    full_root, stop_root, sw_root = rooted(full_power), rooted(stop_power), rooted(sw_power)
    full_gradient = torch.autograd.grad(full_root, parameter, retain_graph=True)[0]
    stop_gradient = torch.autograd.grad(stop_root, parameter, retain_graph=True)[0]
    sw_gradient = torch.autograd.grad(sw_root, parameter)[0]
    response = full_gradient - stop_gradient
    stop_norm = float(stop_gradient.norm())
    ratio = float(response.norm()) / stop_norm if stop_norm > 0.0 else math.nan
    coefficients = effective_coefficients(h, beta)
    negative = coefficients < 0
    coeff_row = {
        "method": method.name, "calibration": method.calibration, "update": update,
        "subject": subject, "seed": seed, "epoch": epoch, "beta_t": beta,
        "effective_coeff_sum": float(coefficients.sum()),
        "effective_coeff_l1": float(coefficients.abs().sum()),
        "negative_coeff_fraction": float(negative.double().mean()),
        "negative_coeff_mass": float(-coefficients.clamp_max(0.0).sum()),
        "positive_coeff_mass": float(coefficients.clamp_min(0.0).sum()),
        "min_effective_coeff": float(coefficients.min()),
        "max_effective_coeff": float(coefficients.max()),
        "bank_seed": bank_seed, "bank_hash": bank_hash,
    }
    lew_before, _ = evaluate_independent_lew(evaluator, basis, parameter)
    full_update = -ETA_ROOT * full_gradient
    stop_update = -ETA_ROOT * stop_gradient
    norm_ratio = float(full_gradient.norm()) / stop_norm if stop_norm > 0.0 else math.nan
    matched_update = -ETA_ROOT * norm_ratio * stop_gradient if stop_norm > 0.0 else torch.full_like(stop_gradient, math.nan)

    def copied_lew(update_tensor: torch.Tensor) -> float:
        copied = before_parameter + update_tensor.detach()
        return evaluate_independent_lew(evaluator, basis, copied)[0]

    lew_full = copied_lew(full_update)
    lew_stop = copied_lew(stop_update)
    lew_matched = copied_lew(matched_update)
    sync(parameter.device)
    diagnostic_ms = 1000.0 * (time.perf_counter() - started)
    assert torch.equal(parameter.detach(), before_parameter)
    response_row = {
        "method": method.name, "calibration": method.calibration, "update": update,
        "subject": subject, "seed": seed, "epoch": epoch, "beta_t": beta,
        "target_rho": resolved_target_rho(method),
        "ess_over_N": ess_rho(alpha_stop), "response_ratio": ratio,
        "cosine_full_stop": cosine(full_gradient, stop_gradient),
        "cosine_full_vs_sw": cosine(full_gradient, sw_gradient),
        "cosine_stop_vs_sw": cosine(stop_gradient, sw_gradient),
        "full_gradient_norm": float(full_gradient.norm()),
        "stop_gradient_norm": stop_norm, "sw_gradient_norm": float(sw_gradient.norm()),
        "bank_seed": bank_seed, "bank_hash": bank_hash,
        "diagnostic_ms": diagnostic_ms,
    }
    one_step_row = {
        "method": method.name, "calibration": method.calibration, "update": update,
        "subject": subject, "seed": seed, "epoch": epoch, "beta_t": beta,
        "LEW_before": lew_before, "LEW_after_full": lew_full,
        "LEW_after_stop": lew_stop, "LEW_after_stop_normmatched": lew_matched,
        "delta_LEW_full": lew_full - lew_before,
        "delta_LEW_stop": lew_stop - lew_before,
        "delta_LEW_stop_normmatched": lew_matched - lew_before,
        "full_minus_stop_delta_LEW": lew_full - lew_stop,
        "full_minus_normmatched_delta_LEW": lew_full - lew_matched,
        "norm_ratio_full_over_stop": norm_ratio,
        "bank_seed": bank_seed, "bank_hash": bank_hash,
        "diagnostic_ms": diagnostic_ms,
    }
    return response_row, coeff_row, one_step_row


def calibration_payload() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        source, target, meta = load_cached_subject(DATASET, subject, DEVICE)
        basis = SvecBasis(meta["d"], DEVICE, DTYPE)
        parameter = basis.forward(source)
        target_vec = basis.forward(target)
        for seed in SEEDS:
            sampled_seed = direction_seed(seed, 0)
            directions = sample_frobenius_directions(N_PROJ, basis, sampled_seed)
            h = w2_squared_per_direction(
                (parameter @ directions.T).T, (target_vec @ directions.T).T
            )
            rows.append({
                "subject": subject, "seed": seed, "N_proj": N_PROJ,
                "mean_h": float(h.mean()), "bank_seed": sampled_seed,
                "bank_hash": tensor_sha256(directions, full=True),
                "h_hash": tensor_sha256(h, full=True),
            })
    hbar = float(np.mean([row["mean_h"] for row in rows]))
    config = registered_config()
    return {
        "hbar": hbar, "beta_scale": 1.0 / hbar,
        "rho_match": RHO_MATCH,
        "spectral_ess": RHO_MATCH * N_PROJ,
        "configuration_hash": canonical_hash(config),
        "configuration_file_sha256": sha256(OUT / "CONFIG.json"),
        "calibrated_before_comparative_training": True,
        "interpretation": "scale calibration diagnostic, not an optimal beta",
        "source_values": rows,
    }


def beta_scale_value() -> float:
    path = OUT / "BETA_CALIBRATION.json"
    if not path.exists():
        raise RuntimeError("BETA_CALIBRATION.json is missing")
    return float(json.loads(path.read_text())["beta_scale"])


def blank_row(
    method: Method, update: str, subject: int, seed: int, epoch: int,
    *, status: str,
) -> dict[str, object]:
    row = {column: math.nan for column in EPOCH_COLUMNS}
    row.update(
        dataset=DATASET, method=method.name, family=method.family,
        response=method.response, calibration=method.calibration, update=update,
        subject=subject, seed=seed, epoch=epoch, bank_epoch=max(0, epoch - 1),
        epochs=EPOCHS, N_proj=N_PROJ, p=P, nan=(status == "nonfinite"),
        diverged=(status == "nonfinite"), solver_failure=(status == "solver_failure"),
        status=status,
    )
    return row


def run_path(method: Method, update: str, subject: int, seed: int) -> Path:
    return OUT / "runs" / update / method.name / f"seed_{seed}" / f"subject_{subject:02d}.csv"


def mechanism_paths(method: Method, update: str, subject: int, seed: int) -> tuple[Path, Path, Path]:
    stem = Path(update) / method.name / f"seed_{seed}" / f"subject_{subject:02d}.csv"
    return (
        OUT / "mechanism" / "response" / stem,
        OUT / "mechanism" / "coefficients" / stem,
        OUT / "mechanism" / "one_step" / stem,
    )


def read_typed_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    strings = {
        "dataset", "method", "family", "response", "calibration", "update",
        "solver_status", "bank_hash", "bank_hash_kind", "h_hash_epoch0",
        "initial_source_hash", "target_hash", "status",
    }
    booleans = {"nan", "diverged", "solver_failure"}
    for column in frame.columns:
        if column in strings:
            continue
        if column in booleans:
            frame[column] = frame[column].map(
                {"True": True, "False": False, "true": True, "false": False, "": False}
            ).astype(bool)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def train_one(
    method: Method, update: str, source: torch.Tensor, target: torch.Tensor,
    *, subject: int, seed: int, beta_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source = source.to(device=DEVICE, dtype=DTYPE)
    target = target.to(device=DEVICE, dtype=DTYPE)
    basis = SvecBasis(source.shape[-1], DEVICE, DTYPE)
    parameter = basis.forward(source).clone().requires_grad_(True)
    target_vec = basis.forward(target)
    evaluator = LEWEvaluator(target)
    lew0, initial_eval_ms = evaluate_independent_lew(evaluator, basis, parameter)
    evaluator.set_baseline(lew0)
    source_hash = tensor_sha256(parameter, full=True)
    target_hash = tensor_sha256(target_vec, full=True)
    ordered_spectral = lognormal_spectral_weights(N_PROJ, SIGMA, DEVICE, DTYPE).detach()
    cumulative_optimization_ms = 0.0
    cumulative_evaluation_ms = initial_eval_ms
    cumulative_diagnostic_ms = 0.0
    rows: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    one_steps: list[dict[str, object]] = []
    initial = blank_row(method, update, subject, seed, 0, status="initial")
    initial.update(
        lew=lew0, relative_lew=1.0, lew_reduction_pct=0.0,
        applied_update_norm=0.0, evaluation_ms=initial_eval_ms,
        cumulative_optimization_ms=0.0,
        cumulative_evaluation_ms=initial_eval_ms,
        cumulative_diagnostic_ms=0.0, nan=False, diverged=False,
        solver_failure=False, initial_source_hash=source_hash, target_hash=target_hash,
    )
    rows.append(initial)
    final_status = "ok"
    for bank_epoch in range(EPOCHS):
        stages = StageTimes()
        sampled_seed = direction_seed(seed, bank_epoch)
        directions, stages.direction_sampling_ms = timed(
            DEVICE, lambda: sample_frobenius_directions(N_PROJ, basis, sampled_seed)
        )
        fingerprint = tensor_sha256(directions, full=(bank_epoch == 0))
        fingerprint_kind = "full_tensor_sha256" if bank_epoch == 0 else "three_row_sha256"
        projected_target, stages.target_projection_ms = timed(
            DEVICE, lambda: target_vec @ directions.T
        )
        projected_source, stages.source_projection_ms = timed(
            DEVICE, lambda: parameter @ directions.T
        )
        h, stages.wasserstein_1d_ms = timed(
            DEVICE, lambda: w2_squared_per_direction(projected_source.T, projected_target.T)
        )
        result, stages.aggregation_ms = timed(
            DEVICE,
            lambda: method_power(
                h, method, beta_scale=beta_scale, ordered_spectral=ordered_spectral
            ),
        )
        if result.solver_status == "FAILED":
            final_status = "solver_failure"
            rows.extend(
                blank_row(method, update, subject, seed, later, status="solver_failure")
                for later in range(bank_epoch + 1, EPOCHS + 1)
            )
            break
        stats = weight_statistics(result.weights)
        if bank_epoch == 0:
            rows[0].update(
                power_objective=float(result.value.detach()),
                rooted_objective=float(rooted(result.value.detach())),
                beta_t=result.beta, target_rho=result.target_rho,
                **stats, realized_rho_error=(
                    result.achieved_rho - result.target_rho
                    if math.isfinite(result.target_rho) else math.nan
                ),
                solver_status=result.solver_status,
                solver_iterations=result.solver_iterations,
                bracket_steps=result.bracket_steps, bank_seed=sampled_seed,
                bank_hash=fingerprint, bank_hash_kind=fingerprint_kind,
                h_hash_epoch0=tensor_sha256(h, full=True),
            )
        if method.family == "ebsw" and method.response == "full" and bank_epoch in DIAGNOSTIC_EPOCHS:
            response_row, coeff_row, one_step_row = mechanism_diagnostic(
                parameter, directions, projected_target, result.beta, evaluator, basis,
                method=method, update=update, subject=subject, seed=seed,
                epoch=bank_epoch, bank_seed=sampled_seed, bank_hash=fingerprint,
            )
            responses.append(response_row)
            coefficients.append(coeff_row)
            one_steps.append(one_step_row)
            cumulative_diagnostic_ms += float(response_row["diagnostic_ms"])
        objective = training_objective(result.value, update)
        gradient, stages.backward_ms = timed(
            DEVICE, lambda: torch.autograd.grad(objective, parameter)[0]
        )
        gradient_norm = float(gradient.norm())

        def update_parameter() -> torch.Tensor:
            update_tensor = applied_update(gradient, update)
            with torch.no_grad():
                parameter.add_(update_tensor)
            return update_tensor

        update_tensor, stages.optimizer_update_ms = timed(DEVICE, update_parameter)
        update_norm = float(update_tensor.norm())
        epoch_ms = stages.total_epoch_ms()
        cumulative_optimization_ms += epoch_ms
        epoch = bank_epoch + 1
        finite = bool(
            torch.isfinite(parameter).all() and torch.isfinite(result.value)
            and math.isfinite(gradient_norm) and math.isfinite(update_norm)
        )
        lew = relative = reduction = math.nan
        evaluation_ms = 0.0
        diverged = not finite
        if finite and epoch % EVAL_EVERY == 0:
            lew, evaluation_ms = evaluate_independent_lew(evaluator, basis, parameter)
            cumulative_evaluation_ms += evaluation_ms
            relative = lew / lew0
            reduction = 100.0 * (lew0 - lew) / lew0
            diverged = evaluator.diverged(lew)
        row = {
            "dataset": DATASET, "method": method.name, "family": method.family,
            "response": method.response, "calibration": method.calibration,
            "update": update, "subject": subject, "seed": seed, "epoch": epoch,
            "bank_epoch": bank_epoch, "epochs": EPOCHS, "N_proj": N_PROJ, "p": P,
            "power_objective": float(result.value.detach()),
            "rooted_objective": float(rooted(result.value.detach())),
            "lew": lew, "relative_lew": relative, "lew_reduction_pct": reduction,
            "raw_gradient_norm": gradient_norm, "applied_update_norm": update_norm,
            "beta_t": result.beta, "target_rho": result.target_rho,
            **stats,
            "realized_rho_error": (
                result.achieved_rho - result.target_rho
                if math.isfinite(result.target_rho) else math.nan
            ),
            "solver_status": result.solver_status,
            "solver_iterations": result.solver_iterations,
            "bracket_steps": result.bracket_steps, "bank_seed": sampled_seed,
            "bank_hash": fingerprint, "bank_hash_kind": fingerprint_kind,
            "h_hash_epoch0": tensor_sha256(h, full=True) if bank_epoch == 0 else "",
            "initial_source_hash": source_hash, "target_hash": target_hash,
            "direction_sampling_ms": stages.direction_sampling_ms,
            "source_projection_ms": stages.source_projection_ms,
            "target_projection_ms": stages.target_projection_ms,
            "wasserstein_1d_ms": stages.wasserstein_1d_ms,
            "aggregation_ms": stages.aggregation_ms, "backward_ms": stages.backward_ms,
            "optimizer_update_ms": stages.optimizer_update_ms,
            "total_epoch_ms": epoch_ms, "evaluation_ms": evaluation_ms,
            "cumulative_optimization_ms": cumulative_optimization_ms,
            "cumulative_evaluation_ms": cumulative_evaluation_ms,
            "cumulative_diagnostic_ms": cumulative_diagnostic_ms,
            "nan": not finite, "diverged": bool(diverged), "solver_failure": False,
            "status": "ok" if finite else "nonfinite",
        }
        rows.append(row)
        if not finite:
            final_status = "nonfinite"
            rows.extend(
                blank_row(method, update, subject, seed, later, status="nonfinite")
                for later in range(epoch + 1, EPOCHS + 1)
            )
            break
    if final_status == "ok" and method.family == "ebsw" and method.response == "full":
        bank_epoch = EPOCHS
        sampled_seed = direction_seed(seed, bank_epoch)
        directions = sample_frobenius_directions(N_PROJ, basis, sampled_seed)
        fingerprint = tensor_sha256(directions, full=False)
        projected_target = target_vec @ directions.T
        h = w2_squared_per_direction((parameter @ directions.T).T, projected_target.T)
        result = method_power(h, method, beta_scale=beta_scale, ordered_spectral=ordered_spectral)
        if result.solver_status == "FAILED":
            final_status = "solver_failure"
            rows[-1]["solver_failure"] = True
            rows[-1]["status"] = "solver_failure_at_final_diagnostic"
        else:
            response_row, coeff_row, one_step_row = mechanism_diagnostic(
                parameter, directions, projected_target, result.beta, evaluator, basis,
                method=method, update=update, subject=subject, seed=seed,
                epoch=EPOCHS, bank_seed=sampled_seed, bank_hash=fingerprint,
            )
            responses.append(response_row)
            coefficients.append(coeff_row)
            one_steps.append(one_step_row)
    frame = pd.DataFrame(rows)[EPOCH_COLUMNS]
    response_frame = pd.DataFrame(responses, columns=RESPONSE_COLUMNS)
    coefficient_frame = pd.DataFrame(coefficients, columns=SIGNED_COLUMNS)
    one_step_frame = pd.DataFrame(one_steps, columns=ONE_STEP_COLUMNS)
    evaluated = frame[np.isfinite(frame.lew)]
    metadata = {
        "method": method.name, "family": method.family, "response": method.response,
        "calibration": method.calibration, "update": update, "subject": subject,
        "seed": seed, "rows": len(frame), "diagnostic_rows": len(response_frame),
        "lew_initial": lew0,
        "lew_final": float(evaluated.lew.iloc[-1]) if not evaluated.empty else math.nan,
        "diverged": bool(frame.diverged.fillna(False).any()),
        "nan": bool(frame["nan"].fillna(False).any()),
        "solver_failure": bool(frame.solver_failure.fillna(False).any()),
        "status": final_status,
    }
    return frame, response_frame, coefficient_frame, one_step_frame, metadata


def run_complete(path: Path, method: Method, response_path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = read_typed_csv(path)
        if len(frame) != EPOCHS + 1 or int(frame.epoch.iloc[-1]) != EPOCHS:
            return False
        if method.family == "ebsw" and method.response == "full" and not bool(frame.solver_failure.any()):
            if not response_path.exists() or len(read_typed_csv(response_path)) != len(DIAGNOSTIC_EPOCHS):
                return False
        return True
    except Exception:
        return False


def metadata_from_frame(path: Path, response_path: Path) -> dict[str, object]:
    frame = read_typed_csv(path)
    evaluated = frame[np.isfinite(frame.lew)]
    return {
        "method": str(frame.method.iloc[0]), "family": str(frame.family.iloc[0]),
        "response": str(frame.response.iloc[0]), "calibration": str(frame.calibration.iloc[0]),
        "update": str(frame["update"].iloc[0]), "subject": int(frame.subject.iloc[0]),
        "seed": int(frame.seed.iloc[0]), "rows": len(frame),
        "diagnostic_rows": len(read_typed_csv(response_path)) if response_path.exists() else 0,
        "lew_initial": float(evaluated.lew.iloc[0]),
        "lew_final": float(evaluated.lew.iloc[-1]),
        "diverged": bool(frame.diverged.fillna(False).any()),
        "nan": bool(frame["nan"].fillna(False).any()),
        "solver_failure": bool(frame.solver_failure.fillna(False).any()),
        "status": "cached_complete",
    }


def execute_grid(*, rerun: bool = False) -> list[dict[str, object]]:
    beta_scale = beta_scale_value()
    records: list[dict[str, object]] = []
    total = len(METHODS) * len(UPDATES) * len(SUBJECTS) * len(SEEDS)
    index = 0
    manifest_path = OUT / "MANIFEST_TRAJECTORIES.json"
    for subject in SUBJECTS:
        source, target, _ = load_cached_subject(DATASET, subject, DEVICE)
        for seed in SEEDS:
            for update in UPDATES:
                for method in METHODS:
                    index += 1
                    path = run_path(method, update, subject, seed)
                    response_path, coefficient_path, one_step_path = mechanism_paths(
                        method, update, subject, seed
                    )
                    try:
                        if rerun or not run_complete(path, method, response_path):
                            frame, responses, coefficients, one_steps, metadata = train_one(
                                method, update, source, target, subject=subject,
                                seed=seed, beta_scale=beta_scale,
                            )
                            path.parent.mkdir(parents=True, exist_ok=True)
                            frame.to_csv(path, index=False)
                            if method.family == "ebsw" and method.response == "full":
                                for output_path, output_frame in (
                                    (response_path, responses),
                                    (coefficient_path, coefficients),
                                    (one_step_path, one_steps),
                                ):
                                    output_path.parent.mkdir(parents=True, exist_ok=True)
                                    output_frame.to_csv(output_path, index=False)
                        else:
                            metadata = metadata_from_frame(path, response_path)
                        record = {
                            **metadata, "run_csv": str(path.relative_to(OUT)),
                            "response_csv": (
                                str(response_path.relative_to(OUT))
                                if method.family == "ebsw" and method.response == "full" else None
                            ),
                            "error": None,
                        }
                        print(
                            f"[{index:03d}/{total:03d}] s{subject:02d} seed={seed} "
                            f"{update:10s} {method.name:24s} "
                            f"LEW {record['lew_initial']:.4f}->{record['lew_final']:.4f} "
                            f"status={record['status']}", flush=True,
                        )
                    except Exception as exc:
                        log_path = OUT / "logs" / f"{update}_{method.name}_seed{seed}_s{subject:02d}.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log_path.write_text(traceback.format_exc())
                        record = {
                            "method": method.name, "family": method.family,
                            "response": method.response, "calibration": method.calibration,
                            "update": update, "subject": subject, "seed": seed,
                            "status": "error", "error": f"{type(exc).__name__}: {exc}",
                            "run_csv": str(path.relative_to(OUT)), "response_csv": None,
                        }
                        print(f"[ERROR] {record['error']}", file=sys.stderr, flush=True)
                    records.append(record)
                    dump_json(manifest_path, records)
        del source, target
        torch.cuda.empty_cache()
    return records


def run_euclidean_context() -> dict[str, object]:
    """Reuse the frozen E5 clouds, direction streams, and normalized step."""
    from coherent_slicing import directional_costs, sample_unit_directions
    from experiments.run_euclidean import flow_initial_clouds, initial_sw_step_norm

    context_methods = (
        METHOD_BY_NAME["sw"], METHOD_BY_NAME["ebsw_full_b1"],
        METHOD_BY_NAME["ebsw_stop_b1"], METHOD_BY_NAME["ebsw_full_essmatch"],
        METHOD_BY_NAME["ebsw_stop_essmatch"],
    )
    epochs, train_count, eval_count, eval_every = 100, 100, 5000, 10
    target_step = float(np.median([
        initial_sw_step_norm(seed, 0.03, train_count) for seed in SEEDS
    ]))
    rows: list[dict[str, object]] = []
    solver_failures = 0
    for seed in SEEDS:
        source, target = flow_initial_clouds(seed)
        eval_directions = sample_unit_directions(eval_count, 2, seed=seed + 4_000_000)
        for method in context_methods:
            parameter = source.clone().requires_grad_(True)
            for epoch in range(epochs + 1):
                if epoch % eval_every == 0:
                    eval_value = float(
                        directional_costs(parameter.detach(), target, eval_directions, p=2).mean().sqrt()
                    )
                    rows.append({
                        "method": method.name, "seed": seed, "epoch": epoch,
                        "independent_sw": eval_value, "target_step_norm": target_step,
                        "rho_match": RHO_MATCH, "solver_failure": False,
                    })
                if epoch == epochs:
                    break
                directions = sample_unit_directions(train_count, 2, seed=seed + epoch * 7919)
                h = directional_costs(parameter, target, directions, p=2)
                if method.family == "sw":
                    power = h.mean()
                else:
                    beta = 1.0
                    if method.calibration == "essmatch":
                        solve = solve_ess_beta(h, RHO_MATCH)
                        if solve.status == "FAILED":
                            solver_failures += 1
                            rows[-1]["solver_failure"] = True
                            break
                        beta = solve.beta
                    weights = stable_exp_weights(h, beta, detach=(method.response == "stop"))
                    power = torch.sum(weights * h)
                gradient = torch.autograd.grad(power, parameter)[0]
                norm = gradient.norm()
                update_tensor = (
                    -target_step * gradient / norm
                    if bool(torch.isfinite(norm)) and float(norm) > 0.0
                    else torch.zeros_like(parameter)
                )
                with torch.no_grad():
                    parameter.add_(update_tensor)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "EUCLIDEAN_CONTEXT.csv", index=False)
    status = {
        "euclidean_context_status": "run_frozen_E5_normalized_protocol",
        "terminal_gate": False, "methods": [method.name for method in context_methods],
        "seeds": list(SEEDS), "epochs": epochs, "train_L": train_count,
        "eval_L": eval_count, "eval_every": eval_every,
        "target_normalized_step": target_step, "solver_failures": solver_failures,
        "invented_distribution": False, "invented_learning_rate": False,
    }
    dump_json(OUT / "EUCLIDEAN_CONTEXT.json", status)
    return status


def relative_auc(group: pd.DataFrame) -> float:
    evaluated = group[np.isfinite(group.lew)].sort_values("epoch")
    if len(evaluated) != EPOCHS // EVAL_EVERY + 1 or int(evaluated.epoch.iloc[-1]) != EPOCHS:
        return math.inf
    return float(np.trapezoid(evaluated.relative_lew, evaluated.epoch) / EPOCHS)


def load_run_frames() -> pd.DataFrame:
    paths = sorted((OUT / "runs").glob("*/*/seed_*/subject_*.csv"))
    if len(paths) != 288:
        raise RuntimeError(f"expected 288 trajectory CSVs, found {len(paths)}")
    return pd.concat([read_typed_csv(path) for path in paths], ignore_index=True)


def load_mechanism(kind: str, columns: list[str]) -> pd.DataFrame:
    paths = sorted((OUT / "mechanism" / kind).glob("*/*/seed_*/subject_*.csv"))
    if len(paths) != 108:
        raise RuntimeError(f"expected 108 {kind} CSVs, found {len(paths)}")
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    strings = {"method", "calibration", "update", "bank_hash"}
    for column in frame.columns:
        if column not in strings:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[columns]


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "family", "response", "calibration", "update", "subject", "seed"]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=False):
        group = group.sort_values("epoch")
        evaluated = group[np.isfinite(group.lew)]
        row = dict(zip(keys, key))
        initial = float(evaluated.lew.iloc[0])
        final = float(evaluated.lew.iloc[-1])
        row.update(
            relative_LEW_AUC=relative_auc(group), LEW_initial=initial,
            LEW_final=final, relative_LEW_final=final / initial,
            LEW_reduction_pct=100.0 * (initial - final) / initial,
            divergence=bool(group.diverged.fillna(False).any()),
            nan=bool(group["nan"].fillna(False).any()),
            solver_failure=bool(group.solver_failure.fillna(False).any()),
            optimization_ms=float(group.cumulative_optimization_ms.dropna().iloc[-1]),
            evaluation_ms=float(group.cumulative_evaluation_ms.dropna().iloc[-1]),
            diagnostic_ms=float(group.cumulative_diagnostic_ms.dropna().iloc[-1]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def pair_names(pair_key: str) -> tuple[str, str]:
    return f"ebsw_full_{pair_key}", f"ebsw_stop_{pair_key}"


def paired_table(summary: pd.DataFrame, comparator: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for update in UPDATES:
        for pair_key in PAIR_KEYS:
            full_name, stop_name = pair_names(pair_key)
            compare_name = stop_name if comparator == "stop" else "sw"
            cell = summary[
                (summary["update"] == update)
                & summary.method.isin([full_name, compare_name])
            ]
            pivot = cell.pivot(index=["subject", "seed"], columns="method", values="relative_LEW_AUC")
            difference = pivot[full_name] - pivot[compare_name]
            full = summary[(summary["update"] == update) & (summary.method == full_name)]
            compare = summary[(summary["update"] == update) & (summary.method == compare_name)]
            rows.append({
                "update": update, "condition": pair_key, "comparator": comparator,
                "mean_AUC_full": float(full.relative_LEW_AUC.mean()),
                f"mean_AUC_{comparator}": float(compare.relative_LEW_AUC.mean()),
                "paired_delta": float(difference.mean()),
                "favorable_runs": int((difference < 0.0).sum()), "total_runs": 9,
                "final_LEW_full": float(full.LEW_final.mean()),
                f"final_LEW_{comparator}": float(compare.LEW_final.mean()),
                "divergence_full": int((full.divergence | full["nan"]).sum()),
                f"divergence_{comparator}": int((compare.divergence | compare["nan"]).sum()),
                "solver_failures_full": int(full.solver_failure.sum()),
            })
    return pd.DataFrame(rows)


def subject_effects(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons: list[tuple[str, str, str]] = []
    for pair_key in PAIR_KEYS:
        full, stop = pair_names(pair_key)
        comparisons.extend([
            (f"full_vs_stop_{pair_key}", full, stop),
            (f"full_vs_sw_{pair_key}", full, "sw"),
        ])
    comparisons.extend([
        ("lpwp_q2_vs_sw", "lpwp_q2", "sw"),
        ("lpwp_q4_vs_sw", "lpwp_q4", "sw"),
        ("spectral_s1_vs_sw", "spectral_s1", "sw"),
        ("lpwp_q2_vs_spectral_s1", "lpwp_q2", "spectral_s1"),
        ("lpwp_q4_vs_spectral_s1", "lpwp_q4", "spectral_s1"),
    ])
    for update in UPDATES:
        for label, left, right in comparisons:
            cell = summary[(summary["update"] == update) & summary.method.isin([left, right])]
            pivot = cell.pivot(index=["subject", "seed"], columns="method", values="relative_LEW_AUC")
            difference = pivot[left] - pivot[right]
            for (subject, seed), value in difference.items():
                rows.append({
                    "comparison": label, "update": update, "level": "run",
                    "subject": subject, "seed": seed, "difference": float(value),
                    "mean": math.nan, "median": math.nan, "sd": math.nan,
                })
            for subject, value in difference.groupby(level="subject").mean().items():
                rows.append({
                    "comparison": label, "update": update, "level": "subject_mean",
                    "subject": subject, "seed": math.nan, "difference": float(value),
                    "mean": math.nan, "median": math.nan, "sd": math.nan,
                })
            rows.append({
                "comparison": label, "update": update, "level": "grand",
                "subject": math.nan, "seed": math.nan, "difference": math.nan,
                "mean": float(difference.mean()), "median": float(difference.median()),
                "sd": float(difference.std(ddof=1)),
            })
    return pd.DataFrame(rows)


def concentration_summary(frame: pd.DataFrame) -> pd.DataFrame:
    trained = frame[(frame.epoch > 0) & ~frame.solver_failure]
    return trained.groupby(["method", "family", "update"], as_index=False).agg(
        mean_ESS_over_N=("ess_over_N", "mean"), median_ESS_over_N=("ess_over_N", "median"),
        mean_entropy=("weight_entropy", "mean"), median_entropy=("weight_entropy", "median"),
        mean_max_weight=("max_weight", "mean"), median_max_weight=("max_weight", "median"),
        mean_top5_mass=("top5_mass", "mean"), median_top5_mass=("top5_mass", "median"),
        mean_top10_mass=("top10_mass", "mean"), median_top10_mass=("top10_mass", "median"),
    )


def bank_audit(frame: pd.DataFrame) -> dict[str, object]:
    trained = frame[(frame.epoch > 0) & ~frame.solver_failure & ~frame["nan"]]
    common_rows = []
    for (subject, seed, bank_epoch), group in trained.groupby(["subject", "seed", "bank_epoch"]):
        common_rows.append({
            "subject": int(subject), "seed": int(seed), "bank_epoch": int(bank_epoch),
            "rows": len(group), "bank_seed_count": int(group.bank_seed.nunique()),
            "bank_hash_count": int(group.bank_hash.nunique()),
        })
    initial = frame[(frame.epoch == 0) & ~frame.solver_failure]
    initial_rows = []
    for (subject, seed), group in initial.groupby(["subject", "seed"]):
        initial_rows.append({
            "subject": int(subject), "seed": int(seed), "rows": len(group),
            "h_hash_count": int(group.h_hash_epoch0.nunique()),
            "source_hash_count": int(group.initial_source_hash.nunique()),
            "target_hash_count": int(group.target_hash.nunique()),
        })
    common_pass = all(
        row["rows"] == len(METHODS) * len(UPDATES)
        and row["bank_seed_count"] == 1 and row["bank_hash_count"] == 1
        for row in common_rows
    )
    initial_pass = all(
        row["rows"] == len(METHODS) * len(UPDATES)
        and row["h_hash_count"] == 1 and row["source_hash_count"] == 1
        and row["target_hash_count"] == 1 for row in initial_rows
    )
    ess = trained[trained.calibration.isin(ESS_PAIR_KEYS)]
    max_ess_error = float(ess.realized_rho_error.abs().max()) if not ess.empty else math.inf
    return {
        "passed": common_pass and initial_pass and max_ess_error <= ESS_RHO_TOL,
        "common_bank_pass": common_pass, "epoch0_h_hash_pass": initial_pass,
        "max_abs_realized_ESS_over_N_error": max_ess_error,
        "ESS_tolerance": ESS_RHO_TOL, "common_bank_rows": common_rows,
        "epoch0_rows": initial_rows,
        "direction_seed_rule": "seed + epoch*(epoch+1)//2",
    }


def tests_pass() -> bool:
    path = OUT / "TEST_RESULTS.xml"
    if not path.exists():
        return False
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return bool(suites) and all(
        int(suite.attrib.get("failures", 0)) == 0
        and int(suite.attrib.get("errors", 0)) == 0 for suite in suites
    )


def gate_results(
    summary: pd.DataFrame, full_stop: pd.DataFrame, full_sw: pd.DataFrame,
    responses: pd.DataFrame, one_step: pd.DataFrame, audit: dict[str, object],
) -> dict[str, object]:
    response_by_condition = responses[
        responses.calibration.isin(ESS_PAIR_KEYS)
    ].groupby("calibration").response_ratio.median()
    active_conditions = response_by_condition[response_by_condition >= 0.10].index.tolist()
    h_active = bool(active_conditions)
    one_normalized = one_step[
        (one_step["update"] == "normalized")
        & one_step.calibration.isin(CALIBRATED_PAIR_KEYS)
    ]
    one_aggregate = one_normalized.groupby(
        ["calibration", "subject", "seed"], as_index=False
    ).full_minus_normmatched_delta_LEW.mean()
    one_counts = one_aggregate.groupby("calibration").full_minus_normmatched_delta_LEW.apply(
        lambda values: int((values < 0.0).sum())
    ).to_dict()
    direction_passes: list[str] = []
    for condition in CALIBRATED_PAIR_KEYS:
        row = full_stop[(full_stop["update"] == "normalized") & (full_stop.condition == condition)].iloc[0]
        if int(row.favorable_runs) >= 7 and float(row.paired_delta) < 0.0 and int(one_counts.get(condition, 0)) >= 7:
            direction_passes.append(condition)
    h_direction = bool(direction_passes)
    complete_passes: list[str] = []
    for condition in CALIBRATED_PAIR_KEYS:
        stop_row = full_stop[(full_stop["update"] == "raw_rooted") & (full_stop.condition == condition)].iloc[0]
        sw_row = full_sw[(full_sw["update"] == "raw_rooted") & (full_sw.condition == condition)].iloc[0]
        if (
            int(stop_row.favorable_runs) >= 7 and float(stop_row.paired_delta) < 0.0
            and int(sw_row.favorable_runs) >= 7 and float(sw_row.paired_delta) < 0.0
            and int(stop_row.divergence_full) < 5
        ):
            complete_passes.append(condition)
    h_complete = bool(complete_passes)
    scale_passes: list[str] = []
    for condition in CALIBRATED_PAIR_KEYS:
        raw = full_stop[(full_stop["update"] == "raw_rooted") & (full_stop.condition == condition)].iloc[0]
        normalized = full_stop[(full_stop["update"] == "normalized") & (full_stop.condition == condition)].iloc[0]
        raw_one = one_step[
            (one_step["update"] == "raw_rooted") & (one_step.calibration == condition)
        ].groupby(["subject", "seed"]).full_minus_normmatched_delta_LEW.mean()
        if (
            int(raw.favorable_runs) >= 7 and float(raw.paired_delta) < 0.0
            and int((raw_one < 0.0).sum()) < 7
            and not (int(normalized.favorable_runs) >= 7 and float(normalized.paired_delta) < 0.0)
        ):
            scale_passes.append(condition)
    h_scale = bool(scale_passes)
    magnitude_passes: list[str] = []
    magnitude_details: dict[str, dict[str, object]] = {}
    sw = summary[(summary["update"] == "raw_rooted") & (summary.method == "sw")]
    for name in ("lpwp_q2", "lpwp_q4"):
        method = summary[(summary["update"] == "raw_rooted") & (summary.method == name)]
        cell = pd.concat([method, sw]).pivot(
            index=["subject", "seed"], columns="method", values="relative_LEW_AUC"
        )
        difference = cell[name] - cell["sw"]
        method_div = int((method.divergence | method["nan"]).sum())
        sw_div = int((sw.divergence | sw["nan"]).sum())
        magnitude_details[name] = {
            "favorable_runs": int((difference < 0.0).sum()),
            "total_runs": int(len(difference)),
            "paired_AUC_delta": float(difference.mean()),
            "mean_AUC_method": float(method.relative_LEW_AUC.mean()),
            "mean_AUC_sw": float(sw.relative_LEW_AUC.mean()),
            "divergence_method": method_div,
            "divergence_sw": sw_div,
        }
        if (
            int((difference < 0.0).sum()) >= 7 and float(difference.mean()) < 0.0
            and method_div < 5 and method_div <= sw_div + 1
        ):
            magnitude_passes.append(name)
    h_mag = bool(magnitude_passes)
    spectral = summary[(summary["update"] == "raw_rooted") & (summary.method == "spectral_s1")]
    spectral_cell = pd.concat([spectral, sw]).pivot(
        index=["subject", "seed"], columns="method", values="relative_LEW_AUC"
    )
    spectral_difference = spectral_cell["spectral_s1"] - spectral_cell["sw"]
    spectral_beats_sw = bool(
        int((spectral_difference < 0.0).sum()) >= 7 and float(spectral_difference.mean()) < 0.0
    )
    solver_failures = int(summary.solver_failure.sum())
    coefficient_path = OUT / "SIGNED_COEFFICIENTS.csv"
    coefficients = pd.read_csv(coefficient_path) if coefficient_path.exists() else pd.DataFrame()
    coeff_error = (
        float((coefficients.effective_coeff_sum - 1.0).abs().max())
        if not coefficients.empty else math.inf
    )
    implementation_audits = bool(
        tests_pass() and audit["passed"] and solver_failures == 0 and coeff_error <= 2e-12
    )
    activation_or_instability = h_active or bool(
        responses[responses.calibration.isin(ESS_PAIR_KEYS)].response_ratio.max() >= 0.10
        and int(summary[summary.calibration.isin(ESS_PAIR_KEYS)].divergence.sum()) > 0
    )
    close = bool(
        implementation_audits and activation_or_instability
        and not h_complete and not h_direction and not h_mag and not spectral_beats_sw
    )
    return {
        "implementation_audits_A": implementation_audits,
        "activation_or_instability_B": activation_or_instability,
        "H-WR-ACTIVE": "PASS" if h_active else "FAIL",
        "H-WR-ACTIVE_conditions": active_conditions,
        "H-WR-DIRECTION": "PASS" if h_direction else "FAIL",
        "H-WR-DIRECTION_conditions": direction_passes,
        "H-WR-COMPLETE": "PASS" if h_complete else "FAIL",
        "H-WR-COMPLETE_conditions": complete_passes,
        "H-WR-SCALE": "PASS" if h_scale else "FAIL",
        "H-WR-SCALE_conditions": scale_passes,
        "H-MAG": "PASS" if h_mag else "FAIL",
        "H-MAG_methods": magnitude_passes,
        "H-MAG_details": magnitude_details,
        "spectral_s1_beats_sw_raw_rooted": spectral_beats_sw,
        "spectral_s1_favorable_runs": int((spectral_difference < 0.0).sum()),
        "spectral_s1_paired_AUC_delta": float(spectral_difference.mean()),
        "one_step_normmatched_favorable_counts": one_counts,
        "response_ratio_median_by_ESS_condition": response_by_condition.to_dict(),
        "maximum_calibrated_response_ratio": float(
            responses[responses.calibration.isin(CALIBRATED_PAIR_KEYS)].response_ratio.max()
        ),
        "effective_coefficient_sum_max_abs_error": coeff_error,
        "solver_failures": solver_failures,
        "TERMINAL_DECISION": "CLOSE" if close else "KEEP",
        "closure_conditions": {
            "A_implementation_audits": implementation_audits,
            "B_response_activated_or_unstable": activation_or_instability,
            "C_complete_fails": not h_complete,
            "D_direction_fails": not h_direction,
            "E_magnitude_coherent_fails": not h_mag,
            "F_spectral_does_not_beat_sw_raw_rooted": not spectral_beats_sw,
        },
    }


def frame_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not math.isfinite(float(value)) else f"{float(value):.7g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_plots(
    summary: pd.DataFrame, full_stop: pd.DataFrame, full_sw: pd.DataFrame,
    responses: pd.DataFrame, coefficients: pd.DataFrame, one_step: pd.DataFrame,
    ess: pd.DataFrame, concentration: pd.DataFrame,
) -> None:
    labels = [method.name for method in METHODS]
    for update, filename in (
        ("normalized", "fig_auc_methods_normalized.png"),
        ("raw_rooted", "fig_auc_methods_raw_rooted.png"),
    ):
        values = summary[summary["update"] == update].groupby("method").relative_LEW_AUC.mean().reindex(labels)
        plt.figure(figsize=(12, 5)); plt.bar(np.arange(len(values)), values)
        plt.xticks(np.arange(len(values)), values.index, rotation=40, ha="right", fontsize=7)
        plt.ylabel("mean relative LEW AUC"); plt.tight_layout(); plt.savefig(OUT / filename, dpi=180); plt.close()
    for table, filename, label in (
        (full_stop, "fig_full_vs_stop.png", "FULL - STOP AUC"),
        (full_sw, "fig_full_vs_sw.png", "FULL - SW AUC"),
    ):
        pivot = table.pivot(index="condition", columns="update", values="paired_delta").reindex(PAIR_KEYS)
        pivot.plot(kind="bar", figsize=(9, 5)); plt.axhline(0, color="black", linewidth=.8)
        plt.ylabel(label); plt.tight_layout(); plt.savefig(OUT / filename, dpi=180); plt.close()
    for column, filename, ylabel in (
        ("response_ratio", "fig_response_ratio_vs_epoch.png", "||full-stop|| / ||stop||"),
        ("cosine_full_stop", "fig_full_stop_cosine_vs_epoch.png", "cosine(full, stop)"),
    ):
        plt.figure(figsize=(10, 5))
        for condition, group in responses.groupby("calibration"):
            curve = group.groupby("epoch")[column].median()
            plt.plot(curve.index, curve.values, marker="o", label=condition)
        plt.xlabel("state epoch"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        plt.savefig(OUT / filename, dpi=180); plt.close()
    signed = coefficients.groupby("calibration")[["negative_coeff_fraction", "negative_coeff_mass", "effective_coeff_l1"]].median()
    signed.plot(kind="bar", figsize=(10, 5)); plt.tight_layout(); plt.savefig(OUT / "fig_signed_effective_coefficients.png", dpi=180); plt.close()
    steps = one_step.groupby("calibration")[["delta_LEW_full", "delta_LEW_stop", "delta_LEW_stop_normmatched"]].mean()
    steps.plot(kind="bar", figsize=(10, 5)); plt.axhline(0, color="black", linewidth=.8)
    plt.ylabel("mean one-step delta LEW"); plt.tight_layout(); plt.savefig(OUT / "fig_one_step_full_stop_normmatched.png", dpi=180); plt.close()
    ess_only = ess[ess.calibration.isin(ESS_PAIR_KEYS)]
    plt.figure(figsize=(10, 5))
    for condition, group in ess_only.groupby("calibration"):
        curve = group.groupby("epoch").beta_t.median()
        plt.plot(curve.index, curve.values, label=condition)
    plt.xlabel("epoch"); plt.ylabel("median beta_t"); plt.legend(); plt.tight_layout(); plt.savefig(OUT / "fig_beta_vs_epoch_by_ess_target.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 6)); plt.scatter(ess_only.target_rho, ess_only.ess_over_N, s=2, alpha=.25)
    plt.plot([0, 1], [0, 1], color="black"); plt.xlabel("target rho"); plt.ylabel("realized ESS/N")
    plt.tight_layout(); plt.savefig(OUT / "fig_realized_ess_vs_target.png", dpi=180); plt.close()
    mag = summary[summary.method.isin(["sw", "spectral_s1", "lpwp_q2", "lpwp_q4"])].groupby(["method", "update"]).relative_LEW_AUC.mean().unstack()
    mag.plot(kind="bar", figsize=(8, 5)); plt.ylabel("mean relative LEW AUC"); plt.tight_layout(); plt.savefig(OUT / "fig_magnitude_coherent_controls.png", dpi=180); plt.close()
    conc = concentration.groupby("method").median_ESS_over_N.mean().reindex(labels)
    plt.figure(figsize=(12, 5)); plt.bar(np.arange(len(conc)), conc)
    plt.xticks(np.arange(len(conc)), conc.index, rotation=40, ha="right", fontsize=7)
    plt.ylabel("median effective weight ESS/N"); plt.tight_layout(); plt.savefig(OUT / "fig_concentration_comparison.png", dpi=180); plt.close()
    context_path = OUT / "EUCLIDEAN_CONTEXT.csv"
    if context_path.exists():
        context = pd.read_csv(context_path)
        plt.figure(figsize=(8, 5))
        for method, group in context.groupby("method"):
            curve = group.groupby("epoch").independent_sw.mean()
            plt.plot(curve.index, curve.values, label=method)
        plt.xlabel("epoch"); plt.ylabel("independent high-L SW"); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(OUT / "fig_euclidean_context.png", dpi=180); plt.close()


def write_tables(
    full_stop: pd.DataFrame, full_sw: pd.DataFrame, summary: pd.DataFrame,
    responses: pd.DataFrame, coefficients: pd.DataFrame, concentration: pd.DataFrame,
) -> None:
    magnitude = summary[summary.method.isin(["sw", "lpwp_q2", "lpwp_q4", "spectral_s1"])].groupby(
        ["update", "method"], as_index=False
    ).agg(mean_AUC=("relative_LEW_AUC", "mean"), final_LEW=("LEW_final", "mean"), divergence=("divergence", "sum"))
    mechanism = responses.groupby(["calibration", "update"], as_index=False).agg(
        median_response_ratio=("response_ratio", "median"),
        median_cosine_full_stop=("cosine_full_stop", "median"),
        median_cosine_full_vs_sw=("cosine_full_vs_sw", "median"),
    ).merge(
        coefficients.groupby(["calibration", "update"], as_index=False).agg(
            median_negative_coeff_fraction=("negative_coeff_fraction", "median"),
            median_negative_coeff_mass=("negative_coeff_mass", "median"),
            median_effective_coeff_l1=("effective_coeff_l1", "median"),
        ), on=["calibration", "update"], how="left"
    )
    text = "\n\n".join([
        "# Registered pairwise tables",
        "## A. FULL vs corresponding STOP\n\n" + frame_markdown(full_stop),
        "## B. FULL vs SW\n\n" + frame_markdown(full_sw),
        "## C. Magnitude-sensitive coherent controls\n\n" + frame_markdown(magnitude),
        "## D. Weight-response mechanism\n\n" + frame_markdown(mechanism),
        "## E. Concentration\n\n" + frame_markdown(concentration),
    ]) + "\n"
    (OUT / "TABLE.md").write_text(text)


def write_report(
    summary: pd.DataFrame, full_stop: pd.DataFrame, full_sw: pd.DataFrame,
    responses: pd.DataFrame, coefficients: pd.DataFrame, one_step: pd.DataFrame,
    concentration: pd.DataFrame, gate: dict[str, object], euclidean: dict[str, object],
) -> None:
    beta = json.loads((OUT / "BETA_CALIBRATION.json").read_text())
    divergence = int((summary.divergence | summary["nan"]).sum())
    solver_failures = int(summary.solver_failure.sum())
    beta1_median = float(responses[responses.calibration == "b1"].response_ratio.median())
    maximum = float(gate["maximum_calibrated_response_ratio"])
    pair_extract = full_stop[[
        "update", "condition", "mean_AUC_full", "mean_AUC_stop", "paired_delta",
        "favorable_runs", "total_runs", "final_LEW_full", "final_LEW_stop",
        "divergence_full", "divergence_stop",
    ]]
    magnitude = summary[summary.method.isin(["sw", "spectral_s1", "lpwp_q2", "lpwp_q4"])].groupby(
        ["update", "method"], as_index=False
    ).agg(mean_AUC=("relative_LEW_AUC", "mean"), final_LEW=("LEW_final", "mean"), divergence=("divergence", "sum"))
    decision_block = (
        "TERMINAL DECISION:\nCLOSE directional-Wasserstein-magnitude adaptive slicing as an\n"
        "optimization direction for SPD/EEG alignment."
        if gate["TERMINAL_DECISION"] == "CLOSE" else
        "TERMINAL DECISION:\nKEEP only the preregistered hypothesis/hypotheses identified below."
    )
    passed = [name for name in ("H-WR-ACTIVE", "H-WR-DIRECTION", "H-WR-COMPLETE", "H-WR-SCALE", "H-MAG") if gate[name] == "PASS"]
    active_reasons = ", ".join(
        f"{condition} median response ratio={value:.6g}"
        for condition, value in gate["response_ratio_median_by_ESS_condition"].items()
        if value >= 0.10
    )
    magnitude_reasons = "; ".join(
        f"{method}: {details['favorable_runs']}/{details['total_runs']} favorable paired runs, "
        f"mean AUC {details['mean_AUC_method']:.9g} vs SW {details['mean_AUC_sw']:.9g}, "
        f"paired delta={details['paired_AUC_delta']:.9g}, divergences "
        f"{details['divergence_method']} vs SW {details['divergence_sw']}"
        for method, details in gate["H-MAG_details"].items()
        if method in gate["H-MAG_methods"]
    )
    keep_reasons = (
        f"H-WR-ACTIVE passed because {active_reasons}. "
        f"H-MAG passed because {magnitude_reasons}. "
        "The practical KEEP decision is supported by H-MAG; H-WR-ACTIVE establishes that "
        "the response mechanism was successfully activated, not that it improved alignment."
        if gate["TERMINAL_DECISION"] == "KEEP" else ""
    )
    text = f"""- regression/audit tests: PASS
- completed BNCI trajectories: {len(summary)}/288
- divergence trajectories: {divergence}
- solver failures: {solver_failures}
- beta_scale: {beta['beta_scale']:.15g}
- spectral rho_match: {RHO_MATCH:.15g}
- median beta=1 response ratio: {beta1_median:.15g}
- maximum calibrated response ratio: {maximum:.15g}
- H-WR-ACTIVE: {gate['H-WR-ACTIVE']}
- H-WR-DIRECTION: {gate['H-WR-DIRECTION']}
- H-WR-COMPLETE: {gate['H-WR-COMPLETE']}
- H-WR-SCALE: {gate['H-WR-SCALE']}
- H-MAG: {gate['H-MAG']}
- TERMINAL DECISION: {gate['TERMINAL_DECISION']}
- Euclidean context: RUN

# Terminal EBSW response audit

## Protocol and implementation audit

This terminal experiment used BNCI2014_001 subjects 1, 3, and 8; seeds 6398, 3654, and 1788; 500 updates; direct resampled `N_proj=500` Frobenius directions; and independent exact LEW at epochs 0,25,…,500. Exactly 16 methods × 2 update formulations × 9 subject-seed cases were run. No HGD, hierarchy, fixed bank, downstream classifier, LR sweep, or raw-power trajectory was added.

FULL exponential EBSW differentiated `alpha=softmax(beta*h)` and therefore included `alpha_i[1+beta(h_i-F)]`. STOP used the identical alpha values and scalar objective but detached alpha. ESS beta was solved deterministically from `h.detach()` and treated as constant during objective differentiation. This is a conditional full EBSW gradient at calibrated beta, with stop-gradient through the beta calibration map.

The analytic FULL and STOP identities, coefficient sum, ESS endpoints and solver tolerance, exact rho match, q=1 equality with SW, copied-state immutability, common banks/h values, frozen step sizes, independent evaluator, absence of raw-power/hierarchy, and frozen prior hashes passed the full regression suite and runtime audits.

`divergence` uses the existing audited LEW evaluator definition: a trajectory is flagged if any evaluated LEW exceeds its epoch-0 LEW or is nonfinite. Thus a finite, completed trajectory can be counted as divergent even when its run status is `ok`.

## FULL versus STOP

{frame_markdown(pair_extract)}

Negative paired delta favors FULL. Run counts retain all divergences and solver failures in the denominator.

## FULL versus SW

{frame_markdown(full_sw)}

## Magnitude-sensitive coherent controls

`lpwp_q2` and `lpwp_q4` are L^(p*q)-aggregations of the directional W_p field. They are not standard SW_pq because the inner directional distance remains W_p.

{frame_markdown(magnitude)}

## Response and signed coefficients

{frame_markdown(responses.groupby(['calibration', 'update'], as_index=False).agg(median_response_ratio=('response_ratio','median'), median_full_stop_cosine=('cosine_full_stop','median'), median_full_vs_sw_cosine=('cosine_full_vs_sw','median'), median_stop_vs_sw_cosine=('cosine_stop_vs_sw','median')))}

{frame_markdown(coefficients.groupby(['calibration', 'update'], as_index=False).agg(median_negative_fraction=('negative_coeff_fraction','median'), median_negative_mass=('negative_coeff_mass','median'), median_positive_mass=('positive_coeff_mass','median'), median_coeff_l1=('effective_coeff_l1','median'), max_coeff_sum_error=('effective_coeff_sum', lambda x: float((x-1).abs().max()))))}

The response coefficients may be signed; negative mass represents redistribution/cancellation, not simply positive emphasis.

## One-step magnitude-versus-direction decomposition

{frame_markdown(one_step.groupby(['calibration', 'update'], as_index=False).agg(mean_delta_full=('delta_LEW_full','mean'), mean_delta_stop=('delta_LEW_stop','mean'), mean_delta_stop_normmatched=('delta_LEW_stop_normmatched','mean'), mean_full_minus_normmatched=('full_minus_normmatched_delta_LEW','mean')))}

The registered subject-seed summary averages FULL-minus-norm-matched-STOP one-step LEW differences across eight states from each normalized FULL trajectory. Negative favors FULL direction beyond scalar norm matching.

## Concentration and beta calibration

`beta_scale=1/hbar` is a scale diagnostic, not an optimized beta. ESS conditions target rho 0.25, 0.50, 0.75, and exact sigma-1 rho match `{RHO_MATCH:.15g}`. Solver errors and realized concentration are retained in `ESS_DIAGNOSTICS.csv`.

{frame_markdown(concentration)}

## Euclidean context

The optional context reused the frozen E5 two-dimensional Gaussian-mixture clouds, direction streams, seeds, epochs, evaluation bank, and common normalized-step construction. It was not part of the BNCI terminal gate and is not claimed as an exact reproduction of an EBSW paper experiment. Status: `{euclidean['euclidean_context_status']}`.

## Terminal hypotheses and decision

Passed hypotheses: {', '.join(passed) if passed else 'none'}.

{decision_block}

Across the preregistered representative value-adaptive families tested, we found no evidence that emphasizing slices according to their directional Wasserstein magnitude improves SPD EEG distribution alignment.

This conclusion is deliberately scoped. It does not invalidate the EBSW triangle-inequality counterexamples, the coherent pair-independent metric construction, the mathematical validity of the spectral metric, or the fixed-bank overfitting findings. A CLOSE decision applies only to the use of directional Wasserstein magnitude as an informativeness signal to improve SPD/EEG distribution-alignment optimization; it is not a universal impossibility theorem about adaptive slicing, coherent metrics, EBSW, or other possible informativeness definitions.
"""
    if gate["TERMINAL_DECISION"] == "KEEP":
        text = text.replace(
            "Across the preregistered representative value-adaptive families tested, we found no evidence that emphasizing slices according to their directional Wasserstein magnitude improves SPD EEG distribution alignment.",
            keep_reasons,
        )
    (OUT / "REPORT.md").write_text(text)


def analyze() -> dict[str, object]:
    require_prepared()
    frame = load_run_frames()
    summary = summarize_runs(frame)
    responses = load_mechanism("response", RESPONSE_COLUMNS)
    coefficients = load_mechanism("coefficients", SIGNED_COLUMNS)
    one_step = load_mechanism("one_step", ONE_STEP_COLUMNS)
    ess = frame[(frame.family == "ebsw") & (frame.epoch > 0)][[
        "method", "calibration", "response", "update", "subject", "seed", "epoch",
        "beta_t", "target_rho", "ess", "ess_over_N", "realized_rho_error",
        "weight_entropy", "max_weight", "top5_mass", "top10_mass",
        "solver_status", "solver_iterations", "bracket_steps",
    ]].copy()
    concentration = concentration_summary(frame)
    subjects = subject_effects(summary)
    full_stop = paired_table(summary, "stop")
    full_sw = paired_table(summary, "sw")
    audit = bank_audit(frame)
    summary.to_csv(OUT / "CORE_RESULTS.csv", index=False)
    subjects.to_csv(OUT / "SUBJECT_RESULTS.csv", index=False)
    responses.to_csv(OUT / "RESPONSE_DIAGNOSTICS.csv", index=False)
    coefficients.to_csv(OUT / "SIGNED_COEFFICIENTS.csv", index=False)
    one_step.to_csv(OUT / "ONE_STEP_DECOMPOSITION.csv", index=False)
    ess.to_csv(OUT / "ESS_DIAGNOSTICS.csv", index=False)
    concentration.to_csv(OUT / "CONCENTRATION_SUMMARY.csv", index=False)
    dump_json(OUT / "BANK_AUDIT.json", audit)
    manifest = json.loads((OUT / "MANIFEST_TRAJECTORIES.json").read_text())
    dump_json(OUT / "RUN_MANIFEST.json", manifest)
    gate = gate_results(summary, full_stop, full_sw, responses, one_step, audit)
    dump_json(OUT / "GATE.json", gate)
    euclidean = json.loads((OUT / "EUCLIDEAN_CONTEXT.json").read_text())
    write_tables(full_stop, full_sw, summary, responses, coefficients, concentration)
    make_plots(summary, full_stop, full_sw, responses, coefficients, one_step, ess, concentration)
    write_report(summary, full_stop, full_sw, responses, coefficients, one_step, concentration, gate, euclidean)
    verify_frozen()
    return {
        "completed": len(summary),
        "divergence": int((summary.divergence | summary["nan"]).sum()),
        "solver_failures": int(summary.solver_failure.sum()),
        **{key: gate[key] for key in ("H-WR-ACTIVE", "H-WR-DIRECTION", "H-WR-COMPLETE", "H-WR-SCALE", "H-MAG", "TERMINAL_DECISION")},
    }


def frozen_snapshot() -> dict[str, object]:
    source_files = [
        PROJECT / "coherent_slicing" / "aggregations.py",
        PROJECT / "coherent_slicing" / "spectral.py",
        PROJECT / "coherent_slicing" / "ot.py",
        PROJECT / "experiments" / "run_euclidean.py",
        PROJECT / "experiments" / "run_spectral_sampling_update_factorial.py",
        EXTERNAL / "evobank" / "data.py", EXTERNAL / "evobank" / "lew.py",
        EXTERNAL / "evobank" / "ot1d.py", EXTERNAL / "evobank" / "svec.py",
    ]
    caches = [
        EXTERNAL / "results" / "pilot_hgd" / "data_cache" / DATASET / f"subject_{subject:02d}_logs.pt"
        for subject in SUBJECTS
    ]
    return {
        "frozen_branch_heads": {name: branch_sha(name) for name in FROZEN_BRANCHES},
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "cache_sha256": {str(path): sha256(path) for path in caches},
        "prior_result_tree_sha256": {
            relative: tree_sha256(PROJECT / relative) for relative in FROZEN_RESULT_HASHES
        },
        "frozen_untracked_sha256": {
            relative: sha256(PROJECT / relative) for relative in FROZEN_UNTRACKED_FILES
        },
    }


def verify_frozen() -> None:
    snapshot = frozen_snapshot()
    if snapshot["frozen_branch_heads"] != FROZEN_BRANCHES:
        raise RuntimeError("a frozen branch head changed")
    if snapshot["prior_result_tree_sha256"] != FROZEN_RESULT_HASHES:
        raise RuntimeError("a frozen prior result tree changed")
    if snapshot["frozen_untracked_sha256"] != FROZEN_UNTRACKED_FILES:
        raise RuntimeError("a frozen abandoned audit source changed")
    path = OUT / "FROZEN_SOURCE_HASHES.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise RuntimeError("a frozen source or cache changed")
    if not path.exists():
        dump_json(path, snapshot)


def prepare() -> None:
    if current_branch() != BRANCH:
        raise RuntimeError(f"must run on {BRANCH}")
    if not tests_pass():
        raise RuntimeError("regression/audit tests must pass before calibration")
    OUT.mkdir(parents=True, exist_ok=True)
    verify_frozen()
    device = check_device()
    config = registered_config()
    config_path = OUT / "CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("refusing to alter frozen CONFIG.json")
    dump_json(config_path, config)
    calibration = calibration_payload()
    calibration_path = OUT / "BETA_CALIBRATION.json"
    if calibration_path.exists() and json.loads(calibration_path.read_text()) != calibration:
        raise RuntimeError("refusing to alter frozen BETA_CALIBRATION.json")
    dump_json(calibration_path, calibration)
    environment = {
        "branch": current_branch(), "commit_at_calibration": branch_sha("HEAD"),
        "python": platform.python_version(), "python_executable": sys.executable,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(), "hostname": platform.node(),
        "dtype": str(DTYPE), "amp": False, "autocast": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": device,
    }
    dump_json(OUT / "ENVIRONMENT.json", environment)


def require_prepared() -> None:
    if current_branch() != BRANCH:
        raise RuntimeError(f"must run on {BRANCH}")
    verify_frozen()
    if not tests_pass():
        raise RuntimeError("regression/audit tests are not passing")
    for name in ("CONFIG.json", "BETA_CALIBRATION.json"):
        if not (OUT / name).exists():
            raise RuntimeError(f"missing frozen calibration artifact {name}")
    if json.loads((OUT / "CONFIG.json").read_text()) != registered_config():
        raise RuntimeError("CONFIG.json no longer matches registered constants")


def manifest_complete() -> bool:
    path = OUT / "MANIFEST_TRAJECTORIES.json"
    if not path.exists():
        return False
    records = json.loads(path.read_text())
    return len(records) == 288 and all(
        record.get("status") in {"ok", "nonfinite", "solver_failure", "cached_complete"}
        for record in records
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True,
        choices=("prepare", "euclidean", "run", "analyze", "all"),
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    configure_numerics()
    if args.phase == "prepare":
        prepare()
        calibration = json.loads((OUT / "BETA_CALIBRATION.json").read_text())
        print(f"[PHASE 4 COMPLETE] beta_scale={calibration['beta_scale']:.15g} rho_match={RHO_MATCH:.15g}")
        return 0
    require_prepared()
    if args.phase in {"euclidean", "all"}:
        status = run_euclidean_context()
        print(f"[PHASE 5 COMPLETE] {status['euclidean_context_status']}")
        if args.phase == "euclidean":
            return 0
    check_device()
    if args.phase in {"run", "all"}:
        if not (OUT / "EUCLIDEAN_CONTEXT.json").exists():
            raise RuntimeError("Phase 5 Euclidean context/status must precede BNCI runs")
        execute_grid(rerun=args.rerun)
        if not manifest_complete():
            raise RuntimeError("all 288 registered trajectories were not retained")
        print("[PHASE 6/7 COMPLETE] 288 trajectories and mechanism diagnostics retained")
    if args.phase in {"analyze", "all"}:
        if not manifest_complete():
            raise RuntimeError("analysis requires all 288 registered trajectories")
        result = analyze()
        print(f"[PHASE 8 COMPLETE] {json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
