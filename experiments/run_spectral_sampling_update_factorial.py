#!/usr/bin/env python
"""Registered BNCI spectral sampling/update 2 x 2 x 3 factorial.

This module evaluates direct ambient SPDSW only.  It deliberately contains no
hierarchical projection-bank or mixture code.
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
from typing import Callable, Iterable, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from coherent_slicing import lognormal_spectral_weights


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "spectral_sampling_update_factorial_v1"
EXTERNAL = Path("/home/pikachu/EBSPDSW")
sys.path.insert(0, str(EXTERNAL))

from evobank.data import load as load_cached_subject  # noqa: E402
from evobank.lew import LEWEvaluator  # noqa: E402
from evobank.ot1d import w2_squared_per_direction  # noqa: E402
from evobank.svec import SvecBasis  # noqa: E402


BRANCH = "exp/spectral-sampling-update-factorial-v1"
DATASET = "BNCI2014_001"
SUBJECTS = (1, 3, 8)
SEEDS = (6398, 3654, 1788)
EPOCHS = 500
N_PROJ = 500
SIGMA = 1.0
P = 2
ETA_NORM = 2.793683898093503
ETA_NORM_SOURCE = (
    "results/high_support_fixed_vs_resampled_spdsw_v2/"
    "BNCI_NORMALIZED_STEP.json (frozen audited BNCI value)"
)
ETA_POWER = 3000.0
ROOT_EPSILON = 0.0
DTYPE = torch.float64
DEVICE = torch.device("cuda:3")
PHYSICAL_GPU = 2
THRESHOLDS = (0.95, 0.90, 0.80, 0.70, 0.60)
DIAGNOSTIC_EPOCHS = (0, 25, 50, 100, 200, 300, 400, 500)

FROZEN_BRANCHES = {
    "main": "4edf5dda470c5e525c5feb274462414751348b4b",
    "exp/lognormal-spectral-spdhsw-v1": "b0e5b47e17d45a94f39b2b5ba08fa965a5d3a77c",
    "exp/fixed-vs-resampled-spectral-spdsw-v1": "ec4a68af6107b3522d5e841dd35f708402a6378b",
    "exp/high-support-fixed-vs-resampled-spdsw-v2": "b9f61cc6824e2fc71e925ceb960c241bdaf5dbec",
    "exp/spectral-raw-optimization-audit-v1": "b9f61cc6824e2fc71e925ceb960c241bdaf5dbec",
    "exp/spectral-update-formulation-audit-v1": "b9f61cc6824e2fc71e925ceb960c241bdaf5dbec",
}

FROZEN_RESULT_HASHES = {
    "results/coherent_sw_overnight": "7d840778efb9f95d6a255961d234a65ce8af55077fa96fa27d8358ce5dd47f48",
    "results/lognormal_spectral_spdhsw_v1": "baa0352a347ca49bc522a6e375755791d464220af5769f7f756d49b2129864e9",
    "results/fixed_vs_resampled_spectral_spdsw_v1": "9eaadb1b5b6136cd418065db22e06cd90387d21ed8529f72ff9a2fbc40e8e451",
    "results/high_support_fixed_vs_resampled_spdsw_v2": "64cc8922267042f98ec4f5ba10344e341a83ffacac6abf4dac45457daa7bdc7f",
}

FROZEN_UNTRACKED_FILES = {
    "experiments/run_spectral_raw_optimization_audit.py":
        "f06c9990f6253c019f1dd5112e514e21c0c1f6cfd4d62fdfd0d7d4349b373686",
}


@dataclass(frozen=True)
class Method:
    name: str
    sampling: str
    aggregation: str
    update: str
    N_proj: int = N_PROJ
    sigma: float = 0.0


@dataclass
class FixedBankState:
    directions: torch.Tensor
    target_projection: torch.Tensor
    bank_seed: int
    bank_hash: str
    target_projection_hash: str
    sampling_ms: float
    target_projection_ms: float


@dataclass
class StageTimes:
    direction_sampling_ms: float = 0.0
    source_projection_ms: float = 0.0
    target_projection_ms: float = 0.0
    wasserstein_1d_ms: float = 0.0
    sorting_aggregation_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_update_ms: float = 0.0

    def total_epoch_ms(self) -> float:
        return float(sum(getattr(self, item.name) for item in fields(self)))


METHODS = (
    Method("fixed_uniform_normalized_power", "fixed", "uniform", "normalized_power"),
    Method("fixed_spectral_normalized_power_s1", "fixed", "spectral", "normalized_power", sigma=SIGMA),
    Method("fixed_uniform_raw_power", "fixed", "uniform", "raw_power"),
    Method("fixed_spectral_raw_power_s1", "fixed", "spectral", "raw_power", sigma=SIGMA),
    Method("fixed_uniform_raw_rooted", "fixed", "uniform", "raw_rooted"),
    Method("fixed_spectral_raw_rooted_s1", "fixed", "spectral", "raw_rooted", sigma=SIGMA),
    Method("resampled_uniform_normalized_power", "resampled", "uniform", "normalized_power"),
    Method("resampled_spectral_normalized_power_s1", "resampled", "spectral", "normalized_power", sigma=SIGMA),
    Method("resampled_uniform_raw_power", "resampled", "uniform", "raw_power"),
    Method("resampled_spectral_raw_power_s1", "resampled", "spectral", "raw_power", sigma=SIGMA),
    Method("resampled_uniform_raw_rooted", "resampled", "uniform", "raw_rooted"),
    Method("resampled_spectral_raw_rooted_s1", "resampled", "spectral", "raw_rooted", sigma=SIGMA),
)

CONFIG_TEMPLATE = {
    "version": "spectral_sampling_update_factorial_v1",
    "created_before_scientific_runs": True,
    "branch": BRANCH,
    "dataset": DATASET,
    "subjects": list(SUBJECTS),
    "seeds": list(SEEDS),
    "epochs": EPOCHS,
    "d": 22,
    "m": 253,
    "p": P,
    "N_proj": N_PROJ,
    "direct_only": True,
    "hierarchical_methods": False,
    "factorial": {
        "sampling": ["fixed", "resampled"],
        "aggregation": ["uniform", "lognormal_spectral_sigma_1.0"],
        "update": ["normalized_power", "raw_power", "raw_rooted"],
        "method_names": [method.name for method in METHODS],
        "trajectory_count": 108,
    },
    "spectral": {
        "sigma": SIGMA,
        "sigma_sweep": False,
        "exact_finite_interval_weights": True,
        "rank_assignment_detached": True,
    },
    "updates": {
        "normalized_eta_norm": ETA_NORM,
        "normalized_eta_norm_source": ETA_NORM_SOURCE,
        "raw_power_learning_rate": ETA_POWER,
        "raw_rooted_eta_source": "ROOTED_STEP_CALIBRATION.json",
        "gradient_clipping": False,
        "momentum": False,
        "weight_decay": False,
        "lr_sweep": False,
    },
    "rooted": {
        "objective": "sqrt(F)",
        "epsilon": ROOT_EPSILON,
        "calibration_subjects": list(SUBJECTS),
        "calibration_seed": 6398,
        "calibration_aggregation": "uniform only",
        "rule": (
            "eta_root = median_subject(||-3000 grad F_uniform||) / "
            "median_subject(||grad sqrt(F_uniform)||)"
        ),
    },
    "sampling": {
        "fixed_bank_shared_by_six_methods": True,
        "resampled_epoch_bank_shared_by_six_methods": True,
        "fixed_resampled_epoch_0_bank_shared": True,
        "fixed_target_projection_cached": True,
    },
    "evaluation": {
        "kind": "independent exact Log-Euclidean Wasserstein",
        "epochs": list(range(EPOCHS + 1)),
        "independent_of_training_banks": True,
        "excluded_from_optimization_wall_clock": True,
    },
    "thresholds": list(THRESHOLDS),
    "diagnostic_epochs": list(DIAGNOSTIC_EPOCHS),
    "no_early_stopping": True,
}

EPOCH_COLUMNS = [
    "dataset", "method", "sampling", "aggregation", "update", "subject", "seed",
    "epoch", "epochs", "N_proj", "sigma", "d", "m", "objective_power",
    "rooted_distance", "lew", "relative_lew", "lew_reduction_pct", "gap_closure",
    "raw_gradient_norm", "applied_update_norm", "mean_h", "std_h", "max_h", "min_h",
    "spectral_effective_N", "spectral_entropy", "spectral_max_weight",
    "spectral_top5_mass", "spectral_top10_mass", "bank_seed", "bank_hash",
    "bank_hash_kind", "target_projection_hash", "initial_source_hash", "target_hash",
    "direction_sampling_ms", "source_projection_ms", "target_projection_ms",
    "wasserstein_1d_ms", "sorting_aggregation_ms", "backward_ms",
    "optimizer_update_ms", "total_epoch_ms", "one_time_bank_sampling_ms",
    "one_time_target_projection_ms", "evaluation_ms", "cumulative_optimization_ms",
    "cumulative_evaluation_ms", "cumulative_direct_projection_count",
    "cumulative_direction_draw_count", "learning_rate", "eta_norm", "eta_root",
    "nan", "diverged", "status",
]

DIAGNOSTIC_COLUMNS = [
    "dataset", "method", "sampling", "aggregation", "update", "subject", "seed",
    "epoch", "raw_gradient_norm", "applied_update_norm", "uniform_spectral_cosine",
    "paired_uniform_gradient_norm", "paired_spectral_gradient_norm", "bank_seed",
    "bank_hash", "spectral_effective_N", "spectral_entropy", "spectral_max_weight",
    "spectral_top5_mass", "spectral_top10_mass", "diagnostic_ms",
]

T = TypeVar("T")
_BANK_HASH_CACHE: dict[tuple[int, int, int], str] = {}


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


def to_builtin(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n")


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
    if processes:
        raise RuntimeError(f"physical GPU {PHYSICAL_GPU} is contaminated: {processes}")
    physical_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(PHYSICAL_GPU), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip().removeprefix("GPU-")
    torch.cuda.set_device(DEVICE)
    properties = torch.cuda.get_device_properties(DEVICE)
    if str(properties.uuid) != physical_uuid:
        raise RuntimeError(f"CUDA ordinal mismatch: physical GPU {PHYSICAL_GPU}")
    return {
        "physical_gpu": PHYSICAL_GPU, "torch_device": str(DEVICE),
        "name": properties.name, "uuid": physical_uuid,
        "total_memory_bytes": properties.total_memory,
        "compute_processes_before_initialization": processes,
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
    """Audited deterministic triangular seed sequence, independent of method."""
    return int(seed + epoch_zero_based * (epoch_zero_based + 1) // 2)


def method_bank_seed(method: Method, seed: int, epoch_zero_based: int) -> int:
    return direction_seed(seed, 0 if method.sampling == "fixed" else epoch_zero_based)


def sample_frobenius_directions(N_proj: int, basis: SvecBasis, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=basis.device).manual_seed(int(seed))
    gaussian = torch.randn(
        N_proj, basis.d, basis.d, generator=generator,
        device=basis.device, dtype=basis.dtype,
    )
    matrices = gaussian + gaussian.transpose(-1, -2)
    matrices = matrices / matrices.norm(dim=(-1, -2), keepdim=True)
    return basis.forward(matrices)


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


def bank_hash(directions: torch.Tensor, seed: int, *, full: bool) -> tuple[str, str]:
    key = (int(directions.shape[0]), int(directions.shape[1]), int(seed))
    if full and key in _BANK_HASH_CACHE:
        return _BANK_HASH_CACHE[key], "full_tensor_sha256"
    value = tensor_sha256(directions, full=full)
    if full:
        _BANK_HASH_CACHE[key] = value
    return value, "full_tensor_sha256" if full else "three_row_sha256"


def build_fixed_bank_state(
    basis: SvecBasis, target_vec: torch.Tensor, seed: int, N_proj: int = N_PROJ
) -> FixedBankState:
    sampled_seed = direction_seed(seed, 0)
    directions, sampling_ms = timed(
        basis.device, lambda: sample_frobenius_directions(N_proj, basis, sampled_seed)
    )
    target_projection, target_ms = timed(basis.device, lambda: target_vec @ directions.T)
    fingerprint, _ = bank_hash(directions, sampled_seed, full=True)
    return FixedBankState(
        directions=directions, target_projection=target_projection,
        bank_seed=sampled_seed, bank_hash=fingerprint,
        target_projection_hash=tensor_sha256(target_projection, full=True),
        sampling_ms=sampling_ms, target_projection_ms=target_ms,
    )


def epoch_bank(
    method: Method, basis: SvecBasis, target_vec: torch.Tensor, seed: int,
    epoch_zero_based: int, fixed_state: FixedBankState | None,
) -> tuple[torch.Tensor, torch.Tensor, int, str, str, str, float, float]:
    sampled_seed = method_bank_seed(method, seed, epoch_zero_based)
    if method.sampling == "fixed":
        if fixed_state is None:
            raise ValueError("fixed method requires fixed bank")
        return (
            fixed_state.directions, fixed_state.target_projection,
            fixed_state.bank_seed, fixed_state.bank_hash, "full_tensor_sha256",
            fixed_state.target_projection_hash, 0.0, 0.0,
        )
    directions, sampling_ms = timed(
        basis.device, lambda: sample_frobenius_directions(method.N_proj, basis, sampled_seed)
    )
    projected_target, target_ms = timed(basis.device, lambda: target_vec @ directions.T)
    # Hash epoch 0 in full so fixed/resampled identity is directly auditable.
    fingerprint, kind = bank_hash(
        directions, sampled_seed, full=(epoch_zero_based == 0)
    )
    return (
        directions, projected_target, sampled_seed, fingerprint, kind,
        tensor_sha256(projected_target, full=False), sampling_ms, target_ms,
    )


def aggregate_directional_costs(
    h: torch.Tensor, aggregation: str, ordered_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if aggregation == "uniform":
        weights = torch.full_like(h, 1.0 / h.numel()).detach()
        return h.mean(), weights
    if aggregation != "spectral" or ordered_weights is None:
        raise ValueError(aggregation)
    order = torch.argsort(h.detach(), stable=True)
    assigned = torch.empty_like(ordered_weights)
    assigned[order] = ordered_weights.detach()
    assigned = assigned.detach()
    return torch.sum(assigned * h), assigned


def rooted_objective(power: torch.Tensor) -> torch.Tensor:
    if ROOT_EPSILON == 0.0:
        return torch.sqrt(power)
    return torch.sqrt(power + ROOT_EPSILON)


def objective_for_update(power: torch.Tensor, update: str) -> torch.Tensor:
    if update in {"normalized_power", "raw_power"}:
        return power
    if update == "raw_rooted":
        return rooted_objective(power)
    raise ValueError(update)


def normalized_update(gradient: torch.Tensor, eta_norm: float = ETA_NORM) -> torch.Tensor:
    norm = gradient.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        return torch.full_like(gradient, math.nan)
    return -float(eta_norm) * gradient / norm


def raw_update(gradient: torch.Tensor, update: str, eta_root: float) -> torch.Tensor:
    if update == "raw_power":
        return -ETA_POWER * gradient
    if update == "raw_rooted":
        return -float(eta_root) * gradient
    if update == "normalized_power":
        return normalized_update(gradient)
    raise ValueError(update)


def distribution_diagnostics(weights: torch.Tensor) -> dict[str, float]:
    weights = weights.detach()
    positive = weights > 0
    return {
        "spectral_effective_N": float(1.0 / weights.square().sum()),
        "spectral_entropy": float(-(weights[positive] * weights[positive].log()).sum()),
        "spectral_max_weight": float(weights.max()),
        "spectral_top5_mass": float(torch.topk(weights, min(5, weights.numel())).values.sum()),
        "spectral_top10_mass": float(torch.topk(weights, min(10, weights.numel())).values.sum()),
    }


def evaluate_independent_lew(
    evaluator: LEWEvaluator, basis: SvecBasis, parameter: torch.Tensor
) -> tuple[float, float]:
    started = time.perf_counter()
    value = evaluator(basis.inverse(parameter.detach()))
    return value, 1000.0 * (time.perf_counter() - started)


def paired_gradient_diagnostic(
    parameter: torch.Tensor, directions: torch.Tensor, projected_target: torch.Tensor,
    update: str, ordered_weights: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Return same-state/bank uniform-vs-spectral gradient norms and cosine."""
    started = time.perf_counter()
    projected_source = parameter @ directions.T
    h = w2_squared_per_direction(projected_source.T, projected_target.T)
    uniform_power, _ = aggregate_directional_costs(h, "uniform", None)
    spectral_power, _ = aggregate_directional_costs(h, "spectral", ordered_weights)
    uniform_objective = objective_for_update(uniform_power, update)
    spectral_objective = objective_for_update(spectral_power, update)
    uniform_gradient = torch.autograd.grad(
        uniform_objective, parameter, retain_graph=True, create_graph=False
    )[0]
    spectral_gradient = torch.autograd.grad(
        spectral_objective, parameter, retain_graph=False, create_graph=False
    )[0]
    uniform_norm = float(uniform_gradient.norm())
    spectral_norm = float(spectral_gradient.norm())
    denominator = uniform_norm * spectral_norm
    cosine = (
        float(torch.sum(uniform_gradient * spectral_gradient)) / denominator
        if denominator > 0.0 else math.nan
    )
    sync(parameter.device)
    return uniform_norm, spectral_norm, cosine, 1000.0 * (time.perf_counter() - started)


def rooted_step_from_norms(power_update_norms: Iterable[float], root_gradient_norms: Iterable[float]) -> float:
    power_values = tuple(float(value) for value in power_update_norms)
    root_values = tuple(float(value) for value in root_gradient_norms)
    if len(power_values) != len(SUBJECTS) or len(root_values) != len(SUBJECTS):
        raise ValueError("rooted calibration requires exactly the three registered subjects")
    denominator = float(np.median(root_values))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("rooted calibration gradient norm must be positive and finite")
    return float(np.median(power_values)) / denominator


def calibrate_rooted_step(*, write: bool = True) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    method = next(method for method in METHODS if method.name == "fixed_uniform_raw_rooted")
    for subject in SUBJECTS:
        source, target, meta = load_cached_subject(DATASET, subject, DEVICE)
        basis = SvecBasis(meta["d"], DEVICE, DTYPE)
        parameter = basis.forward(source).clone().requires_grad_(True)
        target_vec = basis.forward(target)
        sampled_seed = direction_seed(6398, 0)
        directions = sample_frobenius_directions(N_PROJ, basis, sampled_seed)
        h = w2_squared_per_direction(
            (parameter @ directions.T).T, (target_vec @ directions.T).T
        )
        power, _ = aggregate_directional_costs(h, "uniform", None)
        power_gradient = torch.autograd.grad(power, parameter, retain_graph=True)[0]
        root = rooted_objective(power)
        root_gradient = torch.autograd.grad(root, parameter)[0]
        rows.append({
            "dataset": DATASET, "subject": subject, "seed": 6398,
            "N_proj": method.N_proj, "aggregation": "uniform",
            "bank_seed": sampled_seed,
            "bank_hash": bank_hash(directions, sampled_seed, full=True)[0],
            "power_objective": float(power.detach()),
            "rooted_distance": float(root.detach()),
            "power_gradient_norm": float(power_gradient.norm()),
            "U_power": ETA_POWER * float(power_gradient.norm()),
            "G_root": float(root_gradient.norm()),
        })
        del source, target, parameter, target_vec, directions, h
    median_u = float(np.median([row["U_power"] for row in rows]))
    median_g = float(np.median([row["G_root"] for row in rows]))
    eta_root = rooted_step_from_norms(
        [row["U_power"] for row in rows], [row["G_root"] for row in rows]
    )
    payload = {
        "eta_root": eta_root,
        "eta_power_reference": ETA_POWER,
        "median_U_power": median_u,
        "median_G_root": median_g,
        "rule": CONFIG_TEMPLATE["rooted"]["rule"],
        "uniform_only": True,
        "calibrated_before_comparative_training": True,
        "root_epsilon": ROOT_EPSILON,
        "subject_values": rows,
    }
    if write:
        path = OUT / "ROOTED_STEP_CALIBRATION.json"
        if path.exists() and json.loads(path.read_text()) != to_builtin(payload):
            raise RuntimeError("refusing to alter frozen rooted step calibration")
        dump_json(path, payload)
    return payload


def read_eta_root() -> float:
    path = OUT / "ROOTED_STEP_CALIBRATION.json"
    if not path.exists():
        raise RuntimeError("rooted step calibration is missing")
    return float(json.loads(path.read_text())["eta_root"])


def blank_row(method: Method, subject: int, seed: int, epoch: int, eta_root: float) -> dict[str, object]:
    row = {column: math.nan for column in EPOCH_COLUMNS}
    row.update(
        dataset=DATASET, method=method.name, sampling=method.sampling,
        aggregation=method.aggregation, update=method.update, subject=subject,
        seed=seed, epoch=epoch, epochs=EPOCHS, N_proj=N_PROJ,
        sigma=method.sigma, d=22, m=253,
        cumulative_direct_projection_count=N_PROJ * epoch,
        cumulative_direction_draw_count=(N_PROJ if method.sampling == "fixed" else N_PROJ * epoch),
        learning_rate=(ETA_POWER if method.update == "raw_power" else eta_root if method.update == "raw_rooted" else math.nan),
        eta_norm=(ETA_NORM if method.update == "normalized_power" else math.nan),
        eta_root=(eta_root if method.update == "raw_rooted" else math.nan),
        nan=True, diverged=True, status="nonfinite_trajectory",
    )
    return row


def run_path(method: Method, subject: int, seed: int) -> Path:
    return OUT / "runs" / method.name / f"seed_{seed}" / f"subject_{subject:02d}.csv"


def diagnostic_path(method: Method, subject: int, seed: int) -> Path:
    return OUT / "diagnostics" / method.name / f"seed_{seed}" / f"subject_{subject:02d}.csv"


def read_typed_csv(path: Path, *, diagnostic: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    strings = {
        "dataset", "method", "sampling", "aggregation", "update", "bank_hash",
    }
    if not diagnostic:
        strings |= {
            "bank_hash_kind", "target_projection_hash", "initial_source_hash",
            "target_hash", "status",
        }
    booleans = set() if diagnostic else {"nan", "diverged"}
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
    method: Method, source: torch.Tensor, target: torch.Tensor, *, subject: int,
    seed: int, eta_root: float, fixed_state: FixedBankState | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source = source.to(device=DEVICE, dtype=DTYPE)
    target = target.to(device=DEVICE, dtype=DTYPE)
    basis = SvecBasis(source.shape[-1], DEVICE, DTYPE)
    parameter = basis.forward(source).clone().requires_grad_(True)
    target_vec = basis.forward(target)
    if method.sampling == "fixed" and fixed_state is None:
        fixed_state = build_fixed_bank_state(basis, target_vec, seed)
    evaluator = LEWEvaluator(target)
    lew0, evaluation_ms = evaluate_independent_lew(evaluator, basis, parameter)
    evaluator.set_baseline(lew0)
    cumulative_evaluation_ms = evaluation_ms
    setup_ms = (
        fixed_state.sampling_ms + fixed_state.target_projection_ms
        if fixed_state is not None and method.sampling == "fixed" else 0.0
    )
    cumulative_optimization_ms = setup_ms
    source_hash = tensor_sha256(parameter, full=True)
    target_hash = tensor_sha256(target_vec, full=True)
    ordered_weights = lognormal_spectral_weights(N_PROJ, SIGMA, DEVICE, DTYPE).detach()
    spectral_stats = distribution_diagnostics(ordered_weights)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    initial = blank_row(method, subject, seed, 0, eta_root)
    initial.update(
        lew=lew0, relative_lew=1.0, lew_reduction_pct=0.0, gap_closure=0.0,
        raw_gradient_norm=math.nan, applied_update_norm=0.0,
        bank_seed=(fixed_state.bank_seed if fixed_state is not None else direction_seed(seed, 0)),
        bank_hash=(fixed_state.bank_hash if fixed_state is not None else ""),
        bank_hash_kind=("full_tensor_sha256" if fixed_state is not None else ""),
        target_projection_hash=(fixed_state.target_projection_hash if fixed_state is not None else ""),
        initial_source_hash=source_hash, target_hash=target_hash,
        direction_sampling_ms=0.0, source_projection_ms=0.0,
        target_projection_ms=0.0, wasserstein_1d_ms=0.0,
        sorting_aggregation_ms=0.0, backward_ms=0.0,
        optimizer_update_ms=0.0, total_epoch_ms=0.0,
        one_time_bank_sampling_ms=(fixed_state.sampling_ms if fixed_state is not None else 0.0),
        one_time_target_projection_ms=(fixed_state.target_projection_ms if fixed_state is not None else 0.0),
        evaluation_ms=evaluation_ms, cumulative_optimization_ms=cumulative_optimization_ms,
        cumulative_evaluation_ms=cumulative_evaluation_ms,
        cumulative_direct_projection_count=0,
        cumulative_direction_draw_count=(N_PROJ if fixed_state is not None else 0),
        nan=False, diverged=False, status="initial",
    )
    if method.aggregation == "spectral":
        initial.update(spectral_stats)
    rows.append(initial)
    finite = True
    for zero_epoch in range(EPOCHS):
        stages = StageTimes()
        (
            directions, projected_target, sampled_seed, fingerprint,
            fingerprint_kind, target_projection_hash,
            stages.direction_sampling_ms, stages.target_projection_ms,
        ) = epoch_bank(method, basis, target_vec, seed, zero_epoch, fixed_state)
        projected_source, stages.source_projection_ms = timed(
            DEVICE, lambda: parameter @ directions.T
        )
        h, stages.wasserstein_1d_ms = timed(
            DEVICE, lambda: w2_squared_per_direction(projected_source.T, projected_target.T)
        )
        (power, weights), stages.sorting_aggregation_ms = timed(
            DEVICE,
            lambda: aggregate_directional_costs(
                h, method.aggregation,
                ordered_weights if method.aggregation == "spectral" else None,
            ),
        )
        objective = objective_for_update(power, method.update)
        if zero_epoch in DIAGNOSTIC_EPOCHS:
            uniform_norm, spectral_norm, cosine, diagnostic_ms = paired_gradient_diagnostic(
                parameter, directions, projected_target, method.update, ordered_weights
            )
        else:
            uniform_norm = spectral_norm = cosine = diagnostic_ms = math.nan
        _, stages.backward_ms = timed(DEVICE, objective.backward)
        gradient_norm = float(parameter.grad.norm())

        def apply_update() -> torch.Tensor:
            update_tensor = raw_update(parameter.grad, method.update, eta_root)
            with torch.no_grad():
                parameter.add_(update_tensor)
            parameter.grad = None
            return update_tensor

        update_tensor, stages.optimizer_update_ms = timed(DEVICE, apply_update)
        update_norm = float(update_tensor.norm())
        epoch_ms = stages.total_epoch_ms()
        cumulative_optimization_ms += epoch_ms
        epoch = zero_epoch + 1
        finite = bool(
            torch.isfinite(parameter).all() and torch.isfinite(power)
            and math.isfinite(gradient_norm) and math.isfinite(update_norm)
        )
        lew = relative = reduction = closure = math.nan
        current_evaluation_ms = 0.0
        diverged = not finite
        if finite:
            lew, current_evaluation_ms = evaluate_independent_lew(evaluator, basis, parameter)
            cumulative_evaluation_ms += current_evaluation_ms
            relative = lew / lew0
            reduction = 100.0 * (lew0 - lew) / lew0
            closure = evaluator.closed_pct(lew)
            diverged = evaluator.diverged(lew)
        if zero_epoch == 0:
            rows[0]["objective_power"] = float(power.detach())
            rows[0]["rooted_distance"] = float(rooted_objective(power.detach()))
            rows[0]["mean_h"] = float(h.detach().mean())
            rows[0]["std_h"] = float(h.detach().std(unbiased=False))
            rows[0]["max_h"] = float(h.detach().max())
            rows[0]["min_h"] = float(h.detach().min())
            rows[0]["bank_seed"] = sampled_seed
            rows[0]["bank_hash"] = fingerprint
            rows[0]["bank_hash_kind"] = fingerprint_kind
            rows[0]["target_projection_hash"] = target_projection_hash
        stats = distribution_diagnostics(weights)
        row = {
            "dataset": DATASET, "method": method.name, "sampling": method.sampling,
            "aggregation": method.aggregation, "update": method.update,
            "subject": subject, "seed": seed, "epoch": epoch, "epochs": EPOCHS,
            "N_proj": method.N_proj, "sigma": method.sigma, "d": basis.d, "m": basis.m,
            "objective_power": float(power.detach()),
            "rooted_distance": float(rooted_objective(power.detach())),
            "lew": lew, "relative_lew": relative, "lew_reduction_pct": reduction,
            "gap_closure": closure, "raw_gradient_norm": gradient_norm,
            "applied_update_norm": update_norm, "mean_h": float(h.detach().mean()),
            "std_h": float(h.detach().std(unbiased=False)), "max_h": float(h.detach().max()),
            "min_h": float(h.detach().min()),
            **(stats if method.aggregation == "spectral" else {
                key: math.nan for key in spectral_stats
            }),
            "bank_seed": sampled_seed, "bank_hash": fingerprint,
            "bank_hash_kind": fingerprint_kind,
            "target_projection_hash": target_projection_hash,
            "initial_source_hash": source_hash, "target_hash": target_hash,
            "direction_sampling_ms": stages.direction_sampling_ms,
            "source_projection_ms": stages.source_projection_ms,
            "target_projection_ms": stages.target_projection_ms,
            "wasserstein_1d_ms": stages.wasserstein_1d_ms,
            "sorting_aggregation_ms": stages.sorting_aggregation_ms,
            "backward_ms": stages.backward_ms,
            "optimizer_update_ms": stages.optimizer_update_ms,
            "total_epoch_ms": epoch_ms,
            "one_time_bank_sampling_ms": (fixed_state.sampling_ms if fixed_state is not None else 0.0),
            "one_time_target_projection_ms": (fixed_state.target_projection_ms if fixed_state is not None else 0.0),
            "evaluation_ms": current_evaluation_ms,
            "cumulative_optimization_ms": cumulative_optimization_ms,
            "cumulative_evaluation_ms": cumulative_evaluation_ms,
            "cumulative_direct_projection_count": N_PROJ * epoch,
            "cumulative_direction_draw_count": (N_PROJ if method.sampling == "fixed" else N_PROJ * epoch),
            "learning_rate": (ETA_POWER if method.update == "raw_power" else eta_root if method.update == "raw_rooted" else math.nan),
            "eta_norm": (ETA_NORM if method.update == "normalized_power" else math.nan),
            "eta_root": (eta_root if method.update == "raw_rooted" else math.nan),
            "nan": not finite, "diverged": bool(diverged),
            "status": "ok" if finite else "nonfinite",
        }
        rows.append(row)
        diagnostics.append({
            "dataset": DATASET, "method": method.name, "sampling": method.sampling,
            "aggregation": method.aggregation, "update": method.update,
            "subject": subject, "seed": seed, "epoch": zero_epoch,
            "raw_gradient_norm": gradient_norm, "applied_update_norm": update_norm,
            "uniform_spectral_cosine": cosine,
            "paired_uniform_gradient_norm": uniform_norm,
            "paired_spectral_gradient_norm": spectral_norm,
            "bank_seed": sampled_seed, "bank_hash": fingerprint,
            **(spectral_stats if method.aggregation == "spectral" else {
                key: math.nan for key in spectral_stats
            }),
            "diagnostic_ms": diagnostic_ms,
        })
        if not finite:
            for later in range(epoch + 1, EPOCHS + 1):
                rows.append(blank_row(method, subject, seed, later, eta_root))
            break
    # State-epoch 500 same-state diagnostic; no update and no training-clock charge.
    if finite:
        (
            directions, projected_target, sampled_seed, fingerprint, _, _, _, _,
        ) = epoch_bank(method, basis, target_vec, seed, EPOCHS, fixed_state)
        uniform_norm, spectral_norm, cosine, diagnostic_ms = paired_gradient_diagnostic(
            parameter, directions, projected_target, method.update, ordered_weights
        )
        diagnostics.append({
            "dataset": DATASET, "method": method.name, "sampling": method.sampling,
            "aggregation": method.aggregation, "update": method.update,
            "subject": subject, "seed": seed, "epoch": EPOCHS,
            "raw_gradient_norm": math.nan, "applied_update_norm": math.nan,
            "uniform_spectral_cosine": cosine,
            "paired_uniform_gradient_norm": uniform_norm,
            "paired_spectral_gradient_norm": spectral_norm,
            "bank_seed": sampled_seed, "bank_hash": fingerprint,
            **(spectral_stats if method.aggregation == "spectral" else {
                key: math.nan for key in spectral_stats
            }),
            "diagnostic_ms": diagnostic_ms,
        })
    frame = pd.DataFrame(rows)[EPOCH_COLUMNS]
    diagnostic_frame = pd.DataFrame(diagnostics)[DIAGNOSTIC_COLUMNS]
    evaluated = frame[np.isfinite(frame.lew)]
    metadata = {
        "dataset": DATASET, "method": method.name, "sampling": method.sampling,
        "aggregation": method.aggregation, "update": method.update,
        "subject": subject, "seed": seed, "epochs": EPOCHS, "N_proj": N_PROJ,
        "sigma": method.sigma, "rows": len(frame),
        "diagnostic_rows": len(diagnostic_frame), "lew_initial": lew0,
        "lew_final": float(evaluated.lew.iloc[-1]) if not evaluated.empty else math.nan,
        "diverged": bool(frame.diverged.fillna(False).any()),
        "nan": bool(frame["nan"].fillna(False).any()),
        "optimization_ms": float(frame.cumulative_optimization_ms.dropna().iloc[-1]),
        "evaluation_ms": float(frame.cumulative_evaluation_ms.dropna().iloc[-1]),
        "status": "ok" if finite else "nonfinite",
    }
    return frame, diagnostic_frame, metadata


def run_complete(path: Path, diagnostics: Path) -> bool:
    if not path.exists() or not diagnostics.exists():
        return False
    try:
        frame = read_typed_csv(path)
        diagnostic = read_typed_csv(diagnostics, diagnostic=True)
        return (
            len(frame) == EPOCHS + 1 and int(frame.epoch.iloc[-1]) == EPOCHS
            and set(DIAGNOSTIC_EPOCHS).issubset(set(diagnostic.epoch.dropna().astype(int)))
        )
    except Exception:
        return False


def metadata_from_frame(path: Path, diagnostics: Path) -> dict[str, object]:
    frame = read_typed_csv(path)
    evaluated = frame[np.isfinite(frame.lew)]
    return {
        "dataset": DATASET, "method": str(frame.method.iloc[0]),
        "sampling": str(frame.sampling.iloc[0]), "aggregation": str(frame.aggregation.iloc[0]),
        "update": str(frame.update.iloc[0]), "subject": int(frame.subject.iloc[0]),
        "seed": int(frame.seed.iloc[0]), "epochs": EPOCHS, "N_proj": N_PROJ,
        "sigma": float(frame.sigma.iloc[0]), "rows": len(frame),
        "diagnostic_rows": len(read_typed_csv(diagnostics, diagnostic=True)),
        "lew_initial": float(evaluated.lew.iloc[0]),
        "lew_final": float(evaluated.lew.iloc[-1]),
        "diverged": bool(frame.diverged.fillna(False).any()),
        "nan": bool(frame["nan"].fillna(False).any()),
        "optimization_ms": float(frame.cumulative_optimization_ms.dropna().iloc[-1]),
        "evaluation_ms": float(frame.cumulative_evaluation_ms.dropna().iloc[-1]),
        "status": "cached_complete",
    }


def execute_grid(*, rerun: bool = False) -> list[dict[str, object]]:
    eta_root = read_eta_root()
    records: list[dict[str, object]] = []
    manifest_path = OUT / "MANIFEST_TRAJECTORIES.json"
    total = len(METHODS) * len(SUBJECTS) * len(SEEDS)
    index = 0
    for subject in SUBJECTS:
        source, target, meta = load_cached_subject(DATASET, subject, DEVICE)
        basis = SvecBasis(meta["d"], DEVICE, DTYPE)
        target_vec = basis.forward(target)
        for seed in SEEDS:
            fixed_state = build_fixed_bank_state(basis, target_vec, seed)
            for method in METHODS:
                index += 1
                path = run_path(method, subject, seed)
                diag_path = diagnostic_path(method, subject, seed)
                try:
                    if rerun or not run_complete(path, diag_path):
                        frame, diagnostic, metadata = train_one(
                            method, source, target, subject=subject, seed=seed,
                            eta_root=eta_root,
                            fixed_state=(fixed_state if method.sampling == "fixed" else None),
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        diag_path.parent.mkdir(parents=True, exist_ok=True)
                        frame.to_csv(path, index=False)
                        diagnostic.to_csv(diag_path, index=False)
                    else:
                        metadata = metadata_from_frame(path, diag_path)
                    record = {
                        **metadata, "run_csv": str(path.relative_to(OUT)),
                        "diagnostic_csv": str(diag_path.relative_to(OUT)), "error": None,
                    }
                    print(
                        f"[{index:03d}/{total:03d}] s{subject:02d} seed={seed} "
                        f"{method.name:42s} LEW {record['lew_initial']:.4f}->{record['lew_final']:.4f}",
                        flush=True,
                    )
                except Exception as exc:
                    log_path = OUT / "logs" / f"{method.name}_seed{seed}_s{subject:02d}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(traceback.format_exc())
                    record = {
                        "dataset": DATASET, "method": method.name,
                        "sampling": method.sampling, "aggregation": method.aggregation,
                        "update": method.update, "subject": subject, "seed": seed,
                        "epochs": EPOCHS, "N_proj": N_PROJ, "sigma": method.sigma,
                        "status": "error", "error": f"{type(exc).__name__}: {exc}",
                        "run_csv": str(path.relative_to(OUT)),
                        "diagnostic_csv": str(diag_path.relative_to(OUT)),
                    }
                    print(f"[ERROR] {record['error']}", file=sys.stderr, flush=True)
                records.append(record)
                dump_json(manifest_path, records)
        del source, target, target_vec
        torch.cuda.empty_cache()
    return records


def relative_lew_auc(group: pd.DataFrame) -> float:
    evaluated = group[np.isfinite(group.lew)].sort_values("epoch")
    if len(evaluated) != EPOCHS + 1 or int(evaluated.epoch.iloc[-1]) != EPOCHS:
        return math.inf
    return float(np.trapezoid(evaluated.relative_lew, evaluated.epoch) / EPOCHS)


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset", "method", "sampling", "aggregation", "update", "subject", "seed",
        "epochs", "N_proj", "sigma", "d", "m",
    ]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=False):
        group = group.sort_values("epoch")
        evaluated = group[np.isfinite(group.lew)]
        objectives = group[np.isfinite(group.objective_power)]
        initial_lew = float(evaluated.lew.iloc[0])
        final_lew = float(evaluated.lew.iloc[-1])
        initial_objective = float(objectives.objective_power.iloc[0]) if not objectives.empty else math.nan
        final_objective = float(objectives.objective_power.iloc[-1]) if not objectives.empty else math.nan
        row = dict(zip(keys, key))
        row.update(
            lew_initial=initial_lew, lew_final=final_lew,
            relative_lew_auc=relative_lew_auc(group),
            final_relative_lew=final_lew / initial_lew,
            lew_reduction_pct=100.0 * (initial_lew - final_lew) / initial_lew,
            initial_objective_power=initial_objective,
            final_objective_power=final_objective,
            training_objective_reduction_pct=(
                100.0 * (initial_objective - final_objective) / initial_objective
                if initial_objective != 0.0 and math.isfinite(initial_objective) else math.nan
            ),
            optimization_ms=float(group.cumulative_optimization_ms.dropna().iloc[-1]),
            evaluation_ms=float(group.cumulative_evaluation_ms.dropna().iloc[-1]),
            mean_gradient_norm=float(group.raw_gradient_norm.mean()),
            mean_update_norm=float(group.applied_update_norm.mean()),
            diverged=bool(group.diverged.fillna(False).any()),
            nan=bool(group["nan"].fillna(False).any()),
        )
        row["overfit_gap"] = row["training_objective_reduction_pct"] - row["lew_reduction_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_results(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "sampling", "aggregation", "update", "subject", "seed"]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=False):
        group = group.sort_values("epoch")
        for threshold in THRESHOLDS:
            reached = group[np.isfinite(group.relative_lew) & (group.relative_lew <= threshold)]
            first = reached.iloc[0] if not reached.empty else None
            row = dict(zip(keys, key))
            row.update(
                threshold_relative_lew=threshold,
                required_reduction_pct=100.0 * (1.0 - threshold),
                reached=first is not None,
                first_reach_epoch=(int(first.epoch) if first is not None else math.nan),
                first_reach_optimization_ms=(float(first.cumulative_optimization_ms) if first is not None else math.nan),
                first_reach_total_projection_count=(int(first.cumulative_direct_projection_count) if first is not None else math.nan),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def paired_quality_reach(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_keys = ["sampling", "update", "subject", "seed"]
    for key, group in frame.groupby(group_keys, sort=False):
        sampling, update, subject, seed = key
        uniform = group[group.aggregation == "uniform"].sort_values("epoch")
        spectral = group[group.aggregation == "spectral"].sort_values("epoch")
        quality = float(uniform[uniform.epoch == EPOCHS].lew.iloc[0])
        values: dict[str, dict[str, float]] = {}
        for label, candidate in (("uniform", uniform), ("spectral", spectral)):
            reached = candidate[np.isfinite(candidate.lew) & (candidate.lew <= quality)]
            values[label] = {
                "epoch": float(reached.epoch.iloc[0]) if not reached.empty else math.nan,
                "ms": float(reached.cumulative_optimization_ms.iloc[0]) if not reached.empty else math.nan,
            }
        rows.append({
            "reference_type": "paired_uniform_final", "sampling": sampling,
            "update": update, "subject": subject, "seed": seed,
            "quality_lew": quality,
            "uniform_first_reach_epoch": values["uniform"]["epoch"],
            "spectral_first_reach_epoch": values["spectral"]["epoch"],
            "uniform_first_reach_optimization_ms": values["uniform"]["ms"],
            "spectral_first_reach_optimization_ms": values["spectral"]["ms"],
            "spectral_epoch_speedup": values["uniform"]["epoch"] - values["spectral"]["epoch"],
            "spectral_wallclock_speedup_ms": values["uniform"]["ms"] - values["spectral"]["ms"],
        })
    # Optional cross-formulation reference, explicitly descriptive.
    for (subject, seed), group in frame.groupby(["subject", "seed"], sort=False):
        finals = group[(group.aggregation == "uniform") & (group.epoch == EPOCHS)]
        quality = float(finals.lew.min())
        for method, candidate in group.groupby("method", sort=False):
            candidate = candidate.sort_values("epoch")
            reached = candidate[np.isfinite(candidate.lew) & (candidate.lew <= quality)]
            rows.append({
                "reference_type": "best_uniform_final_descriptive",
                "sampling": str(candidate.sampling.iloc[0]),
                "update": str(candidate.update.iloc[0]), "subject": subject, "seed": seed,
                "method": method, "quality_lew": quality,
                "method_first_reach_epoch": (float(reached.epoch.iloc[0]) if not reached.empty else math.nan),
                "method_first_reach_optimization_ms": (
                    float(reached.cumulative_optimization_ms.iloc[0]) if not reached.empty else math.nan
                ),
            })
    return pd.DataFrame(rows)


def exact_first_hit(relative: Iterable[float], threshold: float) -> int | None:
    for epoch, value in enumerate(relative):
        if math.isfinite(float(value)) and float(value) <= threshold:
            return epoch
    return None


def factorial_effect_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = ["subject", "seed"]
    for sampling in ("fixed", "resampled"):
        for update in ("normalized_power", "raw_power", "raw_rooted"):
            cell = summary[(summary.sampling == sampling) & (summary.update == update)]
            pivot = cell.pivot(index=index, columns="aggregation", values="relative_lew_auc")
            for (subject, seed), values in pivot.iterrows():
                rows.append({
                    "effect": "spectral", "sampling": sampling, "update": update,
                    "aggregation": "spectral-minus-uniform", "subject": subject, "seed": seed,
                    "difference": float(values["spectral"] - values["uniform"]),
                })
    for aggregation in ("uniform", "spectral"):
        for update in ("normalized_power", "raw_power", "raw_rooted"):
            cell = summary[(summary.aggregation == aggregation) & (summary.update == update)]
            pivot = cell.pivot(index=index, columns="sampling", values="relative_lew_auc")
            for (subject, seed), values in pivot.iterrows():
                rows.append({
                    "effect": "resampling", "sampling": "resampled-minus-fixed",
                    "update": update, "aggregation": aggregation, "subject": subject,
                    "seed": seed, "difference": float(values["resampled"] - values["fixed"]),
                })
    # Per-run two-factor contrasts. Three-way structure remains visible in the labels.
    for update in ("normalized_power", "raw_power", "raw_rooted"):
        cell = summary[summary.update == update]
        pivot = cell.pivot(index=index, columns=["sampling", "aggregation"], values="relative_lew_auc")
        for (subject, seed), values in pivot.iterrows():
            interaction = (
                values[("resampled", "spectral")] - values[("resampled", "uniform")]
                - values[("fixed", "spectral")] + values[("fixed", "uniform")]
            )
            rows.append({
                "effect": "spectral_x_sampling", "sampling": "interaction",
                "update": update, "aggregation": "interaction", "subject": subject,
                "seed": seed, "difference": float(interaction),
            })
    return pd.DataFrame(rows)


def subject_results(summary: pd.DataFrame) -> pd.DataFrame:
    effects = factorial_effect_rows(summary)
    aggregate_rows: list[dict[str, object]] = []
    group_columns = ["effect", "sampling", "update", "aggregation"]
    for key, group in effects.groupby(group_columns, sort=False):
        base = dict(zip(group_columns, key))
        for level, level_group in (
            ("run", group),
            ("seed_mean", group.groupby("seed", as_index=False).difference.mean()),
            ("subject_mean", group.groupby("subject", as_index=False).difference.mean()),
        ):
            for _, row in level_group.iterrows():
                aggregate_rows.append({
                    **base, "level": level,
                    "subject": row.get("subject", math.nan), "seed": row.get("seed", math.nan),
                    "difference": float(row.difference), "mean": math.nan,
                    "median": math.nan, "sd": math.nan,
                })
        aggregate_rows.append({
            **base, "level": "grand", "subject": math.nan, "seed": math.nan,
            "difference": math.nan, "mean": float(group.difference.mean()),
            "median": float(group.difference.median()),
            "sd": float(group.difference.std(ddof=1)),
        })
    return pd.DataFrame(aggregate_rows)


def load_all_frames() -> pd.DataFrame:
    paths = sorted((OUT / "runs").glob("*/seed_*/subject_*.csv"))
    if len(paths) != 108:
        raise RuntimeError(f"expected 108 run CSVs, found {len(paths)}")
    return pd.concat([read_typed_csv(path) for path in paths], ignore_index=True)


def load_all_diagnostics() -> pd.DataFrame:
    paths = sorted((OUT / "diagnostics").glob("*/seed_*/subject_*.csv"))
    if len(paths) != 108:
        raise RuntimeError(f"expected 108 diagnostic CSVs, found {len(paths)}")
    return pd.concat([read_typed_csv(path, diagnostic=True) for path in paths], ignore_index=True)


def fixed_overfit(frame: pd.DataFrame) -> pd.DataFrame:
    fixed = frame[frame.sampling == "fixed"].copy()
    fixed["initial_training_objective"] = fixed.groupby(
        ["method", "subject", "seed"]
    ).objective_power.transform("first")
    fixed["training_loss_reduction_pct"] = 100.0 * (
        fixed.initial_training_objective - fixed.objective_power
    ) / fixed.initial_training_objective
    fixed["independent_LEW_reduction_pct"] = fixed.lew_reduction_pct
    fixed["overfit_gap"] = (
        fixed.training_loss_reduction_pct - fixed.independent_LEW_reduction_pct
    )
    return fixed[[
        "method", "aggregation", "update", "subject", "seed", "epoch",
        "objective_power", "lew", "training_loss_reduction_pct",
        "independent_LEW_reduction_pct", "overfit_gap",
    ]]


def timing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "direction_sampling_ms", "source_projection_ms", "target_projection_ms",
        "wasserstein_1d_ms", "sorting_aggregation_ms", "backward_ms",
        "optimizer_update_ms", "total_epoch_ms", "evaluation_ms",
    ]
    trained = frame[frame.epoch > 0]
    rows: list[dict[str, object]] = []
    keys = ["method", "sampling", "aggregation", "update", "subject", "seed"]
    for key, group in trained.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        for column in columns:
            row[f"sum_{column}"] = float(group[column].sum())
            row[f"mean_{column}"] = float(group[column].mean())
        first = frame[
            (frame.method == key[0]) & (frame.subject == key[4])
            & (frame.seed == key[5]) & (frame.epoch == 0)
        ].iloc[0]
        row["one_time_bank_sampling_ms"] = float(first.one_time_bank_sampling_ms)
        row["one_time_target_projection_ms"] = float(first.one_time_target_projection_ms)
        row["cumulative_optimization_ms"] = float(group.cumulative_optimization_ms.iloc[-1])
        row["cumulative_evaluation_ms"] = float(group.cumulative_evaluation_ms.iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def bank_audit(frame: pd.DataFrame) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            cell = frame[(frame.subject == subject) & (frame.seed == seed)]
            fixed = cell[(cell.sampling == "fixed") & (cell.epoch > 0)]
            resampled = cell[(cell.sampling == "resampled") & (cell.epoch > 0)]
            fixed_hashes = fixed.groupby("method").bank_hash.apply(lambda x: sorted(set(x)))
            fixed_common = {values[0] for values in fixed_hashes if len(values) == 1}
            epoch_common = all(
                len(set(group.bank_seed.astype(int))) == 1
                for _, group in resampled.groupby("epoch")
            )
            resampled_changes = all(
                group.bank_seed.nunique() == EPOCHS
                for _, group in resampled.groupby("method")
            )
            fixed_epoch0 = int(fixed[fixed.epoch == 1].bank_seed.iloc[0])
            resampled_epoch0 = int(resampled[resampled.epoch == 1].bank_seed.iloc[0])
            fixed_epoch0_hash = str(fixed[fixed.epoch == 1].bank_hash.iloc[0])
            resampled_epoch0_hash = str(resampled[resampled.epoch == 1].bank_hash.iloc[0])
            fixed_target_zero = bool((fixed.target_projection_ms == 0.0).all())
            resampled_target_positive = bool((resampled.target_projection_ms > 0.0).all())
            checks.append({
                "subject": subject, "seed": seed,
                "fixed_six_share_one_hash": len(fixed_common) == 1,
                "fixed_hash_constant_all_epochs": all(len(values) == 1 for values in fixed_hashes),
                "resampled_six_share_each_epoch_seed": epoch_common,
                "resampled_each_method_changes_all_epochs": resampled_changes,
                "fixed_resampled_epoch0_seed_equal": fixed_epoch0 == resampled_epoch0,
                "fixed_resampled_epoch0_hash_equal": fixed_epoch0_hash == resampled_epoch0_hash,
                "fixed_target_projection_zero_per_epoch": fixed_target_zero,
                "resampled_target_projection_positive_per_epoch": resampled_target_positive,
                "initial_source_hash_count": int(cell.initial_source_hash.replace("", np.nan).dropna().nunique()),
                "target_hash_count": int(cell.target_hash.replace("", np.nan).dropna().nunique()),
            })
    passed = all(
        row[check]
        for row in checks
        for check in (
            "fixed_six_share_one_hash", "fixed_hash_constant_all_epochs",
            "resampled_six_share_each_epoch_seed",
            "resampled_each_method_changes_all_epochs",
            "fixed_resampled_epoch0_seed_equal",
            "fixed_resampled_epoch0_hash_equal",
            "fixed_target_projection_zero_per_epoch",
            "resampled_target_projection_positive_per_epoch",
        )
    ) and all(row["initial_source_hash_count"] == 1 and row["target_hash_count"] == 1 for row in checks)
    return {
        "passed": passed, "checks": checks,
        "direction_seed_rule": "seed + epoch*(epoch+1)//2; fixed uses epoch=0",
        "hash_note": (
            "fixed and resampled epoch 0 use full tensor SHA256; later "
            "resampled epochs use deterministic three-row SHA256"
        ),
    }


def classify_spectral_effect(summary: pd.DataFrame, sampling: str, update: str) -> str:
    cell = summary[(summary.sampling == sampling) & (summary.update == update)]
    pivot = cell.pivot(index=["subject", "seed"], columns="aggregation", values="relative_lew_auc")
    differences = pivot.spectral - pivot.uniform
    subject_means = differences.groupby(level="subject").mean()
    if float(differences.mean()) < 0.0 and int((subject_means < 0.0).sum()) >= 2:
        return "IMPROVE"
    if float(differences.mean()) > 0.0 and int((subject_means > 0.0).sum()) >= 2:
        return "WORSE"
    return "NULL"


def fastest_threshold(thresholds: pd.DataFrame, threshold: float) -> str:
    cell = thresholds[
        np.isclose(thresholds.threshold_relative_lew, threshold) & thresholds.reached
    ]
    if cell.empty:
        return "NONE"
    means = cell.groupby("method").first_reach_epoch.mean().sort_values()
    fastest = means[means == means.iloc[0]].index.tolist()
    return " / ".join(fastest)


def plot_trajectory(frame: pd.DataFrame, sampling: str, x: str, filename: str) -> None:
    subset = frame[frame.sampling == sampling]
    plt.figure(figsize=(10, 6))
    for method, group in subset.groupby("method", sort=False):
        curve = group.groupby("epoch")[[x, "relative_lew"]].mean().sort_index()
        plt.plot(curve[x], curve.relative_lew, label=method.replace(f"{sampling}_", ""), linewidth=1.5)
    plt.xlabel("epoch" if x == "epoch" else "cumulative optimization ms")
    plt.ylabel("relative independent LEW")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


def plot_effects(subjects: pd.DataFrame, effect: str, filename: str) -> None:
    grand = subjects[(subjects.level == "grand") & (subjects.effect == effect)]
    labels = [f"{row.sampling}\n{row.update}\n{row.aggregation}" for _, row in grand.iterrows()]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(grand)), grand["mean"])
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(np.arange(len(grand)), labels, rotation=25, ha="right", fontsize=7)
    plt.ylabel("relative LEW AUC difference (lower favors first-named factor)")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


def plot_thresholds(thresholds: pd.DataFrame, value: str, filename: str) -> None:
    grouped = thresholds.groupby(["method", "threshold_relative_lew"])[value].mean().unstack()
    plt.figure(figsize=(12, 6))
    grouped.plot(kind="bar", ax=plt.gca(), width=0.8)
    plt.ylabel(value.replace("_", " "))
    plt.xticks(rotation=35, ha="right", fontsize=7)
    plt.legend(title="relative LEW threshold")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


def plot_metric(frame: pd.DataFrame, column: str, filename: str) -> None:
    plt.figure(figsize=(10, 6))
    for method, group in frame[frame.epoch > 0].groupby("method", sort=False):
        curve = group.groupby("epoch")[column].mean()
        plt.plot(curve.index, curve.values, label=method, linewidth=1.2)
    plt.xlabel("epoch")
    plt.ylabel(column.replace("_", " "))
    plt.yscale("log" if column in {"raw_gradient_norm", "applied_update_norm"} else "linear")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


def make_plots(
    frame: pd.DataFrame, thresholds: pd.DataFrame, paired: pd.DataFrame,
    subjects: pd.DataFrame, diagnostics: pd.DataFrame, overfit: pd.DataFrame,
) -> None:
    plot_trajectory(frame, "fixed", "epoch", "fig_fixed_lew_vs_epoch.png")
    plot_trajectory(frame, "resampled", "epoch", "fig_resampled_lew_vs_epoch.png")
    plot_trajectory(frame, "fixed", "cumulative_optimization_ms", "fig_fixed_lew_vs_wallclock.png")
    plot_trajectory(frame, "resampled", "cumulative_optimization_ms", "fig_resampled_lew_vs_wallclock.png")
    plot_effects(subjects, "spectral", "fig_spectral_effect_by_update.png")
    plot_effects(subjects, "resampling", "fig_sampling_effect_by_update.png")
    plot_thresholds(thresholds, "first_reach_epoch", "fig_epoch_to_threshold.png")
    plot_thresholds(thresholds, "first_reach_optimization_ms", "fig_wallclock_to_threshold.png")
    paired_primary = paired[paired.reference_type == "paired_uniform_final"]
    plt.figure(figsize=(9, 5))
    speed = paired_primary.groupby(["sampling", "update"]).spectral_epoch_speedup.mean()
    speed.plot(kind="bar", ax=plt.gca())
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("spectral epoch speedup to paired uniform final LEW")
    plt.tight_layout()
    plt.savefig(OUT / "fig_paired_uniform_quality_reach.png", dpi=180)
    plt.close()
    plot_metric(frame, "raw_gradient_norm", "fig_gradient_norm.png")
    plot_metric(frame, "applied_update_norm", "fig_update_norm.png")
    selected = diagnostics[np.isfinite(diagnostics.uniform_spectral_cosine)]
    plt.figure(figsize=(9, 5))
    for (sampling, update), group in selected.groupby(["sampling", "update"]):
        curve = group.groupby("epoch").uniform_spectral_cosine.mean()
        plt.plot(curve.index, curve.values, marker="o", label=f"{sampling} {update}")
    plt.xlabel("state epoch")
    plt.ylabel("uniform/spectral gradient cosine")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "fig_gradient_cosine.png", dpi=180)
    plt.close()
    plt.figure(figsize=(9, 6))
    for method, group in overfit.groupby("method", sort=False):
        curve = group.groupby("epoch")[["training_loss_reduction_pct", "independent_LEW_reduction_pct"]].mean()
        plt.plot(curve.independent_LEW_reduction_pct, curve.training_loss_reduction_pct, label=method)
    plt.xlabel("independent LEW reduction %")
    plt.ylabel("fixed-bank training objective reduction %")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUT / "fig_fixed_training_vs_independent_lew.png", dpi=180)
    plt.close()


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
                values.append("NA" if not math.isfinite(float(value)) else f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def key_interpretation_table(
    summary: pd.DataFrame, thresholds: pd.DataFrame, paired: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    paired = paired[paired.reference_type == "paired_uniform_final"]
    for sampling in ("fixed", "resampled"):
        for update in ("normalized_power", "raw_power", "raw_rooted"):
            cell = summary[(summary.sampling == sampling) & (summary.update == update)]
            auc = cell.groupby("aggregation").relative_lew_auc.mean()
            threshold_summary = {}
            for threshold, label in ((0.8, "20pct"), (0.7, "30pct")):
                reach = thresholds[
                    (thresholds.sampling == sampling) & (thresholds.update == update)
                    & np.isclose(thresholds.threshold_relative_lew, threshold)
                ]
                means = reach.groupby("aggregation").first_reach_epoch.mean()
                threshold_summary[label] = (
                    float(means.get("uniform", math.nan) - means.get("spectral", math.nan))
                    if "uniform" in means and "spectral" in means else math.nan
                )
            pair_speed = paired[(paired.sampling == sampling) & (paired.update == update)]
            rows.append({
                "Sampling": sampling.title(), "Update": update,
                "Uniform AUC": float(auc["uniform"]),
                "Spectral AUC": float(auc["spectral"]),
                "Delta": float(auc["spectral"] - auc["uniform"]),
                "20% threshold speedup (epochs)": threshold_summary["20pct"],
                "30% threshold speedup (epochs)": threshold_summary["30pct"],
                "Paired-final speedup (epochs)": float(pair_speed.spectral_epoch_speedup.mean()),
            })
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame, thresholds: pd.DataFrame, paired: pd.DataFrame,
    subjects: pd.DataFrame, timing: pd.DataFrame, overfit: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    eta_root = read_eta_root()
    classes = {
        (sampling, update): classify_spectral_effect(summary, sampling, update)
        for sampling in ("fixed", "resampled")
        for update in ("normalized_power", "raw_power", "raw_rooted")
    }
    fastest20 = fastest_threshold(thresholds, 0.8)
    fastest30 = fastest_threshold(thresholds, 0.7)
    divergence = int((summary.diverged | summary["nan"]).sum())
    key = key_interpretation_table(summary, thresholds, paired)
    core = summary.groupby(["sampling", "aggregation", "update"], as_index=False).agg(
        relative_lew_auc=("relative_lew_auc", "mean"),
        final_lew=("lew_final", "mean"),
        lew_reduction_pct=("lew_reduction_pct", "mean"),
        optimization_ms=("optimization_ms", "mean"),
        divergence_count=("diverged", "sum"),
    )
    gradient = diagnostics[np.isfinite(diagnostics.uniform_spectral_cosine)].groupby(
        ["sampling", "update", "epoch"], as_index=False
    ).uniform_spectral_cosine.mean()
    fixed_final = overfit[overfit.epoch == EPOCHS].groupby(
        ["aggregation", "update"], as_index=False
    )[["training_loss_reduction_pct", "independent_LEW_reduction_pct", "overfit_gap"]].mean()
    text = f"""- regression tests: PASS
- N_proj: {N_PROJ}
- sigma: {SIGMA}
- normalized eta_norm: {ETA_NORM:.15g}
- raw-power LR: {ETA_POWER:g}
- raw-rooted eta_root: {eta_root:.15g}
- fixed normalized spectral effect: {classes[("fixed", "normalized_power")]}
- fixed raw-power spectral effect: {classes[("fixed", "raw_power")]}
- fixed raw-rooted spectral effect: {classes[("fixed", "raw_rooted")]}
- resampled normalized spectral effect: {classes[("resampled", "normalized_power")]}
- resampled raw-power spectral effect: {classes[("resampled", "raw_power")]}
- resampled raw-rooted spectral effect: {classes[("resampled", "raw_rooted")]}
- fastest method to 20% LEW reduction: {fastest20}
- fastest method to 30% LEW reduction: {fastest30}
- divergence/NaN trajectories: {divergence}

# Spectral sampling/update factorial audit

## 1. Exact protocol

BNCI2014_001 subjects 1, 3, and 8 were run with seeds 6398, 3654, and 1788 for 500 updates. Every condition used direct SPDSW with `N_proj=500`, `p=2`, and no hierarchy. The complete registered 2 sampling × 2 aggregation × 3 update factorial contains exactly 108 trajectories. Independent exact Log-Euclidean Wasserstein (LEW) was evaluated at every state epoch 0 through 500 and its time was excluded from optimization wall-clock.

Fixed conditions shared one persistent bank and its cached target projections per subject/seed. Resampled conditions shared one deterministic epoch bank sequence. Fixed and resampled conditions used the same epoch-0 bank and only separated when the resampled policy advanced to its next bank.

The rooted objective used `sqrt(F)` with epsilon exactly `{ROOT_EPSILON}`. The normalized step was the frozen audited BNCI value from `{ETA_NORM_SOURCE}`. Raw power used LR 3000. The one rooted step size was calibrated from the epoch-0 uniform baseline before comparative training.

## 2. 12-condition factorial table

{frame_markdown(key)}

Aggregate results:

{frame_markdown(core)}

Negative Delta means spectral has lower (better) relative-LEW AUC. “Threshold speedup” is uniform first-hit epoch minus spectral first-hit epoch; positive favors spectral. Missing thresholds remain NA and are not extrapolated.

## 3. LEW-vs-epoch

All 501 independently evaluated states per trajectory are retained in the per-run CSVs. Relative-LEW AUC is the trapezoidal AUC over exactly epochs 0…500 divided by 500. Fixed-bank results are not interpreted alone as population behavior.

## 4. Threshold-reaching epochs

The preregistered thresholds were relative LEW 0.95, 0.90, 0.80, 0.70, and 0.60. `THRESHOLD_RESULTS.csv` records the first actually observed hit, with no interpolation. The fastest average method at 20% reduction was **{fastest20}**; at 30% reduction it was **{fastest30}**.

## 5. Threshold-reaching wall-clock

Optimization wall-clock includes fixed one-time bank sampling and target projection setup. It excludes all independent LEW evaluation and paired-gradient diagnostic time. `THRESHOLD_RESULTS.csv` supplies both first-hit epoch and cumulative optimization time. `TIMING.csv` separately reports evaluation overhead and each requested optimization component.

## 6. Paired-uniform-quality reach

`PAIRED_QUALITY_REACH.csv` uses each sampling/update/subject/seed uniform epoch-500 LEW as the paired target. Positive spectral speedup means spectral reached that quality earlier. The optional best-uniform-final rows are clearly marked descriptive and were not used for primary claims.

## 7. Gradient/update magnitudes

{frame_markdown(gradient)}

Raw gradient and applied update norms were recorded for every update. Same-state, same-bank uniform/spectral gradient cosines were computed at state epochs 0, 25, 50, 100, 200, 300, 400, and 500 without mutating training state and without charging diagnostic work to optimization time.

## 8. Fixed-bank overfitting

{frame_markdown(fixed_final)}

The fixed-bank objective value attached to each post-update epoch row is the pre-update objective that generated that update; epoch 0 contains the same initial-state objective. The overfit diagnostic therefore compares the registered training objective trajectory with independent LEW at every available state and should be read with that one-update logging convention in mind.

## 9. Sampling interactions

`SUBJECT_RESULTS.csv` contains run-level, seed-mean, subject-mean, and grand-mean spectral and sampling effects plus the spectral × sampling interaction within each update formulation. With three subjects these are descriptive paired effects, not significance tests. Update-formulation dependence is read by comparing the three registered within-sampling spectral contrasts; no formal significance claim is made.

## 10. Interpretation

Spectral weighting changes both gradient direction and magnitude. Normalized results isolate unit-gradient direction, while the two raw formulations retain the complete vector field. A raw spectral gain is not automatically an LR artifact, and a normalized null does not automatically disprove spectral utility.

“EBSW-style” here means only optimizing the rooted distance with its raw gradient under a common outer-flow philosophy. Conventional exponential IS-EBSW differentiates through energy-dependent weights and therefore includes a weight-response gradient term. The lognormal rank weights here are detached and piecewise fixed: within an ordering region `grad F_spec = sum_i w_i grad h_(i)` and there is no `grad(w_i)` term. Rooted spectral is therefore not algebraically identical to exponential EBSW.

Sigma 1 was preregistered, not optimized. No learning-rate sweep was performed. Thresholds not reached are not extrapolated, and fixed-bank outcomes are not treated as population SPDSW evidence.
"""
    (OUT / "REPORT.md").write_text(text)


def analyze() -> dict[str, object]:
    require_prepared()
    frame = load_all_frames()
    diagnostics = load_all_diagnostics()
    summary = summarize_runs(frame)
    thresholds = threshold_results(frame)
    paired = paired_quality_reach(frame)
    subjects = subject_results(summary)
    overfit = fixed_overfit(frame)
    timing = timing_summary(frame)
    summary.to_csv(OUT / "CORE_RESULTS.csv", index=False)
    subjects.to_csv(OUT / "SUBJECT_RESULTS.csv", index=False)
    thresholds.to_csv(OUT / "THRESHOLD_RESULTS.csv", index=False)
    paired.to_csv(OUT / "PAIRED_QUALITY_REACH.csv", index=False)
    diagnostics.to_csv(OUT / "GRADIENT_DIAGNOSTICS.csv", index=False)
    overfit.to_csv(OUT / "FIXED_OVERFIT.csv", index=False)
    timing.to_csv(OUT / "TIMING.csv", index=False)
    audit = bank_audit(frame)
    if not audit["passed"]:
        raise RuntimeError("bank audit failed")
    dump_json(OUT / "BANK_AUDIT.json", audit)
    manifest = json.loads((OUT / "MANIFEST_TRAJECTORIES.json").read_text())
    dump_json(OUT / "RUN_MANIFEST.json", manifest)
    make_plots(frame, thresholds, paired, subjects, diagnostics, overfit)
    write_report(summary, thresholds, paired, subjects, timing, overfit, diagnostics)
    verify_frozen()
    return {
        "completed": len(summary),
        "divergence_nan": int((summary.diverged | summary["nan"]).sum()),
        "fastest_20": fastest_threshold(thresholds, 0.8),
        "fastest_30": fastest_threshold(thresholds, 0.7),
    }


def frozen_snapshot() -> dict[str, object]:
    source_files = [
        PROJECT / "coherent_slicing" / "spectral.py",
        PROJECT / "coherent_slicing" / "aggregations.py",
        PROJECT / "experiments" / "run_moabb_pilot.py",
        PROJECT / "experiments" / "run_overnight.py",
        PROJECT / "experiments" / "run_logspectral_spdhsw.py",
        PROJECT / "experiments" / "run_fixed_vs_resampled_spectral_spdsw.py",
        PROJECT / "experiments" / "run_high_support_fixed_vs_resampled_spdsw.py",
        EXTERNAL / "evobank" / "data.py", EXTERNAL / "evobank" / "lew.py",
        EXTERNAL / "evobank" / "ot1d.py", EXTERNAL / "evobank" / "svec.py",
    ]
    cache_files = [
        EXTERNAL / "results" / "pilot_hgd" / "data_cache" / DATASET / f"subject_{subject:02d}_logs.pt"
        for subject in SUBJECTS
    ]
    return {
        "frozen_branch_heads": {name: branch_sha(name) for name in FROZEN_BRANCHES},
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "cache_sha256": {str(path): sha256(path) for path in cache_files},
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
        raise RuntimeError("a frozen result tree changed")
    if snapshot["frozen_untracked_sha256"] != FROZEN_UNTRACKED_FILES:
        raise RuntimeError("the frozen abandoned audit source changed")
    path = OUT / "FROZEN_SOURCE_HASHES.json"
    if path.exists() and json.loads(path.read_text()) != snapshot:
        raise RuntimeError("a frozen source or cache changed")
    if not path.exists():
        dump_json(path, snapshot)


def tests_pass() -> bool:
    path = OUT / "TEST_RESULTS.xml"
    if not path.exists():
        return False
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return bool(suites) and all(
        int(suite.attrib.get("failures", 0)) == 0
        and int(suite.attrib.get("errors", 0)) == 0
        for suite in suites
    )


def prepare() -> None:
    if current_branch() != BRANCH:
        raise RuntimeError(f"must run only on {BRANCH}")
    if not tests_pass():
        raise RuntimeError("full regression suite must pass before preparation")
    OUT.mkdir(parents=True, exist_ok=True)
    verify_frozen()
    device = check_device()
    config_path = OUT / "CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text()) != CONFIG_TEMPLATE:
        raise RuntimeError("refusing to alter frozen CONFIG.json")
    dump_json(config_path, CONFIG_TEMPLATE)
    environment = {
        "branch": current_branch(), "commit_at_prepare": branch_sha("HEAD"),
        "python": platform.python_version(), "python_executable": sys.executable,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(), "hostname": platform.node(),
        "dtype": str(DTYPE), "amp": False, "autocast": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": device,
    }
    dump_json(OUT / "ENVIRONMENT.json", environment)
    calibrate_rooted_step(write=True)


def require_prepared() -> None:
    if current_branch() != BRANCH:
        raise RuntimeError(f"must run only on {BRANCH}")
    verify_frozen()
    if not tests_pass() or not (OUT / "CONFIG.json").exists():
        raise RuntimeError("tests/prepare incomplete; scientific runs prohibited")
    if not (OUT / "ROOTED_STEP_CALIBRATION.json").exists():
        raise RuntimeError("rooted step calibration missing")


def manifest_complete() -> bool:
    path = OUT / "MANIFEST_TRAJECTORIES.json"
    if not path.exists():
        return False
    records = json.loads(path.read_text())
    return len(records) == 108 and all(
        record.get("status") in {"ok", "nonfinite", "cached_complete"}
        for record in records
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("prepare", "run", "analyze", "all"))
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    configure_numerics()
    if args.phase == "prepare":
        prepare()
        print(f"prepared {OUT}; eta_root={read_eta_root():.15g}")
        return 0
    require_prepared()
    check_device()
    if args.phase in {"run", "all"}:
        execute_grid(rerun=args.rerun)
        if not manifest_complete():
            raise RuntimeError("all 108 trajectories did not complete")
    if args.phase in {"analyze", "all"}:
        if not manifest_complete():
            raise RuntimeError("analysis requires all 108 trajectories")
        result = analyze()
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
