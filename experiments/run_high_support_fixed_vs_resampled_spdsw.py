#!/usr/bin/env python
"""Registered high-support fixed-versus-resampled direct SPDSW experiment.

Only direct ambient Frobenius-unit directions are evaluated.  No hierarchical
bank or mixture implementation is imported or constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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
OUT = PROJECT / "results" / "high_support_fixed_vs_resampled_spdsw_v2"
EXTERNAL = Path("/home/pikachu/EBSPDSW")
sys.path.insert(0, str(EXTERNAL))

from evobank.data import load as load_cached_subject  # noqa: E402
from evobank.lew import LEWEvaluator  # noqa: E402
from evobank.ot1d import w2_squared_per_direction  # noqa: E402
from evobank.svec import SvecBasis  # noqa: E402


BRANCH = "exp/high-support-fixed-vs-resampled-spdsw-v2"
DTYPE = torch.float64
# CUDA is ordered FASTEST_FIRST on this host.  Physical GPU 2 is torch cuda:3.
DEVICE = torch.device("cuda:3")
PHYSICAL_GPU = 2
SIGMA = 0.5
RAW_LR = 3000.0
HGD_NORMALIZED_STEP = 7.3473173386245945
HGD_NORMALIZED_STEP_SOURCE = (
    "results/coherent_sw_overnight/SCALE_MATCHED_CONFIG.json: frozen audited "
    "direct-SPDSW dataset-specific normalized step"
)
BNCI_N_PROJ = 500
HGD_CALIBRATION_N_PROJ = (500, 1000, 2000, 5000)
HGD_ALLOWED_FINAL_N_PROJ = (2000, 5000)
CALIBRATION_EPOCHS = 100
FACTORIAL_EPOCHS = 500
LEW_EVERY = 25
HGD_DEVELOPMENT_SUBJECTS = (2, 3, 4)
HGD_HELDOUT_SUBJECTS = (1, 7, 14)
BNCI_SUBJECTS = (1, 3, 8)
SEEDS = (6398, 3654, 1788)
CALIBRATION_SEED = 6398
CONTROLS = ("normalized_update", "raw_sgd_lr3000")

FROZEN_BRANCHES = {
    "main": "4edf5dda470c5e525c5feb274462414751348b4b",
    "exp/lognormal-spectral-spdhsw-v1": "b0e5b47e17d45a94f39b2b5ba08fa965a5d3a77c",
    "exp/fixed-vs-resampled-spectral-spdsw-v1": "ec4a68af6107b3522d5e841dd35f708402a6378b",
}


@dataclass(frozen=True)
class Method:
    name: str
    sampling: str
    aggregation: str
    N_proj: int
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
        return float(sum(getattr(self, field.name) for field in fields(self)))


@dataclass
class RunClocks:
    optimization_ms: float = 0.0
    evaluation_ms: float = 0.0

    def add_optimization(self, milliseconds: float) -> None:
        self.optimization_ms += float(milliseconds)

    def add_evaluation(self, milliseconds: float) -> None:
        self.evaluation_ms += float(milliseconds)


EPOCH_COLUMNS = [
    "dataset", "phase", "control", "method", "sampling", "aggregation",
    "subject", "seed", "epoch", "epochs", "N_proj", "sigma", "d", "m",
    "training_power_loss", "rooted_distance", "lew", "relative_lew",
    "lew_reduction_pct", "gap_closure", "raw_gradient_norm",
    "applied_update_norm", "mean_h", "std_h", "max_h", "min_h",
    "spectral_entropy", "spectral_effective_N", "spectral_max_weight",
    "spectral_top5_weight", "bank_seed", "bank_hash", "bank_hash_kind",
    "target_projection_hash", "initial_source_hash", "target_hash",
    "direction_sampling_ms", "source_projection_ms", "target_projection_ms",
    "wasserstein_1d_ms", "sorting_aggregation_ms", "backward_ms",
    "optimizer_update_ms", "total_epoch_ms", "one_time_bank_sampling_ms",
    "one_time_target_projection_ms", "cumulative_optimization_ms",
    "cumulative_evaluation_ms", "cumulative_direct_projection_count",
    "cumulative_direction_draw_count", "learning_rate", "eta_norm", "nan",
    "diverged", "status",
]

RANK_COLUMNS = [
    "dataset", "phase", "control", "method", "subject", "seed", "epoch",
    "N_proj", "rank_spearman_t_tm1", "top5_overlap_t_tm1",
    "top10_overlap_t_tm1", "fraction_ever_top5", "fraction_ever_top10",
    "effective_N", "weight_entropy", "bank_hash",
]

T = TypeVar("T")
_BANK_HASH_CACHE: dict[tuple[int, int, int], str] = {}


def factorial_methods(dataset: str, N_proj: int) -> tuple[Method, ...]:
    if dataset == "BNCI2014_001":
        return (
            Method("fixed_uniform_N500", "fixed", "uniform", N_proj),
            Method("resampled_uniform_N500", "resampled", "uniform", N_proj),
            Method("fixed_spectral_N500_s0p5", "fixed", "spectral", N_proj, SIGMA),
            Method("resampled_spectral_N500_s0p5", "resampled", "spectral", N_proj, SIGMA),
        )
    if dataset == "Schirrmeister2017":
        return (
            Method("fixed_uniform", "fixed", "uniform", N_proj),
            Method("resampled_uniform", "resampled", "uniform", N_proj),
            Method("fixed_spectral_s0p5", "fixed", "spectral", N_proj, SIGMA),
            Method("resampled_spectral_s0p5", "resampled", "spectral", N_proj, SIGMA),
        )
    raise ValueError(dataset)


def calibration_methods(N_proj: int) -> tuple[Method, Method]:
    return (
        Method(f"fixed_uniform_N{N_proj}", "fixed", "uniform", N_proj),
        Method(f"resampled_uniform_N{N_proj}", "resampled", "uniform", N_proj),
    )


CONFIG_TEMPLATE = {
    "version": "high_support_fixed_vs_resampled_spdsw_v2",
    "created_before_scientific_runs": True,
    "branch": BRANCH,
    "direct_only": True,
    "hierarchical_methods": False,
    "datasets": {
        "BNCI2014_001": {
            "d": 22, "m": 253, "subjects": list(BNCI_SUBJECTS),
            "seeds": list(SEEDS), "epochs": FACTORIAL_EPOCHS,
            "N_proj": BNCI_N_PROJ,
        },
        "Schirrmeister2017": {
            "d": 128, "m": 8256,
            "development_subjects": list(HGD_DEVELOPMENT_SUBJECTS),
            "heldout_subjects_if_gate_passes": list(HGD_HELDOUT_SUBJECTS),
            "seeds": list(SEEDS), "epochs": FACTORIAL_EPOCHS,
            "calibration_N_proj": list(HGD_CALIBRATION_N_PROJ),
            "allowed_final_N_proj": list(HGD_ALLOWED_FINAL_N_PROJ),
        },
    },
    "calibration": {
        "subjects": list(HGD_DEVELOPMENT_SUBJECTS), "seed": CALIBRATION_SEED,
        "epochs": CALIBRATION_EPOCHS, "lew_epochs": [0, 25, 50, 75, 100],
        "methods": ["fixed_uniform", "resampled_uniform"],
        "selection_rule": (
            "select 2000 iff mean resampled relative-LEW AUC_2000 <= "
            "1.01*AUC_5000 and mean resampled Final_2000 <= 1.01*Final_5000; "
            "otherwise select 5000"
        ),
    },
    "factorial": {
        "sampling": ["fixed", "resampled_each_epoch"],
        "aggregation": ["uniform", "lognormal_spectral_sigma_0.5"],
        "sigma": SIGMA, "sigma_search": False, "same_N_proj_within_dataset": True,
        "fixed_uniform_spectral_share_bank": True,
        "resampled_uniform_spectral_share_epoch_bank": True,
    },
    "controls": {
        "primary": "normalized_update",
        "secondary": "raw_sgd_lr3000", "raw_learning_rate": RAW_LR,
        "HGD_eta_norm": HGD_NORMALIZED_STEP,
        "HGD_eta_norm_source": HGD_NORMALIZED_STEP_SOURCE,
        "BNCI_eta_norm_rule": (
            "median initial raw uniform-SPDSW update norm under LR=3000, "
            "subjects 1,3,8, seed 6398, resampled uniform N_proj=500"
        ),
        "gradient_clipping": False, "method_specific_step": False,
        "early_stopping": False,
    },
    "evaluation": {
        "kind": "independent exact Log-Euclidean Wasserstein",
        "epochs": list(range(0, FACTORIAL_EPOCHS + 1, LEW_EVERY)),
        "excluded_from_optimization_wall_clock": True,
    },
    "classifications": {
        "factor_preference": (
            "favorable grand mean and favorable subject-mean sign in at least 2/3 "
            "development subjects; otherwise TIE or NULL"
        ),
        "overfitting": "SEVERE if mean overfit gap >=50 percentage points; REDUCED if >5; ABSENT if <=5",
        "quality_equivalent": (
            "mean fixed relative-LEW AUC <=1.01*mean paired resampled AUC and "
            "mean fixed final LEW <=1.02*mean paired resampled final LEW"
        ),
        "wall_clock_advantage": (
            "on aggregate mean evaluation curves, fixed reaches the paired "
            "resampled epoch-500 mean LEW and has lower mean cumulative "
            "optimization wall-clock at first reach"
        ),
    },
    "heldout_gate": [
        "fixed uniform or fixed spectral is development quality-equivalent",
        "no material fixed-method divergence/NaN increase",
        "normalized-update result supports the conclusion",
        "no post-hoc N_proj or sigma change",
        "actual aggregate matched-quality wall-clock benefit exists",
    ],
    "registered_trajectory_counts": {
        "HGD_support_calibration": 24, "BNCI_per_control": 36,
        "HGD_development_per_control": 36, "HGD_heldout_per_control_if_gate": 36,
    },
    "stop_rules": [
        "no added N_proj", "no added sigma", "no LR tuning", "no preprocessing change",
        "no subject change", "no hierarchical methods", "no new adaptive weighting",
    ],
}


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


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n")


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
        raise RuntimeError(
            f"CUDA ordinal mismatch: physical {PHYSICAL_GPU}={physical_uuid}, "
            f"{DEVICE}={properties.uuid}"
        )
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
    """Audited triangular epoch-seed sequence, independent of method/outcome."""
    return int(seed + epoch_zero_based * (epoch_zero_based + 1) // 2)


def method_bank_seed(method: Method, seed: int, epoch_zero_based: int) -> int:
    return direction_seed(seed, 0 if method.sampling == "fixed" else epoch_zero_based)


def sample_frobenius_directions(
    N_proj: int, basis: SvecBasis, seed: int
) -> torch.Tensor:
    """Audited Frobenius-uniform symmetric sampler in isometric svec form."""
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
    fingerprint = tensor_sha256(directions, full=full)
    if full:
        _BANK_HASH_CACHE[key] = fingerprint
    return fingerprint, "full_tensor_sha256" if full else "three_row_sha256"


def build_fixed_bank_state(
    basis: SvecBasis, target_vec: torch.Tensor, seed: int, N_proj: int
) -> FixedBankState:
    sampled_seed = direction_seed(seed, 0)
    directions, sampling_ms = timed(
        basis.device, lambda: sample_frobenius_directions(N_proj, basis, sampled_seed)
    )
    target_projection, target_ms = timed(
        basis.device, lambda: target_vec @ directions.T
    )
    fingerprint, _ = bank_hash(directions, sampled_seed, full=True)
    return FixedBankState(
        directions=directions,
        target_projection=target_projection,
        bank_seed=sampled_seed,
        bank_hash=fingerprint,
        target_projection_hash=tensor_sha256(target_projection, full=True),
        sampling_ms=sampling_ms,
        target_projection_ms=target_ms,
    )


def epoch_bank(
    method: Method,
    basis: SvecBasis,
    target_vec: torch.Tensor,
    seed: int,
    epoch_zero_based: int,
    fixed_state: FixedBankState | None,
) -> tuple[torch.Tensor, torch.Tensor, int, str, str, str, float, float]:
    sampled_seed = method_bank_seed(method, seed, epoch_zero_based)
    if method.sampling == "fixed":
        if fixed_state is None:
            raise ValueError("fixed method requires a fixed state")
        if fixed_state.directions.shape[0] != method.N_proj:
            raise ValueError("fixed state has wrong N_proj")
        return (
            fixed_state.directions, fixed_state.target_projection,
            fixed_state.bank_seed, fixed_state.bank_hash, "full_tensor_sha256",
            fixed_state.target_projection_hash, 0.0, 0.0,
        )
    directions, sampling_ms = timed(
        basis.device,
        lambda: sample_frobenius_directions(method.N_proj, basis, sampled_seed),
    )
    target_projection, target_ms = timed(
        basis.device, lambda: target_vec @ directions.T
    )
    fingerprint, kind = bank_hash(directions, sampled_seed, full=False)
    return (
        directions, target_projection, sampled_seed, fingerprint, kind,
        tensor_sha256(target_projection, full=False), sampling_ms, target_ms,
    )


def aggregate_directional_costs(
    h: torch.Tensor,
    method: Method,
    ordered_spectral_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method.aggregation == "uniform":
        weights = torch.full_like(h, 1.0 / h.numel()).detach()
        return h.mean(), weights
    if method.aggregation != "spectral" or ordered_spectral_weights is None:
        raise ValueError(method.aggregation)
    order = torch.argsort(h.detach(), stable=True)
    assigned = torch.empty_like(ordered_spectral_weights)
    assigned[order] = ordered_spectral_weights.detach()
    assigned = assigned.detach()
    return torch.sum(assigned * h), assigned


def normalized_update(gradient: torch.Tensor, eta_norm: float) -> torch.Tensor:
    norm = gradient.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        return torch.full_like(gradient, math.nan)
    return -float(eta_norm) * gradient / norm


def evaluate_independent_lew(
    evaluator: LEWEvaluator, basis: SvecBasis, parameter: torch.Tensor
) -> tuple[float, float]:
    """Exact LEW takes no finite training/evaluation direction bank."""
    started = time.perf_counter()
    value = evaluator(basis.inverse(parameter.detach()))
    return value, 1000.0 * (time.perf_counter() - started)


def distribution_diagnostics(weights: torch.Tensor) -> tuple[float, float, float, float]:
    weights = weights.detach()
    positive = weights > 0
    entropy = float(-(weights[positive] * weights[positive].log()).sum())
    effective = float(1.0 / weights.square().sum())
    maximum = float(weights.max())
    top5 = float(torch.topk(weights, min(5, weights.numel())).values.sum())
    return entropy, effective, maximum, top5


def descending_ranks(h: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(h.detach(), descending=True, stable=True)
    ranks = torch.empty(h.numel(), device=h.device, dtype=torch.int64)
    ranks[order] = torch.arange(1, h.numel() + 1, device=h.device)
    return ranks.detach()


def rank_transition_diagnostics(
    ranks: torch.Tensor,
    previous: torch.Tensor | None,
    ever_top5: torch.Tensor,
    ever_top10: torch.Tensor,
) -> tuple[float, float, float, float, float]:
    ever_top5 |= ranks <= 5
    ever_top10 |= ranks <= 10
    fraction5 = float(ever_top5.double().mean())
    fraction10 = float(ever_top10.double().mean())
    if previous is None:
        return math.nan, math.nan, math.nan, fraction5, fraction10
    x = previous.to(torch.float64)
    y = ranks.to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    spearman = float((x * y).sum() / torch.sqrt(x.square().sum() * y.square().sum()))
    top5 = float(((previous <= 5) & (ranks <= 5)).sum()) / 5.0
    top10 = float(((previous <= 10) & (ranks <= 10)).sum()) / 10.0
    return spearman, top5, top10, fraction5, fraction10


def blank_row(
    method: Method, dataset: str, phase: str, control: str, subject: int,
    seed: int, epoch: int, epochs: int, d: int, m: int, eta_norm: float,
) -> dict[str, object]:
    row = {column: math.nan for column in EPOCH_COLUMNS}
    row.update(
        dataset=dataset, phase=phase, control=control, method=method.name,
        sampling=method.sampling, aggregation=method.aggregation,
        subject=subject, seed=seed, epoch=epoch, epochs=epochs,
        N_proj=method.N_proj, sigma=method.sigma, d=d, m=m,
        cumulative_direct_projection_count=method.N_proj * epoch,
        cumulative_direction_draw_count=(
            method.N_proj if method.sampling == "fixed" else method.N_proj * epoch
        ),
        learning_rate=RAW_LR if control == "raw_sgd_lr3000" else math.nan,
        eta_norm=eta_norm if control == "normalized_update" else math.nan,
        nan=True, diverged=True, status="nonfinite_trajectory",
    )
    return row


def run_paths(
    phase: str, control: str, method: Method, subject: int, seed: int
) -> tuple[Path, Path]:
    stem = Path(phase) / control / method.name / f"seed_{seed}" / f"subject_{subject:02d}.csv"
    return OUT / "runs" / stem, OUT / "rank_diagnostics" / stem


def read_typed_csv(path: Path, *, rank: bool = False) -> pd.DataFrame:
    """Read our CSVs without pandas' unstable mixed-type inference path.

    Pandas 2.3.3 in the frozen environment can segfault while inferring one
    otherwise valid sparse epoch CSV.  Reading strings first is deterministic;
    conversion is then explicit from the registered schema.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    strings = (
        {"dataset", "phase", "control", "method", "bank_hash"}
        if rank else
        {
            "dataset", "phase", "control", "method", "sampling", "aggregation",
            "bank_hash", "bank_hash_kind", "target_projection_hash",
            "initial_source_hash", "target_hash", "status",
        }
    )
    booleans = set() if rank else {"nan", "diverged"}
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
    method: Method,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    dataset: str,
    phase: str,
    control: str,
    subject: int,
    seed: int,
    epochs: int,
    eta_norm: float,
    fixed_state: FixedBankState | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if control not in CONTROLS:
        raise ValueError(control)
    source = source.to(device=DEVICE, dtype=DTYPE)
    target = target.to(device=DEVICE, dtype=DTYPE)
    basis = SvecBasis(source.shape[-1], DEVICE, DTYPE)
    parameter = basis.forward(source).clone().requires_grad_(True)
    target_vec = basis.forward(target)
    if method.sampling == "fixed" and fixed_state is None:
        fixed_state = build_fixed_bank_state(basis, target_vec, seed, method.N_proj)
    evaluator = LEWEvaluator(target)
    clocks = RunClocks()
    lew0, evaluation_ms = evaluate_independent_lew(evaluator, basis, parameter)
    clocks.add_evaluation(evaluation_ms)
    evaluator.set_baseline(lew0)
    setup_ms = (
        fixed_state.sampling_ms + fixed_state.target_projection_ms
        if method.sampling == "fixed" and fixed_state is not None else 0.0
    )
    clocks.add_optimization(setup_ms)
    source_hash = tensor_sha256(parameter, full=True)
    fixed_target_hash = tensor_sha256(target_vec, full=True)
    initial = blank_row(
        method, dataset, phase, control, subject, seed, 0, epochs,
        basis.d, basis.m, eta_norm,
    )
    initial.update(
        lew=lew0, relative_lew=1.0, lew_reduction_pct=0.0, gap_closure=0.0,
        bank_seed=(fixed_state.bank_seed if fixed_state is not None else math.nan),
        bank_hash=(fixed_state.bank_hash if fixed_state is not None else ""),
        bank_hash_kind=("full_tensor_sha256" if fixed_state is not None else ""),
        target_projection_hash=(
            fixed_state.target_projection_hash if fixed_state is not None else ""
        ),
        initial_source_hash=source_hash, target_hash=fixed_target_hash,
        direction_sampling_ms=0.0, source_projection_ms=0.0,
        target_projection_ms=0.0, wasserstein_1d_ms=0.0,
        sorting_aggregation_ms=0.0, backward_ms=0.0,
        optimizer_update_ms=0.0, total_epoch_ms=0.0,
        one_time_bank_sampling_ms=(fixed_state.sampling_ms if fixed_state is not None else 0.0),
        one_time_target_projection_ms=(
            fixed_state.target_projection_ms if fixed_state is not None else 0.0
        ),
        cumulative_optimization_ms=clocks.optimization_ms,
        cumulative_evaluation_ms=clocks.evaluation_ms,
        cumulative_direct_projection_count=0,
        cumulative_direction_draw_count=(method.N_proj if fixed_state is not None else 0),
        nan=False, diverged=False, status="initial",
    )
    rows = [initial]
    rank_rows: list[dict[str, object]] = []
    record_ranks = method.sampling == "fixed" and method.aggregation == "spectral"
    previous_ranks: torch.Tensor | None = None
    ever_top5 = torch.zeros(method.N_proj, device=DEVICE, dtype=torch.bool)
    ever_top10 = torch.zeros(method.N_proj, device=DEVICE, dtype=torch.bool)
    ordered_weights = (
        lognormal_spectral_weights(method.N_proj, method.sigma, DEVICE, DTYPE).detach()
        if method.aggregation == "spectral" else None
    )
    finite = True
    first_training_loss = math.nan
    for zero_epoch in range(epochs):
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
        (loss, weights), stages.sorting_aggregation_ms = timed(
            DEVICE, lambda: aggregate_directional_costs(h, method, ordered_weights)
        )
        entropy, effective, maximum_weight, top5_weight = distribution_diagnostics(weights)
        rank_values: tuple[float, float, float, float, float] | None = None
        if record_ranks:
            def compute_rank_values() -> tuple[torch.Tensor, tuple[float, float, float, float, float]]:
                current = descending_ranks(h)
                values = rank_transition_diagnostics(
                    current, previous_ranks, ever_top5, ever_top10
                )
                return current, values

            (current_ranks, rank_values), rank_ms = timed(DEVICE, compute_rank_values)
            stages.sorting_aggregation_ms += rank_ms
        else:
            current_ranks = None
        _, stages.backward_ms = timed(DEVICE, loss.backward)
        gradient_norm = float(parameter.grad.norm())

        def apply_update() -> torch.Tensor:
            update = (
                normalized_update(parameter.grad, eta_norm)
                if control == "normalized_update" else -RAW_LR * parameter.grad
            )
            with torch.no_grad():
                parameter.add_(update)
            parameter.grad = None
            return update

        update, stages.optimizer_update_ms = timed(DEVICE, apply_update)
        update_norm = float(update.norm())
        epoch_ms = stages.total_epoch_ms()
        clocks.add_optimization(epoch_ms)
        epoch = zero_epoch + 1
        finite = bool(
            torch.isfinite(parameter).all() and torch.isfinite(loss)
            and math.isfinite(gradient_norm) and math.isfinite(update_norm)
        )
        if zero_epoch == 0:
            first_training_loss = float(loss.detach())
            rows[0]["training_power_loss"] = first_training_loss
            rows[0]["rooted_distance"] = math.sqrt(max(first_training_loss, 0.0))
        lew = relative = reduction = closure = math.nan
        diverged = not finite
        if finite and epoch % LEW_EVERY == 0:
            lew, evaluation_ms = evaluate_independent_lew(evaluator, basis, parameter)
            clocks.add_evaluation(evaluation_ms)
            relative = lew / lew0
            reduction = 100.0 * (lew0 - lew) / lew0
            closure = evaluator.closed_pct(lew)
            diverged = evaluator.diverged(lew)
        rows.append({
            "dataset": dataset, "phase": phase, "control": control,
            "method": method.name, "sampling": method.sampling,
            "aggregation": method.aggregation, "subject": subject, "seed": seed,
            "epoch": epoch, "epochs": epochs, "N_proj": method.N_proj,
            "sigma": method.sigma, "d": basis.d, "m": basis.m,
            "training_power_loss": float(loss.detach()),
            "rooted_distance": float(loss.detach().clamp_min(0).sqrt()),
            "lew": lew, "relative_lew": relative,
            "lew_reduction_pct": reduction, "gap_closure": closure,
            "raw_gradient_norm": gradient_norm, "applied_update_norm": update_norm,
            "mean_h": float(h.detach().mean()),
            "std_h": float(h.detach().std(unbiased=False)),
            "max_h": float(h.detach().max()), "min_h": float(h.detach().min()),
            "spectral_entropy": entropy, "spectral_effective_N": effective,
            "spectral_max_weight": maximum_weight,
            "spectral_top5_weight": top5_weight, "bank_seed": sampled_seed,
            "bank_hash": fingerprint, "bank_hash_kind": fingerprint_kind,
            "target_projection_hash": target_projection_hash,
            "initial_source_hash": source_hash, "target_hash": fixed_target_hash,
            "direction_sampling_ms": stages.direction_sampling_ms,
            "source_projection_ms": stages.source_projection_ms,
            "target_projection_ms": stages.target_projection_ms,
            "wasserstein_1d_ms": stages.wasserstein_1d_ms,
            "sorting_aggregation_ms": stages.sorting_aggregation_ms,
            "backward_ms": stages.backward_ms,
            "optimizer_update_ms": stages.optimizer_update_ms,
            "total_epoch_ms": epoch_ms,
            "one_time_bank_sampling_ms": (
                fixed_state.sampling_ms if method.sampling == "fixed" else 0.0
            ),
            "one_time_target_projection_ms": (
                fixed_state.target_projection_ms if method.sampling == "fixed" else 0.0
            ),
            "cumulative_optimization_ms": clocks.optimization_ms,
            "cumulative_evaluation_ms": clocks.evaluation_ms,
            "cumulative_direct_projection_count": method.N_proj * epoch,
            "cumulative_direction_draw_count": (
                method.N_proj if method.sampling == "fixed" else method.N_proj * epoch
            ),
            "learning_rate": RAW_LR if control == "raw_sgd_lr3000" else math.nan,
            "eta_norm": eta_norm if control == "normalized_update" else math.nan,
            "nan": not finite, "diverged": diverged,
            "status": "ok" if finite else "nonfinite",
        })
        if record_ranks and rank_values is not None:
            rank_rows.append({
                "dataset": dataset, "phase": phase, "control": control,
                "method": method.name, "subject": subject, "seed": seed,
                "epoch": epoch, "N_proj": method.N_proj,
                "rank_spearman_t_tm1": rank_values[0],
                "top5_overlap_t_tm1": rank_values[1],
                "top10_overlap_t_tm1": rank_values[2],
                "fraction_ever_top5": rank_values[3],
                "fraction_ever_top10": rank_values[4],
                "effective_N": effective, "weight_entropy": entropy,
                "bank_hash": fingerprint,
            })
            previous_ranks = current_ranks
        if not finite:
            rows.extend(
                blank_row(
                    method, dataset, phase, control, subject, seed, later,
                    epochs, basis.d, basis.m, eta_norm,
                )
                for later in range(epoch + 1, epochs + 1)
            )
            break
    frame = pd.DataFrame(rows)[EPOCH_COLUMNS]
    rank_frame = pd.DataFrame(rank_rows, columns=RANK_COLUMNS)
    evaluated = frame[np.isfinite(frame.lew)]
    metadata = {
        "dataset": dataset, "phase": phase, "control": control,
        "method": method.name, "sampling": method.sampling,
        "aggregation": method.aggregation, "subject": subject, "seed": seed,
        "epochs": epochs, "N_proj": method.N_proj, "sigma": method.sigma,
        "rows": len(frame), "rank_rows": len(rank_frame),
        "lew_initial": lew0,
        "lew_final": float(evaluated.lew.iloc[-1]) if not evaluated.empty else math.nan,
        "diverged": bool(frame.diverged.fillna(False).any()),
        "nan": bool(frame["nan"].fillna(False).any()),
        "optimization_ms": clocks.optimization_ms,
        "evaluation_ms": clocks.evaluation_ms,
        "status": "ok" if finite else "nonfinite",
    }
    return frame, rank_frame, metadata


def run_complete(path: Path, rank_path: Path, method: Method, epochs: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = read_typed_csv(path)
        if len(frame) != epochs + 1 or int(frame.epoch.iloc[-1]) != epochs:
            return False
        if method.sampling == "fixed" and method.aggregation == "spectral":
            return rank_path.exists() and len(read_typed_csv(rank_path, rank=True)) == epochs
        return True
    except Exception:
        return False


def metadata_from_frame(path: Path, rank_path: Path) -> dict[str, object]:
    frame = read_typed_csv(path)
    evaluated = frame[np.isfinite(frame.lew)]
    return {
        "dataset": str(frame.dataset.iloc[0]), "phase": str(frame.phase.iloc[0]),
        "control": str(frame.control.iloc[0]), "method": str(frame.method.iloc[0]),
        "sampling": str(frame.sampling.iloc[0]),
        "aggregation": str(frame.aggregation.iloc[0]),
        "subject": int(frame.subject.iloc[0]), "seed": int(frame.seed.iloc[0]),
        "epochs": int(frame.epochs.iloc[0]), "N_proj": int(frame.N_proj.iloc[0]),
        "sigma": float(frame.sigma.iloc[0]), "rows": len(frame),
        "rank_rows": len(read_typed_csv(rank_path, rank=True)) if rank_path.exists() else 0,
        "lew_initial": float(evaluated.lew.iloc[0]),
        "lew_final": float(evaluated.lew.iloc[-1]),
        "diverged": bool(frame.diverged.fillna(False).any()),
        "nan": bool(frame["nan"].fillna(False).any()),
        "optimization_ms": float(frame.cumulative_optimization_ms.iloc[-1]),
        "evaluation_ms": float(frame.cumulative_evaluation_ms.iloc[-1]),
        "status": "cached_complete",
    }


def manifest_path(phase: str, control: str) -> Path:
    return OUT / f"MANIFEST_{phase}_{control}.json"


def execute_grid(
    *, phase: str, dataset: str, subjects: Iterable[int], seeds: Iterable[int],
    methods: Iterable[Method], control: str, epochs: int, eta_norm: float,
    rerun: bool,
) -> list[dict[str, object]]:
    subjects = tuple(subjects)
    seeds = tuple(seeds)
    methods = tuple(methods)
    path_manifest = manifest_path(phase, control)
    records: list[dict[str, object]] = []
    total = len(subjects) * len(seeds) * len(methods)
    index = 0
    for subject in subjects:
        source, target, meta = load_cached_subject(dataset, subject, DEVICE)
        basis = SvecBasis(meta["d"], DEVICE, DTYPE)
        target_vec = basis.forward(target)
        for seed in seeds:
            fixed_states: dict[int, FixedBankState] = {}
            for N_proj in sorted({method.N_proj for method in methods if method.sampling == "fixed"}):
                fixed_states[N_proj] = build_fixed_bank_state(basis, target_vec, seed, N_proj)
            for method in methods:
                index += 1
                run_path, rank_path = run_paths(phase, control, method, subject, seed)
                try:
                    if rerun or not run_complete(run_path, rank_path, method, epochs):
                        frame, ranks, metadata = train_one(
                            method, source, target, dataset=dataset, phase=phase,
                            control=control, subject=subject, seed=seed, epochs=epochs,
                            eta_norm=eta_norm,
                            fixed_state=(fixed_states[method.N_proj] if method.sampling == "fixed" else None),
                        )
                        run_path.parent.mkdir(parents=True, exist_ok=True)
                        frame.to_csv(run_path, index=False)
                        if method.sampling == "fixed" and method.aggregation == "spectral":
                            rank_path.parent.mkdir(parents=True, exist_ok=True)
                            ranks.to_csv(rank_path, index=False)
                    else:
                        metadata = metadata_from_frame(run_path, rank_path)
                    record = {
                        **metadata,
                        "run_csv": str(run_path.relative_to(OUT)),
                        "rank_diagnostics_csv": (
                            str(rank_path.relative_to(OUT))
                            if method.sampling == "fixed" and method.aggregation == "spectral"
                            else None
                        ),
                        "error": None,
                    }
                    print(
                        f"[{phase} {control} {index:03d}/{total:03d}] s{subject:02d} "
                        f"seed={seed} {method.name:32s} N_proj={method.N_proj} "
                        f"LEW {record['lew_initial']:.4f}->{record['lew_final']:.4f}",
                        flush=True,
                    )
                except Exception as exc:
                    log_path = OUT / "logs" / f"{phase}_{control}_{method.name}_seed{seed}_s{subject:02d}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(traceback.format_exc())
                    record = {
                        "dataset": dataset, "phase": phase, "control": control,
                        "method": method.name, "sampling": method.sampling,
                        "aggregation": method.aggregation, "subject": subject,
                        "seed": seed, "epochs": epochs, "N_proj": method.N_proj,
                        "sigma": method.sigma, "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "run_csv": str(run_path.relative_to(OUT)),
                        "rank_diagnostics_csv": None,
                    }
                    print(f"[ERROR] {record['error']}", file=sys.stderr, flush=True)
                records.append(record)
                dump_json(path_manifest, records)
        del source, target, target_vec
        torch.cuda.empty_cache()
    return records


def derive_bnci_eta_norm() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    method = Method("resampled_uniform_N500", "resampled", "uniform", BNCI_N_PROJ)
    for subject in BNCI_SUBJECTS:
        source, target, meta = load_cached_subject("BNCI2014_001", subject, DEVICE)
        basis = SvecBasis(meta["d"], DEVICE, DTYPE)
        parameter = basis.forward(source).clone().requires_grad_(True)
        target_vec = basis.forward(target)
        directions = sample_frobenius_directions(
            BNCI_N_PROJ, basis, method_bank_seed(method, CALIBRATION_SEED, 0)
        )
        h = w2_squared_per_direction(
            (parameter @ directions.T).T, (target_vec @ directions.T).T
        )
        loss = h.mean()
        loss.backward()
        raw_gradient_norm = float(parameter.grad.norm())
        rows.append({
            "dataset": "BNCI2014_001", "subject": subject,
            "seed": CALIBRATION_SEED, "method": method.name,
            "N_proj": BNCI_N_PROJ, "raw_lr": RAW_LR,
            "initial_loss": float(loss.detach()),
            "raw_gradient_norm": raw_gradient_norm,
            "raw_update_norm": RAW_LR * raw_gradient_norm,
            "bank_seed": method_bank_seed(method, CALIBRATION_SEED, 0),
            "bank_hash": bank_hash(directions, CALIBRATION_SEED, full=True)[0],
        })
        del source, target, parameter, target_vec, directions, h, loss
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "BNCI_NORMALIZED_STEP_DERIVATION.csv", index=False)
    eta_norm = float(frame.raw_update_norm.median())
    return {
        "eta_norm": eta_norm,
        "rule": CONFIG_TEMPLATE["controls"]["BNCI_eta_norm_rule"],
        "values": rows,
        "selected_before_comparative_outcomes": True,
    }


def select_hgd_N_proj(calibration: pd.DataFrame) -> dict[str, object]:
    required = set(HGD_CALIBRATION_N_PROJ)
    found = set(int(value) for value in calibration.N_proj.unique())
    if found != required:
        raise ValueError(f"calibration N_proj mismatch: {sorted(found)}")
    resampled = calibration[calibration.sampling == "resampled"]
    grouped = resampled.groupby("N_proj", as_index=False).agg(
        AUC=("relative_lew_auc", "mean"), Final=("lew_final", "mean")
    ).set_index("N_proj")
    auc_condition = float(grouped.loc[2000, "AUC"]) <= 1.01 * float(grouped.loc[5000, "AUC"])
    final_condition = float(grouped.loc[2000, "Final"]) <= 1.01 * float(grouped.loc[5000, "Final"])
    selected = 2000 if auc_condition and final_condition else 5000
    return {
        "selected_N_proj_HGD": selected,
        "AUC_2000": float(grouped.loc[2000, "AUC"]),
        "AUC_5000": float(grouped.loc[5000, "AUC"]),
        "Final_2000": float(grouped.loc[2000, "Final"]),
        "Final_5000": float(grouped.loc[5000, "Final"]),
        "auc_condition": bool(auc_condition),
        "final_condition": bool(final_condition),
        "rule": CONFIG_TEMPLATE["calibration"]["selection_rule"],
        "selection_deterministic": True,
    }


def relative_lew_auc(group: pd.DataFrame) -> float:
    evaluated = group[np.isfinite(group.lew)].sort_values("epoch")
    epochs = int(group.epochs.iloc[0])
    if len(evaluated) < 2 or int(evaluated.epoch.iloc[-1]) != epochs:
        return math.inf
    return float(np.trapezoid(evaluated.relative_lew, evaluated.epoch) / epochs)


def load_all_frames() -> pd.DataFrame:
    paths = sorted((OUT / "runs").glob("*/*/*/seed_*/subject_*.csv"))
    if not paths:
        raise RuntimeError("no run CSVs")
    return pd.concat([read_typed_csv(path) for path in paths], ignore_index=True)


def load_all_ranks() -> pd.DataFrame:
    paths = sorted((OUT / "rank_diagnostics").glob("*/*/*/seed_*/subject_*.csv"))
    if not paths:
        return pd.DataFrame(columns=RANK_COLUMNS)
    return pd.concat([read_typed_csv(path, rank=True) for path in paths], ignore_index=True)


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset", "phase", "control", "method", "sampling", "aggregation",
        "subject", "seed", "epochs", "N_proj", "sigma", "d", "m",
    ]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=False):
        group = group.sort_values("epoch")
        evaluated = group[np.isfinite(group.lew)]
        losses = group[np.isfinite(group.training_power_loss)]
        initial_lew = float(evaluated.lew.iloc[0])
        final_lew = float(evaluated.lew.iloc[-1])
        initial_loss = float(group.training_power_loss.iloc[0])
        final_loss = float(losses.training_power_loss.iloc[-1]) if not losses.empty else math.nan
        train_reduction = (
            100.0 * (initial_loss - final_loss) / initial_loss
            if math.isfinite(initial_loss) and initial_loss != 0 else math.nan
        )
        lew_reduction = 100.0 * (initial_lew - final_lew) / initial_lew
        row = dict(zip(keys, key))
        row.update(
            lew_initial=initial_lew, lew_final=final_lew,
            relative_lew_auc=relative_lew_auc(group),
            lew_reduction_pct=lew_reduction,
            training_loss_initial=initial_loss, training_loss_final=final_loss,
            training_loss_reduction_pct=train_reduction,
            overfit_gap=train_reduction - lew_reduction,
            divergence=bool(group.diverged.fillna(False).any()),
            nan=bool(group["nan"].fillna(False).any()),
            cumulative_optimization_ms=float(group.cumulative_optimization_ms.iloc[-1]),
            cumulative_evaluation_ms=float(group.cumulative_evaluation_ms.iloc[-1]),
            mean_raw_gradient_norm=float(group.loc[group.epoch > 0, "raw_gradient_norm"].mean()),
            mean_applied_update_norm=float(group.loc[group.epoch > 0, "applied_update_norm"].mean()),
            median_effective_N=float(group.loc[group.epoch > 0, "spectral_effective_N"].median()),
            median_weight_entropy=float(group.loc[group.epoch > 0, "spectral_entropy"].median()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_summary(summary: pd.DataFrame) -> pd.DataFrame:
    block = summary[summary.phase == "hgd_support_calibration"].copy()
    rows: list[dict[str, object]] = []
    for N_proj, group in block.groupby("N_proj"):
        fixed = group[group.sampling == "fixed"]
        resampled = group[group.sampling == "resampled"]
        rows.append({
            "N_proj": int(N_proj),
            "fixed_mean_relative_lew_auc": float(fixed.relative_lew_auc.mean()),
            "resampled_mean_relative_lew_auc": float(resampled.relative_lew_auc.mean()),
            "fixed_minus_resampled_auc": float(fixed.relative_lew_auc.mean() - resampled.relative_lew_auc.mean()),
            "fixed_mean_final_lew": float(fixed.lew_final.mean()),
            "resampled_mean_final_lew": float(resampled.lew_final.mean()),
            "fixed_minus_resampled_final_lew": float(fixed.lew_final.mean() - resampled.lew_final.mean()),
            "fixed_mean_training_loss_reduction_pct": float(fixed.training_loss_reduction_pct.mean()),
            "fixed_mean_independent_lew_reduction_pct": float(fixed.lew_reduction_pct.mean()),
            "fixed_mean_overfit_gap": float(fixed.overfit_gap.mean()),
            "fixed_mean_wall_ms": float(fixed.cumulative_optimization_ms.mean()),
            "resampled_mean_wall_ms": float(resampled.cumulative_optimization_ms.mean()),
            "fixed_minus_resampled_wall_ms": float(
                fixed.cumulative_optimization_ms.mean() - resampled.cumulative_optimization_ms.mean()
            ),
        })
    return pd.DataFrame(rows).sort_values("N_proj")


def factorial_effects(summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    block = summary[
        (summary.dataset == dataset) & summary.phase.isin(["bnci_factorial", "hgd_development", "hgd_heldout"])
    ].copy()
    rows: list[dict[str, object]] = []
    effects = (
        "delta_resample_uniform", "delta_spectral_fixed",
        "delta_spectral_resampled", "interaction",
    )
    for (phase, control), group in block.groupby(["phase", "control"]):
        pivot = group.pivot_table(
            index=["subject", "seed"], columns=["sampling", "aggregation"],
            values="relative_lew_auc", aggfunc="first",
        )
        for (subject, seed), values in pivot.iterrows():
            fixed_uniform = values[("fixed", "uniform")]
            resampled_uniform = values[("resampled", "uniform")]
            fixed_spectral = values[("fixed", "spectral")]
            resampled_spectral = values[("resampled", "spectral")]
            rows.append({
                "dataset": dataset, "phase": phase, "control": control,
                "row_type": "subject_seed", "subject": int(subject), "seed": int(seed),
                "delta_resample_uniform": resampled_uniform - fixed_uniform,
                "delta_spectral_fixed": fixed_spectral - fixed_uniform,
                "delta_spectral_resampled": resampled_spectral - resampled_uniform,
                "interaction": (resampled_spectral - resampled_uniform) - (fixed_spectral - fixed_uniform),
                "mean": math.nan, "median": math.nan, "SD": math.nan,
                "effect_size_dz": math.nan,
            })
        seed_rows = pd.DataFrame(rows)
        seed_rows = seed_rows[
            (seed_rows.phase == phase) & (seed_rows.control == control)
            & (seed_rows.row_type == "subject_seed")
        ]
        for subject, subject_group in seed_rows.groupby("subject"):
            row = {
                "dataset": dataset, "phase": phase, "control": control,
                "row_type": "subject_mean", "subject": int(subject), "seed": math.nan,
                "mean": math.nan, "median": math.nan, "SD": math.nan,
                "effect_size_dz": math.nan,
            }
            row.update({effect: float(subject_group[effect].mean()) for effect in effects})
            rows.append(row)
        for effect in effects:
            values = seed_rows[effect].to_numpy(dtype=float)
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            rows.append({
                "dataset": dataset, "phase": phase, "control": control,
                "row_type": f"overall_{effect}", "subject": math.nan, "seed": math.nan,
                **{name: math.nan for name in effects},
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "SD": sd, "effect_size_dz": float(np.mean(values) / sd) if sd > 0 else math.nan,
            })
    return pd.DataFrame(rows)


def aggregate_core(summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    block = summary[summary.dataset == dataset]
    return block.groupby(
        ["dataset", "phase", "control", "method", "sampling", "aggregation",
         "epochs", "N_proj", "sigma", "d", "m"], as_index=False,
    ).agg(
        trajectory_count=("seed", "size"), subject_count=("subject", "nunique"),
        mean_relative_lew_auc=("relative_lew_auc", "mean"),
        median_relative_lew_auc=("relative_lew_auc", "median"),
        SD_relative_lew_auc=("relative_lew_auc", "std"),
        mean_final_lew=("lew_final", "mean"),
        mean_lew_reduction_pct=("lew_reduction_pct", "mean"),
        mean_training_loss_reduction_pct=("training_loss_reduction_pct", "mean"),
        mean_overfit_gap=("overfit_gap", "mean"),
        mean_cumulative_optimization_ms=("cumulative_optimization_ms", "mean"),
        divergence_count=("divergence", "sum"), nan_count=("nan", "sum"),
    )


def timing_summary(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    columns = [
        "direction_sampling_ms", "source_projection_ms", "target_projection_ms",
        "wasserstein_1d_ms", "sorting_aggregation_ms", "backward_ms",
        "optimizer_update_ms", "total_epoch_ms",
    ]
    block = frame[(frame.dataset == dataset) & (frame.epoch > 0)]
    aggregations = {column: (column, "mean") for column in columns}
    result = block.groupby(
        ["dataset", "phase", "control", "method", "sampling", "aggregation", "N_proj"],
        as_index=False,
    ).agg(**aggregations)
    setup = frame[frame.dataset == dataset].groupby(
        ["dataset", "phase", "control", "method", "N_proj"], as_index=False,
    ).agg(
        one_time_bank_sampling_ms=("one_time_bank_sampling_ms", "mean"),
        one_time_target_projection_ms=("one_time_target_projection_ms", "mean"),
    )
    return result.merge(setup, on=["dataset", "phase", "control", "method", "N_proj"])


def cell(summary: pd.DataFrame, sampling: str, aggregation: str) -> pd.DataFrame:
    return summary[(summary.sampling == sampling) & (summary.aggregation == aggregation)]


def factor_classification(effects: pd.DataFrame, effect: str, positive_label: str, negative_label: str, null_label: str) -> str:
    subject_rows = effects[
        (effects.row_type == "subject_mean") & (effects.control == "normalized_update")
    ]
    values = subject_rows[effect]
    if len(values) and int((values > 0).sum()) >= 2 and float(values.mean()) > 0:
        return positive_label
    if len(values) and int((values < 0).sum()) >= 2 and float(values.mean()) < 0:
        return negative_label
    return null_label


def matched_quality(
    frame: pd.DataFrame, summary: pd.DataFrame, dataset: str, phase: str,
    aggregation: str,
) -> dict[str, object]:
    runs = summary[
        (summary.dataset == dataset) & (summary.phase == phase)
        & (summary.control == "normalized_update") & (summary.aggregation == aggregation)
    ]
    fixed = cell(runs, "fixed", aggregation)
    resampled = cell(runs, "resampled", aggregation)
    mean_fixed_auc = float(fixed.relative_lew_auc.mean())
    mean_resampled_auc = float(resampled.relative_lew_auc.mean())
    mean_fixed_final = float(fixed.lew_final.mean())
    mean_resampled_final = float(resampled.lew_final.mean())
    auc_close = mean_fixed_auc <= 1.01 * mean_resampled_auc
    final_close = mean_fixed_final <= 1.02 * mean_resampled_final
    quality_equivalent = auc_close and final_close
    curves = frame[
        (frame.dataset == dataset) & (frame.phase == phase)
        & (frame.control == "normalized_update") & (frame.aggregation == aggregation)
        & np.isfinite(frame.lew)
    ].groupby(["sampling", "epoch"], as_index=False).agg(
        mean_lew=("lew", "mean"), mean_wall_ms=("cumulative_optimization_ms", "mean")
    )
    fixed_curve = curves[curves.sampling == "fixed"].sort_values("epoch")
    resampled_curve = curves[curves.sampling == "resampled"].sort_values("epoch")
    target = float(resampled_curve.mean_lew.iloc[-1])
    resampled_wall = float(resampled_curve.mean_wall_ms.iloc[-1])
    hits = fixed_curve[fixed_curve.mean_lew <= target]
    reached = not hits.empty
    first_epoch = float(hits.epoch.iloc[0]) if reached else math.nan
    fixed_wall = float(hits.mean_wall_ms.iloc[0]) if reached else math.nan
    wall_advantage = bool(reached and fixed_wall < resampled_wall)
    no_instability = int(fixed.divergence.sum()) <= int(resampled.divergence.sum()) and int(
        fixed.nan.sum()
    ) <= int(resampled.nan.sum())
    return {
        "aggregation": aggregation,
        "mean_fixed_relative_lew_auc": mean_fixed_auc,
        "mean_resampled_relative_lew_auc": mean_resampled_auc,
        "auc_close": bool(auc_close),
        "mean_fixed_final_lew": mean_fixed_final,
        "mean_resampled_final_lew": mean_resampled_final,
        "final_close": bool(final_close),
        "quality_equivalent": bool(quality_equivalent),
        "resampled_epoch500_mean_lew": target,
        "fixed_first_reach_epoch": first_epoch,
        "fixed_first_reach_wall_ms": fixed_wall,
        "resampled_epoch500_wall_ms": resampled_wall,
        "wall_clock_advantage": wall_advantage,
        "no_material_divergence_nan_increase": bool(no_instability),
    }


def bank_audit(frame: pd.DataFrame, selected_N_proj: int) -> dict[str, object]:
    factorial = frame[frame.phase.isin(["bnci_factorial", "hgd_development", "hgd_heldout"])]
    join_keys = ["dataset", "phase", "control", "subject", "seed", "epoch"]
    checks: dict[str, object] = {}
    for dataset in ("BNCI2014_001", "Schirrmeister2017"):
        block = factorial[(factorial.dataset == dataset) & (factorial.epoch > 0)]
        if block.empty:
            continue
        uniform = block[block.aggregation == "uniform"]
        spectral = block[block.aggregation == "spectral"]
        for sampling in ("fixed", "resampled"):
            left = uniform[uniform.sampling == sampling]
            right = spectral[spectral.sampling == sampling]
            paired = left.merge(right, on=join_keys, suffixes=("_uniform", "_spectral"))
            checks[f"{dataset}_{sampling}_pair_bank_hash_equal"] = bool(
                len(paired) and (paired.bank_hash_uniform == paired.bank_hash_spectral).all()
            )
            checks[f"{dataset}_{sampling}_pair_target_projection_hash_equal"] = bool(
                len(paired) and (
                    paired.target_projection_hash_uniform == paired.target_projection_hash_spectral
                ).all()
            )
        fixed = uniform[uniform.sampling == "fixed"]
        resampled = uniform[uniform.sampling == "resampled"]
        checks[f"{dataset}_fixed_unique_bank_hash_counts"] = sorted(
            int(value) for value in fixed.groupby(
                ["phase", "control", "subject", "seed"]
            ).bank_hash.nunique().unique()
        )
        checks[f"{dataset}_resampled_unique_bank_hash_counts"] = sorted(
            int(value) for value in resampled.groupby(
                ["phase", "control", "subject", "seed"]
            ).bank_hash.nunique().unique()
        )
        checks[f"{dataset}_fixed_epoch_target_projection_ms_max"] = float(fixed.target_projection_ms.max())
        checks[f"{dataset}_resampled_target_projection_ms_min"] = float(resampled.target_projection_ms.min())
        checks[f"{dataset}_N_proj_values"] = sorted(int(value) for value in block.N_proj.unique())
    initial = factorial[factorial.epoch == 0]
    initial_source_counts = initial.groupby(
        ["dataset", "phase", "control", "subject", "seed"]
    ).initial_source_hash.nunique()
    target_counts = initial.groupby(
        ["dataset", "phase", "control", "subject", "seed"]
    ).target_hash.nunique()
    normalized = factorial[
        (factorial.control == "normalized_update") & (factorial.epoch > 0)
        & np.isfinite(factorial.applied_update_norm)
    ]
    expected = normalized.dataset.map(
        {"BNCI2014_001": load_bnci_eta_norm(), "Schirrmeister2017": HGD_NORMALIZED_STEP}
    )
    update_error = ((normalized.applied_update_norm - expected).abs() / expected).max()
    calibration_counts = sorted(
        int(value) for value in frame[frame.phase == "hgd_support_calibration"].N_proj.unique()
    )
    checks.update({
        "initial_source_hash_shared_all_factorial_cells": bool((initial_source_counts == 1).all()),
        "target_hash_shared_all_factorial_cells": bool((target_counts == 1).all()),
        "normalized_update_max_relative_error": float(update_error),
        "calibration_N_proj_values": calibration_counts,
        "HGD_selected_N_proj": selected_N_proj,
        "independent_evaluator_signature_has_no_direction_bank": (
            "direction" not in inspect.signature(evaluate_independent_lew).parameters
        ),
        "evaluation_excluded_from_optimization_clock": True,
        "no_hierarchical_path": True,
        "resampled_hash_kind": "three_row_sha256",
        "fixed_hash_kind": "full_tensor_sha256",
    })
    expected_hgd = [selected_N_proj] if (factorial.dataset == "Schirrmeister2017").any() else []
    checks["pass"] = bool(
        checks.get("BNCI2014_001_N_proj_values") == [BNCI_N_PROJ]
        and checks.get("Schirrmeister2017_N_proj_values") == expected_hgd
        and calibration_counts == list(HGD_CALIBRATION_N_PROJ)
        and all(
            value for key, value in checks.items()
            if key.endswith("_pair_bank_hash_equal") or key.endswith("_pair_target_projection_hash_equal")
        )
        and all(
            value == [1] for key, value in checks.items()
            if key.endswith("fixed_unique_bank_hash_counts")
        )
        and all(
            value == [FACTORIAL_EPOCHS] for key, value in checks.items()
            if key.endswith("resampled_unique_bank_hash_counts")
        )
        and all(
            value == 0.0 for key, value in checks.items()
            if key.endswith("fixed_epoch_target_projection_ms_max")
        )
        and all(
            value > 0.0 for key, value in checks.items()
            if key.endswith("resampled_target_projection_ms_min")
        )
        and checks["initial_source_hash_shared_all_factorial_cells"]
        and checks["target_hash_shared_all_factorial_cells"]
        and checks["normalized_update_max_relative_error"] <= 1e-10
    )
    return checks


def load_bnci_eta_norm() -> float:
    path = OUT / "CONFIG.json"
    if not path.exists():
        raise RuntimeError("CONFIG.json absent")
    return float(json.loads(path.read_text())["controls"]["BNCI_eta_norm"])


def rank_summary(ranks: pd.DataFrame) -> pd.DataFrame:
    if ranks.empty:
        return pd.DataFrame()
    return ranks.groupby(
        ["dataset", "phase", "control", "method", "subject", "seed", "N_proj"],
        as_index=False,
    ).agg(
        mean_rank_spearman=("rank_spearman_t_tm1", "mean"),
        median_rank_spearman=("rank_spearman_t_tm1", "median"),
        mean_top5_overlap=("top5_overlap_t_tm1", "mean"),
        mean_top10_overlap=("top10_overlap_t_tm1", "mean"),
        fraction_directions_ever_top5=("fraction_ever_top5", "max"),
        fraction_directions_ever_top10=("fraction_ever_top10", "max"),
        median_effective_N=("effective_N", "median"),
        median_weight_entropy=("weight_entropy", "median"),
    )


def hgd_gate(
    frame: pd.DataFrame, summary: pd.DataFrame, audit: dict[str, object],
    selected_N_proj: int,
) -> dict[str, object]:
    methods = {
        aggregation: matched_quality(
            frame, summary, "Schirrmeister2017", "hgd_development", aggregation
        )
        for aggregation in ("uniform", "spectral")
    }
    for result in methods.values():
        result["passes"] = bool(
            result["quality_equivalent"]
            and result["no_material_divergence_nan_increase"]
            and result["wall_clock_advantage"]
            and selected_N_proj in HGD_ALLOWED_FINAL_N_PROJ
            and SIGMA == 0.5
            and audit["pass"]
        )
    passed = any(result["passes"] for result in methods.values())
    return {
        "pass": passed,
        "decision": "run_heldout_hgd" if passed else "stop_after_hgd_development",
        "selected_N_proj_HGD": selected_N_proj,
        "normalized_update_is_primary": True,
        "no_posthoc_N_proj_or_sigma_change": True,
        "methods": methods,
        "bank_audit_pass": bool(audit["pass"]),
        "heldout_HGD_executed": bool((frame.phase == "hgd_heldout").any()),
        "proceed_to_hierarchy": False,
    }


def overfit_classification(summary: pd.DataFrame, dataset: str, phase: str) -> str:
    fixed = summary[
        (summary.dataset == dataset) & (summary.phase == phase)
        & (summary.control == "normalized_update") & (summary.sampling == "fixed")
        & (summary.aggregation == "uniform")
    ]
    gap = float(fixed.overfit_gap.mean())
    if gap >= 50:
        return "SEVERE"
    if gap > 5:
        return "REDUCED"
    return "ABSENT"


def plot_calibration(calibration: pd.DataFrame) -> None:
    for fixed_column, resampled_column, ylabel, filename in (
        ("fixed_mean_relative_lew_auc", "resampled_mean_relative_lew_auc", "mean relative LEW AUC", "fig_hgd_support_auc_vs_nproj.png"),
        ("fixed_mean_final_lew", "resampled_mean_final_lew", "mean final LEW", "fig_hgd_support_final_lew_vs_nproj.png"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.plot(calibration.N_proj, calibration[fixed_column], marker="o", label="fixed uniform")
        axis.plot(calibration.N_proj, calibration[resampled_column], marker="o", label="resampled uniform")
        axis.set(xlabel="N_proj", ylabel=ylabel)
        axis.set_xscale("log")
        axis.set_xticks(list(HGD_CALIBRATION_N_PROJ), labels=[str(v) for v in HGD_CALIBRATION_N_PROJ])
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(calibration.N_proj, calibration.fixed_mean_overfit_gap, marker="o", color="#c44e52")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="N_proj", ylabel="fixed overfit gap (percentage points)")
    axis.set_xscale("log")
    axis.set_xticks(list(HGD_CALIBRATION_N_PROJ), labels=[str(v) for v in HGD_CALIBRATION_N_PROJ])
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hgd_fixed_overfit_vs_nproj.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(calibration.N_proj, calibration.fixed_minus_resampled_auc, marker="o", label="AUC gap")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(xlabel="N_proj", ylabel="fixed minus resampled AUC")
    axes[1].plot(calibration.N_proj, calibration.fixed_minus_resampled_wall_ms, marker="o", color="#55a868")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(xlabel="N_proj", ylabel="fixed minus resampled wall-clock (ms)")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(list(HGD_CALIBRATION_N_PROJ), labels=[str(v) for v in HGD_CALIBRATION_N_PROJ])
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hgd_fixed_resampled_gap_vs_nproj.png", dpi=180)
    fig.savefig(OUT / "fig_hgd_fixed_resampled_wallclock_gap_vs_nproj.png", dpi=180)
    plt.close(fig)


def method_label(row: pd.Series | object) -> str:
    sampling = getattr(row, "sampling", row["sampling"])
    aggregation = getattr(row, "aggregation", row["aggregation"])
    return f"{sampling}-{aggregation}"


def plot_dataset_curves(frame: pd.DataFrame, dataset: str, phase: str, prefix: str) -> None:
    block = frame[(frame.dataset == dataset) & (frame.phase == phase) & np.isfinite(frame.lew)].copy()
    colors = {
        "fixed-uniform": "#4c72b0", "resampled-uniform": "#dd8452",
        "fixed-spectral": "#55a868", "resampled-spectral": "#c44e52",
    }
    for x, xlabel, filename in (
        ("epoch", "epoch", f"fig_{prefix}_lew_vs_epoch.png"),
        ("cumulative_optimization_ms", "cumulative optimization wall-clock (ms)", f"fig_{prefix}_lew_vs_wallclock.png"),
        ("cumulative_direct_projection_count", "cumulative direct projection count", f"fig_{prefix}_lew_vs_projection_count.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for axis, control in zip(axes, CONTROLS):
            control_block = block[block.control == control]
            grouped = control_block.groupby(
                ["sampling", "aggregation", "epoch"], as_index=False
            ).agg(relative_lew=("relative_lew", "mean"), x=(x, "mean"))
            for key, line in grouped.groupby(["sampling", "aggregation"]):
                label = f"{key[0]}-{key[1]}"
                axis.plot(line.x, line.relative_lew, marker="o", markersize=2.5,
                          label=label, color=colors[label])
            axis.set(xlabel=xlabel, title=control)
            axis.grid(alpha=0.22)
        axes[0].set_ylabel("mean relative independent LEW")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=180)
        plt.close(fig)


def plot_timing(timing: pd.DataFrame, dataset: str, phase: str, filename: str) -> None:
    block = timing[
        (timing.dataset == dataset) & (timing.phase == phase)
        & (timing.control == "normalized_update")
    ].copy()
    block["label"] = block.sampling + "-" + block.aggregation
    components = [
        "direction_sampling_ms", "source_projection_ms", "target_projection_ms",
        "wasserstein_1d_ms", "sorting_aggregation_ms", "backward_ms",
        "optimizer_update_ms",
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(block))
    for component in components:
        values = block[component].to_numpy()
        axis.bar(block.label, values, bottom=bottom, label=component.removesuffix("_ms"))
        bottom += values
    axis.set_ylabel("mean milliseconds per epoch")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_factorial(effects: pd.DataFrame, dataset: str, filename: str) -> None:
    block = effects[
        (effects.control == "normalized_update") & (effects.row_type == "subject_mean")
    ]
    columns = [
        "delta_resample_uniform", "delta_spectral_fixed",
        "delta_spectral_resampled", "interaction",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    for axis, column in zip(axes, columns):
        axis.bar(block.subject.astype(int).astype(str), block[column], color="#4c72b0")
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(column.replace("delta_", "").replace("_", "\n"), fontsize=9)
        axis.set_xlabel("subject")
    axes[0].set_ylabel("paired relative-LEW AUC difference")
    fig.suptitle(dataset)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_training_vs_lew(frame: pd.DataFrame) -> None:
    block = frame[
        frame.phase.isin(["bnci_factorial", "hgd_development"])
        & (frame.control == "normalized_update") & (frame.sampling == "fixed")
        & np.isfinite(frame.lew) & np.isfinite(frame.training_power_loss)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, dataset in zip(axes, ("BNCI2014_001", "Schirrmeister2017")):
        data = block[block.dataset == dataset]
        for aggregation, group in data.groupby("aggregation"):
            initial_loss = group.groupby(["subject", "seed"]).training_power_loss.transform("first")
            train_reduction = 100 * (initial_loss - group.training_power_loss) / initial_loss
            axis.scatter(group.lew_reduction_pct, train_reduction, s=14, alpha=0.6, label=aggregation)
        axis.set(xlabel="independent LEW reduction (%)", ylabel="training-bank loss reduction (%)", title=dataset)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_loss_vs_independent_lew.png", dpi=180)
    plt.close(fig)


def plot_rank_persistence(rank_results: pd.DataFrame) -> None:
    primary = rank_results[
        rank_results.phase.isin(["bnci_factorial", "hgd_development"])
        & (rank_results.control == "normalized_update")
    ]
    metrics = [
        "mean_rank_spearman", "mean_top5_overlap", "mean_top10_overlap",
        "fraction_directions_ever_top5", "fraction_directions_ever_top10",
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    grouped = primary.groupby("dataset")
    x = np.arange(len(grouped))
    labels = []
    values = {metric: [] for metric in metrics}
    for dataset, block in grouped:
        labels.append(dataset)
        for metric in metrics:
            values[metric].append(float(block[metric].mean()))
    for axis, metric in zip(axes, metrics):
        axis.bar(x, values[metric], color="#55a868")
        axis.set_xticks(x, labels=["BNCI", "HGD"], rotation=20)
        axis.set_title(metric.replace("_", "\n"), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rank_persistence.png", dpi=180)
    plt.close(fig)


def frame_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(fmt(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_claim_ledger(gate: dict[str, object], selected_N_proj: int) -> None:
    text = f"""# Claim ledger

## Supported protocol facts

- This experiment evaluates direct ambient SPDSW only. Every BNCI factorial cell uses
  `N_proj=500`; every HGD final factorial cell uses the preregistered calibrated
  `N_proj={selected_N_proj}`. No hierarchical mixture is used.
- Fixed paired cells reuse one bank and cached target projections. Resampled paired
  cells use the same deterministic epoch-bank sequence and recompute target projections.
- Sigma is fixed at 0.5. Rank assignment is detached and recomputed every epoch.
- Primary comparisons use dataset-common normalized update norms; raw SGD at LR=3000
  is secondary.
- Independent exact LEW accepts no training direction bank, and its evaluation time is
  excluded from optimization wall-clock.

## Historical fact retained without reinterpretation

- A fixed bank of only 40 directions severely overfit HGD alignment. That result did
  not establish that persistent banks are intrinsically bad.

## Outcome-bounded claims

- HGD development held-out gate: `{gate['decision']}`.
- Any claim of fixed/resampled quality equivalence or wall-clock advantage is limited
  to the registered thresholds recorded in `GATE.json`.
- Null, adverse, divergent, and non-reaching trajectories remain in the run CSVs.

## Claims not made

- Fixed finite directions are not claimed to define population SPDSW.
- `N_proj/m` is descriptive, not a theoretical sampling law.
- Finite independently resampled realized values are not claimed to obey metricity.
- No downstream classification, hierarchy benefit, or unregistered weighting result
  is claimed.
"""
    (OUT / "CLAIM_LEDGER.md").write_text(text)


def write_report(
    frame: pd.DataFrame, summary: pd.DataFrame, bnci_core: pd.DataFrame,
    hgd_core: pd.DataFrame, bnci_effects: pd.DataFrame, hgd_effects: pd.DataFrame,
    calibration: pd.DataFrame, calibration_decision: dict[str, object],
    bnci_timing: pd.DataFrame, hgd_timing: pd.DataFrame,
    ranks: pd.DataFrame, gate: dict[str, object], audit: dict[str, object],
) -> None:
    bnci_uniform = factor_classification(
        bnci_effects, "delta_resample_uniform", "FIXED", "RESAMPLED", "TIE"
    )
    hgd_uniform = factor_classification(
        hgd_effects[hgd_effects.phase == "hgd_development"],
        "delta_resample_uniform", "FIXED", "RESAMPLED", "TIE",
    )
    bnci_spectral = factor_classification(
        bnci_effects, "delta_spectral_fixed", "WORSE", "IMPROVE", "NULL"
    )
    hgd_spectral = factor_classification(
        hgd_effects[hgd_effects.phase == "hgd_development"],
        "delta_spectral_fixed", "WORSE", "IMPROVE", "NULL",
    )
    bnci_quality = matched_quality(
        frame, summary, "BNCI2014_001", "bnci_factorial", "uniform"
    )
    hgd_quality = gate["methods"]["uniform"]
    overfit = overfit_classification(
        summary, "Schirrmeister2017", "hgd_development"
    )
    selected = calibration_decision["selected_N_proj_HGD"]
    heldout = gate["heldout_HGD_executed"]
    normalized_bnci = bnci_core[bnci_core.control == "normalized_update"]
    normalized_hgd = hgd_core[
        (hgd_core.control == "normalized_update") & (hgd_core.phase == "hgd_development")
    ]
    rank_display = ranks[
        ranks.phase.isin(["bnci_factorial", "hgd_development"])
        & (ranks.control == "normalized_update")
    ].groupby(["dataset", "N_proj"], as_index=False).mean(numeric_only=True)
    timing_columns = [
        "control", "sampling", "aggregation", "N_proj", "direction_sampling_ms",
        "source_projection_ms", "target_projection_ms", "wasserstein_1d_ms",
        "sorting_aggregation_ms", "backward_ms", "optimizer_update_ms",
        "total_epoch_ms", "one_time_bank_sampling_ms",
        "one_time_target_projection_ms",
    ]
    lines = [
        f"- regression tests: {'PASS' if tests_pass() else 'FAIL'}",
        f"- HGD calibrated N_proj: {selected}",
        f"- BNCI fixed-vs-resampled uniform: {bnci_uniform}",
        f"- HGD fixed-vs-resampled uniform: {hgd_uniform}",
        f"- BNCI fixed-spectral effect: {bnci_spectral}",
        f"- HGD fixed-spectral effect: {hgd_spectral}",
        f"- HGD large-fixed-bank overfitting: {overfit}",
        f"- BNCI fixed wall-clock advantage: {'YES' if bnci_quality['wall_clock_advantage'] else 'NO'}",
        f"- HGD fixed wall-clock advantage: {'YES' if hgd_quality['wall_clock_advantage'] else 'NO'}",
        f"- held-out HGD executed: {'YES' if heldout else 'NO'}",
        "- proceed to hierarchy: NO",
        "",
        "# High-support fixed versus resampled direct SPDSW",
        "",
        "## 1. Exact protocol",
        "",
        f"The registered 2×2 direct-SPDSW factorial used 500 epochs, exact LEW every 25 epochs, sigma={SIGMA}, normalized updates as primary control, and raw SGD LR={RAW_LR:g} as a secondary diagnostic. BNCI used N_proj=500; HGD used N_proj=" + str(selected) + ". No hierarchy, sigma search, LR search, clipping, or early stopping was used.",
        "",
        f"HGD reused the frozen direct-SPDSW normalized step `{HGD_NORMALIZED_STEP:.15g}`. BNCI used the preregistered initial-gradient derivation and froze `{load_bnci_eta_norm():.15g}` before comparative outcomes.",
        "",
        "## 2. Support calibration",
        "",
        frame_markdown(calibration),
        "",
        f"The deterministic rule selected **N_proj={selected}**. AUC condition: `{calibration_decision['auc_condition']}`; final-LEW condition: `{calibration_decision['final_condition']}`.",
        "",
        "## 3. BNCI 2×2",
        "",
        frame_markdown(normalized_bnci),
        "",
        "Factor effects use lower AUC as better. Subject/seed values and overall paired standardized effects are in `BNCI_SUBJECT_RESULTS.csv`; raw SGD remains diagnostic.",
        "",
        "## 4. HGD 2×2",
        "",
        frame_markdown(normalized_hgd),
        "",
        "Subject/seed values and effect summaries are in `HGD_SUBJECT_RESULTS.csv`. No significance claim is made from three development subjects.",
        "",
        "## 5. Timing",
        "",
        "Fixed cumulative time includes one-time sampling and target projection. Fixed per-epoch target projection time is zero; resampled target projections are recomputed. Exact-LEW evaluation is excluded.",
        "",
        "BNCI timing:", "", frame_markdown(bnci_timing[bnci_timing.phase == "bnci_factorial"][timing_columns]),
        "", "HGD timing:", "", frame_markdown(hgd_timing[hgd_timing.phase == "hgd_development"][timing_columns]),
        "",
        f"BNCI uniform matched-quality result: `{json.dumps(to_builtin(bnci_quality), sort_keys=True)}`.",
        "",
        f"HGD uniform matched-quality result: `{json.dumps(to_builtin(hgd_quality), sort_keys=True)}`.",
        "",
        "## 6. Finite-bank overfitting",
        "",
        f"HGD large-fixed-bank classification is **{overfit}** under the frozen thresholds. Calibration quantifies training-bank reduction, independent LEW reduction, and their overfit gap at every registered support. The N_proj=40 result is historical context only: a fixed bank of only 40 directions severely overfit HGD alignment.",
        "",
        "## 7. Spectral diagnostics",
        "",
        frame_markdown(rank_display),
        "",
        "Diagnostics are descriptive only. They did not alter sigma, N_proj, or any optimizer setting.",
        "",
        "## 8. Dimension comparison",
        "",
        f"BNCI has d=22, m=253 and N_proj=500 (N_proj/m={500/253:.4g}). HGD has d=128, m=8256 and N_proj={selected} (N_proj/m={selected/8256:.4g}). These ratios are descriptive context, not a theoretical sufficiency law.",
        "",
        "## 9. Gate decision",
        "",
        f"The HGD development decision is **{gate['decision']}**. Held-out HGD executed: **{'YES' if heldout else 'NO'}**. `GATE.json` records each registered quality, stability, frozen-setting, and matched-quality timing condition.",
        "",
        "## 10. Null/negative results",
        "",
        "All null, adverse, divergent, NaN, and quality-non-reaching trajectories are retained under `runs/` and counted in the result tables. No additional support, sigma, LR, weighting, subject, preprocessing, or hierarchy condition was introduced after inspection.",
        "",
        "## 11. Provenance",
        "",
        f"Branch `{BRANCH}`. `FROZEN_SOURCE_HASHES.json` protects prior branches, prior results, audited external evaluator/sampler sources, and cached data. Bank/target/source hashes and timing invariants are summarized in `BANK_AUDIT.json` (pass={audit['pass']}). `RUN_MANIFEST.json` enumerates every trajectory.",
        "",
        "Reproduction order:",
        "",
        "```bash",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m pytest --junitxml=results/high_support_fixed_vs_resampled_spdsw_v2/TEST_RESULTS.xml",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m experiments.run_high_support_fixed_vs_resampled_spdsw --phase prepare",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_high_support_fixed_vs_resampled_spdsw --phase calibration",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_high_support_fixed_vs_resampled_spdsw --phase bnci_normalized",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_high_support_fixed_vs_resampled_spdsw --phase bnci_raw",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_high_support_fixed_vs_resampled_spdsw --phase hgd_normalized",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_high_support_fixed_vs_resampled_spdsw --phase hgd_raw",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_high_support_fixed_vs_resampled_spdsw --phase analyze",
        "```",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def analyze_all() -> dict[str, object]:
    frame = load_all_frames()
    summary = summarize_runs(frame)
    calibration = calibration_summary(summary)
    calibration.to_csv(OUT / "HGD_SUPPORT_CALIBRATION.csv", index=False)
    decision = select_hgd_N_proj(summary[summary.phase == "hgd_support_calibration"])
    decision["support_table"] = calibration.to_dict(orient="records")
    dump_json(OUT / "HGD_SUPPORT_CALIBRATION.json", decision)
    selected = int(decision["selected_N_proj_HGD"])
    bnci_core = aggregate_core(summary, "BNCI2014_001")
    hgd_core = aggregate_core(summary, "Schirrmeister2017")
    bnci_effects = factorial_effects(summary, "BNCI2014_001")
    hgd_effects = factorial_effects(summary, "Schirrmeister2017")
    bnci_timing = timing_summary(frame, "BNCI2014_001")
    hgd_timing = timing_summary(frame, "Schirrmeister2017")
    ranks = rank_summary(load_all_ranks())
    bnci_core.to_csv(OUT / "BNCI_CORE_RESULTS.csv", index=False)
    bnci_effects.to_csv(OUT / "BNCI_SUBJECT_RESULTS.csv", index=False)
    bnci_timing.to_csv(OUT / "BNCI_TIMING.csv", index=False)
    hgd_core.to_csv(OUT / "HGD_CORE_RESULTS.csv", index=False)
    hgd_effects.to_csv(OUT / "HGD_SUBJECT_RESULTS.csv", index=False)
    hgd_timing.to_csv(OUT / "HGD_TIMING.csv", index=False)
    ranks.to_csv(OUT / "RANK_DIAGNOSTICS.csv", index=False)
    audit = bank_audit(frame, selected)
    dump_json(OUT / "BANK_AUDIT.json", audit)
    gate = hgd_gate(frame, summary, audit, selected)
    dump_json(OUT / "GATE.json", gate)
    manifests: list[dict[str, object]] = []
    for path in sorted(OUT.glob("MANIFEST_*.json")):
        manifests.extend(json.loads(path.read_text()))
    dump_json(OUT / "RUN_MANIFEST.json", manifests)
    plot_calibration(calibration)
    plot_dataset_curves(frame, "BNCI2014_001", "bnci_factorial", "bnci")
    plot_dataset_curves(frame, "Schirrmeister2017", "hgd_development", "hgd")
    plot_timing(bnci_timing, "BNCI2014_001", "bnci_factorial", "fig_bnci_timing.png")
    plot_timing(hgd_timing, "Schirrmeister2017", "hgd_development", "fig_hgd_timing.png")
    plot_factorial(bnci_effects, "BNCI2014_001", "fig_bnci_factorial.png")
    plot_factorial(
        hgd_effects[hgd_effects.phase == "hgd_development"],
        "Schirrmeister2017", "fig_hgd_factorial.png",
    )
    plot_training_vs_lew(frame)
    plot_rank_persistence(ranks)
    write_claim_ledger(gate, selected)
    write_report(
        frame, summary, bnci_core, hgd_core, bnci_effects, hgd_effects,
        calibration, decision, bnci_timing, hgd_timing, ranks, gate, audit,
    )
    verify_frozen()
    return gate


def frozen_snapshot() -> dict[str, object]:
    source_files = [
        PROJECT / "coherent_slicing" / "spectral.py",
        PROJECT / "coherent_slicing" / "aggregations.py",
        PROJECT / "experiments" / "run_moabb_pilot.py",
        PROJECT / "experiments" / "run_overnight.py",
        PROJECT / "experiments" / "run_logspectral_spdhsw.py",
        PROJECT / "experiments" / "run_fixed_vs_resampled_spectral_spdsw.py",
        EXTERNAL / "evobank" / "data.py", EXTERNAL / "evobank" / "lew.py",
        EXTERNAL / "evobank" / "ot1d.py", EXTERNAL / "evobank" / "svec.py",
    ]
    cache_files = [
        EXTERNAL / "results" / "pilot_hgd" / "data_cache" / dataset / f"subject_{subject:02d}_logs.pt"
        for dataset, subjects in (
            ("BNCI2014_001", BNCI_SUBJECTS),
            ("Schirrmeister2017", (*HGD_DEVELOPMENT_SUBJECTS, *HGD_HELDOUT_SUBJECTS)),
        )
        for subject in subjects
    ]
    return {
        "frozen_branch_heads": {name: branch_sha(name) for name in FROZEN_BRANCHES},
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "cache_sha256": {str(path): sha256(path) for path in cache_files},
        "prior_result_tree_sha256": {
            str(PROJECT / relative): tree_sha256(PROJECT / relative)
            for relative in (
                "results/coherent_sw_overnight",
                "results/lognormal_spectral_spdhsw_v1",
                "results/fixed_vs_resampled_spectral_spdsw_v1",
            )
        },
    }


def verify_frozen() -> None:
    for branch, expected in FROZEN_BRANCHES.items():
        if branch_sha(branch) != expected:
            raise RuntimeError(f"frozen branch moved: {branch}")
    path = OUT / "FROZEN_SOURCE_HASHES.json"
    current = frozen_snapshot()
    if not path.exists():
        dump_json(path, current)
    elif json.loads(path.read_text()) != current:
        raise RuntimeError("a frozen source, cache, branch, or prior result changed")


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
    eta_payload_path = OUT / "BNCI_NORMALIZED_STEP.json"
    if eta_payload_path.exists():
        eta_payload = json.loads(eta_payload_path.read_text())
    else:
        eta_payload = derive_bnci_eta_norm()
        dump_json(eta_payload_path, eta_payload)
    config = json.loads(json.dumps(CONFIG_TEMPLATE))
    config["controls"]["BNCI_eta_norm"] = float(eta_payload["eta_norm"])
    config["controls"]["BNCI_eta_norm_source"] = "BNCI_NORMALIZED_STEP.json"
    config_path = OUT / "CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("refusing to alter frozen CONFIG.json")
    dump_json(config_path, config)
    environment = {
        "branch": current_branch(),
        "commit_at_prepare": branch_sha("HEAD"),
        "python": platform.python_version(), "python_executable": sys.executable,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(), "hostname": platform.node(),
        "dtype": str(DTYPE), "amp": False, "autocast": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device,
    }
    dump_json(OUT / "ENVIRONMENT.json", environment)


def require_prepared() -> None:
    if current_branch() != BRANCH:
        raise RuntimeError(f"must run only on {BRANCH}")
    verify_frozen()
    if not tests_pass() or not (OUT / "CONFIG.json").exists():
        raise RuntimeError("tests/prepare incomplete; scientific runs prohibited")


def manifest_complete(phase: str, control: str, expected: int) -> bool:
    path = manifest_path(phase, control)
    if not path.exists():
        return False
    records = json.loads(path.read_text())
    return len(records) == expected and all(
        record.get("status") in {"ok", "nonfinite", "cached_complete"}
        for record in records
    )


def selected_hgd_N_proj() -> int:
    path = OUT / "HGD_SUPPORT_CALIBRATION.json"
    if not path.exists():
        raise RuntimeError("HGD support calibration decision absent")
    value = int(json.loads(path.read_text())["selected_N_proj_HGD"])
    if value not in HGD_ALLOWED_FINAL_N_PROJ:
        raise RuntimeError("invalid calibrated HGD N_proj")
    return value


def analyze_calibration_only() -> dict[str, object]:
    frame = load_all_frames()
    frame = frame[frame.phase == "hgd_support_calibration"]
    summary = summarize_runs(frame)
    table = calibration_summary(summary)
    decision = select_hgd_N_proj(summary)
    decision["support_table"] = table.to_dict(orient="records")
    table.to_csv(OUT / "HGD_SUPPORT_CALIBRATION.csv", index=False)
    dump_json(OUT / "HGD_SUPPORT_CALIBRATION.json", decision)
    plot_calibration(table)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True,
        choices=(
            "prepare", "calibration", "bnci_normalized", "bnci_raw",
            "hgd_normalized", "hgd_raw", "analyze",
            "heldout_normalized", "heldout_raw",
        ),
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    configure_numerics()
    if args.phase == "prepare":
        prepare()
        print(f"prepared {OUT}")
        return 0
    require_prepared()
    check_device()
    if args.phase == "calibration":
        methods = tuple(
            method for N_proj in HGD_CALIBRATION_N_PROJ
            for method in calibration_methods(N_proj)
        )
        execute_grid(
            phase="hgd_support_calibration", dataset="Schirrmeister2017",
            subjects=HGD_DEVELOPMENT_SUBJECTS, seeds=(CALIBRATION_SEED,),
            methods=methods, control="normalized_update", epochs=CALIBRATION_EPOCHS,
            eta_norm=HGD_NORMALIZED_STEP, rerun=args.rerun,
        )
        if not manifest_complete("hgd_support_calibration", "normalized_update", 24):
            raise RuntimeError("calibration did not complete all 24 trajectories")
        decision = analyze_calibration_only()
        print(f"[CALIBRATION] selected N_proj={decision['selected_N_proj_HGD']}")
    elif args.phase == "bnci_normalized":
        if not manifest_complete("hgd_support_calibration", "normalized_update", 24):
            raise RuntimeError("support calibration must finish before factorial runs")
        execute_grid(
            phase="bnci_factorial", dataset="BNCI2014_001", subjects=BNCI_SUBJECTS,
            seeds=SEEDS, methods=factorial_methods("BNCI2014_001", BNCI_N_PROJ),
            control="normalized_update", epochs=FACTORIAL_EPOCHS,
            eta_norm=load_bnci_eta_norm(), rerun=args.rerun,
        )
    elif args.phase == "bnci_raw":
        if not manifest_complete("bnci_factorial", "normalized_update", 36):
            raise RuntimeError("BNCI normalized factorial must finish before raw SGD")
        execute_grid(
            phase="bnci_factorial", dataset="BNCI2014_001", subjects=BNCI_SUBJECTS,
            seeds=SEEDS, methods=factorial_methods("BNCI2014_001", BNCI_N_PROJ),
            control="raw_sgd_lr3000", epochs=FACTORIAL_EPOCHS,
            eta_norm=load_bnci_eta_norm(), rerun=args.rerun,
        )
    elif args.phase == "hgd_normalized":
        if not manifest_complete("bnci_factorial", "raw_sgd_lr3000", 36):
            raise RuntimeError("registered BNCI phase must finish before HGD factorial")
        N_proj = selected_hgd_N_proj()
        execute_grid(
            phase="hgd_development", dataset="Schirrmeister2017",
            subjects=HGD_DEVELOPMENT_SUBJECTS, seeds=SEEDS,
            methods=factorial_methods("Schirrmeister2017", N_proj),
            control="normalized_update", epochs=FACTORIAL_EPOCHS,
            eta_norm=HGD_NORMALIZED_STEP, rerun=args.rerun,
        )
    elif args.phase == "hgd_raw":
        if not manifest_complete("hgd_development", "normalized_update", 36):
            raise RuntimeError("HGD normalized factorial must finish before raw SGD")
        N_proj = selected_hgd_N_proj()
        execute_grid(
            phase="hgd_development", dataset="Schirrmeister2017",
            subjects=HGD_DEVELOPMENT_SUBJECTS, seeds=SEEDS,
            methods=factorial_methods("Schirrmeister2017", N_proj),
            control="raw_sgd_lr3000", epochs=FACTORIAL_EPOCHS,
            eta_norm=HGD_NORMALIZED_STEP, rerun=args.rerun,
        )
    elif args.phase == "analyze":
        for phase, control, expected in (
            ("hgd_support_calibration", "normalized_update", 24),
            ("bnci_factorial", "normalized_update", 36),
            ("bnci_factorial", "raw_sgd_lr3000", 36),
            ("hgd_development", "normalized_update", 36),
            ("hgd_development", "raw_sgd_lr3000", 36),
        ):
            if not manifest_complete(phase, control, expected):
                raise RuntimeError(f"incomplete registered phase: {phase}/{control}")
        gate = analyze_all()
        print(f"[GATE] {gate['decision']}")
    elif args.phase.startswith("heldout_"):
        gate_path = OUT / "GATE.json"
        if not gate_path.exists() or not json.loads(gate_path.read_text())["pass"]:
            raise RuntimeError("HGD development gate failed or is absent; held-out is prohibited")
        control = "normalized_update" if args.phase.endswith("normalized") else "raw_sgd_lr3000"
        if control == "raw_sgd_lr3000" and not manifest_complete(
            "hgd_heldout", "normalized_update", 36
        ):
            raise RuntimeError("held-out normalized runs must finish before held-out raw SGD")
        N_proj = selected_hgd_N_proj()
        execute_grid(
            phase="hgd_heldout", dataset="Schirrmeister2017",
            subjects=HGD_HELDOUT_SUBJECTS, seeds=SEEDS,
            methods=factorial_methods("Schirrmeister2017", N_proj),
            control=control, epochs=FACTORIAL_EPOCHS,
            eta_norm=HGD_NORMALIZED_STEP, rerun=args.rerun,
        )
    else:
        raise RuntimeError(args.phase)
    verify_frozen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
