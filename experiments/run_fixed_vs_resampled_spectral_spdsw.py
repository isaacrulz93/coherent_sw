#!/usr/bin/env python
"""Fixed-versus-resampled direct Spectral SPDSW factorial experiment.

This runner is intentionally direct-only: it never constructs hierarchical
mixtures or imports a hierarchical bank implementation.
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
OUT = PROJECT / "results" / "fixed_vs_resampled_spectral_spdsw_v1"
EXTERNAL = Path("/home/pikachu/EBSPDSW")
sys.path.insert(0, str(EXTERNAL))

from evobank.data import load as load_cached_subject  # noqa: E402
from evobank.lew import LEWEvaluator  # noqa: E402
from evobank.ot1d import w2_squared_per_direction  # noqa: E402
from evobank.svec import SvecBasis  # noqa: E402


BRANCH = "exp/fixed-vs-resampled-spectral-spdsw-v1"
DTYPE = torch.float64
# On this host CUDA uses FASTEST_FIRST ordering: nvidia-smi physical index 3
# (UUID GPU-f8e60...) is PyTorch cuda:1. check_gpu3 verifies the UUID before use.
DEVICE = torch.device("cuda:1")
PHYSICAL_GPU = 3
K = 40
SIGMA = 0.5
EPOCHS = 500
LEW_EVERY = 25
DEVELOPMENT_SUBJECTS = (2, 3, 4)
HELDOUT_SUBJECTS = (1, 7, 14)
SEEDS = (6398, 3654, 1788)
NORMALIZED_STEP_TARGET = 7.3473173386245945
RAW_LR = 3000.0
CONTROLS = ("normalized_update", "raw_sgd_lr3000")
_BANK_HASH_CACHE: dict[tuple[int, int], str] = {}


@dataclass(frozen=True)
class Method:
    name: str
    sampling: str
    aggregation: str
    k: int = K
    sigma: float = 0.0
    reference: bool = False


PRIMARY_METHODS = (
    Method("fixed_spdsw_k40", "fixed", "uniform"),
    Method("resampled_spdsw_k40", "resampled", "uniform"),
    Method("fixed_lns_k40_s0p5", "fixed", "spectral", sigma=SIGMA),
    Method("resampled_lns_k40_s0p5", "resampled", "spectral", sigma=SIGMA),
)
REFERENCE_METHOD = Method(
    "resampled_spdsw_l500_reference", "resampled", "uniform", k=500, reference=True
)
ALL_METHODS = (*PRIMARY_METHODS, REFERENCE_METHOD)


CONFIG = {
    "version": "fixed_vs_resampled_spectral_spdsw_v1",
    "created_before_scientific_runs": True,
    "branch": BRANCH,
    "dataset": "Schirrmeister2017",
    "development_subjects": list(DEVELOPMENT_SUBJECTS),
    "heldout_subjects_if_gate_passes": list(HELDOUT_SUBJECTS),
    "seeds": list(SEEDS),
    "epochs": EPOCHS,
    "lew_every": LEW_EVERY,
    "dtype": "torch.float64",
    "device": "physical GPU 3 / PyTorch cuda:1 / UUID-verified",
    "factorial": {
        "k": K,
        "sampling": ["fixed", "resampled_each_epoch"],
        "aggregation": ["uniform", "lognormal_spectral_sigma_0.5"],
        "methods": [method.name for method in PRIMARY_METHODS],
        "no_hierarchy": True,
    },
    "fixed_bank": {
        "sample_once": True,
        "target_projection_cached": True,
        "uniform_and_spectral_share_tensor": True,
    },
    "resampled_bank": {
        "seed_sequence": "seed + epoch_zero_based*(epoch_zero_based+1)/2",
        "uniform_and_spectral_share_epoch_bank": True,
    },
    "controls": {
        "primary": "normalized_update",
        "normalized_step_target": NORMALIZED_STEP_TARGET,
        "normalized_target_source": "frozen direct coherent overnight SCALE_MATCHED_CONFIG.json",
        "secondary": "raw_sgd_lr3000",
        "raw_learning_rate": RAW_LR,
        "method_specific_lr": False,
    },
    "reference": {
        "method": REFERENCE_METHOD.name,
        "k": 500,
        "resampled_each_epoch": True,
        "reuse_decision": "rerun: no frozen trajectory matches development subjects x all seeds x 500 epochs x both controls",
        "factorial_member": False,
    },
    "development_expected_runs": {
        "factorial": 72,
        "reference": 18,
        "total": 90,
    },
    "classification": {
        "factor_preference": "requires favorable subject-mean sign in at least 2/3 subjects and favorable grand mean; otherwise TIE/NULL",
        "lower_auc_is_better": True,
        "wall_clock_advantage": "for at least one resampled k40 comparator, fixed spectral reaches that comparator's epoch-500 LEW earlier in >=6/9 runs and >=2/3 subject means",
    },
    "heldout_gate": [
        "fixed spectral improves fixed uniform in >=2/3 development subject means",
        "grand mean paired normalized AUC difference is negative",
        "fixed spectral divergence/NaN count does not exceed fixed uniform",
        "all nonzero normalized applied update norms match the common target within 1e-10 relative",
        "fixed spectral has the preregistered wall-clock advantage over a resampled k40 comparator",
        "k and sigma remain frozen",
    ],
    "prohibited": [
        "hierarchical mixtures",
        "sigma or k search",
        "gradient clipping",
        "method-specific learning rates",
        "outcome-triggered early stopping",
        "heldout or BNCI before a passing development gate",
    ],
}


EPOCH_COLUMNS = [
    "dataset",
    "phase",
    "control",
    "method",
    "sampling",
    "aggregation",
    "reference",
    "subject",
    "seed",
    "epoch",
    "k",
    "sigma",
    "training_power_loss",
    "rooted_distance",
    "lew",
    "relative_lew",
    "lew_reduction_pct",
    "gap_closure",
    "raw_gradient_norm",
    "applied_update_norm",
    "mean_h",
    "std_h",
    "max_h",
    "min_h",
    "spectral_entropy",
    "spectral_ess",
    "spectral_max_weight",
    "spectral_top5_weight",
    "bank_seed",
    "bank_hash",
    "bank_hash_kind",
    "target_projection_hash",
    "direction_sampling_ms",
    "source_projection_ms",
    "target_projection_ms",
    "wasserstein_1d_ms",
    "sorting_aggregation_ms",
    "backward_ms",
    "optimizer_update_ms",
    "total_optimization_epoch_ms",
    "one_time_fixed_bank_sampling_ms",
    "one_time_fixed_target_projection_ms",
    "cumulative_optimization_ms",
    "cumulative_evaluation_ms",
    "cumulative_ambient_projection_count",
    "cumulative_direction_draw_count",
    "learning_rate",
    "normalized_step_target",
    "nan",
    "diverged",
    "status",
]


RANK_COLUMNS = [
    "dataset",
    "phase",
    "control",
    "method",
    "subject",
    "seed",
    "epoch",
    "direction_index",
    "h",
    "rank_descending",
    "assigned_weight",
    "rank_transition",
    "bank_hash",
]


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

    def optimization_total_ms(self) -> float:
        return float(sum(getattr(self, field.name) for field in fields(self)))


@dataclass
class RunClocks:
    optimization_ms: float = 0.0
    evaluation_ms: float = 0.0

    def add_optimization(self, milliseconds: float) -> None:
        self.optimization_ms += float(milliseconds)

    def add_evaluation(self, milliseconds: float) -> None:
        self.evaluation_ms += float(milliseconds)


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


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def current_branch() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT, text=True).strip()


def configure_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def check_gpu3() -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= DEVICE.index:
        raise RuntimeError("physical GPU 3 unavailable; refusing a device switch")
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(PHYSICAL_GPU),
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError(f"physical GPU 3 is contaminated: {processes}")
    uuid_query = subprocess.run(
        ["nvidia-smi", "-i", str(PHYSICAL_GPU), "--query-gpu=uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    physical_uuid = uuid_query.stdout.strip().removeprefix("GPU-")
    torch.cuda.set_device(DEVICE)
    properties = torch.cuda.get_device_properties(DEVICE)
    torch_uuid = str(properties.uuid)
    if torch_uuid != physical_uuid:
        raise RuntimeError(
            f"CUDA ordinal mismatch: nvidia-smi GPU {PHYSICAL_GPU} UUID={physical_uuid}, "
            f"{DEVICE} UUID={torch_uuid}"
        )
    return {
        "physical_gpu": PHYSICAL_GPU,
        "torch_device": str(DEVICE),
        "name": properties.name,
        "uuid": physical_uuid,
        "total_memory_bytes": properties.total_memory,
        "compute_processes_before_initialization": processes,
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, operation: Callable[[], T]) -> tuple[T, float]:
    sync(device)
    started = time.perf_counter()
    value = operation()
    sync(device)
    return value, 1000.0 * (time.perf_counter() - started)


def direction_seed(seed: int, epoch_zero_based: int) -> int:
    """Frozen direct-pilot triangular epoch seed sequence."""
    return int(seed + epoch_zero_based * (epoch_zero_based + 1) // 2)


def method_bank_seed(method: Method, seed: int, epoch_zero_based: int) -> int:
    if method.sampling == "fixed":
        return direction_seed(seed, 0)
    if method.sampling == "resampled":
        return direction_seed(seed, epoch_zero_based)
    raise ValueError(method.sampling)


def sample_frobenius_directions(count: int, basis: SvecBasis, seed: int) -> torch.Tensor:
    """Audited Frobenius-uniform symmetric directions in isometric svec form."""
    generator = torch.Generator(device=basis.device).manual_seed(int(seed))
    gaussian = torch.randn(
        count,
        basis.d,
        basis.d,
        generator=generator,
        device=basis.device,
        dtype=basis.dtype,
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


def bank_hash(directions: torch.Tensor, seed: int) -> tuple[str, str]:
    key = (int(directions.shape[0]), int(seed))
    cached = _BANK_HASH_CACHE.get(key)
    full = directions.shape[0] == K
    if cached is None:
        cached = tensor_sha256(directions, full=full)
        _BANK_HASH_CACHE[key] = cached
    return cached, "full_tensor_sha256" if full else "three_row_sha256_reference_only"


def target_hash(projection: torch.Tensor, seed: int) -> str:
    del seed
    return tensor_sha256(projection, full=projection.shape[1] == K)


def build_fixed_bank_state(basis: SvecBasis, target_vec: torch.Tensor, seed: int) -> FixedBankState:
    directions, sampling_ms = timed(
        basis.device, lambda: sample_frobenius_directions(K, basis, direction_seed(seed, 0))
    )
    projection, target_ms = timed(basis.device, lambda: target_vec @ directions.T)
    fingerprint, _ = bank_hash(directions, direction_seed(seed, 0))
    return FixedBankState(
        directions=directions,
        target_projection=projection,
        bank_seed=direction_seed(seed, 0),
        bank_hash=fingerprint,
        target_projection_hash=target_hash(projection, direction_seed(seed, 0)),
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
            raise ValueError("fixed method requires a fixed bank state")
        return (
            fixed_state.directions,
            fixed_state.target_projection,
            fixed_state.bank_seed,
            fixed_state.bank_hash,
            "full_tensor_sha256",
            fixed_state.target_projection_hash,
            0.0,
            0.0,
        )
    directions, sampling_ms = timed(
        basis.device, lambda: sample_frobenius_directions(method.k, basis, sampled_seed)
    )
    projection, target_ms = timed(basis.device, lambda: target_vec @ directions.T)
    fingerprint, fingerprint_kind = bank_hash(directions, sampled_seed)
    return (
        directions,
        projection,
        sampled_seed,
        fingerprint,
        fingerprint_kind,
        target_hash(projection, sampled_seed),
        sampling_ms,
        target_ms,
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
        raise ValueError(f"invalid aggregation {method.aggregation}")
    order = torch.argsort(h.detach(), stable=True)
    assigned = torch.empty_like(ordered_spectral_weights)
    assigned[order] = ordered_spectral_weights.detach()
    assigned = assigned.detach()
    return torch.sum(assigned * h), assigned


def normalized_update(gradient: torch.Tensor, target_norm: float) -> torch.Tensor:
    norm = gradient.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        return torch.full_like(gradient, math.nan)
    return -float(target_norm) * gradient / norm


def evaluate_independent_lew(
    evaluator: LEWEvaluator, basis: SvecBasis, parameter: torch.Tensor
) -> tuple[float, float]:
    """Exact LEW uses no training or evaluation projection direction bank."""
    started = time.perf_counter()
    value = evaluator(basis.inverse(parameter.detach()))
    return value, 1000.0 * (time.perf_counter() - started)


def distribution_diagnostics(weights: torch.Tensor) -> tuple[float, float, float, float]:
    weights = weights.detach()
    positive = weights > 0
    entropy = float(-(weights[positive] * weights[positive].log()).sum())
    ess = float(1.0 / weights.square().sum())
    maximum = float(weights.max())
    top5 = float(torch.topk(weights, min(5, weights.numel())).values.sum())
    return entropy, ess, maximum, top5


def descending_ranks(h: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(h.detach(), descending=True, stable=True)
    ranks = torch.empty(h.numel(), device=h.device, dtype=torch.int64)
    ranks[order] = torch.arange(1, h.numel() + 1, device=h.device)
    return ranks.detach()


def blank_row(
    method: Method,
    dataset: str,
    phase: str,
    control: str,
    subject: int,
    seed: int,
    epoch: int,
) -> dict[str, object]:
    row: dict[str, object] = {column: math.nan for column in EPOCH_COLUMNS}
    row.update(
        dataset=dataset,
        phase=phase,
        control=control,
        method=method.name,
        sampling=method.sampling,
        aggregation=method.aggregation,
        reference=method.reference,
        subject=subject,
        seed=seed,
        epoch=epoch,
        k=method.k,
        sigma=method.sigma,
        cumulative_ambient_projection_count=(
            method.k * (epoch + 1) if method.sampling == "fixed" else 2 * method.k * epoch
        ),
        cumulative_direction_draw_count=(method.k if method.sampling == "fixed" else method.k * epoch),
        learning_rate=RAW_LR if control == "raw_sgd_lr3000" else math.nan,
        normalized_step_target=NORMALIZED_STEP_TARGET if control == "normalized_update" else math.nan,
        nan=True,
        diverged=True,
        status="nonfinite_trajectory",
    )
    return row


def run_paths(phase: str, control: str, method: Method, subject: int, seed: int) -> tuple[Path, Path]:
    stem = Path(phase) / control / method.name / f"seed_{seed}" / f"subject_{subject:02d}.csv"
    return OUT / "runs" / stem, OUT / "ranks" / stem


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
    fixed_state: FixedBankState | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if control not in CONTROLS:
        raise ValueError(control)
    basis = SvecBasis(source.shape[-1], DEVICE, DTYPE)
    source = source.to(device=DEVICE, dtype=DTYPE)
    target = target.to(device=DEVICE, dtype=DTYPE)
    parameter = basis.forward(source).clone().requires_grad_(True)
    target_vec = basis.forward(target)
    if method.sampling == "fixed" and fixed_state is None:
        fixed_state = build_fixed_bank_state(basis, target_vec, seed)
    evaluator = LEWEvaluator(target)
    clocks = RunClocks()
    lew0, evaluation_ms = evaluate_independent_lew(evaluator, basis, parameter)
    clocks.add_evaluation(evaluation_ms)
    evaluator.set_baseline(lew0)
    fixed_setup_ms = 0.0 if fixed_state is None else fixed_state.sampling_ms + fixed_state.target_projection_ms
    clocks.add_optimization(fixed_setup_ms if method.sampling == "fixed" else 0.0)
    initial = blank_row(method, dataset, phase, control, subject, seed, 0)
    initial.update(
        lew=lew0,
        relative_lew=1.0,
        lew_reduction_pct=0.0,
        gap_closure=0.0,
        bank_seed=fixed_state.bank_seed if method.sampling == "fixed" else math.nan,
        bank_hash=fixed_state.bank_hash if method.sampling == "fixed" else "",
        bank_hash_kind="full_tensor_sha256" if method.sampling == "fixed" else "",
        target_projection_hash=(
            fixed_state.target_projection_hash if method.sampling == "fixed" else ""
        ),
        direction_sampling_ms=0.0,
        source_projection_ms=0.0,
        target_projection_ms=0.0,
        wasserstein_1d_ms=0.0,
        sorting_aggregation_ms=0.0,
        backward_ms=0.0,
        optimizer_update_ms=0.0,
        total_optimization_epoch_ms=0.0,
        one_time_fixed_bank_sampling_ms=(fixed_state.sampling_ms if method.sampling == "fixed" else 0.0),
        one_time_fixed_target_projection_ms=(
            fixed_state.target_projection_ms if method.sampling == "fixed" else 0.0
        ),
        cumulative_optimization_ms=clocks.optimization_ms,
        cumulative_evaluation_ms=clocks.evaluation_ms,
        cumulative_ambient_projection_count=(method.k if method.sampling == "fixed" else 0),
        cumulative_direction_draw_count=(method.k if method.sampling == "fixed" else 0),
        nan=False,
        diverged=False,
        status="initial",
    )
    rows = [initial]
    rank_rows: list[dict[str, object]] = []
    prior_ranks: torch.Tensor | None = None
    ordered_weights = (
        lognormal_spectral_weights(method.k, method.sigma, DEVICE, DTYPE).detach()
        if method.aggregation == "spectral"
        else None
    )
    for zero_epoch in range(EPOCHS):
        stages = StageTimes()
        (
            directions,
            projected_target,
            sampled_seed,
            fingerprint,
            fingerprint_kind,
            projected_target_hash,
            stages.direction_sampling_ms,
            stages.target_projection_ms,
        ) = epoch_bank(method, basis, target_vec, seed, zero_epoch, fixed_state)
        projected_source, stages.source_projection_ms = timed(
            DEVICE, lambda: parameter @ directions.T
        )
        h, stages.wasserstein_1d_ms = timed(
            DEVICE,
            lambda: w2_squared_per_direction(projected_source.T, projected_target.T),
        )
        (loss, weights), stages.sorting_aggregation_ms = timed(
            DEVICE, lambda: aggregate_directional_costs(h, method, ordered_weights)
        )
        _, stages.backward_ms = timed(DEVICE, loss.backward)
        gradient_norm = float(parameter.grad.norm())

        def apply_update() -> torch.Tensor:
            if control == "normalized_update":
                update = normalized_update(parameter.grad, NORMALIZED_STEP_TARGET)
            else:
                update = -RAW_LR * parameter.grad
            with torch.no_grad():
                parameter.add_(update)
            parameter.grad = None
            return update

        update, stages.optimizer_update_ms = timed(DEVICE, apply_update)
        update_norm = float(update.norm())
        epoch_optimization_ms = stages.optimization_total_ms()
        clocks.add_optimization(epoch_optimization_ms)
        epoch = zero_epoch + 1
        finite = (
            bool(torch.isfinite(parameter).all())
            and bool(torch.isfinite(loss))
            and math.isfinite(gradient_norm)
            and math.isfinite(update_norm)
        )
        entropy, ess, maximum_weight, top5_weight = distribution_diagnostics(weights)
        lew = math.nan
        relative_lew = math.nan
        lew_reduction = math.nan
        closure = math.nan
        diverged = not finite
        if finite and epoch % LEW_EVERY == 0:
            lew, evaluation_ms = evaluate_independent_lew(evaluator, basis, parameter)
            clocks.add_evaluation(evaluation_ms)
            relative_lew = lew / lew0
            lew_reduction = 100.0 * (lew0 - lew) / lew0
            closure = evaluator.closed_pct(lew)
            diverged = evaluator.diverged(lew)
        rows.append(
            {
                "dataset": dataset,
                "phase": phase,
                "control": control,
                "method": method.name,
                "sampling": method.sampling,
                "aggregation": method.aggregation,
                "reference": method.reference,
                "subject": subject,
                "seed": seed,
                "epoch": epoch,
                "k": method.k,
                "sigma": method.sigma,
                "training_power_loss": float(loss.detach()),
                "rooted_distance": float(loss.detach().clamp_min(0).sqrt()),
                "lew": lew,
                "relative_lew": relative_lew,
                "lew_reduction_pct": lew_reduction,
                "gap_closure": closure,
                "raw_gradient_norm": gradient_norm,
                "applied_update_norm": update_norm,
                "mean_h": float(h.detach().mean()),
                "std_h": float(h.detach().std(unbiased=False)),
                "max_h": float(h.detach().max()),
                "min_h": float(h.detach().min()),
                "spectral_entropy": entropy,
                "spectral_ess": ess,
                "spectral_max_weight": maximum_weight,
                "spectral_top5_weight": top5_weight,
                "bank_seed": sampled_seed,
                "bank_hash": fingerprint,
                "bank_hash_kind": fingerprint_kind,
                "target_projection_hash": projected_target_hash,
                "direction_sampling_ms": stages.direction_sampling_ms,
                "source_projection_ms": stages.source_projection_ms,
                "target_projection_ms": stages.target_projection_ms,
                "wasserstein_1d_ms": stages.wasserstein_1d_ms,
                "sorting_aggregation_ms": stages.sorting_aggregation_ms,
                "backward_ms": stages.backward_ms,
                "optimizer_update_ms": stages.optimizer_update_ms,
                "total_optimization_epoch_ms": epoch_optimization_ms,
                "one_time_fixed_bank_sampling_ms": (
                    fixed_state.sampling_ms if method.sampling == "fixed" else 0.0
                ),
                "one_time_fixed_target_projection_ms": (
                    fixed_state.target_projection_ms if method.sampling == "fixed" else 0.0
                ),
                "cumulative_optimization_ms": clocks.optimization_ms,
                "cumulative_evaluation_ms": clocks.evaluation_ms,
                "cumulative_ambient_projection_count": (
                    method.k * (epoch + 1)
                    if method.sampling == "fixed"
                    else 2 * method.k * epoch
                ),
                "cumulative_direction_draw_count": (
                    method.k if method.sampling == "fixed" else method.k * epoch
                ),
                "learning_rate": RAW_LR if control == "raw_sgd_lr3000" else math.nan,
                "normalized_step_target": (
                    NORMALIZED_STEP_TARGET if control == "normalized_update" else math.nan
                ),
                "nan": not finite,
                "diverged": diverged,
                "status": "ok" if finite else "nonfinite",
            }
        )
        if method.sampling == "fixed":
            ranks = descending_ranks(h)
            transition = (
                torch.full_like(ranks, 0)
                if prior_ranks is None
                else ranks - prior_ranks
            )
            h_cpu = h.detach().cpu()
            ranks_cpu = ranks.cpu()
            weights_cpu = weights.detach().cpu()
            transition_cpu = transition.cpu()
            rank_rows.extend(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "control": control,
                    "method": method.name,
                    "subject": subject,
                    "seed": seed,
                    "epoch": epoch,
                    "direction_index": index,
                    "h": float(h_cpu[index]),
                    "rank_descending": int(ranks_cpu[index]),
                    "assigned_weight": float(weights_cpu[index]),
                    "rank_transition": (
                        math.nan if prior_ranks is None else int(transition_cpu[index])
                    ),
                    "bank_hash": fingerprint,
                }
                for index in range(method.k)
            )
            prior_ranks = ranks
        if not finite:
            rows.extend(
                blank_row(method, dataset, phase, control, subject, seed, later)
                for later in range(epoch + 1, EPOCHS + 1)
            )
            break
    frame = pd.DataFrame(rows)[EPOCH_COLUMNS]
    ranks_frame = pd.DataFrame(rank_rows, columns=RANK_COLUMNS)
    evaluated = frame[np.isfinite(frame.lew)]
    final_lew = float(evaluated.lew.iloc[-1]) if not evaluated.empty else math.nan
    metadata: dict[str, object] = {
        "phase": phase,
        "control": control,
        "method": method.name,
        "subject": subject,
        "seed": seed,
        "rows": len(frame),
        "rank_rows": len(ranks_frame),
        "lew_initial": lew0,
        "lew_final": final_lew,
        "diverged": bool(frame.diverged.fillna(False).any()),
        "nan": bool(frame["nan"].fillna(False).any()),
        "optimization_ms": clocks.optimization_ms,
        "evaluation_ms": clocks.evaluation_ms,
        "status": "ok" if finite else "nonfinite",
    }
    return frame, ranks_frame, metadata


def run_complete(path: Path, rank_path: Path, method: Method) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
        if len(frame) != EPOCHS + 1 or int(frame.epoch.iloc[-1]) != EPOCHS:
            return False
        if method.sampling == "fixed":
            return rank_path.exists() and len(pd.read_csv(rank_path)) == EPOCHS * method.k
        return True
    except Exception:
        return False


def metadata_from_frame(path: Path, rank_path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    evaluated = frame[np.isfinite(frame.lew)]
    return {
        "phase": str(frame.phase.iloc[0]),
        "control": str(frame.control.iloc[0]),
        "method": str(frame.method.iloc[0]),
        "subject": int(frame.subject.iloc[0]),
        "seed": int(frame.seed.iloc[0]),
        "rows": len(frame),
        "rank_rows": len(pd.read_csv(rank_path)) if rank_path.exists() else 0,
        "lew_initial": float(evaluated.lew.iloc[0]),
        "lew_final": float(evaluated.lew.iloc[-1]),
        "diverged": bool(frame.diverged.fillna(False).any()),
        "nan": bool(frame["nan"].fillna(False).any()),
        "optimization_ms": float(frame.cumulative_optimization_ms.iloc[-1]),
        "evaluation_ms": float(frame.cumulative_evaluation_ms.iloc[-1]),
        "status": "cached_complete",
    }


def execute_grid(
    *,
    phase: str,
    dataset: str,
    subjects: Iterable[int],
    control: str,
    rerun: bool,
) -> list[dict[str, object]]:
    methods = ALL_METHODS
    subjects = tuple(subjects)
    manifest_path = OUT / f"MANIFEST_{phase}_{control}.json"
    records: list[dict[str, object]] = []
    total = len(subjects) * len(SEEDS) * len(methods)
    index = 0
    for subject in subjects:
        source, target, meta = load_cached_subject(dataset, subject, DEVICE)
        basis = SvecBasis(meta["d"], DEVICE, DTYPE)
        target_vec = basis.forward(target)
        for seed in SEEDS:
            fixed_state = build_fixed_bank_state(basis, target_vec, seed)
            for method in methods:
                index += 1
                run_path, rank_path = run_paths(phase, control, method, subject, seed)
                try:
                    if rerun or not run_complete(run_path, rank_path, method):
                        frame, ranks, metadata = train_one(
                            method,
                            source,
                            target,
                            dataset=dataset,
                            phase=phase,
                            control=control,
                            subject=subject,
                            seed=seed,
                            fixed_state=fixed_state if method.sampling == "fixed" else None,
                        )
                        run_path.parent.mkdir(parents=True, exist_ok=True)
                        frame.to_csv(run_path, index=False)
                        if method.sampling == "fixed":
                            rank_path.parent.mkdir(parents=True, exist_ok=True)
                            ranks.to_csv(rank_path, index=False)
                    else:
                        metadata = metadata_from_frame(run_path, rank_path)
                    record = {
                        **metadata,
                        "k": method.k,
                        "sampling": method.sampling,
                        "aggregation": method.aggregation,
                        "reference": method.reference,
                        "run_csv": str(run_path.relative_to(OUT)),
                        "rank_csv": str(rank_path.relative_to(OUT)) if method.sampling == "fixed" else None,
                        "error": None,
                    }
                    print(
                        f"[{phase} {control} {index:03d}/{total:03d}] s{subject:02d} seed={seed} "
                        f"{method.name:38s} LEW {record['lew_initial']:.3f}->{record['lew_final']:.3f}",
                        flush=True,
                    )
                except Exception as exc:
                    log_path = OUT / "logs" / f"{phase}_{control}_{method.name}_seed{seed}_s{subject:02d}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(traceback.format_exc())
                    record = {
                        "phase": phase,
                        "control": control,
                        "method": method.name,
                        "subject": subject,
                        "seed": seed,
                        "k": method.k,
                        "sampling": method.sampling,
                        "aggregation": method.aggregation,
                        "reference": method.reference,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "run_csv": str(run_path.relative_to(OUT)),
                        "rank_csv": str(rank_path.relative_to(OUT)) if method.sampling == "fixed" else None,
                    }
                    print(f"[ERROR] {record['error']}", file=sys.stderr, flush=True)
                records.append(record)
                dump_json(manifest_path, records)
        del source, target, target_vec
        torch.cuda.empty_cache()
    return records


def load_phase_frames(phase: str) -> pd.DataFrame:
    paths = sorted((OUT / "runs" / phase).glob("*/*/seed_*/subject_*.csv"))
    if not paths:
        raise RuntimeError(f"no run CSVs for {phase}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def load_phase_ranks(phase: str) -> pd.DataFrame:
    paths = sorted((OUT / "ranks" / phase).glob("*/*/seed_*/subject_*.csv"))
    if not paths:
        raise RuntimeError(f"no fixed-bank rank CSVs for {phase}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def relative_lew_auc(group: pd.DataFrame) -> float:
    evaluated = group[np.isfinite(group.lew)].sort_values("epoch")
    if len(evaluated) < 2 or int(evaluated.epoch.iloc[-1]) != EPOCHS:
        return math.inf
    return float(np.trapezoid(evaluated.relative_lew, evaluated.epoch) / EPOCHS)


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset", "phase", "control", "method", "sampling", "aggregation",
        "reference", "subject", "seed", "k", "sigma",
    ]
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=False):
        group = group.sort_values("epoch")
        evaluated = group[np.isfinite(group.lew)]
        initial_lew = float(evaluated.lew.iloc[0])
        final_lew = float(evaluated.lew.iloc[-1])
        losses = group[np.isfinite(group.training_power_loss)]
        initial_loss = float(losses.training_power_loss.iloc[0]) if not losses.empty else math.nan
        final_loss = float(losses.training_power_loss.iloc[-1]) if not losses.empty else math.nan
        training_reduction = (
            100.0 * (initial_loss - final_loss) / initial_loss
            if math.isfinite(initial_loss) and initial_loss != 0.0
            else math.nan
        )
        lew_reduction = 100.0 * (initial_lew - final_lew) / initial_lew
        row = dict(zip(keys, key))
        row.update(
            lew_initial=initial_lew,
            lew_final=final_lew,
            relative_lew_auc=relative_lew_auc(group),
            lew_reduction_pct=lew_reduction,
            training_loss_initial=initial_loss,
            training_loss_final=final_loss,
            training_loss_reduction_pct=training_reduction,
            train_minus_lew_reduction_pct=training_reduction - lew_reduction,
            divergence=bool(group.diverged.fillna(False).any()),
            nan=bool(group["nan"].fillna(False).any()),
            cumulative_optimization_ms=float(group.cumulative_optimization_ms.iloc[-1]),
            cumulative_evaluation_ms=float(group.cumulative_evaluation_ms.iloc[-1]),
            one_time_fixed_bank_sampling_ms=float(group.one_time_fixed_bank_sampling_ms.max()),
            one_time_fixed_target_projection_ms=float(group.one_time_fixed_target_projection_ms.max()),
            mean_gradient_norm=float(group.loc[group.epoch > 0, "raw_gradient_norm"].mean()),
            mean_update_norm=float(group.loc[group.epoch > 0, "applied_update_norm"].mean()),
            median_spectral_ess=float(group.loc[group.epoch > 0, "spectral_ess"].median()),
            median_spectral_entropy=float(group.loc[group.epoch > 0, "spectral_entropy"].median()),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    references = summary[summary.reference == True][
        ["phase", "control", "subject", "seed", "lew_final"]
    ].rename(columns={"lew_final": "reference_l500_lew_final"})
    summary = summary.merge(references, on=["phase", "control", "subject", "seed"], how="left")
    reach_epochs: list[float] = []
    reach_times: list[float] = []
    for record in summary.itertuples(index=False):
        block = frame[
            (frame.phase == record.phase)
            & (frame.control == record.control)
            & (frame.method == record.method)
            & (frame.subject == record.subject)
            & (frame.seed == record.seed)
            & np.isfinite(frame.lew)
        ].sort_values("epoch")
        hits = block[block.lew <= record.reference_l500_lew_final]
        if hits.empty:
            reach_epochs.append(math.nan)
            reach_times.append(math.nan)
        else:
            first = hits.iloc[0]
            reach_epochs.append(float(first.epoch))
            reach_times.append(float(first.cumulative_optimization_ms))
    summary["epoch_reach_l500_final_quality"] = reach_epochs
    summary["wall_ms_reach_l500_final_quality"] = reach_times
    return summary


def factorial_subject_results(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary[summary.reference == False]
    cells = primary.groupby(["phase", "control", "subject", "method"], as_index=False).agg(
        relative_lew_auc=("relative_lew_auc", "mean"),
        lew_final=("lew_final", "mean"),
    )
    auc = cells.pivot(index=["phase", "control", "subject"], columns="method", values="relative_lew_auc")
    final = cells.pivot(index=["phase", "control", "subject"], columns="method", values="lew_final")
    output = auc.reset_index().rename(
        columns={name: f"auc_{name}" for name in [method.name for method in PRIMARY_METHODS]}
    )
    final = final.reset_index().rename(
        columns={name: f"final_lew_{name}" for name in [method.name for method in PRIMARY_METHODS]}
    )
    output = output.merge(final, on=["phase", "control", "subject"])
    output["delta_resample_uniform"] = (
        output[f"auc_{PRIMARY_METHODS[1].name}"] - output[f"auc_{PRIMARY_METHODS[0].name}"]
    )
    output["delta_spectral_fixed"] = (
        output[f"auc_{PRIMARY_METHODS[2].name}"] - output[f"auc_{PRIMARY_METHODS[0].name}"]
    )
    output["delta_spectral_resampled"] = (
        output[f"auc_{PRIMARY_METHODS[3].name}"] - output[f"auc_{PRIMARY_METHODS[1].name}"]
    )
    output["interaction"] = output.delta_spectral_resampled - output.delta_spectral_fixed
    return output


def rank_dynamics(ranks: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "phase", "control", "method", "subject", "seed"]
    rows: list[dict[str, object]] = []
    for key, group in ranks.groupby(keys, sort=False):
        pivot = group.pivot(index="epoch", columns="direction_index", values="rank_descending").sort_index()
        correlations: list[float] = []
        overlaps: list[float] = []
        values = pivot.to_numpy()
        for previous, current in zip(values[:-1], values[1:]):
            correlations.append(float(np.corrcoef(previous, current)[0, 1]))
            overlaps.append(
                len(set(np.flatnonzero(previous <= 5)) & set(np.flatnonzero(current <= 5))) / 5.0
            )
        transition = group[np.isfinite(group.rank_transition)].rank_transition.abs()
        record = dict(zip(keys, key))
        record.update(
            mean_consecutive_spearman=float(np.mean(correlations)),
            median_consecutive_spearman=float(np.median(correlations)),
            mean_consecutive_top5_overlap=float(np.mean(overlaps)),
            fraction_directions_ever_top5=float((pivot <= 5).any(axis=0).mean()),
            mean_absolute_rank_transition=float(transition.mean()),
        )
        block = frame[
            (frame.phase == record["phase"])
            & (frame.control == record["control"])
            & (frame.method == record["method"])
            & (frame.subject == record["subject"])
            & (frame.seed == record["seed"])
            & (frame.epoch > 0)
        ]
        record["median_weight_entropy"] = float(block.spectral_entropy.median())
        record["median_effective_directions"] = float(block.spectral_ess.median())
        rows.append(record)
    return pd.DataFrame(rows)


def timing_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "direction_sampling_ms", "source_projection_ms", "target_projection_ms",
        "wasserstein_1d_ms", "sorting_aggregation_ms", "backward_ms",
        "optimizer_update_ms", "total_optimization_epoch_ms",
    ]
    positive = frame[frame.epoch > 0]
    aggregations = {column: (column, "mean") for column in columns}
    result = positive.groupby(
        ["dataset", "phase", "control", "method", "sampling", "aggregation", "reference", "k"],
        as_index=False,
    ).agg(**aggregations)
    setup = frame.groupby(["dataset", "phase", "control", "method"], as_index=False).agg(
        one_time_fixed_bank_sampling_ms=("one_time_fixed_bank_sampling_ms", "mean"),
        one_time_fixed_target_projection_ms=("one_time_fixed_target_projection_ms", "mean"),
    )
    return result.merge(setup, on=["dataset", "phase", "control", "method"], how="left")


def wallclock_targets(frame: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    fixed_name = "fixed_lns_k40_s0p5"
    comparators = ("resampled_spdsw_k40", "resampled_lns_k40_s0p5")
    rows: list[dict[str, object]] = []
    for control in CONTROLS:
        for subject in sorted(summary.subject.unique()):
            for seed in SEEDS:
                fixed = frame[
                    (frame.control == control)
                    & (frame.method == fixed_name)
                    & (frame.subject == subject)
                    & (frame.seed == seed)
                    & np.isfinite(frame.lew)
                ].sort_values("epoch")
                for comparator in comparators:
                    compared = frame[
                        (frame.control == control)
                        & (frame.method == comparator)
                        & (frame.subject == subject)
                        & (frame.seed == seed)
                        & np.isfinite(frame.lew)
                    ].sort_values("epoch")
                    target = float(compared.lew.iloc[-1])
                    hits = fixed[fixed.lew <= target]
                    reached = not hits.empty
                    first_time = float(hits.cumulative_optimization_ms.iloc[0]) if reached else math.nan
                    first_epoch = float(hits.epoch.iloc[0]) if reached else math.nan
                    comparator_time = float(compared.cumulative_optimization_ms.iloc[-1])
                    rows.append(
                        {
                            "phase": str(fixed.phase.iloc[0]),
                            "control": control,
                            "subject": int(subject),
                            "seed": int(seed),
                            "fixed_method": fixed_name,
                            "comparator": comparator,
                            "comparator_epoch500_lew": target,
                            "fixed_reached": reached,
                            "fixed_first_reach_epoch": first_epoch,
                            "fixed_first_reach_wall_ms": first_time,
                            "comparator_epoch500_wall_ms": comparator_time,
                            "wall_advantage": bool(reached and first_time < comparator_time),
                        }
                    )
    return pd.DataFrame(rows)


def effect_classification(subjects: pd.DataFrame, column: str) -> str:
    values = subjects[column]
    if int((values < 0.0).sum()) >= 2 and float(values.mean()) < 0.0:
        return "IMPROVE"
    if int((values > 0.0).sum()) >= 2 and float(values.mean()) > 0.0:
        return "WORSE"
    return "NULL"


def sampling_classification(subjects: pd.DataFrame) -> str:
    values = subjects.delta_resample_uniform
    if int((values > 0.0).sum()) >= 2 and float(values.mean()) > 0.0:
        return "FIXED"
    if int((values < 0.0).sum()) >= 2 and float(values.mean()) < 0.0:
        return "RESAMPLED"
    return "TIE"


def bank_audit(frame: pd.DataFrame) -> dict[str, object]:
    primary = frame[(frame.reference == False) & (frame.epoch > 0)].copy()
    fixed_uniform = primary[primary.method == "fixed_spdsw_k40"]
    fixed_spectral = primary[primary.method == "fixed_lns_k40_s0p5"]
    resampled_uniform = primary[primary.method == "resampled_spdsw_k40"]
    resampled_spectral = primary[primary.method == "resampled_lns_k40_s0p5"]
    join_keys = ["phase", "control", "subject", "seed", "epoch"]
    fixed_pair = fixed_uniform.merge(
        fixed_spectral, on=join_keys, suffixes=("_uniform", "_spectral")
    )
    resampled_pair = resampled_uniform.merge(
        resampled_spectral, on=join_keys, suffixes=("_uniform", "_spectral")
    )
    fixed_groups = fixed_uniform.groupby(["phase", "control", "subject", "seed"])
    resampled_groups = resampled_uniform.groupby(["phase", "control", "subject", "seed"])
    normalized = primary[(primary.control == "normalized_update") & np.isfinite(primary.applied_update_norm)]
    relative_update_error = (
        (normalized.applied_update_norm - NORMALIZED_STEP_TARGET).abs() / NORMALIZED_STEP_TARGET
    )
    initial = frame[(frame.reference == False) & (frame.epoch == 0)]
    initial_spread = initial.groupby(["phase", "control", "subject", "seed"]).lew.agg(
        lambda values: float(values.max() - values.min())
    )
    audit = {
        "primary_k_values": sorted(int(value) for value in primary.k.unique()),
        "fixed_pair_bank_hash_equal_all_epochs": bool(
            (fixed_pair.bank_hash_uniform == fixed_pair.bank_hash_spectral).all()
        ),
        "resampled_pair_bank_hash_equal_all_epochs": bool(
            (resampled_pair.bank_hash_uniform == resampled_pair.bank_hash_spectral).all()
        ),
        "fixed_pair_target_hash_equal_all_epochs": bool(
            (fixed_pair.target_projection_hash_uniform == fixed_pair.target_projection_hash_spectral).all()
        ),
        "resampled_pair_target_hash_equal_all_epochs": bool(
            (
                resampled_pair.target_projection_hash_uniform
                == resampled_pair.target_projection_hash_spectral
            ).all()
        ),
        "fixed_unique_bank_hash_counts": sorted(int(value) for value in fixed_groups.bank_hash.nunique().unique()),
        "resampled_unique_bank_hash_counts": sorted(
            int(value) for value in resampled_groups.bank_hash.nunique().unique()
        ),
        "fixed_unique_target_hash_counts": sorted(
            int(value) for value in fixed_groups.target_projection_hash.nunique().unique()
        ),
        "fixed_epoch_direction_sampling_ms_max": float(fixed_uniform.direction_sampling_ms.max()),
        "fixed_epoch_target_projection_ms_max": float(fixed_uniform.target_projection_ms.max()),
        "resampled_direction_sampling_ms_min": float(resampled_uniform.direction_sampling_ms.min()),
        "resampled_target_projection_ms_min": float(resampled_uniform.target_projection_ms.min()),
        "initial_lew_max_spread_across_methods": float(initial_spread.max()),
        "normalized_update_max_relative_error": float(relative_update_error.max()),
        "evaluation_kind": "exact LEW with no projection directions",
        "no_hierarchy_method": True,
        "bank_hash_kind_primary": "full_tensor_sha256",
        "bank_hash_examples": {
            "fixed": str(fixed_uniform.bank_hash.iloc[0]),
            "resampled_epoch1": str(resampled_uniform.sort_values("epoch").bank_hash.iloc[0]),
            "resampled_epoch500": str(resampled_uniform.sort_values("epoch").bank_hash.iloc[-1]),
        },
    }
    audit["pass"] = bool(
        audit["primary_k_values"] == [K]
        and audit["fixed_pair_bank_hash_equal_all_epochs"]
        and audit["resampled_pair_bank_hash_equal_all_epochs"]
        and audit["fixed_pair_target_hash_equal_all_epochs"]
        and audit["resampled_pair_target_hash_equal_all_epochs"]
        and audit["fixed_unique_bank_hash_counts"] == [1]
        and audit["resampled_unique_bank_hash_counts"] == [EPOCHS]
        and audit["fixed_unique_target_hash_counts"] == [1]
        and audit["fixed_epoch_direction_sampling_ms_max"] == 0.0
        and audit["fixed_epoch_target_projection_ms_max"] == 0.0
        and audit["initial_lew_max_spread_across_methods"] == 0.0
        and audit["normalized_update_max_relative_error"] <= 1e-10
    )
    return audit


def development_gate(
    summary: pd.DataFrame,
    subjects: pd.DataFrame,
    wallclock: pd.DataFrame,
    audit: dict[str, object],
) -> dict[str, object]:
    normalized_subjects = subjects[subjects.control == "normalized_update"]
    normalized_runs = summary[(summary.control == "normalized_update") & (summary.reference == False)]
    fixed_uniform = normalized_runs[normalized_runs.method == "fixed_spdsw_k40"]
    fixed_spectral = normalized_runs[normalized_runs.method == "fixed_lns_k40_s0p5"]
    paired = fixed_spectral[["subject", "seed", "relative_lew_auc"]].merge(
        fixed_uniform[["subject", "seed", "relative_lew_auc"]],
        on=["subject", "seed"],
        suffixes=("_spectral", "_uniform"),
    )
    paired["difference"] = paired.relative_lew_auc_spectral - paired.relative_lew_auc_uniform
    subject_difference = paired.groupby("subject").difference.mean()
    wall_primary = wallclock[wallclock.control == "normalized_update"]
    comparator_status: dict[str, object] = {}
    wall_advantage_any = False
    for comparator, block in wall_primary.groupby("comparator"):
        subject_favorable = block.groupby("subject").wall_advantage.sum() >= 2
        favorable_runs = int(block.wall_advantage.sum())
        favorable_subjects = int(subject_favorable.sum())
        passes = favorable_runs >= 6 and favorable_subjects >= 2
        comparator_status[str(comparator)] = {
            "favorable_runs": favorable_runs,
            "favorable_subjects": favorable_subjects,
            "passes": bool(passes),
        }
        wall_advantage_any = wall_advantage_any or bool(passes)
    no_extra_instability = int(fixed_spectral.divergence.sum()) <= int(fixed_uniform.divergence.sum()) and int(
        fixed_spectral.nan.sum()
    ) <= int(fixed_uniform.nan.sum())
    conditions = {
        "fixed_spectral_improves_at_least_2_subjects": int((subject_difference < 0.0).sum()) >= 2,
        "fixed_spectral_mean_auc_difference_favorable": float(paired.difference.mean()) < 0.0,
        "no_material_instability_increase": bool(no_extra_instability),
        "normalized_update_norms_matched": float(audit["normalized_update_max_relative_error"]) <= 1e-10,
        "fixed_spectral_wall_clock_advantage": wall_advantage_any,
        "no_posthoc_hyperparameter_change": True,
    }
    gate_pass = all(conditions.values()) and bool(audit["pass"])
    fixed_status = effect_classification(normalized_subjects, "delta_spectral_fixed")
    resampled_status = effect_classification(normalized_subjects, "delta_spectral_resampled")
    sampling_status = sampling_classification(normalized_subjects)
    fixed_lns_subject = normalized_subjects[f"auc_fixed_lns_k40_s0p5"]
    resampled_uniform_subject = normalized_subjects[f"auc_resampled_spdsw_k40"]
    if gate_pass:
        case = "A_strongest_positive"
    elif fixed_status == "WORSE" and resampled_status == "WORSE":
        case = "D_spectral_hurts_both"
    elif (
        sampling_status == "RESAMPLED"
        and int((resampled_uniform_subject < fixed_lns_subject).sum()) >= 2
        and float(resampled_uniform_subject.mean()) < float(fixed_lns_subject.mean())
    ):
        case = "C_resampling_remains_superior"
    elif sampling_status == "FIXED" and fixed_status != "IMPROVE":
        case = "B_fixed_helps_spectral_does_not"
    else:
        case = "mixed_or_null"
    return {
        "pass": gate_pass,
        "decision": "proceed_to_heldout_hgd" if gate_pass else "stop_after_development_gate",
        "case": case,
        "fixed_vs_resampled_result": sampling_status,
        "spectral_under_fixed": fixed_status,
        "spectral_under_resampled": resampled_status,
        "fixed_spectral_wall_clock_advantage": wall_advantage_any,
        "fixed_spectral_improved_subjects": int((subject_difference < 0.0).sum()),
        "mean_fixed_spectral_paired_auc_difference": float(paired.difference.mean()),
        "conditions": conditions,
        "wall_clock_comparators": comparator_status,
        "bank_audit_pass": bool(audit["pass"]),
    }


def frame_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def format_cell(value: object) -> str:
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
        "| " + " | ".join(format_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def aggregate_core(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.groupby(
        ["dataset", "phase", "control", "method", "sampling", "aggregation", "reference", "k", "sigma"],
        as_index=False,
    ).agg(
        run_count=("seed", "size"),
        subject_count=("subject", "nunique"),
        mean_relative_lew_auc=("relative_lew_auc", "mean"),
        std_relative_lew_auc=("relative_lew_auc", "std"),
        mean_final_lew=("lew_final", "mean"),
        mean_lew_reduction_pct=("lew_reduction_pct", "mean"),
        mean_training_loss_reduction_pct=("training_loss_reduction_pct", "mean"),
        mean_train_minus_lew_reduction_pct=("train_minus_lew_reduction_pct", "mean"),
        mean_optimization_ms=("cumulative_optimization_ms", "mean"),
        mean_one_time_bank_sampling_ms=("one_time_fixed_bank_sampling_ms", "mean"),
        mean_one_time_target_projection_ms=("one_time_fixed_target_projection_ms", "mean"),
        l500_quality_reached_runs=("epoch_reach_l500_final_quality", "count"),
        mean_epoch_reach_l500_quality=("epoch_reach_l500_final_quality", "mean"),
        mean_wall_ms_reach_l500_quality=("wall_ms_reach_l500_final_quality", "mean"),
        divergence_count=("divergence", "sum"),
        nan_count=("nan", "sum"),
    )


def plot_mean_curves(
    frame: pd.DataFrame,
    *,
    x: str,
    output: Path,
    xlabel: str,
    title: str,
) -> None:
    shown = [method.name for method in ALL_METHODS]
    colors = {
        "fixed_spdsw_k40": "#277da1",
        "resampled_spdsw_k40": "#43aa8b",
        "fixed_lns_k40_s0p5": "#f8961e",
        "resampled_lns_k40_s0p5": "#f94144",
        "resampled_spdsw_l500_reference": "#7b2cbf",
    }
    evaluated = frame[(frame.control == "normalized_update") & np.isfinite(frame.lew)]
    fig, axis = plt.subplots(figsize=(8.4, 5.1))
    for method in shown:
        block = evaluated[evaluated.method == method]
        grouped = block.groupby("epoch", as_index=False).agg(
            x_value=(x, "mean"),
            mean=("relative_lew", "mean"),
            minimum=("relative_lew", "min"),
            maximum=("relative_lew", "max"),
        )
        axis.plot(grouped.x_value, grouped["mean"], label=method, color=colors[method], linewidth=2)
        axis.fill_between(
            grouped.x_value, grouped.minimum, grouped.maximum, color=colors[method], alpha=0.08
        )
    axis.set(xlabel=xlabel, ylabel="relative exact LEW (lower is better)", title=title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_figures(
    development_frame: pd.DataFrame,
    development_subjects: pd.DataFrame,
    timing: pd.DataFrame,
    ranks: pd.DataFrame,
    rank_summary: pd.DataFrame,
) -> None:
    plot_mean_curves(
        development_frame,
        x="epoch",
        output=OUT / "fig_lew_vs_epoch.png",
        xlabel="epoch",
        title="HGD development: normalized-update LEW versus epoch",
    )
    plot_mean_curves(
        development_frame,
        x="cumulative_optimization_ms",
        output=OUT / "fig_lew_vs_wallclock.png",
        xlabel="cumulative optimization time (ms; LEW evaluation excluded)",
        title="HGD development: LEW versus optimization wall-clock",
    )
    plot_mean_curves(
        development_frame,
        x="cumulative_ambient_projection_count",
        output=OUT / "fig_lew_vs_projection_count.png",
        xlabel="cumulative source/target ambient directional projections",
        title="HGD development: LEW versus directional projection count",
    )

    components = [
        "direction_sampling_ms", "source_projection_ms", "target_projection_ms",
        "wasserstein_1d_ms", "sorting_aggregation_ms", "backward_ms", "optimizer_update_ms",
    ]
    block = timing[(timing.phase == "development") & (timing.control == "normalized_update")]
    fig, axis = plt.subplots(figsize=(10.0, 5.2))
    bottom = np.zeros(len(block))
    for component in components:
        values = block[component].to_numpy()
        axis.bar(block.method, values, bottom=bottom, label=component.removesuffix("_ms"))
        bottom += values
    axis.set(ylabel="mean milliseconds per optimization epoch", title="Timing decomposition")
    axis.tick_params(axis="x", rotation=25, labelsize=8)
    axis.legend(fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_timing_breakdown.png", dpi=180)
    plt.close(fig)

    subject_block = development_subjects[development_subjects.control == "normalized_update"]
    effects = [
        ("delta_resample_uniform", "resampling | uniform"),
        ("delta_spectral_fixed", "spectral | fixed"),
        ("delta_spectral_resampled", "spectral | resampled"),
        ("interaction", "interaction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharey=True)
    for axis, (column, title) in zip(axes.flat, effects):
        axis.bar(subject_block.subject.astype(str), subject_block[column], color="#577590")
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set(title=title, xlabel="subject", ylabel="paired relative-LEW AUC difference")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("2x2 factorial effects (negative favors first named change)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_factorial_effects.png", dpi=180)
    plt.close(fig)

    rank_block = rank_summary[
        (rank_summary.phase == "development") & (rank_summary.control == "normalized_update")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    metrics = [
        ("mean_consecutive_spearman", "consecutive-rank Spearman"),
        ("mean_consecutive_top5_overlap", "consecutive top-5 overlap"),
        ("fraction_directions_ever_top5", "fraction ever entering top-5"),
    ]
    for axis, (column, title) in zip(axes, metrics):
        grouped = rank_block.groupby("method")[column].agg(["mean", "min", "max"]).reset_index()
        error = np.vstack([grouped["mean"] - grouped["min"], grouped["max"] - grouped["mean"]])
        axis.bar(grouped.method, grouped["mean"], yerr=error, capsize=3, color=["#277da1", "#f8961e"])
        axis.set(title=title, ylabel="mean and run range")
        axis.tick_params(axis="x", rotation=20, labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rank_persistence.png", dpi=180)
    plt.close(fig)

    fixed = development_frame[
        (development_frame.control == "normalized_update")
        & (development_frame.sampling == "fixed")
        & (development_frame.epoch > 0)
    ].copy()
    initial_loss = fixed[fixed.epoch == 1][
        ["method", "subject", "seed", "training_power_loss"]
    ].rename(columns={"training_power_loss": "initial_training_loss"})
    fixed = fixed.merge(initial_loss, on=["method", "subject", "seed"])
    fixed["training_loss_reduction_pct"] = 100.0 * (
        fixed.initial_training_loss - fixed.training_power_loss
    ) / fixed.initial_training_loss
    fixed_eval = fixed[np.isfinite(fixed.lew_reduction_pct)]
    grouped = fixed_eval.groupby(["method", "epoch"], as_index=False).agg(
        train_reduction=("training_loss_reduction_pct", "mean"),
        lew_reduction=("lew_reduction_pct", "mean"),
    )
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    for method, method_block in grouped.groupby("method"):
        axis.plot(
            method_block.train_reduction,
            method_block.lew_reduction,
            marker="o",
            label=method,
        )
    limits = [
        min(float(grouped.train_reduction.min()), float(grouped.lew_reduction.min())),
        max(float(grouped.train_reduction.max()), float(grouped.lew_reduction.max())),
    ]
    axis.plot(limits, limits, linestyle="--", color="black", linewidth=1, label="equal reduction")
    axis.set(
        xlabel="fixed training-bank loss reduction (%)",
        ylabel="independent exact-LEW reduction (%)",
        title="Fixed-bank optimization versus independent LEW",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_loss_vs_independent_lew.png", dpi=180)
    plt.close(fig)

    representative = ranks[
        (ranks.phase == "development")
        & (ranks.control == "normalized_update")
        & (ranks.method == "fixed_lns_k40_s0p5")
        & (ranks.subject == DEVELOPMENT_SUBJECTS[0])
        & (ranks.seed == SEEDS[0])
    ]
    frequency = representative.groupby("direction_index").rank_descending.apply(lambda values: float((values <= 5).mean()))
    chosen = list(frequency.sort_values(ascending=False).head(8).index)
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for direction in chosen:
        line = representative[representative.direction_index == direction]
        axis.plot(line.epoch, line.rank_descending, label=f"A{direction}", alpha=0.85)
    axis.invert_yaxis()
    axis.set(
        xlabel="epoch",
        ylabel="descending cost rank (1 is highest)",
        title="Representative fixed-bank rank trajectories (descriptive top-5 frequency selection)",
    )
    axis.legend(fontsize=7, ncol=4)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rank_trajectories.png", dpi=180)
    plt.close(fig)


def write_claim_ledger(gate: dict[str, object]) -> None:
    text = f"""# Claim ledger

## Established implementation facts

- Every primary method uses 40 direct Frobenius-unit ambient directions; no
  SPDHSW hierarchy or mixture direction is constructed.
- Fixed banks permit deterministic target-projection caching and persistent
  direction identities. Resampled banks use the frozen triangular epoch-seed
  sequence and have no cross-epoch identity interpretation.
- Lognormal-spectral weights are fixed at sigma 0.5 while their detached rank
  assignment changes with the observed directional costs.

## Finite realized statements

- Bank-hash equality, target-cache reuse, update-norm equality, timing, rank
  persistence, and LEW outcomes are finite-run audit or empirical statements.
- Fixed finite directions are not claimed to define a full metric.
- Independently resampled realized estimates are not claimed to obey a
  realization-wise triangle inequality.

## Empirical finding

- Registered development case: `{gate['case']}`.
- Spectral-under-fixed classification: `{gate['spectral_under_fixed']}`;
  spectral-under-resampled classification: `{gate['spectral_under_resampled']}`.
- These HGD alignment results measure exact LEW only and make no downstream
  classification claim.

## Unsupported or prohibited claims

- Nonzero spectral weighting is not the same population target as uniform
  SPDSW and is not described as an unbiased estimator of it.
- A fixed bank is not claimed to solve ambient directional coverage.
- Fixed spectral weighting is not claimed to replace resampling unless every
  registered development gate item passes.
- No hierarchy benefit is inferred because hierarchy is absent here.
"""
    (OUT / "CLAIM_LEDGER.md").write_text(text)


def write_report(
    core: pd.DataFrame,
    subjects: pd.DataFrame,
    timing: pd.DataFrame,
    rank_summary: pd.DataFrame,
    wallclock: pd.DataFrame,
    gate_bundle: dict[str, object],
) -> None:
    gate = gate_bundle["development"]
    heldout_run = gate_bundle.get("heldout") is not None
    normalized_core = core[
        (core.phase == "development")
        & (core.control == "normalized_update")
        & (core.reference == False)
    ][
        [
            "method", "mean_relative_lew_auc", "std_relative_lew_auc", "mean_final_lew",
            "mean_lew_reduction_pct", "mean_optimization_ms", "divergence_count", "nan_count",
        ]
    ]
    subject_block = subjects[
        (subjects.phase == "development") & (subjects.control == "normalized_update")
    ][
        [
            "subject", "delta_resample_uniform", "delta_spectral_fixed",
            "delta_spectral_resampled", "interaction",
        ]
    ]
    timing_block = timing[
        (timing.phase == "development") & (timing.control == "normalized_update")
    ]
    timing_columns = [
        "method", "direction_sampling_ms", "source_projection_ms", "target_projection_ms",
        "wasserstein_1d_ms", "sorting_aggregation_ms", "backward_ms",
        "optimizer_update_ms", "total_optimization_epoch_ms",
        "one_time_fixed_bank_sampling_ms", "one_time_fixed_target_projection_ms",
    ]
    rank_block = rank_summary[
        (rank_summary.phase == "development") & (rank_summary.control == "normalized_update")
    ].groupby("method", as_index=False).agg(
        consecutive_spearman=("mean_consecutive_spearman", "mean"),
        top5_overlap=("mean_consecutive_top5_overlap", "mean"),
        ever_top5_fraction=("fraction_directions_ever_top5", "mean"),
        median_effective_directions=("median_effective_directions", "mean"),
    )
    overfit = core[
        (core.phase == "development")
        & (core.control == "normalized_update")
        & (core.sampling == "fixed")
    ][
        [
            "method", "mean_training_loss_reduction_pct", "mean_lew_reduction_pct",
            "mean_train_minus_lew_reduction_pct",
        ]
    ]
    wall = wallclock[wallclock.control == "normalized_update"].groupby("comparator", as_index=False).agg(
        favorable_runs=("wall_advantage", "sum"),
        reached_runs=("fixed_reached", "sum"),
        mean_fixed_reach_ms=("fixed_first_reach_wall_ms", "mean"),
        mean_comparator_epoch500_ms=("comparator_epoch500_wall_ms", "mean"),
    )
    failures = sorted((OUT / "logs").glob("*.log")) if (OUT / "logs").exists() else []
    lines = [
        "- theorem/regression tests: PASS",
        f"- development fixed-vs-resampled result: {gate['fixed_vs_resampled_result']}",
        f"- development spectral-under-fixed result: {gate['spectral_under_fixed']}",
        f"- development spectral-under-resampled result: {gate['spectral_under_resampled']}",
        f"- fixed-spectral wall-clock advantage: {'YES' if gate['fixed_spectral_wall_clock_advantage'] else 'NO'}",
        f"- held-out HGD run: {'YES' if heldout_run else 'NO'}",
        "- proceed to hierarchy after this experiment: NO",
        "",
        "# Fixed versus resampled Spectral SPDSW v1",
        "",
        f"Registered decision: `{gate_bundle['decision']}`. Development classification: `{gate['case']}`.",
        "Previous direct and hierarchical null results remain frozen and are not reinterpreted.",
        "",
        "## 1. Exact protocol",
        "",
        "- HGD cached 0train -> 1test log-SPD blocks; development subjects 2, 3, 4; seeds 6398, 3654, 1788; 500 epochs; exact LEW every 25 epochs.",
        "- The 2x2 factorial uses direct k=40 directions only and sigma=0.5 only. Primary updates use the frozen direct-SPDSW step norm 7.3473173386245945; secondary raw SGD uses LR=3000 for every method.",
        "- Fixed uniform and spectral runs share one physical fixed-bank tensor and cached target projection per subject/seed/control. Resampled pairs use the same deterministic epoch bank.",
        "- All tensors are float64; AMP, autocast, TF32, clipping, early stopping, method-specific LR, and hierarchy are absent.",
        "- Cumulative optimization time includes fixed one-time bank/target setup and excludes exact-LEW evaluation.",
        "",
        "### L500 compatibility decision",
        "",
        "No prior result matched the complete development protocol. The prior spectral-development L500 runs cover only seed 6398, 100 epochs, and step norm 3.8661. The overnight L500 runs cover subjects 1, 7, 14 under raw LR=10000. Therefore both new controls were rerun for subjects 2, 3, 4 and all three seeds.",
        "",
        "## 2. Normalized-update 2x2 table",
        "",
        frame_markdown(normalized_core),
        "",
        "## 3. Subject-wise factorial differences",
        "",
        "All differences are right-minus-left relative-LEW AUC; negative is favorable to resampling or spectral weighting according to the column definition.",
        "",
        frame_markdown(subject_block),
        "",
        "## 4–5. Epoch and wall-clock curves",
        "",
        "See `fig_lew_vs_epoch.png`, `fig_lew_vs_wallclock.png`, and `fig_lew_vs_projection_count.png`. Projection count is source-plus-target directional projections: fixed uses 40 cached target projections once plus 40 source projections per epoch; resampled uses 40 source and 40 target projections per epoch.",
        "",
        "Fixed-spectral time-to-resampled-quality diagnostic:",
        "",
        frame_markdown(wall),
        "",
        "## 6. Timing decomposition",
        "",
        frame_markdown(timing_block[timing_columns]),
        "",
        "## 7. Fixed-bank rank dynamics",
        "",
        frame_markdown(rank_block),
        "",
        "Rank persistence is descriptive and did not tune sigma. Resampled banks have no asserted cross-epoch identity. See `fig_rank_persistence.png` and the additional descriptive `fig_rank_trajectories.png`.",
        "",
        "## 8. Fixed-bank overfitting diagnostic",
        "",
        "Training-loss reduction uses the first pre-update training loss (epoch 1) as baseline; LEW reduction uses epoch 0 to epoch 500.",
        "",
        frame_markdown(overfit),
        "",
        "## 9. Gate decision",
        "",
        f"The development gate {'PASSED' if gate['pass'] else 'FAILED'}: `{gate_bundle['decision']}`.",
        "",
        "```json",
        json.dumps(gate["conditions"], indent=2, sort_keys=True),
        "```",
        "",
        "## 10. Nulls, failures, and scope",
        "",
        f"- Execution failure logs: {len(failures)}. Every log, if any, remains under `logs/`.",
        "- Raw-SGD outcomes are retained in CORE_RESULTS.csv and SUBJECT_RESULTS.csv but do not determine the primary conclusion.",
        "- Missing L500-quality reach times mean the k=40 method never reached its paired reference's epoch-500 LEW.",
        "- CLAIM_LEDGER.md separates implementation facts, finite-run statements, empirical findings, and prohibited claims.",
        "",
        "## Commands and provenance",
        "",
        "```bash",
        "nvidia-smi -i 3",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m experiments.run_fixed_vs_resampled_spectral_spdsw --phase prepare",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m pytest -q --junitxml=results/fixed_vs_resampled_spectral_spdsw_v1/TEST_RESULTS.xml",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_fixed_vs_resampled_spectral_spdsw --phase development_normalized",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_fixed_vs_resampled_spectral_spdsw --phase development_raw",
        "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_fixed_vs_resampled_spectral_spdsw --phase analyze",
        "```",
        "",
        f"- Branch: `{BRANCH}`; analysis invocation commit: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()}`.",
        f"- Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}, physical GPU 3.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def analyze_all() -> dict[str, object]:
    phases = [path.name for path in sorted((OUT / "runs").iterdir()) if path.is_dir()]
    frames = pd.concat([load_phase_frames(phase) for phase in phases], ignore_index=True)
    rank_frames = pd.concat([load_phase_ranks(phase) for phase in phases], ignore_index=True)
    summary = summarize_runs(frames)
    subjects = factorial_subject_results(summary)
    timings = timing_breakdown(frames)
    rank_summary = rank_dynamics(rank_frames, frames)
    development_frame = frames[frames.phase == "development"]
    development_summary = summary[summary.phase == "development"]
    development_subjects = subjects[subjects.phase == "development"]
    development_wall = wallclock_targets(development_frame, development_summary)
    audit = bank_audit(development_frame)
    gate = development_gate(development_summary, development_subjects, development_wall, audit)
    heldout_gate = None
    if "heldout_hgd" in phases:
        heldout_frame = frames[frames.phase == "heldout_hgd"]
        heldout_summary = summary[summary.phase == "heldout_hgd"]
        heldout_subjects = subjects[subjects.phase == "heldout_hgd"]
        heldout_wall = wallclock_targets(heldout_frame, heldout_summary)
        heldout_audit = bank_audit(heldout_frame)
        heldout_gate = development_gate(heldout_summary, heldout_subjects, heldout_wall, heldout_audit)
    decision = (
        "stop_after_development_gate"
        if not gate["pass"]
        else "proceed_to_bnci_transfer"
        if heldout_gate and heldout_gate["pass"]
        else "stop_after_heldout_gate"
        if heldout_gate
        else "proceed_to_heldout_hgd"
    )
    gate_bundle: dict[str, object] = {
        "decision": decision,
        "development": gate,
        "heldout": heldout_gate,
        "bnci_executed": "bnci_transfer" in phases,
        "proceed_to_hierarchy": False,
    }
    core = aggregate_core(summary)
    core.to_csv(OUT / "CORE_RESULTS.csv", index=False)
    subjects.to_csv(OUT / "SUBJECT_RESULTS.csv", index=False)
    timings.to_csv(OUT / "TIMING_BREAKDOWN.csv", index=False)
    rank_summary.to_csv(OUT / "RANK_DYNAMICS.csv", index=False)
    development_wall.to_csv(OUT / "WALLCLOCK_TARGETS.csv", index=False)
    dump_json(OUT / "BANK_AUDIT.json", audit)
    manifests: list[dict[str, object]] = []
    for path in sorted(OUT.glob("MANIFEST_*.json")):
        manifests.extend(json.loads(path.read_text()))
    dump_json(OUT / "RUN_MANIFEST.json", manifests)
    dump_json(OUT / "GATE.json", gate_bundle)
    write_figures(development_frame, development_subjects, timings, rank_frames, rank_summary)
    write_claim_ledger(gate)
    write_report(core, subjects, timings, rank_summary, development_wall, gate_bundle)
    verify_frozen()
    return gate_bundle


def frozen_inputs() -> dict[str, str]:
    files = [
        PROJECT / "coherent_slicing" / "spectral.py",
        PROJECT / "coherent_slicing" / "aggregations.py",
        PROJECT / "experiments" / "run_moabb_pilot.py",
        PROJECT / "experiments" / "run_overnight.py",
        PROJECT / "experiments" / "run_logspectral_spdhsw.py",
        EXTERNAL / "evobank" / "data.py",
        EXTERNAL / "evobank" / "lew.py",
        EXTERNAL / "evobank" / "ot1d.py",
        EXTERNAL / "evobank" / "svec.py",
    ]
    payload = {str(path): sha256(path) for path in files}
    for relative in ("results/coherent_sw_overnight", "results/lognormal_spectral_spdhsw_v1"):
        payload[str(PROJECT / relative)] = tree_sha256(PROJECT / relative)
    return payload


def verify_frozen() -> None:
    path = OUT / "FROZEN_INPUT_HASHES.json"
    current = frozen_inputs()
    if not path.exists():
        dump_json(path, current)
    elif json.loads(path.read_text()) != current:
        raise RuntimeError("a frozen source or prior result bundle changed")


def tests_pass() -> bool:
    path = OUT / "TEST_RESULTS.xml"
    if not path.exists():
        return False
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return bool(suites) and all(
        int(suite.attrib.get("failures", 0)) == 0 and int(suite.attrib.get("errors", 0)) == 0
        for suite in suites
    )


def prepare() -> None:
    if current_branch() != BRANCH:
        raise RuntimeError(f"must run only on {BRANCH}")
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / "CONFIG.json"
    if config_path.exists() and json.loads(config_path.read_text()) != CONFIG:
        raise RuntimeError("refusing to change frozen CONFIG.json")
    if not config_path.exists():
        dump_json(config_path, CONFIG)
    verify_frozen()
    device = check_gpu3()
    environment = {
        "branch": current_branch(),
        "commit_at_prepare": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "dtype": str(DTYPE),
        "amp": False,
        "autocast": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device,
    }
    dump_json(OUT / "ENVIRONMENT.json", environment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "prepare",
            "development_normalized",
            "development_raw",
            "analyze",
            "heldout_normalized",
            "heldout_raw",
            "bnci_normalized",
            "bnci_raw",
        ),
    )
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    configure_numerics()
    if args.phase == "prepare":
        prepare()
        print(f"prepared {OUT}")
        return 0
    if current_branch() != BRANCH:
        raise RuntimeError(f"must run only on {BRANCH}")
    verify_frozen()
    if not tests_pass():
        raise RuntimeError("full regression suite has not passed; scientific runs are prohibited")
    if args.phase == "analyze":
        gate = analyze_all()
        print(f"[GATE] {gate['decision']}", flush=True)
        return 0
    check_gpu3()
    if args.phase == "development_normalized":
        execute_grid(
            phase="development",
            dataset="Schirrmeister2017",
            subjects=DEVELOPMENT_SUBJECTS,
            control="normalized_update",
            rerun=args.rerun,
        )
    elif args.phase == "development_raw":
        normalized_manifest = OUT / "MANIFEST_development_normalized_update.json"
        if not normalized_manifest.exists() or len(json.loads(normalized_manifest.read_text())) != 45:
            raise RuntimeError("all normalized development runs must finish before raw SGD")
        execute_grid(
            phase="development",
            dataset="Schirrmeister2017",
            subjects=DEVELOPMENT_SUBJECTS,
            control="raw_sgd_lr3000",
            rerun=args.rerun,
        )
    elif args.phase.startswith("heldout_"):
        gate_path = OUT / "GATE.json"
        if not gate_path.exists() or not json.loads(gate_path.read_text())["development"]["pass"]:
            raise RuntimeError("development gate failed or is absent; held-out HGD is prohibited")
        control = "normalized_update" if args.phase.endswith("normalized") else "raw_sgd_lr3000"
        if control == "raw_sgd_lr3000":
            manifest = OUT / "MANIFEST_heldout_hgd_normalized_update.json"
            if not manifest.exists() or len(json.loads(manifest.read_text())) != 45:
                raise RuntimeError("held-out normalized runs must finish before held-out raw SGD")
        execute_grid(
            phase="heldout_hgd",
            dataset="Schirrmeister2017",
            subjects=HELDOUT_SUBJECTS,
            control=control,
            rerun=args.rerun,
        )
    elif args.phase.startswith("bnci_"):
        gate_path = OUT / "GATE.json"
        if not gate_path.exists():
            raise RuntimeError("held-out gate is absent; BNCI is prohibited")
        bundle = json.loads(gate_path.read_text())
        if not bundle.get("heldout") or not bundle["heldout"]["pass"]:
            raise RuntimeError("held-out HGD gate failed; BNCI is prohibited")
        control = "normalized_update" if args.phase.endswith("normalized") else "raw_sgd_lr3000"
        if control == "raw_sgd_lr3000":
            manifest = OUT / "MANIFEST_bnci_transfer_normalized_update.json"
            if not manifest.exists() or len(json.loads(manifest.read_text())) != 45:
                raise RuntimeError("BNCI normalized runs must finish before BNCI raw SGD")
        execute_grid(
            phase="bnci_transfer",
            dataset="BNCI2014_001",
            subjects=(1, 3, 8),
            control=control,
            rerun=args.rerun,
        )
    else:
        raise RuntimeError(args.phase)
    verify_frozen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
