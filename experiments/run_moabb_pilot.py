#!/usr/bin/env python
"""Preregistered direct-SPDSW coherent-risk MOABB pilot.

The script consumes the existing read-only log-SPD caches and exact LEW
evaluator.  It never imports or introduces a hierarchical projection in the
training loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from coherent_slicing import (
    cvar_power,
    ebsw_exp_power,
    entropic_power,
    evar_power,
    power_ebsw_power,
    sw_power,
    w_p_power_per_direction,
)


PROJECT = Path(__file__).resolve().parents[1]
SIBLING = Path("/home/pikachu/EBSPDSW")
sys.path.insert(0, str(SIBLING))
from evobank import data as sibling_data  # noqa: E402
from evobank.lew import LEWEvaluator  # noqa: E402
from evobank.svec import SvecBasis  # noqa: E402


DEFAULT_OUT = PROJECT / "results" / "coherent_sw" / "moabb_pilot_v1"
DTYPE = torch.float64

PREREGISTERED = {
    "version": "moabb_pilot_v1",
    "created_before_pilot": True,
    "theorem_gate": "21 pytest cases and E1-E5 must pass",
    "datasets": {
        "BNCI2014_001": {"subjects": [1], "split": "session 1 -> session 2", "role": "sanity"},
        "Schirrmeister2017": {
            "subjects": [1, 7, 14],
            "development_subjects": [2, 3, 4],
            "split": "0train -> 1test",
            "channels": 128,
            "role": "primary",
        },
    },
    "seed": 6398,
    "epochs": 500,
    "development_epochs": 100,
    "p": 2,
    "lew_every": 25,
    "direction_budgets": [40, 500],
    "direction_resampling": "existing cumulative seed + epoch rule; common across L=40 methods",
    "beta_grid": [0.1, 1.0, 10.0],
    "beta_selection": "lowest mean relative exact-LEW AUC on HGD development subjects 2,3,4",
    "learning_rate_grid": [3000.0, 10000.0, 30000.0],
    "learning_rate_selection": (
        "per-method global grid point whose median initial development-subject update norm "
        "is closest to SPDSW-L40 at lr=10000; no evaluation outcome used"
    ),
    "normalized_update": (
        "constant per-epoch step norm equal to the median initial SPDSW-L40 lr=10000 "
        "update norm over the subjects in that dataset arm"
    ),
    "ebsw_gradient": "full gradient through self-normalized weights",
    "power_ebsw_gradient": "full gradient through pure-power weights",
    "evar_gradient": "Danskin; detached optimal KL-ball weights",
    "cvar_gradient": "detached optimal density-cap subgradient",
    "aggregation_overhead_gate_seconds_per_epoch": 0.001,
    "stop_rule": {
        "secondary_lr_auc_improved_subjects": 2,
        "normalized_auc_improved_subjects": 2,
        "no_divergence_increase": True,
        "exclude_sampled_max_endpoint": True,
        "negligible_aggregation_overhead": True,
    },
}

FROZEN_SOURCES = [
    Path("/home/pikachu/edubridge_SPDHSW/spdsw/spdsw.py"),
    Path("/home/pikachu/edubridge_SPDHSW/spdsw/spdhsw.py"),
    SIBLING / "evobank" / "data.py",
    SIBLING / "evobank" / "svec.py",
    SIBLING / "evobank" / "ot1d.py",
    SIBLING / "evobank" / "lew.py",
    SIBLING / "evobank" / "trainer.py",
    SIBLING / "evobank" / "baselines.py",
]

EPOCH_COLUMNS = [
    "dataset",
    "subject",
    "seed",
    "control",
    "method",
    "family",
    "L",
    "epoch",
    "direction_seed",
    "training_power_loss",
    "rooted_distance",
    "lew",
    "gap_closure_pct",
    "gradient_norm",
    "update_norm",
    "mean_h",
    "std_h",
    "max_h",
    "entropy",
    "kl_uniform",
    "ess",
    "beta_star",
    "achieved_kl",
    "active_tail_count",
    "aggregation_seconds",
    "optimization_seconds_cum",
    "evaluation_seconds_cum",
    "wall_seconds_cum",
    "cumulative_ambient_projections",
    "learning_rate",
    "normalized_step_target",
    "nan",
    "diverged",
    "status",
]


@dataclass(frozen=True)
class Method:
    name: str
    family: str
    L: int = 40
    beta: float | None = None
    gamma: float | None = None
    kappa: float | None = None
    alpha: float | None = None


def method_grid(beta_ebsw: float, beta_entropic: float) -> list[Method]:
    methods = [
        Method("spdsw_l40", "sw"),
        Method(f"ebsw_exp_b{tag(beta_ebsw)}", "ebsw_exp", beta=beta_ebsw),
        Method("power_ebsw_g1", "power_ebsw", gamma=1.0),
        Method(f"entropic_b{tag(beta_entropic)}", "entropic", beta=beta_entropic),
    ]
    methods.extend(Method(f"evar_k{tag(value)}", "evar", kappa=value) for value in (0.1, 0.5, 1.0, 2.0))
    methods.extend(Method(f"cvar_a{tag(value)}", "cvar", alpha=value) for value in (0.5, 0.2, 0.1, 0.05))
    methods.append(Method("sampled_max_l40", "sampled_max"))
    methods.append(Method("spdsw_l500", "sw", L=500))
    return methods


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {str(path): sha256(path) for path in FROZEN_SOURCES}


def direction_seed(seed: int, epoch_zero_based: int) -> int:
    """Mirror the existing cumulative ``random_state += step`` sequence."""
    return int(seed + epoch_zero_based * (epoch_zero_based + 1) // 2)


def frobenius_directions(
    count: int,
    basis: SvecBasis,
    seed: int,
) -> torch.Tensor:
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


def distribution_diagnostics(weights: torch.Tensor) -> tuple[float, float, float]:
    weights = weights.detach()
    positive = weights > 0
    entropy = float(-(weights[positive] * weights[positive].log()).sum())
    kl = float((weights[positive] * (weights[positive].log() + math.log(weights.numel()))).sum())
    effective = float(1.0 / weights.square().sum())
    return entropy, kl, effective


def aggregate(method: Method, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, str]:
    uniform = torch.full_like(h, 1.0 / h.numel())
    if method.family == "sw":
        return sw_power(h), uniform, math.nan, "mean"
    if method.family == "ebsw_exp":
        weights = torch.softmax(float(method.beta) * h.detach(), dim=0)
        return ebsw_exp_power(h, float(method.beta), full_gradient=True), weights, float(method.beta), "fixed_beta"
    if method.family == "power_ebsw":
        if float(h.detach().max()) == 0.0:
            weights = uniform
        else:
            weights = h.detach().pow(float(method.gamma))
            weights = weights / weights.sum()
        return power_ebsw_power(h, float(method.gamma), full_gradient=True), weights, math.nan, "fixed_gamma"
    if method.family == "entropic":
        weights = torch.softmax(float(method.beta) * h.detach(), dim=0)
        return entropic_power(h, float(method.beta)), weights, float(method.beta), "fixed_beta"
    if method.family == "evar":
        result = evar_power(h, float(method.kappa))
        return result.value, result.weights, result.beta, result.status
    if method.family == "cvar":
        result = cvar_power(h, float(method.alpha))
        return result.value, result.weights, math.nan, "exact_cap"
    if method.family == "sampled_max":
        weights = torch.zeros_like(h)
        weights[int(torch.argmax(h.detach()))] = 1.0
        return h.max(), weights, math.inf, "sampled_max"
    raise ValueError(method.family)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def initial_gradient_norm(method: Method, source: torch.Tensor, target: torch.Tensor, seed: int) -> float:
    device = source.device
    basis = SvecBasis(source.shape[-1], device, DTYPE)
    parameter = basis.forward(source).clone().requires_grad_(True)
    target_vec = basis.forward(target)
    directions = frobenius_directions(method.L, basis, direction_seed(seed, 0))
    h = w_p_power_per_direction((parameter @ directions.T).T, (target_vec @ directions.T).T, p=2)
    loss, _, _, _ = aggregate(method, h)
    loss.backward()
    value = float(parameter.grad.norm())
    del parameter, target_vec, directions, h, loss
    return value


def nan_row(
    dataset: str,
    subject: int,
    seed: int,
    control: str,
    method: Method,
    epoch: int,
    learning_rate: float,
    normalized_target: float,
    optimization_seconds: float,
    evaluation_seconds: float,
    status: str,
) -> dict:
    row = {column: math.nan for column in EPOCH_COLUMNS}
    row.update(
        dataset=dataset,
        subject=subject,
        seed=seed,
        control=control,
        method=method.name,
        family=method.family,
        L=method.L,
        epoch=epoch,
        direction_seed=direction_seed(seed, max(0, epoch - 1)),
        learning_rate=learning_rate,
        normalized_step_target=normalized_target,
        optimization_seconds_cum=optimization_seconds,
        evaluation_seconds_cum=evaluation_seconds,
        wall_seconds_cum=optimization_seconds,
        cumulative_ambient_projections=method.L * epoch,
        nan=True,
        diverged=True,
        status=status,
    )
    return row


def train_one(
    method: Method,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    dataset: str,
    subject: int,
    seed: int,
    epochs: int,
    lew_every: int,
    control: str,
    learning_rate: float,
    normalized_target: float,
) -> tuple[pd.DataFrame, dict]:
    if control not in (
        "normalized",
        "selected_lr",
        "development",
        "common_lr10000",
        "lr3000",
    ):
        raise ValueError(control)
    device = source.device
    basis = SvecBasis(source.shape[-1], device, DTYPE)
    source0 = source.to(device=device, dtype=DTYPE)
    target = target.to(device=device, dtype=DTYPE)
    parameter = basis.forward(source0).clone().requires_grad_(True)
    target_vec = basis.forward(target)
    evaluator = LEWEvaluator(target)
    eval_seconds = 0.0
    optimization_seconds = 0.0

    tick = time.perf_counter()
    lew0 = evaluator(source0)
    eval_seconds += time.perf_counter() - tick
    evaluator.set_baseline(lew0)
    rows = [
        {
            "dataset": dataset,
            "subject": subject,
            "seed": seed,
            "control": control,
            "method": method.name,
            "family": method.family,
            "L": method.L,
            "epoch": 0,
            "direction_seed": direction_seed(seed, 0),
            "training_power_loss": math.nan,
            "rooted_distance": math.nan,
            "lew": lew0,
            "gap_closure_pct": 0.0,
            "gradient_norm": math.nan,
            "update_norm": math.nan,
            "mean_h": math.nan,
            "std_h": math.nan,
            "max_h": math.nan,
            "entropy": math.nan,
            "kl_uniform": math.nan,
            "ess": math.nan,
            "beta_star": math.nan,
            "achieved_kl": math.nan,
            "active_tail_count": math.nan,
            "aggregation_seconds": 0.0,
            "optimization_seconds_cum": 0.0,
            "evaluation_seconds_cum": eval_seconds,
            "wall_seconds_cum": 0.0,
            "cumulative_ambient_projections": 0,
            "learning_rate": learning_rate,
            "normalized_step_target": normalized_target,
            "nan": False,
            "diverged": False,
            "status": "initial",
        }
    ]
    failed = False
    for zero_epoch in range(epochs):
        synchronize(device)
        epoch_tick = time.perf_counter()
        sampled_seed = direction_seed(seed, zero_epoch)
        directions = frobenius_directions(method.L, basis, sampled_seed)
        projected_source = parameter @ directions.T
        projected_target = target_vec @ directions.T
        h = w_p_power_per_direction(projected_source.T, projected_target.T, p=2)
        synchronize(device)
        aggregate_tick = time.perf_counter()
        loss, weights, beta_star, status = aggregate(method, h)
        synchronize(device)
        aggregation_seconds = time.perf_counter() - aggregate_tick
        loss.backward()
        gradient_norm = float(parameter.grad.norm())
        if control in ("normalized", "development"):
            if gradient_norm > 0.0 and math.isfinite(gradient_norm):
                update = -float(normalized_target) * parameter.grad / gradient_norm
            else:
                update = torch.zeros_like(parameter)
        else:
            update = -float(learning_rate) * parameter.grad
        update_norm = float(update.norm())
        with torch.no_grad():
            parameter.add_(update)
        parameter.grad = None
        synchronize(device)
        optimization_seconds += time.perf_counter() - epoch_tick

        epoch = zero_epoch + 1
        finite = bool(torch.isfinite(parameter).all()) and bool(torch.isfinite(loss))
        entropy_value, kl_value, ess_value = distribution_diagnostics(weights)
        lew_value = math.nan
        closure = math.nan
        diverged = False
        if finite and (epoch % lew_every == 0 or epoch == epochs):
            eval_tick = time.perf_counter()
            lew_value = evaluator(basis.inverse(parameter.detach()))
            eval_seconds += time.perf_counter() - eval_tick
            closure = evaluator.closed_pct(lew_value)
            diverged = evaluator.diverged(lew_value)
        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "seed": seed,
                "control": control,
                "method": method.name,
                "family": method.family,
                "L": method.L,
                "epoch": epoch,
                "direction_seed": sampled_seed,
                "training_power_loss": float(loss.detach()),
                "rooted_distance": float(loss.detach().clamp_min(0).sqrt()),
                "lew": lew_value,
                "gap_closure_pct": closure,
                "gradient_norm": gradient_norm,
                "update_norm": update_norm,
                "mean_h": float(h.detach().mean()),
                "std_h": float(h.detach().std(unbiased=False)),
                "max_h": float(h.detach().max()),
                "entropy": entropy_value,
                "kl_uniform": kl_value,
                "ess": ess_value,
                "beta_star": beta_star,
                "achieved_kl": kl_value if method.family == "evar" else math.nan,
                "active_tail_count": int((weights.detach() > 0).sum()) if method.family == "cvar" else math.nan,
                "aggregation_seconds": aggregation_seconds,
                "optimization_seconds_cum": optimization_seconds,
                "evaluation_seconds_cum": eval_seconds,
                "wall_seconds_cum": optimization_seconds,
                "cumulative_ambient_projections": method.L * epoch,
                "learning_rate": learning_rate,
                "normalized_step_target": normalized_target,
                "nan": not finite,
                "diverged": diverged or not finite,
                "status": status,
            }
        )
        if not finite:
            failed = True
            for later in range(epoch + 1, epochs + 1):
                rows.append(
                    nan_row(
                        dataset,
                        subject,
                        seed,
                        control,
                        method,
                        later,
                        learning_rate,
                        normalized_target,
                        optimization_seconds,
                        eval_seconds,
                        "non_finite_after_epoch_%d" % epoch,
                    )
                )
            break
    frame = pd.DataFrame(rows)[EPOCH_COLUMNS]
    final_lew = float(frame.lew.dropna().iloc[-1]) if frame.lew.notna().any() else math.nan
    metadata = {
        "lew_initial": lew0,
        "lew_final": final_lew,
        "diverged": failed or evaluator.diverged(final_lew),
        "optimization_seconds": optimization_seconds,
        "evaluation_seconds": eval_seconds,
        "aggregation_seconds": float(frame.aggregation_seconds.fillna(0).sum()),
    }
    return frame, metadata


def relative_lew_auc(frame: pd.DataFrame) -> float:
    evaluated = frame[np.isfinite(frame.lew)].sort_values("epoch")
    if len(evaluated) < 2:
        return math.inf
    baseline = float(evaluated.lew.iloc[0])
    horizon = float(evaluated.epoch.iloc[-1] - evaluated.epoch.iloc[0])
    if not math.isfinite(baseline) or baseline <= 0 or horizon <= 0:
        return math.inf
    return float(np.trapezoid(evaluated.lew / baseline, evaluated.epoch) / horizon)


def complete_csv(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
        return len(frame) == epochs + 1 and int(frame.epoch.iloc[-1]) == epochs
    except Exception:
        return False


def load_subject(dataset: str, subject: int, device: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    source, target, meta = sibling_data.load(dataset, subject, device)
    if dataset == "Schirrmeister2017":
        if str(meta["source_level"]) != "0train" or str(meta["target_level"]) != "1test":
            raise RuntimeError(f"unexpected HGD split: {meta}")
        if int(meta["d"]) != 128:
            raise RuntimeError(f"HGD is not the 128-channel cache: {meta}")
    return source, target, meta


def normalized_target_for_subjects(dataset: str, subjects: Iterable[int], device: str) -> float:
    values = []
    sw = Method("spdsw_l40", "sw")
    for subject in subjects:
        source, target, _ = load_subject(dataset, subject, device)
        values.append(10000.0 * initial_gradient_norm(sw, source, target, PREREGISTERED["seed"]))
        del source, target
    return float(np.median(values))


def calibrate_beta(out: Path, device: str, dev_target: float, rerun: bool) -> dict[str, float]:
    development = out / "development" / "beta_grid"
    rows = []
    for family in ("ebsw_exp", "entropic"):
        for beta in PREREGISTERED["beta_grid"]:
            method = Method(f"{family}_b{tag(beta)}", family, beta=beta)
            for subject in PREREGISTERED["datasets"]["Schirrmeister2017"]["development_subjects"]:
                path = development / family / f"beta_{tag(beta)}" / f"subject_{subject:02d}.csv"
                if rerun or not complete_csv(path, PREREGISTERED["development_epochs"]):
                    source, target, _ = load_subject("Schirrmeister2017", subject, device)
                    frame, _ = train_one(
                        method,
                        source,
                        target,
                        dataset="Schirrmeister2017",
                        subject=subject,
                        seed=PREREGISTERED["seed"],
                        epochs=PREREGISTERED["development_epochs"],
                        lew_every=PREREGISTERED["lew_every"],
                        control="development",
                        learning_rate=math.nan,
                        normalized_target=dev_target,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(path, index=False)
                    del source, target
                frame = pd.read_csv(path)
                rows.append(
                    {
                        "family": family,
                        "beta": beta,
                        "subject": subject,
                        "relative_lew_auc": relative_lew_auc(frame),
                        "lew_final": float(frame.lew.dropna().iloc[-1]),
                        "diverged": bool(frame["diverged"].fillna(False).any()),
                    }
                )
                print(f"[DEV beta] {family} beta={beta:g} s{subject:02d} AUC={rows[-1]['relative_lew_auc']:.6f}")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "development" / "beta_grid.csv", index=False)
    aggregate_frame = (
        frame.groupby(["family", "beta"], as_index=False)
        .agg(mean_relative_lew_auc=("relative_lew_auc", "mean"), divergence_count=("diverged", "sum"))
        .sort_values(["family", "divergence_count", "mean_relative_lew_auc", "beta"])
    )
    aggregate_frame.to_csv(out / "development" / "beta_grid_summary.csv", index=False)
    selected = {}
    for family in ("ebsw_exp", "entropic"):
        selected[family] = float(aggregate_frame[aggregate_frame.family == family].iloc[0].beta)
    return selected


def select_learning_rates(
    out: Path,
    device: str,
    methods: list[Method],
    dev_target: float,
) -> dict[str, float]:
    gradients = []
    subjects = PREREGISTERED["datasets"]["Schirrmeister2017"]["development_subjects"]
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for subject in subjects:
        source, target, _ = load_subject("Schirrmeister2017", subject, device)
        cache[subject] = (source, target)
    for method in methods:
        for subject in subjects:
            source, target = cache[subject]
            value = initial_gradient_norm(method, source, target, PREREGISTERED["seed"])
            gradients.append({"method": method.name, "family": method.family, "subject": subject, "gradient_norm": value})
    frame = pd.DataFrame(gradients)
    candidates = []
    selected = {}
    for method in methods:
        median_gradient = float(frame[frame.method == method.name].gradient_norm.median())
        for learning_rate in PREREGISTERED["learning_rate_grid"]:
            update = learning_rate * median_gradient
            mismatch = abs(math.log(max(update, 1e-300) / max(dev_target, 1e-300)))
            candidates.append(
                {
                    "method": method.name,
                    "family": method.family,
                    "median_initial_gradient_norm": median_gradient,
                    "candidate_lr": learning_rate,
                    "median_update_norm": update,
                    "log_step_mismatch": mismatch,
                }
            )
        block = [row for row in candidates if row["method"] == method.name]
        selected[method.name] = float(min(block, key=lambda row: (row["log_step_mismatch"], row["candidate_lr"]))["candidate_lr"])
    frame.to_csv(out / "development" / "initial_gradient_norms.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "development" / "learning_rate_selection.csv", index=False)
    del cache
    return selected


def run_arm(
    out: Path,
    dataset: str,
    subjects: list[int],
    methods: list[Method],
    control: str,
    normalized_target: float,
    selected_lrs: dict[str, float],
    device: str,
    rerun: bool,
) -> list[dict]:
    manifest = []
    for subject in subjects:
        source, target, meta = load_subject(dataset, subject, device)
        for method in methods:
            path = out / "runs" / dataset / control / method.name / f"subject_{subject:02d}.csv"
            try:
                if rerun or not complete_csv(path, PREREGISTERED["epochs"]):
                    frame, metadata = train_one(
                        method,
                        source,
                        target,
                        dataset=dataset,
                        subject=subject,
                        seed=PREREGISTERED["seed"],
                        epochs=PREREGISTERED["epochs"],
                        lew_every=PREREGISTERED["lew_every"],
                        control=control,
                        learning_rate=selected_lrs.get(method.name, 10000.0),
                        normalized_target=normalized_target,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(path, index=False)
                else:
                    frame = pd.read_csv(path)
                    metadata = {
                        "lew_initial": float(frame.lew.dropna().iloc[0]),
                        "lew_final": float(frame.lew.dropna().iloc[-1]),
                        "diverged": bool(frame["diverged"].fillna(False).any()),
                        "optimization_seconds": float(frame.optimization_seconds_cum.iloc[-1]),
                        "evaluation_seconds": float(frame.evaluation_seconds_cum.iloc[-1]),
                        "aggregation_seconds": float(frame.aggregation_seconds.fillna(0).sum()),
                    }
                record = {
                    "dataset": dataset,
                    "subject": subject,
                    "method": method.name,
                    "family": method.family,
                    "L": method.L,
                    "control": control,
                    "path": str(path),
                    "n_source": meta["n_source"],
                    "n_target": meta["n_target"],
                    "d": meta["d"],
                    "status": "ok",
                    "error": "",
                    **metadata,
                }
                print(
                    f"[PILOT] {dataset} {control} s{subject:02d} {method.name:20s} "
                    f"LEW {metadata['lew_initial']:.4f}->{metadata['lew_final']:.4f} "
                    f"opt={metadata['optimization_seconds']:.3f}s",
                    flush=True,
                )
            except Exception as exc:
                log = out / "logs" / f"{dataset}_{control}_{method.name}_s{subject:02d}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(traceback.format_exc())
                record = {
                    "dataset": dataset,
                    "subject": subject,
                    "method": method.name,
                    "family": method.family,
                    "L": method.L,
                    "control": control,
                    "path": str(path),
                    "n_source": meta["n_source"],
                    "n_target": meta["n_target"],
                    "d": meta["d"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[FAIL] {record['error']}", file=sys.stderr, flush=True)
            manifest.append(record)
            pd.DataFrame(manifest).to_csv(out / f"manifest_{dataset}_{control}.csv", index=False)
        del source, target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return manifest


def all_hgd_frames(out: Path) -> pd.DataFrame:
    frames = []
    root = out / "runs" / "Schirrmeister2017"
    for path in root.glob("*/*/subject_*.csv"):
        frame = pd.read_csv(path)
        frames.append(frame)
    if not frames:
        raise RuntimeError("no HGD run CSVs found")
    return pd.concat(frames, ignore_index=True)


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "subject", "seed", "control", "method", "family", "L"]
    for key, group in frame.groupby(keys, sort=False):
        evaluated = group[np.isfinite(group.lew)].sort_values("epoch")
        initial = float(evaluated.lew.iloc[0]) if len(evaluated) else math.nan
        final = float(evaluated.lew.iloc[-1]) if len(evaluated) else math.nan
        rows.append(
            {
                **dict(zip(keys, key)),
                "lew_initial": initial,
                "lew_final": final,
                "relative_lew_auc": relative_lew_auc(group),
                "epoch500_gap_closure_pct": 100.0 * (initial - final) / initial,
                "diverged": bool(group["diverged"].fillna(False).any()) or not math.isfinite(final) or final > initial,
                "nan_count": int(group["nan"].fillna(False).sum()),
                "optimization_seconds": float(group.optimization_seconds_cum.iloc[-1]),
                "evaluation_seconds": float(group.evaluation_seconds_cum.iloc[-1]),
                "aggregation_seconds": float(group.aggregation_seconds.fillna(0).sum()),
                "aggregation_ms_per_epoch": 1000.0 * float(group.aggregation_seconds.fillna(0).sum()) / max(int(group.epoch.max()), 1),
                "ambient_projections": int(group.cumulative_ambient_projections.max()),
            }
        )
    return pd.DataFrame(rows)


def make_plots(out: Path, frame: pd.DataFrame) -> None:
    display = frame[np.isfinite(frame.lew)].copy()
    for x, name, label in (
        ("epoch", "fig_lew_vs_epoch.png", "epoch"),
        ("optimization_seconds_cum", "fig_lew_vs_wall.png", "optimization wall time, evaluation excluded (s)"),
        ("cumulative_ambient_projections", "fig_lew_vs_projections.png", "cumulative ambient directions"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
        for axis, subject in zip(axes, (1, 7, 14)):
            block = display[(display.control == "normalized") & (display.subject == subject)]
            for method, group in block.groupby("method"):
                axis.plot(group[x], group.lew, label=method, linewidth=1)
            axis.set_title(f"HGD subject {subject}")
            axis.set_xlabel(label)
            axis.set_ylabel("exact-OT LEW")
            axis.grid(alpha=0.25)
        axes[-1].legend(fontsize=5, ncol=2)
        figure.tight_layout()
        figure.savefig(out / name, dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    diagnostics = frame[(frame.control == "normalized") & (frame.epoch > 0)]
    for method, group in diagnostics.groupby("method"):
        curve = group.groupby("epoch", as_index=False).agg(
            gradient_norm=("gradient_norm", "median"), update_norm=("update_norm", "median")
        )
        axes[0].plot(curve.epoch, curve.gradient_norm, label=method, linewidth=0.8)
        axes[1].plot(curve.epoch, curve.update_norm, label=method, linewidth=0.8)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("median gradient norm")
    axes[1].set_ylabel("median parameter-update norm")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=5, ncol=2)
    figure.tight_layout()
    figure.savefig(out / "fig_gradient_update_stability.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method, group in diagnostics.groupby("method"):
        curve = group.groupby("epoch", as_index=False).agg(entropy=("entropy", "median"), kl=("kl_uniform", "median"), ess=("ess", "median"))
        axes[0].plot(curve.epoch, curve.entropy, label=method, linewidth=0.8)
        axes[1].plot(curve.epoch, curve.kl, label=method, linewidth=0.8)
        axes[2].plot(curve.epoch, curve.ess, label=method, linewidth=0.8)
    for axis, ylabel in zip(axes, ("entropy", "KL to uniform", "ESS")):
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=5, ncol=2)
    figure.tight_layout()
    figure.savefig(out / "fig_entropy_kl_ess.png", dpi=180)
    plt.close(figure)


def evaluate_gate(out: Path, summary: pd.DataFrame, methods: list[Method]) -> dict:
    hgd = summary[summary.dataset == "Schirrmeister2017"]
    candidates = [method for method in methods if method.family in ("evar", "cvar")]
    records = []
    for method in candidates:
        row = {"method": method.name, "family": method.family}
        improvement_counts = {}
        no_divergence = True
        for control in ("selected_lr", "normalized"):
            block = hgd[hgd.control == control]
            baseline = block[block.method == "spdsw_l40"].set_index("subject")
            candidate = block[block.method == method.name].set_index("subject")
            common = baseline.index.intersection(candidate.index)
            improved = sum(
                float(candidate.loc[subject, "relative_lew_auc"])
                < float(baseline.loc[subject, "relative_lew_auc"])
                for subject in common
            )
            improvement_counts[control] = int(improved)
            no_divergence &= int(candidate.loc[common].diverged.sum()) <= int(baseline.loc[common].diverged.sum())
        overhead = float(hgd[(hgd.control == "normalized") & (hgd.method == method.name)].aggregation_ms_per_epoch.median())
        if method.family == "evar":
            not_endpoint = float(method.kappa) < math.log(method.L)
        else:
            not_endpoint = float(method.alpha) > 1.0 / method.L
        row.update(
            selected_lr_improved_subjects=improvement_counts.get("selected_lr", 0),
            normalized_improved_subjects=improvement_counts.get("normalized", 0),
            no_divergence_increase=bool(no_divergence),
            not_sampled_max_endpoint=bool(not_endpoint),
            aggregation_ms_per_epoch=overhead,
            negligible_aggregation_overhead=overhead
            <= 1000.0 * PREREGISTERED["aggregation_overhead_gate_seconds_per_epoch"],
        )
        row["passes"] = bool(
            row["selected_lr_improved_subjects"] >= 2
            and row["normalized_improved_subjects"] >= 2
            and row["no_divergence_increase"]
            and row["not_sampled_max_endpoint"]
            and row["negligible_aggregation_overhead"]
        )
        records.append(row)
    gate_frame = pd.DataFrame(records)
    gate_frame.to_csv(out / "stop_rule_by_setting.csv", index=False)
    passing = gate_frame[gate_frame.passes].method.tolist()
    gate = {
        "pass": bool(passing),
        "passing_settings": passing,
        "decision": "eligible_for_separate_followup_specification" if passing else "stop_null_result_no_expansion",
        "all_subject_seed_expansion_run": False,
        "hierarchical_stage_implemented": False,
    }
    json_dump(out / "stop_rule.json", gate)
    return gate


def write_report(out: Path, summary: pd.DataFrame, gate: dict, selected: dict, failures: list[dict]) -> None:
    hgd = summary[summary.dataset == "Schirrmeister2017"]
    normalized = hgd[hgd.control == "normalized"]
    best = normalized.sort_values("relative_lew_auc").head(5)
    lines = [
        "# Direct SPDSW coherent-risk MOABB pilot",
        "",
        f"Status: {'completed' if not failures else 'completed with failures'}. Stop-rule decision: `{gate['decision']}`.",
        "",
        "## Frozen selections",
        "",
        f"- EBSW-exp beta: {selected['beta']['ebsw_exp']:g}; entropic beta: {selected['beta']['entropic']:g}.",
        "- Learning rates were selected globally by method from `{3000,10000,30000}` using only initial development-subject gradient norms.",
        f"- HGD normalized step target: {selected['normalized_targets']['Schirrmeister2017']:.8g}.",
        "",
        "## Results and neutral interpretation",
        "",
        "Lower relative LEW AUC is better. The five lowest normalized-control subject/run rows were:",
        "",
        best[["subject", "method", "relative_lew_auc", "epoch500_gap_closure_pct", "diverged"]].to_markdown(index=False),
        "",
        f"The preregistered gate {'passed for ' + ', '.join(gate['passing_settings']) if gate['pass'] else 'did not pass for any intermediate EVaR/CVaR setting'}.",
        "No all-subject/all-seed expansion was run. No hierarchical EVaR/CVaR method was implemented.",
        "",
        "Common random numbers mean every method uses the same deterministic direction tensor at a given epoch and L. "
        "The epoch-0 directional cost vector is therefore identical across L=40 methods. After the first update, method-specific particle states differ, so their numerical h vectors appropriately differ even though the directions remain paired.",
        "",
        "## Failures and negative results",
        "",
    ]
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- No execution errors were recorded.")
    if not gate["pass"]:
        lines.append("- The stop rule failed; this null result is terminal for the requested pilot and no extra hyperparameters were tried.")
    lines.extend(
        [
            "- Sampled Max-SPDSW is a finite-bank endpoint reference, not a sphere-optimized Max-SW solver.",
            "- Aggregation overhead excludes exact-OT evaluation time; evaluation seconds are logged separately.",
            "",
            "## Outputs",
            "",
            "Per-epoch CSVs are under `runs/`; development-only calibration is under `development/`; "
            "`pilot_summary.csv`, `stop_rule_by_setting.csv`, figures, configs, and frozen-source hashes are in this directory.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def validate_prerequisites() -> None:
    theorem = PROJECT / "results" / "theorem_regression.xml"
    euclidean = PROJECT / "results" / "coherent_sw" / "euclidean_v1" / "summary.json"
    if not theorem.exists():
        raise RuntimeError("theorem regression artifact is missing")
    payload = json.loads(euclidean.read_text()) if euclidean.exists() else {}
    if payload.get("failures"):
        raise RuntimeError(f"Euclidean gate failed: {payload['failures']}")
    if set(payload.get("summaries", {})) != {"E1", "E2", "E3", "E4", "E5"}:
        raise RuntimeError("Euclidean E1-E5 completion artifact is missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--skip-development", action="store_true", help="requires an existing selected_config.json")
    args = parser.parse_args()
    validate_prerequisites()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.out.mkdir(parents=True, exist_ok=True)

    prereg = {
        **PREREGISTERED,
        "device": str(device),
        "device_name": torch.cuda.get_device_properties(device).name if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    prereg_path = args.out / "preregistered_config.json"
    if prereg_path.exists() and json.loads(prereg_path.read_text()) != prereg:
        raise RuntimeError(f"refusing to overwrite different preregistration: {prereg_path}")
    if not prereg_path.exists():
        json_dump(prereg_path, prereg)
    before_hashes = source_hashes()
    hashes_path = args.out / "frozen_source_hashes.json"
    if hashes_path.exists() and json.loads(hashes_path.read_text()) != before_hashes:
        raise RuntimeError("a frozen source changed since the pilot was preregistered")
    if not hashes_path.exists():
        json_dump(hashes_path, before_hashes)

    selected_path = args.out / "selected_config.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
    else:
        if args.skip_development:
            raise RuntimeError("--skip-development needs an existing selected_config.json")
        dev_subjects = PREREGISTERED["datasets"]["Schirrmeister2017"]["development_subjects"]
        dev_target = normalized_target_for_subjects("Schirrmeister2017", dev_subjects, args.device)
        beta = calibrate_beta(args.out, args.device, dev_target, args.rerun)
        methods = method_grid(beta["ebsw_exp"], beta["entropic"])
        lrs = select_learning_rates(args.out, args.device, methods, dev_target)
        normalized_targets = {
            "Schirrmeister2017": normalized_target_for_subjects(
                "Schirrmeister2017", PREREGISTERED["datasets"]["Schirrmeister2017"]["subjects"], args.device
            ),
            "BNCI2014_001": normalized_target_for_subjects(
                "BNCI2014_001", PREREGISTERED["datasets"]["BNCI2014_001"]["subjects"], args.device
            ),
        }
        selected = {
            "beta": beta,
            "learning_rates": lrs,
            "normalized_targets": normalized_targets,
            "development_normalized_target": dev_target,
            "selection_frozen_before_primary_pilot": True,
        }
        json_dump(selected_path, selected)

    methods = method_grid(selected["beta"]["ebsw_exp"], selected["beta"]["entropic"])
    manifests = []
    manifests += run_arm(
        args.out,
        "BNCI2014_001",
        PREREGISTERED["datasets"]["BNCI2014_001"]["subjects"],
        methods,
        "normalized",
        selected["normalized_targets"]["BNCI2014_001"],
        selected["learning_rates"],
        args.device,
        args.rerun,
    )
    pilot_subjects = PREREGISTERED["datasets"]["Schirrmeister2017"]["subjects"]
    for control in ("normalized", "selected_lr"):
        manifests += run_arm(
            args.out,
            "Schirrmeister2017",
            pilot_subjects,
            methods,
            control,
            selected["normalized_targets"]["Schirrmeister2017"],
            selected["learning_rates"],
            args.device,
            args.rerun,
        )

    all_frames = []
    for path in (args.out / "runs").glob("*/*/*/subject_*.csv"):
        all_frames.append(pd.read_csv(path))
    combined = pd.concat(all_frames, ignore_index=True)
    summary = summarize_runs(combined)
    summary.to_csv(args.out / "pilot_summary.csv", index=False)
    (args.out / "pilot_summary.md").write_text(summary.to_markdown(index=False) + "\n")
    hgd = combined[combined.dataset == "Schirrmeister2017"]
    make_plots(args.out, hgd)
    gate = evaluate_gate(args.out, summary, methods)
    failures = [record for record in manifests if record.get("status") != "ok"]
    write_report(args.out, summary, gate, selected, failures)

    after_hashes = source_hashes()
    if after_hashes != before_hashes:
        raise RuntimeError("frozen source hash changed during the pilot")
    print(f"[DONE] gate={gate['decision']} -> {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
