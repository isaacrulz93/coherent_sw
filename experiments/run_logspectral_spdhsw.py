#!/usr/bin/env python
"""Falsification-first lognormal-spectral normalized-SPDHSW pilot."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from coherent_slicing import lognormal_spectral_power, lognormal_spectral_weights


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results" / "lognormal_spectral_spdhsw_v1"
EXTERNAL = Path("/home/pikachu/EBSPDSW")
sys.path.insert(0, str(EXTERNAL))

from evobank.data import load as load_cached_subject  # noqa: E402
from evobank.lew import LEWEvaluator  # noqa: E402
from evobank.ot1d import w2_squared_per_direction  # noqa: E402
from evobank.svec import SvecBasis  # noqa: E402


DTYPE = torch.float64
PHYSICAL_GPU = 3
DEVICE = torch.device("cuda:3")
SIGMAS = (0.0, 0.5, 1.0, 1.25, 1.5)
NONZERO_SIGMAS = SIGMAS[1:]
K = 40
L_HIER = 500
SYNTHETIC_DIMENSIONS = (253, 2016, 8256)
SYNTHETIC_DRAWS = 200
DEV_SUBJECTS = (2, 3, 4)
DEV_SEED = 6398
DEV_EPOCHS = 100
LEW_EVERY = 25
RAW_REFERENCE_LR = 10000.0
HELDOUT_SUBJECTS = (1, 7, 14)
HELDOUT_SEEDS = (6398, 3654, 1788)
HELDOUT_EPOCHS = 500
RAW_LR_GRID = (1000.0, 3000.0, 5000.0, 10000.0)
_SPECTRUM_CACHE: dict[tuple[int, float], torch.Tensor] = {}


FROZEN_FILES = {
    Path("/home/pikachu/EBSPDSW/evobank/bank.py"): "895d288b1863af2ee46285e039704522ea6e13e35746b9bfdd85d48b8aa8bf0a",
    Path("/home/pikachu/EBSPDSW/evobank/data.py"): "17f0e97ac34edd7a71da696e5ae02b5ec4c253852c45547e6cb71ba2b9a2c6dd",
    Path("/home/pikachu/EBSPDSW/evobank/lew.py"): "4b8b64311b47bc6a438a8fd1e93b9906f2da9f1642a558745aaf580ea12d5a63",
    Path("/home/pikachu/EBSPDSW/evobank/make_tables.py"): "01d42b19c22bb478845ca1b309e1a39179abf9b5e75e5d7ac9c42abb9374eac4",
    Path("/home/pikachu/EBSPDSW/evobank/ot1d.py"): "a5ab21c6f9d0f58cd8aaf2ae0caca077ad035fadbb45a6c87e47ff6a933cb958",
    Path("/home/pikachu/EBSPDSW/evobank/svec.py"): "98becc334f71404d6ff3aec00f81f4e444f9b84d747edee03584d5bd243175ea",
    Path("/home/pikachu/EBSPDSW/evobank/trainer.py"): "45ece814c0a83bde53ef2f999bce96998f9d82a9e1623ef43dd922a778754cb7",
    Path("/home/pikachu/edubridge_SPDHSW/spdsw/spdhsw.py"): "84c296ed6612883332d780e9db6e1a790fd7bd693a9eaa23f9ae0a87fc7045b3",
    Path("/home/pikachu/edubridge_SPDHSW/spdsw/spdsw.py"): "a1c7ee9e0512b7a1e45d3e6218d9a62357c33ac968f34e3208d07779e3015bf7",
    PROJECT / "coherent_slicing" / "aggregations.py": "fa11e9e303357495ee79ba0c601e6427f87e4872ec89a6a4ceaf73a7fdfd2c8a",
    PROJECT / "experiments" / "run_moabb_pilot.py": "c85a466d30e696a05271cf2519e2fc57f9cd4f6a5d6388b6c460a91b1b0b208c",
    PROJECT / "experiments" / "run_overnight.py": "1a4a4e81d3b4f5727ebd816a638d77f5e682b3d927a9fbb2b4b9ad57bb3a7954",
    PROJECT / "tests" / "test_coherent_slicing.py": "d54a8e3a1ca44c5575381208dec62e5edc9dc739e462a54f3f8800513f960ff8",
}
OVERNIGHT_MANIFEST_SHA = "e40bb36ee02d8384617628f7f3de975e2c67ea80f3103e34eded73f139ae99e9"


@dataclass(frozen=True)
class Method:
    name: str
    family: str
    L: int
    sigma: float
    hierarchical: bool


EPOCH_COLUMNS = [
    "dataset",
    "phase",
    "control",
    "method",
    "family",
    "hierarchical",
    "subject",
    "seed",
    "epoch",
    "k",
    "L",
    "sigma",
    "training_power_loss",
    "rooted_distance",
    "lew",
    "gap_closure_pct",
    "gradient_norm",
    "update_norm",
    "mean_h",
    "std_h",
    "max_h",
    "min_h",
    "spectral_entropy",
    "spectral_ess",
    "spectral_max_weight",
    "spectral_top10_weight",
    "gram_condition",
    "aggregation_seconds",
    "optimization_seconds_cum",
    "evaluation_seconds_cum",
    "cumulative_ambient_projections",
    "learning_rate",
    "normalized_step_target",
    "bank_seed",
    "mix_seed",
    "nan",
    "diverged",
    "status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def overnight_manifest_sha() -> str:
    digest = hashlib.sha256()
    root = PROJECT / "results" / "coherent_sw_overnight"
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(PROJECT)
        digest.update(f"{sha256(path)}  {relative}\n".encode())
    return digest.hexdigest()


def verify_frozen() -> None:
    mismatches = [str(path) for path, expected in FROZEN_FILES.items() if sha256(path) != expected]
    if overnight_manifest_sha() != OVERNIGHT_MANIFEST_SHA:
        mismatches.append("results/coherent_sw_overnight aggregate manifest")
    if mismatches:
        raise RuntimeError(f"frozen input changed: {mismatches}")


def configure_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def check_gpu3() -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= PHYSICAL_GPU:
        raise RuntimeError("physical GPU 3 unavailable; GPU phases must not switch devices")
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
    compute_processes = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if compute_processes:
        raise RuntimeError(f"GPU compute process contamination detected; refusing device switch: {compute_processes}")
    torch.cuda.set_device(DEVICE)
    properties = torch.cuda.get_device_properties(DEVICE)
    return {
        "physical_gpu": PHYSICAL_GPU,
        "torch_device": str(DEVICE),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_processes_before_cuda_initialization": compute_processes,
    }


def sync() -> None:
    torch.cuda.synchronize(DEVICE)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def frame_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def sigma_tag(sigma: float) -> str:
    return f"{sigma:g}".replace(".", "p")


def hierarchy_method(sigma: float) -> Method:
    if sigma == 0.0:
        return Method("normalized_spdhsw_k40_l500", "lognormal_spectral", L_HIER, sigma, True)
    return Method(f"lns_spdhsw_k40_l500_s{sigma_tag(sigma)}", "lognormal_spectral", L_HIER, sigma, True)


def direct_method(L: int, sigma: float) -> Method:
    if sigma == 0.0:
        return Method(f"spdsw_l{L}", "sw", L, sigma, False)
    return Method(f"direct_lns_l{L}_s{sigma_tag(sigma)}", "lognormal_spectral", L, sigma, False)


def development_methods() -> list[Method]:
    return [
        hierarchy_method(0.0),
        *[hierarchy_method(sigma) for sigma in NONZERO_SIGMAS],
        direct_method(K, 0.0),
        *[direct_method(K, sigma) for sigma in NONZERO_SIGMAS],
        direct_method(500, 0.0),
    ]


def spectrum_diagnostics(weights: torch.Tensor) -> tuple[float, float, float, float]:
    detached = weights.detach()
    positive = detached > 0
    entropy = float(-(detached[positive] * detached[positive].log()).sum())
    ess = float(1.0 / detached.square().sum())
    maximum = float(detached.max())
    top10 = float(torch.topk(detached, min(10, detached.numel())).values.sum())
    return entropy, ess, maximum, top10


def aggregate_costs(h: torch.Tensor, sigma: float) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float, float, float]]:
    if sigma == 0.0:
        weights = torch.full_like(h, 1.0 / h.numel())
        value = h.mean()
        return value, weights, spectrum_diagnostics(weights)
    key = (h.numel(), float(sigma))
    ordered = _SPECTRUM_CACHE.get(key)
    if ordered is None or ordered.device != h.device or ordered.dtype != h.dtype:
        ordered = lognormal_spectral_weights(h.numel(), sigma, h.device, h.dtype).detach()
        _SPECTRUM_CACHE[key] = ordered
    order = torch.argsort(h.detach(), stable=True)
    weights = torch.empty_like(ordered)
    weights[order] = ordered
    weights = weights.detach()
    return torch.sum(weights * h), weights, spectrum_diagnostics(weights)


def sample_spherical_hierarchy(m: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    bank = torch.randn(K, m, generator=generator, device=DEVICE, dtype=DTYPE)
    bank = bank / bank.norm(dim=1, keepdim=True)
    psi = torch.randn(L_HIER, K, generator=generator, device=DEVICE, dtype=DTYPE)
    psi = psi / psi.norm(dim=1, keepdim=True)
    gram = bank @ bank.T
    scale2 = torch.einsum("la,ab,lb->l", psi, gram, psi)
    if not bool(torch.isfinite(scale2).all()) or bool((scale2 <= 0).any()):
        raise RuntimeError("invalid synthetic effective-direction scale")
    return bank, psi, gram, scale2.sqrt()


def synthetic_phase() -> dict:
    phase_out = OUT / "synthetic"
    phase_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dimension_index, m in enumerate(SYNTHETIC_DIMENSIONS):
        delta = torch.zeros(m, device=DEVICE, dtype=DTYPE)
        delta[0] = 1.0
        ambient_mean = 1.0 / m
        for draw in range(SYNTHETIC_DRAWS):
            seed = 810_000_000 + 100_000 * dimension_index + draw
            bank, psi, gram, scale = sample_spherical_hierarchy(m, seed)
            bottleneck_signal = bank @ delta
            h_direct = bottleneck_signal.square()
            h_hierarchical = ((psi @ bottleneck_signal) / scale).square()
            eigenvalues = torch.linalg.eigvalsh(gram)
            condition = float(eigenvalues[-1] / eigenvalues[0])
            oracle = float(bottleneck_signal @ torch.linalg.solve(gram, bottleneck_signal))
            best = float(h_hierarchical.max())
            direct_uniform = float(h_direct.mean())
            hierarchical_uniform = float(h_hierarchical.mean())
            finite_base = all(
                math.isfinite(value) and value >= 0.0
                for value in (condition, oracle, best, direct_uniform, hierarchical_uniform)
            )
            for sigma in SIGMAS:
                direct_result = lognormal_spectral_power(h_direct, sigma)
                hierarchy_result = lognormal_spectral_power(h_hierarchical, sigma)
                direct_value = float(direct_result.value)
                hierarchy_value = float(hierarchy_result.value)
                direct_gain = direct_value - direct_uniform
                hierarchy_gain = hierarchy_value - hierarchical_uniform
                interaction = hierarchy_gain - direct_gain
                finite = finite_base and all(
                    math.isfinite(value)
                    for value in (direct_value, hierarchy_value, direct_gain, hierarchy_gain, interaction)
                )
                rows.append(
                    {
                        "bank_model": "iid_spherical_equivalent_to_Frobenius_svec",
                        "m": m,
                        "draw": draw,
                        "seed": seed,
                        "k": K,
                        "L": L_HIER,
                        "sigma": sigma,
                        "ambient_spdsw_mean": ambient_mean,
                        "direct_uniform_ratio": direct_uniform / ambient_mean,
                        "direct_spectral_ratio": direct_value / ambient_mean,
                        "hierarchical_uniform_ratio": hierarchical_uniform / ambient_mean,
                        "hierarchical_spectral_ratio": hierarchy_value / ambient_mean,
                        "best_of_L_ratio": best / ambient_mean,
                        "within_span_oracle_ratio": oracle / ambient_mean,
                        "best_of_L_over_oracle": best / oracle,
                        "hierarchical_minus_uniform": hierarchy_gain,
                        "direct_minus_uniform": direct_gain,
                        "interaction": interaction,
                        "interaction_over_ambient_mean": interaction / ambient_mean,
                        "direct_spectrum_entropy": direct_result.entropy,
                        "direct_spectrum_ess": direct_result.ess,
                        "hierarchy_spectrum_entropy": hierarchy_result.entropy,
                        "hierarchy_spectrum_ess": hierarchy_result.ess,
                        "gram_condition": condition,
                        "finite": finite,
                    }
                )
        print(f"[SYNTHETIC] m={m} draws={SYNTHETIC_DRAWS}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(phase_out / "draws.csv", index=False)
    numeric = [
        "direct_uniform_ratio",
        "direct_spectral_ratio",
        "hierarchical_uniform_ratio",
        "hierarchical_spectral_ratio",
        "best_of_L_ratio",
        "within_span_oracle_ratio",
        "best_of_L_over_oracle",
        "hierarchical_minus_uniform",
        "direct_minus_uniform",
        "interaction",
        "interaction_over_ambient_mean",
        "direct_spectrum_entropy",
        "direct_spectrum_ess",
        "hierarchy_spectrum_entropy",
        "hierarchy_spectrum_ess",
        "gram_condition",
    ]
    summary = frame.groupby(["m", "sigma"], as_index=False)[numeric].agg(["mean", "std", "median"])
    summary.columns = ["_".join(str(part) for part in column if part != "") for column in summary.columns]
    summary.to_csv(phase_out / "summary.csv", index=False)

    sigma_zero_error = float(
        np.max(
            np.abs(
                frame.loc[frame.sigma == 0.0, "hierarchical_spectral_ratio"]
                - frame.loc[frame.sigma == 0.0, "hierarchical_uniform_ratio"]
            )
        )
    )
    interaction_means = frame.groupby(["sigma", "m"], as_index=False).interaction.mean()
    positive_candidates = []
    condition_robust_candidates = []
    for sigma in NONZERO_SIGMAS:
        block = interaction_means[interaction_means.sigma == sigma]
        if len(block) == len(SYNTHETIC_DIMENSIONS) and bool((block.interaction > 0.0).all()):
            positive_candidates.append(sigma)
            robust = True
            for m in SYNTHETIC_DIMENSIONS:
                cell = frame[(frame.sigma == sigma) & (frame.m == m)]
                threshold = float(cell.gram_condition.median())
                robust &= float(cell[cell.gram_condition <= threshold].interaction.mean()) > 0.0
            if robust:
                condition_robust_candidates.append(sigma)
    finite = bool(frame.finite.all()) and bool(np.isfinite(frame.select_dtypes(include=[np.number])).all().all())
    gate_pass = (
        sigma_zero_error <= 2e-14
        and bool(positive_candidates)
        and bool(condition_robust_candidates)
        and finite
    )
    gate = {
        "pass": gate_pass,
        "sigma_zero_max_ratio_error": sigma_zero_error,
        "positive_interaction_all_dimensions_sigmas": positive_candidates,
        "positive_in_condition_number_lower_half_all_dimensions_sigmas": condition_robust_candidates,
        "all_values_finite": finite,
        "ill_conditioning_rule": "interaction mean > 0 within each dimension's gram-condition lower half",
        "decision": "proceed_to_hgd_development" if gate_pass else "stop_after_synthetic_null",
    }
    dump_json(phase_out / "GATE.json", gate)
    make_spectrum_figure()
    make_synthetic_figure(frame)
    update_global_outputs(synthetic_gate=gate)
    verify_frozen()
    return gate


def make_spectrum_figure() -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    ranks = np.arange(1, L_HIER + 1) / L_HIER
    for sigma in SIGMAS:
        weights = lognormal_spectral_weights(L_HIER, sigma, "cpu", DTYPE).numpy()
        axis.plot(ranks, L_HIER * weights, label=f"sigma={sigma:g}")
    axis.set_xlabel("increasing rank i/L")
    axis.set_ylabel("L × exact cell weight")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT / "fig_spectrum_weights.png", dpi=180)
    plt.close(figure)


def make_synthetic_figure(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    grouped = frame.groupby(["m", "sigma"], as_index=False).agg(
        interaction=("interaction_over_ambient_mean", "mean"),
        capture=("best_of_L_over_oracle", "mean"),
    )
    for m, block in grouped.groupby("m"):
        axes[0].plot(block.sigma, block.interaction, marker="o", label=f"m={m}")
        axes[1].plot(block.sigma, block.capture, marker="o", label=f"m={m}")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("spectral interaction / ambient mean")
    axes[1].set_ylabel("best-of-L / within-span oracle")
    for axis in axes:
        axis.set_xlabel("sigma")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUT / "fig_synthetic_capture.png", dpi=180)
    plt.close(figure)


def sample_frobenius_bank(basis: SvecBasis, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    raw = torch.randn(count, basis.d, basis.d, generator=generator, device=DEVICE, dtype=DTYPE)
    matrices = raw + raw.transpose(-1, -2)
    matrices = matrices / matrices.norm(dim=(-1, -2), keepdim=True)
    return basis.forward(matrices)


def sample_normalized_hierarchy(
    basis: SvecBasis,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    raw = torch.randn(K, basis.d, basis.d, generator=generator, device=DEVICE, dtype=DTYPE)
    matrices = raw + raw.transpose(-1, -2)
    matrices = matrices / matrices.norm(dim=(-1, -2), keepdim=True)
    bank = basis.forward(matrices)
    psi = torch.randn(L_HIER, K, generator=generator, device=DEVICE, dtype=DTYPE)
    psi = psi / psi.norm(dim=1, keepdim=True)
    gram = bank @ bank.T
    scale2 = torch.einsum("la,ab,lb->l", psi, gram, psi)
    if not bool(torch.isfinite(scale2).all()) or bool((scale2 <= 0).any()):
        raise RuntimeError("invalid normalized hierarchy scale")
    return bank, psi, gram, scale2.sqrt()


def method_costs(
    method: Method,
    parameter: torch.Tensor,
    target_vec: torch.Tensor,
    basis: SvecBasis,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if method.hierarchical:
        bank, psi, gram, scale = sample_normalized_hierarchy(basis, random_seed)
        source_bottleneck = parameter @ bank.T
        target_bottleneck = target_vec @ bank.T
        projected_source = (source_bottleneck @ psi.T) / scale[None, :]
        projected_target = (target_bottleneck @ psi.T) / scale[None, :]
        return w2_squared_per_direction(projected_source.T, projected_target.T), gram
    bank = sample_frobenius_bank(basis, method.L, random_seed)
    projected_source = parameter @ bank.T
    projected_target = target_vec @ bank.T
    return w2_squared_per_direction(projected_source.T, projected_target.T), None


def initial_gradient_norm(method: Method, source: torch.Tensor, target: torch.Tensor, seed: int) -> float:
    basis = SvecBasis(source.shape[-1], DEVICE, DTYPE)
    parameter = basis.forward(source).clone().requires_grad_(True)
    target_vec = basis.forward(target)
    h, _ = method_costs(method, parameter, target_vec, basis, seed)
    value, _, _ = aggregate_costs(h, method.sigma)
    value.backward()
    result = float(parameter.grad.norm())
    del parameter, target_vec, h, value
    return result


def _blank_epoch_row(method: Method, phase: str, control: str, subject: int, seed: int, epoch: int) -> dict:
    row = {column: math.nan for column in EPOCH_COLUMNS}
    row.update(
        dataset="Schirrmeister2017",
        phase=phase,
        control=control,
        method=method.name,
        family=method.family,
        hierarchical=method.hierarchical,
        subject=subject,
        seed=seed,
        epoch=epoch,
        k=K if method.hierarchical else method.L,
        L=method.L,
        sigma=method.sigma,
        cumulative_ambient_projections=(K if method.hierarchical else method.L) * epoch,
        bank_seed=seed + max(0, epoch - 1),
        mix_seed=(seed + max(0, epoch - 1)) if method.hierarchical else math.nan,
        nan=True,
        diverged=True,
        status="nonfinite_trajectory",
    )
    return row


def train_alignment(
    method: Method,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    phase: str,
    control: str,
    subject: int,
    seed: int,
    epochs: int,
    normalized_step_target: float,
    learning_rate: float,
) -> tuple[pd.DataFrame, dict]:
    basis = SvecBasis(source.shape[-1], DEVICE, DTYPE)
    source = source.to(device=DEVICE, dtype=DTYPE)
    target = target.to(device=DEVICE, dtype=DTYPE)
    parameter = basis.forward(source).clone().requires_grad_(True)
    target_vec = basis.forward(target)
    evaluator = LEWEvaluator(target)
    evaluation_seconds = 0.0
    optimization_seconds = 0.0
    tick = time.perf_counter()
    lew0 = evaluator(source)
    evaluation_seconds += time.perf_counter() - tick
    evaluator.set_baseline(lew0)
    initial = _blank_epoch_row(method, phase, control, subject, seed, 0)
    initial.update(
        training_power_loss=math.nan,
        rooted_distance=math.nan,
        lew=lew0,
        gap_closure_pct=0.0,
        optimization_seconds_cum=0.0,
        evaluation_seconds_cum=evaluation_seconds,
        cumulative_ambient_projections=0,
        learning_rate=learning_rate,
        normalized_step_target=normalized_step_target,
        bank_seed=seed,
        mix_seed=seed if method.hierarchical else math.nan,
        nan=False,
        diverged=False,
        status="initial",
    )
    rows = [initial]
    nonfinite_epoch: int | None = None
    aggregation_total = 0.0
    for zero_epoch in range(epochs):
        random_seed = seed + zero_epoch
        sync()
        epoch_tick = time.perf_counter()
        h, gram = method_costs(method, parameter, target_vec, basis, random_seed)
        sync()
        aggregation_tick = time.perf_counter()
        loss, weights, diagnostics = aggregate_costs(h, method.sigma)
        sync()
        aggregation_seconds = time.perf_counter() - aggregation_tick
        aggregation_total += aggregation_seconds
        loss.backward()
        gradient_norm = float(parameter.grad.norm())
        if control == "normalized_update":
            if gradient_norm > 0.0 and math.isfinite(gradient_norm):
                update = -normalized_step_target * parameter.grad / gradient_norm
            else:
                update = torch.full_like(parameter, math.nan)
        elif control == "raw_sgd_initial_update_matched":
            update = -learning_rate * parameter.grad
        else:
            raise ValueError(control)
        update_norm = float(update.norm())
        with torch.no_grad():
            parameter.add_(update)
        parameter.grad = None
        sync()
        optimization_seconds += time.perf_counter() - epoch_tick
        epoch = zero_epoch + 1
        finite = bool(torch.isfinite(parameter).all()) and bool(torch.isfinite(loss)) and math.isfinite(gradient_norm)
        gram_condition = math.nan
        if gram is not None:
            eigenvalues = torch.linalg.eigvalsh(gram.detach())
            gram_condition = float(eigenvalues[-1] / eigenvalues[0])
        entropy, ess, max_weight, top10_weight = diagnostics
        want_lew = epoch % LEW_EVERY == 0 or epoch == epochs
        lew = math.nan
        closure = math.nan
        diverged = not finite
        if finite and want_lew:
            eval_tick = time.perf_counter()
            lew = evaluator(basis.inverse(parameter.detach()))
            evaluation_seconds += time.perf_counter() - eval_tick
            closure = evaluator.closed_pct(lew)
            diverged = evaluator.diverged(lew)
        rows.append(
            {
                "dataset": "Schirrmeister2017",
                "phase": phase,
                "control": control,
                "method": method.name,
                "family": method.family,
                "hierarchical": method.hierarchical,
                "subject": subject,
                "seed": seed,
                "epoch": epoch,
                "k": K if method.hierarchical else method.L,
                "L": method.L,
                "sigma": method.sigma,
                "training_power_loss": float(loss.detach()),
                "rooted_distance": float(loss.detach().clamp_min(0).sqrt()),
                "lew": lew,
                "gap_closure_pct": closure,
                "gradient_norm": gradient_norm,
                "update_norm": update_norm,
                "mean_h": float(h.detach().mean()),
                "std_h": float(h.detach().std(unbiased=False)),
                "max_h": float(h.detach().max()),
                "min_h": float(h.detach().min()),
                "spectral_entropy": entropy,
                "spectral_ess": ess,
                "spectral_max_weight": max_weight,
                "spectral_top10_weight": top10_weight,
                "gram_condition": gram_condition,
                "aggregation_seconds": aggregation_seconds,
                "optimization_seconds_cum": optimization_seconds,
                "evaluation_seconds_cum": evaluation_seconds,
                "cumulative_ambient_projections": (K if method.hierarchical else method.L) * epoch,
                "learning_rate": learning_rate,
                "normalized_step_target": normalized_step_target,
                "bank_seed": random_seed,
                "mix_seed": random_seed if method.hierarchical else math.nan,
                "nan": not finite,
                "diverged": diverged,
                "status": "ok" if finite else "nonfinite",
            }
        )
        if not finite:
            nonfinite_epoch = epoch
            rows.extend(_blank_epoch_row(method, phase, control, subject, seed, later) for later in range(epoch + 1, epochs + 1))
            break
    frame = pd.DataFrame(rows)[EPOCH_COLUMNS]
    evaluated = frame[np.isfinite(frame.lew)]
    final_lew = float(evaluated.lew.iloc[-1]) if len(evaluated) else math.nan
    metadata = {
        "lew_initial": lew0,
        "lew_final": final_lew,
        "diverged": bool(frame.diverged.fillna(False).any()) or evaluator.diverged(final_lew),
        "nan_epoch": nonfinite_epoch,
        "optimization_seconds": optimization_seconds,
        "evaluation_seconds": evaluation_seconds,
        "aggregation_seconds": aggregation_total,
    }
    return frame, metadata


def complete_csv(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
        return len(frame) == epochs + 1 and int(frame.epoch.iloc[-1]) == epochs
    except Exception:
        return False


def relative_auc(frame: pd.DataFrame) -> float:
    evaluated = frame[np.isfinite(frame.lew)].sort_values("epoch")
    if len(evaluated) < 2:
        return math.inf
    initial = float(evaluated.lew.iloc[0])
    horizon = float(evaluated.epoch.iloc[-1] - evaluated.epoch.iloc[0])
    return float(np.trapezoid(evaluated.lew / initial, evaluated.epoch) / horizon)


def run_grid(
    *,
    phase: str,
    run_root: Path,
    subjects: Iterable[int],
    seeds: Iterable[int],
    methods: list[Method],
    epochs: int,
    control: str,
    normalized_step_target: float,
    learning_rates: dict[str, float] | None = None,
    rerun: bool = False,
) -> pd.DataFrame:
    records = []
    subjects = tuple(subjects)
    seeds = tuple(seeds)
    total = len(subjects) * len(seeds) * len(methods)
    index = 0
    for subject in subjects:
        source, target, meta = load_cached_subject("Schirrmeister2017", subject, DEVICE)
        for seed in seeds:
            for method in methods:
                index += 1
                path = run_root / control / method.name / f"seed_{seed}" / f"subject_{subject:02d}.csv"
                lr = math.nan if learning_rates is None else learning_rates[method.name]
                try:
                    if rerun or not complete_csv(path, epochs):
                        frame, metadata = train_alignment(
                            method,
                            source,
                            target,
                            phase=phase,
                            control=control,
                            subject=subject,
                            seed=seed,
                            epochs=epochs,
                            normalized_step_target=normalized_step_target,
                            learning_rate=lr,
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        frame.to_csv(path, index=False)
                    else:
                        frame = pd.read_csv(path)
                        evaluated = frame[np.isfinite(frame.lew)]
                        metadata = {
                            "lew_initial": float(evaluated.lew.iloc[0]),
                            "lew_final": float(evaluated.lew.iloc[-1]),
                            "diverged": bool(frame.diverged.fillna(False).any()),
                            "nan_epoch": None if not bool(frame["nan"].fillna(False).any()) else int(frame[frame["nan"]].epoch.iloc[0]),
                            "optimization_seconds": float(frame.optimization_seconds_cum.iloc[-1]),
                            "evaluation_seconds": float(frame.evaluation_seconds_cum.iloc[-1]),
                            "aggregation_seconds": float(frame.aggregation_seconds.fillna(0).sum()),
                        }
                    record = {
                        "phase": phase,
                        "control": control,
                        "method": method.name,
                        "family": method.family,
                        "hierarchical": method.hierarchical,
                        "sigma": method.sigma,
                        "L": method.L,
                        "subject": subject,
                        "seed": seed,
                        "d": meta["d"],
                        "n_source": meta["n_source"],
                        "n_target": meta["n_target"],
                        "learning_rate": lr,
                        "normalized_step_target": normalized_step_target,
                        "status": "ok",
                        "error": "",
                        **metadata,
                    }
                    print(
                        f"[{phase.upper()} {index:03d}/{total:03d}] {control} s{subject:02d} seed={seed} "
                        f"{method.name:35s} LEW {metadata['lew_initial']:.3f}->{metadata['lew_final']:.3f}",
                        flush=True,
                    )
                except Exception as exc:
                    log = OUT / "logs" / f"{phase}_{control}_{method.name}_seed{seed}_s{subject:02d}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text(traceback.format_exc())
                    record = {
                        "phase": phase,
                        "control": control,
                        "method": method.name,
                        "family": method.family,
                        "hierarchical": method.hierarchical,
                        "sigma": method.sigma,
                        "L": method.L,
                        "subject": subject,
                        "seed": seed,
                        "d": meta["d"],
                        "n_source": meta["n_source"],
                        "n_target": meta["n_target"],
                        "learning_rate": lr,
                        "normalized_step_target": normalized_step_target,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"[FAIL] {record['error']}", file=sys.stderr, flush=True)
                records.append(record)
                pd.DataFrame(records).to_csv(OUT / f"MANIFEST_{phase}_{control}.csv", index=False)
        del source, target
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def load_run_frames(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/*/seed_*/subject_*.csv"))
    if not paths:
        raise RuntimeError(f"no run CSVs under {root}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["phase", "control", "method", "family", "hierarchical", "sigma", "L", "subject", "seed"]
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        group = group.sort_values("epoch")
        evaluated = group[np.isfinite(group.lew)]
        initial = float(evaluated.lew.iloc[0])
        final = float(evaluated.lew.iloc[-1])
        row = dict(zip(keys, key))
        row.update(
            lew_initial=initial,
            lew_final=final,
            relative_lew_auc=relative_auc(group),
            gap_closure_100=float(group.loc[group.epoch == 100, "gap_closure_pct"].iloc[0]),
            gap_closure_500=(
                float(group.loc[group.epoch == 500, "gap_closure_pct"].iloc[0])
                if bool((group.epoch == 500).any())
                else math.nan
            ),
            diverged=bool(group.diverged.fillna(False).any()) or not math.isfinite(final),
            nan_count=int(group["nan"].fillna(False).sum()),
            optimization_seconds=float(group.optimization_seconds_cum.iloc[-1]),
            aggregation_seconds=float(group.aggregation_seconds.fillna(0).sum()),
            median_ess=float(group.loc[group.epoch > 0, "spectral_ess"].median()),
            median_entropy=float(group.loc[group.epoch > 0, "spectral_entropy"].median()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def development_step_target() -> tuple[float, pd.DataFrame]:
    method = hierarchy_method(0.0)
    rows = []
    for subject in DEV_SUBJECTS:
        source, target, _ = load_cached_subject("Schirrmeister2017", subject, DEVICE)
        gradient = initial_gradient_norm(method, source, target, DEV_SEED)
        rows.append(
            {
                "subject": subject,
                "seed": DEV_SEED,
                "method": method.name,
                "gradient_norm": gradient,
                "raw_reference_lr": RAW_REFERENCE_LR,
                "initial_update_norm": RAW_REFERENCE_LR * gradient,
            }
        )
    frame = pd.DataFrame(rows)
    return float(frame.initial_update_norm.median()), frame


def development_phase(rerun: bool = False) -> dict:
    synthetic_gate = json.loads((OUT / "synthetic" / "GATE.json").read_text())
    if not synthetic_gate["pass"]:
        raise RuntimeError("synthetic gate failed; development is prohibited")
    phase_out = OUT / "development"
    phase_out.mkdir(parents=True, exist_ok=True)
    target_path = phase_out / "NORMALIZED_STEP_TARGET.json"
    if target_path.exists():
        target_payload = json.loads(target_path.read_text())
        step_target = float(target_payload["normalized_step_target"])
    else:
        step_target, target_frame = development_step_target()
        target_frame.to_csv(phase_out / "initial_gradient_audit.csv", index=False)
        target_payload = {
            "definition": "median initial normalized-SPDHSW raw-LR-10000 update norm over subjects 2,3,4",
            "normalized_step_target": step_target,
        }
        dump_json(target_path, target_payload)
    methods = development_methods()
    manifest = run_grid(
        phase="development",
        run_root=phase_out / "runs",
        subjects=DEV_SUBJECTS,
        seeds=(DEV_SEED,),
        methods=methods,
        epochs=DEV_EPOCHS,
        control="normalized_update",
        normalized_step_target=step_target,
        rerun=rerun,
    )
    frames = load_run_frames(phase_out / "runs")
    summary = summarize_runs(frames)
    summary.to_csv(phase_out / "RUN_SUMMARY.csv", index=False)
    baseline = summary[summary.method == hierarchy_method(0.0).name][["subject", "relative_lew_auc"]].rename(
        columns={"relative_lew_auc": "uniform_hierarchy_auc"}
    )
    candidate_rows = []
    for sigma in NONZERO_SIGMAS:
        method = hierarchy_method(sigma)
        block = summary[summary.method == method.name].merge(baseline, on="subject")
        block["paired_auc_difference"] = block.relative_lew_auc - block.uniform_hierarchy_auc
        candidate_rows.append(
            {
                "method": method.name,
                "sigma": sigma,
                "mean_relative_lew_auc": float(block.relative_lew_auc.mean()),
                "mean_uniform_hierarchy_auc": float(block.uniform_hierarchy_auc.mean()),
                "mean_paired_auc_difference": float(block.paired_auc_difference.mean()),
                "improved_subjects": int((block.paired_auc_difference < 0.0).sum()),
                "divergence_count": int(block.diverged.sum()),
                "nan_count": int(block.nan_count.sum()),
                "eligible_no_divergence": not bool(block.diverged.any()) and int(block.nan_count.sum()) == 0,
            }
        )
    selection = pd.DataFrame(candidate_rows).sort_values(
        ["eligible_no_divergence", "mean_relative_lew_auc", "sigma"],
        ascending=[False, True, True],
    )
    winner = selection.iloc[0]
    selection["selected"] = selection.method == winner.method
    selection.to_csv(phase_out / "SELECTION.csv", index=False)
    execution_errors = int((manifest.status != "ok").sum())
    passed = bool(winner.eligible_no_divergence) and int(winner.improved_subjects) >= 2 and execution_errors == 0
    gate = {
        "pass": passed,
        "selected_method": str(winner.method),
        "selected_sigma": float(winner.sigma),
        "selected_improved_subjects": int(winner.improved_subjects),
        "selected_mean_paired_auc_difference": float(winner.mean_paired_auc_difference),
        "selected_divergence_count": int(winner.divergence_count),
        "execution_errors": execution_errors,
        "decision": "proceed_to_heldout_hgd" if passed else "stop_after_development_null",
    }
    dump_json(phase_out / "GATE.json", gate)
    dump_json(
        phase_out / "SELECTED_SIGMA.json",
        {
            "selected_method": str(winner.method),
            "selected_sigma": float(winner.sigma),
            "selection_rule": "no divergence, lowest mean relative LEW AUC, smaller-sigma tie break",
            "all_candidates_file": "SELECTION.csv",
        },
    )
    update_global_outputs(synthetic_gate=synthetic_gate, development_gate=gate)
    verify_frozen()
    return gate


def _development_interaction(summary: pd.DataFrame) -> pd.DataFrame:
    """Development-only hierarchy-versus-direct diagnostic by subject/sigma."""
    hierarchy_zero = summary[summary.method == hierarchy_method(0.0).name][
        ["subject", "relative_lew_auc"]
    ].rename(columns={"relative_lew_auc": "hierarchy_uniform_auc"})
    direct_zero = summary[summary.method == direct_method(K, 0.0).name][
        ["subject", "relative_lew_auc"]
    ].rename(columns={"relative_lew_auc": "direct_uniform_auc"})
    rows: list[pd.DataFrame] = []
    for sigma in NONZERO_SIGMAS:
        hierarchy = summary[summary.method == hierarchy_method(sigma).name][
            ["subject", "relative_lew_auc"]
        ].rename(columns={"relative_lew_auc": "hierarchy_spectral_auc"})
        direct = summary[summary.method == direct_method(K, sigma).name][
            ["subject", "relative_lew_auc"]
        ].rename(columns={"relative_lew_auc": "direct_spectral_auc"})
        block = hierarchy.merge(hierarchy_zero, on="subject").merge(direct, on="subject").merge(
            direct_zero, on="subject"
        )
        block.insert(0, "sigma", sigma)
        block["hierarchy_paired_difference"] = (
            block.hierarchy_spectral_auc - block.hierarchy_uniform_auc
        )
        block["direct_paired_difference"] = block.direct_spectral_auc - block.direct_uniform_auc
        block["interaction"] = block.hierarchy_paired_difference - block.direct_paired_difference
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def _plot_mean_curves(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    output: Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    selected = [
        hierarchy_method(0.0).name,
        hierarchy_method(0.5).name,
        direct_method(K, 0.0).name,
        direct_method(K, 0.5).name,
        direct_method(500, 0.0).name,
    ]
    labels = {
        hierarchy_method(0.0).name: "normalized SPDHSW (s=0)",
        hierarchy_method(0.5).name: "spectral SPDHSW (s=0.5)",
        direct_method(K, 0.0).name: "SPDSW L=40",
        direct_method(K, 0.5).name: "direct spectral L=40 (s=0.5)",
        direct_method(500, 0.0).name: "SPDSW L=500",
    }
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    for method in selected:
        block = frame[(frame.method == method) & np.isfinite(frame[y])].copy()
        if block.empty:
            continue
        if y == "lew":
            initial = block[block.epoch == 0][["subject", "seed", "lew"]].rename(
                columns={"lew": "initial_lew"}
            )
            block = block.merge(initial, on=["subject", "seed"])
            block["plot_value"] = block.lew / block.initial_lew
        else:
            block["plot_value"] = block[y]
        grouped = block.groupby("epoch", as_index=False).agg(
            x_value=(x, "mean"), mean=("plot_value", "mean"), minimum=("plot_value", "min"), maximum=("plot_value", "max")
        )
        axis.plot(grouped.x_value, grouped["mean"], label=labels[method], linewidth=2)
        axis.fill_between(grouped.x_value, grouped.minimum, grouped.maximum, alpha=0.10)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_development_figures(frame: pd.DataFrame, summary: pd.DataFrame, interaction: pd.DataFrame) -> None:
    evaluated = frame[np.isfinite(frame.lew)].copy()
    _plot_mean_curves(
        evaluated,
        x="epoch",
        y="lew",
        output=OUT / "fig_lew_vs_epoch.png",
        xlabel="epoch",
        ylabel="mean relative exact LEW",
        title="HGD development (normalized update; mean and subject range)",
    )
    _plot_mean_curves(
        evaluated,
        x="optimization_seconds_cum",
        y="lew",
        output=OUT / "fig_lew_vs_wallclock.png",
        xlabel="optimization-loop seconds (LEW evaluation excluded)",
        ylabel="mean relative exact LEW",
        title="HGD development: LEW versus optimization wall-clock",
    )
    _plot_mean_curves(
        evaluated,
        x="cumulative_ambient_projections",
        y="lew",
        output=OUT / "fig_lew_vs_ambient_projections.png",
        xlabel="cumulative expensive ambient projections",
        ylabel="mean relative exact LEW",
        title="HGD development: LEW versus ambient projection count",
    )

    diagnostic = interaction.groupby("sigma", as_index=False).agg(
        mean_interaction=("interaction", "mean"),
        minimum=("interaction", "min"),
        maximum=("interaction", "max"),
    )
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    error = np.vstack(
        [diagnostic.mean_interaction - diagnostic.minimum, diagnostic.maximum - diagnostic.mean_interaction]
    )
    axis.bar(diagnostic.sigma.astype(str), diagnostic.mean_interaction, color="#577590", yerr=error, capsize=4)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(
        xlabel="sigma",
        ylabel="development AUC interaction",
        title="Development diagnostic only (held-out interaction not run)",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_interaction.png", dpi=180)
    plt.close(fig)

    selected = [hierarchy_method(0.0).name, hierarchy_method(0.5).name, direct_method(K, 0.0).name]
    labels = {
        hierarchy_method(0.0).name: "normalized SPDHSW",
        hierarchy_method(0.5).name: "spectral SPDHSW s=0.5",
        direct_method(K, 0.0).name: "SPDSW L=40",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    positive_epochs = frame[frame.epoch > 0]
    for method in selected:
        block = positive_epochs[positive_epochs.method == method]
        grouped = block.groupby("epoch", as_index=False).agg(
            gradient=("gradient_norm", "mean"), update=("update_norm", "mean")
        )
        axes[0].plot(grouped.epoch, grouped.gradient, label=labels[method])
        axes[1].plot(grouped.epoch, grouped["update"], label=labels[method])
    axes[0].set(xlabel="epoch", ylabel="mean Frobenius norm", title="Gradient norm")
    axes[1].set(xlabel="epoch", ylabel="mean Frobenius norm", title="Update norm")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("HGD development stability under normalized updates")
    fig.tight_layout()
    fig.savefig(OUT / "fig_gradient_update_stability.png", dpi=180)
    plt.close(fig)

    spectrum = summary.groupby(["hierarchical", "sigma", "L"], as_index=False).agg(
        ess=("median_ess", "mean"), entropy=("median_entropy", "mean")
    )
    spectrum["relative_ess"] = spectrum.ess / spectrum.L
    spectrum["relative_entropy"] = spectrum.entropy / np.log(spectrum.L)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for hierarchical, label in [(False, "direct L=40"), (True, "hierarchy L=500")]:
        expected_l = L_HIER if hierarchical else K
        block = spectrum[
            (spectrum.hierarchical == hierarchical)
            & (spectrum.L == expected_l)
            & (spectrum.sigma.isin(SIGMAS))
        ].sort_values("sigma")
        axes[0].plot(block.sigma, block.relative_ess, marker="o", label=label)
        axes[1].plot(block.sigma, block.relative_entropy, marker="o", label=label)
    axes[0].set(xlabel="sigma", ylabel="ESS / L", title="Relative effective sample size")
    axes[1].set(xlabel="sigma", ylabel="entropy / log(L)", title="Relative entropy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ess_entropy.png", dpi=180)
    plt.close(fig)


def finalize_development_null() -> dict:
    """Package the preregistered development null without launching later phases."""
    phase_out = OUT / "development"
    development_gate = json.loads((phase_out / "GATE.json").read_text())
    synthetic_gate = json.loads((OUT / "synthetic" / "GATE.json").read_text())
    if development_gate.get("pass"):
        raise RuntimeError("development gate passed; null finalizer is not applicable")
    frame = load_run_frames(phase_out / "runs")
    summary = pd.read_csv(phase_out / "RUN_SUMMARY.csv")
    interaction = _development_interaction(summary)
    interaction.to_csv(phase_out / "INTERACTION_DIAGNOSTIC.csv", index=False)

    aggregate = summary.groupby(
        ["phase", "control", "method", "family", "hierarchical", "sigma", "L"], as_index=False
    ).agg(
        subjects=("subject", "nunique"),
        mean_relative_lew_auc=("relative_lew_auc", "mean"),
        std_relative_lew_auc=("relative_lew_auc", "std"),
        mean_final_lew=("lew_final", "mean"),
        mean_gap_closure_100=("gap_closure_100", "mean"),
        divergence_count=("diverged", "sum"),
        nan_count=("nan_count", "sum"),
        mean_optimization_seconds=("optimization_seconds", "mean"),
        mean_aggregation_seconds=("aggregation_seconds", "mean"),
        median_ess=("median_ess", "median"),
        median_entropy=("median_entropy", "median"),
    )
    aggregate["aggregation_overhead_pct"] = (
        100.0 * aggregate.mean_aggregation_seconds / aggregate.mean_optimization_seconds
    )
    aggregate["selected_development_candidate"] = aggregate.method == development_gate["selected_method"]
    aggregate["interpretation"] = "development-only; held-out Phase C prohibited by failed gate"
    aggregate.to_csv(OUT / "CORE_RESULTS.csv", index=False)

    seed_columns = ["bank_seed", "mix_seed"]
    seed_range = frame.groupby(["subject", "seed", "epoch", "hierarchical"])[seed_columns].agg(
        lambda values: int(values.dropna().nunique())
    )
    h_columns = ["mean_h", "std_h", "max_h", "min_h"]
    initial = frame[frame.epoch == 1]
    initial_ranges: dict[str, float] = {}
    for hierarchical, label in [(False, "direct"), (True, "hierarchical")]:
        block = initial[
            (initial.hierarchical == hierarchical)
            & (initial.L == (L_HIER if hierarchical else K))
        ]
        for column in h_columns:
            ranges = block.groupby(["subject", "seed"])[column].agg(lambda values: float(values.max() - values.min()))
            initial_ranges[f"{label}_{column}_max_range"] = float(ranges.max())
    eval_epochs = sorted(int(value) for value in frame.loc[np.isfinite(frame.lew), "epoch"].unique())
    validation = {
        "run_csv_count": int(summary.shape[0]),
        "expected_run_csv_count": len(DEV_SUBJECTS) * len(development_methods()),
        "rows_per_run_values": sorted(int(value) for value in frame.groupby(["method", "subject", "seed"]).size().unique()),
        "expected_rows_per_run": DEV_EPOCHS + 1,
        "evaluation_epochs": eval_epochs,
        "expected_evaluation_epochs": list(range(0, DEV_EPOCHS + 1, LEW_EVERY)),
        "divergence_count": int(summary.diverged.sum()),
        "nan_count": int(summary.nan_count.sum()),
        "bank_seed_unique_count_max_within_epoch_group": int(seed_range.bank_seed.max()),
        "mix_seed_unique_count_max_within_hierarchical_epoch_group": int(
            seed_range.reset_index().query("hierarchical == True").mix_seed.max()
        ),
        "initial_directional_field_ranges": initial_ranges,
        "heldout_phase_run": False,
        "validation_pass": bool(
            summary.shape[0] == len(DEV_SUBJECTS) * len(development_methods())
            and (frame.groupby(["method", "subject", "seed"]).size() == DEV_EPOCHS + 1).all()
            and eval_epochs == list(range(0, DEV_EPOCHS + 1, LEW_EVERY))
            and int(summary.diverged.sum()) == 0
            and int(summary.nan_count.sum()) == 0
            and int(seed_range.bank_seed.max()) == 1
            and int(seed_range.reset_index().query("hierarchical == True").mix_seed.max()) == 1
        ),
    }
    dump_json(phase_out / "COMMON_RANDOM_NUMBERS_AUDIT.json", validation)
    _write_development_figures(frame, summary, interaction)

    root_runs = OUT / "runs"
    root_runs.mkdir(exist_ok=True)
    (root_runs / "NOT_RUN.md").write_text(
        "# Held-out runs: not run\n\n"
        "Phase C was prohibited because the preregistered Phase B development gate failed. "
        "No subjects 1, 7, or 14, Phase D baselines, or BNCI transfer runs were launched.\n"
    )
    update_global_outputs(synthetic_gate=synthetic_gate, development_gate=development_gate)
    verify_frozen()
    return validation


def write_claim_ledger() -> None:
    text = """# Claim ledger

## Proved population statements

- A fixed nonnegative nondecreasing spectrum with unit integral defines a
  coherent spectral risk functional on directional p-costs.
- The lognormal-quantile density is nonnegative, integrates to one, and reduces
  to uniform weighting at sigma zero.
- Positive cost scaling preserves ranks and therefore preserves assigned
  lognormal-quantile weights; the spectral power is positively homogeneous.

## Finite common-direction pseudometric statements

- With one shared finite direction family for every pair, the rooted ordered
  weighted Lp construction obeys the triangle inequality (up to the usual
  identity-of-indiscernibles limitation of a finite direction set).
- The regression suite supports this statement for explicit and random shared
  fields. It does not cover independently resampled pairwise banks.

## Empirical findings

- The registered synthetic interaction gate passed in m=253, 2016, and 8256.
- In HGD development subjects 2, 3, and 4, every registered nonzero sigma had
  worse mean normalized-update relative LEW AUC than uniform normalized
  SPDHSW. The selected sigma 0.5 differed by +0.00354608 (lower is better)
  and improved 0 of 3 subjects.
- The failed development gate stopped held-out HGD, raw-SGD controls, matched
  concentration baselines, and BNCI transfer. No inference about those unrun
  phases is made.

## Unsupported or prohibited claims

- Nonzero sigma is not claimed to estimate the same population quantity as
  uniform SPDSW.
- Finite spectral order statistics are not claimed to be unbiased estimators.
- A large-sigma hierarchical candidate maximum is not global Max-SPDSW.
- Hierarchical mixtures never explore outside the bottleneck span.
- Smaller k is not claimed to solve shared-bottleneck variance.
- Metricity is not claimed for independently resampled realized finite values.
- The lognormal spectrum itself is not presented as the main novelty; the
  working method is Spectral SPDHSW with lognormal-quantile weighting.
"""
    (OUT / "CLAIM_LEDGER.md").write_text(text)


def update_global_outputs(
    *,
    synthetic_gate: dict | None = None,
    development_gate: dict | None = None,
    heldout_gate: dict | None = None,
) -> None:
    if synthetic_gate is None and (OUT / "synthetic" / "GATE.json").exists():
        synthetic_gate = json.loads((OUT / "synthetic" / "GATE.json").read_text())
    if development_gate is None and (OUT / "development" / "GATE.json").exists():
        development_gate = json.loads((OUT / "development" / "GATE.json").read_text())
    theorem_pass = (OUT / "TEST_RESULTS.xml").exists() and 'failures="0"' in (OUT / "TEST_RESULTS.xml").read_text()
    synthetic_yes = bool(synthetic_gate and synthetic_gate.get("pass"))
    if heldout_gate is None:
        hgd = hierarchy = quality = "NOT RUN"
    else:
        hgd = "YES" if heldout_gate.get("normalized_improvement") else "NO"
        hierarchy = "YES" if heldout_gate.get("hierarchy_interaction") else "NO"
        quality = "YES" if heldout_gate.get("reaches_spdsw_l500_quality") else "NO"
    overall_decision = (
        heldout_gate.get("decision")
        if heldout_gate
        else development_gate.get("decision")
        if development_gate
        else synthetic_gate.get("decision")
        if synthetic_gate
        else "not_started"
    )
    gate = {
        "theorem_regression": theorem_pass,
        "synthetic": synthetic_gate,
        "development": development_gate,
        "heldout": heldout_gate,
        "decision": overall_decision,
        "proceed_to_full_expansion": False,
    }
    dump_json(OUT / "GATE.json", gate)
    write_claim_ledger()
    lines = [
        f"- theorem regression: {'PASS' if theorem_pass else 'FAIL'}",
        f"- synthetic hierarchy interaction: {'YES' if synthetic_yes else 'NO'}",
        f"- HGD normalized-update improvement: {hgd}",
        f"- hierarchy-specific interaction: {hierarchy}",
        f"- reaches SPDSW-L500 quality: {quality}",
        "- proceed to full expansion: NO",
        "",
        "# Lognormal-spectral hierarchical SPDSW pilot",
        "",
        f"Current registered decision: `{overall_decision}`.",
        "The completed direct-pilot decision `stop_after_hgd_null` remains unchanged and is not reinterpreted.",
        "",
        "## Fixed method and controls",
        "",
        "- Spectral SPDHSW uses lognormal-quantile spectral weighting over freshly resampled normalized mixtures.",
        "- dtype is torch.float64; AMP, autocast, and TF32 are disabled; no clipping or outcome-triggered early stopping is used.",
        "- Physical GPU 3 is the only GPU authorized. No persistent/evolving bank is used.",
        "",
        "## Phase status",
        "",
    ]
    if synthetic_gate:
        lines.append(
            f"- Synthetic gate: {'PASS' if synthetic_gate['pass'] else 'FAIL'}; "
            f"positive all-dimension sigmas={synthetic_gate['positive_interaction_all_dimensions_sigmas']}; "
            f"condition-robust sigmas={synthetic_gate['positive_in_condition_number_lower_half_all_dimensions_sigmas']}."
        )
    else:
        lines.append("- Synthetic phase: NOT RUN.")
    if development_gate:
        lines.append(
            f"- Development gate: {'PASS' if development_gate['pass'] else 'FAIL'}; "
            f"selected sigma={development_gate['selected_sigma']}, "
            f"improved subjects={development_gate['selected_improved_subjects']}/3; "
            f"mean paired relative-LEW AUC difference={development_gate['selected_mean_paired_auc_difference']:+.8f} "
            "(lower is better)."
        )
    else:
        lines.append("- HGD development: NOT RUN.")
    if development_gate and not development_gate["pass"]:
        selection = pd.read_csv(OUT / "development" / "SELECTION.csv")
        selection_table = selection[
            [
                "sigma",
                "mean_relative_lew_auc",
                "mean_uniform_hierarchy_auc",
                "mean_paired_auc_difference",
                "improved_subjects",
                "divergence_count",
                "selected",
            ]
        ]
        summary = pd.read_csv(OUT / "development" / "RUN_SUMMARY.csv")
        selected_summary = summary[
            summary.method.isin(
                [
                    hierarchy_method(0.0).name,
                    hierarchy_method(float(development_gate["selected_sigma"])).name,
                    direct_method(K, 0.0).name,
                    direct_method(K, float(development_gate["selected_sigma"])).name,
                    direct_method(500, 0.0).name,
                ]
            )
        ][["method", "subject", "lew_initial", "lew_final", "relative_lew_auc", "gap_closure_100"]]
        step_target = json.loads((OUT / "development" / "NORMALIZED_STEP_TARGET.json").read_text())[
            "normalized_step_target"
        ]
        validation_path = OUT / "development" / "COMMON_RANDOM_NUMBERS_AUDIT.json"
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
        lines.extend(
            [
                "",
                "## Development gate result",
                "",
                "The mechanism did not survive the registered HGD development selection. Every nonzero sigma was worse "
                "than sigma=0 in mean relative exact-LEW AUC, and none improved 2 of 3 development subjects. "
                "Consequently Phase C held-out HGD, the raw-SGD control, Phase D matched-concentration baselines, "
                "BNCI transfer, and full expansion were not run.",
                "",
                f"The common normalized step target was `{step_target:.12g}`, derived exactly as preregistered from "
                "the median initial normalized-SPDHSW update norm at raw LR 10000.",
                "",
                frame_markdown(selection_table),
                "",
                "Per-subject outcomes for the selected development comparison and references:",
                "",
                frame_markdown(selected_summary.sort_values(["method", "subject"])),
                "",
                "## Run integrity",
                "",
                f"- Development run CSVs: {validation.get('run_csv_count', 'pending')}/"
                f"{validation.get('expected_run_csv_count', 'pending')}; rows per run: "
                f"{validation.get('rows_per_run_values', 'pending')}.",
                f"- Exact LEW epochs: {validation.get('evaluation_epochs', 'pending')}.",
                f"- Divergences: {validation.get('divergence_count', 'pending')}; NaN rows: "
                f"{validation.get('nan_count', 'pending')}.",
                "- Fresh deterministic epoch seeds were shared across methods; the CRN audit is in "
                "`development/COMMON_RANDOM_NUMBERS_AUDIT.json`.",
                "- All run CSVs, including negative outcomes, are retained. No clipping, early stopping, "
                "preprocessing change, or post-hoc sigma expansion was used.",
            ]
        )
    lines.extend(
        [
            "",
            "## Failures and negative results",
            "",
            "- The first synthetic invocation stopped before producing scientific draw records because CUDA "
            "`ndtri` required an unavailable NVRTC builtins library. The failure log is retained as "
            "`logs/synthetic_attempt1.log`. The deterministic spectrum construction was moved to float64 CPU "
            "and copied to physical GPU 3; a GPU smoke regression was then added and all 63 tests passed.",
            "- The synthetic mechanism gate passed, but this did not predict an HGD optimization gain. The HGD "
            "development gate failed cleanly with no divergence or nonfinite values.",
            "- Because the registered gate failed, no held-out or transfer result exists; `runs/NOT_RUN.md` records "
            "that deliberate stop.",
            "",
            "## Figures and tables",
            "",
            "- `fig_spectrum_weights.png` and `fig_synthetic_capture.png` summarize Phase A.",
            "- `fig_lew_vs_epoch.png`, `fig_lew_vs_wallclock.png`, and "
            "`fig_lew_vs_ambient_projections.png` show development-only trajectories.",
            "- `fig_interaction.png` is explicitly a development diagnostic, not the unrun held-out primary statistic.",
            "- `fig_gradient_update_stability.png` and `fig_ess_entropy.png` show optimization and concentration controls.",
            "- `CORE_RESULTS.csv` is development-only because the held-out gate was never reached.",
        ]
    )
    lines.extend(
        [
            "",
            "## Exact commands and environment",
            "",
            "```bash",
            "nvidia-smi -i 3",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m pytest -q --junitxml=results/lognormal_spectral_spdhsw_v1/TEST_RESULTS.xml",
            "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_logspectral_spdhsw --phase synthetic",
            "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_logspectral_spdhsw --phase development",
            "PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_logspectral_spdhsw --phase finalize",
            "```",
            "",
            f"- Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}.",
            f"- Host: {platform.node()}; branch: exp/lognormal-spectral-spdhsw-v1.",
            f"- Starting direct-pilot commit: 4edf5dda470c5e525c5feb274462414751348b4b; "
            "audit/tests checkpoint: 39bc01f2aec9e9cd1b5d145319c53d601cc9fd86; "
            "synthetic checkpoint: dfc7645eee08ef714a0589ec638ad4ca6f18b30c.",
            f"- Finalization parent commit: {subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()}.",
            "- Device: physical GPU 3, NVIDIA RTX 6000 Ada Generation.",
            "",
            "## Scope and claims",
            "",
            "See `CLAIM_LEDGER.md`. Negative gates stop expansion; no prohibited claim is inferred from a finite-bank outcome.",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def write_environment(device_info: dict[str, object]) -> None:
    payload = {
        "command_python": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": str(PROJECT),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT, text=True).strip(),
        "commit_at_invocation": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "amp": False,
        "autocast": False,
        "dtype": str(DTYPE),
        "device": device_info,
    }
    dump_json(OUT / "ENVIRONMENT.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("synthetic", "development", "finalize"), required=True)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    verify_frozen()
    configure_numerics()
    if args.phase == "finalize":
        validation = finalize_development_null()
        print(
            f"[FINALIZE] stop_after_development_null validation_pass={validation['validation_pass']}",
            flush=True,
        )
        return 0
    device_info = check_gpu3()
    write_environment(device_info)
    if args.phase == "synthetic":
        gate = synthetic_phase()
    else:
        gate = development_phase(rerun=args.rerun)
    print(f"[{args.phase.upper()} GATE] {gate['decision']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
