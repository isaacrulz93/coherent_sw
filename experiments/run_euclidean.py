#!/usr/bin/env python
"""Run the preregistered controlled Euclidean experiments E1--E5."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from coherent_slicing import (
    cvar_power,
    directional_costs,
    ebsw_exp_power,
    entropic_power,
    evar_power,
    power_ebsw_power,
    sample_unit_directions,
    sw_power,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "coherent_sw" / "euclidean_v1"
DTYPE = torch.float64

CONFIG = {
    "version": "euclidean_v1",
    "seed": 6398,
    "dtype": "float64",
    "p": 2,
    "fixed_beta": 2.0,
    "power_gamma": 1.0,
    "evar_kappa": 0.5,
    "cvar_alpha": 0.2,
    "E1": {"angular_directions": 2**18},
    "E2": {"scales": [0.25, 0.5, 1.0, 2.0, 4.0], "directions": 4096},
    "E3": {
        "directions": 4096,
        "kappas": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        "alphas": [1.0, 0.5, 0.2, 0.1, 0.05, 0.02],
    },
    "E4": {
        "dimensions": [16, 64, 256],
        "L": [20, 40, 100, 200, 500, 1000],
        "repeats": 16,
        "reference_L": 20000,
        "samples": 96,
    },
    "E5": {
        "seeds": [6398, 3654, 1788],
        "epochs": 100,
        "train_L": 100,
        "eval_L": 5000,
        "eval_every": 10,
        "learning_rates": {
            "sw": [0.01, 0.03, 0.1],
            "ebsw_exp": [0.003, 0.01, 0.03],
            "evar": [0.01, 0.03, 0.1],
            "cvar": [0.01, 0.03, 0.1],
            "sampled_max": [0.003, 0.01, 0.03],
        },
        "normalized_reference_lr": 0.03,
    },
}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def angular_directions(count: int) -> torch.Tensor:
    angle = (torch.arange(count, dtype=DTYPE) + 0.5) * math.pi / count
    return torch.stack((angle.cos(), angle.sin()), dim=1)


def exact_triple() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[-2.0, 2.0], [-2.0, 2.0]], dtype=DTYPE),
        torch.tensor([[-1.0, -1.0], [0.0, 1.0]], dtype=DTYPE),
        torch.tensor([[1.0, -2.0], [1.0, -2.0]], dtype=DTYPE),
    )


def entropy(weights: torch.Tensor) -> float:
    positive = weights > 0
    return float(-(weights[positive] * weights[positive].log()).sum())


def kl_uniform(weights: torch.Tensor) -> float:
    positive = weights > 0
    return float((weights[positive] * (weights[positive].log() + math.log(weights.numel()))).sum())


def ess(weights: torch.Tensor) -> float:
    return float(1.0 / weights.square().sum())


def method_power(name: str, h: torch.Tensor, *, gamma: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    count = h.numel()
    uniform = torch.full_like(h, 1.0 / count)
    if name == "sw":
        return sw_power(h), uniform
    if name == "ebsw_exp":
        weights = torch.softmax(CONFIG["fixed_beta"] * h, dim=0)
        return ebsw_exp_power(h, CONFIG["fixed_beta"]), weights.detach()
    if name == "power_ebsw":
        exponent = CONFIG["power_gamma"] if gamma is None else gamma
        if exponent == 0:
            weights = uniform
        elif float(h.max()) == 0:
            weights = uniform
        else:
            weights = (h / h.max()).pow(exponent)
            weights = weights / weights.sum()
        return power_ebsw_power(h, exponent), weights.detach()
    if name == "entropic":
        weights = torch.softmax(CONFIG["fixed_beta"] * h, dim=0)
        return entropic_power(h, CONFIG["fixed_beta"]), weights.detach()
    if name == "evar":
        result = evar_power(h, CONFIG["evar_kappa"])
        return result.value, result.weights
    if name == "cvar":
        result = cvar_power(h, CONFIG["cvar_alpha"])
        return result.value, result.weights
    if name == "sampled_max":
        weights = torch.zeros_like(h)
        weights[int(torch.argmax(h))] = 1.0
        return h.max(), weights
    raise ValueError(name)


METHODS = ("sw", "ebsw_exp", "power_ebsw", "entropic", "evar", "cvar", "sampled_max")


def e1(out: Path) -> dict:
    directions = angular_directions(CONFIG["E1"]["angular_directions"])
    mu, nu, eta = exact_triple()
    pairs = ((mu, nu), (nu, eta), (mu, eta))
    costs = [directional_costs(a, b, directions, p=1) for a, b in pairs]
    rows = []
    fixtures = [
        ("exact_p1_gamma1", 1.0),
        ("power_regime_uniform", 0.0),
        ("power_regime_weak", 0.25),
        ("power_regime_violating", 1.0),
        ("power_regime_max_limit", 8.0),
    ]
    for fixture, fixture_gamma in fixtures:
        for method in METHODS:
            gamma = fixture_gamma if method == "power_ebsw" else None
            distance = [float(method_power(method, h, gamma=gamma)[0]) for h in costs]
            rows.append(
                {
                    "fixture": fixture,
                    "method": method,
                    "p": 1,
                    "gamma": gamma,
                    "D_mu_nu": distance[0],
                    "D_nu_eta": distance[1],
                    "D_mu_eta": distance[2],
                    "triangle_slack": distance[2] - distance[0] - distance[1],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "e1_triangle_slack.csv", index=False)
    exact_direct = 5.0 * math.pi / 4.0
    exact_leg = 5.0 * (math.pi + 1.0) / (2.0 * (math.sqrt(10.0) + math.sqrt(5.0)))

    # Descriptive independent-bank control.  It is intentionally not a gate.
    generator = torch.Generator().manual_seed(1729)
    clouds = [torch.randn(8, 4, generator=generator, dtype=DTYPE) for _ in range(3)]
    controls = []
    for seed in range(100):
        for method in ("evar", "cvar"):
            h = [
                directional_costs(
                    clouds[a], clouds[b], sample_unit_directions(1, 4, seed=1000 * seed + pair), p=2
                )
                for pair, (a, b) in enumerate(((0, 1), (1, 2), (0, 2)))
            ]
            distance = [float(method_power(method, value)[0].sqrt()) for value in h]
            controls.append(
                {
                    "seed": seed,
                    "method": method,
                    "L_per_pair": 1,
                    "shared_directions": False,
                    "triangle_slack": distance[2] - distance[0] - distance[1],
                    "theorem_test": False,
                }
            )
    pd.DataFrame(controls).to_csv(out / "e1_independent_direction_control.csv", index=False)
    return {
        "exact_leg": exact_leg,
        "exact_direct": exact_direct,
        "exact_power_ebsw_violation": exact_direct - 2 * exact_leg,
        "worst_evar_shared_slack": float(frame[frame.method == "evar"].triangle_slack.max()),
        "worst_cvar_shared_slack": float(frame[frame.method == "cvar"].triangle_slack.max()),
        "independent_control_max_slack": float(pd.DataFrame(controls).triangle_slack.max()),
    }


def gaussian_mixture_pair(q: int, samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    half = samples // 2
    diagonal = torch.linspace(0.25, 1.5, q, dtype=DTYPE).sqrt()
    source = torch.randn(samples, q, generator=generator, dtype=DTYPE) * diagonal
    target = torch.randn(samples, q, generator=generator, dtype=DTYPE) * diagonal.flip(0)
    shift = torch.zeros(q, dtype=DTYPE)
    shift[: min(5, q)] = torch.linspace(1.5, 0.3, min(5, q), dtype=DTYPE)
    source[:half] -= 0.35 * shift
    source[half:] += 0.35 * shift
    target[:half] -= 0.2 * shift
    target[half:] += shift
    return source, target


def e2(out: Path) -> dict:
    source, target = gaussian_mixture_pair(2, 128, CONFIG["seed"])
    directions = sample_unit_directions(CONFIG["E2"]["directions"], 2, seed=CONFIG["seed"] + 1)
    rows = []
    for scale in CONFIG["E2"]["scales"]:
        h = directional_costs(scale * source, scale * target, directions, p=2)
        for method in ("ebsw_exp", "evar"):
            value, weights = method_power(method, h)
            distance = float(value.sqrt())
            beta_star = evar_power(h, CONFIG["evar_kappa"]).beta if method == "evar" else CONFIG["fixed_beta"]
            rows.append(
                {
                    "scale": scale,
                    "method": method,
                    "power_value": float(value),
                    "distance": distance,
                    "distance_over_scale": distance / scale,
                    "entropy": entropy(weights),
                    "kl_uniform": kl_uniform(weights),
                    "ess": ess(weights),
                    "beta": beta_star,
                    "max_weight": float(weights.max()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "e2_scale_equivariance.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for method, group in frame.groupby("method"):
        group = group.sort_values("scale")
        axes[0].plot(group.scale, group.distance_over_scale, marker="o", label=method)
        axes[1].plot(group.scale, group.kl_uniform, marker="o", label=method)
        axes[2].plot(group.scale, group.ess, marker="o", label=method)
    axes[0].set_ylabel("D(cX,cY) / c")
    axes[1].set_ylabel("KL(weights || uniform)")
    axes[2].set_ylabel("effective support (ESS)")
    for axis in axes:
        axis.set_xlabel("scale c")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(out / "fig_scale_equivariance.png", dpi=180)
    plt.close(figure)

    ev = frame[frame.method == "evar"]
    return {
        "evar_D_over_c_range": float(ev.distance_over_scale.max() - ev.distance_over_scale.min()),
        "evar_KL_range": float(ev.kl_uniform.max() - ev.kl_uniform.min()),
        "ebsw_D_over_c_range": float(
            frame[frame.method == "ebsw_exp"].distance_over_scale.max()
            - frame[frame.method == "ebsw_exp"].distance_over_scale.min()
        ),
    }


def e3(out: Path) -> dict:
    source, target = gaussian_mixture_pair(8, 128, CONFIG["seed"] + 20)
    directions = sample_unit_directions(CONFIG["E3"]["directions"], 8, seed=CONFIG["seed"] + 21)
    h = directional_costs(source, target, directions, p=2)
    max_distance = float(h.max().sqrt())
    rows = []
    for kappa in CONFIG["E3"]["kappas"]:
        result = evar_power(h, kappa)
        distance = float(result.value.sqrt())
        rows.append(
            {
                "family": "evar",
                "parameter": kappa,
                "power_value": float(result.value),
                "distance": distance,
                "entropy": result.entropy,
                "ess": ess(result.weights),
                "distance_to_max": max_distance - distance,
                "kl_uniform": result.achieved_kl,
                "beta_star": result.beta,
                "status": result.status,
            }
        )
    for alpha in CONFIG["E3"]["alphas"]:
        result = cvar_power(h, alpha)
        distance = float(result.value.sqrt())
        rows.append(
            {
                "family": "cvar",
                "parameter": alpha,
                "power_value": float(result.value),
                "distance": distance,
                "entropy": entropy(result.weights),
                "ess": ess(result.weights),
                "distance_to_max": max_distance - distance,
                "kl_uniform": kl_uniform(result.weights),
                "beta_star": math.nan,
                "status": "exact_cap",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "e3_interpolation.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for family, group in frame.groupby("family"):
        axes[0].plot(group.parameter, group.power_value, marker="o", label=family)
        axes[1].plot(group.parameter, group.ess, marker="o", label=family)
        axes[2].plot(group.parameter, group.distance_to_max, marker="o", label=family)
    axes[0].set_ylabel("aggregated power value")
    axes[1].set_ylabel("effective support (ESS)")
    axes[2].set_ylabel("Max-SW reference minus distance")
    for axis in axes:
        axis.set_xlabel("kappa (EVaR) or alpha (CVaR)")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(out / "fig_interpolation.png", dpi=180)
    plt.close(figure)
    return {"max_sw_reference": max_distance, "rows": len(frame)}


def e4(out: Path, quick: bool) -> dict:
    setting = dict(CONFIG["E4"])
    if quick:
        setting.update(repeats=4, reference_L=5000)
    rows = []
    reference_rows = []
    for q in setting["dimensions"]:
        source, target = gaussian_mixture_pair(q, setting["samples"], CONFIG["seed"] + q)
        reference_directions = sample_unit_directions(
            setting["reference_L"], q, seed=CONFIG["seed"] + 10_000 + q
        )
        reference_h = directional_costs(source, target, reference_directions, p=2)
        references = {}
        for method in METHODS:
            value, _ = method_power(method, reference_h)
            references[method] = float(value.sqrt())
            reference_rows.append(
                {"q": q, "method": method, "reference_L": setting["reference_L"], "distance": references[method]}
            )
        del reference_directions, reference_h
        for count in setting["L"]:
            for repeat in range(setting["repeats"]):
                directions = sample_unit_directions(
                    count, q, seed=CONFIG["seed"] + q * 100_000 + count * 100 + repeat
                )
                h = directional_costs(source, target, directions, p=2)
                for method in METHODS:
                    value, _ = method_power(method, h)
                    estimate = float(value.sqrt())
                    rows.append(
                        {
                            "q": q,
                            "L": count,
                            "repeat": repeat,
                            "method": method,
                            "estimate": estimate,
                            "own_metric_reference": references[method],
                            "error": estimate - references[method],
                        }
                    )
    estimates = pd.DataFrame(rows)
    estimates.to_csv(out / "e4_finite_direction_estimates.csv", index=False)
    references_frame = pd.DataFrame(reference_rows)
    references_frame.to_csv(out / "e4_metric_references.csv", index=False)
    summary = (
        estimates.assign(squared_error=lambda x: x.error**2)
        .groupby(["q", "L", "method"], as_index=False)
        .agg(rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))), bias=("error", "mean"))
    )
    summary.to_csv(out / "e4_finite_direction_rmse.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for axis, q in zip(axes, setting["dimensions"]):
        block = summary[summary.q == q]
        for method, group in block.groupby("method"):
            axis.plot(group.L, group.rmse, marker="o", label=method)
        axis.set_title(f"q={q}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("directions L")
        axis.set_ylabel("RMSE to own high-L reference")
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, loc="best")
    figure.tight_layout()
    figure.savefig(out / "fig_finite_direction_error.png", dpi=180)
    plt.close(figure)
    return {
        "reference_L": setting["reference_L"],
        "repeats": setting["repeats"],
        "max_rmse": float(summary.rmse.max()),
    }


FLOW_METHODS = ("sw", "ebsw_exp", "evar", "cvar", "sampled_max")


def flow_power(method: str, h: torch.Tensor) -> torch.Tensor:
    return method_power(method, h)[0]


def evaluation_sw(source: torch.Tensor, target: torch.Tensor, directions: torch.Tensor) -> float:
    return float(directional_costs(source, target, directions, p=2).mean().sqrt())


def flow_initial_clouds(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    source, target = gaussian_mixture_pair(2, 32, seed)
    source = source + torch.tensor([-2.0, 1.25], dtype=DTYPE)
    target = target + torch.tensor([1.0, -0.5], dtype=DTYPE)
    return source, target


def initial_sw_step_norm(seed: int, base_lr: float, count: int) -> float:
    source, target = flow_initial_clouds(seed)
    parameter = source.clone().requires_grad_(True)
    directions = sample_unit_directions(count, 2, seed=seed + 900_000)
    loss = sw_power(directional_costs(parameter, target, directions, p=2))
    loss.backward()
    return base_lr * float(parameter.grad.norm())


def one_flow(
    method: str,
    seed: int,
    *,
    learning_rate: float,
    normalized_step: float | None,
    eval_directions: torch.Tensor,
    epochs: int,
    train_count: int,
    eval_every: int,
) -> list[dict]:
    source, target = flow_initial_clouds(seed)
    parameter = source.clone().requires_grad_(True)
    rows = []
    elapsed = 0.0
    for epoch in range(epochs + 1):
        if epoch % eval_every == 0:
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "control": "normalized" if normalized_step is not None else "lr_grid",
                    "learning_rate": learning_rate,
                    "target_step_norm": normalized_step,
                    "epoch": epoch,
                    "eval_independent_sw": evaluation_sw(parameter.detach(), target, eval_directions),
                    "train_power_loss": math.nan,
                    "gradient_norm": math.nan,
                    "update_norm": math.nan,
                    "seconds_cum": elapsed,
                    "diverged": False,
                }
            )
        if epoch == epochs:
            break
        tick = time.perf_counter()
        directions = sample_unit_directions(train_count, 2, seed=seed + epoch * 7919)
        h = directional_costs(parameter, target, directions, p=2)
        loss = flow_power(method, h)
        loss.backward()
        gradient_norm = float(parameter.grad.norm())
        if normalized_step is None:
            update = -learning_rate * parameter.grad
        elif gradient_norm > 0 and math.isfinite(gradient_norm):
            update = -normalized_step * parameter.grad / gradient_norm
        else:
            update = torch.zeros_like(parameter)
        update_norm = float(update.norm())
        with torch.no_grad():
            parameter.add_(update)
        parameter.grad = None
        elapsed += time.perf_counter() - tick
        if rows and rows[-1]["epoch"] == epoch:
            rows[-1].update(
                train_power_loss=float(loss.detach()),
                gradient_norm=gradient_norm,
                update_norm=update_norm,
                seconds_cum=elapsed,
                diverged=not bool(torch.isfinite(parameter).all()),
            )
        if not bool(torch.isfinite(parameter).all()):
            break
    return rows


def e5(out: Path, quick: bool) -> dict:
    setting = dict(CONFIG["E5"])
    epochs = 40 if quick else setting["epochs"]
    eval_count = 2000 if quick else setting["eval_L"]
    seeds = setting["seeds"][:1] if quick else setting["seeds"]
    # This target is computed before any adaptive run and then frozen.
    initial_steps = [
        initial_sw_step_norm(seed, setting["normalized_reference_lr"], setting["train_L"])
        for seed in seeds
    ]
    target_step = float(np.median(initial_steps))
    rows = []
    for seed in seeds:
        eval_directions = sample_unit_directions(eval_count, 2, seed=seed + 4_000_000)
        for method in FLOW_METHODS:
            for learning_rate in setting["learning_rates"][method]:
                rows.extend(
                    one_flow(
                        method,
                        seed,
                        learning_rate=learning_rate,
                        normalized_step=None,
                        eval_directions=eval_directions,
                        epochs=epochs,
                        train_count=setting["train_L"],
                        eval_every=setting["eval_every"],
                    )
                )
            rows.extend(
                one_flow(
                    method,
                    seed,
                    learning_rate=math.nan,
                    normalized_step=target_step,
                    eval_directions=eval_directions,
                    epochs=epochs,
                    train_count=setting["train_L"],
                    eval_every=setting["eval_every"],
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "e5_gradient_flow.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    grid = frame[frame.control == "lr_grid"]
    for (method, lr), group in grid.groupby(["method", "learning_rate"]):
        curve = group.groupby("epoch", as_index=False).eval_independent_sw.mean()
        axes[0].plot(curve.epoch, curve.eval_independent_sw, label=f"{method} lr={lr:g}")
    normalized = frame[frame.control == "normalized"]
    for method, group in normalized.groupby("method"):
        curve = group.groupby("epoch", as_index=False).eval_independent_sw.mean()
        axes[1].plot(curve.epoch, curve.eval_independent_sw, label=method)
    axes[0].set_title("preregistered learning-rate grids")
    axes[1].set_title("normalized-update control")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.set_ylabel("independent high-L SW")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(out / "fig_gradient_flow.png", dpi=180)
    plt.close(figure)
    return {
        "target_normalized_step_norm": target_step,
        "runs": len(FLOW_METHODS) * len(seeds) * 4,
        "diverged_records": int(frame.diverged.sum()),
        "final_normalized": frame[(frame.control == "normalized") & (frame.epoch == epochs)]
        .groupby("method")
        .eval_independent_sw.mean()
        .to_dict(),
    }


def report(out: Path, summaries: dict, elapsed: float, failures: list[str], quick: bool) -> None:
    e1s, e2s, e4s, e5s = (summaries[key] for key in ("E1", "E2", "E4", "E5"))
    lines = [
        "# Coherent sliced-Wasserstein controlled experiments",
        "",
        f"Status: {'PASS' if not failures else 'FAIL'} (mode={'quick' if quick else 'full'}, {elapsed:.2f} s).",
        "",
        "## Validation and notable outcomes",
        "",
        f"- The analytic p=gamma=1 pure-power EBSW counterexample was reproduced: "
        f"leg={e1s['exact_leg']:.12f}, direct={e1s['exact_direct']:.12f}, "
        f"triangle violation={e1s['exact_power_ebsw_violation']:.12f}.",
        f"- In the common-direction E1 audit, worst EVaR slack was "
        f"{e1s['worst_evar_shared_slack']:.3e} and worst CVaR slack was "
        f"{e1s['worst_cvar_shared_slack']:.3e}.",
        f"- The independent-direction control is descriptive only; its maximum observed slack was "
        f"{e1s['independent_control_max_slack']:.3e}. No theorem claim is attached to it.",
        f"- EVaR scale audit: range of D(cX,cY)/c={e2s['evar_D_over_c_range']:.3e}; "
        f"fixed-beta EBSW range={e2s['ebsw_D_over_c_range']:.3e}.",
        f"- E4 compares every estimator only with its own L={e4s['reference_L']} reference "
        f"({e4s['repeats']} repeats); SW is never used as the target for EVaR/CVaR.",
        f"- E5 normalized-update target norm was {e5s['target_normalized_step_norm']:.6g}; "
        f"diverged records={e5s['diverged_records']}.",
        "",
        "## Negative results and failures",
        "",
    ]
    if failures:
        lines.extend([f"- {item}" for item in failures])
    else:
        lines.append("- No numerical or execution failures were observed in E1--E5.")
    lines.extend(
        [
            "- Fixed-beta and pure-power EBSW are baselines, not asserted metrics; positive triangle slacks are retained in the CSV.",
            "- The sampled maximum is a finite-bank reference, not a sphere-optimized Max-SW implementation.",
            "- E4 high-direction references remain finite Monte Carlo references, not exact population values.",
            "",
            "## Artifacts",
            "",
            "All CSVs, figures, and the frozen configuration are in this directory. "
            "The MOABB pilot has a separate output tree and is gated on this report and the theorem tests.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="reduced E4/E5 repetitions for smoke validation")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = {
        **CONFIG,
        "quick": bool(args.quick),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "created_before_experiment": True,
    }
    config_path = args.out / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        comparable_existing = {k: existing.get(k) for k in frozen if k not in {"platform"}}
        comparable_new = {k: frozen[k] for k in frozen if k not in {"platform"}}
        if comparable_existing != comparable_new:
            raise RuntimeError(f"refusing to overwrite a different frozen config: {config_path}")
    else:
        write_json(config_path, frozen)

    started = time.perf_counter()
    summaries = {}
    failures = []
    experiments: list[tuple[str, Callable[[], dict]]] = [
        ("E1", lambda: e1(args.out)),
        ("E2", lambda: e2(args.out)),
        ("E3", lambda: e3(args.out)),
        ("E4", lambda: e4(args.out, args.quick)),
        ("E5", lambda: e5(args.out, args.quick)),
    ]
    for name, function in experiments:
        tick = time.perf_counter()
        try:
            summaries[name] = function()
            print(f"[{name}] PASS ({time.perf_counter() - tick:.2f}s)", flush=True)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            summaries[name] = {"error": failures[-1]}
            print(f"[{name}] FAIL: {failures[-1]}", file=sys.stderr, flush=True)
            break
    elapsed = time.perf_counter() - started
    write_json(args.out / "summary.json", {"summaries": summaries, "failures": failures, "elapsed_seconds": elapsed})
    if all(key in summaries and "error" not in summaries[key] for key in ("E1", "E2", "E4", "E5")):
        report(args.out, summaries, elapsed, failures, args.quick)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
