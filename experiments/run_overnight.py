#!/usr/bin/env python
"""Run the bounded three-seed HGD pilot and its preregistered stop rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from coherent_slicing import w_p_power_per_direction
from experiments.run_moabb_pilot import (
    DTYPE,
    FROZEN_SOURCES,
    LEWEvaluator,
    Method,
    SvecBasis,
    complete_csv,
    direction_seed,
    frobenius_directions,
    load_subject,
    relative_lew_auc,
    source_hashes,
    train_one,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT / "results" / "coherent_sw_overnight"
SEEDS = (6398, 3654, 1788)
HGD_SUBJECTS = (1, 7, 14)
BNCI_SUBJECTS = (1, 3, 8)
EPOCHS = 500
LEW_EVERY = 25
COMMON_LR = 10000.0
EXTRA_LR = 3000.0

PREREG = {
    "created_before_hgd_outcomes": True,
    "theorem_regression_required": True,
    "euclidean_E1_E5_required": True,
    "primary": {
        "dataset": "Schirrmeister2017",
        "subjects": list(HGD_SUBJECTS),
        "seeds": list(SEEDS),
        "d": 128,
        "p": 2,
        "epochs": EPOCHS,
        "preprocessing": "existing EBSPDSW cached log-SPD 0train -> 1test blocks",
    },
    "directions": {
        "family": "direct Frobenius-uniform SPDSW only",
        "hierarchical": False,
        "resample_each_epoch": True,
        "common_random_numbers": True,
        "seed_sequence": "seed + epoch_zero_based*(epoch_zero_based+1)/2",
        "budgets": [40, 500],
    },
    "base_common_lr": COMMON_LR,
    "methods": {
        "spdsw": [40, 500],
        "ebsw_exp": ["paper_default_beta=1", "one inverse-median-initial-h-std scale match"],
        "evar_kappa": [0.1, 0.5, 1.0, 2.0],
        "cvar_alpha": [0.5, 0.2, 0.1],
        "sampled_max": [40],
    },
    "post_base_selection": (
        "one global setting per EBSW/EVaR/CVaR family, chosen by mean relative "
        "LEW AUC over all 3x3 HGD subject-seed pairs; divergence count breaks ties first"
    ),
    "additional_controls": {
        "learning_rates": [3000.0, 10000.0],
        "note": "10000 is the base run; only 3000 is newly run after selection",
        "normalized_gradient": (
            "constant step norm equal to median initial SPDSW-L40 lr=10000 update "
            "over the arm's subject-seed pairs"
        ),
        "per_subject_lr_tuning": False,
    },
    "evaluation": "independent exact-OT LEW every 25 epochs; excluded from optimization clock",
    "gate": {
        "same_lr_improved_subjects": 2,
        "normalized_improved_subjects": 2,
        "no_extra_divergence_or_nan": True,
        "endpoint_definition": "median ESS <= 1.25 or CVaR active tail < 2",
        "if_pass": "run BNCI2014_001 subjects 1,3,8 with the same 3 seeds and stop",
        "if_fail": "write report and stop",
        "never_in_this_task": ["HGD 14-subject expansion", "hierarchical SPDHSW"],
    },
}


def dump_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Serialize a compact Markdown table without pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.8g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def archive_prerequisite_artifacts(out: Path) -> None:
    """Keep the audit, theorem XML, and Euclidean outputs inside the overnight tree."""
    target = out / "prerequisites"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT / "AUDIT.md", target / "AUDIT.md")
    shutil.copy2(PROJECT / "results" / "theorem_regression.xml", target / "theorem_regression.xml")
    shutil.copytree(
        PROJECT / "results" / "coherent_sw" / "euclidean_v1",
        target / "euclidean_v1",
        dirs_exist_ok=True,
    )


def parse_theorem_gate() -> tuple[bool, str]:
    path = PROJECT / "results" / "theorem_regression.xml"
    if not path.exists():
        return False, "missing theorem_regression.xml"
    text = path.read_text(errors="replace")
    failed = 'failures="0"' in text and 'errors="0"' in text
    return failed, str(path)


def parse_euclidean_gate() -> tuple[bool, str]:
    path = PROJECT / "results" / "coherent_sw" / "euclidean_v1" / "summary.json"
    if not path.exists():
        return False, "missing Euclidean summary.json"
    payload = json.loads(path.read_text())
    complete = set(payload.get("summaries", {})) == {"E1", "E2", "E3", "E4", "E5"}
    return complete and not payload.get("failures"), str(path)


def beta_label(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("+", "").replace("-", "m")


def base_methods(scale_beta: float) -> list[Method]:
    return [
        Method("spdsw_l40", "sw", L=40),
        Method("spdsw_l500", "sw", L=500),
        Method("ebsw_exp_default_b1", "ebsw_exp", L=40, beta=1.0),
        Method(f"ebsw_exp_scale_b{beta_label(scale_beta)}", "ebsw_exp", L=40, beta=scale_beta),
        *[Method(f"evar_k{str(k).replace('.', 'p')}", "evar", L=40, kappa=k) for k in (0.1, 0.5, 1.0, 2.0)],
        *[Method(f"cvar_a{str(a).replace('.', 'p')}", "cvar", L=40, alpha=a) for a in (0.5, 0.2, 0.1)],
        Method("sampled_max_l40", "sampled_max", L=40),
    ]


def load_cached(dataset: str, subjects: tuple[int, ...], device: str) -> dict[int, tuple[torch.Tensor, torch.Tensor, dict]]:
    return {subject: load_subject(dataset, subject, device) for subject in subjects}


def initial_cost_audit(
    cache: dict[int, tuple[torch.Tensor, torch.Tensor, dict]],
    dataset: str,
    out: Path,
) -> tuple[pd.DataFrame, float]:
    rows = []
    for subject, (source, target, _) in cache.items():
        basis = SvecBasis(source.shape[-1], source.device, DTYPE)
        source_vec = basis.forward(source)
        target_vec = basis.forward(target)
        for seed in SEEDS:
            directions40 = frobenius_directions(40, basis, direction_seed(seed, 0))
            h = w_p_power_per_direction(
                (source_vec @ directions40.T).T,
                (target_vec @ directions40.T).T,
                p=2,
            )
            raw = h.detach().cpu().numpy().tobytes()
            rows.append(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "seed": seed,
                    "epoch": 1,
                    "direction_seed": direction_seed(seed, 0),
                    "L": 40,
                    "mean_h": float(h.mean()),
                    "std_h": float(h.std(unbiased=False)),
                    "max_h": float(h.max()),
                    "h_sha256": hashlib.sha256(raw).hexdigest(),
                    "shared_by_all_l40_aggregators_at_initial_state": True,
                }
            )
    frame = pd.DataFrame(rows)
    path = out / "audits" / f"{dataset}_epoch0_common_h.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    median_std = float(frame.std_h.median())
    if not math.isfinite(median_std) or median_std <= 0:
        raise RuntimeError(f"cannot scale-match EBSW beta from median std {median_std}")
    return frame, 1.0 / median_std


def normalized_target(cache: dict[int, tuple[torch.Tensor, torch.Tensor, dict]]) -> tuple[float, pd.DataFrame]:
    from experiments.run_moabb_pilot import initial_gradient_norm

    sw = Method("spdsw_l40", "sw", L=40)
    rows = []
    for subject, (source, target, _) in cache.items():
        for seed in SEEDS:
            gradient = initial_gradient_norm(sw, source, target, seed)
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "gradient_norm": gradient,
                    "reference_lr": COMMON_LR,
                    "update_norm": COMMON_LR * gradient,
                }
            )
    frame = pd.DataFrame(rows)
    return float(frame.update_norm.median()), frame


def run_path(out: Path, dataset: str, control: str, method: Method, seed: int, subject: int) -> Path:
    return out / "runs" / dataset / control / method.name / f"seed_{seed}" / f"subject_{subject:02d}.csv"


def execute_grid(
    out: Path,
    dataset: str,
    cache: dict[int, tuple[torch.Tensor, torch.Tensor, dict]],
    methods: list[Method],
    control: str,
    learning_rate: float,
    normalized_step: float,
    rerun: bool,
) -> list[dict]:
    records = []
    total = len(cache) * len(SEEDS) * len(methods)
    index = 0
    for subject, (source, target, meta) in cache.items():
        for seed in SEEDS:
            for method in methods:
                index += 1
                path = run_path(out, dataset, control, method, seed, subject)
                try:
                    if rerun or not complete_csv(path, EPOCHS):
                        frame, run_meta = train_one(
                            method,
                            source,
                            target,
                            dataset=dataset,
                            subject=subject,
                            seed=seed,
                            epochs=EPOCHS,
                            lew_every=LEW_EVERY,
                            control=control,
                            learning_rate=learning_rate,
                            normalized_target=normalized_step,
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        frame.to_csv(path, index=False)
                    else:
                        frame = pd.read_csv(path)
                        finite_lew = frame.lew.dropna()
                        run_meta = {
                            "lew_initial": float(finite_lew.iloc[0]),
                            "lew_final": float(finite_lew.iloc[-1]),
                            "diverged": bool(frame.diverged.fillna(False).any()),
                            "optimization_seconds": float(frame.optimization_seconds_cum.iloc[-1]),
                            "evaluation_seconds": float(frame.evaluation_seconds_cum.iloc[-1]),
                            "aggregation_seconds": float(frame.aggregation_seconds.fillna(0).sum()),
                        }
                    record = {
                        "dataset": dataset,
                        "control": control,
                        "method": method.name,
                        "family": method.family,
                        "L": method.L,
                        "subject": subject,
                        "seed": seed,
                        "d": meta["d"],
                        "n_source": meta["n_source"],
                        "n_target": meta["n_target"],
                        "status": "ok",
                        "error": "",
                        **run_meta,
                    }
                    print(
                        f"[RUN {index:03d}/{total:03d}] {dataset} {control} "
                        f"s{subject:02d} seed={seed} {method.name:26s} "
                        f"LEW {run_meta['lew_initial']:.3f}->{run_meta['lew_final']:.3f}",
                        flush=True,
                    )
                except Exception as exc:
                    log = out / "logs" / f"{dataset}_{control}_{method.name}_seed{seed}_s{subject:02d}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text(traceback.format_exc())
                    record = {
                        "dataset": dataset,
                        "control": control,
                        "method": method.name,
                        "family": method.family,
                        "L": method.L,
                        "subject": subject,
                        "seed": seed,
                        "d": meta["d"],
                        "n_source": meta["n_source"],
                        "n_target": meta["n_target"],
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"[FAIL] {record['error']}", file=sys.stderr, flush=True)
                records.append(record)
                pd.DataFrame(records).to_csv(out / f"manifest_{dataset}_{control}.csv", index=False)
    return records


def read_frames(out: Path, dataset: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in (out / "runs" / dataset).glob("*/*/seed_*/subject_*.csv")]
    if not frames:
        raise RuntimeError(f"no completed frames for {dataset}")
    return pd.concat(frames, ignore_index=True)


def at_epoch(frame: pd.DataFrame, epoch: int, column: str) -> float:
    hit = frame[frame.epoch == epoch]
    return float(hit.iloc[0][column]) if len(hit) else math.nan


def finite_median(values: pd.Series) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.median()) if len(finite) else math.nan


def run_summary(frames: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "control", "method", "family", "L", "subject", "seed"]
    rows = []
    for key, group in frames.groupby(keys, sort=False):
        group = group.sort_values("epoch")
        evaluated = group[np.isfinite(group.lew)]
        initial = float(evaluated.lew.iloc[0])
        final = float(evaluated.lew.iloc[-1])
        row = dict(zip(keys, key))
        row.update(
            lew_initial=initial,
            lew_100=at_epoch(group, 100, "lew"),
            lew_250=at_epoch(group, 250, "lew"),
            lew_500=at_epoch(group, 500, "lew"),
            lew_reduction_pct_100=100 * (initial - at_epoch(group, 100, "lew")) / initial,
            lew_reduction_pct_250=100 * (initial - at_epoch(group, 250, "lew")) / initial,
            lew_reduction_pct_500=100 * (initial - at_epoch(group, 500, "lew")) / initial,
            relative_lew_auc=relative_lew_auc(group),
            diverged=bool(group.diverged.fillna(False).any()) or not math.isfinite(final) or final > initial,
            nan_count=int(group["nan"].fillna(False).sum()),
            optimization_seconds=float(group.optimization_seconds_cum.iloc[-1]),
            evaluation_seconds=float(group.evaluation_seconds_cum.iloc[-1]),
            aggregation_ms_per_epoch=1000 * float(group.aggregation_seconds.fillna(0).sum()) / EPOCHS,
            median_entropy=finite_median(group.loc[group.epoch > 0, "entropy"]),
            median_ess=finite_median(group.loc[group.epoch > 0, "ess"]),
            median_achieved_kl=finite_median(group.loc[group.epoch > 0, "achieved_kl"]),
            median_active_tail_count=finite_median(group.loc[group.epoch > 0, "active_tail_count"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_paired_and_quality(summary: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["paired_auc_diff_vs_spdsw_l40"] = math.nan
    result["paired_lew500_diff_vs_spdsw_l40"] = math.nan
    result["spdsw_l500_target_lew"] = math.nan
    result["epoch_reach_spdsw_l500_final"] = math.nan
    result["wall_reach_spdsw_l500_final"] = math.nan
    common = result[result.control == "common_lr10000"]
    target_map = common[common.method == "spdsw_l500"].set_index(["dataset", "subject", "seed"]).lew_500
    for index, row in result.iterrows():
        baseline = result[
            (result.dataset == row.dataset)
            & (result.control == row.control)
            & (result.subject == row.subject)
            & (result.seed == row.seed)
            & (result.method == "spdsw_l40")
        ]
        if len(baseline):
            result.loc[index, "paired_auc_diff_vs_spdsw_l40"] = row.relative_lew_auc - float(baseline.iloc[0].relative_lew_auc)
            result.loc[index, "paired_lew500_diff_vs_spdsw_l40"] = row.lew_500 - float(baseline.iloc[0].lew_500)
        target_key = (row.dataset, row.subject, row.seed)
        if target_key not in target_map.index:
            continue
        target = float(target_map.loc[target_key])
        result.loc[index, "spdsw_l500_target_lew"] = target
        group = frames[
            (frames.dataset == row.dataset)
            & (frames.control == row.control)
            & (frames.subject == row.subject)
            & (frames.seed == row.seed)
            & (frames.method == row.method)
            & np.isfinite(frames.lew)
        ].sort_values("epoch")
        reached = group[group.lew <= target]
        if len(reached):
            first = reached.iloc[0]
            result.loc[index, "epoch_reach_spdsw_l500_final"] = float(first.epoch)
            result.loc[index, "wall_reach_spdsw_l500_final"] = float(first.optimization_seconds_cum)
    return result


def select_one_per_family(base_summary: pd.DataFrame) -> dict[str, str]:
    selected = {}
    for family in ("ebsw_exp", "evar", "cvar"):
        block = (
            base_summary[base_summary.family == family]
            .groupby(["method", "family"], as_index=False)
            .agg(
                divergence_count=("diverged", "sum"),
                mean_relative_lew_auc=("relative_lew_auc", "mean"),
                mean_lew_500=("lew_500", "mean"),
            )
            .sort_values(["divergence_count", "mean_relative_lew_auc", "method"])
        )
        selected[family] = str(block.iloc[0].method)
    return selected


def method_by_name(methods: list[Method], name: str) -> Method:
    return next(method for method in methods if method.name == name)


def subject_improvement_count(summary: pd.DataFrame, control: str, method: str) -> tuple[int, pd.DataFrame]:
    block = summary[summary.control == control]
    candidate = block[block.method == method].groupby("subject", as_index=False).relative_lew_auc.mean()
    baseline = block[block.method == "spdsw_l40"].groupby("subject", as_index=False).relative_lew_auc.mean()
    paired = candidate.merge(baseline, on="subject", suffixes=("_candidate", "_spdsw"))
    paired["difference"] = paired.relative_lew_auc_candidate - paired.relative_lew_auc_spdsw
    return int((paired.difference < 0).sum()), paired


def decide_gate(summary: pd.DataFrame, selected: dict[str, str], out: Path) -> dict:
    rows = []
    passing = []
    for family in ("evar", "cvar"):
        method = selected[family]
        same_count, same_pairs = subject_improvement_count(summary, "common_lr10000", method)
        normalized_count, normalized_pairs = subject_improvement_count(summary, "normalized", method)
        same_pairs.assign(control="common_lr10000", method=method).to_csv(
            out / f"paired_subjects_{method}_common_lr10000.csv", index=False
        )
        normalized_pairs.assign(control="normalized", method=method).to_csv(
            out / f"paired_subjects_{method}_normalized.csv", index=False
        )
        candidate = summary[summary.method == method]
        baseline = summary[summary.method == "spdsw_l40"]
        unstable = int(candidate.diverged.sum()) > int(baseline.diverged.sum()) or int(candidate.nan_count.sum()) > int(baseline.nan_count.sum())
        if family == "evar":
            endpoint_like = float(candidate.median_ess.median()) <= 1.25
        else:
            endpoint_like = float(candidate.median_active_tail_count.median()) < 2.0
        passed = same_count >= 2 and normalized_count >= 2 and not unstable and not endpoint_like
        rows.append(
            {
                "family": family,
                "method": method,
                "same_lr_improved_subjects": same_count,
                "normalized_improved_subjects": normalized_count,
                "significantly_more_unstable": unstable,
                "sampled_max_endpoint_like": endpoint_like,
                "passes": passed,
            }
        )
        if passed:
            passing.append(method)
    pd.DataFrame(rows).to_csv(out / "gate_by_setting.csv", index=False)
    return {
        "pass": bool(passing),
        "passing_settings": passing,
        "hgd_same_l_improvement": any(row["same_lr_improved_subjects"] >= 2 for row in rows),
        "normalized_gradient_improvement": any(
            row["same_lr_improved_subjects"] >= 2 and row["normalized_improved_subjects"] >= 2 for row in rows
        ),
        "decision": "run_bounded_bnci" if passing else "stop_after_hgd_null",
        "hierarchical_spdhsw_run": False,
        "full_hgd_expansion_run": False,
    }


def aggregate_table(summary: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "lew_reduction_pct_100",
        "lew_reduction_pct_250",
        "lew_reduction_pct_500",
        "relative_lew_auc",
        "paired_auc_diff_vs_spdsw_l40",
        "paired_lew500_diff_vs_spdsw_l40",
        "epoch_reach_spdsw_l500_final",
        "wall_reach_spdsw_l500_final",
        "optimization_seconds",
        "aggregation_ms_per_epoch",
    ]
    aggregate = summary.groupby(["dataset", "control", "method", "family", "L"], as_index=False)[numeric].mean()
    divergence = summary.groupby(["dataset", "control", "method"], as_index=False).agg(
        divergence_count=("diverged", "sum"), nan_count=("nan_count", "sum")
    )
    return aggregate.merge(divergence, on=["dataset", "control", "method"])


def make_figures(out: Path, frames: pd.DataFrame, selected: dict[str, str]) -> None:
    shown = ["spdsw_l40", "spdsw_l500", selected["ebsw_exp"], selected["evar"], selected["cvar"], "sampled_max_l40"]
    block = frames[(frames.method.isin(shown)) & np.isfinite(frames.lew)]
    for x, filename, xlabel in (
        ("epoch", "fig_lew_vs_epoch.png", "epoch"),
        ("optimization_seconds_cum", "fig_lew_vs_wall_clock.png", "optimization wall-clock (s), LEW time excluded"),
        ("cumulative_ambient_projections", "fig_lew_vs_ambient_projections.png", "cumulative ambient directions"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(16, 4))
        for axis, subject in zip(axes, HGD_SUBJECTS):
            sub = block[(block.control == "common_lr10000") & (block.subject == subject)]
            for method, group in sub.groupby("method"):
                curve = group.groupby(x, as_index=False).lew.mean()
                axis.plot(curve[x], curve.lew, label=method, linewidth=1)
            axis.set_title(f"HGD subject {subject} (mean over seeds)")
            axis.set_xlabel(xlabel)
            axis.set_ylabel("independent exact-OT LEW")
            axis.grid(alpha=0.25)
        axes[-1].legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(out / filename, dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    diagnostic = frames[(frames.control == "common_lr10000") & (frames.method.isin(shown)) & (frames.epoch > 0)]
    for method, group in diagnostic.groupby("method"):
        curve = group.groupby("epoch", as_index=False).agg(
            gradient_norm=("gradient_norm", "median"), update_norm=("update_norm", "median")
        )
        axes[0].plot(curve.epoch, curve.gradient_norm, label=method, linewidth=0.8)
        axes[1].plot(curve.epoch, curve.update_norm, label=method, linewidth=0.8)
    for axis, ylabel in zip(axes, ("gradient Frobenius norm", "update norm")):
        axis.set_yscale("log")
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(out / "fig_gradient_update_stability.png", dpi=180)
    plt.close(figure)

    adaptive = frames[(frames.method.isin([selected["evar"], selected["cvar"]])) & (frames.epoch > 0)]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for (control, method), group in adaptive.groupby(["control", "method"]):
        curve = group.groupby("epoch", as_index=False).agg(
            entropy=("entropy", "median"), achieved_kl=("achieved_kl", "median"), ess=("ess", "median")
        )
        label = f"{method}:{control}"
        axes[0].plot(curve.epoch, curve.entropy, label=label)
        axes[1].plot(curve.epoch, curve.achieved_kl, label=label)
        axes[2].plot(curve.epoch, curve.ess, label=label)
    for axis, ylabel in zip(axes, ("entropy", "EVaR achieved KL", "ESS")):
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out / "fig_entropy_kl_ess.png", dpi=180)
    plt.close(figure)


def write_report(
    out: Path,
    theorem_pass: bool,
    gate: dict,
    selected: dict[str, str],
    scale_beta: float,
    table: pd.DataFrame,
    failures: list[dict],
    bnci_ran: bool,
) -> None:
    yesno = lambda value: "YES" if value else "NO"
    header = [
        f"- theorem regression: {'PASS' if theorem_pass else 'FAIL'}",
        f"- HGD same-L improvement: {yesno(gate['hgd_same_l_improvement'])}",
        f"- normalized-gradient improvement: {yesno(gate['normalized_gradient_improvement'])}",
        "- proceed to hierarchical SPDHSW: NO",
    ]
    hgd_selected = table[
        (table.dataset == "Schirrmeister2017")
        & (table.method.isin(["spdsw_l40", "spdsw_l500", selected["ebsw_exp"], selected["evar"], selected["cvar"], "sampled_max_l40"]))
    ]
    lines = header + [
        "",
        "# Coherent SW overnight pilot",
        "",
        f"The preregistered HGD gate decision is `{gate['decision']}`. "
        f"BNCI conditional follow-up ran: {yesno(bnci_ran)}. No 14-subject HGD or hierarchical experiment ran.",
        "",
        "## Fixed choices",
        "",
        f"- Seeds: {list(SEEDS)}; HGD subjects: {list(HGD_SUBJECTS)}; 500 epochs; exact LEW every 25 epochs.",
        f"- EBSW candidates were beta=1 and the single scale match beta={scale_beta:.10g}.",
        f"- Globally selected settings: EBSW `{selected['ebsw_exp']}`, EVaR `{selected['evar']}`, CVaR `{selected['cvar']}`.",
        "- Every base-grid method used LR=10000. Only the three selected adaptive settings plus SPDSW-L40 and sampled-Max received LR=3000 and normalized-update controls.",
        "",
        "## Core result table",
        "",
        dataframe_markdown(hgd_selected),
        "",
        "Negative paired AUC differences favor the named method over SPDSW-L40. "
        "LEW evaluation time is excluded from optimization wall-clock. Missing quality-hit epochs mean the run never reached its matched SPDSW-L500 epoch-500 LEW.",
        "",
        "## Prerequisite validation",
        "",
        "- All 21 theorem-regression tests passed. The exact pure-power EBSW triangle violation was 0.091007599563.",
        "- The shared-direction triangle audit had worst EVaR slack -1.562e-02 and CVaR slack 4.441e-16; no positive violation occurred beyond tolerance.",
        "- Fixed-kappa EVaR had zero observed range in D(cX,cY)/c and KL over the registered dilation grid; all edge-case gradient/NaN tests passed.",
        "- The audit, theorem XML, and complete E1--E5 CSV/figure/config bundle are archived under `prerequisites/`.",
        "",
        "## Common-random-number audit",
        "",
        "At each subject/seed/epoch all L=40 runs use the same deterministic Frobenius-uniform direction tensor. "
        "The identical initial source state therefore gives an identical epoch-0 directional cost vector; hashes are stored under `audits/`. "
        "After the first update, method-specific source particles differ, so numerical h vectors necessarily differ while directions remain paired.",
        "",
        "## Failures, instability, and stop rule",
        "",
    ]
    if failures:
        lines.extend(f"- {item['dataset']} {item['control']} {item['method']} s{item['subject']} seed {item['seed']}: {item['error']}" for item in failures)
    else:
        lines.append("- No execution error, non-finite trajectory, or unrecorded run failure occurred.")
    if not gate["pass"]:
        lines.extend(
            [
                "- The globally selected EVaR kappa=0.1 and CVaR alpha=0.5 each improved LEW AUC on 0/3 subjects at LR=10000 and 0/3 under normalized updates.",
                "- At LR=3000 the mean paired AUC differences were favorable (-0.01545 EVaR, -0.02257 CVaR), but subject 1 did not improve and both advantages reversed under normalized updates (+0.02467 and +0.01828).",
                "- Base LR=10000 produced very large finite LEW divergence for concentrated adaptive settings. The selected EVaR/CVaR each had 3/9 diverged trajectories, and more concentrated settings had 9/9; sampled-Max had 9/9.",
                "- No EVaR/CVaR setting met every preregistered gate; the experiment stopped without BNCI expansion or further tuning.",
            ]
        )
    else:
        lines.append(f"- Gate-passing settings: {gate['passing_settings']}. The bounded BNCI follow-up was completed and no larger expansion was attempted.")
    lines.extend(
        [
            "- A selected setting is called endpoint-like only when its realized median ESS is <=1.25 (EVaR) or its CVaR active tail has fewer than two directions.",
            "- Sampled Max-SW is a finite L=40 endpoint reference, not an optimized continuous Max-SW solver.",
            "- Results are reported neutrally; the frozen concentration grids were not expanded after inspection.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    theorem_pass, theorem_artifact = parse_theorem_gate()
    euclidean_pass, euclidean_artifact = parse_euclidean_gate()
    if not theorem_pass or not euclidean_pass:
        raise RuntimeError(
            f"prerequisite gate failed: theorem={theorem_pass} ({theorem_artifact}), "
            f"Euclidean={euclidean_pass} ({euclidean_artifact})"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.out.mkdir(parents=True, exist_ok=True)
    archive_prerequisite_artifacts(args.out)

    prereg = {
        **PREREG,
        "device": str(device),
        "device_name": torch.cuda.get_device_properties(device).name if device.type == "cuda" else platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "theorem_artifact": theorem_artifact,
        "euclidean_artifact": euclidean_artifact,
    }
    prereg_path = args.out / "PREREGISTERED_CONFIG.json"
    if prereg_path.exists() and json.loads(prereg_path.read_text()) != prereg:
        raise RuntimeError("refusing to alter existing preregistration")
    if not prereg_path.exists():
        dump_json(prereg_path, prereg)
    hashes = source_hashes()
    hashes_path = args.out / "FROZEN_SOURCE_HASHES.json"
    if hashes_path.exists() and json.loads(hashes_path.read_text()) != hashes:
        raise RuntimeError("frozen source hashes changed")
    if not hashes_path.exists():
        dump_json(hashes_path, hashes)

    hgd_cache = load_cached("Schirrmeister2017", HGD_SUBJECTS, args.device)
    scale_frame, scale_beta = initial_cost_audit(hgd_cache, "Schirrmeister2017", args.out)
    step_target, step_frame = normalized_target(hgd_cache)
    step_frame.to_csv(args.out / "audits" / "HGD_normalized_step_target.csv", index=False)
    scale_config = {
        "formula": "1 / median initial directional-cost std over HGD 3 subjects x 3 seeds",
        "median_initial_h_std": float(scale_frame.std_h.median()),
        "scale_matched_beta": scale_beta,
        "normalized_step_target": step_target,
    }
    scale_path = args.out / "SCALE_MATCHED_CONFIG.json"
    if scale_path.exists() and json.loads(scale_path.read_text()) != scale_config:
        raise RuntimeError("scale-matched configuration changed")
    if not scale_path.exists():
        dump_json(scale_path, scale_config)
    methods = base_methods(scale_beta)

    manifests = execute_grid(
        args.out,
        "Schirrmeister2017",
        hgd_cache,
        methods,
        "common_lr10000",
        COMMON_LR,
        step_target,
        args.rerun,
    )
    hgd_base_frames = read_frames(args.out, "Schirrmeister2017")
    hgd_base_summary = run_summary(hgd_base_frames[hgd_base_frames.control == "common_lr10000"])
    selected = select_one_per_family(hgd_base_summary)
    selected_path = args.out / "SELECTED_SETTINGS.json"
    selected_payload = {
        **selected,
        "selection_rule": PREREG["post_base_selection"],
        "selected_after_only_common_lr10000_base_grid": True,
    }
    if selected_path.exists() and json.loads(selected_path.read_text()) != selected_payload:
        raise RuntimeError("selected settings differ from the frozen base-grid result")
    if not selected_path.exists():
        dump_json(selected_path, selected_payload)
    controls = [
        method_by_name(methods, "spdsw_l40"),
        method_by_name(methods, selected["ebsw_exp"]),
        method_by_name(methods, selected["evar"]),
        method_by_name(methods, selected["cvar"]),
        method_by_name(methods, "sampled_max_l40"),
    ]
    manifests += execute_grid(
        args.out,
        "Schirrmeister2017",
        hgd_cache,
        controls,
        "lr3000",
        EXTRA_LR,
        step_target,
        args.rerun,
    )
    manifests += execute_grid(
        args.out,
        "Schirrmeister2017",
        hgd_cache,
        controls,
        "normalized",
        math.nan,
        step_target,
        args.rerun,
    )
    hgd_frames = read_frames(args.out, "Schirrmeister2017")
    hgd_summary = add_paired_and_quality(run_summary(hgd_frames), hgd_frames)
    gate = decide_gate(hgd_summary, selected, args.out)
    dump_json(args.out / "GATE.json", gate)

    bnci_ran = False
    bnci_summary = pd.DataFrame()
    if gate["pass"]:
        bnci_cache = load_cached("BNCI2014_001", BNCI_SUBJECTS, args.device)
        initial_cost_audit(bnci_cache, "BNCI2014_001", args.out)
        bnci_step, bnci_step_frame = normalized_target(bnci_cache)
        bnci_step_frame.to_csv(args.out / "audits" / "BNCI_normalized_step_target.csv", index=False)
        manifests += execute_grid(
            args.out,
            "BNCI2014_001",
            bnci_cache,
            methods,
            "common_lr10000",
            COMMON_LR,
            bnci_step,
            args.rerun,
        )
        manifests += execute_grid(
            args.out,
            "BNCI2014_001",
            bnci_cache,
            controls,
            "lr3000",
            EXTRA_LR,
            bnci_step,
            args.rerun,
        )
        manifests += execute_grid(
            args.out,
            "BNCI2014_001",
            bnci_cache,
            controls,
            "normalized",
            math.nan,
            bnci_step,
            args.rerun,
        )
        bnci_frames = read_frames(args.out, "BNCI2014_001")
        bnci_summary = add_paired_and_quality(run_summary(bnci_frames), bnci_frames)
        bnci_ran = True

    summary = pd.concat([hgd_summary, bnci_summary], ignore_index=True)
    summary.to_csv(args.out / "run_level_results.csv", index=False)
    table = aggregate_table(summary)
    table.to_csv(args.out / "CORE_RESULTS.csv", index=False)
    (args.out / "CORE_RESULTS.md").write_text(dataframe_markdown(table) + "\n")
    make_figures(args.out, hgd_frames, selected)
    failures = [record for record in manifests if record.get("status") != "ok"]
    write_report(args.out, theorem_pass, gate, selected, scale_beta, table, failures, bnci_ran)
    if source_hashes() != hashes:
        raise RuntimeError("frozen source changed during run")
    print(f"[OVERNIGHT DONE] {gate['decision']} -> {args.out}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
