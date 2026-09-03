- theorem regression: PASS
- synthetic hierarchy interaction: YES
- HGD normalized-update improvement: NOT RUN
- hierarchy-specific interaction: NOT RUN
- reaches SPDSW-L500 quality: NOT RUN
- proceed to full expansion: NO

# Lognormal-spectral hierarchical SPDSW pilot

Current registered decision: `stop_after_development_null`.
The completed direct-pilot decision `stop_after_hgd_null` remains unchanged and is not reinterpreted.

## Fixed method and controls

- Spectral SPDHSW uses lognormal-quantile spectral weighting over freshly resampled normalized mixtures.
- dtype is torch.float64; AMP, autocast, and TF32 are disabled; no clipping or outcome-triggered early stopping is used.
- Physical GPU 3 is the only GPU authorized. No persistent/evolving bank is used.

## Phase status

- Synthetic gate: PASS; positive all-dimension sigmas=[0.5, 1.0, 1.25, 1.5]; condition-robust sigmas=[0.5, 1.0, 1.25, 1.5].
- Development gate: FAIL; selected sigma=0.5, improved subjects=0/3; mean paired relative-LEW AUC difference=+0.00354608 (lower is better).

## Development gate result

The mechanism did not survive the registered HGD development selection. Every nonzero sigma was worse than sigma=0 in mean relative exact-LEW AUC, and none improved 2 of 3 development subjects. Consequently Phase C held-out HGD, the raw-SGD control, Phase D matched-concentration baselines, BNCI transfer, and full expansion were not run.

The common normalized step target was `3.8661274383`, derived exactly as preregistered from the median initial normalized-SPDHSW update norm at raw LR 10000.

| sigma | mean_relative_lew_auc | mean_uniform_hierarchy_auc | mean_paired_auc_difference | improved_subjects | divergence_count | selected |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.96625723 | 0.96271116 | 0.0035460764 | 0 | 0 | True |
| 1 | 0.96963203 | 0.96271116 | 0.0069208734 | 0 | 0 | False |
| 1.25 | 0.97137614 | 0.96271116 | 0.0086649797 | 0 | 0 | False |
| 1.5 | 0.97324184 | 0.96271116 | 0.010530686 | 0 | 0 | False |

Per-subject outcomes for the selected development comparison and references:

| method | subject | lew_initial | lew_final | relative_lew_auc | gap_closure_100 |
| --- | --- | --- | --- | --- | --- |
| direct_lns_l40_s0p5 | 2 | 92.15362 | 86.202383 | 0.96691708 | 6.4579524 |
| direct_lns_l40_s0p5 | 3 | 71.129925 | 66.436874 | 0.9657397 | 6.5978583 |
| direct_lns_l40_s0p5 | 4 | 89.933719 | 85.220162 | 0.97293577 | 5.2411448 |
| lns_spdhsw_k40_l500_s0p5 | 2 | 92.15362 | 85.819108 | 0.96469686 | 6.8738615 |
| lns_spdhsw_k40_l500_s0p5 | 3 | 71.129925 | 66.132619 | 0.963507 | 7.0256034 |
| lns_spdhsw_k40_l500_s0p5 | 4 | 89.933719 | 84.79927 | 0.97056784 | 5.7091474 |
| normalized_spdhsw_k40_l500 | 2 | 92.15362 | 85.269438 | 0.96154916 | 7.4703325 |
| normalized_spdhsw_k40_l500 | 3 | 71.129925 | 65.510612 | 0.9588755 | 7.9000689 |
| normalized_spdhsw_k40_l500 | 4 | 89.933719 | 84.297042 | 0.9677088 | 6.2675906 |
| spdsw_l40 | 2 | 92.15362 | 85.272361 | 0.96150078 | 7.4671613 |
| spdsw_l40 | 3 | 71.129925 | 65.584316 | 0.95923972 | 7.7964506 |
| spdsw_l40 | 4 | 89.933719 | 84.495664 | 0.96877846 | 6.0467358 |
| spdsw_l500 | 2 | 92.15362 | 71.052363 | 0.86889948 | 22.897914 |
| spdsw_l500 | 3 | 71.129925 | 54.869492 | 0.86550445 | 22.860187 |
| spdsw_l500 | 4 | 89.933719 | 73.456519 | 0.89415343 | 18.321493 |

## Run integrity

- Development run CSVs: 33/33; rows per run: [101].
- Exact LEW epochs: [0, 25, 50, 75, 100].
- Divergences: 0; NaN rows: 0.
- Fresh deterministic epoch seeds were shared across methods; the CRN audit is in `development/COMMON_RANDOM_NUMBERS_AUDIT.json`.
- All run CSVs, including negative outcomes, are retained. No clipping, early stopping, preprocessing change, or post-hoc sigma expansion was used.

## Failures and negative results

- The first synthetic invocation stopped before producing scientific draw records because CUDA `ndtri` required an unavailable NVRTC builtins library. The failure log is retained as `logs/synthetic_attempt1.log`. The deterministic spectrum construction was moved to float64 CPU and copied to physical GPU 3; a GPU smoke regression was then added and all 63 tests passed.
- The synthetic mechanism gate passed, but this did not predict an HGD optimization gain. The HGD development gate failed cleanly with no divergence or nonfinite values.
- Because the registered gate failed, no held-out or transfer result exists; `runs/NOT_RUN.md` records that deliberate stop.

## Figures and tables

- `fig_spectrum_weights.png` and `fig_synthetic_capture.png` summarize Phase A.
- `fig_lew_vs_epoch.png`, `fig_lew_vs_wallclock.png`, and `fig_lew_vs_ambient_projections.png` show development-only trajectories.
- `fig_interaction.png` is explicitly a development diagnostic, not the unrun held-out primary statistic.
- `fig_gradient_update_stability.png` and `fig_ess_entropy.png` show optimization and concentration controls.
- `CORE_RESULTS.csv` is development-only because the held-out gate was never reached.

## Exact commands and environment

```bash
nvidia-smi -i 3
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m pytest -q --junitxml=results/lognormal_spectral_spdhsw_v1/TEST_RESULTS.xml
PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_logspectral_spdhsw --phase synthetic
PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_logspectral_spdhsw --phase development
PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_logspectral_spdhsw --phase finalize
```

- Python 3.10.19, PyTorch 2.11.0+cu130, CUDA runtime 13.0.
- Host: affctiv; branch: exp/lognormal-spectral-spdhsw-v1.
- Starting direct-pilot commit: 4edf5dda470c5e525c5feb274462414751348b4b; audit/tests checkpoint: 39bc01f2aec9e9cd1b5d145319c53d601cc9fd86; synthetic checkpoint: dfc7645eee08ef714a0589ec638ad4ca6f18b30c.
- Development/null checkpoint: 488672c609b04caceef5cac760412c1025429c3d.
- Device: physical GPU 3, NVIDIA RTX 6000 Ada Generation.

## Scope and claims

See `CLAIM_LEDGER.md`. Negative gates stop expansion; no prohibited claim is inferred from a finite-bank outcome.
