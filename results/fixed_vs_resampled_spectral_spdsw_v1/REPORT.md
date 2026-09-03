- theorem/regression tests: PASS
- development fixed-vs-resampled result: RESAMPLED
- development spectral-under-fixed result: IMPROVE
- development spectral-under-resampled result: WORSE
- fixed-spectral wall-clock advantage: NO
- held-out HGD run: NO
- proceed to hierarchy after this experiment: NO

# Fixed versus resampled Spectral SPDSW v1

Registered decision: `stop_after_development_gate`. Development classification: `C_resampling_remains_superior`.
Previous direct and hierarchical null results remain frozen and are not reinterpreted.

## 1. Exact protocol

- HGD cached 0train -> 1test log-SPD blocks; development subjects 2, 3, 4; seeds 6398, 3654, 1788; 500 epochs; exact LEW every 25 epochs.
- The 2x2 factorial uses direct k=40 directions only and sigma=0.5 only. Primary updates use the frozen direct-SPDSW step norm 7.3473173386245945; secondary raw SGD uses LR=3000 for every method.
- Fixed uniform and spectral runs share one physical fixed-bank tensor and cached target projection per subject/seed/control. Resampled pairs use the same deterministic epoch bank.
- All tensors are float64; AMP, autocast, TF32, clipping, early stopping, method-specific LR, and hierarchy are absent.
- Cumulative optimization time includes fixed one-time bank/target setup and excludes exact-LEW evaluation.

### L500 compatibility decision

No prior result matched the complete development protocol. The prior spectral-development L500 runs cover only seed 6398, 100 epochs, and step norm 3.8661. The overnight L500 runs cover subjects 1, 7, 14 under raw LR=10000. Therefore both new controls were rerun for subjects 2, 3, 4 and all three seeds.

## 2. Normalized-update 2x2 table

| method | mean_relative_lew_auc | std_relative_lew_auc | mean_final_lew | mean_lew_reduction_pct | mean_optimization_ms | divergence_count | nan_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_lns_k40_s0p5 | 0.99910961 | 0.00021409627 | 84.335416 | 0.082548241 | 1422.7599 | 0 | 0 |
| fixed_spdsw_k40 | 0.99917544 | 0.00022545611 | 84.35136 | 0.063348832 | 1392.4008 | 0 | 0 |
| resampled_lns_k40_s0p5 | 0.87263906 | 0.016885197 | 68.66449 | 18.641521 | 1602.0231 | 0 | 0 |
| resampled_spdsw_k40 | 0.8553141 | 0.018029839 | 67.473169 | 20.040963 | 1604.7113 | 0 | 0 |

Secondary raw-SGD LR=3000 2x2 table (diagnostic only):

| method | mean_relative_lew_auc | std_relative_lew_auc | mean_final_lew | mean_lew_reduction_pct | mean_optimization_ms | divergence_count | nan_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_lns_k40_s0p5 | 0.99891194 | 0.0002146569 | 84.311264 | 0.1116182 | 1354.0174 | 0 | 0 |
| fixed_spdsw_k40 | 0.99891175 | 0.00021454552 | 84.311253 | 0.11163 | 1321.3143 | 0 | 0 |
| resampled_lns_k40_s0p5 | 0.91808458 | 0.011252096 | 72.403646 | 14.211225 | 1545.5002 | 0 | 0 |
| resampled_spdsw_k40 | 0.93505347 | 0.0093914613 | 74.509668 | 11.709345 | 1535.8547 | 0 | 0 |

The normalized mean resampling effect under uniform weighting was `-0.14386133`. The fixed spectral effect was favorable but only `-0.00006582`, whereas the resampled spectral effect was adverse at `+0.01732495`. Under raw SGD, the resampled spectral effect reversed sign to `-0.01696889`; this secondary reversal does not override the primary normalized-update conclusion.

## 3. Subject-wise factorial differences

All differences are right-minus-left relative-LEW AUC; negative is favorable to resampling or spectral weighting according to the column definition.

| subject | delta_resample_uniform | delta_spectral_fixed | delta_spectral_resampled | interaction |
| --- | --- | --- | --- | --- |
| 2 | -0.16216105 | -4.6413078e-05 | 0.018027558 | 0.018073971 |
| 3 | -0.14784392 | -6.5776237e-05 | 0.018329155 | 0.018394931 |
| 4 | -0.12157903 | -8.5282874e-05 | 0.015618152 | 0.015703435 |

## 4–5. Epoch and wall-clock curves

See `fig_lew_vs_epoch.png`, `fig_lew_vs_wallclock.png`, and `fig_lew_vs_projection_count.png`. Projection count is source-plus-target directional projections: fixed uses 40 cached target projections once plus 40 source projections per epoch; resampled uses 40 source and 40 target projections per epoch.

Fixed-spectral time-to-resampled-quality diagnostic:

| comparator | favorable_runs | reached_runs | mean_fixed_reach_ms | mean_comparator_epoch500_ms |
| --- | --- | --- | --- | --- |
| resampled_lns_k40_s0p5 | 0 | 0 |  | 1602.0231 |
| resampled_spdsw_k40 | 0 | 0 |  | 1604.7113 |

Paired L500 epoch-500 quality reach summary:

| control | method | l500_quality_reached_runs | mean_epoch_reach_l500_quality | mean_wall_ms_reach_l500_quality |
| --- | --- | --- | --- | --- |
| normalized_update | fixed_lns_k40_s0p5 | 0 |  |  |
| normalized_update | fixed_spdsw_k40 | 0 |  |  |
| normalized_update | resampled_lns_k40_s0p5 | 0 |  |  |
| normalized_update | resampled_spdsw_k40 | 0 |  |  |
| raw_sgd_lr3000 | fixed_lns_k40_s0p5 | 0 |  |  |
| raw_sgd_lr3000 | fixed_spdsw_k40 | 0 |  |  |
| raw_sgd_lr3000 | resampled_lns_k40_s0p5 | 9 | 411.11111 | 1272.8203 |
| raw_sgd_lr3000 | resampled_spdsw_k40 | 0 |  |  |

## 6. Timing decomposition

| method | direction_sampling_ms | source_projection_ms | target_projection_ms | wasserstein_1d_ms | sorting_aggregation_ms | backward_ms | optimizer_update_ms | total_optimization_epoch_ms | one_time_fixed_bank_sampling_ms | one_time_fixed_target_projection_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_lns_k40_s0p5 | 0 | 0.86542381 | 0 | 0.4963505 | 0.12027795 | 0.93042793 | 0.40409503 | 2.8165752 | 9.1191821 | 5.3531016 |
| fixed_spdsw_k40 | 0 | 0.86432342 | 0 | 0.5123498 | 0.047464145 | 0.92672976 | 0.40498983 | 2.755857 | 9.1191821 | 5.3531016 |
| resampled_lns_k40_s0p5 | 0.18705748 | 0.85810063 | 0.2040165 | 0.49944677 | 0.12255188 | 0.92717139 | 0.40570157 | 3.2040462 | 0 | 0 |
| resampled_spdsw_k40 | 0.18698348 | 0.8743078 | 0.20652408 | 0.50133086 | 0.044629863 | 0.98853772 | 0.40710877 | 3.2094226 | 0 | 0 |
| resampled_spdsw_l500_reference | 0.80605005 | 5.9554171 | 1.2618301 | 0.51658464 | 0.045405492 | 6.3321512 | 0.40720842 | 15.324647 | 0 | 0 |

## 7. Fixed-bank rank dynamics

| method | consecutive_spearman | top5_overlap | ever_top5_fraction | median_effective_directions |
| --- | --- | --- | --- | --- |
| fixed_lns_k40_s0p5 | 0.98333649 | 0.99674905 | 0.30833333 | 31.364933 |
| fixed_spdsw_k40 | 0.99562807 | 0.99403251 | 0.38055556 | 40 |

Rank persistence is descriptive and did not tune sigma. Resampled banks have no asserted cross-epoch identity. See `fig_rank_persistence.png` and the additional descriptive `fig_rank_trajectories.png`.

## 8. Fixed-bank overfitting diagnostic

Training-loss reduction uses the first pre-update training loss (epoch 1) as baseline; LEW reduction uses epoch 0 to epoch 500.

| method | mean_training_loss_reduction_pct | mean_lew_reduction_pct | mean_train_minus_lew_reduction_pct |
| --- | --- | --- | --- |
| fixed_lns_k40_s0p5 | 73.667228 | 0.082548241 | 73.58468 |
| fixed_spdsw_k40 | 94.227368 | 0.063348832 | 94.164019 |

## 9. Gate decision

The development gate FAILED: `stop_after_development_gate`.
The favorable fixed-spectral AUC difference was `-0.00006582` over 3/3 subject means, but it was far too small to make fixed spectral competitive with either resampled k=40 method and yielded no registered wall-clock quality advantage.

```json
{
  "fixed_spectral_improves_at_least_2_subjects": true,
  "fixed_spectral_mean_auc_difference_favorable": true,
  "fixed_spectral_wall_clock_advantage": false,
  "no_material_instability_increase": true,
  "no_posthoc_hyperparameter_change": true,
  "normalized_update_norms_matched": true
}
```

## 10. Nulls, failures, and scope

- Execution failure logs: 0. Every log, if any, remains under `logs/`.
- Completed trajectories: 90/90; manifest failures: 0; non-development trajectories: 0.
- The development sample is only three subjects; effect sizes and subject-wise paired differences are reported without significance claims.
- Raw-SGD outcomes are retained in CORE_RESULTS.csv and SUBJECT_RESULTS.csv but do not determine the primary conclusion.
- Missing L500-quality reach times mean the k=40 method never reached its paired reference's epoch-500 LEW.
- CLAIM_LEDGER.md separates implementation facts, finite-run statements, empirical findings, and prohibited claims.

## Commands and provenance

```bash
nvidia-smi -i 3
PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m experiments.run_fixed_vs_resampled_spectral_spdsw --phase prepare
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m pytest -q --junitxml=results/fixed_vs_resampled_spectral_spdsw_v1/TEST_RESULTS.xml
PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_fixed_vs_resampled_spectral_spdsw --phase development_normalized
PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_fixed_vs_resampled_spectral_spdsw --phase development_raw
PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_fixed_vs_resampled_spectral_spdsw --phase analyze
```

- Branch: `exp/fixed-vs-resampled-spectral-spdsw-v1`; analysis invocation commit: `87ff8034ae04f80891f1d98d852ecdab323e43e3`.
- Python 3.10.19, PyTorch 2.11.0+cu130, CUDA runtime 13.0, physical GPU 3 (UUID-verified PyTorch cuda:1).
- Full regression result before scientific runs: 84 passed; TEST_RESULTS.xml contains the machine-readable result.
