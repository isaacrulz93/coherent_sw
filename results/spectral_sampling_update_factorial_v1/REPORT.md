- regression tests: PASS
- N_proj: 500
- sigma: 1.0
- normalized eta_norm: 2.7936838980935
- raw-power LR: 3000
- raw-rooted eta_root: 589.107249530589
- fixed normalized spectral effect: IMPROVE
- fixed raw-power spectral effect: IMPROVE
- fixed raw-rooted spectral effect: WORSE
- resampled normalized spectral effect: WORSE
- resampled raw-power spectral effect: IMPROVE
- resampled raw-rooted spectral effect: WORSE
- fastest method to 20% LEW reduction: resampled_uniform_normalized_power
- fastest method to 30% LEW reduction: resampled_uniform_normalized_power
- divergence/NaN trajectories: 0

# Spectral sampling/update factorial audit

## 1. Exact protocol

BNCI2014_001 subjects 1, 3, and 8 were run with seeds 6398, 3654, and 1788 for 500 updates. Every condition used direct SPDSW with `N_proj=500`, `p=2`, and no hierarchy. The complete registered 2 sampling × 2 aggregation × 3 update factorial contains exactly 108 trajectories. Independent exact Log-Euclidean Wasserstein (LEW) was evaluated at every state epoch 0 through 500 and its time was excluded from optimization wall-clock.

Fixed conditions shared one persistent bank and its cached target projections per subject/seed. Resampled conditions shared one deterministic epoch bank sequence. Fixed and resampled conditions used the same epoch-0 bank and only separated when the resampled policy advanced to its next bank.

The rooted objective used `sqrt(F)` with epsilon exactly `0.0`. The normalized step was the frozen audited BNCI value from `results/high_support_fixed_vs_resampled_spdsw_v2/BNCI_NORMALIZED_STEP.json (frozen audited BNCI value)`. Raw power used LR 3000. The one rooted step size was calibrated from the epoch-0 uniform baseline before comparative training.

## 2. 12-condition factorial table

| Sampling | Update | Uniform AUC | Spectral AUC | Delta | 20% threshold speedup (epochs) | 20% paired n | 20% hits U/S | 30% threshold speedup (epochs) | 30% paired n | 30% hits U/S | Paired-final speedup (epochs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed | normalized_power | 0.796267 | 0.795908 | -0.000359582 | -2 | 6 | 6/6 | NA | 0 | 0/0 | -8.11111 |
| Fixed | raw_power | 0.798037 | 0.795147 | -0.00288979 | 14.3333 | 6 | 6/6 | NA | 0 | 0/0 | 73.5556 |
| Fixed | raw_rooted | 0.797734 | 0.800787 | 0.00305272 | 2.33333 | 6 | 6/6 | NA | 0 | 0/0 | NA |
| Resampled | normalized_power | 0.661391 | 0.697512 | 0.0361203 | -6.33333 | 9 | 9/9 | -39.1667 | 6 | 9/6 | NA |
| Resampled | raw_power | 0.744015 | 0.724404 | -0.0196114 | 31.4444 | 9 | 9/9 | 132 | 5 | 5/6 | 211.889 |
| Resampled | raw_rooted | 0.677245 | 0.699342 | 0.0220975 | -0.666667 | 9 | 9/9 | -18.1667 | 6 | 9/6 | NA |

Aggregate results:

| sampling | aggregation | update | relative_lew_auc | final_lew | lew_reduction_pct | optimization_ms | divergence_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | spectral | normalized_power | 0.795908 | 7.21116 | 20.5466 | 1017.26 | 0 |
| fixed | spectral | raw_power | 0.795147 | 7.20492 | 20.6157 | 955.065 | 0 |
| fixed | spectral | raw_rooted | 0.800787 | 7.26202 | 19.9711 | 963.505 | 0 |
| fixed | uniform | normalized_power | 0.796267 | 7.21274 | 20.5334 | 975.825 | 0 |
| fixed | uniform | raw_power | 0.798037 | 7.20843 | 20.5769 | 907.872 | 0 |
| fixed | uniform | raw_rooted | 0.797734 | 7.22462 | 20.401 | 926.889 | 0 |
| resampled | spectral | normalized_power | 0.697512 | 5.83711 | 35.5192 | 1230.97 | 0 |
| resampled | spectral | raw_power | 0.724404 | 6.21811 | 31.5106 | 1171.63 | 0 |
| resampled | spectral | raw_rooted | 0.699342 | 5.91304 | 34.6649 | 1190.25 | 0 |
| resampled | uniform | normalized_power | 0.661391 | 5.23089 | 42.1971 | 1176.75 | 0 |
| resampled | uniform | raw_power | 0.744015 | 6.47723 | 28.6823 | 1121.63 | 0 |
| resampled | uniform | raw_rooted | 0.677245 | 5.45224 | 39.9358 | 1146.91 | 0 |

Negative Delta means spectral has lower (better) relative-LEW AUC. “Threshold speedup” is uniform first-hit epoch minus spectral first-hit epoch; positive favors spectral. Missing thresholds remain NA and are not extrapolated.

## 3. LEW-vs-epoch

All 501 independently evaluated states per trajectory are retained in the per-run CSVs. Relative-LEW AUC is the trapezoidal AUC over exactly epochs 0…500 divided by 500. Fixed-bank results are not interpreted alone as population behavior.

## 4. Threshold-reaching epochs

The preregistered thresholds were relative LEW 0.95, 0.90, 0.80, 0.70, and 0.60. `THRESHOLD_RESULTS.csv` records the first actually observed hit, with no interpolation. The fastest average method at 20% reduction was **resampled_uniform_normalized_power**; at 30% reduction it was **resampled_uniform_normalized_power**.

The means below are over reached runs only, while `reached_runs/total_runs` exposes censoring explicitly. “Fastest” above requires 9/9 reaches, preventing an incomplete method from winning by NA exclusion.

| sampling | aggregation | update | threshold_relative_lew | reached_runs | total_runs | mean_first_reach_epoch | mean_first_reach_optimization_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | spectral | normalized_power | 0.7 | 0 | 9 | NA | NA |
| fixed | spectral | normalized_power | 0.8 | 6 | 9 | 10.3333 | 21.5973 |
| fixed | spectral | raw_power | 0.7 | 0 | 9 | NA | NA |
| fixed | spectral | raw_power | 0.8 | 6 | 9 | 11.3333 | 21.8363 |
| fixed | spectral | raw_rooted | 0.7 | 0 | 9 | NA | NA |
| fixed | spectral | raw_rooted | 0.8 | 6 | 9 | 8.33333 | 15.7648 |
| fixed | uniform | normalized_power | 0.7 | 0 | 9 | NA | NA |
| fixed | uniform | normalized_power | 0.8 | 6 | 9 | 8.33333 | 16.5651 |
| fixed | uniform | raw_power | 0.7 | 0 | 9 | NA | NA |
| fixed | uniform | raw_power | 0.8 | 6 | 9 | 25.6667 | 47.8976 |
| fixed | uniform | raw_rooted | 0.7 | 0 | 9 | NA | NA |
| fixed | uniform | raw_rooted | 0.8 | 6 | 9 | 10.6667 | 19.9802 |
| resampled | spectral | normalized_power | 0.7 | 6 | 9 | 108.333 | 270.432 |
| resampled | spectral | normalized_power | 0.8 | 9 | 9 | 22.6667 | 54.2512 |
| resampled | spectral | raw_power | 0.7 | 6 | 9 | 210 | 490.396 |
| resampled | spectral | raw_power | 0.8 | 9 | 9 | 38.4444 | 88.7257 |
| resampled | spectral | raw_rooted | 0.7 | 6 | 9 | 108.333 | 256.129 |
| resampled | spectral | raw_rooted | 0.8 | 9 | 9 | 20.2222 | 48.3634 |
| resampled | uniform | normalized_power | 0.7 | 9 | 9 | 132.222 | 313.13 |
| resampled | uniform | normalized_power | 0.8 | 9 | 9 | 16.3333 | 36.5126 |
| resampled | uniform | raw_power | 0.7 | 5 | 9 | 329.8 | 734.776 |
| resampled | uniform | raw_power | 0.8 | 9 | 9 | 69.8889 | 155.945 |
| resampled | uniform | raw_rooted | 0.7 | 9 | 9 | 162.111 | 368.246 |
| resampled | uniform | raw_rooted | 0.8 | 9 | 9 | 19.5556 | 43.7562 |

## 5. Threshold-reaching wall-clock

Optimization wall-clock includes fixed one-time bank sampling and target projection setup. It excludes all independent LEW evaluation and paired-gradient diagnostic time. `THRESHOLD_RESULTS.csv` supplies both first-hit epoch and cumulative optimization time. `TIMING.csv` separately reports evaluation overhead and each requested optimization component.

| sampling | aggregation | update | mean_cumulative_optimization_ms | mean_cumulative_evaluation_ms | mean_direction_sampling_ms | mean_source_projection_ms | mean_target_projection_ms | mean_wasserstein_1d_ms | mean_sorting_aggregation_ms | mean_backward_ms | mean_optimizer_update_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | spectral | normalized_power | 1017.26 | 5575.14 | 0 | 123.353 | 0 | 325.713 | 76.2626 | 379.602 | 98.9324 |
| fixed | spectral | raw_power | 955.065 | 5595.53 | 0 | 123.453 | 0 | 327.818 | 76.5927 | 379.438 | 34.3616 |
| fixed | spectral | raw_rooted | 963.505 | 5594.93 | 0 | 123.149 | 0 | 326.833 | 76.396 | 389.174 | 34.5505 |
| fixed | uniform | normalized_power | 975.825 | 5631.41 | 0 | 123.577 | 0 | 336.105 | 29.9109 | 372.906 | 99.9231 |
| fixed | uniform | raw_power | 907.872 | 5702.71 | 0 | 124.245 | 0 | 334.287 | 28.473 | 372.5 | 34.9647 |
| fixed | uniform | raw_rooted | 926.889 | 5628.69 | 0 | 123.645 | 0 | 331.148 | 28.2469 | 395.534 | 34.9123 |
| resampled | spectral | normalized_power | 1230.97 | 5132.63 | 130.173 | 114.509 | 117.843 | 323.247 | 77.5224 | 368.211 | 99.4671 |
| resampled | spectral | raw_power | 1171.63 | 5301.4 | 131.814 | 114.708 | 118.033 | 326.446 | 78.2201 | 367.287 | 35.1223 |
| resampled | spectral | raw_rooted | 1190.25 | 5052.97 | 129.385 | 114.467 | 117.471 | 321.679 | 76.8031 | 395.505 | 34.9403 |
| resampled | uniform | normalized_power | 1176.75 | 4976.36 | 129.465 | 114.372 | 117.34 | 322.512 | 28.1752 | 365.979 | 98.9077 |
| resampled | uniform | raw_power | 1121.63 | 5335.33 | 130.556 | 114.53 | 118.104 | 323.601 | 28.2889 | 371.548 | 34.9973 |
| resampled | uniform | raw_rooted | 1146.91 | 5083.67 | 130.887 | 114.578 | 117.843 | 325.374 | 28.4811 | 394.391 | 35.3596 |

## 6. Paired-uniform-quality reach

`PAIRED_QUALITY_REACH.csv` uses each sampling/update/subject/seed uniform epoch-500 LEW as the paired target. Positive spectral speedup means spectral reached that quality earlier. The optional best-uniform-final rows are clearly marked descriptive and were not used for primary claims.

## 7. Gradient/update magnitudes

The sigma-1 finite-weight concentration was fixed throughout all spectral runs:

| spectral_effective_N | spectral_entropy | spectral_max_weight | spectral_top5_mass | spectral_top10_mass |
| --- | --- | --- | --- | --- |
| 188.241 | 5.71625 | 0.0301795 | 0.0923622 | 0.145999 |

| sampling | update | epoch | uniform_spectral_cosine |
| --- | --- | --- | --- |
| fixed | normalized_power | 0 | 0.87058 |
| fixed | normalized_power | 25 | 0.914734 |
| fixed | normalized_power | 50 | 0.932056 |
| fixed | normalized_power | 100 | 0.933744 |
| fixed | normalized_power | 200 | 0.934325 |
| fixed | normalized_power | 300 | 0.934758 |
| fixed | normalized_power | 400 | 0.934997 |
| fixed | normalized_power | 500 | 0.935108 |
| fixed | raw_power | 0 | 0.87058 |
| fixed | raw_power | 25 | 0.587284 |
| fixed | raw_power | 50 | 0.520232 |
| fixed | raw_power | 100 | 0.441473 |
| fixed | raw_power | 200 | 0.211056 |
| fixed | raw_power | 300 | 0.128369 |
| fixed | raw_power | 400 | 0.0977944 |
| fixed | raw_power | 500 | 0.078518 |
| fixed | raw_rooted | 0 | 0.87058 |
| fixed | raw_rooted | 25 | 0.913826 |
| fixed | raw_rooted | 50 | 0.937922 |
| fixed | raw_rooted | 100 | 0.938227 |
| fixed | raw_rooted | 200 | 0.938172 |
| fixed | raw_rooted | 300 | 0.938264 |
| fixed | raw_rooted | 400 | 0.93831 |
| fixed | raw_rooted | 500 | 0.938308 |
| resampled | normalized_power | 0 | 0.87058 |
| resampled | normalized_power | 25 | 0.641192 |
| resampled | normalized_power | 50 | 0.63768 |
| resampled | normalized_power | 100 | 0.63765 |
| resampled | normalized_power | 200 | 0.636929 |
| resampled | normalized_power | 300 | 0.634964 |
| resampled | normalized_power | 400 | 0.634308 |
| resampled | normalized_power | 500 | 0.637002 |
| resampled | raw_power | 0 | 0.87058 |
| resampled | raw_power | 25 | 0.680409 |
| resampled | raw_power | 50 | 0.640455 |
| resampled | raw_power | 100 | 0.638607 |
| resampled | raw_power | 200 | 0.635033 |
| resampled | raw_power | 300 | 0.638107 |
| resampled | raw_power | 400 | 0.638006 |
| resampled | raw_power | 500 | 0.637942 |
| resampled | raw_rooted | 0 | 0.87058 |
| resampled | raw_rooted | 25 | 0.643688 |
| resampled | raw_rooted | 50 | 0.638515 |
| resampled | raw_rooted | 100 | 0.637222 |
| resampled | raw_rooted | 200 | 0.637428 |
| resampled | raw_rooted | 300 | 0.639874 |
| resampled | raw_rooted | 400 | 0.636331 |
| resampled | raw_rooted | 500 | 0.639317 |

Raw gradient and applied update norms were recorded for every update. Same-state, same-bank uniform/spectral gradient cosines were computed at state epochs 0, 25, 50, 100, 200, 300, 400, and 500 without mutating training state and without charging diagnostic work to optimization time.

## 8. Fixed-bank overfitting

| aggregation | update | training_loss_reduction_pct | independent_LEW_reduction_pct | overfit_gap |
| --- | --- | --- | --- | --- |
| spectral | normalized_power | 98.2421 | 20.5466 | 77.6955 |
| spectral | raw_power | 99.9563 | 20.6157 | 79.3406 |
| spectral | raw_rooted | 87.6108 | 19.9711 | 67.6396 |
| uniform | normalized_power | 98.5651 | 20.5334 | 78.0317 |
| uniform | raw_power | 99.8898 | 20.5769 | 79.3129 |
| uniform | raw_rooted | 97.9043 | 20.401 | 77.5034 |

The fixed-bank objective value attached to each post-update epoch row is the pre-update objective that generated that update; epoch 0 contains the same initial-state objective. The overfit diagnostic therefore compares the registered training objective trajectory with independent LEW at every available state and should be read with that one-update logging convention in mind.

## 9. Sampling interactions

`SUBJECT_RESULTS.csv` contains run-level, seed-mean, subject-mean, and grand-mean spectral and sampling effects, spectral × sampling interactions, pairwise spectral × update interactions, and pairwise sampling × update interactions.

| effect | sampling | update | aggregation | mean | median | sd |
| --- | --- | --- | --- | --- | --- | --- |
| spectral | fixed | normalized_power | spectral-minus-uniform | -0.000359582 | -0.000229505 | 0.000382866 |
| spectral | fixed | raw_power | spectral-minus-uniform | -0.00288979 | -0.00275553 | 0.000618039 |
| spectral | fixed | raw_rooted | spectral-minus-uniform | 0.00305272 | 0.00235559 | 0.00160898 |
| spectral | resampled | normalized_power | spectral-minus-uniform | 0.0361203 | 0.0390341 | 0.00618389 |
| spectral | resampled | raw_power | spectral-minus-uniform | -0.0196114 | -0.0212618 | 0.00347166 |
| spectral | resampled | raw_rooted | spectral-minus-uniform | 0.0220975 | 0.0203847 | 0.00674401 |
| resampling | resampled-minus-fixed | normalized_power | uniform | -0.134876 | -0.143013 | 0.0222313 |
| resampling | resampled-minus-fixed | raw_power | uniform | -0.0540218 | -0.0564641 | 0.00882855 |
| resampling | resampled-minus-fixed | raw_rooted | uniform | -0.12049 | -0.12934 | 0.0183749 |
| resampling | resampled-minus-fixed | normalized_power | spectral | -0.0983963 | -0.102268 | 0.01631 |
| resampling | resampled-minus-fixed | raw_power | spectral | -0.0707433 | -0.0760461 | 0.0121846 |
| resampling | resampled-minus-fixed | raw_rooted | spectral | -0.101445 | -0.108909 | 0.0178915 |
| spectral_x_sampling | interaction | normalized_power | interaction | 0.0364799 | 0.040055 | 0.00639297 |
| spectral_x_sampling | interaction | raw_power | interaction | -0.0167216 | -0.0186992 | 0.00364211 |
| spectral_x_sampling | interaction | raw_rooted | interaction | 0.0190448 | 0.0174908 | 0.00532729 |
| spectral_x_update | fixed | raw_power-minus-normalized_power | spectral-effect contrast | -0.00253021 | -0.00234173 | 0.000677992 |
| spectral_x_update | fixed | raw_rooted-minus-normalized_power | spectral-effect contrast | 0.0034123 | 0.00309292 | 0.00152714 |
| spectral_x_update | fixed | raw_rooted-minus-raw_power | spectral-effect contrast | 0.00594251 | 0.00560773 | 0.00114583 |
| spectral_x_update | resampled | raw_power-minus-normalized_power | spectral-effect contrast | -0.0557316 | -0.0613458 | 0.00945531 |
| spectral_x_update | resampled | raw_rooted-minus-normalized_power | spectral-effect contrast | -0.0140228 | -0.0103952 | 0.00810075 |
| spectral_x_update | resampled | raw_rooted-minus-raw_power | spectral-effect contrast | 0.0417089 | 0.0386325 | 0.00792017 |
| sampling_x_update | resampling-effect contrast | raw_power-minus-normalized_power | uniform | 0.0808544 | 0.0839559 | 0.0139615 |
| sampling_x_update | resampling-effect contrast | raw_rooted-minus-normalized_power | uniform | 0.0143864 | 0.0113508 | 0.0065771 |
| sampling_x_update | resampling-effect contrast | raw_rooted-minus-raw_power | uniform | -0.066468 | -0.072781 | 0.00983654 |
| sampling_x_update | resampling-effect contrast | raw_power-minus-normalized_power | spectral | 0.0276529 | 0.0249186 | 0.00598417 |
| sampling_x_update | resampling-effect contrast | raw_rooted-minus-normalized_power | spectral | -0.00304866 | -0.00282879 | 0.00286272 |
| sampling_x_update | resampling-effect contrast | raw_rooted-minus-raw_power | spectral | -0.0307016 | -0.0281186 | 0.00662609 |

With three subjects these are descriptive paired effects, not significance tests.

## 10. Interpretation

Spectral weighting changes both gradient direction and magnitude. Normalized results isolate unit-gradient direction, while the two raw formulations retain the complete vector field. A raw spectral gain is not automatically an LR artifact, and a normalized null does not automatically disprove spectral utility.

“EBSW-style” here means only optimizing the rooted distance with its raw gradient under a common outer-flow philosophy. Conventional exponential IS-EBSW differentiates through energy-dependent weights and therefore includes a weight-response gradient term. The lognormal rank weights here are detached and piecewise fixed: within an ordering region `grad F_spec = sum_i w_i grad h_(i)` and there is no `grad(w_i)` term. Rooted spectral is therefore not algebraically identical to exponential EBSW.

Sigma 1 was preregistered, not optimized. No learning-rate sweep was performed. Thresholds not reached are not extrapolated, and fixed-bank outcomes are not treated as population SPDSW evidence.

Numerically, the registered pattern is Case B together with Case F. Spectral improved raw-power AUC within fixed support (Delta = -0.002890) and especially under resampling (Delta = -0.019611), but worsened raw-rooted AUC within fixed (Delta = +0.003053) and resampled (Delta = +0.022098) conditions. Under resampling it also worsened normalized AUC (Delta = +0.036120). Thus the raw-power gain is sensitive to the power-versus-root objective scaling and does not provide robust spectral-geometry evidence; an LR-equivalence audit would be required before further development. The raw-rooted failure in both sampling regimes means the weak spectral result cannot be attributed merely to normalized updates or merely to having optimized the p-th power instead of the rooted distance.

Resampling was substantially better than fixed support across every registered aggregation/update cell. Fixed training-objective reductions of 87.6–100.0% accompanied only about 20.0–20.6% independent-LEW reductions, leaving mean overfit gaps of 67.6–79.3 percentage points. This is an alignment diagnostic, not a downstream classification claim.
