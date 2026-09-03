- theorem regression: PASS
- HGD same-L improvement: NO
- normalized-gradient improvement: NO
- proceed to hierarchical SPDHSW: NO

# Coherent SW overnight pilot

The preregistered HGD gate decision is `stop_after_hgd_null`. BNCI conditional follow-up ran: NO. No 14-subject HGD or hierarchical experiment ran.

## Fixed choices

- Seeds: [6398, 3654, 1788]; HGD subjects: [1, 7, 14]; 500 epochs; exact LEW every 25 epochs.
- EBSW candidates were beta=1 and the single scale match beta=485.6709795.
- Globally selected settings: EBSW `ebsw_exp_default_b1`, EVaR `evar_k0p1`, CVaR `cvar_a0p5`.
- Every base-grid method used LR=10000. Only the three selected adaptive settings plus SPDSW-L40 and sampled-Max received LR=3000 and normalized-update controls.

## Core result table

| dataset | control | method | family | L | lew_reduction_pct_100 | lew_reduction_pct_250 | lew_reduction_pct_500 | relative_lew_auc | paired_auc_diff_vs_spdsw_l40 | paired_lew500_diff_vs_spdsw_l40 | epoch_reach_spdsw_l500_final | wall_reach_spdsw_l500_final | optimization_seconds | aggregation_ms_per_epoch | divergence_count | nan_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Schirrmeister2017 | common_lr10000 | cvar_a0p5 | cvar | 40 | -13.334795 | -223.38262 | -12320.349 | 16.533164 | 15.674283 | 12291.068 |  |  | 1.3475806 | 0.33750686 | 3 | 0 |
| Schirrmeister2017 | common_lr10000 | ebsw_exp_default_b1 | ebsw_exp | 40 | 7.2287625 | 15.312706 | 23.742535 | 0.85859731 | -0.00028351148 | -0.23178068 |  |  | 1.336541 | 0.26223802 | 3 | 0 |
| Schirrmeister2017 | common_lr10000 | evar_k0p1 | evar | 40 | -21.662537 | -513.00745 | -59117.363 | 65.438947 | 64.580066 | 58844.616 |  |  | 2.10745 | 1.7600725 | 3 | 0 |
| Schirrmeister2017 | common_lr10000 | sampled_max_l40 | sampled_max | 40 | -8.989111e+57 | -9.2685496e+142 | -1.6102525e+287 | 4.0256312e+283 | 4.0256312e+283 | 1.601754e+287 |  |  | 1.2837732 | 0.1459497 | 9 | 0 |
| Schirrmeister2017 | common_lr10000 | spdsw_l40 | sw | 40 | 7.1888126 | 15.284704 | 23.701762 | 0.85888082 | 0 | 0 |  |  | 1.2481324 | 0.18000597 | 3 | 0 |
| Schirrmeister2017 | common_lr10000 | spdsw_l500 | sw | 500 | 12.978861 | 22.846314 | 31.12537 | 0.7931147 | -0.065766114 | -22.101493 | 500 | 6.5093608 | 6.5093608 | 0.23452755 | 0 | 0 |
| Schirrmeister2017 | lr3000 | cvar_a0p5 | cvar | 40 | 5.7556854 | 12.008399 | 18.688163 | 0.88864919 | -0.022570263 | -12.098367 |  |  | 1.3444061 | 0.33847885 | 0 | 0 |
| Schirrmeister2017 | lr3000 | ebsw_exp_default_b1 | ebsw_exp | 40 | 4.5244732 | 9.8625295 | 16.211652 | 0.9071826 | -0.0040368467 | -2.6497905 |  |  | 1.3427101 | 0.26255266 | 0 | 0 |
| Schirrmeister2017 | lr3000 | evar_k0p1 | evar | 40 | 5.2180775 | 11.188831 | 17.816632 | 0.89576805 | -0.015451397 | -9.1150618 |  |  | 2.141549 | 1.7870978 | 0 | 0 |
| Schirrmeister2017 | lr3000 | sampled_max_l40 | sampled_max | 40 | -4.5924404e+09 | -8.2901971e+22 | -6.1605857e+44 | 1.5629082e+41 | 1.5629082e+41 | 6.128072e+44 |  |  | 1.2929146 | 0.14570288 | 9 | 0 |
| Schirrmeister2017 | lr3000 | spdsw_l40 | sw | 40 | 4.2629461 | 9.393293 | 15.664041 | 0.91121945 | 0 | 0 |  |  | 1.256896 | 0.18455494 | 0 | 0 |
| Schirrmeister2017 | normalized | cvar_a0p5 | cvar | 40 | 4.4292782 | 9.8388473 | 16.356125 | 0.90720232 | 0.018283839 | 6.6272443 |  |  | 1.3767787 | 0.33620428 | 3 | 0 |
| Schirrmeister2017 | normalized | ebsw_exp_default_b1 | ebsw_exp | 40 | 5.2626855 | 11.775685 | 19.255809 | 0.88967709 | 0.00075860441 | 0.53084916 |  |  | 1.3772105 | 0.25922828 | 1 | 0 |
| Schirrmeister2017 | normalized | evar_k0p1 | evar | 40 | 3.9797526 | 9.1183729 | 15.493573 | 0.91359292 | 0.024674433 | 9.8513122 |  |  | 2.1744125 | 1.7938296 | 3 | 0 |
| Schirrmeister2017 | normalized | sampled_max_l40 | sampled_max | 40 | -2.2628541 | -4.6042647 | -6.9548216 | 1.0424668 | 0.15354831 | 56.552123 |  |  | 1.324834 | 0.14299352 | 6 | 0 |
| Schirrmeister2017 | normalized | spdsw_l40 | sw | 40 | 5.3065418 | 11.861683 | 19.367787 | 0.88891848 | 0 | 0 |  |  | 1.3006493 | 0.18621513 | 1 | 0 |

Negative paired AUC differences favor the named method over SPDSW-L40. LEW evaluation time is excluded from optimization wall-clock. Missing quality-hit epochs mean the run never reached its matched SPDSW-L500 epoch-500 LEW.

## Prerequisite validation

- All 21 theorem-regression tests passed. The exact pure-power EBSW triangle violation was 0.091007599563.
- The shared-direction triangle audit had worst EVaR slack -1.562e-02 and CVaR slack 4.441e-16; no positive violation occurred beyond tolerance.
- Fixed-kappa EVaR had zero observed range in D(cX,cY)/c and KL over the registered dilation grid; all edge-case gradient/NaN tests passed.
- The audit, theorem XML, and complete E1--E5 CSV/figure/config bundle are archived under `prerequisites/`.

## Common-random-number audit

At each subject/seed/epoch all L=40 runs use the same deterministic Frobenius-uniform direction tensor. The identical initial source state therefore gives an identical epoch-0 directional cost vector; hashes are stored under `audits/`. After the first update, method-specific source particles differ, so numerical h vectors necessarily differ while directions remain paired.

## Failures, instability, and stop rule

- No execution error, non-finite trajectory, or unrecorded run failure occurred.
- The globally selected EVaR kappa=0.1 and CVaR alpha=0.5 each improved LEW AUC on 0/3 subjects at LR=10000 and 0/3 under normalized updates.
- At LR=3000 the mean paired AUC differences were favorable (-0.01545 EVaR, -0.02257 CVaR), but subject 1 did not improve and both advantages reversed under normalized updates (+0.02467 and +0.01828).
- Base LR=10000 produced very large finite LEW divergence for concentrated adaptive settings. The selected EVaR/CVaR each had 3/9 diverged trajectories, and more concentrated settings had 9/9; sampled-Max had 9/9.
- No EVaR/CVaR setting met every preregistered gate; the experiment stopped without BNCI expansion or further tuning.
- A selected setting is called endpoint-like only when its realized median ESS is <=1.25 (EVaR) or its CVaR active tail has fewer than two directions.
- Sampled Max-SW is a finite L=40 endpoint reference, not an optimized continuous Max-SW solver.
- Results are reported neutrally; the frozen concentration grids were not expanded after inspection.
