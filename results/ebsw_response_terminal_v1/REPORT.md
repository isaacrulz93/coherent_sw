- regression/audit tests: PASS
- completed BNCI trajectories: 288/288
- divergence trajectories: 26
- solver failures: 0
- beta_scale: 122.776923381967
- spectral rho_match: 0.376481475220964
- median beta=1 response ratio: 0.00013889927721305
- maximum calibrated response ratio: 3.19535454598008
- H-WR-ACTIVE: PASS
- H-WR-DIRECTION: FAIL
- H-WR-COMPLETE: FAIL
- H-WR-SCALE: FAIL
- H-MAG: PASS
- TERMINAL DECISION: KEEP
- Euclidean context: RUN

# Terminal EBSW response audit

## Protocol and implementation audit

This terminal experiment used BNCI2014_001 subjects 1, 3, and 8; seeds 6398, 3654, and 1788; 500 updates; direct resampled `N_proj=500` Frobenius directions; and independent exact LEW at epochs 0,25,…,500. Exactly 16 methods × 2 update formulations × 9 subject-seed cases were run. No HGD, hierarchy, fixed bank, downstream classifier, LR sweep, or raw-power trajectory was added.

FULL exponential EBSW differentiated `alpha=softmax(beta*h)` and therefore included `alpha_i[1+beta(h_i-F)]`. STOP used the identical alpha values and scalar objective but detached alpha. ESS beta was solved deterministically from `h.detach()` and treated as constant during objective differentiation. This is a conditional full EBSW gradient at calibrated beta, with stop-gradient through the beta calibration map.

The analytic FULL and STOP identities, coefficient sum, ESS endpoints and solver tolerance, exact rho match, q=1 equality with SW, copied-state immutability, common banks/h values, frozen step sizes, independent evaluator, absence of raw-power/hierarchy, and frozen prior hashes passed the full regression suite and runtime audits.

`divergence` uses the existing audited LEW evaluator definition: a trajectory is flagged if any evaluated LEW exceeds its epoch-0 LEW or is nonfinite. Thus a finite, completed trajectory can be counted as divergent even when its run status is `ok`.

## FULL versus STOP

| update | condition | mean_AUC_full | mean_AUC_stop | paired_delta | favorable_runs | total_runs | final_LEW_full | final_LEW_stop | divergence_full | divergence_stop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normalized | b1 | 0.6657453 | 0.66721 | -0.001464614 | 6 | 9 | 5.234565 | 5.247204 | 0 | 0 |
| normalized | bscale | 0.6716774 | 0.6677438 | 0.003933581 | 1 | 9 | 5.27242 | 5.227301 | 0 | 0 |
| normalized | ess025 | 0.7700365 | 0.7251134 | 0.04492307 | 0 | 9 | 6.781234 | 6.192337 | 0 | 0 |
| normalized | ess050 | 0.7426351 | 0.6980025 | 0.0446326 | 0 | 9 | 6.437864 | 5.789334 | 0 | 0 |
| normalized | ess075 | 0.7123846 | 0.6756887 | 0.03669586 | 0 | 9 | 6.013302 | 5.418342 | 0 | 0 |
| normalized | essmatch | 0.7548643 | 0.7102151 | 0.04464925 | 0 | 9 | 6.593575 | 5.973735 | 0 | 0 |
| raw_rooted | b1 | 0.6803956 | 0.6807967 | -0.0004010503 | 4 | 9 | 5.443502 | 5.455977 | 0 | 0 |
| raw_rooted | bscale | 4.094037 | 2.197277 | 1.89676 | 0 | 9 | 37.16466 | 22.14922 | 9 | 5 |
| raw_rooted | ess025 | 1.017344 | 0.7384156 | 0.2789287 | 0 | 9 | 8.975952 | 6.340616 | 9 | 0 |
| raw_rooted | ess050 | 0.8206827 | 0.6979637 | 0.122719 | 0 | 9 | 7.200759 | 5.807115 | 0 | 0 |
| raw_rooted | ess075 | 0.7363971 | 0.6756697 | 0.06072748 | 0 | 9 | 6.301843 | 5.383949 | 0 | 0 |
| raw_rooted | essmatch | 0.8882993 | 0.7157725 | 0.1725268 | 0 | 9 | 7.85753 | 6.102991 | 3 | 0 |

Negative paired delta favors FULL. Run counts retain all divergences and solver failures in the denominator.

## FULL versus SW

| update | condition | comparator | mean_AUC_full | mean_AUC_sw | paired_delta | favorable_runs | total_runs | final_LEW_full | final_LEW_sw | divergence_full | divergence_sw | solver_failures_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normalized | b1 | sw | 0.6657453 | 0.6650382 | 0.0007071435 | 3 | 9 | 5.234565 | 5.230888 | 0 | 0 | 0 |
| normalized | bscale | sw | 0.6716774 | 0.6650382 | 0.006639202 | 0 | 9 | 5.27242 | 5.230888 | 0 | 0 | 0 |
| normalized | ess025 | sw | 0.7700365 | 0.6650382 | 0.1049983 | 0 | 9 | 6.781234 | 5.230888 | 0 | 0 | 0 |
| normalized | ess050 | sw | 0.7426351 | 0.6650382 | 0.0775969 | 0 | 9 | 6.437864 | 5.230888 | 0 | 0 | 0 |
| normalized | ess075 | sw | 0.7123846 | 0.6650382 | 0.0473464 | 0 | 9 | 6.013302 | 5.230888 | 0 | 0 | 0 |
| normalized | essmatch | sw | 0.7548643 | 0.6650382 | 0.08982609 | 0 | 9 | 6.593575 | 5.230888 | 0 | 0 | 0 |
| raw_rooted | b1 | sw | 0.6803956 | 0.6808033 | -0.000407689 | 7 | 9 | 5.443502 | 5.452243 | 0 | 0 | 0 |
| raw_rooted | bscale | sw | 4.094037 | 0.6808033 | 3.413233 | 0 | 9 | 37.16466 | 5.452243 | 9 | 0 | 0 |
| raw_rooted | ess025 | sw | 1.017344 | 0.6808033 | 0.336541 | 0 | 9 | 8.975952 | 5.452243 | 9 | 0 | 0 |
| raw_rooted | ess050 | sw | 0.8206827 | 0.6808033 | 0.1398794 | 0 | 9 | 7.200759 | 5.452243 | 0 | 0 | 0 |
| raw_rooted | ess075 | sw | 0.7363971 | 0.6808033 | 0.05559384 | 0 | 9 | 6.301843 | 5.452243 | 0 | 0 | 0 |
| raw_rooted | essmatch | sw | 0.8882993 | 0.6808033 | 0.207496 | 0 | 9 | 7.85753 | 5.452243 | 3 | 0 | 0 |

## Magnitude-sensitive coherent controls

`lpwp_q2` and `lpwp_q4` are L^(p*q)-aggregations of the directional W_p field. They are not standard SW_pq because the inner directional distance remains W_p.

| update | method | mean_AUC | final_LEW | divergence |
| --- | --- | --- | --- | --- |
| normalized | lpwp_q2 | 0.6659319 | 5.235693 | 0 |
| normalized | lpwp_q4 | 0.7180772 | 6.077404 | 0 |
| normalized | spectral_s1 | 0.7010398 | 5.837113 | 0 |
| normalized | sw | 0.6650382 | 5.230888 | 0 |
| raw_rooted | lpwp_q2 | 0.6725794 | 5.310425 | 0 |
| raw_rooted | lpwp_q4 | 0.7961079 | 7.09556 | 0 |
| raw_rooted | spectral_s1 | 0.7033769 | 5.913042 | 0 |
| raw_rooted | sw | 0.6808033 | 5.452243 | 0 |

## Response and signed coefficients

| calibration | update | median_response_ratio | median_full_stop_cosine | median_full_vs_sw_cosine | median_stop_vs_sw_cosine |
| --- | --- | --- | --- | --- | --- |
| b1 | normalized | 0.0001332554 | 1 | 1 | 1 |
| b1 | raw_rooted | 0.0001401276 | 1 | 1 | 1 |
| bscale | normalized | 0.01716844 | 0.999876 | 0.9995002 | 0.9998754 |
| bscale | raw_rooted | 0.0002429596 | 1 | 0.2516881 | 0.2639092 |
| ess025 | normalized | 2.173539 | 0.9565615 | 0.279988 | 0.5136474 |
| ess025 | raw_rooted | 2.144671 | 0.9665404 | 0.5360184 | 0.7217985 |
| ess050 | normalized | 1.645256 | 0.9326102 | 0.3928096 | 0.6829649 |
| ess050 | raw_rooted | 1.618922 | 0.9590534 | 0.6355654 | 0.8213097 |
| ess075 | normalized | 1.000071 | 0.9278527 | 0.5851389 | 0.8371404 |
| ess075 | raw_rooted | 1.074806 | 0.9534022 | 0.7122388 | 0.8890785 |
| essmatch | normalized | 1.863227 | 0.9452222 | 0.341399 | 0.6091583 |
| essmatch | raw_rooted | 1.928949 | 0.9588156 | 0.5705103 | 0.7722162 |

| calibration | update | median_negative_fraction | median_negative_mass | median_positive_mass | median_coeff_l1 | max_coeff_sum_error |
| --- | --- | --- | --- | --- | --- | --- |
| b1 | normalized | 0 | -0 | 1 | 1 | 2.220446e-16 |
| b1 | raw_rooted | 0 | -0 | 1 | 1 | 2.220446e-16 |
| bscale | normalized | 0 | -0 | 1 | 1 | 3.108624e-15 |
| bscale | raw_rooted | 0.998 | 0.0001656242 | 1.000166 | 1.000331 | 4.951595e-14 |
| ess025 | normalized | 0.512 | 0.0764663 | 1.076466 | 1.152933 | 8.881784e-16 |
| ess025 | raw_rooted | 0.38 | 0.01380511 | 1.013805 | 1.02761 | 4.440892e-16 |
| ess050 | normalized | 0.124 | 0.005719947 | 1.00572 | 1.01144 | 4.440892e-16 |
| ess050 | raw_rooted | 0 | -0 | 1 | 1 | 2.220446e-16 |
| ess075 | normalized | 0 | -0 | 1 | 1 | 2.220446e-16 |
| ess075 | raw_rooted | 0 | -0 | 1 | 1 | 2.220446e-16 |
| essmatch | normalized | 0.338 | 0.03046813 | 1.030468 | 1.060936 | 8.881784e-16 |
| essmatch | raw_rooted | 0 | -0 | 1 | 1 | 2.220446e-16 |

The response coefficients may be signed; negative mass represents redistribution/cancellation, not simply positive emphasis.

## One-step magnitude-versus-direction decomposition

| calibration | update | mean_delta_full | mean_delta_stop | mean_delta_stop_normmatched | mean_full_minus_normmatched |
| --- | --- | --- | --- | --- | --- |
| b1 | normalized | -0.05249468 | -0.05185913 | -0.05249771 | 3.02827e-06 |
| b1 | raw_rooted | -0.04701126 | -0.04637585 | -0.04701416 | 2.899479e-06 |
| bscale | normalized | 0.7046783 | 0.1365256 | 0.6459448 | 0.05873349 |
| bscale | raw_rooted | 0.8378076 | -0.01921591 | 0.6341609 | 0.2036467 |
| ess025 | normalized | 0.5419519 | -0.07133796 | 0.3667165 | 0.1752353 |
| ess025 | raw_rooted | 0.01764169 | -0.5205206 | -0.5348823 | 0.552524 |
| ess050 | normalized | 0.05937324 | -0.07850547 | -0.04152586 | 0.1008991 |
| ess050 | raw_rooted | -0.0161091 | -0.2393259 | -0.2241062 | 0.2079971 |
| ess075 | normalized | -0.05707693 | -0.07092848 | -0.101583 | 0.04450611 |
| ess075 | raw_rooted | -0.06556745 | -0.1155499 | -0.131308 | 0.06574053 |
| essmatch | normalized | 0.1639065 | -0.08283777 | 0.03891889 | 0.1249876 |
| essmatch | raw_rooted | 0.02568861 | -0.3283484 | -0.3099914 | 0.33568 |

The registered subject-seed summary averages FULL-minus-norm-matched-STOP one-step LEW differences across eight states from each normalized FULL trajectory. Negative favors FULL direction beyond scalar norm matching.

## Concentration and beta calibration

`beta_scale=1/hbar` is a scale diagnostic, not an optimized beta. ESS conditions target rho 0.25, 0.50, 0.75, and exact sigma-1 rho match `0.376481475220964`. Solver errors and realized concentration are retained in `ESS_DIAGNOSTICS.csv`.

| method | family | update | mean_ESS_over_N | median_ESS_over_N | mean_entropy | median_entropy | mean_max_weight | median_max_weight | mean_top5_mass | median_top5_mass | mean_top10_mass | median_top10_mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ebsw_full_b1 | ebsw | normalized | 0.9999992 | 1 | 6.214608 | 6.214608 | 0.002002394 | 0.002000985 | 0.01000898 | 0.01000395 | 0.02001521 | 0.02000691 |
| ebsw_full_b1 | ebsw | raw_rooted | 0.9999992 | 1 | 6.214608 | 6.214608 | 0.002002442 | 0.002001015 | 0.01000917 | 0.0100041 | 0.02001553 | 0.02000712 |
| ebsw_full_bscale | ebsw | normalized | 0.9680208 | 0.9998343 | 6.119699 | 6.214526 | 0.01486379 | 0.002124733 | 0.02921677 | 0.01050282 | 0.04134805 | 0.02087637 |
| ebsw_full_bscale | ebsw | raw_rooted | 0.002260928 | 0.002 | 0.0698856 | 6.193309e-07 | 0.9736515 | 1 | 0.9991061 | 1 | 0.9992223 | 1 |
| ebsw_full_ess025 | ebsw | normalized | 0.25 | 0.25 | 5.68596 | 5.679219 | 0.05270331 | 0.05180254 | 0.1390554 | 0.1400043 | 0.1974534 | 0.2021109 |
| ebsw_full_ess025 | ebsw | raw_rooted | 0.25 | 0.25 | 5.75312 | 5.749778 | 0.05565277 | 0.05524737 | 0.1424196 | 0.1441565 | 0.1973153 | 0.2011268 |
| ebsw_full_ess050 | ebsw | normalized | 0.5 | 0.5 | 5.94851 | 5.943724 | 0.02610424 | 0.02520234 | 0.0788189 | 0.07891645 | 0.1210418 | 0.1219658 |
| ebsw_full_ess050 | ebsw | raw_rooted | 0.5 | 0.5 | 5.993042 | 5.989641 | 0.0296513 | 0.02911985 | 0.0842972 | 0.08466111 | 0.1245511 | 0.1266449 |
| ebsw_full_ess075 | ebsw | normalized | 0.75 | 0.75 | 6.098006 | 6.096416 | 0.01334488 | 0.01275962 | 0.04549435 | 0.04549263 | 0.07472279 | 0.07474761 |
| ebsw_full_ess075 | ebsw | raw_rooted | 0.75 | 0.75 | 6.116997 | 6.115295 | 0.01627582 | 0.01575624 | 0.0506814 | 0.05076812 | 0.07929759 | 0.07988572 |
| ebsw_full_essmatch | ebsw | normalized | 0.3764815 | 0.3764815 | 5.842482 | 5.836884 | 0.0358969 | 0.03490625 | 0.1025014 | 0.1027989 | 0.1521384 | 0.1547783 |
| ebsw_full_essmatch | ebsw | raw_rooted | 0.3764815 | 0.3764815 | 5.902508 | 5.899519 | 0.03966903 | 0.03911782 | 0.1071509 | 0.107985 | 0.1535428 | 0.1567188 |
| ebsw_stop_b1 | ebsw | normalized | 0.9999992 | 1 | 6.214608 | 6.214608 | 0.0020024 | 0.002000989 | 0.01000899 | 0.01000399 | 0.02001523 | 0.02000691 |
| ebsw_stop_b1 | ebsw | raw_rooted | 0.9999992 | 1 | 6.214608 | 6.214608 | 0.002002452 | 0.002001007 | 0.01000921 | 0.01000407 | 0.0200156 | 0.0200071 |
| ebsw_stop_bscale | ebsw | normalized | 0.9771913 | 0.999835 | 6.149518 | 6.214526 | 0.01083729 | 0.002124519 | 0.0233378 | 0.01050163 | 0.03498616 | 0.02087607 |
| ebsw_stop_bscale | ebsw | raw_rooted | 0.4406844 | 0.002397264 | 2.796918 | 0.3189026 | 0.5397991 | 0.9091614 | 0.5621309 | 1 | 0.5672245 | 1 |
| ebsw_stop_ess025 | ebsw | normalized | 0.25 | 0.25 | 5.659463 | 5.651185 | 0.05087668 | 0.04969589 | 0.1384243 | 0.1391161 | 0.198929 | 0.2033122 |
| ebsw_stop_ess025 | ebsw | raw_rooted | 0.25 | 0.25 | 5.742193 | 5.74054 | 0.05542462 | 0.05487032 | 0.1413737 | 0.1429563 | 0.1957569 | 0.1999338 |
| ebsw_stop_ess050 | ebsw | normalized | 0.5 | 0.5 | 5.94512 | 5.94077 | 0.02574299 | 0.02487365 | 0.07865542 | 0.07866854 | 0.1210312 | 0.1219584 |
| ebsw_stop_ess050 | ebsw | raw_rooted | 0.5 | 0.5 | 5.945321 | 5.941072 | 0.02580733 | 0.0248887 | 0.07858122 | 0.0786729 | 0.1209634 | 0.1217704 |
| ebsw_stop_ess075 | ebsw | normalized | 0.75 | 0.75 | 6.09745 | 6.095679 | 0.0132037 | 0.01255663 | 0.04533055 | 0.04530454 | 0.07458536 | 0.07462572 |
| ebsw_stop_ess075 | ebsw | raw_rooted | 0.75 | 0.75 | 6.097664 | 6.096043 | 0.01326454 | 0.01268313 | 0.04542133 | 0.0454419 | 0.07464912 | 0.07471688 |
| ebsw_stop_essmatch | ebsw | normalized | 0.3764815 | 0.3764815 | 5.835216 | 5.829073 | 0.03552462 | 0.03441544 | 0.101957 | 0.1021916 | 0.1518617 | 0.1538304 |
| ebsw_stop_essmatch | ebsw | raw_rooted | 0.3764815 | 0.3764815 | 5.834989 | 5.828492 | 0.03538726 | 0.03443615 | 0.1020436 | 0.1024874 | 0.1520961 | 0.1542595 |
| lpwp_q2 | lpwp | normalized | 0.827733 | 0.8381102 | 6.114884 | 6.124574 | 0.006560229 | 0.006182181 | 0.02809322 | 0.0269815 | 0.05140168 | 0.04962246 |
| lpwp_q2 | lpwp | raw_rooted | 0.8274402 | 0.8363033 | 6.115664 | 6.123802 | 0.006599149 | 0.006248989 | 0.02818606 | 0.02722719 | 0.05151943 | 0.04999659 |
| lpwp_q4 | lpwp | normalized | 0.2393801 | 0.2494485 | 5.36385 | 5.435858 | 0.04557238 | 0.03725551 | 0.1405585 | 0.1267509 | 0.2150041 | 0.1993934 |
| lpwp_q4 | lpwp | raw_rooted | 0.05076062 | 0.04867689 | 3.98642 | 4.009629 | 0.1430455 | 0.1250645 | 0.3748039 | 0.3669547 | 0.5146519 | 0.5127226 |
| spectral_s1 | spectral | normalized | 0.3764815 | 0.3764815 | 5.716248 | 5.716248 | 0.03017952 | 0.03017952 | 0.09236225 | 0.09236225 | 0.1459989 | 0.1459989 |
| spectral_s1 | spectral | raw_rooted | 0.3764815 | 0.3764815 | 5.716248 | 5.716248 | 0.03017952 | 0.03017952 | 0.09236225 | 0.09236225 | 0.1459989 | 0.1459989 |
| sw | sw | normalized | 1 | 1 | 6.214608 | 6.214608 | 0.002 | 0.002 | 0.01 | 0.01 | 0.02 | 0.02 |
| sw | sw | raw_rooted | 1 | 1 | 6.214608 | 6.214608 | 0.002 | 0.002 | 0.01 | 0.01 | 0.02 | 0.02 |

## Euclidean context

The optional context reused the frozen E5 two-dimensional Gaussian-mixture clouds, direction streams, seeds, epochs, evaluation bank, and common normalized-step construction. It was not part of the BNCI terminal gate and is not claimed as an exact reproduction of an EBSW paper experiment. Status: `run_frozen_E5_normalized_protocol`.

## Terminal hypotheses and decision

Passed hypotheses: H-WR-ACTIVE, H-MAG.

TERMINAL DECISION:
KEEP only the preregistered hypothesis/hypotheses identified below.

H-WR-ACTIVE passed because ess025 median response ratio=2.15479, ess050 median response ratio=1.62694, ess075 median response ratio=1.03224, essmatch median response ratio=1.88166. H-MAG passed because lpwp_q2: 9/9 favorable paired runs, mean AUC 0.67257944 vs SW 0.680803304, paired delta=-0.00822386378, divergences 0 vs SW 0. The practical KEEP decision is supported by H-MAG; H-WR-ACTIVE establishes that the response mechanism was successfully activated, not that it improved alignment.

This conclusion is deliberately scoped. It does not invalidate the EBSW triangle-inequality counterexamples, the coherent pair-independent metric construction, the mathematical validity of the spectral metric, or the fixed-bank overfitting findings. A CLOSE decision applies only to the use of directional Wasserstein magnitude as an informativeness signal to improve SPD/EEG distribution-alignment optimization; it is not a universal impossibility theorem about adaptive slicing, coherent metrics, EBSW, or other possible informativeness definitions.
