# Coherent sliced-Wasserstein controlled experiments

Status: PASS (mode=full, 16.12 s).

## Validation and notable outcomes

- The analytic p=gamma=1 pure-power EBSW counterexample was reproduced: leg=1.917991608712, direct=3.926990816987, triangle violation=0.091007599563.
- In the common-direction E1 audit, worst EVaR slack was -1.562e-02 and worst CVaR slack was 4.441e-16.
- The independent-direction control is descriptive only; its maximum observed slack was 4.594e-03. No theorem claim is attached to it.
- EVaR scale audit: range of D(cX,cY)/c=0.000e+00; fixed-beta EBSW range=3.214e-01.
- E4 compares every estimator only with its own L=20000 reference (16 repeats); SW is never used as the target for EVaR/CVaR.
- E5 normalized-update target norm was 0.0219529; diverged records=0.

## Negative results and failures

- No numerical or execution failures were observed in E1--E5.
- Fixed-beta and pure-power EBSW are baselines, not asserted metrics; positive triangle slacks are retained in the CSV.
- The sampled maximum is a finite-bank reference, not a sphere-optimized Max-SW implementation.
- E4 high-direction references remain finite Monte Carlo references, not exact population values.

## Artifacts

All CSVs, figures, and the frozen configuration are in this directory. The MOABB pilot has a separate output tree and is gated on this report and the theorem tests.
