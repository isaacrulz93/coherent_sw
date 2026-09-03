# Read-only audit — lognormal-spectral hierarchical SPDSW v1

Audit date: 2026-09-03 (Asia/Seoul). Base commit before additive work:
`4edf5dda470c5e525c5feb274462414751348b4b`.

No API below is assumed from a name alone; the cited implementations were read
before the new experiment code was written.

## Current coherent-slicing repository

- Existing typed aggregators are in `coherent_slicing/aggregations.py`: result
  tuples at lines 16-27, validation at 30-40, SW at 43-46, exponential EBSW at
  49-64, pure-power EBSW at 67-98, entropic risk at 101-115, EVaR at 128-225,
  and CVaR at 228-249.
- The existing theorem/numerical suite is
  `tests/test_coherent_slicing.py:25-219` (21 tests before this experiment).
- The completed direct pilot entry point is
  `experiments/run_overnight.py`. Its initial shared-cost audit is at 179-219,
  and its subject/seed/method grid dispatch is at 247-335.
- Direct Frobenius directions and common random numbers are implemented in
  `experiments/run_moabb_pilot.py:194-215` and used in its epoch loop at
  388-398. The seed is `seed + epoch*(epoch+1)/2`; every same-L method gets the
  same direction tensor at a fixed subject/seed/epoch. The frozen direct result
  remains `stop_after_hgd_null` and is not reinterpreted here.

## External read-only implementation

- Frobenius-isometric symmetric vectorization is
  `/home/pikachu/EBSPDSW/evobank/svec.py:26-61`. It stores the upper triangle,
  multiplies off-diagonals by `sqrt(2)`, and supplies the exact inverse.
- Frobenius-uniform symmetric banks are sampled by
  `/home/pikachu/EBSPDSW/evobank/bank.py:68-83` as
  `(G+G.T)/||G+G.T||_F`. Unit mixing rows are sampled at lines 86-95.
- Fresh nonpersistent `r0` sampling is defined at
  `/home/pikachu/EBSPDSW/evobank/bank.py:157-173`: a generator seeded by
  `base_seed + step` draws the bank first and mixing rows second.
- Normalized effective directions are encoded by
  `/home/pikachu/EBSPDSW/evobank/bank.py:207-232`. In particular,
  `G=A A.T` and `sqrt(psi G psi.T)` are the effective Frobenius norms.
- The cheap normalized hierarchical projection is used at
  `/home/pikachu/EBSPDSW/evobank/trainer.py:196-214`: bottleneck projections,
  mixing, division by the effective-direction scale, then directional W2^2.
- Per-direction empirical one-dimensional W2^2 is
  `/home/pikachu/EBSPDSW/evobank/ot1d.py:20-22`; its audited inverse-CDF
  implementation is at 25-41.
- HGD cached log-SPD coordinates are rooted at
  `/home/pikachu/EBSPDSW/results/pilot_hgd/data_cache/Schirrmeister2017/`.
  All subjects 1-14 exist. Cache naming/loading is defined in
  `/home/pikachu/EBSPDSW/evobank/data.py:48-50,118-134`; preparation uses the
  first two split levels (`0train -> 1test`) at lines 78-105.
- Independent exact-OT LEW is
  `/home/pikachu/EBSPDSW/evobank/lew.py:29-76`: full squared Euclidean ground
  cost in log coordinates with uniform-marginal `ot.emd2`, and divergence when
  LEW is nonfinite or exceeds its initial value.
- The existing normalized `k=40,L=500` implementation defaults are
  `/home/pikachu/EBSPDSW/evobank/trainer.py:45-59`; the frozen HGD protocol is
  explicitly summarized in `/home/pikachu/EBSPDSW/evobank/make_tables.py:186-197`
  as d=128, m=8256, k=40, L=500, 500 epochs, seeds 6398/3654/1788, float64,
  log-space Euclidean SGD, raw LR 10000.
- `/home/pikachu/edubridge_SPDHSW/spdsw/spdhsw.py:49-96` is the original shared
  two-layer sampler, and lines 98-124 implement its SPDHSW loss. That version
  does **not** divide mixed projections by their effective Frobenius norms;
  therefore the normalized regression target for this task is the later
  EBSPDSW implementation cited above.

## Reproduction decision

The normalized hierarchical projection can be reproduced exactly: use the
same svec convention, single seeded generator, bank-then-mix draw order,
`G=A A.T`, and per-row `sqrt(psi G psi.T)` divisor. No explicit materialized
ambient mixture is required in the training path. Implementation may proceed.

## Frozen scope

`/home/pikachu/SPDSW`, `/home/pikachu/edubridge_SPDHSW`,
`/home/pikachu/EBSPDSW`, `experiments/run_overnight.py`, and all 239 files
already under `results/coherent_sw_overnight/` are read-only. The aggregate
SHA256 of the sorted per-file hash listing of the latter bundle is recorded in
`FROZEN_SOURCE_HASHES.json`.

