# Repository audit (2026-09-03)

This project is an additive experiment tree.  No file in the existing SPDSW,
SPDHSW, EBSPDSW, or rebuttal trees is modified.

## Located components

- Direct SPDSW Frobenius-uniform directions: `/home/pikachu/edubridge_SPDHSW/spdsw/spdsw.py:68-100`.
- Direct log-SPD projection and 1-D inverse-CDF Wasserstein: the same file,
  lines 143-216.  The per-direction audited wrapper is
  `/home/pikachu/EBSPDSW/evobank/ot1d.py:20-22`.
- Frobenius-isometric `svec`: `/home/pikachu/EBSPDSW/evobank/svec.py:26-61`.
  It uses upper-triangular entries with a `sqrt(2)` multiplier off diagonal.
- Frobenius-uniform sampler and normalized hierarchical effective direction:
  `/home/pikachu/EBSPDSW/evobank/bank.py:68-95` and lines 207-232.
- Cross-session loaders and caches: `/home/pikachu/EBSPDSW/evobank/data.py`.
  The HGD cache uses the existing first-level to second-level split
  (`0train -> 1test`); BNCI2014-001 uses session 1 -> session 2.
- Cached log coordinates: `/home/pikachu/EBSPDSW/results/pilot_hgd/data_cache/`.
  HGD subjects 1-14 and BNCI subjects 1-9 were present at audit time.
- Log-space Euclidean SGD: `/home/pikachu/EBSPDSW/evobank/baselines.py:31-143`
  and `/home/pikachu/EBSPDSW/evobank/trainer.py:106-269`.
- Independent exact-OT LEW evaluator:
  `/home/pikachu/EBSPDSW/evobank/lew.py:29-76`.
- Frozen HGD protocol: 128 channels, subjects 1-14, seeds
  `(6398, 3654, 1788, 8515, 264)`, learning rate 10000, 500 epochs, and LEW
  every 25 epochs.  The present pilot preregisters subjects `(1, 7, 14)` and
  seeds `(6398, 3654, 1788)`.

## Frozen scope

The complete `/home/pikachu/SPDSW` rebuttal tree, `/home/pikachu/edubridge_SPDHSW`
implementation/experiments/results/caches, and the existing `/home/pikachu/EBSPDSW`
code/results are treated as read-only.  This project writes only below
`/home/pikachu/coherent_sw`.
