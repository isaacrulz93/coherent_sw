# Claim ledger

## Established implementation facts

- Every primary method uses 40 direct Frobenius-unit ambient directions; no
  SPDHSW hierarchy or mixture direction is constructed.
- Fixed banks permit deterministic target-projection caching and persistent
  direction identities. Resampled banks use the frozen triangular epoch-seed
  sequence and have no cross-epoch identity interpretation.
- Lognormal-spectral weights are fixed at sigma 0.5 while their detached rank
  assignment changes with the observed directional costs.

## Finite realized statements

- Bank-hash equality, target-cache reuse, update-norm equality, timing, rank
  persistence, and LEW outcomes are finite-run audit or empirical statements.
- Fixed finite directions are not claimed to define a full metric.
- Independently resampled realized estimates are not claimed to obey a
  realization-wise triangle inequality.

## Empirical finding

- Registered development case: `C_resampling_remains_superior`.
- Spectral-under-fixed classification: `IMPROVE`;
  spectral-under-resampled classification: `WORSE`.
- These HGD alignment results measure exact LEW only and make no downstream
  classification claim.

## Unsupported or prohibited claims

- Nonzero spectral weighting is not the same population target as uniform
  SPDSW and is not described as an unbiased estimator of it.
- A fixed bank is not claimed to solve ambient directional coverage.
- Fixed spectral weighting is not claimed to replace resampling unless every
  registered development gate item passes.
- No hierarchy benefit is inferred because hierarchy is absent here.
