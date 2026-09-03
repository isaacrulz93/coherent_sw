# Theorem and numerical regression

Status: **PASS**

Command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
  /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m pytest -q \
  --junitxml=results/lognormal_spectral_spdhsw_v1/TEST_RESULTS.xml
```

Final post-gate verification result: `63 passed` in 3.22 seconds. The suite ran on CPU
except for the explicit fixed-weight smoke test on physical GPU 3. The warnings are 73 upstream
TorchScript deprecations from imported dependencies and three pre-existing
pytest xUnit property-format warnings; there was no failed, skipped, or xfailed
test.

## Lognormal-spectrum checks

- `sigma=0` returns exactly uniform weights and uses `h.mean()` exactly.
- For every registered `(L, sigma)` pair, including `L=2000,sigma=1.5`, cell
  weights are finite, nonnegative, nondecreasing, and sum to one at float64
  tolerance.
- Closed-form cell weights agree with independent high-accuracy quadrature.
- Permutation invariance and positive homogeneity hold numerically.
- Shared-direction triangle regressions pass for the explicit empirical triple
  and random nonnegative component fields.
- Away from ties, autograd equals the assigned rank weights exactly. Ties use a
  deterministic stable ordering and return a finite valid subgradient.
- The sigma-zero hierarchical loss exactly equals the existing normalized
  SPDHSW mean for identical bank/mix draws.
- Cheap two-stage coordinates agree with projection onto explicitly
  materialized normalized effective directions.
- Assigned weights are invariant under positive rescaling of costs.
- No NaN or Inf occurs at the registered maximum `L=2000,sigma=1.5`.
- The registered maximum weights are finite and unit-mass after transfer to
  physical GPU 3; this guards the exact device path used by the experiments.

These are finite common-direction and implementation regressions. They do not
assert metricity for independently resampled realized direction banks or that a
nonzero spectral estimator estimates the same population quantity as uniform
SPDSW.
