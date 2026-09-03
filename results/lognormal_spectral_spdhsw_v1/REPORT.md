- theorem regression: PASS
- synthetic hierarchy interaction: YES
- HGD normalized-update improvement: NOT RUN
- hierarchy-specific interaction: NOT RUN
- reaches SPDSW-L500 quality: NOT RUN
- proceed to full expansion: NO

# Lognormal-spectral hierarchical SPDSW pilot

Current registered decision: `proceed_to_hgd_development`.
The completed direct-pilot decision `stop_after_hgd_null` remains unchanged and is not reinterpreted.

## Fixed method and controls

- Spectral SPDHSW uses lognormal-quantile spectral weighting over freshly resampled normalized mixtures.
- dtype is torch.float64; AMP, autocast, and TF32 are disabled; no clipping or outcome-triggered early stopping is used.
- Physical GPU 3 is the only GPU authorized. No persistent/evolving bank is used.

## Phase status

- Synthetic gate: PASS; positive all-dimension sigmas=[0.5, 1.0, 1.25, 1.5]; condition-robust sigmas=[0.5, 1.0, 1.25, 1.5].
- HGD development: NOT RUN.

## Exact commands and environment

```bash
nvidia-smi -i 3
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -m pytest -q --junitxml=results/lognormal_spectral_spdhsw_v1/TEST_RESULTS.xml
PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_logspectral_spdhsw --phase synthetic
PYTHONPATH=. /home/pikachu/miniconda3/envs/spdsw_hsw/bin/python -u -m experiments.run_logspectral_spdhsw --phase development
```

- Python 3.10.19, PyTorch 2.11.0+cu130, CUDA runtime 13.0.
- Host: affctiv; branch: exp/lognormal-spectral-spdhsw-v1.
- Audit/tests checkpoint: 39bc01f; runner invocation parent commit: 39bc01f2aec9e9cd1b5d145319c53d601cc9fd86.
- Device: physical GPU 3, NVIDIA RTX 6000 Ada Generation.

## Scope and claims

See `CLAIM_LEDGER.md`. Negative gates stop expansion; no prohibited claim is inferred from a finite-bank outcome.
