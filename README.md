# Coherent sliced-Wasserstein pilot

Numerically stable PyTorch implementations and regression tests for SW,
exponential and pure-power EBSW, fixed-multiplier entropic aggregation,
KL-ball EVaR-SW, and density-capped CVaR-SW.

The repository also contains the preregistered Euclidean experiments and the
three-subject, three-seed direct-SPDSW HGD pilot. The pilot stopped at its
registered negative gate, so BNCI expansion and hierarchical SPDHSW were not
run.

## Key artifacts

- `AUDIT.md`: read-only source and protocol audit.
- `tests/test_coherent_slicing.py`: theorem and numerical regression suite.
- `results/coherent_sw_overnight/REPORT.md`: final pilot report.
- `results/coherent_sw_overnight/CORE_RESULTS.csv`: aggregate HGD results.
- `results/coherent_sw_overnight/runs/`: all 198 HGD run trajectories.

## Tests

```bash
python -m pip install -e .
python -m pytest -q
```

The MOABB entry point consumes the existing read-only log-SPD caches and helper
modules under `/home/pikachu/EBSPDSW`; those external cached datasets are not
vendored here. The complete generated result bundle is committed.
