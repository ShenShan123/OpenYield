# Yield estimation in OpenYield V2.2.4

The current compiler can generate reproducible, independent per-device process samples. The legacy importance-sampling implementations in `model_lib/` are research code and are **not a validated V2.2.4 yield flow**. They still depend on machine-local paths, optional ML packages absent from `environment.yml`, and the older custom-variation interface. The old root debug driver was removed because it did not provide a working current-release tutorial.

## Generate current process samples

Run from the repository root with the OpenYield environment and Xyce available:

```bash
python -m sram_compiler.per_device_mc.run \
  --rows 8 --cols 4 --operation read \
  --real-cell-mode 0 --variation-mode per-device \
  --mc-runs 100 --seed 3 --run-xyce \
  --output-dir outputs/yield_study
```

Use mode 0 for full device coverage and record the corner, VDD, temperature, wire model, timing table, driver baseline, model cards, and seed with the results. The runner writes a summary and measurement data to a configuration-specific output directory. `--variation-mode nominal --mc-runs 1` provides a fixed-corner reference. See the [runner tutorial](../sram_compiler/per_device_mc/README.md) for sample and output semantics.

This command produces circuit results, not a qualified failure probability. A yield study must define failure using waveform-based read, write, retention, sensing, and recovery checks, count numerical non-completion separately, and establish coverage across PVT and mismatch. The [V2.2.4 screen](../docs/design/PHASED_CONTROL_V2_2_4.md) has not run. The [open items](../docs/README.md#open-items) describe the work needed to connect samples to release checks and validate the estimators.

## Legacy algorithm modules

| Module | Research method |
|---|---|
| `model_lib/MC.py` | Direct Monte Carlo baseline |
| `model_lib/MNIS.py` | Mean-shifted importance sampling |
| `model_lib/AIS.py` | Adaptive importance sampling |
| `model_lib/ACS.py` | Adaptive compressed sampling |
| `model_lib/HSCS.py` | High-dimensional sparse compressed sampling |

`tool/Distribution/` contains their distribution helpers. These modules remain available for porting and comparison, but no supported CLI currently runs them end to end. The import-time deletion of a machine-local simulation directory has been removed from `tool/delete.py`; callers still need to supply their own output directories.
