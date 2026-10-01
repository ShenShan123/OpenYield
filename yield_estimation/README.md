# Yield estimation in OpenYield V2.2.6

The current compiler can generate reproducible, independent per-device process samples. The legacy importance-sampling implementations in `model_lib/` are research code and are **not a validated V2.2.5 yield flow**. They still depend on machine-local paths, optional ML packages absent from `environment.yml`, and the older custom-variation interface. The old root debug driver was removed because it did not provide a working current-release tutorial.

## Generate current process samples

From the repository root, set `ARRAY = [8, 4, False]`, `OPERATION = "read"`, `REAL_CELL_MODE = 0`, `MC_RUNS = 100`, `MC_SEED = 3`, and `RUN_XYCE = True` in `main_sram.py`. With the OpenYield environment and Xyce available, run:

```bash
python main_sram.py
```

Use mode 0 for full device coverage and record the corner, VDD, temperature, wire model, timing table, driver baseline, model cards, and seed with the results. The main entrance writes measurement data under `outputs/main_sram/`. Set `VARIATION_MODE = "nominal"` and `MC_RUNS = 1` for a fixed-corner diagnostic reference. See the [batch runner tutorial](../sram_compiler/per_device_mc/README.md) for sample and output semantics.

This command produces circuit results, not a qualified failure probability. A yield study must define failure using waveform-based read, write, retention, sensing, and recovery checks, count numerical non-completion separately, and establish coverage across PVT and mismatch. The [V2.2.5 screen](../docs/V2_2_5_FULL_COVERAGE_EVALUATION.md) has not completed. The [open items](../docs/README.md#open-items) describe the work needed to connect samples to release checks and validate the estimators.

## Legacy algorithm modules

| Module | Research method |
|---|---|
| `model_lib/MC.py` | Direct Monte Carlo baseline |
| `model_lib/MNIS.py` | Mean-shifted importance sampling |
| `model_lib/AIS.py` | Adaptive importance sampling |
| `model_lib/ACS.py` | Adaptive compressed sampling |
| `model_lib/HSCS.py` | High-dimensional sparse compressed sampling |

`tool/Distribution/` contains their distribution helpers. These modules remain available for porting and comparison, but no supported CLI currently runs them end to end. The import-time deletion of a machine-local simulation directory has been removed from `tool/delete.py`; callers still need to supply their own output directories.
