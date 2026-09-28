# Circuit-backed NSGA-II

This directory provides the evolutionary search used by [`../demo_nsgaii.py`](../demo_nsgaii.py). Run the SRAM entry point from the repository root; the former generic ZDT1 test driver in this directory has been removed.

```bash
python size_optimization/demo_nsgaii.py --max_iter 50 --circuit_mode 1
```

`--max_iter` sets the generation limit. Circuit mode `1` uses the real circuit and `2` the approximate equivalent-cell path. The script evaluates SNM, power, delay, and area and writes results under `size_optimization/experiment/NSGA-II/`. It requires the optional optimizer dependencies and Xyce. For joint architecture and transistor sizing, select NSGA-II from `python size_optimization/experiment.py`.

See the [optimization guide](../README.md) for setup and the [equivalent-model guide](../../sram_compiler/equivalent_modeling/README.md) for the accuracy limits of mode `2`.
