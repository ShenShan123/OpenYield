# Circuit-backed MOBO

`bayesian_optimizer.py` and the nearby modules implement the multi-objective Bayesian optimizer used by [`../demo_mobo.py`](../demo_mobo.py). Run the SRAM entry point from the repository root; the former generic ZDT1 test driver in this directory has been removed.

```bash
python size_optimization/demo_mobo.py --max_iter 50 --circuit_mode 1
```

`--max_iter` sets the number of optimization iterations. `--circuit_mode 1` evaluates the real circuit; `2` uses the approximate equivalent-cell path. The optimizer searches SNM, power, delay, and area and writes its Pareto CSV and logs under `size_optimization/experiment/MOBO/`. Circuit-backed evaluations require the optional optimizer packages and Xyce. See the [optimization guide](../README.md) for setup and the [equivalent-model guide](../../sram_compiler/equivalent_modeling/README.md) before comparing the two circuit modes.

For architecture plus transistor sizing, `python size_optimization/experiment.py` also offers MOBO as a joint search algorithm. The [offline OpenYield V2 package](../openyield_v2/README.md) is a separate surrogate workflow with its own data and budgets.
