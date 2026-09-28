# SRAM sizing optimization

This directory contains two independent workflows: circuit-backed searches that call the current SRAM compiler and Xyce, and the offline `openyield_v2/` surrogate package. Run commands from the repository root. The release version remains V2.2.4; current array and clock limits are in the [compiler tutorial](../sram_compiler/README.md).

## Circuit-backed search

Install the OpenYield environment from `environment.yml`, confirm `Xyce` is on `PATH`, then install any optional packages required by the chosen method. `config_sram.yaml` and the circuit YAMLs define the parameter ranges; `exp_utils.py` connects them to the compiler. The driver sizes and timing lookup remain fixed across cell candidates and PVT samples.

| Method | Script |
|---|---|
| Simulated annealing, particle swarm, random | `demo_sa.py`, `demo_pso.py`, `demo_random.py` |
| Bayesian and surrogate methods | `demo_cbo.py`, `demo_tssbo.py`, `demo_cpn.py`, `demo_roseopt.py` |
| CMA-ES and SMAC | `demo_cmaes.py`, `demo_smac.py` |
| Multi-objective search | `demo_nsgaii.py`, `demo_moead.py`, `demo_mobo.py` |

A small entry-point example:

```bash
python size_optimization/demo_sa.py
```

`main_opt.py` at the repository root offers an interactive selector for the demo algorithms. `experiment.py` offers joint architecture and transistor sizing or sizing on one or all five fixed array configurations:

```bash
python size_optimization/experiment.py
```

The earlier standalone architecture-stage SMAC class was unused and has been removed. `experiment.py` still uses the historical `TwoStageOptimizer` class name internally; its current interactive choices are joint search and fixed-configuration sizing. Search outputs are written under the optimizer's experiment directories; keep generated runs out of Git.

The [detailed algorithm notes](电路算法说明文档.md) describe the parameter space and objective. The [MOBO](MOBO/README.md) and [NSGA-II](NSGA-II/Readme.md) pages show their direct SRAM entry points. Equivalent circuit evaluations are approximations; compare with a full-transistor mode 0 run before interpreting their metrics.

## Offline OpenYield V2 package

`openyield_v2/` reads bundled static 6T and 10T datasets. It does not call Xyce during optimization. Those datasets were collected under earlier cell and equivalent-model settings and are not current per-device yield evidence.

```bash
python -m pip install -r size_optimization/openyield_v2/requirements.txt
python -m size_optimization.openyield_v2.run_experiment --dry-run
python -m size_optimization.openyield_v2.run_experiment
```

See the [offline package guide](openyield_v2/README.md) for algorithm selection, budgets, and output files.
