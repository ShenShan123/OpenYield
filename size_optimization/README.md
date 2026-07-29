# SRAM sizing optimization

This directory contains the original circuit-backed optimization scripts and the separate OpenYield V2 offline optimizer package.

## Circuit-backed scripts

The original scripts call the current OpenYield configuration and evaluation code through `exp_utils.py`.

| Method | Entry point |
|---|---|
| Simulated annealing | `demo_sa.py` |
| Particle swarm optimization | `demo_pso.py` |
| Constrained Bayesian optimization | `demo_cbo.py` |
| RoSE-Opt | `demo_roseopt.py` |
| CMA-ES | `demo_cmaes.py` |
| SMAC | `demo_smac.py` |
| NSGA-II | `demo_nsgaii.py` |
| MOEA/D | `demo_moead.py` |
| Multi-objective BO | `demo_mobo.py` |
| Random search | `demo_random.py` |

Run scripts from the repository root so the YAML and model-card paths resolve consistently:

```bash
python size_optimization/demo_sa.py
python size_optimization/demo_pso.py
python size_optimization/demo_cbo.py
```

`experiment.py` runs the existing two-stage architecture and transistor-sizing flow. Parameter ranges are read from `config_sram.yaml` and the circuit YAML files.

## OpenYield V2

`openyield_v2/` is an offline surrogate-optimization package. It does not replace the circuit-backed scripts and does not call Xyce during optimization. Its bundled 6T and 10T datasets are fixed training samples with per-device variation disabled.

```bash
python -m pip install -r size_optimization/openyield_v2/requirements.txt
python -m size_optimization.openyield_v2.run_experiment --dry-run
python -m size_optimization.openyield_v2.run_experiment
```

See [`openyield_v2/README.md`](openyield_v2/README.md) for algorithm selection and output files.
