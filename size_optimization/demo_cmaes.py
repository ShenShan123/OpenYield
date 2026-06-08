"""
SRAM Circuit Optimization using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
使用CMA-ES的SRAM电路优化

Uses the pycma library (pip install cma).
Reference: Hansen, N. "The CMA Evolution Strategy: A Tutorial", 2016.
"""

import os
import sys
import time
import numpy as np
import torch
import warnings

import cma

# Project utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from size_optimization.exp_utils import (
    seed_set,
    create_directories,
    evaluate_sram,
    CompositeSRAMParameterSpace,
    get_default_normalized_vector,
    get_composite_initial_params,
    estimate_scaled_total_area,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Parameter space wrapper
# ---------------------------------------------------------------------------
class SRAMParameterSpace:
    def __init__(self, config_path="config_sram.yaml"):
        try:
            self._space = CompositeSRAMParameterSpace(config_path)
            self.bounds = self._space.bounds
        except Exception:
            print("Warning: Config loading failed, using default 6-dim space")
            self._space = None
            self.bounds = torch.tensor([[0.0] * 6, [1.0] * 6])

    @property
    def dim(self):
        return self.bounds.shape[1]

    def convert_params(self, x):
        if self._space is not None:
            return self._space.convert_params(x)
        raise RuntimeError("CompositeSRAMParameterSpace not initialized")

    def print_params(self, params):
        if self._space is not None:
            return self._space.print_params(params)
        for k, v in params.items():
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# CMA-ES Optimizer wrapper
# ---------------------------------------------------------------------------
class CMAESOptimizer:
    """Wraps pycma's CMAEvolutionStrategy for SRAM circuit optimisation.

    CMA-ES is a derivative-free evolutionary strategy for continuous optimisation.
    It works in [0,1]^d via built-in bounds handling.
    We negate the merit (CMA-ES minimises) to maximise FoM.
    """

    def __init__(self, parameter_space, sigma0=0.3, popsize=None, seed=None):
        self.parameter_space = parameter_space
        self.dim = parameter_space.dim

        # CMA-ES options
        opts = {
            "bounds": [0, 1],
            "tolfun": 1e-11,
            "tolx": 1e-11,
            "verbose": -1,         # suppress pycma verbosity
            "verb_disp": 0,
            "verb_log": 0,
        }
        if popsize is not None:
            opts["popsize"] = popsize
        if seed is not None:
            opts["seed"] = seed

        # Start from centre of parameter space
        x0 = [0.5] * self.dim
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        # Best tracking
        self.best_merit = float("-inf")
        self.best_params = None
        self.best_result = None
        self.total_evals = 0

    def run(self, eval_fn, max_iter):
        """Run CMA-ES optimisation for max_iter function evaluations."""
        generation = 0

        while self.total_evals < max_iter and not self.es.stop():
            generation += 1
            t0 = time.time()

            # Ask for candidates
            remaining = max_iter - self.total_evals
            solutions = self.es.ask()
            # Limit to remaining budget
            if len(solutions) > remaining:
                solutions = solutions[:remaining]

            fitness_values = []
            for x in solutions:
                x_clipped = np.clip(x, 0.0, 1.0)
                x_tensor = torch.tensor(x_clipped, dtype=torch.float32)
                params = self.parameter_space.convert_params(x_tensor)

                objectives, constraints, result, success = eval_fn(params)
                self.total_evals += 1

                if success and result:
                    merit = result.get("merit", float("-inf"))
                    # CMA-ES minimises → negate merit
                    fitness_values.append(-merit)

                    if merit > self.best_merit:
                        self.best_merit = merit
                        self.best_params = params
                        self.best_result = result
                else:
                    # Penalty for failed simulations
                    fitness_values.append(1e6)

            # Tell CMA-ES the fitness values
            # Must match the full population for tell()
            if len(fitness_values) == len(solutions):
                self.es.tell(solutions, fitness_values)
            else:
                # Pad if we truncated solutions
                self.es.tell(solutions[:len(fitness_values)], fitness_values)

            t1 = time.time()
            print(
                f"  CMA-ES gen {generation}: evals={self.total_evals}/{max_iter}, "
                f"best_merit={self.best_merit:.4f}, sigma={self.es.sigma:.4f}, "
                f"time={t1 - t0:.1f}s"
            )

        return {
            "params": self.best_params,
            "merit": self.best_merit,
            "fom": self.best_merit,
            "result": self.best_result,
            "iteration": self.total_evals,
        }


# ---------------------------------------------------------------------------
# Main entry point (unified interface)
# ---------------------------------------------------------------------------
def main(config_path="config_sram.yaml", problem=None, max_iter=400, **kwargs):
    """Run CMA-ES optimisation for SRAM sizing.

    Follows the same interface as demo_cbo.main / demo_pso.main etc.
    """
    print("===== SRAM optimization using CMA-ES =====")
    create_directories()

    # Resolve parameter space and eval backend
    if problem is not None:
        try:
            parameter_space, eval_fn, _ = problem
        except Exception:
            parameter_space = SRAMParameterSpace(config_path)
            eval_fn = evaluate_sram
    else:
        parameter_space = SRAMParameterSpace(config_path)
        eval_fn = evaluate_sram

    # Pre-evaluate initial point for baseline
    initial_params = get_composite_initial_params()
    print("Pre-evaluating initial point for baseline...")
    objectives, constraints, initial_result, success = eval_fn(initial_params)
    if success and initial_result:
        print(f"Initial merit: {initial_result.get('merit', 'N/A')}")
    else:
        print("Warning: initial point evaluation failed")

    # Construct and run optimizer
    optimizer = CMAESOptimizer(
        parameter_space,
        sigma0=0.3,
        popsize=None,    # auto from dimension
        seed=None,
    )
    ret = optimizer.run(eval_fn, max_iter=max_iter)

    # Output
    print("\n===== CMA-ES Optimization Best Results =====")
    if ret["merit"] is not None and ret["merit"] != float("-inf"):
        print(f"Best Merit: {ret['merit']:.6e}")
        if ret["result"]:
            r = ret["result"]
            print(f"Min SNM: {r.get('min_snm', 'N/A')}")
            print(f"Max power: {r.get('max_power', 'N/A'):.6e}")
            print(f"Area: {r.get('area', 0) * 1e12:.2f} µm²")
    else:
        print("No valid solution found")

    return ret


if __name__ == "__main__":
    SEED = 42
    seed_set(seed=SEED)
    main()
