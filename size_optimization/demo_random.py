"""
Random Search baseline for SRAM circuit optimization.
随机搜索基线算法
"""

import os
import sys
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from size_optimization.exp_utils import (
    seed_set, create_directories, evaluate_sram,
    CompositeSRAMParameterSpace, get_composite_initial_params,
)


def main(config_path="config_sram.yaml", problem=None, max_iter=None):
    """Random search over SRAM parameter space."""
    print("===== SRAM optimization using Random Search =====")
    create_directories()

    if problem is not None and isinstance(problem, (tuple, list)) and len(problem) >= 2:
        parameter_space = problem[0]
        external_eval_fn = problem[1]
    else:
        parameter_space = CompositeSRAMParameterSpace(config_path)
        external_eval_fn = None

    _max_iter = max_iter if isinstance(max_iter, int) and max_iter > 0 else 5

    best_merit = float("-inf")
    best_params = None
    best_result = None
    eval_fn = external_eval_fn if external_eval_fn is not None else evaluate_sram

    for i in range(_max_iter):
        x = np.random.uniform(0, 1, parameter_space.bounds.shape[1])
        params = parameter_space.convert_params(torch.tensor(x, dtype=torch.float32))
        print(f"\n[Random search] Iteration {i+1}/{_max_iter}")
        objectives, constraints, result, success = eval_fn(params)
        if success and result:
            merit = result.get("merit", result.get("fom", float("-inf")))
            if merit > best_merit:
                best_merit = merit
                best_params = params
                best_result = result
                print(f"  New best merit: {best_merit:.6e}")

    print("\n===== Random Search completed =====")
    return {"params": best_params, "merit": best_merit, "result": best_result, "iteration": _max_iter}


if __name__ == "__main__":
    seed_set(42)
    main()
