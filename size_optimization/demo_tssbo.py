"""
SRAM Circuit Optimization using tSS-BO (truncated Subspace Sampling Bayesian Optimization)
使用tSS-BO的SRAM电路优化

Reference: Gu et al., "tSS-BO: Scalable Bayesian Optimization for Analog Circuit
Sizing via Truncated Subspace Sampling", DATE 2024.
"""

import os
import sys
import time
import math
import numpy as np
import torch
import pickle
import warnings

# ---------------------------------------------------------------------------
# Path setup – make tSS-BO source importable
# ---------------------------------------------------------------------------
_TSS_BO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tSS-BO")
if not os.path.isdir(_TSS_BO_ROOT):
    # Fallback: look for tSS-BO in user home or alongside project root
    _TSS_BO_ROOT = os.path.expanduser("~/tSS-BO")
if _TSS_BO_ROOT not in sys.path:
    sys.path.insert(0, _TSS_BO_ROOT)

# tSS-BO core components
from src.tSS_BO import tSubspace
from src.util import select_training_set, select_candidate_EI_unconstrained

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
# Parameter space wrapper (matches CBO / other demo conventions)
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
# tSS-BO optimizer adapted for SRAM multi-objective constrained optimisation
# ---------------------------------------------------------------------------
class TSSBOOptimizer:
    """Wraps the tSS-BO algorithm for SRAM circuit optimisation.

    tSS-BO works natively in a normalised space [-0.5, 0.5]^d.  Our framework
    uses [0, 1]^d.  We map between the two internally:
        tss_x = framework_x - 0.5
    """

    def __init__(
        self,
        parameter_space,
        sigma=0.2,
        mu=0.5,
        k=None,
        batch_size=1,
        n_candidates=200,
        n_resample=10,
        n_training=None,
    ):
        self.parameter_space = parameter_space
        self.dim = parameter_space.dim
        # tSS-BO bounds are [-0.5, 0.5]^d
        self.tss_bounds = torch.tensor([[-0.5, 0.5]] * self.dim)

        if k is None:
            k = max(100, 3 * self.dim)
        if n_training is None:
            n_training = min(self.dim * 2, 500)

        self.sigma = sigma
        self.mu = mu
        self.k = k
        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.n_resample = n_resample
        self.n_training = n_training

        # tSS-BO core subspace
        self.subspace = tSubspace(
            self.dim,
            self.tss_bounds,
            sigma=sigma,
            mu=mu,
            k=k,
        )

        # History (framework [0,1] space)
        self.eval_x = torch.empty(0, self.dim)
        self.eval_y = torch.empty(0)            # scalar merit (to maximise → we minimise negated)

        # Best tracking
        self.best_merit = float("-inf")
        self.best_params = None
        self.best_result = None

    # ---- coordinate transforms ----
    @staticmethod
    def _to_tss(x_01):
        """[0,1] → [-0.5, 0.5]"""
        return x_01 - 0.5

    @staticmethod
    def _to_framework(x_tss):
        """[-0.5, 0.5] → [0,1]"""
        return x_tss + 0.5

    # ---- main loop helpers ----
    def _eval_batch(self, x_01_batch, eval_fn):
        """Evaluate a batch of [0,1] normalised vectors.

        Returns list of (objectives, constraints, result, success, merit, x_01).
        """
        results = []
        for idx in range(x_01_batch.shape[0]):
            x_01 = x_01_batch[idx]
            params = self.parameter_space.convert_params(x_01)
            objectives, constraints, result, success = eval_fn(params)
            if success and result:
                merit = result.get("merit", float("-inf"))
            else:
                merit = float("-inf")
            results.append((objectives, constraints, result, success, merit, x_01))

            # Update best
            if merit > self.best_merit:
                self.best_merit = merit
                self.best_params = params
                self.best_result = result
        return results

    def run(self, eval_fn, max_iter, initial_params=None):
        """Run the tSS-BO optimisation loop."""
        iteration = 0
        g = 0  # generation counter

        # ---- optional initial point ----
        if initial_params is not None:
            x_01 = get_default_normalized_vector(self.parameter_space)
            if x_01:
                x_01 = torch.tensor(x_01, dtype=torch.float32)
            else:
                x_01 = torch.rand(self.dim)
            batch_results = self._eval_batch(x_01.unsqueeze(0), eval_fn)
            _, _, _, success, merit, _ = batch_results[0]
            if success:
                x_tss = self._to_tss(x_01)
                y_scalar = -merit  # tSS-BO minimises
                self.subspace.set_new_mean(x_tss, y_scalar)
                self.subspace.mean_f = torch.tensor(y_scalar)
                self.eval_x = torch.cat([self.eval_x, x_01.unsqueeze(0)])
                self.eval_y = torch.cat([self.eval_y, torch.tensor([y_scalar])])
            iteration += 1

        # ---- main iterations ----
        while iteration < max_iter:
            g += 1
            print(f"\n===== tSS-BO generation {g}, evals so far {iteration}/{max_iter} =====")
            t0 = time.time()

            # 1. Sample candidates in tSS space
            X_candidates_tss = self.subspace.sample_candidates(
                self.n_candidates, self.n_resample
            ).t()  # shape (n_candidates, dim)

            # 2. Select best candidates via EI (or raw if not enough data)
            model_list = None
            m_n_s = None
            if self.eval_x.shape[0] >= 2:
                X_train_tss = self._to_tss(self.eval_x)
                Y_train = self.eval_y  # already negated merit
                X_center = self.subspace.mean.ravel()

                if X_train_tss.shape[0] >= self.n_training:
                    D, B = self.subspace._eigen_decomposition()
                    X_train_tss, Y_train = select_training_set(
                        X_center, X_train_tss, Y_train, B, D, n_training=self.n_training
                    )

                try:
                    X_cand_tss, model_list, m_n_s = select_candidate_EI_unconstrained(
                        X_train_tss,
                        Y_train,
                        X_candidates_tss,
                        batch_size=min(self.batch_size, max_iter - iteration),
                    )
                except Exception as e:
                    print(f"EI selection failed ({e}), using top candidates from sampling")
                    X_cand_tss = X_candidates_tss[: self.batch_size]
            else:
                X_cand_tss = X_candidates_tss[: self.batch_size]

            # 3. Evaluate selected candidates
            X_cand_01 = self._to_framework(X_cand_tss).clamp(0, 1)
            batch_results = self._eval_batch(X_cand_01, eval_fn)

            # Collect y values for subspace update
            y_vals = []
            for objectives, constraints, result, success, merit, x_01 in batch_results:
                y_scalar = -merit if merit != float("-inf") else 1e6
                y_vals.append(y_scalar)
                self.eval_x = torch.cat([self.eval_x, x_01.unsqueeze(0)])
                self.eval_y = torch.cat([self.eval_y, torch.tensor([y_scalar])])
                iteration += 1

                if success and result:
                    print(f"  iter {iteration}: merit={merit:.4f}, min_snm={result.get('min_snm', 0):.4f}")
                else:
                    print(f"  iter {iteration}: FAILED")

            Y_cand = torch.tensor(y_vals, dtype=torch.float32)

            # 4. Update subspace
            if self.subspace.mean_f is None:
                self.subspace.set_mean_f(Y_cand[0])
                if X_cand_tss.shape[0] > 1:
                    self.subspace.update_subspace(
                        X_cand_tss[1:].t(), Y_cand[1:],
                        GP_model_list=None, mean_and_std=None,
                    )
            else:
                self.subspace.update_subspace(
                    X_cand_tss.t(), Y_cand,
                    GP_model_list=None, mean_and_std=None,
                )

            t1 = time.time()
            print(f"  generation time: {t1 - t0:.1f}s, best merit so far: {self.best_merit:.4f}")

        return {
            "params": self.best_params,
            "merit": self.best_merit,
            "fom": self.best_merit,
            "result": self.best_result,
            "iteration": iteration,
        }


# ---------------------------------------------------------------------------
# Main entry point (unified interface)
# ---------------------------------------------------------------------------
def main(config_path="config_sram.yaml", problem=None, max_iter=400, **kwargs):
    """Run tSS-BO optimisation for SRAM sizing.

    Follows the same interface as demo_cbo.main / demo_pso.main etc.
    """
    print("===== SRAM optimization using tSS-BO =====")
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

    dim = parameter_space.dim

    # Initial params
    initial_params = get_composite_initial_params()

    # Pre-evaluate initial point for baseline
    print("Pre-evaluating initial point for baseline...")
    objectives, constraints, initial_result, success = eval_fn(initial_params)
    if success and initial_result:
        print(f"Initial merit: {initial_result.get('merit', 'N/A')}")
    else:
        print("Warning: initial point evaluation failed")

    # Construct optimizer
    optimizer = TSSBOOptimizer(
        parameter_space,
        sigma=0.2,
        mu=0.5,
        k=max(100, 3 * dim),
        batch_size=1,          # sequential evaluation, one at a time
        n_candidates=200,
        n_resample=10,
        n_training=min(dim * 2, 500),
    )

    # Run
    ret = optimizer.run(eval_fn, max_iter=max_iter, initial_params=initial_params)

    # Output
    print("\n===== tSS-BO Optimization Best Results =====")
    if ret["merit"] is not None and ret["merit"] != float("-inf"):
        print(f"Best Merit: {ret['merit']:.6e}")
        if ret["result"]:
            print(f"Min SNM: {ret['result'].get('min_snm', 'N/A')}")
            r = ret["result"]
            area = r.get("area", 0)
            print(f"Max power: {r.get('max_power', 'N/A'):.6e}")
            print(f"Area: {area * 1e12:.2f} µm²")
    else:
        print("No valid solution found")

    return ret


if __name__ == "__main__":
    SEED = 42
    seed_set(seed=SEED)
    main()
