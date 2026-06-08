"""
SRAM Circuit Optimization using CPN (TabPFN-based Bayesian Optimization)
使用CPN (TabPFN) 的SRAM电路优化

Adapts the DirectFoMOptimizer from CPN/TabPFN which uses TabPFNRegressor as
the surrogate model and DEI (Distributional Expected Improvement) as the
acquisition function.

Reference: CPN/TabPFN directFOM optimizer (tabpfn_main_directFOM.py)
"""

import os
import sys
import time
import numpy as np
import torch
import warnings

from tabpfn import TabPFNRegressor

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
# TabPFN Direct-FoM Optimizer for SRAM
# ---------------------------------------------------------------------------
class TabPFNDirectFoMOptimizer:
    """Simplified Direct-FoM Bayesian Optimizer using TabPFNRegressor.

    Ported from CPN/TabPFN DirectFoMOptimizer, adapted for the SRAM sizing
    interface.  The surrogate predicts -merit (since we minimise internally
    similar to the original).  Acquisition is DEI computed from the bar-
    distribution output of TabPFN.
    """

    def __init__(self, dim, device="cpu", n_candidates=500):
        self.dim = dim
        self.device = device
        self.n_candidates = n_candidates

        self.tabpfn = TabPFNRegressor(device=self.device)

        # Observation storage (normalised [0,1]^d inputs)
        self.X_obs = []          # list of np arrays
        self.y_neg_merit = []    # -merit (for minimisation)

        self.best_merit = float("-inf")
        self.best_x = None

        self.iterations = 0

    # ---- public API -------------------------------------------------------

    def add_observation(self, x_np, merit_value):
        """Record one observation: x in [0,1]^d, merit (higher = better)."""
        self.X_obs.append(np.asarray(x_np, dtype=np.float64))
        self.y_neg_merit.append(-merit_value)  # model learns -merit
        if merit_value > self.best_merit:
            self.best_merit = merit_value
            self.best_x = x_np.copy()

    def get_next_point(self):
        """Return the next normalised point to evaluate."""
        self.iterations += 1

        # Not enough data – random exploration
        if len(self.X_obs) < 5:
            return np.random.rand(self.dim)

        X_train = np.array(self.X_obs)
        y_train = np.array(self.y_neg_merit)

        # Filter out non-finite y values (e.g. +inf from infeasible -(-inf) merit)
        finite_mask = np.isfinite(y_train)
        if finite_mask.sum() < 5:
            return np.random.rand(self.dim)
        X_train = X_train[finite_mask]
        y_train = y_train[finite_mask]

        # Generate candidates
        X_cand = self._generate_candidates(self.n_candidates)

        # Fit surrogate
        self.tabpfn.fit(X_train, y_train)

        # Predict with full distributional output
        try:
            pred = self.tabpfn.predict(X_cand, output_type="full")
        except Exception as e:
            print(f"  TabPFN predict failed: {e}, returning random point")
            return X_cand[np.random.randint(len(X_cand))]

        # --- DEI acquisition (from DirectFoMOptimizer) ---------------------
        logits_t = pred["logits"].squeeze(1)
        criterion = pred["criterion"]
        borders_t = criterion.borders.to(logits_t.device)
        probs_t = torch.softmax(logits_t, dim=-1)
        bin_centers_t = (borders_t[:-1] + borders_t[1:]) / 2.0

        y_best_t = torch.tensor(
            float(np.min(y_train)), device=logits_t.device, dtype=torch.float32
        )
        improvement = torch.clamp(y_best_t - bin_centers_t, min=0)
        ei = torch.sum(probs_t * improvement, dim=-1)

        best_idx = torch.argmax(ei).item()
        return X_cand[best_idx]

    # ---- candidate generation (LHS + best-region + boundary + random) ------

    def _generate_candidates(self, n):
        try:
            from scipy.stats import qmc
        except ImportError:
            qmc = None

        parts = []

        # 50 % LHS
        n_lhs = int(n * 0.5)
        if n_lhs > 0:
            if qmc is not None:
                sampler = qmc.LatinHypercube(d=self.dim, seed=self.iterations)
                parts.append(sampler.random(n=n_lhs))
            else:
                parts.append(np.random.rand(n_lhs, self.dim))

        # 20 % near best
        n_best = int(n * 0.2)
        if n_best > 0 and self.best_x is not None:
            tiled = np.tile(self.best_x, (n_best, 1))
            noise = np.random.randn(n_best, self.dim) * 0.1
            parts.append(np.clip(tiled + noise, 0, 1))
        elif n_best > 0:
            parts.append(np.random.rand(n_best, self.dim))

        # 20 % boundary
        n_bnd = int(n * 0.2)
        if n_bnd > 0:
            samples = []
            for _ in range(n_bnd):
                x = np.random.rand(self.dim)
                mask = np.random.rand(self.dim) < 0.3
                if np.any(mask):
                    x[mask] = np.random.randint(0, 2, size=int(mask.sum())).astype(float)
                samples.append(x)
            parts.append(np.array(samples))

        # fill rest with random
        total = sum(p.shape[0] for p in parts)
        n_rand = n - total
        if n_rand > 0:
            parts.append(np.random.rand(n_rand, self.dim))

        return np.vstack(parts)[:n]


# ---------------------------------------------------------------------------
# Main entry point (unified interface)
# ---------------------------------------------------------------------------
def main(config_path="config_sram.yaml", problem=None, max_iter=400, **kwargs):
    """Run CPN (TabPFN Direct-FoM) optimisation for SRAM sizing.

    Follows the same interface as demo_cbo.main / demo_pso.main etc.
    """
    print("===== SRAM optimization using CPN (TabPFN Direct-FoM) =====")
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

    # Pre-evaluate initial point for baseline
    initial_params = get_composite_initial_params()
    print("Pre-evaluating initial point for baseline...")
    objectives, constraints, initial_result, success = eval_fn(initial_params)
    if success and initial_result:
        print(f"Initial merit: {initial_result.get('merit', 'N/A')}")
    else:
        print("Warning: initial point evaluation failed")

    # Construct CPN optimizer
    optimizer = TabPFNDirectFoMOptimizer(dim=dim, device="cpu", n_candidates=500)

    # Tracking
    best_merit = float("-inf")
    best_params = None
    best_result = None
    best_iter = 0

    for it in range(1, max_iter + 1):
        t0 = time.time()

        # Get next candidate
        x_np = optimizer.get_next_point()
        x_clipped = np.clip(x_np, 0.0, 1.0)
        x_tensor = torch.tensor(x_clipped, dtype=torch.float32)
        params = parameter_space.convert_params(x_tensor)

        # Evaluate
        objectives, constraints, result, ok = eval_fn(params)
        merit = result.get("merit", float("-inf")) if ok and result else float("-inf")

        # Record observation
        optimizer.add_observation(x_clipped, merit)

        # Track best
        if merit > best_merit:
            best_merit = merit
            best_params = params
            best_result = result
            best_iter = it

        elapsed = time.time() - t0
        if it % 10 == 0 or it <= 5 or it == max_iter:
            print(
                f"  CPN iter {it}/{max_iter}: merit={merit:.4f}, "
                f"best={best_merit:.4f} (iter {best_iter}), "
                f"time={elapsed:.1f}s"
            )

    # Output
    print("\n===== CPN (TabPFN) Optimization Best Results =====")
    if best_merit != float("-inf"):
        print(f"Best Merit: {best_merit:.6e}  (found at iteration {best_iter})")
        if best_result:
            r = best_result
            print(f"Min SNM: {r.get('min_snm', 'N/A')}")
            print(f"Max power: {r.get('max_power', 'N/A'):.6e}")
            print(f"Area: {r.get('area', 0) * 1e12:.2f} µm²")
    else:
        print("No valid solution found")

    return {
        "params": best_params,
        "merit": best_merit,
        "fom": best_merit,
        "result": best_result,
        "iteration": best_iter,
    }


if __name__ == "__main__":
    SEED = 42
    seed_set(seed=SEED)
    main()
