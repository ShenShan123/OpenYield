"""OpenYield yield-estimation entry point.

The old script imported the removed ``sram_yield_estimation`` package and a
deleted ``spice.py`` module. This module is now import-safe and exposes the
unified facade while leaving SRAM testbench construction to callers.
"""

from __future__ import annotations

from yield_estimation import (
    EXPERIMENTAL_ALGORITHMS,
    STABLE_ALGORITHMS,
    SimulationRunner,
    YieldEstimator,
)


def create_estimator(
    *,
    model,
    algorithm_choice="MC",
    basic_params=None,
    algo_params=None,
    spice_params=None,
) -> YieldEstimator:
    """Create a configured estimator without starting a simulation."""
    return YieldEstimator(
        model=model,
        algorithm_choice=algorithm_choice,
        basic_params=basic_params,
        algo_params=algo_params,
        spice_params=spice_params,
    )


__all__ = [
    "EXPERIMENTAL_ALGORITHMS",
    "STABLE_ALGORITHMS",
    "SimulationRunner",
    "YieldEstimator",
    "create_estimator",
]
