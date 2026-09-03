"""Unified simulation and yield-estimation API."""

from .distributions import GaussianDistribution, LegacyDistributionAdapter
from .estimators import (
    ACS,
    AIS,
    BIBD,
    EFIAL,
    EXPERIMENTAL_ALGORITHMS,
    FUSIS,
    HSCS,
    MC,
    MNIS,
    OPT,
    STABLE_ALGORITHMS,
)
from .facade import YieldEstimator
from .results import EstimationResult, MultiConditionEstimationResult
from .simulation import (
    BudgetExceeded,
    BudgetLedger,
    SimulationBatch,
    SimulationRunner,
    TargetCellTestbenchAdapter,
)

__all__ = [
    "ACS", "AIS", "BIBD", "BudgetExceeded", "BudgetLedger", "EFIAL",
    "EXPERIMENTAL_ALGORITHMS", "EstimationResult", "FUSIS", "GaussianDistribution",
    "HSCS", "LegacyDistributionAdapter", "MC", "MNIS", "MultiConditionEstimationResult",
    "OPT", "STABLE_ALGORITHMS", "SimulationBatch", "SimulationRunner",
    "TargetCellTestbenchAdapter", "YieldEstimator",
]
