"""Online multi-objective TabPFN coarse-to-refine optimization for SRAM.

``max_iter`` is the exact real-SPICE attempt budget.  Small budgets collect only
online context data.  Full CTR runs use at most 300 context attempts and reserve
at least 30 calls for final audit.  The final TabPFN context drives a four-
objective SMSEMOA coarse search; KMeans representatives of the combined
context/coarse Pareto set are refined with an equal-weight relative-change
LogSumExp loss.  Final audit prioritizes predicted FoM top-K, then greedily
maximizes marginal normalized hypervolume.

TabPFN predictions made by active selection, SMSEMOA coarse search, and
differentiable refinement are counted separately and never consume SPICE budget.
No bundled CSV dataset is read.
"""

from __future__ import annotations

import math
import os
import sys
import time
import heapq
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


OBJECTIVE_NAMES = ("neg_min_snm", "max_power", "area", "max_delay")

# Default online-budget policy.  These defaults can be overridden from
# online_ctr/configs/online.yaml or keyword arguments passed to main().
TARGET_CONTEXT_SIZE = 300
MIN_CONTEXT_FOR_CTR = 30
MIN_FINAL_AUDITS = 30
INITIAL_LHS_SIZE = 100

# Backward-compatible names retained for callers that imported the old
# constants.  They no longer impose a 100-sample TabPFN requirement.
MIN_TRAIN_SAMPLES = MIN_CONTEXT_FOR_CTR
MAX_TRAIN_SAMPLES = TARGET_CONTEXT_SIZE

Evaluation = Tuple[Sequence[float], Sequence[float], Optional[Dict[str, Any]], bool]


def _project_utils() -> Dict[str, Any]:
    from size_optimization.exp_utils import (  # pylint: disable=import-outside-toplevel
        CompositeSRAMParameterSpace,
        create_directories,
        evaluate_sram,
        get_default_normalized_vector,
        seed_set,
    )

    return {
        "space": CompositeSRAMParameterSpace,
        "create_directories": create_directories,
        "evaluate": evaluate_sram,
        "default_vector": get_default_normalized_vector,
        "seed_set": seed_set,
    }


def _build_tabpfn_manager(device: str):
    from size_optimization.openyield_v2.surrogate_utils import (  # pylint: disable=import-outside-toplevel
        MultiTargetSurrogateManager,
    )

    return MultiTargetSurrogateManager(target_names=list(OBJECTIVE_NAMES), device=device)


def _resolve_device(requested: str, gpu_index: Optional[int] = None) -> str:
    requested = str(requested).lower()
    preferred_index = 0 if gpu_index is None else int(gpu_index)
    if preferred_index < 0:
        raise ValueError("gpu_index must be non-negative")

    if requested == "cpu":
        return "cpu"
    if requested != "auto" and requested != "cuda" and not requested.startswith("cuda:"):
        raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        if requested == "auto":
            return "cpu"
        raise RuntimeError(
            f"device='{requested}' requested, but PyTorch is not installed"
        ) from exc

    if requested == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        if preferred_index >= torch.cuda.device_count():
            print(
                f"CTR gpu_index={preferred_index} is unavailable; "
                "falling back to cuda:0."
            )
            preferred_index = 0
        return f"cuda:{preferred_index}"

    if requested == "cuda":
        requested = f"cuda:{preferred_index}"
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device='{requested}' requested, but CUDA is unavailable")
        try:
            requested_index = int(requested.split(":", 1)[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("CUDA device must use the form cuda:<index>") from exc
        if requested_index < 0 or requested_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"device='{requested}' requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible"
            )
        return f"cuda:{requested_index}"
    raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")


def _numpy_bounds(parameter_space: Any) -> Tuple[np.ndarray, np.ndarray]:
    bounds = parameter_space.bounds
    if hasattr(bounds, "detach"):
        bounds = bounds.detach().cpu().numpy()
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] == 0:
        raise ValueError("parameter_space.bounds must have shape (2, dimension)")
    lower, upper = bounds[0].copy(), bounds[1].copy()
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("CTR requires finite bounds")
    if np.any(upper <= lower):
        raise ValueError("every upper bound must be greater than its lower bound")
    return lower, upper


def resolve_training_target(
    total_budget: int,
    requested: Optional[int] = None,
    *,
    target_context_size: int = TARGET_CONTEXT_SIZE,
    min_context_for_ctr: int = MIN_CONTEXT_FOR_CTR,
    min_final_audits: int = MIN_FINAL_AUDITS,
) -> int:
    """Return the number of real-SPICE *attempts* assigned to context data.

    Budgets below ``min_context_for_ctr + min_final_audits`` run in acquisition-
    only mode, so every evaluation is assigned to the online context.  Larger
    budgets cap context acquisition at ``target_context_size`` and reserve at
    least ``min_final_audits`` calls for final real-SPICE auditing.
    """
    total_budget = int(total_budget)
    if total_budget <= 0:
        raise ValueError("max_iter must be positive")
    target_context_size = max(1, int(target_context_size))
    min_context_for_ctr = max(1, int(min_context_for_ctr))
    min_final_audits = max(1, int(min_final_audits))

    full_ctr_threshold = min_context_for_ctr + min_final_audits
    if total_budget < full_ctr_threshold:
        if requested is not None and int(requested) != total_budget:
            raise ValueError(
                "acquisition-only CTR uses the complete real-SPICE budget as "
                f"context attempts; expected training_evals={total_budget}"
            )
        return total_budget

    if requested is None:
        requested = min(target_context_size, total_budget - min_final_audits)
    requested = int(requested)
    max_context = min(target_context_size, total_budget - min_final_audits)
    if not min_context_for_ctr <= requested <= max_context:
        raise ValueError(
            f"training_evals must be in [{min_context_for_ctr}, "
            f"{max_context}], got {requested}"
        )
    return requested


def is_pareto_mask(costs: np.ndarray) -> np.ndarray:
    """Return nondominated mask for finite objectives that are all minimized."""
    values = np.asarray(costs, dtype=float)
    if values.ndim != 2:
        raise ValueError("costs must be a 2-D matrix")
    mask = np.all(np.isfinite(values), axis=1)
    valid_indices = np.flatnonzero(mask)
    valid = values[valid_indices]
    keep = np.ones(len(valid), dtype=bool)
    for i in range(len(valid)):
        if not keep[i]:
            continue
        dominated = np.all(valid <= valid[i], axis=1) & np.any(valid < valid[i], axis=1)
        if np.any(dominated):
            keep[i] = False
    result = np.zeros(len(values), dtype=bool)
    result[valid_indices] = keep
    return result


def _predicted_fom(costs: np.ndarray) -> np.ndarray:
    values = np.asarray(costs, dtype=float)
    snm = -values[:, 0]
    power = values[:, 1]
    area = values[:, 2]
    delay = values[:, 3]
    valid = (
        np.isfinite(values).all(axis=1)
        & (snm > 0)
        & (power > 0)
        & (area > 0)
        & (delay > 0)
    )
    fom = np.full(len(values), -np.inf, dtype=float)
    fom[valid] = np.log10(snm[valid] / (power[valid] * np.sqrt(area[valid]) * delay[valid]))
    return fom


def _physical_mask(costs: np.ndarray) -> np.ndarray:
    values = np.asarray(costs, dtype=float)
    return (
        np.all(np.isfinite(values), axis=1)
        & (values[:, 0] < 0.0)
        & np.all(values[:, 1:] > 0.0, axis=1)
    )


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values, axis=0)
    hi = np.nanmax(values, axis=0)
    return (values - lo) / np.maximum(hi - lo, 1e-30)


def _farthest_indices(costs: np.ndarray, count: int, priority: Iterable[int] = ()) -> List[int]:
    """Select well-spread objective-space representatives with fixed priorities."""
    costs = np.asarray(costs, dtype=float)
    if count <= 0 or len(costs) == 0:
        return []
    chosen: List[int] = []
    for index in priority:
        index = int(index)
        if 0 <= index < len(costs) and index not in chosen:
            chosen.append(index)
        if len(chosen) >= count:
            return chosen
    z = _normalized(costs)
    if not chosen:
        chosen.append(int(np.argmin(np.mean(z, axis=1))))
    while len(chosen) < min(count, len(costs)):
        distance = np.linalg.norm(z[:, None, :] - z[np.asarray(chosen)][None, :, :], axis=2)
        nearest = np.min(distance, axis=1)
        nearest[chosen] = -np.inf
        chosen.append(int(np.argmax(nearest)))
    return chosen


def _kmeans_representative_indices(
    costs: np.ndarray,
    count: int,
    *,
    seed: int,
) -> List[int]:
    """Choose objective-space medoids from KMeans clusters.

    Objectives are normalized before clustering so SNM, power, area, and delay
    contribute on comparable scales.  The returned points are existing designs
    nearest to cluster centers; synthetic KMeans centers are never optimized.
    """
    values = np.asarray(costs, dtype=float)
    if values.ndim != 2:
        raise ValueError("costs must be a 2-D matrix")
    if count <= 0 or len(values) == 0:
        return []
    n_clusters = min(int(count), len(values))
    if n_clusters == len(values):
        return list(range(len(values)))

    from sklearn.cluster import KMeans  # pylint: disable=import-outside-toplevel

    normalized = _normalized(values)
    model = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=10)
    labels = model.fit_predict(normalized)
    chosen: List[int] = []
    for cluster_index, center in enumerate(model.cluster_centers_):
        members = np.flatnonzero(labels == cluster_index)
        if len(members) == 0:
            continue
        distances = np.linalg.norm(normalized[members] - center, axis=1)
        chosen.append(int(members[int(np.argmin(distances))]))
    return chosen


def _greedy_hv_indices(
    candidate_costs: np.ndarray,
    count: int,
    *,
    baseline_costs: Optional[np.ndarray] = None,
    reference_margin: float = 0.10,
) -> List[int]:
    """Greedily maximize marginal hypervolume with lazy CELF updates.

    All objectives are minimized.  A single normalization transform is fitted
    to the baseline and candidate pool, and the reference point is fixed for the
    complete selection.  CELF preserves greedy marginal-HV semantics without
    recomputing every candidate after every selected point.
    """
    candidates = np.asarray(candidate_costs, dtype=float)
    if count <= 0 or len(candidates) == 0:
        return []
    count = min(int(count), len(candidates))
    if count == len(candidates):
        return list(range(len(candidates)))

    baseline = np.empty((0, candidates.shape[1]), dtype=float)
    if baseline_costs is not None:
        raw_baseline = np.asarray(baseline_costs, dtype=float)
        if raw_baseline.size:
            baseline = raw_baseline.reshape(-1, candidates.shape[1])

    all_values = np.vstack([baseline, candidates])
    lo = np.min(all_values, axis=0)
    hi = np.max(all_values, axis=0)
    scale = np.maximum(hi - lo, 1e-30)
    normalized_candidates = np.clip((candidates - lo) / scale, 0.0, 1.0)
    normalized_baseline = np.clip((baseline - lo) / scale, 0.0, 1.0)
    reference = np.full(candidates.shape[1], 1.0 + max(0.0, float(reference_margin)))

    from pymoo.indicators.hv import HV  # pylint: disable=import-outside-toplevel

    indicator = HV(ref_point=reference)

    def hv(points: np.ndarray) -> float:
        if points.size == 0:
            return 0.0
        finite = points[np.all(np.isfinite(points), axis=1)]
        return float(indicator.do(finite)) if len(finite) else 0.0

    selected: List[int] = []
    current_points = normalized_baseline.copy()
    current_hv = hv(current_points)
    heap: List[Tuple[float, int, int]] = []
    for index, point in enumerate(normalized_candidates):
        gain = hv(np.vstack([current_points, point])) - current_hv
        heapq.heappush(heap, (-gain, index, 0))

    selected_set: set[int] = set()
    while heap and len(selected) < count:
        neg_gain, index, stamp = heapq.heappop(heap)
        if index in selected_set:
            continue
        if stamp == len(selected):
            selected.append(index)
            selected_set.add(index)
            current_points = np.vstack([current_points, normalized_candidates[index]])
            current_hv = hv(current_points)
            continue
        gain = hv(np.vstack([current_points, normalized_candidates[index]])) - current_hv
        heapq.heappush(heap, (-gain, index, len(selected)))

    return selected


def _value_key(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), 15)
    if isinstance(value, (int, np.integer, str, bool, type(None))):
        return value
    if isinstance(value, dict):
        return tuple(sorted((str(k), _value_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_value_key(v) for v in value)
    return str(value)


def _params_key(params: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(sorted((str(k), _value_key(v)) for k, v in params.items()))


def _extract_continuous_mask(space: Any) -> np.ndarray:
    """Infer which normalized dimensions may move during gradient refinement."""
    dim = int(space.bounds.shape[1])

    # JointParameterSpace: rows/cols are discrete, followed by base_space.
    if all(hasattr(space, attr) for attr in ("row_choices", "column_choices", "base_space")):
        child = _extract_continuous_mask(space.base_space)
        mask = np.concatenate([np.zeros(2, dtype=bool), child])
        return mask[:dim]

    # CompositeSRAMParameterSpace: bitcell dimensions followed by peripherals.
    if hasattr(space, "bitcell_space") and hasattr(space, "peripheral_defs"):
        bitcell = _extract_continuous_mask(space.bitcell_space)
        peripheral = np.asarray(
            [str(item.get("type", "")).lower() == "continuous" for item in space.peripheral_defs],
            dtype=bool,
        )
        mask = np.concatenate([bitcell, peripheral])
        if len(mask) == dim:
            return mask

    if hasattr(space, "param_info"):
        mask: List[bool] = []
        for info in space.param_info.values():
            kind = str(info.get("type", "")).lower()
            count = int(info.get("count", 1))
            mask.extend([kind.startswith("continuous")] * count)
        if len(mask) == dim:
            return np.asarray(mask, dtype=bool)

    # Unknown injected spaces are treated as continuous.
    return np.ones(dim, dtype=bool)


@dataclass
class RealObservation:
    x: np.ndarray
    params: Dict[str, Any]
    objectives: np.ndarray
    constraints: Sequence[float]
    result: Optional[Dict[str, Any]]
    success: bool
    source: str
    elapsed: float

    @property
    def merit(self) -> float:
        if not self.success or not self.result:
            return float("-inf")
        try:
            value = float(self.result.get("merit", self.result.get("fom", -np.inf)))
        except (TypeError, ValueError):
            return float("-inf")
        return value if math.isfinite(value) else float("-inf")


class OnlineTabPFNCTR:
    def __init__(
        self,
        parameter_space: Any,
        *,
        seed: int,
        device: str,
        gpu_index: int,
        training_evals: Optional[int],
        target_context_size: int,
        min_context_for_ctr: int,
        min_final_audits: int,
        initial_lhs_size: int,
        active_batch_size: int,
        active_candidates: int,
        pareto_bonus: float,
        novelty_weight: float,
        coarse_evals: int,
        coarse_pop_size: int,
        representative_points: int,
        refine_steps: int,
        refine_lr: float,
        refine_smoothing: float,
        physical_penalty: float,
        gradient_clip: float,
        fom_top_k: int,
        hv_reference_margin: float,
    ) -> None:
        self.space = parameter_space
        self.lower, self.upper = _numpy_bounds(parameter_space)
        self.dim = len(self.lower)
        self.continuous_mask = _extract_continuous_mask(parameter_space)
        self.seed = int(seed)
        self.device = _resolve_device(device, gpu_index)
        print(f"CTR surrogate device: {self.device}")
        self.requested_training_evals = training_evals
        self.target_context_size = max(1, int(target_context_size))
        self.min_context_for_ctr = max(1, int(min_context_for_ctr))
        self.min_final_audits = max(1, int(min_final_audits))
        self.initial_lhs_size = max(1, int(initial_lhs_size))
        self.active_batch_size = max(1, int(active_batch_size))
        self.active_candidates = max(128, int(active_candidates))
        self.pareto_bonus = max(0.0, float(pareto_bonus))
        self.novelty_weight = max(0.0, float(novelty_weight))
        self.coarse_pop_size = max(4, int(coarse_pop_size))
        requested_coarse = max(self.coarse_pop_size, int(coarse_evals))
        self.coarse_evals = (
            int(math.ceil(requested_coarse / self.coarse_pop_size)) * self.coarse_pop_size
        )
        self.representative_points = max(1, int(representative_points))
        self.refine_steps = max(1, int(refine_steps))
        self.refine_lr = float(refine_lr)
        self.refine_smoothing = float(refine_smoothing)
        self.physical_penalty = max(0.0, float(physical_penalty))
        self.gradient_clip = max(0.0, float(gradient_clip))
        self.fom_top_k = max(0, int(fom_top_k))
        self.hv_reference_margin = max(0.0, float(hv_reference_margin))
        self.rng = np.random.default_rng(self.seed)

        self.real_observations: List[RealObservation] = []
        self.real_design_keys: set[Tuple[Any, ...]] = set()
        self.manager: Any = None
        self.surrogate_queries = 0
        self.fit_count = 0
        self.coarse_history_x: List[np.ndarray] = []
        self.coarse_history_y: List[np.ndarray] = []
        self.refine_history_x: List[np.ndarray] = []
        self.refine_history_y: List[np.ndarray] = []

    def _decode(self, x: np.ndarray) -> Dict[str, Any]:
        return self.space.convert_params(np.asarray(x, dtype=np.float32))

    def _is_unseen(self, x: np.ndarray) -> bool:
        try:
            return _params_key(self._decode(x)) not in self.real_design_keys
        except Exception:
            return False

    def _scale_unit(self, x_unit: np.ndarray) -> np.ndarray:
        return self.lower + np.asarray(x_unit) * (self.upper - self.lower)

    def _unit(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x) - self.lower) / (self.upper - self.lower)

    @staticmethod
    def _objective_vector(evaluation: Evaluation) -> Tuple[np.ndarray, bool]:
        objectives, _, result, success = evaluation
        try:
            values = np.asarray(objectives, dtype=float).reshape(-1)
        except Exception:
            values = np.empty(0, dtype=float)
        physical = bool(len(values) >= 4 and _physical_mask(values[:4].reshape(1, -1))[0])
        valid = bool(success and result and physical)
        return values[:4] if len(values) >= 4 else np.full(4, np.nan), valid

    def evaluate_real(
        self,
        x: np.ndarray,
        eval_fn: Callable[[Dict[str, Any]], Evaluation],
        source: str,
    ) -> RealObservation:
        x = np.asarray(x, dtype=float).copy()
        params = self._decode(x)
        started = time.time()
        evaluation = eval_fn(params)
        elapsed = time.time() - started
        objectives, valid = self._objective_vector(evaluation)
        _, constraints, result, success = evaluation
        observation = RealObservation(
            x=x,
            params=params,
            objectives=objectives,
            constraints=list(constraints) if constraints is not None else [],
            result=result,
            success=bool(success) and valid,
            source=source,
            elapsed=elapsed,
        )
        self.real_observations.append(observation)
        self.real_design_keys.add(_params_key(params))
        status = f"FoM={observation.merit:.6g}" if observation.success else "FAILED"
        print(
            f"  CTR SPICE {len(self.real_observations)} [{source}]: "
            f"{status}, valid_train={len(self.successful_observations)}, time={elapsed:.1f}s"
        )
        return observation

    @property
    def successful_observations(self) -> List[RealObservation]:
        return [obs for obs in self.real_observations if obs.success]

    def fit_tabpfn(self) -> None:
        valid = self.successful_observations
        if len(valid) < 2:
            raise RuntimeError(
                "CTR needs at least 2 valid online samples to build a surrogate; "
                f"only {len(valid)} are available"
            )
        x_train = np.vstack([obs.x for obs in valid])
        y_train = pd.DataFrame(
            np.vstack([obs.objectives for obs in valid]), columns=OBJECTIVE_NAMES
        )
        self.manager = _build_tabpfn_manager(self.device)
        self.manager.fit_all(x_train, y_train)
        self.fit_count += 1
        print(f"  TabPFN context updated: {len(valid)} samples (fit #{self.fit_count})")

    def predict(self, x: np.ndarray, *, return_std: bool = False):
        x = np.asarray(x, dtype=np.float32)
        self.surrogate_queries += len(x)
        prediction = self.manager.predict(x, return_std=return_std)
        if return_std:
            means, stds = prediction
            mean_matrix = np.column_stack([means[name] for name in OBJECTIVE_NAMES])
            std_matrix = np.column_stack([stds[name] for name in OBJECTIVE_NAMES])
            return mean_matrix, std_matrix
        return np.column_stack([prediction[name] for name in OBJECTIVE_NAMES])

    def _space_filling_point(self, index: int) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed + index)
        for _ in range(100):
            point = self._scale_unit(sampler.random(1)[0])
            if self._is_unseen(point):
                return point
        return self.rng.uniform(self.lower, self.upper)

    def _active_points(self, count: int) -> List[np.ndarray]:
        unit = qmc.LatinHypercube(
            d=self.dim, seed=self.seed + 10000 + len(self.real_observations)
        ).random(self.active_candidates)
        candidates = self._scale_unit(unit)
        candidates = np.asarray([x for x in candidates if self._is_unseen(x)])
        if len(candidates) == 0:
            return []
        means, stds = self.predict(candidates, return_std=True)
        finite = np.all(np.isfinite(means), axis=1) & np.all(np.isfinite(stds), axis=1)
        candidates, means, stds = candidates[finite], means[finite], stds[finite]
        if len(candidates) == 0:
            return []

        physical = _physical_mask(means)
        pareto = np.zeros(len(means), dtype=bool)
        if np.any(physical):
            physical_indices = np.flatnonzero(physical)
            pareto[physical_indices] = is_pareto_mask(means[physical])
        scale = np.maximum(np.ptp(means, axis=0), 1e-30)
        uncertainty = np.mean(stds / scale, axis=1)
        scores = uncertainty.copy()
        scores[pareto] += self.pareto_bonus + uncertainty[pareto]

        selected: List[int] = []
        unit_candidates = self._unit(candidates)
        for _ in range(min(count, len(candidates))):
            if not selected:
                index = int(np.argmax(scores))
            else:
                novelty = np.min(
                    np.linalg.norm(
                        unit_candidates[:, None, :] - unit_candidates[np.asarray(selected)][None, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                combined = scores + self.novelty_weight * novelty
                combined[selected] = -np.inf
                index = int(np.argmax(combined))
            selected.append(index)
        return [candidates[index] for index in selected]

    def collect_training_data(
        self,
        eval_fn: Callable[[Dict[str, Any]], Evaluation],
        context_attempt_budget: int,
        *,
        initial_x: Optional[Sequence[float]] = None,
    ) -> None:
        """Collect online context while counting every real-SPICE attempt.

        The first part is one genuine LHS design.  Remaining attempts use
        Pareto/uncertainty active learning whenever at least two valid context
        samples are available; failed simulations still consume context budget.
        """
        context_attempt_budget = max(1, int(context_attempt_budget))
        lhs_count = min(
            context_attempt_budget,
            self.initial_lhs_size,
            max(4, int(math.ceil(context_attempt_budget / 3.0))),
        )
        point_index = 0
        lhs_design = self._scale_unit(
            qmc.LatinHypercube(d=self.dim, seed=self.seed).random(lhs_count)
        )
        if initial_x is not None:
            initial = np.asarray(initial_x, dtype=float).reshape(-1)
            if len(initial) == self.dim and np.all(np.isfinite(initial)):
                lhs_design[0] = np.clip(initial, self.lower, self.upper)

        for point in lhs_design:
            if len(self.real_observations) >= context_attempt_budget:
                break
            candidate = point if self._is_unseen(point) else self._space_filling_point(point_index)
            self.evaluate_real(candidate, eval_fn, "train-lhs")
            point_index += 1

        if len(self.successful_observations) >= 2:
            self.fit_tabpfn()

        while len(self.real_observations) < context_attempt_budget:
            remaining_attempts = context_attempt_budget - len(self.real_observations)
            batch_size = min(self.active_batch_size, remaining_attempts)
            points = self._active_points(batch_size) if self.manager is not None else []
            if not points:
                points = [self._space_filling_point(point_index + i) for i in range(batch_size)]
                point_index += batch_size
            for point in points:
                if len(self.real_observations) >= context_attempt_budget:
                    break
                self.evaluate_real(point, eval_fn, "train-pareto-active")
            if len(self.successful_observations) >= 2:
                self.fit_tabpfn()

    def run_coarse(self) -> Tuple[np.ndarray, np.ndarray]:
        from pymoo.algorithms.moo.sms import SMSEMOA  # pylint: disable=import-outside-toplevel
        from pymoo.core.problem import Problem  # pylint: disable=import-outside-toplevel
        from pymoo.optimize import minimize  # pylint: disable=import-outside-toplevel

        owner = self

        class SurrogateProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=owner.dim,
                    n_obj=4,
                    xl=owner.lower,
                    xu=owner.upper,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                values = owner.predict(np.asarray(x), return_std=False)
                owner.coarse_history_x.extend(np.asarray(x, dtype=float).copy())
                owner.coarse_history_y.extend(np.asarray(values, dtype=float).copy())
                optimized = values.copy()
                optimized[~_physical_mask(values)] = 1.0e30
                out["F"] = optimized

        algorithm = SMSEMOA(pop_size=self.coarse_pop_size, eliminate_duplicates=True)
        minimize(
            SurrogateProblem(),
            algorithm,
            termination=("n_eval", self.coarse_evals),
            seed=self.seed,
            verbose=False,
        )
        x = np.asarray(self.coarse_history_x, dtype=float)
        y = np.asarray(self.coarse_history_y, dtype=float)
        if len(x) == 0:
            raise RuntimeError("TabPFN coarse search produced no candidates")
        physical = _physical_mask(y)
        physical_front = np.zeros(len(y), dtype=bool)
        if np.any(physical):
            physical_indices = np.flatnonzero(physical)
            physical_front[physical_indices] = is_pareto_mask(y[physical])
        print(
            f"  Coarse complete: {len(x)} TabPFN candidates, "
            f"predicted physical Pareto size={int(physical_front.sum())}"
        )
        return x, y

    def refine(self, coarse_x: np.ndarray, coarse_y: np.ndarray) -> None:
        import torch  # pylint: disable=import-outside-toplevel

        # Combine trusted online context designs with the complete coarse-search
        # history, then cluster their joint nondominated set.  KMeans medoids give
        # deterministic, well-spread existing designs without forcing extreme
        # single-objective points into the refinement set.
        valid_context = self.successful_observations
        context_x = (
            np.vstack([obs.x for obs in valid_context])
            if valid_context
            else np.empty((0, self.dim), dtype=float)
        )
        context_y = (
            np.vstack([obs.objectives for obs in valid_context])
            if valid_context
            else np.empty((0, len(OBJECTIVE_NAMES)), dtype=float)
        )
        combined_x = np.vstack([context_x, np.asarray(coarse_x, dtype=float)])
        combined_y = np.vstack([context_y, np.asarray(coarse_y, dtype=float)])

        unique_x: List[np.ndarray] = []
        unique_y: List[np.ndarray] = []
        seen: set[Tuple[Any, ...]] = set()
        for x, y in zip(combined_x, combined_y):
            if not _physical_mask(np.asarray(y).reshape(1, -1))[0]:
                continue
            try:
                key = _params_key(self._decode(x))
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique_x.append(np.asarray(x, dtype=float))
            unique_y.append(np.asarray(y, dtype=float))

        if not unique_x:
            raise RuntimeError("No physical context/coarse design is available for refinement")
        seed_pool_x = np.vstack(unique_x)
        seed_pool_y = np.vstack(unique_y)
        physical_indices = np.flatnonzero(_physical_mask(seed_pool_y))
        pareto_indices = (
            physical_indices[is_pareto_mask(seed_pool_y[physical_indices])]
            if len(physical_indices)
            else np.empty(0, dtype=int)
        )
        if len(pareto_indices) == 0:
            raise RuntimeError("No finite predicted Pareto seed is available")
        local_indices = _kmeans_representative_indices(
            seed_pool_y[pareto_indices],
            min(self.representative_points, len(pareto_indices)),
            seed=self.seed,
        )
        seed_indices = pareto_indices[np.asarray(local_indices)]
        lower_t = torch.tensor(self.lower, dtype=torch.float32, device=self.device)
        upper_t = torch.tensor(self.upper, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(self.continuous_mask, dtype=torch.float32, device=self.device)

        for seed_number, history_index in enumerate(seed_indices):
            seed_x = np.asarray(seed_pool_x[history_index], dtype=np.float32)
            unit = np.clip(self._unit(seed_x), 1e-4, 1.0 - 1e-4)
            z = torch.nn.Parameter(
                torch.tensor(np.log(unit / (1.0 - unit)), dtype=torch.float32, device=self.device)
            )
            seed_t = torch.tensor(seed_x, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                seed_prediction = self.manager.predict_tensor(seed_t.reshape(1, -1))
                self.surrogate_queries += 1
                ref_t = torch.stack(
                    [seed_prediction[name].reshape(-1)[0] for name in OBJECTIVE_NAMES]
                ).detach()
            optimizer = torch.optim.Adam([z], lr=self.refine_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.refine_steps, eta_min=self.refine_lr * 0.1
            )

            for _step in range(self.refine_steps):
                optimizer.zero_grad(set_to_none=True)
                candidate = lower_t + torch.sigmoid(z) * (upper_t - lower_t)
                candidate = seed_t + mask_t * (candidate - seed_t)
                raw = self.manager.predict_tensor(candidate.reshape(1, -1))
                self.surrogate_queries += 1
                objectives = torch.stack(
                    [raw[name].reshape(-1)[0] for name in OBJECTIVE_NAMES]
                )
                relative = (objectives - ref_t) / torch.clamp(torch.abs(ref_t), min=1e-30)
                mu = max(self.refine_smoothing, 1e-6)
                loss = mu * (
                    torch.logsumexp(relative / mu, dim=0) - math.log(len(OBJECTIVE_NAMES))
                )
                physical_violation = (
                    torch.relu(objectives[0]) / torch.clamp(torch.abs(ref_t[0]), min=1e-30)
                    + torch.sum(
                        torch.relu(-objectives[1:])
                        / torch.clamp(torch.abs(ref_t[1:]), min=1e-30)
                    )
                )
                loss = loss + self.physical_penalty * physical_violation
                if not torch.isfinite(loss):
                    break
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_([z], max_norm=self.gradient_clip)
                optimizer.step()
                scheduler.step()

                # Record exactly one new post-update prediction per gradient step.
                with torch.no_grad():
                    updated = lower_t + torch.sigmoid(z) * (upper_t - lower_t)
                    updated = seed_t + mask_t * (updated - seed_t)
                    updated_raw = self.manager.predict_tensor(updated.reshape(1, -1))
                    self.surrogate_queries += 1
                    updated_objectives = torch.stack(
                        [updated_raw[name].reshape(-1)[0] for name in OBJECTIVE_NAMES]
                    )
                self.refine_history_x.append(updated.detach().cpu().numpy().copy())
                self.refine_history_y.append(updated_objectives.detach().cpu().numpy().copy())

            print(
                f"  Refine seed {seed_number + 1}/{len(seed_indices)}: "
                f"steps={self.refine_steps}, fixed_dims={int((~self.continuous_mask).sum())}"
            )

    def _surrogate_history(self) -> Tuple[np.ndarray, np.ndarray]:
        blocks_x = [np.asarray(self.coarse_history_x, dtype=float)]
        blocks_y = [np.asarray(self.coarse_history_y, dtype=float)]
        if self.refine_history_x:
            blocks_x.append(np.asarray(self.refine_history_x, dtype=float))
            blocks_y.append(np.asarray(self.refine_history_y, dtype=float))
        return np.vstack(blocks_x), np.vstack(blocks_y)

    def select_audit_candidates(
        self, count: int
    ) -> Tuple[List[Tuple[np.ndarray, str]], Dict[str, int]]:
        history_x, history_y = self._surrogate_history()

        # Deduplicate decoded physical designs and remove designs already simulated.
        unique_x: List[np.ndarray] = []
        unique_y: List[np.ndarray] = []
        keys: set[Tuple[Any, ...]] = set()
        for x, y in zip(history_x, history_y):
            try:
                key = _params_key(self._decode(x))
            except Exception:
                continue
            if (
                key in keys
                or key in self.real_design_keys
                or not _physical_mask(np.asarray(y).reshape(1, -1))[0]
            ):
                continue
            keys.add(key)
            unique_x.append(np.asarray(x, dtype=float))
            unique_y.append(np.asarray(y, dtype=float))
        if not unique_x:
            raise RuntimeError("No new coarse/refine design is available for SPICE audit")

        x = np.vstack(unique_x)
        y = np.vstack(unique_y)
        fom = _predicted_fom(y)
        top10 = [
            int(i) for i in np.argsort(fom)[::-1] if np.isfinite(fom[i])
        ][: self.fom_top_k]
        pareto = np.flatnonzero(is_pareto_mask(y)).tolist()

        selected: List[int] = []
        for index in top10:
            if index not in selected:
                selected.append(index)
            if len(selected) >= count:
                break

        pareto_remaining = [index for index in pareto if index not in selected]
        slots = count - len(selected)
        if slots > 0 and pareto_remaining:
            if len(pareto_remaining) <= slots:
                selected.extend(pareto_remaining)
            else:
                valid_context = self.successful_observations
                baseline_blocks = [
                    np.vstack([obs.objectives for obs in valid_context])
                    if valid_context
                    else np.empty((0, len(OBJECTIVE_NAMES)), dtype=float)
                ]
                if selected:
                    baseline_blocks.append(y[np.asarray(selected)])
                baseline = np.vstack(baseline_blocks)
                local = _greedy_hv_indices(
                    y[pareto_remaining],
                    slots,
                    baseline_costs=baseline,
                    reference_margin=self.hv_reference_margin,
                )
                selected.extend([pareto_remaining[i] for i in local])

        # Spend every remaining audit call on the best unselected history points.
        if len(selected) < count:
            for index in np.argsort(fom)[::-1]:
                index = int(index)
                if index not in selected:
                    selected.append(index)
                if len(selected) >= count:
                    break

        summary = {
            "history_unique": len(x),
            "predicted_pareto": len(pareto),
            "predicted_fom_top10": len(top10),
            "selected": min(count, len(selected)),
            "selected_fom_top10": sum(index in set(top10) for index in selected[:count]),
            "selected_pareto": sum(index in set(pareto) for index in selected[:count]),
        }
        labeled: List[Tuple[np.ndarray, str]] = []
        top10_set, pareto_set = set(top10), set(pareto)
        for index in selected[:count]:
            if index in top10_set:
                label = "final-fom-top10-audit"
            elif index in pareto_set:
                label = "final-pareto-audit"
            else:
                label = "final-history-fill"
            labeled.append((x[index], label))
        return labeled, summary

    def audit_remaining_budget(
        self,
        eval_fn: Callable[[Dict[str, Any]], Evaluation],
        total_budget: int,
    ) -> Dict[str, int]:
        remaining = total_budget - len(self.real_observations)
        try:
            candidates, summary = self.select_audit_candidates(remaining)
        except RuntimeError:
            candidates = []
            summary = {
                "history_unique": 0,
                "predicted_pareto": 0,
                "predicted_fom_top10": 0,
                "selected": 0,
                "selected_fom_top10": 0,
                "selected_pareto": 0,
            }
        for candidate, source in candidates:
            if len(self.real_observations) >= total_budget:
                break
            if self._is_unseen(candidate):
                self.evaluate_real(candidate, eval_fn, source)

        # Extremely small/quantized spaces can collapse history designs. Fill any
        # residual budget with unseen random designs so the SPICE budget is exact.
        attempts = 0
        while len(self.real_observations) < total_budget and attempts < 10000:
            attempts += 1
            candidate = self.rng.uniform(self.lower, self.upper)
            if self._is_unseen(candidate):
                self.evaluate_real(candidate, eval_fn, "final-budget-fill")
        if len(self.real_observations) != total_budget:
            raise RuntimeError(
                f"CTR used {len(self.real_observations)} SPICE evaluations; "
                f"expected exactly {total_budget}"
            )
        return summary

    def final_result(
        self,
        context_attempt_budget: int,
        audit_summary: Dict[str, int],
        *,
        mode: str,
    ) -> Dict[str, Any]:
        valid = self.successful_observations
        if not valid:
            training_attempts = sum(
                obs.source.startswith("train-") for obs in self.real_observations
            )
            return {
                "algorithm": "Online-CTR",
                "params": None,
                "merit": None,
                "fom": None,
                "result": None,
                "iteration": len(self.real_observations),
                "n_evaluations": len(self.real_observations),
                "circuit_evaluations": len(self.real_observations),
                "successful_evaluations": 0,
                "tabpfn_queries": self.surrogate_queries,
                "context_attempt_budget": context_attempt_budget,
                "training_target": context_attempt_budget,
                "training_attempts": training_attempts,
                "training_samples": 0,
                "final_audit_evaluations": len(self.real_observations) - training_attempts,
                "tabpfn_fit_count": self.fit_count,
                "coarse_queries": len(self.coarse_history_x),
                "refine_queries": len(self.refine_history_x),
                "audit_selection": audit_summary,
                "history": [],
                "pareto_solutions": [],
                "num_solutions": 0,
                "is_multiobjective": True,
                "mode": mode,
            }

        values = np.vstack([obs.objectives for obs in valid])
        front_indices = np.flatnonzero(is_pareto_mask(values))
        best = max(valid, key=lambda obs: obs.merit)
        pareto_solutions: List[Dict[str, Any]] = []
        for index in front_indices:
            obs = valid[int(index)]
            pareto_solutions.append(
                {
                    "params": obs.params,
                    "result": obs.result,
                    "merit": obs.merit,
                    "min_snm": -float(obs.objectives[0]),
                    "max_power": float(obs.objectives[1]),
                    "area": float(obs.objectives[2]),
                    "max_delay": float(obs.objectives[3]),
                    "source": obs.source,
                }
            )

        history = [
            {
                "iteration": i,
                "source": obs.source,
                "success": obs.success,
                "objectives": obs.objectives.tolist(),
                "constraints": list(obs.constraints),
                "merit": obs.merit if obs.success else None,
                "params": obs.params,
                "result": obs.result,
                "elapsed": obs.elapsed,
            }
            for i, obs in enumerate(self.real_observations, start=1)
        ]
        training_attempts = sum(obs.source.startswith("train-") for obs in self.real_observations)
        final_audits = len(self.real_observations) - training_attempts
        return {
            "algorithm": "Online-CTR",
            "params": best.params,
            "merit": best.merit,
            "fom": best.merit,
            "result": best.result,
            "iteration": len(self.real_observations),
            "n_evaluations": len(self.real_observations),
            "circuit_evaluations": len(self.real_observations),
            "successful_evaluations": len(valid),
            "mode": mode,
            "context_attempt_budget": context_attempt_budget,
            "training_target": context_attempt_budget,
            "training_attempts": training_attempts,
            "training_samples": sum(
                obs.success and obs.source.startswith("train-") for obs in self.real_observations
            ),
            "final_audit_evaluations": final_audits,
            "tabpfn_queries": self.surrogate_queries,
            "tabpfn_fit_count": self.fit_count,
            "coarse_queries": len(self.coarse_history_x),
            "refine_queries": len(self.refine_history_x),
            "audit_selection": audit_summary,
            "history": history,
            "pareto_solutions": pareto_solutions,
            "num_solutions": len(pareto_solutions),
            "is_multiobjective": True,
        }

    def run(
        self,
        eval_fn: Callable[[Dict[str, Any]], Evaluation],
        total_budget: int,
        *,
        initial_x: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        context_attempt_budget = resolve_training_target(
            total_budget,
            self.requested_training_evals,
            target_context_size=self.target_context_size,
            min_context_for_ctr=self.min_context_for_ctr,
            min_final_audits=self.min_final_audits,
        )
        full_ctr_threshold = self.min_context_for_ctr + self.min_final_audits
        acquisition_only = int(total_budget) < full_ctr_threshold
        audit_budget = int(total_budget) - context_attempt_budget
        print(
            f"CTR budget: SPICE total={total_budget}, context attempts="
            f"{context_attempt_budget}, final audits={audit_budget}, "
            f"mode={'acquisition-only' if acquisition_only else 'full-ctr'}"
        )
        self.collect_training_data(
            eval_fn,
            context_attempt_budget,
            initial_x=initial_x,
        )

        empty_audit_summary = {
            "history_unique": 0,
            "predicted_pareto": 0,
            "predicted_fom_top10": 0,
            "selected": 0,
            "selected_fom_top10": 0,
            "selected_pareto": 0,
        }
        if acquisition_only:
            return self.final_result(
                context_attempt_budget,
                empty_audit_summary,
                mode="acquisition-only",
            )

        # If too few simulations were valid to fit a surrogate, preserve the
        # exact real-SPICE budget with unseen random audits and return real data.
        if self.manager is None:
            attempts = 0
            while len(self.real_observations) < total_budget and attempts < 10000:
                attempts += 1
                candidate = self.rng.uniform(self.lower, self.upper)
                if self._is_unseen(candidate):
                    self.evaluate_real(candidate, eval_fn, "final-budget-fill")
            if len(self.real_observations) != total_budget:
                raise RuntimeError(
                    f"CTR used {len(self.real_observations)} SPICE evaluations; "
                    f"expected exactly {total_budget}"
                )
            return self.final_result(
                context_attempt_budget,
                empty_audit_summary,
                mode="surrogate-unavailable",
            )

        coarse_x, coarse_y = self.run_coarse()
        self.refine(coarse_x, coarse_y)
        audit_summary = self.audit_remaining_budget(eval_fn, total_budget)
        return self.final_result(
            context_attempt_budget,
            audit_summary,
            mode="full-ctr",
        )


def _load_ctr_config(path: Optional[str] = None) -> Dict[str, Any]:
    import yaml  # pylint: disable=import-outside-toplevel

    config_path = path or os.path.join(
        current_dir, "online_ctr", "configs", "online.yaml"
    )
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("CTR configuration must be a YAML mapping")
    flat: Dict[str, Any] = {}
    for value in loaded.values():
        if isinstance(value, dict):
            flat.update(value)
    flat.update({key: value for key, value in loaded.items() if not isinstance(value, dict)})
    return flat


def main(config_path="config_sram.yaml", problem=None, max_iter=400, **kwargs):
    """Run online TabPFN CTR using exactly ``max_iter`` real SPICE calls."""
    if kwargs.get("check_only", False):
        ctr_config = _load_ctr_config(kwargs.get("ctr_config_path"))
        if problem is None:
            parameter_space = _project_utils()["space"](config_path)
        else:
            parameter_space = problem[0]
        lower, upper = _numpy_bounds(parameter_space)
        report = {
            "algorithm": "Online-CTR",
            "status": "ready",
            "dimension": int(len(lower)),
            "bounds_finite": bool(np.isfinite(lower).all() and np.isfinite(upper).all()),
            "config_loaded": bool(ctr_config),
            "config_path": str(
                kwargs.get("ctr_config_path")
                or os.path.join(current_dir, "online_ctr", "configs", "online.yaml")
            ),
            "data_source": "online SPICE context (no bundled CSV)",
        }
        print("Online-CTR check passed: " + ", ".join(
            f"{key}={value}" for key, value in report.items() if key != "algorithm"
        ))
        return report

    print("===== SRAM optimization using online TabPFN Coarse-to-Refine (Online-CTR) =====")
    try:
        _project_utils()["create_directories"]()
    except Exception:
        if problem is None:
            raise

    metadata: Dict[str, Any] = {}
    initial_x = kwargs.get("initial_x")
    if problem is not None:
        try:
            parameter_space, eval_fn, raw_metadata = problem
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata
        except (TypeError, ValueError) as exc:
            raise ValueError("problem must be (parameter_space, eval_fn, metadata)") from exc
    else:
        utils = _project_utils()
        parameter_space = utils["space"](config_path)
        eval_fn = utils["evaluate"]
        try:
            initial_x = utils["default_vector"](parameter_space)
        except Exception:
            initial_x = None

    if initial_x is None:
        initial_x = metadata.get("initial_x")

    ctr_config = _load_ctr_config(kwargs.get("ctr_config_path"))

    def option(name: str, default: Any) -> Any:
        value = kwargs.get(name)
        return value if value is not None else ctr_config.get(name, default)

    seed = int(option("seed", 42))
    np.random.seed(seed)
    optimizer = OnlineTabPFNCTR(
        parameter_space,
        seed=seed,
        device=str(option("device", "auto")),
        gpu_index=int(option("gpu_index", 1)),
        training_evals=option("training_evals", None),
        target_context_size=int(option("target_context_size", TARGET_CONTEXT_SIZE)),
        min_context_for_ctr=int(option("min_context_for_ctr", MIN_CONTEXT_FOR_CTR)),
        min_final_audits=int(option("min_final_audits", MIN_FINAL_AUDITS)),
        initial_lhs_size=int(option("initial_lhs_size", INITIAL_LHS_SIZE)),
        active_batch_size=int(option("active_batch_size", 10)),
        active_candidates=int(option("active_candidates", 2000)),
        pareto_bonus=float(option("pareto_bonus", 2.0)),
        novelty_weight=float(option("novelty_weight", 0.25)),
        coarse_evals=int(option("coarse_evals", 1500)),
        coarse_pop_size=int(option("coarse_pop_size", 50)),
        representative_points=int(option("representative_points", 20)),
        refine_steps=int(option("refine_steps", 80)),
        refine_lr=float(option("refine_lr", 0.05)),
        refine_smoothing=float(option("refine_smoothing", 0.05)),
        physical_penalty=float(option("physical_penalty", 10.0)),
        gradient_clip=float(option("gradient_clip", 5.0)),
        fom_top_k=int(option("fom_top_k", 10)),
        hv_reference_margin=float(option("hv_reference_margin", 0.10)),
    )
    result = optimizer.run(eval_fn, int(max_iter), initial_x=initial_x)

    print("\n===== Online-CTR Final Real-SPICE Result =====")
    print(
        f"SPICE evaluations={result['circuit_evaluations']}, "
        f"training samples={result.get('training_samples')}, "
        f"final audits={result.get('final_audit_evaluations')}, "
        f"TabPFN queries={result['tabpfn_queries']}"
    )
    if result["result"] is None:
        print("No valid real-SPICE solution found")
    else:
        print(f"Best real FoM: {result['merit']:.6e}")
        print(f"Real Pareto front size: {result['num_solutions']}")
    return result


if __name__ == "__main__":
    utilities = _project_utils()
    utilities["seed_set"](42)
    main(max_iter=400, seed=42)
