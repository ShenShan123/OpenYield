"""Unified estimators that consume only ``SimulationRunner``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
import socket
import time

import numpy as np

from .distributions import GaussianDistribution, GaussianMixture
from .results import EstimationResult, MultiConditionEstimationResult
from .simulation import SimulationBatch, SimulationRunner


def _logaddexp_mixture(log_a: np.ndarray, log_b: np.ndarray, ratio_b: float) -> np.ndarray:
    if ratio_b <= 0:
        return log_a
    if ratio_b >= 1:
        return log_b
    return np.logaddexp(np.log1p(-ratio_b) + log_a, np.log(ratio_b) + log_b)


def _farthest_centers(points: np.ndarray, count: int, reference: np.ndarray) -> np.ndarray:
    """Deterministic farthest-point clustering without a sklearn dependency."""
    if points.shape[0] <= count:
        return points.copy()
    distances = np.linalg.norm(points - reference[None, :], axis=1)
    selected = [int(np.argmin(distances))]
    nearest = np.linalg.norm(points - points[selected[0]][None, :], axis=1)
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[index][None, :], axis=1))
    return points[np.asarray(selected)]


class BaseEstimator:
    algorithm = "BASE"

    def __init__(
        self,
        *,
        runner: SimulationRunner,
        distribution: GaussianDistribution,
        threshold: float,
        seed: int = 0,
        failure_direction: str = "greater",
        operation: str = "read",
        target_row: int = 0,
        target_col: int = 0,
        temperature: float = 27,
        **config: Any,
    ) -> None:
        if failure_direction not in {"greater", "less"}:
            raise ValueError("failure_direction must be 'greater' or 'less'")
        self.runner = runner
        self.distribution = distribution
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.failure_direction = failure_direction
        self.operation = operation
        self.target_row = int(target_row)
        self.target_col = int(target_col)
        self.temperature = float(temperature)
        self.config = dict(config)
        self.result: Optional[Union[EstimationResult, MultiConditionEstimationResult]] = None
        self._artifacts: list[str] = []

    def _failed(self, values: np.ndarray) -> np.ndarray:
        failed = (
            values > self.threshold
            if self.failure_direction == "greater"
            else values < self.threshold
        )
        if bool(self.config.get("failure_if_nonpositive", False)):
            failed = failed | (values <= 0.0)
        return failed

    def _simulate(self, samples: np.ndarray, run_name: str) -> SimulationBatch:
        batch = self.runner.run_mc_simulation(
            operation=self.operation,
            target_row=self.target_row,
            target_col=self.target_col,
            mc_runs=samples.shape[0],
            temperature=self.temperature,
            vars=samples,
            run_name=run_name,
        )
        self._artifacts.extend(batch.artifacts)
        return batch

    def _simulate_many(self, samples: np.ndarray, run_name: str) -> SimulationBatch:
        """Evaluate a logical sample array in isolated native-sized batches."""
        batch_size = max(1, int(self.config.get("batch_size", 1000)))
        values: list[np.ndarray] = []
        statuses: list[str] = []
        errors: list[Optional[str]] = []
        artifacts: list[str] = []
        start_charged = self.runner.ledger.charged_calls
        start_live = self.runner.ledger.live_calls
        start_retry = self.runner.ledger.retry_calls
        offset = 0
        part = 0
        while offset < samples.shape[0] and self.runner.ledger.remaining > 0:
            count = min(batch_size, samples.shape[0] - offset, self.runner.ledger.remaining)
            batch = self._simulate(samples[offset:offset + count], f"{run_name}_{part:05d}")
            values.append(batch.values)
            statuses.extend(batch.statuses)
            errors.extend(batch.errors)
            artifacts.extend(batch.artifacts)
            offset += count
            part += 1
        if offset < samples.shape[0]:
            missing = samples.shape[0] - offset
            values.append(np.full(missing, np.nan, dtype=float))
            statuses.extend("simulator_error" for _ in range(missing))
            errors.extend("charged budget consumed by retries" for _ in range(missing))
        return SimulationBatch(
            values=np.concatenate(values),
            statuses=tuple(statuses),
            errors=tuple(errors),
            charged_calls=self.runner.ledger.charged_calls - start_charged,
            live_calls=self.runner.ledger.live_calls - start_live,
            retry_calls=self.runner.ledger.retry_calls - start_retry,
            run_name=run_name,
            artifacts=tuple(artifacts),
        )

    def start_estimate(self, max_num: int = 5000):
        self.runner.reset_budget(max_num)
        self._artifacts = []
        self.result = self._estimate(int(max_num))
        return self.result

    def run(self, max_num: int = 5000):
        return self.start_estimate(max_num=max_num)

    def _result(
        self,
        *,
        start: float,
        max_num: int,
        contributions: np.ndarray,
    ) -> EstimationResult:
        ledger = self.runner.ledger
        if ledger is None:
            raise RuntimeError("budget ledger was not initialized")
        finite = np.isfinite(contributions)
        usable = contributions[finite]
        if usable.size:
            probability = float(np.clip(usable.mean(), 0.0, 1.0))
            standard_error = float(usable.std(ddof=1) / np.sqrt(usable.size)) if usable.size > 1 else 0.0
        else:
            probability = 0.0
            standard_error = 0.0
        if ledger.simulator_errors:
            status = "simulator_failure"
        elif probability == 0.0:
            status = "ok_zero_failure"
        else:
            status = "ok"
        metadata = {
            "threshold": self.threshold,
            "failure_direction": self.failure_direction,
            "operation": self.operation,
            "target_row": self.target_row,
            "target_col": self.target_col,
            "temperature": self.temperature,
            "node": socket.gethostname(),
        }
        supplied_metadata = self.config.get("metadata")
        if isinstance(supplied_metadata, dict):
            metadata.update(supplied_metadata)
        return EstimationResult(
            algorithm=self.algorithm,
            status=status,
            failure_probability=probability,
            yield_probability=1.0 - probability,
            standard_error=standard_error,
            budget_target=max_num,
            charged_calls=ledger.charged_calls,
            live_calls=ledger.live_calls,
            retry_calls=ledger.retry_calls,
            simulator_errors=ledger.simulator_errors,
            samples_used=int(usable.size),
            elapsed_seconds=time.perf_counter() - start,
            seed=self.seed,
            artifacts=tuple(self._artifacts),
            metadata=metadata,
        )

    def _estimate(self, max_num: int) -> EstimationResult:
        raise NotImplementedError


class MC(BaseEstimator):
    algorithm = "MC"

    def _estimate(self, max_num: int) -> EstimationResult:
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        batch_size = max(1, int(self.config.get("batch_size", min(1000, max_num))))
        contribution_parts: list[np.ndarray] = []
        batch_index = 0
        while self.runner.ledger.remaining > 0:
            count = min(batch_size, self.runner.ledger.remaining)
            samples = self.distribution.sample(count, rng)
            batch = self._simulate(samples, f"mc_{batch_index:05d}")
            valid = batch.valid_mask
            contributions = np.full(count, np.nan, dtype=float)
            contributions[valid] = self._failed(batch.values[valid]).astype(float)
            contribution_parts.append(contributions)
            batch_index += 1
        return self._result(
            start=start,
            max_num=max_num,
            contributions=np.concatenate(contribution_parts),
        )


class AdaptiveImportanceEstimator(BaseEstimator):
    strategy = "all_failures"

    def _select_centers(
        self,
        pilot_samples: np.ndarray,
        pilot_values: np.ndarray,
        pilot_valid: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        valid_samples = pilot_samples[pilot_valid]
        valid_values = pilot_values[pilot_valid]
        failed = self._failed(valid_values)
        failures = valid_samples[failed]
        max_components = int(self.config.get("max_components", 64))
        if failures.size == 0:
            if valid_samples.size == 0:
                return self.distribution.mean.reshape(1, -1), None
            extreme = int(np.argmax(valid_values) if self.failure_direction == "greater" else np.argmin(valid_values))
            return valid_samples[extreme].reshape(1, -1), None

        if self.strategy in {"minimum_norm", "opt"}:
            standardized = self.distribution.standardize(failures)
            center = failures[int(np.argmin(np.linalg.norm(standardized, axis=1)))]
            blend = float(self.config.get("center_blend", 1.0 if self.strategy == "opt" else 0.6))
            center = blend * center + (1.0 - blend) * self.distribution.mean
            return center.reshape(1, -1), None
        if self.strategy == "clustered":
            count = min(max_components, max(1, int(np.sqrt(failures.shape[0]))))
            return _farthest_centers(failures, count, self.distribution.mean), None
        if self.strategy == "directional_clustered":
            standardized = self.distribution.standardize(failures)
            norms = np.linalg.norm(standardized, axis=1)
            directions = standardized / np.maximum(norms[:, None], np.finfo(float).tiny)
            count = min(max_components, max(1, int(np.sqrt(failures.shape[0]))))
            direction_centers = _farthest_centers(
                directions, count, np.zeros(directions.shape[1], dtype=float)
            )
            labels = np.argmax(directions @ direction_centers.T, axis=1)
            selected: list[int] = []
            cluster_weights: list[float] = []
            for label in range(direction_centers.shape[0]):
                members = np.flatnonzero(labels == label)
                if members.size == 0:
                    continue
                selected.append(int(members[np.argmin(norms[members])]))
                cluster_weights.append(float(members.size))
            weights = np.asarray(cluster_weights, dtype=float)
            weights /= weights.sum()
            return failures[np.asarray(selected)], weights
        if self.strategy == "boundary_weighted":
            standardized = self.distribution.standardize(failures)
            order = np.argsort(np.linalg.norm(standardized, axis=1))
            failures = failures[order[:max_components]]
            log_weights = self.distribution.log_pdf(failures)
            weights = np.exp(log_weights - np.max(log_weights))
            return failures, weights / weights.sum()
        if failures.shape[0] > max_components:
            standardized = self.distribution.standardize(failures)
            order = np.argsort(np.linalg.norm(standardized, axis=1))
            indexes = np.linspace(0, order.size - 1, max_components, dtype=int)
            failures = failures[order[indexes]]
        if self.strategy == "density_weighted":
            log_weights = self.distribution.log_pdf(failures)
            weights = np.exp(log_weights - np.max(log_weights))
            return failures, weights / weights.sum()
        return failures, None

    def _estimate(self, max_num: int) -> EstimationResult:
        if max_num < 2:
            raise ValueError(f"{self.algorithm} requires max_num >= 2")
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        pilot_fraction = float(self.config.get("pilot_fraction", 0.4))
        pilot_count = min(max_num - 1, max(1, int(round(max_num * pilot_fraction))))
        is_count = max_num - pilot_count

        pilot_samples = self.distribution.sample(pilot_count, rng)
        pilot_batch = self._simulate_many(pilot_samples, f"{self.algorithm.lower()}_pilot")
        is_count = self.runner.ledger.remaining
        if is_count <= 0:
            contributions = np.full(pilot_count, np.nan, dtype=float)
            contributions[pilot_batch.valid_mask] = self._failed(
                pilot_batch.values[pilot_batch.valid_mask]
            ).astype(float)
            return self._result(start=start, max_num=max_num, contributions=contributions)
        pilot_valid = pilot_batch.valid_mask
        centers, center_weights = self._select_centers(
            pilot_samples, pilot_batch.values, pilot_valid
        )
        proposal_scale = float(self.config.get("proposal_scale", 1.0))
        base_variance = np.diag(self.distribution.covariance) * proposal_scale
        proposal = GaussianMixture(centers, base_variance, center_weights)
        defensive_ratio = float(self.config.get("defensive_ratio", 0.1))
        if not 0.0 < defensive_ratio < 1.0:
            raise ValueError("defensive_ratio must be between 0 and 1")
        defensive_count = max(1, min(is_count - 1, int(round(is_count * defensive_ratio)))) if is_count > 1 else 1
        proposal_count = is_count - defensive_count
        proposal_samples = proposal.sample(proposal_count, rng)
        defensive_samples = self.distribution.sample(defensive_count, rng)
        is_samples = np.vstack([proposal_samples, defensive_samples])
        order = rng.permutation(is_samples.shape[0])
        is_samples = is_samples[order]
        is_batch = self._simulate_many(is_samples, f"{self.algorithm.lower()}_is")

        pilot_contribution = np.full(pilot_count, np.nan, dtype=float)
        pilot_contribution[pilot_valid] = self._failed(pilot_batch.values[pilot_valid]).astype(float)
        is_valid = is_batch.valid_mask
        is_contribution = np.full(is_count, np.nan, dtype=float)
        log_f = self.distribution.log_pdf(is_samples)
        log_g = proposal.log_pdf(is_samples)
        log_q = _logaddexp_mixture(log_g, log_f, defensive_count / is_count)
        weights = np.exp(np.clip(log_f - log_q, -745.0, 700.0))
        is_failed = np.zeros(is_count, dtype=bool)
        is_failed[is_valid] = self._failed(is_batch.values[is_valid])
        is_contribution[is_valid] = weights[is_valid] * is_failed[is_valid].astype(float)
        contributions = np.concatenate([pilot_contribution, is_contribution])
        result = self._result(
            start=start,
            max_num=max_num,
            contributions=contributions,
        )
        metadata = dict(result.metadata)
        metadata.update(
            pilot_samples=pilot_count,
            importance_samples=is_count,
            proposal_components=int(centers.shape[0]),
            proposal_strategy=self.strategy,
            defensive_ratio=defensive_count / is_count,
        )
        return EstimationResult(**{**result.__dict__, "metadata": metadata})


class MNIS(AdaptiveImportanceEstimator):
    algorithm = "MNIS"
    strategy = "minimum_norm"


class AIS(AdaptiveImportanceEstimator):
    algorithm = "AIS"
    strategy = "all_failures"


class ACS(AdaptiveImportanceEstimator):
    algorithm = "ACS"
    strategy = "clustered"


class HSCS(AdaptiveImportanceEstimator):
    algorithm = "HSCS"
    strategy = "directional_clustered"


class EFIAL(AdaptiveImportanceEstimator):
    algorithm = "EFIAL"
    strategy = "density_weighted"


class FUSIS(AdaptiveImportanceEstimator):
    algorithm = "FUSIS"
    strategy = "boundary_weighted"


class OPT(AdaptiveImportanceEstimator):
    algorithm = "OPT"
    strategy = "opt"


class BIBD(BaseEstimator):
    algorithm = "BIBD"

    def __init__(self, *, conditions: Optional[list[dict[str, Any]]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.conditions = conditions or [{"name": "default"}]

    def start_estimate(self, max_num: int = 5000) -> MultiConditionEstimationResult:
        start = time.perf_counter()
        if max_num < len(self.conditions):
            raise ValueError("BIBD budget must cover every condition")
        per_condition = [max_num // len(self.conditions)] * len(self.conditions)
        for index in range(max_num % len(self.conditions)):
            per_condition[index] += 1
        results: dict[str, EstimationResult] = {}
        charged = 0
        artifacts: list[str] = []
        for index, (condition, budget) in enumerate(zip(self.conditions, per_condition)):
            condition_runner = SimulationRunner(
                self.runner.model,
                self.runner.simulation_root / f"condition_{index:02d}",
                metric=self.runner.metric,
                input_space=self.runner.input_space,
                nominal=self.runner.nominal,
                sigma=self.runner.sigma,
                max_retries=self.runner.max_retries,
                quiet=self.runner.quiet,
            )
            condition_config = dict(self.config)
            condition_config.pop("conditions", None)
            estimator = MC(
                runner=condition_runner,
                distribution=self.distribution,
                threshold=float(condition.get("threshold", self.threshold)),
                seed=self.seed + index,
                failure_direction=str(condition.get("failure_direction", self.failure_direction)),
                operation=str(condition.get("operation", self.operation)),
                target_row=int(condition.get("target_row", self.target_row)),
                target_col=int(condition.get("target_col", self.target_col)),
                temperature=float(condition.get("temperature", self.temperature)),
                **condition_config,
            )
            result = estimator.start_estimate(budget)
            name = str(condition.get("name", f"condition_{index}"))
            results[name] = result
            charged += result.charged_calls
            artifacts.extend(result.artifacts)
        status = "ok" if all(result.status in {"ok", "ok_zero_failure"} for result in results.values()) else "simulator_failure"
        self.result = MultiConditionEstimationResult(
            algorithm=self.algorithm,
            status=status,
            budget_target=max_num,
            charged_calls=charged,
            elapsed_seconds=time.perf_counter() - start,
            conditions=results,
            artifacts=tuple(artifacts),
        )
        return self.result


ESTIMATOR_REGISTRY = {
    estimator.algorithm: estimator
    for estimator in (MC, MNIS, AIS, ACS, HSCS, EFIAL, FUSIS, OPT, BIBD)
}


STABLE_ALGORITHMS = ("MC", "MNIS", "AIS", "ACS", "HSCS", "EFIAL")
EXPERIMENTAL_ALGORITHMS = ("FUSIS", "OPT", "BIBD")
