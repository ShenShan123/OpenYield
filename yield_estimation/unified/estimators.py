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


class CrossTopoImportanceEstimator(BaseEstimator):
    """Shared runner-only mechanics for the CrossTopo baseline ports.

    The donor implementations mixed algorithm logic with cached failure files,
    simulator-directory mutation, and convergence-based stopping.  This class
    retains their proposal and adaptation rules while delegating every circuit
    evaluation and every charged-budget decision to ``SimulationRunner``.
    """

    donor_repository = "IceLab-JCIE/EDA26-Yield-Array-Transfer-Nanlin"

    def _pilot(self, max_num: int, rng: np.random.Generator) -> tuple[np.ndarray, SimulationBatch]:
        pilot_fraction = float(self.config.get("pilot_fraction", 0.4))
        if not 0.0 < pilot_fraction < 1.0:
            raise ValueError("pilot_fraction must be between 0 and 1")
        count = min(max_num - 1, max(1, int(round(max_num * pilot_fraction))))
        samples = self.distribution.sample(count, rng)
        return samples, self._simulate_many(samples, f"{self.algorithm.lower()}_pilot")

    def _proposal_variance(self, component_count: int = 1) -> np.ndarray:
        scale = float(
            self.config.get(
                "proposal_scale",
                self.config.get("g_cal_val", self.config.get("g_sam_val", 1.0)),
            )
        )
        if scale <= 0:
            raise ValueError("proposal_scale/g_cal_val must be positive")
        diagonal = np.diag(self.distribution.covariance) * scale
        return np.tile(diagonal, (component_count, 1))

    def _pilot_parts(
        self, samples: np.ndarray, batch: SimulationBatch
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        valid = batch.valid_mask
        valid_samples = samples[valid]
        valid_values = batch.values[valid]
        failed = self._failed(valid_values)
        contribution = np.full(samples.shape[0], np.nan, dtype=float)
        contribution[valid] = failed.astype(float)
        return valid_samples, valid_values, failed, contribution

    def _fallback_anchor(
        self, valid_samples: np.ndarray, valid_values: np.ndarray
    ) -> np.ndarray:
        if valid_samples.shape[0] == 0:
            return self.distribution.mean.reshape(1, -1)
        index = int(
            np.argmax(valid_values)
            if self.failure_direction == "greater"
            else np.argmin(valid_values)
        )
        return valid_samples[index].reshape(1, -1)

    def _draw_importance(
        self,
        proposal: GaussianMixture,
        count: int,
        rng: np.random.Generator,
        defensive_ratio: float,
    ) -> tuple[np.ndarray, float]:
        if not 0.0 <= defensive_ratio < 1.0:
            raise ValueError("defensive_ratio must be in [0, 1)")
        defensive_count = int(round(count * defensive_ratio))
        if defensive_ratio > 0 and count > 1:
            defensive_count = max(1, min(count - 1, defensive_count))
        proposal_count = count - defensive_count
        samples = np.vstack([
            proposal.sample(proposal_count, rng),
            self.distribution.sample(defensive_count, rng),
        ])
        return samples, defensive_count / count

    def _importance_contribution(
        self,
        samples: np.ndarray,
        batch: SimulationBatch,
        proposal: GaussianMixture,
        defensive_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_f = self.distribution.log_pdf(samples)
        log_g = proposal.log_pdf(samples)
        log_q = _logaddexp_mixture(log_g, log_f, defensive_ratio)
        log_weight = np.clip(log_f - log_q, -745.0, 700.0)
        valid = batch.valid_mask
        failed = np.zeros(samples.shape[0], dtype=bool)
        failed[valid] = self._failed(batch.values[valid])
        contribution = np.full(samples.shape[0], np.nan, dtype=float)
        contribution[valid] = np.exp(log_weight[valid]) * failed[valid]
        return contribution, failed, log_weight

    def _finish(
        self,
        *,
        start: float,
        max_num: int,
        contribution_parts: list[np.ndarray],
        **metadata: Any,
    ) -> EstimationResult:
        result = self._result(
            start=start,
            max_num=max_num,
            contributions=np.concatenate(contribution_parts),
        )
        combined = dict(result.metadata)
        combined.update(
            implementation_source=self.donor_repository,
            implementation_family="CrossTopo baseline port",
            **metadata,
        )
        return EstimationResult(**{**result.__dict__, "metadata": combined})


class MNIS(CrossTopoImportanceEstimator):
    algorithm = "MNIS"

    def _estimate(self, max_num: int) -> EstimationResult:
        if max_num < 2:
            raise ValueError("MNIS requires max_num >= 2")
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        pilot_samples, pilot_batch = self._pilot(max_num, rng)
        valid_samples, valid_values, failed, pilot_contribution = self._pilot_parts(
            pilot_samples, pilot_batch
        )
        failures = valid_samples[failed]
        if failures.shape[0]:
            standardized = self.distribution.standardize(failures)
            anchor = failures[int(np.argmin(np.linalg.norm(standardized, axis=1)))]
        else:
            anchor = self._fallback_anchor(valid_samples, valid_values)[0]
        # CrossTopo MNIS moves the minimum-norm failure 40% away from nominal.
        center_blend = float(self.config.get("center_blend", 0.4))
        center = (
            center_blend * anchor
            + (1.0 - center_blend) * self.distribution.mean
        ).reshape(1, -1)
        proposal = GaussianMixture(center, self._proposal_variance())
        remaining = self.runner.ledger.remaining
        contribution_parts = [pilot_contribution]
        actual_ratio = 0.0
        if remaining:
            samples, actual_ratio = self._draw_importance(
                proposal,
                remaining,
                rng,
                float(self.config.get("defensive_ratio", 0.2)),
            )
            batch = self._simulate_many(samples, "mnis_is")
            contribution, _, _ = self._importance_contribution(
                samples, batch, proposal, actual_ratio
            )
            contribution_parts.append(contribution)
        return self._finish(
            start=start,
            max_num=max_num,
            contribution_parts=contribution_parts,
            proposal_rule="minimum_norm_failure",
            pilot_failures=int(failures.shape[0]),
            center_blend=center_blend,
            defensive_ratio=actual_ratio,
            proposal_components=1,
        )


class AIS(CrossTopoImportanceEstimator):
    algorithm = "AIS"

    def _estimate(self, max_num: int) -> EstimationResult:
        if max_num < 2:
            raise ValueError("AIS requires max_num >= 2")
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        pilot_samples, pilot_batch = self._pilot(max_num, rng)
        valid_samples, valid_values, failed, pilot_contribution = self._pilot_parts(
            pilot_samples, pilot_batch
        )
        anchors = valid_samples[failed]
        if anchors.shape[0] == 0:
            anchors = self._fallback_anchor(valid_samples, valid_values)
        anchor_limit = int(self.config.get("initial_failed_data_num", 100))
        anchors = anchors[:max(1, anchor_limit)]
        contribution_parts = [pilot_contribution]
        defensive_ratio = float(self.config.get("defensive_ratio", 0.1))
        target_hit_rate = float(self.config.get("target_hit_rate", 0.3))
        round_size = max(1, int(self.config.get("is_batch_size", anchor_limit)))
        rounds = 0

        while self.runner.ledger.remaining:
            proposal = GaussianMixture(
                anchors,
                self._proposal_variance(anchors.shape[0]),
            )
            count = min(round_size, self.runner.ledger.remaining)
            samples, actual_ratio = self._draw_importance(
                proposal, count, rng, defensive_ratio
            )
            batch = self._simulate_many(samples, f"ais_is_{rounds:05d}")
            contribution, is_failed, log_weight = self._importance_contribution(
                samples, batch, proposal, actual_ratio
            )
            contribution_parts.append(contribution)

            finite = np.isfinite(log_weight)
            if finite.any():
                shifted = np.exp(log_weight[finite] - np.max(log_weight[finite]))
                ess_ratio = float(
                    shifted.sum() ** 2
                    / max(np.square(shifted).sum() * shifted.size, np.finfo(float).tiny)
                )
            else:
                ess_ratio = 0.0
            proposal_count = count - int(round(count * actual_ratio))
            hit_rate = float(is_failed[:proposal_count].mean()) if proposal_count else 0.0
            failed_weights = log_weight[is_failed & finite]
            if failed_weights.size:
                scaled = np.exp(failed_weights - np.max(failed_weights))
                max_share = float(scaled.max() / scaled.sum())
            else:
                max_share = 0.0
            update = (1.0 / (ess_ratio + 1e-6)) * np.sqrt(
                (target_hit_rate + 0.01) / (hit_rate + 0.01)
            )
            if max_share > 0.3:
                update *= 1.5
            defensive_ratio = float(np.clip(defensive_ratio * np.clip(update, 0.8, 1.25), 0.01, 0.95))

            new_anchors = samples[is_failed & batch.valid_mask]
            if new_anchors.shape[0]:
                candidate = np.vstack([anchors, new_anchors])
                density_order = np.argsort(self.distribution.log_pdf(candidate))[::-1]
                anchors = candidate[density_order[:anchor_limit]]
            rounds += 1

        return self._finish(
            start=start,
            max_num=max_num,
            contribution_parts=contribution_parts,
            proposal_rule="adaptive_failure_gmm",
            pilot_failures=int(failed.sum()),
            proposal_components=int(anchors.shape[0]),
            adaptation_rounds=rounds,
            final_defensive_ratio=defensive_ratio,
        )


class ACS(CrossTopoImportanceEstimator):
    algorithm = "ACS"

    def __init__(self, *, mode: str = "original", **kwargs: Any) -> None:
        if mode not in {"original", "improved"}:
            raise ValueError("ACS mode must be 'original' or 'improved'")
        super().__init__(**kwargs)
        self.mode = mode

    def _directional_groups(self, anchors: np.ndarray) -> tuple[np.ndarray, int]:
        standardized = self.distribution.standardize(anchors)
        norms = np.linalg.norm(standardized, axis=1)
        directions = standardized / np.maximum(norms[:, None], np.finfo(float).tiny)
        requested = self.config.get("cluster_num")
        count = (
            int(requested)
            if requested is not None
            else max(1, int(round(np.sqrt(anchors.shape[0]))))
        )
        count = min(count, anchors.shape[0], int(self.config.get("max_components", 64)))
        centers = _farthest_centers(directions, count, np.zeros(directions.shape[1]))
        return np.argmax(directions @ centers.T, axis=1), count

    def _proposal(self, anchors: np.ndarray) -> tuple[GaussianMixture, int]:
        labels, groups = self._directional_groups(anchors)
        log_density = self.distribution.log_pdf(anchors)
        raw = np.exp(log_density - np.max(log_density))
        weights = np.zeros_like(raw)
        for label in range(groups):
            members = labels == label
            if members.any():
                weights[members] = raw[members] / raw[members].sum() / groups
        if self.mode == "original":
            return GaussianMixture(
                anchors,
                self._proposal_variance(anchors.shape[0]),
                weights,
            ), groups

        compressed = []
        compressed_weights = []
        for label in range(groups):
            members = labels == label
            if not members.any():
                continue
            local = weights[members]
            local /= local.sum()
            compressed.append(np.sum(anchors[members] * local[:, None], axis=0))
            compressed_weights.append(float(weights[members].sum()))
        nominal_ratio = float(self.config.get("nominal_proposal_ratio", 0.2))
        if not 0.0 < nominal_ratio < 1.0:
            raise ValueError("nominal_proposal_ratio must be between 0 and 1")
        centers = np.vstack([self.distribution.mean, np.asarray(compressed)])
        component_weights = np.hstack([
            nominal_ratio,
            (1.0 - nominal_ratio)
            * np.asarray(compressed_weights)
            / np.sum(compressed_weights),
        ])
        variances = np.vstack([
            np.diag(self.distribution.covariance),
            self._proposal_variance(len(compressed)),
        ])
        return GaussianMixture(centers, variances, component_weights), groups

    def _estimate(self, max_num: int) -> EstimationResult:
        if max_num < 2:
            raise ValueError("ACS requires max_num >= 2")
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        pilot_samples, pilot_batch = self._pilot(max_num, rng)
        valid_samples, valid_values, failed, pilot_contribution = self._pilot_parts(
            pilot_samples, pilot_batch
        )
        failure_anchors = valid_samples[failed]
        if self.mode == "original":
            anchors = failure_anchors
        else:
            requested = max(1, int(self.config.get("initial_fail_num", 32)))
            if failure_anchors.shape[0] >= requested:
                anchors = failure_anchors[:requested]
            else:
                order = np.argsort(valid_values)
                if self.failure_direction == "greater":
                    order = order[::-1]
                anchors = valid_samples[order[:requested]]
        if anchors.shape[0] == 0:
            anchors = self._fallback_anchor(valid_samples, valid_values)

        contribution_parts = [pilot_contribution]
        round_size = max(
            1, int(self.config.get("is_batch_size", self.config.get("batch_size", 100)))
        )
        groups = 1
        rounds = 0
        while self.runner.ledger.remaining:
            proposal, groups = self._proposal(anchors)
            count = min(round_size, self.runner.ledger.remaining)
            samples = proposal.sample(count, rng)
            batch = self._simulate_many(samples, f"acs_{self.mode}_is_{rounds:05d}")
            contribution, is_failed, _ = self._importance_contribution(
                samples, batch, proposal, 0.0
            )
            contribution_parts.append(contribution)
            new_failures = samples[is_failed & batch.valid_mask]
            if new_failures.shape[0]:
                if self.mode == "original":
                    anchors = new_failures
                else:
                    anchors = np.vstack([anchors, new_failures])
                    limit = int(self.config.get("max_anchor_history", 256))
                    if anchors.shape[0] > limit:
                        anchors = anchors[-limit:]
            rounds += 1

        return self._finish(
            start=start,
            max_num=max_num,
            contribution_parts=contribution_parts,
            mode=self.mode,
            proposal_rule=(
                "paper_aligned_failure_components"
                if self.mode == "original"
                else "compressed_cones_with_nominal_component"
            ),
            pilot_failures=int(failure_anchors.shape[0]),
            proposal_components=(int(anchors.shape[0]) if self.mode == "original" else groups + 1),
            cone_groups=groups,
            adaptation_rounds=rounds,
        )


class HSCS(CrossTopoImportanceEstimator):
    algorithm = "HSCS"

    def _estimate(self, max_num: int) -> EstimationResult:
        if max_num < 2:
            raise ValueError("HSCS requires max_num >= 2")
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        pilot_samples, pilot_batch = self._pilot(max_num, rng)
        valid_samples, valid_values, failed, pilot_contribution = self._pilot_parts(
            pilot_samples, pilot_batch
        )
        failures = valid_samples[failed]
        if failures.shape[0] == 0:
            failures = self._fallback_anchor(valid_samples, valid_values)
        standardized = self.distribution.standardize(failures)
        norms = np.linalg.norm(standardized, axis=1)
        directions = standardized / np.maximum(norms[:, None], np.finfo(float).tiny)
        cluster_count = min(
            failures.shape[0],
            int(self.config.get("cluster_num", max(1, int(round(np.sqrt(failures.shape[0])))))),
            int(self.config.get("max_components", 32)),
        )
        direction_centers = _farthest_centers(
            directions, cluster_count, np.zeros(directions.shape[1])
        )
        labels = np.argmax(directions @ direction_centers.T, axis=1)
        centers = []
        weights = []
        for label in range(cluster_count):
            members = np.flatnonzero(labels == label)
            if members.size:
                selected = members[int(np.argmin(norms[members]))]
                centers.append(failures[selected])
                weights.append(float(members.size))
        centers_array = np.asarray(centers)
        weights_array = np.asarray(weights) / np.sum(weights)

        # HSCS keeps a target-distribution component in its proposal and uses
        # one minimum-radius representative for every discovered failure cone.
        defensive_ratio = float(self.config.get("defensive_ratio", self.config.get("ratio", 0.1)))
        proposal = GaussianMixture(
            centers_array,
            self._proposal_variance(centers_array.shape[0]),
            weights_array,
        )
        remaining = self.runner.ledger.remaining
        contribution_parts = [pilot_contribution]
        actual_ratio = 0.0
        if remaining:
            samples, actual_ratio = self._draw_importance(
                proposal, remaining, rng, defensive_ratio
            )
            batch = self._simulate_many(samples, "hscs_is")
            contribution, _, _ = self._importance_contribution(
                samples, batch, proposal, actual_ratio
            )
            contribution_parts.append(contribution)
        return self._finish(
            start=start,
            max_num=max_num,
            contribution_parts=contribution_parts,
            proposal_rule="minimum_radius_failure_per_directional_cone",
            pilot_failures=int(failed.sum()),
            cone_groups=int(centers_array.shape[0]),
            proposal_components=int(centers_array.shape[0]),
            defensive_ratio=actual_ratio,
        )


class EFIAL(CrossTopoImportanceEstimator):
    """CrossTopo EFIAL: iterative density-weighted failure-point GMM."""

    algorithm = "EFIAL"

    def _estimate(self, max_num: int) -> EstimationResult:
        if max_num < 2:
            raise ValueError("EFIAL requires max_num >= 2")
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        pilot_samples, pilot_batch = self._pilot(max_num, rng)
        valid_samples, valid_values, failed, pilot_contribution = self._pilot_parts(
            pilot_samples, pilot_batch
        )
        anchors = valid_samples[failed]
        if anchors.shape[0] == 0:
            anchors = self._fallback_anchor(valid_samples, valid_values)

        contribution_parts = [pilot_contribution]
        defensive_ratio = float(self.config.get("defensive_ratio", 0.1))
        round_size = max(
            1, int(self.config.get("is_batch_size", self.config.get("batch_size", 100)))
        )
        history_limit = max(1, int(self.config.get("max_anchor_history", 256)))
        rounds = 0
        while self.runner.ledger.remaining:
            log_density = self.distribution.log_pdf(anchors)
            weights = np.exp(log_density - np.max(log_density))
            weights /= weights.sum()
            proposal = GaussianMixture(
                anchors,
                self._proposal_variance(anchors.shape[0]),
                weights,
            )
            count = min(round_size, self.runner.ledger.remaining)
            samples, actual_ratio = self._draw_importance(
                proposal, count, rng, defensive_ratio
            )
            batch = self._simulate_many(samples, f"efial_is_{rounds:05d}")
            contribution, is_failed, _ = self._importance_contribution(
                samples, batch, proposal, actual_ratio
            )
            contribution_parts.append(contribution)
            new_failures = samples[is_failed & batch.valid_mask]
            if new_failures.shape[0]:
                anchors = np.vstack([anchors, new_failures])[-history_limit:]
            rounds += 1

        return self._finish(
            start=start,
            max_num=max_num,
            contribution_parts=contribution_parts,
            proposal_rule="iterative_density_weighted_failure_gmm",
            pilot_failures=int(failed.sum()),
            proposal_components=int(anchors.shape[0]),
            adaptation_rounds=rounds,
            defensive_ratio=defensive_ratio,
        )


class _FailureProbabilitySurrogate:
    """RBF-SVM surrogate used by the normalized CrossTopo FUSIS flow.

    The CrossTopo donor trains a neural feature extractor before its RBF SVM.
    OpenYield's base environment does not include Torch, so this adapter keeps
    the same probabilistic-classifier contract and uses sklearn's RBF SVM when
    available, with a deterministic NumPy RBF fallback.
    """

    def __init__(self, seed: int, config: dict[str, Any]) -> None:
        self.seed = seed
        self.config = config
        self.backend = "numpy_rbf"
        self._model: Any = None
        self._scaler: Any = None
        self._torch: Any = None
        self._feature_extractor: Any = None
        self._output_layer: Any = None
        self._constant: Optional[float] = None
        self._positive = np.empty((0, 0))
        self._negative = np.empty((0, 0))
        self._bandwidth = 1.0

    def fit(self, samples: np.ndarray, labels: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=float)
        labels = np.asarray(labels, dtype=bool).reshape(-1)
        classes = np.unique(labels)
        if classes.size < 2:
            self._constant = float(classes[0]) if classes.size else 0.0
            self.backend = "constant"
            return
        self._constant = None
        requested = str(self.config.get("surrogate_backend", "auto"))
        if requested not in {"auto", "deep_kernel_svm", "rbf_svm", "numpy_rbf"}:
            raise ValueError(
                "surrogate_backend must be 'auto', 'deep_kernel_svm', "
                "'rbf_svm', or 'numpy_rbf'"
            )
        if requested in {"auto", "deep_kernel_svm"}:
            try:
                import torch
                import torch.nn as nn
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import SVC

                torch.manual_seed(self.seed)
                input_dimension = samples.shape[1]
                if self._feature_extractor is None:
                    self._feature_extractor = nn.Sequential(
                        nn.Linear(input_dimension, input_dimension * 4),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(input_dimension * 4, input_dimension * 2),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(input_dimension * 2, input_dimension),
                    )
                    self._output_layer = nn.Linear(input_dimension, 1)
                    x_tensor = torch.tensor(samples, dtype=torch.float32)
                    y_tensor = torch.tensor(
                        labels.astype(float), dtype=torch.float32
                    ).reshape(-1, 1)
                    positive = float(y_tensor.sum())
                    negative = float(len(y_tensor) - positive)
                    criterion = nn.BCEWithLogitsLoss(
                        pos_weight=torch.tensor([negative / max(positive, 1e-6)])
                    )
                    optimizer = torch.optim.Adam(
                        list(self._feature_extractor.parameters())
                        + list(self._output_layer.parameters()),
                        lr=float(self.config.get("dnn_lr", 1e-3)),
                    )
                    self._feature_extractor.train()
                    for _ in range(int(self.config.get("n_dnn_epochs", 20))):
                        optimizer.zero_grad()
                        logits = self._output_layer(self._feature_extractor(x_tensor))
                        loss = criterion(logits, y_tensor)
                        loss.backward()
                        optimizer.step()
                self._feature_extractor.eval()
                with torch.no_grad():
                    features = self._feature_extractor(
                        torch.tensor(samples, dtype=torch.float32)
                    ).numpy()
                self._scaler = StandardScaler().fit(features)
                self._model = SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    C=float(self.config.get("svm_c", 10.0)),
                    gamma=self.config.get("svm_gamma", 0.1),
                    random_state=self.seed,
                ).fit(self._scaler.transform(features), labels.astype(int))
                self._torch = torch
                self.backend = "deep_kernel_svm"
                return
            except ImportError:
                if requested == "deep_kernel_svm":
                    raise ImportError(
                        "FUSIS surrogate_backend='deep_kernel_svm' requires "
                        "both torch and scikit-learn"
                    )
        if requested != "numpy_rbf":
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.svm import SVC

                self._scaler = StandardScaler().fit(samples)
                transformed = self._scaler.transform(samples)
                self._model = SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    C=float(self.config.get("svm_c", 10.0)),
                    gamma=self.config.get("svm_gamma", "scale"),
                    random_state=self.seed,
                ).fit(transformed, labels.astype(int))
                self.backend = "rbf_svm"
                return
            except ImportError:
                if requested == "rbf_svm":
                    raise
        self._positive = samples[labels]
        self._negative = samples[~labels]
        pair_scale = np.std(samples, axis=0)
        positive_scale = pair_scale[pair_scale > 0]
        self._bandwidth = float(np.median(positive_scale)) if positive_scale.size else 1.0
        self.backend = "numpy_rbf"

    def predict_failure_probability(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=float)
        if self._constant is not None:
            probability = np.full(values.shape[0], self._constant, dtype=float)
        elif self.backend == "deep_kernel_svm":
            self._feature_extractor.eval()
            with self._torch.no_grad():
                features = self._feature_extractor(
                    self._torch.tensor(values, dtype=self._torch.float32)
                ).numpy()
            probability = self._model.predict_proba(
                self._scaler.transform(features)
            )[:, 1]
        elif self.backend == "rbf_svm":
            probability = self._model.predict_proba(
                self._scaler.transform(values)
            )[:, 1]
        else:
            def nearest(reference: np.ndarray) -> np.ndarray:
                diff = values[:, None, :] - reference[None, :, :]
                return np.min(np.sum(diff * diff, axis=2), axis=1)

            positive_distance = nearest(self._positive)
            negative_distance = nearest(self._negative)
            logits = np.clip(
                (negative_distance - positive_distance)
                / max(self._bandwidth ** 2, np.finfo(float).tiny),
                -40.0,
                40.0,
            )
            probability = 1.0 / (1.0 + np.exp(-logits))
        clip = self.config.get("probability_clip", (1e-6, 1.0 - 1e-6))
        return np.clip(probability, float(clip[0]), float(clip[1]))


class FUSIS(CrossTopoImportanceEstimator):
    """CrossTopo FUSIS: surrogate-guided MCMC plus true-simulation correction."""

    algorithm = "FUSIS"

    def _mcmc(
        self,
        surrogate: _FailureProbabilitySurrogate,
        training_samples: np.ndarray,
        training_labels: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        failed_indexes = np.flatnonzero(training_labels)
        if failed_indexes.size:
            current = training_samples[int(rng.choice(failed_indexes))].copy()
        else:
            current = training_samples[int(rng.integers(training_samples.shape[0]))].copy()

        def log_target(point: np.ndarray) -> float:
            log_probability = np.log(
                surrogate.predict_failure_probability(point.reshape(1, -1))[0]
            )
            return float(log_probability + self.distribution.log_pdf(point.reshape(1, -1))[0])

        step_size = float(self.config.get("mcmc_step_size", 0.2))
        burn_in = max(count, int(self.config.get("mcmc_burn_in", count)))
        output = np.empty((count, current.size), dtype=float)
        current_log = log_target(current)
        accepted = 0
        total = burn_in + count
        for index in range(total):
            proposal = current + rng.normal(0.0, step_size, current.size)
            proposal_log = log_target(proposal)
            if np.log(rng.random()) < min(0.0, proposal_log - current_log):
                current = proposal
                current_log = proposal_log
                accepted += 1
            if index < burn_in and index and index % 50 == 0:
                rate = accepted / (index + 1)
                if rate < 0.2:
                    step_size /= 1.2
                elif rate > 0.4:
                    step_size *= 1.2
            if index >= burn_in:
                output[index - burn_in] = current
        return output, accepted / total

    def _estimate(self, max_num: int) -> EstimationResult:
        if max_num < 2:
            raise ValueError("FUSIS requires max_num >= 2")
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        pilot_samples, pilot_batch = self._pilot(max_num, rng)
        valid_samples, _, pilot_failed, pilot_contribution = self._pilot_parts(
            pilot_samples, pilot_batch
        )
        training_samples = valid_samples.copy()
        training_labels = pilot_failed.copy()
        contribution_parts = [pilot_contribution]
        round_size = max(
            1,
            int(
                self.config.get(
                    "verification_batch_size", self.config.get("batch_size", 100)
                )
            ),
        )
        surrogate_mc = max(10, int(self.config.get("surrogate_mc_samples", 2000)))
        rounds = 0
        acceptance_rates: list[float] = []
        backend = "constant"
        surrogate = _FailureProbabilitySurrogate(self.seed, self.config)

        while self.runner.ledger.remaining:
            surrogate.fit(training_samples, training_labels)
            backend = surrogate.backend
            target_draws = self.distribution.sample(surrogate_mc, rng)
            surrogate_probability = float(
                surrogate.predict_failure_probability(target_draws).mean()
            )
            count = min(round_size, self.runner.ledger.remaining)
            if training_labels.any() and (~training_labels).any():
                samples, acceptance = self._mcmc(
                    surrogate,
                    training_samples,
                    training_labels,
                    count,
                    rng,
                )
            else:
                samples = self.distribution.sample(count, rng)
                acceptance = 0.0
                surrogate_probability = 1.0
            batch = self._simulate_many(samples, f"fusis_verify_{rounds:05d}")
            valid = batch.valid_mask
            labels = np.zeros(count, dtype=bool)
            labels[valid] = self._failed(batch.values[valid])
            contribution = np.full(count, np.nan, dtype=float)
            # MH targets q(v) proportional to p_hat(fail|v) * f(v).  Its
            # normalizer is E_f[p_hat] (``surrogate_probability``), hence the
            # true-simulator correction is Z * I(v) / p_hat(fail|v).
            verification_probability = surrogate.predict_failure_probability(samples)
            contribution[valid] = (
                surrogate_probability
                * labels[valid].astype(float)
                / verification_probability[valid]
            )
            contribution_parts.append(contribution)
            if valid.any():
                training_samples = np.vstack([training_samples, samples[valid]])
                training_labels = np.hstack([training_labels, labels[valid]])
            acceptance_rates.append(acceptance)
            rounds += 1

        return self._finish(
            start=start,
            max_num=max_num,
            contribution_parts=contribution_parts,
            proposal_rule="surrogate_probability_mcmc_correction",
            surrogate_backend=backend,
            pilot_failures=int(pilot_failed.sum()),
            adaptation_rounds=rounds,
            mean_mcmc_acceptance=(
                float(np.mean(acceptance_rates)) if acceptance_rates else 0.0
            ),
        )


class _NFlowsProposal:
    """Small adapter around the CNF library used by the CrossTopo OPT donor."""

    def __init__(self, failures: np.ndarray, seed: int, config: dict[str, Any]) -> None:
        import torch
        from nflows.distributions.normal import StandardNormal
        from nflows.flows.base import Flow
        from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
        from nflows.transforms.base import CompositeTransform
        from nflows.transforms.permutations import RandomPermutation

        torch.manual_seed(seed)
        dimension = failures.shape[1]
        transforms = []
        for _ in range(int(config.get("num_flow_steps", 4))):
            transforms.extend([
                RandomPermutation(features=dimension),
                MaskedAffineAutoregressiveTransform(
                    features=dimension,
                    hidden_features=int(config.get("hidden_features", max(16, 2 * dimension))),
                    num_blocks=int(config.get("num_hidden_layers", 2)),
                ),
            ])
        self._torch = torch
        self._flow = Flow(
            CompositeTransform(transforms), StandardNormal(shape=[dimension])
        )
        optimizer = torch.optim.Adam(
            self._flow.parameters(), lr=float(config.get("flow_lr", 1e-4))
        )
        data = torch.tensor(failures, dtype=torch.float32)
        self._flow.train()
        for _ in range(int(config.get("n_train_epochs", 20))):
            optimizer.zero_grad()
            loss = -self._flow.log_prob(data).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._flow.parameters(), 1.0)
            optimizer.step()

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        self._torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        self._flow.eval()
        with self._torch.no_grad():
            return self._flow.sample(count).cpu().numpy().astype(float)

    def log_pdf(self, samples: np.ndarray) -> np.ndarray:
        self._flow.eval()
        with self._torch.no_grad():
            tensor = self._torch.tensor(samples, dtype=self._torch.float32)
            return self._flow.log_prob(tensor).cpu().numpy().astype(float)


class OPT(CrossTopoImportanceEstimator):
    """CrossTopo OPT: failure-trained flow mixed with the target distribution."""

    algorithm = "OPT"

    def _fit_proposal(
        self, failures: np.ndarray, rng: np.random.Generator
    ) -> tuple[Any, str]:
        backend = str(self.config.get("flow_backend", "auto"))
        if backend not in {"auto", "nflows", "gaussian"}:
            raise ValueError("flow_backend must be 'auto', 'nflows', or 'gaussian'")
        if backend != "gaussian":
            try:
                return _NFlowsProposal(
                    failures,
                    int(rng.integers(0, 2**31 - 1)),
                    self.config,
                ), "nflows"
            except ImportError:
                if backend == "nflows":
                    raise ImportError(
                        "OPT flow_backend='nflows' requires both torch and nflows"
                    )
        center = failures.mean(axis=0, keepdims=True)
        target_variance = np.diag(self.distribution.covariance)
        if failures.shape[0] > 1:
            variance = np.var(failures, axis=0)
        else:
            variance = target_variance.copy()
        floor = float(self.config.get("flow_variance_floor", 0.1))
        variance = np.maximum(variance, floor * target_variance)
        return GaussianMixture(center, variance), "gaussian_fallback"

    def _estimate(self, max_num: int) -> EstimationResult:
        if max_num < 2:
            raise ValueError("OPT requires max_num >= 2")
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        pilot_samples, pilot_batch = self._pilot(max_num, rng)
        valid_samples, valid_values, failed, pilot_contribution = self._pilot_parts(
            pilot_samples, pilot_batch
        )
        failures = valid_samples[failed]
        if failures.shape[0] == 0:
            failures = self._fallback_anchor(valid_samples, valid_values)
        contribution_parts = [pilot_contribution]
        target_ratio = float(self.config.get("target_ratio", self.config.get("alpha", 0.8)))
        if not 0.0 < target_ratio < 1.0:
            raise ValueError("target_ratio/alpha must be between 0 and 1")
        round_size = max(
            1, int(self.config.get("is_batch_size", self.config.get("batch_size", 100)))
        )
        history_limit = max(1, int(self.config.get("max_failure_history", 512)))
        rounds = 0
        actual_backend = ""

        while self.runner.ledger.remaining:
            proposal, actual_backend = self._fit_proposal(failures, rng)
            count = min(round_size, self.runner.ledger.remaining)
            proposal_count = int(round(count * (1.0 - target_ratio)))
            proposal_count = min(count - 1, max(1, proposal_count)) if count > 1 else 0
            target_count = count - proposal_count
            samples = np.vstack([
                self.distribution.sample(target_count, rng),
                proposal.sample(proposal_count, rng),
            ])
            batch = self._simulate_many(samples, f"opt_is_{rounds:05d}")
            log_f = self.distribution.log_pdf(samples)
            log_g = proposal.log_pdf(samples)
            actual_target_ratio = target_count / count
            log_q = _logaddexp_mixture(log_g, log_f, actual_target_ratio)
            valid = batch.valid_mask
            is_failed = np.zeros(count, dtype=bool)
            is_failed[valid] = self._failed(batch.values[valid])
            contribution = np.full(count, np.nan, dtype=float)
            contribution[valid] = (
                np.exp(np.clip(log_f[valid] - log_q[valid], -745.0, 700.0))
                * is_failed[valid]
            )
            contribution_parts.append(contribution)
            new_failures = samples[is_failed & valid]
            if new_failures.shape[0]:
                failures = np.vstack([failures, new_failures])[-history_limit:]
            rounds += 1

        return self._finish(
            start=start,
            max_num=max_num,
            contribution_parts=contribution_parts,
            proposal_rule="failure_trained_flow_target_mixture",
            flow_backend=actual_backend,
            pilot_failures=int(failed.sum()),
            adaptation_rounds=rounds,
            target_ratio=target_ratio,
        )


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
