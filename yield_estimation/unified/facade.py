"""Backward-compatible ``YieldEstimator`` facade and artifact management."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
import csv
import hashlib
import json
import os
import uuid

import numpy as np

from .distributions import GaussianDistribution, LegacyDistributionAdapter
from .estimators import ESTIMATOR_REGISTRY
from .results import EstimationResult, MultiConditionEstimationResult
from .simulation import SimulationRunner


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return f"<{type(value).__module__}.{type(value).__name__}>"


class YieldEstimator:
    """Create and run any registered yield estimator with one call style."""

    def __init__(
        self,
        *,
        model: Any,
        algorithm_choice: str = "MC",
        basic_params: Optional[dict[str, Any]] = None,
        algo_params: Optional[dict[str, Any]] = None,
        spice_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.algorithm_choice = algorithm_choice.upper()
        self.basic_params = dict(basic_params or {})
        self.algo_params = dict(algo_params or {})
        self.spice_params = dict(spice_params or {})
        if self.algorithm_choice not in ESTIMATOR_REGISTRY:
            choices = ", ".join(sorted(ESTIMATOR_REGISTRY))
            raise ValueError(f"unknown algorithm {algorithm_choice!r}; choose one of {choices}")
        self.run_root = self._make_run_root()
        self.runner = self._make_runner()
        self.distribution = self._make_distribution()
        self.estimator = self.create()
        self.result: Optional[Union[EstimationResult, MultiConditionEstimationResult]] = None

    def _make_run_root(self) -> Path:
        configured = self.spice_params.get("run_root")
        if configured is not None:
            path = Path(configured)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            suffix = uuid.uuid4().hex[:8]
            path = Path("results/unified") / self.algorithm_choice.lower() / f"{timestamp}_{os.getpid()}_{suffix}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _make_runner(self) -> SimulationRunner:
        if isinstance(self.model, SimulationRunner):
            return self.model
        return SimulationRunner(
            self.model,
            self.run_root / "sim",
            metric=self.spice_params.get("metric", 0),
            input_space=str(self.spice_params.get("input_space", "physical")),
            nominal=self.spice_params.get("nominal"),
            sigma=self.spice_params.get("sigma"),
            max_retries=int(self.spice_params.get("max_retries", 0)),
            quiet=bool(self.spice_params.get("quiet", False)),
        )

    def _make_distribution(self):
        supplied = self.basic_params.get("distribution") or self.basic_params.get("f_norm")
        mean_value = self.basic_params.get("mean", self.basic_params.get("means"))
        if mean_value is None:
            dimension = self.basic_params.get("dimension", self.basic_params.get("feature_num"))
            if dimension is None:
                raise ValueError("basic_params needs mean/means or dimension/feature_num")
            mean = np.zeros(int(dimension), dtype=float)
        else:
            mean = np.asarray(mean_value, dtype=float).reshape(-1)
        if supplied is not None:
            if isinstance(supplied, GaussianDistribution):
                return supplied
            return LegacyDistributionAdapter(supplied, mean)
        covariance = self.basic_params.get("covariance", self.basic_params.get("cov_matrix"))
        if covariance is None:
            sigma = self.basic_params.get("sigma")
            if sigma is None:
                covariance = np.eye(mean.size, dtype=float)
            else:
                sigma_array = np.asarray(sigma, dtype=float)
                if sigma_array.ndim == 0:
                    sigma_array = np.full(mean.size, float(sigma_array))
                covariance = np.diag(sigma_array.reshape(-1) ** 2)
        return GaussianDistribution(mean, np.asarray(covariance, dtype=float))

    def create(self):
        estimator_class = ESTIMATOR_REGISTRY[self.algorithm_choice]
        parameters = dict(self.algo_params)
        parameters.pop("threshold", None)
        parameters.pop("seed", None)
        if self.algorithm_choice == "BIBD":
            parameters["conditions"] = parameters.get(
                "conditions", self.spice_params.get("conditions")
            )
        return estimator_class(
            runner=self.runner,
            distribution=self.distribution,
            threshold=float(self.basic_params.get("threshold", self.algo_params.get("threshold", 0.0))),
            seed=int(self.basic_params.get("seed", self.algo_params.get("seed", 0))),
            failure_direction=str(self.basic_params.get("failure_direction", "greater")),
            operation=str(self.spice_params.get("operation", "read")),
            target_row=int(self.spice_params.get("target_row", 0)),
            target_col=int(self.spice_params.get("target_col", 0)),
            temperature=float(self.spice_params.get("temperature", 27)),
            **parameters,
        )

    def _write_config(self, max_num: int) -> Path:
        path = self.run_root / "config.json"
        payload = {
            "algorithm_choice": self.algorithm_choice,
            "max_num": max_num,
            "basic_params": _serializable(self.basic_params),
            "algo_params": _serializable(self.algo_params),
            "spice_params": _serializable(self.spice_params),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _write_summary(self, result: Union[EstimationResult, MultiConditionEstimationResult]) -> Path:
        path = self.run_root / "summary.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "algorithm", "status", "budget_target", "charged_calls", "live_calls",
                    "simulator_errors", "failure_probability", "yield_probability", "standard_error",
                ],
            )
            writer.writeheader()
            if isinstance(result, EstimationResult):
                writer.writerow({
                    "algorithm": result.algorithm,
                    "status": result.status,
                    "budget_target": result.budget_target,
                    "charged_calls": result.charged_calls,
                    "live_calls": result.live_calls,
                    "simulator_errors": result.simulator_errors,
                    "failure_probability": result.failure_probability,
                    "yield_probability": result.yield_probability,
                    "standard_error": result.standard_error,
                })
            else:
                for name, condition in result.conditions.items():
                    writer.writerow({
                        "algorithm": f"BIBD:{name}",
                        "status": condition.status,
                        "budget_target": condition.budget_target,
                        "charged_calls": condition.charged_calls,
                        "live_calls": condition.live_calls,
                        "simulator_errors": condition.simulator_errors,
                        "failure_probability": condition.failure_probability,
                        "yield_probability": condition.yield_probability,
                        "standard_error": condition.standard_error,
                    })
        return path

    def _write_manifest(self) -> Path:
        manifest = self.run_root / "MANIFEST.sha256"
        lines: list[str] = []
        for path in sorted(self.run_root.rglob("*")):
            if path.is_file() and path != manifest:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines.append(f"{digest}  {path.relative_to(self.run_root)}")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest

    def run(self, max_num: int = 5000):
        config_path = self._write_config(max_num)
        result = self.estimator.start_estimate(max_num=max_num)
        result_path = self.run_root / "result.json"
        summary_path = self._write_summary(result)
        done_path = self.run_root / "DONE"
        done_path.write_text(f"{result.status}\n", encoding="utf-8")
        base_artifacts = tuple(result.artifacts) + (
            str(config_path), str(result_path), str(summary_path), str(done_path),
        )
        result = replace(result, artifacts=base_artifacts)
        result.write_json(result_path)
        manifest = self._write_manifest()
        result = replace(result, artifacts=result.artifacts + (str(manifest),))
        result.write_json(result_path)
        self._write_manifest()
        self.result = result
        self.estimator.result = result
        return result
