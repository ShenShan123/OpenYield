"""Structured results shared by all unified yield estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union
import json
import math


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


@dataclass(frozen=True)
class EstimationResult:
    algorithm: str
    status: str
    failure_probability: float
    yield_probability: float
    standard_error: float
    budget_target: int
    charged_calls: int
    live_calls: int
    retry_calls: int
    simulator_errors: int
    samples_used: int
    elapsed_seconds: float
    seed: int
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def write_json(self, path: Union[str, Path]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return destination


@dataclass(frozen=True)
class MultiConditionEstimationResult:
    algorithm: str
    status: str
    budget_target: int
    charged_calls: int
    elapsed_seconds: float
    conditions: dict[str, EstimationResult]
    artifacts: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "algorithm": self.algorithm,
            "status": self.status,
            "budget_target": self.budget_target,
            "charged_calls": self.charged_calls,
            "elapsed_seconds": self.elapsed_seconds,
            "conditions": {key: value.to_dict() for key, value in self.conditions.items()},
            "artifacts": list(self.artifacts),
        })

    def write_json(self, path: Union[str, Path]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return destination
