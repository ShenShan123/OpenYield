"""Shared objective and constraint specification for every optimizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


METRIC_UNITS = {
    "hold_snm": "V",
    "read_snm": "V",
    "write_snm": "V",
    "min_snm": "V",
    "raw_read_delay": "s",
    "raw_write_delay": "s",
    "read_delay": "s",
    "write_delay": "s",
    "max_delay": "s",
    "read_pstc": "W",
    "read_pdyn": "W",
    "write_pstc": "W",
    "write_pdyn": "W",
    "read_power": "W",
    "write_power": "W",
    "max_power": "W",
    "single_array_area": "m^2",
    "area": "m^2",
    "power_delay_product": "W*s",
}

AVAILABLE_OPTIMIZATION_METRICS = {
    "hold_snm",
    "read_snm",
    "write_snm",
    "min_snm",
    "read_delay",
    "write_delay",
    "max_delay",
    "read_power",
    "write_power",
    "max_power",
    "area",
    "power_delay_product",
}
PHYSICAL_POSITIVE_METRICS = [
    "hold_snm",
    "read_snm",
    "write_snm",
    "read_delay",
    "write_delay",
    "read_power",
    "write_power",
    "area",
]

DEFAULT_OBJECTIVES: List[Dict[str, Any]] = [
    {"name": "max_power", "source": "max_power", "direction": "min"},
    {"name": "max_delay", "source": "max_delay", "direction": "min"},
    {"name": "area", "source": "area", "direction": "min"},
    {"name": "min_snm", "source": "min_snm", "direction": "max"},
]

DEFAULT_CONSTRAINTS: List[Dict[str, Any]] = [
    {
        "name": "minimum_snm",
        "metric": "min_snm",
        "operator": ">=",
        "value": 0.1,
    },
    {
        "name": "maximum_delay",
        "metric": "max_delay",
        "operator": "<=",
        "value": 3.0e-9,
    },
    {
        "name": "maximum_power",
        "metric": "max_power",
        "operator": "<=",
        "value": 0.02,
    },
]


def _normalise_objectives(raw: Any) -> List[Dict[str, Any]]:
    values = DEFAULT_OBJECTIVES if raw is None else raw
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("optimization_problem.objectives must contain at least 2 items.")
    result: List[Dict[str, Any]] = []
    for index, item in enumerate(values, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Objective {index} must be a mapping.")
        source = str(item.get("source", item.get("name", ""))).strip()
        name = str(item.get("name", source)).strip()
        direction = str(item.get("direction", "")).strip().lower()
        if not name or source not in AVAILABLE_OPTIMIZATION_METRICS:
            raise ValueError(
                f"Objective {index} has unsupported source={source!r}; "
                f"available={sorted(AVAILABLE_OPTIMIZATION_METRICS)}"
            )
        if direction not in {"min", "max"}:
            raise ValueError(f"Objective {index} direction must be 'min' or 'max'.")
        result.append(
            {
                "name": name,
                "source": source,
                "direction": direction,
                "label": str(item.get("label", name)),
                "unit": str(item.get("unit", METRIC_UNITS.get(source, ""))),
            }
        )
    names = [item["name"] for item in result]
    sources = [item["source"] for item in result]
    if len(set(names)) != len(names) or len(set(sources)) != len(sources):
        raise ValueError("Objective names and sources must be unique.")
    return result


def _normalise_constraints(raw: Any) -> List[Dict[str, Any]]:
    values = DEFAULT_CONSTRAINTS if raw is None else raw
    if not isinstance(values, list):
        raise TypeError("optimization_problem.constraints must be a list.")
    result: List[Dict[str, Any]] = []
    for index, item in enumerate(values, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Constraint {index} must be a mapping.")
        metric = str(item.get("metric", "")).strip()
        operator = str(item.get("operator", "")).strip()
        if metric not in AVAILABLE_OPTIMIZATION_METRICS:
            raise ValueError(
                f"Constraint {index} has unsupported metric={metric!r}; "
                f"available={sorted(AVAILABLE_OPTIMIZATION_METRICS)}"
            )
        if operator not in {"<=", ">="}:
            raise ValueError(f"Constraint {index} operator must be '<=' or '>='.")
        value = float(item["value"])
        scale = float(item.get("scale", max(abs(value), 1.0e-30)))
        if not np.isfinite(value) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Constraint {index} value/scale must be finite; scale > 0.")
        result.append(
            {
                "name": str(item.get("name", f"{metric}_{operator}_{value:g}")),
                "metric": metric,
                "operator": operator,
                "value": value,
                "scale": scale,
            }
        )
    names = [item["name"] for item in result]
    if len(set(names)) != len(names):
        raise ValueError("Constraint names must be unique.")
    return result


def load_problem_spec(
    config_path: str | Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = raw.get("optimization_problem", {})
    objectives = _normalise_objectives(section.get("objectives"))
    constraints = _normalise_constraints(section.get("constraints"))
    enforce_physical = bool(section.get("enforce_physical_validity", True))
    return objectives, constraints, enforce_physical


def objective_matrix(
    metrics_df: pd.DataFrame,
    objectives: Sequence[Dict[str, Any]],
) -> np.ndarray:
    columns: List[np.ndarray] = []
    for spec in objectives:
        source = str(spec["source"])
        if source not in metrics_df.columns:
            raise KeyError(f"Objective metric is missing: {source}")
        values = pd.to_numeric(metrics_df[source], errors="coerce").to_numpy(float)
        columns.append(-values if spec["direction"] == "max" else values)
    return np.column_stack(columns)


def non_dominated_mask(costs: np.ndarray) -> np.ndarray:
    """Return the minimization Pareto mask for a finite 2-D cost matrix."""
    values = np.asarray(costs, dtype=float)
    if values.ndim != 2:
        raise ValueError("costs must be a two-dimensional matrix.")
    keep = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        if not keep[index]:
            continue
        dominated = np.all(values <= point, axis=1) & np.any(
            values < point, axis=1
        )
        dominated[index] = False
        if np.any(dominated):
            keep[index] = False
    return keep


def pareto_front_from_evaluations(
    evaluations: pd.DataFrame,
    objectives: Sequence[Dict[str, Any]],
    *,
    feasible_column: str = "constraint_valid",
) -> pd.DataFrame:
    """Compute a front from all recorded queries without calling a surrogate."""
    if feasible_column not in evaluations.columns:
        raise KeyError(f"Feasibility column is missing: {feasible_column}")
    feasible = evaluations.loc[
        evaluations[feasible_column].astype(bool)
    ].copy()
    if feasible.empty:
        return feasible.reset_index(drop=True)

    costs = objective_matrix(feasible, objectives)
    finite = np.all(np.isfinite(costs), axis=1)
    feasible = feasible.loc[finite].copy().reset_index(drop=True)
    costs = costs[finite]
    if feasible.empty:
        return feasible

    front = feasible.loc[non_dominated_mask(costs)].copy()
    first = objectives[0]
    return front.sort_values(
        str(first["source"]),
        ascending=(str(first["direction"]) == "min"),
        kind="mergesort",
    ).reset_index(drop=True)


def feasible_mask(
    metrics_df: pd.DataFrame,
    constraints: Sequence[Dict[str, Any]],
    *,
    enforce_physical_validity: bool,
) -> np.ndarray:
    valid = np.ones(len(metrics_df), dtype=bool)
    if enforce_physical_validity:
        missing = [
            metric for metric in PHYSICAL_POSITIVE_METRICS
            if metric not in metrics_df.columns
        ]
        if missing:
            raise KeyError(f"Physical-validity metrics are missing: {missing}")
        for metric in PHYSICAL_POSITIVE_METRICS:
            values = pd.to_numeric(metrics_df[metric], errors="coerce").to_numpy(float)
            valid &= np.isfinite(values) & (values > 0)

    for spec in constraints:
        metric = str(spec["metric"])
        if metric not in metrics_df.columns:
            raise KeyError(f"Constraint metric is missing: {metric}")
        values = pd.to_numeric(metrics_df[metric], errors="coerce").to_numpy(float)
        valid &= np.isfinite(values)
        if spec["operator"] == "<=":
            valid &= values <= float(spec["value"])
        else:
            valid &= values >= float(spec["value"])
    return valid


def constraint_violation(
    metrics_df: pd.DataFrame,
    constraints: Sequence[Dict[str, Any]],
    *,
    enforce_physical_validity: bool,
) -> np.ndarray:
    """Return one non-negative aggregate violation per row."""
    total = np.zeros(len(metrics_df), dtype=float)
    if enforce_physical_validity:
        for metric in PHYSICAL_POSITIVE_METRICS:
            values = pd.to_numeric(metrics_df[metric], errors="coerce").to_numpy(float)
            scale = max(float(np.nanmedian(np.abs(values))), 1.0e-30)
            term = np.maximum(-values / scale, 0.0)
            term[~np.isfinite(values)] = 1.0e6
            total += term * term
    for spec in constraints:
        values = pd.to_numeric(
            metrics_df[str(spec["metric"])], errors="coerce"
        ).to_numpy(float)
        value = float(spec["value"])
        scale = float(spec["scale"])
        if spec["operator"] == "<=":
            term = np.maximum((values - value) / scale, 0.0)
        else:
            term = np.maximum((value - values) / scale, 0.0)
        term[~np.isfinite(values)] = 1.0e6
        total += term * term
    return total
