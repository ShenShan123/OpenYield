"""Deterministic constrained-LHS sampling for SramOpt."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from scipy.stats import qmc


_MODEL_RE = re.compile(r"^\s*\.model\s+(\S+)\s+(nmos|pmos)\b", re.IGNORECASE)
_PARAM_RE = re.compile(
    r"\b(vth0|u0)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ModelDrive:
    """Small model-card subset needed by the cheap strength screen."""

    device_type: str
    vth0: float
    u0: float


def parse_model_drives(path: Path) -> Dict[str, ModelDrive]:
    """Read VTH0 and U0 from the nominal BSIM model card."""
    blocks: Dict[str, Dict[str, Any]] = {}
    current: Optional[str] = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _MODEL_RE.match(line)
        if match:
            current = match.group(1).upper()
            blocks[current] = {"device_type": match.group(2).lower()}
        if current is None:
            continue
        for name, raw_value in _PARAM_RE.findall(line):
            blocks[current][name.lower()] = float(raw_value)

    result: Dict[str, ModelDrive] = {}
    for name, values in blocks.items():
        if "vth0" not in values or "u0" not in values:
            continue
        result[name] = ModelDrive(
            device_type=str(values["device_type"]),
            vth0=float(values["vth0"]),
            u0=float(values["u0"]),
        )
    if not result:
        raise ValueError(f"no complete MOS model definitions found in {path}")
    return result


def _drive_proxy(width: float, length: float, model: ModelDrive) -> float:
    if width <= 0.0 or length <= 0.0:
        return math.nan
    overdrive = max(1.0 - abs(model.vth0), 0.0)
    return model.u0 * (width / length) * overdrive**2


def physical_screen(
    design: Mapping[str, Any],
    models: Optional[Mapping[str, ModelDrive]] = None,
    *,
    read_strength_min: float = 0.0,
    write_strength_min: float = 0.0,
) -> Tuple[bool, Dict[str, float]]:
    """Apply the inexpensive cell-strength screen used before circuit calls."""
    required = ("pg_width", "pd_width", "pu_width", "length")
    values = {name: float(design[name]) for name in required}
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        return False, {}

    if models is None:
        pg_drive = values["pg_width"] / values["length"]
        pd_drive = values["pd_width"] / values["length"]
        pu_drive = values["pu_width"] / values["length"]
    else:
        nmos = models[str(design["nmos_model_name"]).upper()]
        pmos = models[str(design["pmos_model_name"]).upper()]
        pg_drive = _drive_proxy(values["pg_width"], values["length"], nmos)
        pd_drive = _drive_proxy(values["pd_width"], values["length"], nmos)
        pu_drive = _drive_proxy(values["pu_width"], values["length"], pmos)

    metrics = {
        "write_strength_ratio": pg_drive / pu_drive,
        "read_strength_ratio": pd_drive / pg_drive,
    }
    passed = (
        all(math.isfinite(value) for value in metrics.values())
        and metrics["read_strength_ratio"] >= float(read_strength_min)
        and metrics["write_strength_ratio"] >= float(write_strength_min)
    )
    return passed, metrics


def maximin_indices(points: np.ndarray, count: int) -> List[int]:
    """Select a deterministic, space-filling subset of candidate points."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("points must be a non-empty two-dimensional array")
    if count <= 0 or count > len(values):
        raise ValueError("count must be between one and the candidate count")
    if count == len(values):
        return list(range(len(values)))

    selected = [int(np.argmin(np.sum((values - 0.5) ** 2, axis=1)))]
    minimum_distance = np.sum((values - values[selected[0]]) ** 2, axis=1)
    minimum_distance[selected[0]] = -1.0
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        candidate_distance = np.sum((values - values[next_index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, candidate_distance)
        minimum_distance[selected] = -1.0
    return selected


def constrained_lhs(
    space: Any,
    count: int,
    *,
    seed: int,
    oversample_factor: int = 5,
    models: Optional[Mapping[str, ModelDrive]] = None,
    read_strength_min: float = 0.0,
    write_strength_min: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a reproducible screened LHS and retain a maximin subset."""
    if count <= 0:
        raise ValueError("count must be positive")
    if oversample_factor <= 0:
        raise ValueError("oversample_factor must be positive")
    dimension = int(space.search_dimension)
    pool_size = int(count) * int(oversample_factor)
    candidates = qmc.LatinHypercube(d=dimension, seed=int(seed)).random(pool_size)

    accepted: List[np.ndarray] = []
    for point in candidates:
        design = space.feature_row(point)
        passed, _ = physical_screen(
            design,
            models,
            read_strength_min=read_strength_min,
            write_strength_min=write_strength_min,
        )
        if passed:
            accepted.append(point)
    if len(accepted) < count:
        raise RuntimeError(
            f"only {len(accepted)} of {pool_size} SramOpt LHS candidates passed "
            f"the physical screen; need {count}"
        )

    feasible = np.asarray(accepted, dtype=float)
    selected = feasible[maximin_indices(feasible, count)]
    return selected, {
        "method": "deterministic constrained LHS with maximin selection",
        "seed": int(seed),
        "candidate_count": pool_size,
        "accepted_count": len(feasible),
        "selected_count": len(selected),
        "read_strength_min": float(read_strength_min),
        "write_strength_min": float(write_strength_min),
        "model_aware": models is not None,
    }


def default_model_card() -> Path:
    return Path(__file__).resolve().parents[2] / "tran_models" / "models_TT.spice"


def load_default_model_drives() -> Optional[Dict[str, ModelDrive]]:
    """Load the repository model card when available; otherwise disable it."""
    path = default_model_card()
    return parse_model_drives(path) if path.is_file() else None


__all__: Iterable[str] = (
    "ModelDrive",
    "constrained_lhs",
    "load_default_model_drives",
    "maximin_indices",
    "parse_model_drives",
    "physical_screen",
)
