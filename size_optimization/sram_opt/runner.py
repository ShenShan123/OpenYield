"""Runnable SramOpt entry point for OpenYield V2.2.6."""

from __future__ import annotations

import math
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contract import FEATURE_COLUMNS
from .predictor import TabPFNYieldPredictor
from .sampling import constrained_lhs, load_default_model_drives, physical_screen
from .space import SramOptDesignSpace


Evaluation = Tuple[Sequence[float], Sequence[float], Optional[Dict[str, Any]], bool]


def _bundle_status(bundle: Optional[str], checkpoint: Optional[str]) -> Dict[str, Any]:
    bundle_path = Path(bundle).expanduser().resolve() if bundle else None
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint else None
    return {
        "bundle_path": str(bundle_path) if bundle_path else None,
        "bundle_available": bool(bundle_path and bundle_path.is_file()),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_available": bool(checkpoint_path and checkpoint_path.is_file()),
    }


def check_integration(
    config_path: str,
    *,
    bundle: Optional[str] = None,
    checkpoint: Optional[str] = None,
    space: Optional[Any] = None,
) -> Dict[str, Any]:
    """Check code/config wiring without loading data, TabPFN, or Xyce."""
    from sram_compiler.version import VERSION

    parameter_space = space or SramOptDesignSpace(config_path)
    default_point = np.asarray(parameter_space.get_default_normalized_vector(), dtype=float)
    design = parameter_space.feature_row(default_point)
    report = {
        "algorithm": "SramOpt",
        "status": "ready-data-free",
        "compiler_version": VERSION,
        "feature_count": len(FEATURE_COLUMNS),
        "search_dimension": int(parameter_space.search_dimension),
        "default_design_complete": tuple(design) == FEATURE_COLUMNS,
        "fixed_capacity_bits": int(design["rows"] * design["cols"] * design["num_arrays"]),
        "screening_topology": "equivalent-array mode 1 (cross)",
        "screening_variation": "nominal",
        "yield_evidence": "external per-device-MC TabPFN bundle",
        "final_validation_required": "mode 0 full-transistor per-device MC",
        **_bundle_status(bundle, checkpoint),
    }
    print(
        "SramOpt check passed: "
        + ", ".join(f"{key}={value}" for key, value in report.items() if key != "algorithm")
    )
    return report


def _merit(result: Mapping[str, Any]) -> float:
    snm = float(result.get("min_snm", 0.0))
    power = float(result.get("max_power", 0.0))
    area = float(result.get("area", 0.0))
    delay = float(
        result.get(
            "max_delay",
            max(float(result.get("read_delay", 0.0)), float(result.get("write_delay", 0.0))),
        )
    )
    values = (snm, power, area, delay)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        return float("-inf")
    return float(np.log10(snm / (power * math.sqrt(area) * delay)))


def run_search(
    space: Any,
    evaluator: Callable[..., Evaluation],
    predictor: Any,
    *,
    max_iter: int,
    seed: int,
    yield_target: Optional[float] = None,
    oversample_factor: int = 5,
    read_strength_min: float = 0.0,
    write_strength_min: float = 0.0,
    model_aware_screen: bool = True,
    backend: str = "SA",
) -> Dict[str, Any]:
    """Run feasibility-first SA or a constrained-LHS search baseline."""
    budget = int(max_iter)
    if budget <= 0:
        raise ValueError("max_iter must be positive")
    target = float(
        predictor.feasibility_target if yield_target is None else yield_target
    )
    if not math.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("yield_target must be in [0, 1]")
    backend = str(backend).upper()
    if backend not in ("SA", "LHS"):
        raise ValueError("SramOpt backend must be SA or LHS")

    models = load_default_model_drives() if model_aware_screen else None
    points, sampling = constrained_lhs(
        space,
        budget,
        seed=int(seed),
        oversample_factor=int(oversample_factor),
        models=models,
        read_strength_min=float(read_strength_min),
        write_strength_min=float(write_strength_min),
    )

    rng = np.random.default_rng(int(seed))
    history = []
    best: Optional[Dict[str, Any]] = None
    current_point = points[0]
    current_rank = None
    warmup = min(10, max(1, budget // 4))
    for iteration in range(1, budget + 1):
        point = points[iteration - 1]
        temperature = 0.01 ** ((iteration - 1) / max(1, budget - 1))
        if backend == "SA" and iteration > warmup:
            # Mutate categorical and continuous normalized coordinates together.
            for _ in range(30):
                proposal = np.clip(
                    current_point + rng.normal(0.0, 0.10 + 0.25 * temperature, len(point)),
                    0.0, 1.0,
                )
                passed, _ = physical_screen(
                    space.feature_row(proposal), models,
                    read_strength_min=read_strength_min,
                    write_strength_min=write_strength_min,
                )
                if passed:
                    point = proposal
                    break
        params = space.convert_params(point)
        feature_row = space.feature_row(point)
        prediction = np.asarray(predictor.predict([feature_row]), dtype=float)
        if prediction.shape != (1,) or not np.isfinite(prediction).all():
            raise ValueError("yield predictor must return one finite probability per design")
        predicted_yield = float(prediction[0])
        if not 0.0 <= predicted_yield <= 1.0:
            raise ValueError("predicted yield must be a probability in [0, 1]")
        objectives, constraints, result, success = evaluator(params)
        merit = _merit(result or {}) if success else float("-inf")
        constraint_values = np.asarray(constraints, dtype=float)
        constraint_valid = bool(np.isfinite(constraint_values).all())
        violation = float(np.maximum(constraint_values, 0.0).sum()) if constraint_valid else math.inf
        nominal_valid = bool(success and math.isfinite(merit) and constraint_valid)
        feasible = bool(nominal_valid and predicted_yield >= target and violation <= 0.0)
        rank = (
            2 if feasible else 1 if nominal_valid else 0,
            merit if feasible else -(max(0.0, target - predicted_yield) + violation),
        )
        accept = current_rank is None or rank >= current_rank
        if not accept and rank[0] == current_rank[0]:
            delta = rank[1] - current_rank[1]
            accept = bool(math.isfinite(delta) and rng.random() < math.exp(max(-700.0, delta / temperature)))
        if accept:
            current_point = point.copy()
            current_rank = rank
        record = {
            "iteration": iteration,
            "params": params,
            "predicted_yield": predicted_yield,
            "yield_feasible": predicted_yield >= target,
            "circuit_success": bool(success),
            "objectives": list(objectives),
            "constraints": list(constraints),
            "merit": merit,
            "result": result,
            "feasible": feasible,
        }
        history.append(record)
        print(
            f"[SramOpt] {iteration}/{budget}: yield={predicted_yield:.6f}, "
            f"target={target:.6f}, merit={merit:.6e}, feasible={feasible}"
        )
        if feasible and (best is None or merit > best["merit"]):
            best = record

    return {
        "algorithm": "SramOpt",
        "backend": backend,
        "mode": "V2.2.6 mode-1 nominal screening with external yield predictor",
        "params": None if best is None else best["params"],
        "merit": float("-inf") if best is None else best["merit"],
        "predicted_yield": None if best is None else best["predicted_yield"],
        "result": None if best is None else best["result"],
        "iteration": None if best is None else best["iteration"],
        "n_evaluations": len(history),
        "feasible_evaluations": sum(record["feasible"] for record in history),
        "yield_target": target,
        "sampling": sampling,
        "history": history,
        "validation_warning": (
            "Equivalent-array mode 1 is a reduced-order screen, not full-array yield "
            "evidence. Validate the selected candidate with mode 0 per-device Monte Carlo."
        ),
    }


def main(config_path: str, problem: Optional[Any] = None, max_iter: int = 50, **kwargs: Any):
    """Run SramOpt or its data-free integration check."""
    bundle = kwargs.get("bundle")
    checkpoint = kwargs.get("checkpoint")
    if kwargs.get("check_only", False):
        injected_space = problem[0] if problem is not None else None
        return check_integration(
            config_path,
            bundle=bundle,
            checkpoint=checkpoint,
            space=injected_space,
        )

    threads = int(kwargs.get("threads", 2))
    if threads <= 0:
        raise ValueError("threads must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))

    output_dir = Path(kwargs.get("output_dir") or (
        Path(__file__).resolve().parents[2] / "outputs" / "sram_opt"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )).expanduser().resolve()
    if problem is None:
        from .evaluator import NominalCircuitEvaluator

        space = SramOptDesignSpace(config_path)
        predictor = TabPFNYieldPredictor(
            bundle,
            checkpoint=checkpoint,
            device=str(kwargs.get("device", "cpu")),
            seed=int(kwargs.get("seed", 42)),
        )
        evaluator = NominalCircuitEvaluator(
            output_dir, seed=int(kwargs.get("seed", 42))
        )
    else:
        try:
            space, evaluator, predictor = problem
        except (TypeError, ValueError) as exc:
            raise ValueError("problem must be (space, evaluator, predictor)") from exc

    print("===== SramOpt joint architecture/sizing optimization =====")
    result = run_search(
        space,
        evaluator,
        predictor,
        max_iter=int(max_iter),
        seed=int(kwargs.get("seed", 42)),
        yield_target=kwargs.get("yield_target"),
        oversample_factor=int(kwargs.get("oversample_factor", 5)),
        read_strength_min=float(kwargs.get("read_strength_min", 0.0)),
        write_strength_min=float(kwargs.get("write_strength_min", 0.0)),
        model_aware_screen=bool(kwargs.get("model_aware_screen", True)),
        backend=str(kwargs.get("backend", "SA")),
    )
    result["seed"] = int(kwargs.get("seed", 42))
    result["predictor_provenance"] = getattr(predictor, "provenance", {})
    if problem is None or kwargs.get("output_dir"):
        output_dir.mkdir(parents=True, exist_ok=True)
        result["output_dir"] = str(output_dir)

        def clean(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            if isinstance(value, np.ndarray):
                return clean(value.tolist())
            if isinstance(value, (float, np.floating)):
                return float(value) if math.isfinite(value) else None
            if isinstance(value, np.integer):
                return int(value)
            return value

        (output_dir / "result.json").write_text(
            json.dumps(clean(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        "SramOpt completed: "
        f"evaluations={result['n_evaluations']}, "
        f"feasible={result['feasible_evaluations']}"
    )
    return result


__all__ = ["check_integration", "main", "run_search"]
