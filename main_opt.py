"""Unified interactive and command-line entry point for SRAM optimization."""

from __future__ import annotations

import argparse
import importlib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "size_optimization" / "config_sram.yaml"


@dataclass(frozen=True)
class Algorithm:
    name: str
    description: str
    module: str


ALGORITHMS = (
    Algorithm("SA", "Simulated Annealing", "size_optimization.demo_sa"),
    Algorithm("PSO", "Particle Swarm Optimization", "size_optimization.demo_pso"),
    Algorithm("CBO", "Constrained Bayesian Optimization", "size_optimization.demo_cbo"),
    Algorithm("RoSE_Opt", "RL-enhanced Bayesian Optimization", "size_optimization.demo_roseopt"),
    Algorithm("SMAC", "Sequential Model-based Algorithm Configuration", "size_optimization.demo_smac"),
    Algorithm("CMA-ES", "Covariance Matrix Adaptation Evolution Strategy", "size_optimization.demo_cmaes"),
    Algorithm("MOEAD", "Multi-Objective Evolutionary Algorithm", "size_optimization.demo_moead"),
    Algorithm("MOBO", "Multi-Objective Bayesian Optimization", "size_optimization.demo_mobo"),
    Algorithm("NSGA-II", "Non-dominated Sorting Genetic Algorithm II", "size_optimization.demo_nsgaii"),
    Algorithm("tSS-BO", "Truncated Subspace Sampling BO", "size_optimization.demo_tssbo"),
    Algorithm("CPN", "TabPFN-based Bayesian Optimization", "size_optimization.demo_cpn"),
    Algorithm("Random", "Random Search baseline", "size_optimization.demo_random"),
    Algorithm("SramOpt", "Yield-constrained joint architecture/sizing", "size_optimization.sram_opt.runner"),
    Algorithm("Online-CTR", "Online TabPFN coarse-to-refine optimization", "size_optimization.demo_online_ctr"),
)


def _algorithm_lookup() -> Dict[str, Algorithm]:
    result: Dict[str, Algorithm] = {}
    for index, algorithm in enumerate(ALGORITHMS, start=1):
        result[str(index)] = algorithm
        result[algorithm.name.lower()] = algorithm
    result["sramopt"] = next(item for item in ALGORITHMS if item.name == "SramOpt")
    result["sram_opt"] = result["sramopt"]
    result["online_ctr"] = next(item for item in ALGORITHMS if item.name == "Online-CTR")
    return result


def print_algorithms() -> None:
    print("Available optimization algorithms:")
    for index, algorithm in enumerate(ALGORITHMS, start=1):
        print(f"  {index:>2}. {algorithm.name:<12} - {algorithm.description}")


def _resolve_algorithms(value: str) -> List[Algorithm]:
    lookup = _algorithm_lookup()
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("no algorithm was selected")
    if len(tokens) == 1 and tokens[0].lower() == "all":
        return list(ALGORITHMS)
    unknown = [token for token in tokens if token.lower() not in lookup]
    if unknown:
        raise ValueError("unknown algorithm(s): " + ", ".join(unknown))
    selected = []
    for token in tokens:
        algorithm = lookup[token.lower()]
        if algorithm not in selected:
            selected.append(algorithm)
    return selected


def _interactive_selection() -> List[Algorithm]:
    print_algorithms()
    print("\nEnter numbers separated by commas, 'all', or 'none'.")
    choice = input("Your choice: ").strip()
    if choice.lower() == "none":
        return []
    try:
        return _resolve_algorithms(choice)
    except ValueError as exc:
        print(f"{exc}. Running SA as the default.")
        return [ALGORITHMS[0]]


def _new_method_kwargs(algorithm: Algorithm, args: argparse.Namespace) -> Dict[str, Any]:
    common: Dict[str, Any] = {"check_only": bool(args.check), "seed": args.seed}
    if args.device is not None:
        common["device"] = args.device
    if algorithm.name == "SramOpt":
        common.update(
            bundle=args.bundle, checkpoint=args.checkpoint,
            yield_target=args.yield_target, threads=args.threads,
            backend=args.backend, output_dir=args.output_dir,
        )
    elif algorithm.name == "Online-CTR":
        common["ctr_config_path"] = args.ctr_config
        if args.gpu_index is not None:
            common["gpu_index"] = args.gpu_index
    return common


def run_algorithm(algorithm: Algorithm, config_path: Path, args: argparse.Namespace) -> Any:
    print(f"\n{'=' * 20} Running {algorithm.name} {'=' * 20}")
    module = importlib.import_module(algorithm.module)
    if args.check and algorithm.name not in ("SramOpt", "Online-CTR"):
        result = {"algorithm": algorithm.name, "status": "import-ready"}
        print(f"{algorithm.name} import check passed.")
        return result

    kwargs: Dict[str, Any] = {}
    if args.max_iter is not None:
        kwargs["max_iter"] = args.max_iter
    if algorithm.name in ("SramOpt", "Online-CTR"):
        kwargs.update(_new_method_kwargs(algorithm, args))
    result = module.main(str(config_path), **kwargs)
    print(f"{algorithm.name} completed.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", help="name/number, comma-separated list, or all")
    parser.add_argument("--list", action="store_true", help="list algorithms and exit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--check", "--dry-run", dest="check", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="TabPFN device (SramOpt: cpu; Online-CTR: config)")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--bundle", help="trusted SramOpt TabPFN joblib bundle")
    parser.add_argument("--checkpoint", help="SramOpt TabPFN checkpoint override")
    parser.add_argument("--yield-target", type=float)
    parser.add_argument("--backend", choices=("SA", "LHS"), default="SA")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ctr-config", help="Online-CTR YAML configuration")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    if args.max_iter is not None and args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.gpu_index is not None and args.gpu_index < 0:
        parser.error("--gpu-index must be non-negative")
    if args.yield_target is not None and not 0.0 <= args.yield_target <= 1.0:
        parser.error("--yield-target must be in [0, 1]")
    path = args.config.expanduser().resolve()
    if not path.is_file():
        parser.error(f"configuration file not found: {path}")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print_algorithms()
        return 0
    config_path = _validate_args(parser, args)
    print("\n" + "=" * 60)
    print("Starting SRAM Circuit Optimization Algorithms")
    print("=" * 60)
    print(f"Using configuration file: {config_path}")
    try:
        selected = _interactive_selection() if args.algorithm is None else _resolve_algorithms(args.algorithm)
    except ValueError as exc:
        parser.error(str(exc))

    failures = 0
    for algorithm in selected:
        try:
            result = run_algorithm(algorithm, config_path, args)
            if (not args.check and isinstance(result, dict)
                    and algorithm.name in ("SramOpt", "Online-CTR")
                    and result.get("result") is None):
                failures += 1
                print(f"{algorithm.name}: no feasible circuit solution was found.")
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            failures += 1
            print(f"Error running {algorithm.name}: {exc}", flush=True)
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SRAM Optimization Session Completed")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
