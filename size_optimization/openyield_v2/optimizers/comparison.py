#!/usr/bin/env python3
"""Run selected optimizers and save one comparable result set per algorithm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "experiment.yaml"
DEFAULT_OUTPUT = PACKAGE_ROOT / "runs" / "optimization" / "comparison"

EVOLUTIONARY = {"NSGA2", "SPEA2", "UNSGA3", "CTAEA"}
BAYESIAN = {"GPBO", "PAREGO", "MACE"}
SUPPORTED = (*sorted(EVOLUTIONARY), *sorted(BAYESIAN), "PROPOSED")
ALIASES = {
    "NSGA-II": "NSGA2",
    "NSGA_II": "NSGA2",
    "U-NSGA-III": "UNSGA3",
    "U_NSGA_III": "UNSGA3",
    "PAR-EGO": "PAREGO",
    "PAR_EGO": "PAREGO",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--algorithms",
        default="NSGA2,SPEA2,UNSGA3,CTAEA,GPBO,PAREGO,MACE,PROPOSED",
        help="Comma-separated subset of supported algorithms.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Existing comparison run. Only missing algorithms are launched.",
    )
    parser.add_argument("--max-evals", type=int)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--max-train-per-topology", type=int, default=250)
    parser.add_argument(
        "--balance-topologies",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verbose-library-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--bounds-lower-q", type=float, default=0.01)
    parser.add_argument("--bounds-upper-q", type=float, default=0.99)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--init-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=10000)
    parser.add_argument("--gp-max-train", type=int, default=600)
    parser.add_argument("--gp-noise", type=float, default=1.0e-6)
    parser.add_argument("--lcb-beta", type=float, default=2.0)
    parser.add_argument("--parego-rho", type=float, default=0.05)
    parser.add_argument("--diversity-radius", type=float, default=0.08)
    parser.add_argument("--coarse-evals", type=int)
    parser.add_argument("--coarse-pop-size", type=int, default=50)
    parser.add_argument("--representative-points", type=int)
    parser.add_argument("--preferences-per-point", type=int)
    parser.add_argument("--refine-steps", type=int)
    parser.add_argument("--max-train-samples", type=int, default=500)
    parser.add_argument("--preference-edge", type=float, default=0.10)
    parser.add_argument("--hard-audit-every", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--continuous-lr", type=float, default=0.10)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--lr-min-ratio", type=float, default=0.10)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1.0e-6)
    parser.add_argument("--early-stop-min-steps", type=int, default=40)
    parser.add_argument("--topology-lr", type=float, default=0.08)
    parser.add_argument("--discrete-lr", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--refine-smoothing", type=float, default=0.05)
    parser.add_argument("--tau-start", type=float, default=1.2)
    parser.add_argument("--tau-end", type=float, default=0.15)
    parser.add_argument("--topology-init-bias", type=float, default=0.5)
    parser.add_argument("--discrete-init-bias", type=float, default=2.0)
    parser.add_argument(
        "--optimize-discrete",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-gumbel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reset-torch-seed-each-step",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--seed-ref-source",
        choices=["coarse", "surrogate"],
        default="coarse",
    )
    parser.add_argument(
        "--enable-simple-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--constraint-penalty", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_algorithms(raw: str) -> List[str]:
    algorithms: List[str] = []
    for item in str(raw).split(","):
        key = ALIASES.get(item.strip().upper(), item.strip().upper())
        if not key:
            continue
        if key not in SUPPORTED:
            raise ValueError(
                f"Unsupported algorithm {item!r}. Supported: {', '.join(SUPPORTED)}"
            )
        if key not in algorithms:
            algorithms.append(key)
    if not algorithms:
        raise ValueError("At least one algorithm must be selected.")
    return algorithms


def read_config(path: Path) -> Tuple[Path, Dict[str, Any]]:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Experiment config not found: {resolved}")
    return resolved, yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


def validate_comparison_args(
    args: argparse.Namespace,
    algorithms: List[str],
    max_evals: int,
) -> None:
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")
    if not (0.0 <= args.bounds_lower_q < args.bounds_upper_q <= 1.0):
        raise ValueError("Bounds quantiles must satisfy 0 <= lower < upper <= 1.")
    if algorithms and any(name in EVOLUTIONARY for name in algorithms):
        if args.pop_size <= 1 or max_evals % args.pop_size:
            raise ValueError(
                "Evolutionary max_evals must be divisible by pop_size."
            )
    if algorithms and any(name in BAYESIAN for name in algorithms):
        if not (1 < args.init_samples <= max_evals):
            raise ValueError("init_samples must be in [2, max_evals].")
        if args.batch_size <= 0 or args.candidate_pool < args.batch_size:
            raise ValueError("candidate_pool must be at least batch_size > 0.")
        if args.gp_max_train != 0 and args.gp_max_train < 2:
            raise ValueError("gp_max_train must be 0 or at least 2.")


def _config_int(
    explicit: int | None,
    section: Dict[str, Any],
    key: str,
    default: int,
) -> int:
    return int(explicit if explicit is not None else section.get(key, default))


def proposed_budget_options(
    args: argparse.Namespace,
    budget: Dict[str, Any],
    max_evals: int,
) -> Dict[str, Any]:
    coarse = _config_int(
        args.coarse_evals, budget, "proposed_coarse_evals", max_evals // 2
    )
    representatives = _config_int(
        args.representative_points,
        budget,
        "proposed_representative_points",
        10,
    )
    preferences = _config_int(
        args.preferences_per_point,
        budget,
        "proposed_preferences_per_point",
        1,
    )
    runs = representatives * preferences
    if runs <= 0:
        raise ValueError("The number of refine runs must be positive.")
    seed_reevaluations = (
        representatives if args.seed_ref_source == "surrogate" else 0
    )

    def planned_for_steps(steps: int) -> int:
        audit_steps = {0, int(steps)}
        if args.hard_audit_every > 0:
            audit_steps.update(
                range(0, int(steps) + 1, args.hard_audit_every)
            )
        return (
            coarse
            + seed_reevaluations
            + runs * ((int(steps) + 1) + len(audit_steps))
        )

    if args.refine_steps is not None:
        refine_steps = int(args.refine_steps)
    elif budget.get("proposed_refine_steps") is not None:
        refine_steps = int(budget["proposed_refine_steps"])
    else:
        candidates = [
            steps
            for steps in range(1, max_evals + 1)
            if planned_for_steps(steps) == max_evals
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Could not infer a unique refine_steps value for the requested "
                "Proposed budget. Set proposed_refine_steps explicitly."
            )
        refine_steps = candidates[0]

    planned = planned_for_steps(refine_steps)
    if planned != max_evals:
        raise ValueError(
            "Proposed budget mismatch: "
            f"coarse={coarse}, runs={runs}, refine_steps={refine_steps}, "
            f"planned={planned}, expected={max_evals}."
        )
    return {
        "coarse_evals": coarse,
        "coarse_pop_size": int(args.coarse_pop_size),
        "representative_points": representatives,
        "preferences_per_point": preferences,
        "refine_steps": refine_steps,
        "hard_audit_every": int(args.hard_audit_every),
        "seed_ref_source": str(args.seed_ref_source),
    }


def build_command(
    algorithm: str,
    *,
    args: argparse.Namespace,
    config_path: Path,
    max_evals: int,
    temporary_root: Path,
    proposed_options: Dict[str, Any],
) -> List[str]:
    child = [
        "--output-dir",
        str(temporary_root),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--max-evals",
        str(max_evals),
    ]
    if algorithm in EVOLUTIONARY:
        family = "evolutionary"
        child.extend(
            [
                "--algorithms",
                algorithm,
                "--pop-size",
                str(args.pop_size),
                "--test-size",
                str(args.test_size),
                "--max-train-per-topology",
                str(args.max_train_per_topology),
                "--bounds-lower-q",
                str(args.bounds_lower_q),
                "--bounds-upper-q",
                str(args.bounds_upper_q),
            ]
        )
    elif algorithm in BAYESIAN:
        family = "bayesian"
        child.extend(
            [
                "--algorithms",
                algorithm,
                "--init-samples",
                str(args.init_samples),
                "--batch-size",
                str(args.batch_size),
                "--candidate-pool",
                str(args.candidate_pool),
                "--test-size",
                str(args.test_size),
                "--max-train-per-topology",
                str(args.max_train_per_topology),
                "--gp-max-train",
                str(args.gp_max_train),
                "--gp-noise",
                str(args.gp_noise),
                "--lcb-beta",
                str(args.lcb_beta),
                "--parego-rho",
                str(args.parego_rho),
                "--diversity-radius",
                str(args.diversity_radius),
                "--bounds-lower-q",
                str(args.bounds_lower_q),
                "--bounds-upper-q",
                str(args.bounds_upper_q),
            ]
        )
    else:
        family = "proposed"
        child.extend(
            [
                "--train-ratio",
                str(1.0 - args.test_size),
                "--test-ratio",
                str(args.test_size),
                "--max-train-samples",
                str(args.max_train_samples),
                "--coarse-evals",
                str(proposed_options["coarse_evals"]),
                "--coarse-pop-size",
                str(proposed_options["coarse_pop_size"]),
                "--representative-points",
                str(proposed_options["representative_points"]),
                "--preferences-per-point",
                str(proposed_options["preferences_per_point"]),
                "--refine-steps",
                str(proposed_options["refine_steps"]),
                "--preference-edge",
                str(args.preference_edge),
                "--hard-audit-every",
                str(proposed_options["hard_audit_every"]),
                "--print-every",
                str(args.print_every),
                "--continuous-lr",
                str(args.continuous_lr),
                "--lr-scheduler",
                str(args.lr_scheduler),
                "--lr-min-ratio",
                str(args.lr_min_ratio),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--early-stop-min-delta",
                str(args.early_stop_min_delta),
                "--early-stop-min-steps",
                str(args.early_stop_min_steps),
                "--topology-lr",
                str(args.topology_lr),
                "--discrete-lr",
                str(args.discrete_lr),
                "--grad-clip-norm",
                str(args.grad_clip_norm),
                "--refine-smoothing",
                str(args.refine_smoothing),
                "--tau-start",
                str(args.tau_start),
                "--tau-end",
                str(args.tau_end),
                "--topology-init-bias",
                str(args.topology_init_bias),
                "--discrete-init-bias",
                str(args.discrete_init_bias),
                "--seed-ref-source",
                str(proposed_options["seed_ref_source"]),
                "--constraint-penalty",
                str(args.constraint_penalty),
                "--bounds-lower-q",
                str(args.bounds_lower_q),
                "--bounds-upper-q",
                str(args.bounds_upper_q),
            ]
        )
        for enabled, positive, negative in (
            (args.optimize_discrete, "--optimize-discrete", "--no-optimize-discrete"),
            (args.use_gumbel, "--use-gumbel", "--no-use-gumbel"),
            (
                args.reset_torch_seed_each_step,
                "--reset-torch-seed-each-step",
                "--no-reset-torch-seed-each-step",
            ),
            (
                args.enable_simple_constraints,
                "--enable-simple-constraints",
                "--no-enable-simple-constraints",
            ),
        ):
            child.append(positive if enabled else negative)
    child.append(
        "--balance-topologies"
        if args.balance_topologies
        else "--no-balance-topologies"
    )
    child.append(
        "--verbose-library-training"
        if args.verbose_library_training
        else "--no-verbose-library-training"
    )
    return [
        sys.executable,
        "-u",
        "-m",
        "size_optimization.openyield_v2.run",
        "optimize",
        "--family",
        family,
        "--config",
        str(config_path),
        "--",
        *child,
    ]


def newest_run_directory(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No optimizer run directory was created under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _completed(
    evaluation_path: Path,
    front_path: Path,
    expected_rows: int,
) -> bool:
    if not evaluation_path.exists() or not front_path.exists():
        return False
    try:
        return len(pd.read_csv(evaluation_path)) == expected_rows
    except Exception:
        return False


def run() -> Path:
    args = parse_args()
    algorithms = normalize_algorithms(args.algorithms)
    config_path, raw_config = read_config(args.config)
    budget = raw_config.get("optimization_budget", {})
    max_evals = int(
        args.max_evals
        if args.max_evals is not None
        else budget.get("tabpfn_design_queries_per_algorithm", 1000)
    )
    if max_evals <= 0:
        raise ValueError("max_evals must be positive.")
    if args.gpu_id is not None and args.gpu_id < 0:
        raise ValueError("--gpu-id must be non-negative.")
    validate_comparison_args(args, algorithms, max_evals)
    proposed_options = (
        proposed_budget_options(args, budget, max_evals)
        if "PROPOSED" in algorithms
        else {}
    )

    if args.resume_dir:
        output_dir = args.resume_dir.expanduser().resolve()
        if not output_dir.exists():
            raise FileNotFoundError(f"Comparison run not found: {output_dir}")
    else:
        output_base = args.output_dir.expanduser()
        if not output_base.is_absolute():
            output_base = (Path.cwd() / output_base).resolve()
        output_dir = output_base / time.strftime("%Y%m%d_%H%M%S")

    evaluations_dir = output_dir / "evaluations"
    fronts_dir = output_dir / "pareto_fronts"
    if not args.dry_run:
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        fronts_dir.mkdir(parents=True, exist_ok=True)

    commands: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="openyield_compare_") as temp_dir:
        temp_base = Path(temp_dir)
        for algorithm in algorithms:
            destination_evaluations = evaluations_dir / f"{algorithm}.csv"
            destination_front = fronts_dir / f"{algorithm}.csv"
            if _completed(
                destination_evaluations, destination_front, max_evals
            ):
                print(f"[Comparison] Reusing {algorithm}")
                continue

            temporary_root = temp_base / algorithm
            temporary_root.mkdir(parents=True, exist_ok=True)
            command = build_command(
                algorithm,
                args=args,
                config_path=config_path,
                max_evals=max_evals,
                temporary_root=temporary_root,
                proposed_options=proposed_options,
            )
            commands.append({"algorithm": algorithm, "command": command})
            print(f"\n[Comparison] Running {algorithm}")
            print(" ".join(command))
            if args.dry_run:
                continue

            environment = os.environ.copy()
            if args.gpu_id is not None:
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                check=True,
            )

            run_dir = newest_run_directory(temporary_root)
            source_evaluations = run_dir / "evaluations" / f"{algorithm}.csv"
            source_front = run_dir / "pareto_fronts" / f"{algorithm}.csv"
            if not source_evaluations.exists() or not source_front.exists():
                raise FileNotFoundError(
                    f"{algorithm} did not produce the shared output contract."
                )
            evaluations = pd.read_csv(source_evaluations)
            if len(evaluations) != max_evals:
                raise RuntimeError(
                    f"{algorithm} produced {len(evaluations)} rows; "
                    f"expected {max_evals}."
                )
            evaluations.to_csv(destination_evaluations, index=False)
            pd.read_csv(source_front).to_csv(destination_front, index=False)

    manifest = {
        "algorithms": algorithms,
        "max_evals_per_algorithm": max_evals,
        "config": str(config_path),
        "seed": args.seed,
        "device": args.device,
        "physical_gpu_id": args.gpu_id,
        "proposed": proposed_options,
        "commands": commands,
        "dry_run": bool(args.dry_run),
        "output_contract": {
            "evaluations": "exactly one row per optimizer query",
            "pareto_fronts": "non-dominated points over every configured objective",
        },
    }
    if args.dry_run:
        print(f"\nDry-run output target: {output_dir}")
        return output_dir

    (output_dir / "run_config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_parts: List[pd.DataFrame] = []
    for algorithm in algorithms:
        evaluation_path = evaluations_dir / f"{algorithm}.csv"
        front_path = fronts_dir / f"{algorithm}.csv"
        evaluations = pd.read_csv(evaluation_path)
        front = pd.read_csv(front_path)
        summary_parts.append(
            pd.DataFrame(
                [
                    {
                        "algorithm": algorithm,
                        "evaluations": len(evaluations),
                        "feasible_evaluations": int(
                            evaluations["constraint_valid"].astype(bool).sum()
                        ),
                        "pareto_front_size": len(front),
                    }
                ]
            )
        )
    pd.concat(summary_parts, ignore_index=True).to_csv(
        output_dir / "algorithm_summary.csv", index=False
    )
    print(f"\nComparison results: {output_dir}")
    return output_dir


def main() -> None:
    run()


if __name__ == "__main__":
    main()
