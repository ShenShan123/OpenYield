#!/usr/bin/env python3
"""Command-line entry point for the OpenYieldV2 optimizers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _call_main(main: Callable[[], None], forwarded: List[str]) -> None:
    old = sys.argv
    try:
        sys.argv = [old[0], *forwarded]
        main()
    finally:
        sys.argv = old


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    optimize = sub.add_parser("optimize", help="Run one optimization family.")
    optimize.add_argument(
        "--family",
        choices=["evolutionary", "bayesian", "proposed"],
        required=True,
    )
    optimize.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "experiment.yaml",
        help=(
            "Shared experiment YAML. Dataset paths and optimization budgets are "
            "used as defaults; explicit child arguments after '--' take priority."
        ),
    )
    optimize.add_argument(
        "--gpu-id",
        type=int,
        help=(
            "Physical CUDA GPU index, for example 4. It is mapped to logical "
            "cuda:0 before importing torch/TabPFN."
        ),
    )
    optimize.add_argument("args", nargs=argparse.REMAINDER)

    compare = sub.add_parser(
        "compare",
        help="Run selected algorithms with one shared output format.",
    )
    compare.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _without_separator(values: List[str]) -> List[str]:
    return values[1:] if values and values[0] == "--" else values


def _has_option(values: List[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in values)


def _has_boolean_option(values: List[str], option: str) -> bool:
    negative = f"--no-{option[2:]}"
    return _has_option(values, option) or _has_option(values, negative)


def _configured_dataset_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # experiment.yaml documents dataset paths relative to openyield_v2.
    return (PACKAGE_ROOT / path).resolve()


def _apply_optimization_config(
    family: str,
    forwarded: List[str],
    config_path: Path,
) -> List[str]:
    """Use experiment.yaml as defaults without overriding explicit CLI flags."""
    import yaml

    config_path = config_path.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    datasets = raw.get("datasets", {})
    budget = raw.get("optimization_budget", {})
    shared = raw.get("shared_optimizer", {})
    resolved = list(forwarded)

    if not _has_option(resolved, "--problem-config"):
        resolved.extend(["--problem-config", str(config_path)])

    for option, key in (("--data-6t", "data_6t"), ("--data-10t", "data_10t")):
        value = datasets.get(key)
        if value is not None and not _has_option(resolved, option):
            resolved.extend(
                [option, str(_configured_dataset_path(str(value), config_path))]
            )

    option = "--max-evals"
    value = budget.get("tabpfn_design_queries_per_algorithm")
    if value is not None and not _has_option(resolved, option):
        resolved.extend([option, str(int(value))])

    if family == "proposed":
        proposed_options = (
            ("--coarse-evals", "proposed_coarse_evals"),
            ("--refine-steps", "proposed_refine_steps"),
            ("--representative-points", "proposed_representative_points"),
            ("--preferences-per-point", "proposed_preferences_per_point"),
            ("--print-every", "proposed_print_every"),
            ("--hard-audit-every", "proposed_hard_audit_every"),
        )
        for option, key in proposed_options:
            value = budget.get(key)
            if value is not None and not _has_option(resolved, option):
                resolved.extend([option, str(int(value))])

    shared_options = {
        "--bounds-lower-q": "bounds_lower_q",
        "--bounds-upper-q": "bounds_upper_q",
    }
    if family == "proposed":
        shared_options["--max-train-samples"] = "max_train_samples"
        test_size = shared.get("test_size")
        if test_size is not None:
            if not _has_option(resolved, "--test-ratio"):
                resolved.extend(["--test-ratio", str(test_size)])
            if not _has_option(resolved, "--train-ratio"):
                resolved.extend(["--train-ratio", str(1.0 - float(test_size))])
    else:
        shared_options["--test-size"] = "test_size"
        shared_options["--max-train-per-topology"] = "max_train_per_topology"
    for option, key in shared_options.items():
        value = shared.get(key)
        if value is not None and not _has_option(resolved, option):
            resolved.extend([option, str(value)])
    for option, key, default in (
        ("--balance-topologies", "balance_topologies", True),
        ("--verbose-library-training", "verbose_library_training", False),
    ):
        if not _has_boolean_option(resolved, option):
            enabled = bool(shared.get(key, default))
            resolved.append(option if enabled else f"--no-{option[2:]}")

    family_sections = {
        "evolutionary": {
            "--pop-size": "pop_size",
        },
        "bayesian": {
            "--init-samples": "init_samples",
            "--batch-size": "batch_size",
            "--candidate-pool": "candidate_pool",
            "--gp-max-train": "gp_max_train",
            "--gp-noise": "gp_noise",
            "--lcb-beta": "lcb_beta",
            "--parego-rho": "parego_rho",
            "--diversity-radius": "diversity_radius",
        },
        "proposed": {
            "--coarse-pop-size": "coarse_pop_size",
            "--preference-edge": "preference_edge",
            "--continuous-lr": "continuous_lr",
            "--lr-scheduler": "lr_scheduler",
            "--lr-min-ratio": "lr_min_ratio",
            "--early-stop-patience": "early_stop_patience",
            "--early-stop-min-delta": "early_stop_min_delta",
            "--early-stop-min-steps": "early_stop_min_steps",
            "--topology-lr": "topology_lr",
            "--discrete-lr": "discrete_lr",
            "--grad-clip-norm": "grad_clip_norm",
            "--refine-smoothing": "refine_smoothing",
            "--tau-start": "tau_start",
            "--tau-end": "tau_end",
            "--topology-init-bias": "topology_init_bias",
            "--discrete-init-bias": "discrete_init_bias",
            "--seed-ref-source": "seed_ref_source",
            "--constraint-penalty": "constraint_penalty",
        },
    }
    section = raw.get(family, {})
    for option, key in family_sections[family].items():
        value = section.get(key)
        if value is not None and not _has_option(resolved, option):
            resolved.extend([option, str(value)])
    if family == "proposed":
        for option, key, default in (
            ("--optimize-discrete", "optimize_discrete", False),
            ("--use-gumbel", "use_gumbel", False),
            (
                "--reset-torch-seed-each-step",
                "reset_torch_seed_each_step",
                False,
            ),
            ("--enable-simple-constraints", "enable_simple_constraints", True),
        ):
            if not _has_boolean_option(resolved, option):
                enabled = bool(section.get(key, default))
                resolved.append(option if enabled else f"--no-{option[2:]}")
    return resolved


def main() -> None:
    args = parse_args()
    if args.command == "optimize" and args.gpu_id is not None:
        if args.gpu_id < 0:
            raise ValueError("--gpu-id must be non-negative.")
        # This must happen before importing any optimizer, torch or TabPFN.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(
            f"[CUDA] Physical GPU {args.gpu_id} selected; "
            "it is visible inside Python as cuda:0."
        )
    forwarded = _without_separator(args.args)
    if args.command == "compare":
        from .optimizers import comparison

        _call_main(comparison.main, forwarded)
        return

    forwarded = _apply_optimization_config(
        args.family,
        forwarded,
        args.config,
    )
    if args.family == "evolutionary":
        from .optimizers import evolutionary

        _call_main(evolutionary.main, forwarded)
    elif args.family == "bayesian":
        from .optimizers import bayesian

        _call_main(bayesian.main, forwarded)
    elif args.family == "proposed":
        from .optimizers import proposed

        _call_main(proposed.main, forwarded)


if __name__ == "__main__":
    main()
