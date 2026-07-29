#!/usr/bin/env python3
"""Run the default algorithm comparison configured in experiment.yaml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "experiment.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Optional failed comparison timestamp directory to resume.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every resolved child command without launching an optimizer.",
    )
    return parser.parse_args()


def resolve_config(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    return path


def append_option(command: List[str], option: str, value: Any) -> None:
    if value is not None:
        command.extend([option, str(value)])


def append_boolean_option(
    command: List[str],
    option: str,
    value: Any,
) -> None:
    if value is None:
        return
    command.append(option if bool(value) else f"--no-{option[2:]}")


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    settings: Dict[str, Any] = raw.get("comparison", {})
    if not settings:
        raise ValueError(
            f"{config_path} has no comparison section. "
            "Copy the documented comparison defaults into that YAML."
        )

    command = [
        sys.executable,
        "-u",
        "-m",
        "size_optimization.openyield_v2.run",
        "compare",
        "--",
        "--config",
        str(config_path),
    ]
    comparison_map = {
        "algorithms": "--algorithms",
        "device": "--device",
        "gpu_id": "--gpu-id",
        "seed": "--seed",
        "output_dir": "--output-dir",
        "max_evals": "--max-evals",
    }
    shared_map = {
        "test_size": "--test-size",
        "max_train_per_topology": "--max-train-per-topology",
        "max_train_samples": "--max-train-samples",
        "bounds_lower_q": "--bounds-lower-q",
        "bounds_upper_q": "--bounds-upper-q",
    }
    evolutionary_map = {
        "pop_size": "--pop-size",
    }
    bayesian_map = {
        "init_samples": "--init-samples",
        "batch_size": "--batch-size",
        "candidate_pool": "--candidate-pool",
        "gp_max_train": "--gp-max-train",
        "gp_noise": "--gp-noise",
        "lcb_beta": "--lcb-beta",
        "parego_rho": "--parego-rho",
        "diversity_radius": "--diversity-radius",
    }
    proposed_map = {
        "coarse_pop_size": "--coarse-pop-size",
        "preference_edge": "--preference-edge",
        "continuous_lr": "--continuous-lr",
        "lr_scheduler": "--lr-scheduler",
        "lr_min_ratio": "--lr-min-ratio",
        "early_stop_patience": "--early-stop-patience",
        "early_stop_min_delta": "--early-stop-min-delta",
        "early_stop_min_steps": "--early-stop-min-steps",
        "topology_lr": "--topology-lr",
        "discrete_lr": "--discrete-lr",
        "grad_clip_norm": "--grad-clip-norm",
        "refine_smoothing": "--refine-smoothing",
        "tau_start": "--tau-start",
        "tau_end": "--tau-end",
        "topology_init_bias": "--topology-init-bias",
        "discrete_init_bias": "--discrete-init-bias",
        "seed_ref_source": "--seed-ref-source",
        "constraint_penalty": "--constraint-penalty",
    }
    budget_map = {
        "proposed_coarse_evals": "--coarse-evals",
        "proposed_refine_steps": "--refine-steps",
        "proposed_representative_points": "--representative-points",
        "proposed_preferences_per_point": "--preferences-per-point",
        "proposed_print_every": "--print-every",
        "proposed_hard_audit_every": "--hard-audit-every",
    }
    sections = (
        (settings, comparison_map),
        (raw.get("shared_optimizer", {}), shared_map),
        (raw.get("evolutionary", {}), evolutionary_map),
        (raw.get("bayesian", {}), bayesian_map),
        (raw.get("proposed", {}), proposed_map),
        (raw.get("optimization_budget", {}), budget_map),
    )
    for section, option_map in sections:
        for key, option in option_map.items():
            value = section.get(key)
            if key == "output_dir" and value is not None:
                output_path = Path(str(value)).expanduser()
                if not output_path.is_absolute():
                    output_path = (PACKAGE_ROOT / output_path).resolve()
                value = output_path
            append_option(command, option, value)

    shared = raw.get("shared_optimizer", {})
    proposed = raw.get("proposed", {})
    for key, option, default in (
        ("balance_topologies", "--balance-topologies", True),
        ("verbose_library_training", "--verbose-library-training", False),
    ):
        append_boolean_option(command, option, shared.get(key, default))
    for key, option, default in (
        ("optimize_discrete", "--optimize-discrete", False),
        ("use_gumbel", "--use-gumbel", False),
        ("reset_torch_seed_each_step", "--reset-torch-seed-each-step", False),
        ("enable_simple_constraints", "--enable-simple-constraints", True),
    ):
        append_boolean_option(command, option, proposed.get(key, default))
    if args.resume_dir is not None:
        append_option(command, "--resume-dir", args.resume_dir)
    if args.dry_run:
        command.append("--dry-run")

    print("Running configured comparison:")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
