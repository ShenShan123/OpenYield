"""CPU-only regression tests for configuration, budgets and Pareto semantics."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from size_optimization.openyield_v2.optimization_spec import (
    pareto_front_from_evaluations,
)
from size_optimization.openyield_v2.optimizers import comparison
from size_optimization.openyield_v2.run import _apply_optimization_config
from size_optimization.openyield_v2 import run_experiment


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_ROOT / "configs" / "experiment.yaml"


class ConfigurationTests(unittest.TestCase):
    def _comparison_args(self) -> argparse.Namespace:
        with patch(
            "sys.argv",
            [
                "comparison",
                "--config",
                str(CONFIG),
                "--algorithms",
                "NSGA2,PAREGO,PROPOSED",
            ],
        ):
            args = comparison.parse_args()
        raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        for section_name in (
            "shared_optimizer",
            "evolutionary",
            "bayesian",
            "proposed",
        ):
            for key, value in raw[section_name].items():
                if hasattr(args, key):
                    setattr(args, key, value)
        return args

    def test_default_proposed_budget_is_exact(self) -> None:
        args = self._comparison_args()
        budget = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
            "optimization_budget"
        ]
        options = comparison.proposed_budget_options(args, budget, 1000)
        self.assertEqual(options["coarse_evals"], 500)
        self.assertEqual(options["refine_steps"], 47)

    def test_child_commands_forward_algorithm_hyperparameters(self) -> None:
        args = self._comparison_args()
        budget = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
            "optimization_budget"
        ]
        proposed = comparison.proposed_budget_options(args, budget, 1000)
        temporary = PACKAGE_ROOT / "runs" / "_command_test"

        nsga2 = comparison.build_command(
            "NSGA2",
            args=args,
            config_path=CONFIG,
            max_evals=1000,
            temporary_root=temporary,
            proposed_options=proposed,
        )
        self.assertIn("--pop-size", nsga2)
        self.assertIn("--bounds-lower-q", nsga2)

        parego = comparison.build_command(
            "PAREGO",
            args=args,
            config_path=CONFIG,
            max_evals=1000,
            temporary_root=temporary,
            proposed_options=proposed,
        )
        for option in (
            "--gp-max-train",
            "--gp-noise",
            "--lcb-beta",
            "--parego-rho",
            "--diversity-radius",
        ):
            self.assertIn(option, parego)

        proposed_command = comparison.build_command(
            "PROPOSED",
            args=args,
            config_path=CONFIG,
            max_evals=1000,
            temporary_root=temporary,
            proposed_options=proposed,
        )
        for option in (
            "--continuous-lr",
            "--lr-min-ratio",
            "--grad-clip-norm",
            "--tau-start",
            "--seed-ref-source",
            "--constraint-penalty",
            "--no-optimize-discrete",
        ):
            self.assertIn(option, proposed_command)

    def test_direct_family_yaml_defaults_do_not_override_cli(self) -> None:
        resolved = _apply_optimization_config(
            "evolutionary",
            ["--pop-size", "20"],
            CONFIG,
        )
        self.assertEqual(resolved.count("--pop-size"), 1)
        self.assertEqual(resolved[resolved.index("--pop-size") + 1], "20")
        self.assertIn("--bounds-lower-q", resolved)

    def test_run_experiment_forwards_all_config_sections(self) -> None:
        with patch("sys.argv", ["run_experiment"]), patch(
            "size_optimization.openyield_v2.run_experiment.subprocess.run"
        ) as mocked_run:
            run_experiment.main()
        command = mocked_run.call_args.args[0]
        for option in (
            "--pop-size",
            "--gp-max-train",
            "--continuous-lr",
            "--lr-min-ratio",
            "--coarse-evals",
            "--hard-audit-every",
            "--balance-topologies",
            "--no-optimize-discrete",
        ):
            self.assertIn(option, command)

    def test_comparison_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_base = Path(temp_dir) / "comparison"
            with patch(
                "sys.argv",
                [
                    "comparison",
                    "--config",
                    str(CONFIG),
                    "--algorithms",
                    "NSGA2",
                    "--output-dir",
                    str(output_base),
                    "--dry-run",
                ],
            ):
                planned = comparison.run()
            self.assertFalse(planned.exists())


class ParetoTests(unittest.TestCase):
    def test_front_uses_only_recorded_feasible_evaluations(self) -> None:
        evaluations = pd.DataFrame(
            {
                "evaluation": [1, 2, 3, 4],
                "constraint_valid": [True, True, True, False],
                "power": [1.0, 2.0, 0.5, 0.1],
                "snm": [1.0, 2.0, 0.5, 10.0],
            }
        )
        objectives = [
            {"source": "power", "direction": "min"},
            {"source": "snm", "direction": "max"},
        ]
        front = pareto_front_from_evaluations(evaluations, objectives)
        self.assertEqual(front["evaluation"].tolist(), [3, 1, 2])
        self.assertNotIn(4, front["evaluation"].tolist())


if __name__ == "__main__":
    unittest.main()
