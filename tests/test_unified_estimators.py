from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from yield_estimation import EXPERIMENTAL_ALGORITHMS, STABLE_ALGORITHMS, YieldEstimator


EXACT_PFAIL = 0.15865525393145707


def linear_simulator(samples: np.ndarray) -> np.ndarray:
    return samples[:, 0]


class EstimatorTests(unittest.TestCase):
    def _estimator(self, directory: str, algorithm: str, **algo_params):
        return YieldEstimator(
            model=linear_simulator,
            algorithm_choice=algorithm,
            basic_params={
                "mean": np.zeros(2),
                "covariance": np.eye(2),
                "threshold": 1.0,
                "seed": 123,
            },
            algo_params=algo_params,
            spice_params={"run_root": Path(directory) / algorithm.lower()},
        )

    def test_all_stable_estimators_use_exact_budget_and_match_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            for algorithm in STABLE_ALGORITHMS:
                with self.subTest(algorithm=algorithm):
                    estimator = self._estimator(
                        directory, algorithm, pilot_fraction=0.4, max_components=16
                    )
                    result = estimator.run(max_num=2000)
                    self.assertEqual(result.charged_calls, 2000)
                    self.assertEqual(result.live_calls, 2000)
                    self.assertEqual(result.simulator_errors, 0)
                    self.assertEqual(result.status, "ok")
                    self.assertLess(abs(result.failure_probability - EXACT_PFAIL), 0.04)

    def test_all_stable_estimators_share_one_small_budget_call_format(self):
        with tempfile.TemporaryDirectory() as directory:
            results = {}
            for algorithm in STABLE_ALGORITHMS:
                estimator = YieldEstimator(
                    model=linear_simulator,
                    algorithm_choice=algorithm,
                    basic_params={
                        "dimension": 2,
                        "threshold": 0.5,
                        "seed": 19,
                    },
                    algo_params={
                        "pilot_fraction": 0.5,
                        "is_batch_size": 24,
                        "max_components": 8,
                    },
                    spice_params={
                        "run_root": Path(directory) / f"same_format_{algorithm.lower()}",
                    },
                )
                results[algorithm] = estimator.run(max_num=120)

            for algorithm, result in results.items():
                with self.subTest(algorithm=algorithm):
                    self.assertEqual(result.algorithm, algorithm)
                    self.assertEqual(result.budget_target, 120)
                    self.assertEqual(result.charged_calls, 120)
                    self.assertEqual(result.live_calls, 120)
                    self.assertEqual(result.simulator_errors, 0)
                    self.assertIn(result.status, {"ok", "ok_zero_failure"})

    def test_acs_defaults_to_original_and_accepts_improved(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self._estimator(
                directory, "ACS", pilot_fraction=0.5, is_batch_size=20
            ).run(max_num=120)
            improved = YieldEstimator(
                model=linear_simulator,
                algorithm_choice="ACS",
                basic_params={
                    "dimension": 2,
                    "threshold": 0.5,
                    "seed": 23,
                },
                algo_params={
                    "mode": "improved",
                    "pilot_fraction": 0.5,
                    "is_batch_size": 20,
                },
                spice_params={"run_root": Path(directory) / "acs_improved"},
            ).run(max_num=120)

            self.assertEqual(original.metadata["mode"], "original")
            self.assertEqual(improved.metadata["mode"], "improved")
            self.assertEqual(original.charged_calls, 120)
            self.assertEqual(improved.charged_calls, 120)

    def test_acs_rejects_unknown_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "original.*improved"):
                self._estimator(directory, "ACS", mode="unknown")

    def test_method_specific_experimental_ports_use_the_common_contract(self):
        expected_rules = {
            "FUSIS": "surrogate_probability_mcmc_correction",
            "OPT": "failure_trained_flow_target_mixture",
        }
        self.assertTrue(set(expected_rules).issubset(EXPERIMENTAL_ALGORITHMS))
        with tempfile.TemporaryDirectory() as directory:
            for algorithm, rule in expected_rules.items():
                with self.subTest(algorithm=algorithm):
                    estimator = YieldEstimator(
                        model=linear_simulator,
                        algorithm_choice=algorithm,
                        basic_params={
                            "dimension": 2,
                            "threshold": 0.5,
                            "seed": 29,
                        },
                        algo_params={
                            "pilot_fraction": 0.5,
                            "is_batch_size": 20,
                            "verification_batch_size": 20,
                            "surrogate_mc_samples": 100,
                            "flow_backend": "gaussian",
                        },
                        spice_params={
                            "run_root": Path(directory) / algorithm.lower()
                        },
                    )
                    result = estimator.run(max_num=80)
                    self.assertEqual(result.charged_calls, 80)
                    self.assertEqual(result.live_calls, 80)
                    self.assertEqual(result.simulator_errors, 0)
                    self.assertEqual(result.metadata["proposal_rule"], rule)
                    self.assertEqual(
                        result.metadata["implementation_family"],
                        "CrossTopo baseline port",
                    )

    def test_zero_failure_status(self):
        with tempfile.TemporaryDirectory() as directory:
            estimator = YieldEstimator(
                model=lambda samples: np.zeros(len(samples)),
                algorithm_choice="MC",
                basic_params={"dimension": 2, "threshold": 1.0, "seed": 1},
                spice_params={"run_root": Path(directory) / "zero"},
            )
            result = estimator.run(100)
            self.assertEqual(result.status, "ok_zero_failure")
            self.assertEqual(result.failure_probability, 0.0)

    def test_nonpositive_metric_can_be_a_physical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            estimator = YieldEstimator(
                model=lambda samples: np.where(samples[:, 0] > 0, 1.0, -1.0),
                algorithm_choice="MC",
                basic_params={"dimension": 2, "threshold": 2.0, "seed": 1},
                algo_params={"failure_if_nonpositive": True},
                spice_params={"run_root": Path(directory) / "functional"},
            )
            result = estimator.run(1000)
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.simulator_errors, 0)
            self.assertGreater(result.failure_probability, 0.4)

    def test_artifacts_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            estimator = self._estimator(directory, "MC")
            estimator.run(100)
            root = Path(directory) / "mc"
            for name in ("config.json", "result.json", "summary.csv", "DONE", "MANIFEST.sha256"):
                self.assertTrue((root / name).exists(), name)
            for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), digest)

    def test_bibd_keeps_condition_results_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            estimator = YieldEstimator(
                model=linear_simulator,
                algorithm_choice="BIBD",
                basic_params={"dimension": 2, "threshold": 1.0, "seed": 10},
                algo_params={
                    "batch_size": 17,
                    "failure_if_nonpositive": True,
                    "conditions": [
                        {"name": "cold", "threshold": 0.5},
                        {"name": "hot", "threshold": 1.0},
                    ]
                },
                spice_params={"run_root": Path(directory) / "bibd"},
            )
            result = estimator.run(1000)
            self.assertEqual(result.charged_calls, 1000)
            self.assertEqual(set(result.conditions), {"cold", "hot"})
            self.assertTrue(all(item.simulator_errors == 0 for item in result.conditions.values()))
            self.assertGreater(
                result.conditions["cold"].failure_probability,
                result.conditions["hot"].failure_probability,
            )

    def test_algorithm_source_has_no_direct_simulator_management(self):
        source = (Path(__file__).parents[1] / "yield_estimation/unified/estimators.py").read_text()
        self.assertNotIn("subprocess", source)
        self.assertNotIn("sim_path", source)


if __name__ == "__main__":
    unittest.main()
