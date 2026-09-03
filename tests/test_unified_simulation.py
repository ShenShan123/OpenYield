from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from yield_estimation import SimulationRunner, TargetCellTestbenchAdapter
from yield_estimation.unified.simulation import BudgetExceeded


class TupleTestbench:
    def __init__(self) -> None:
        self.sim_path = ""
        self.received = None

    def run_mc_simulation(
        self, operation="read", target_row=0, target_col=0,
        mc_runs=100, temperature=27, vars=None,
    ):
        self.received = np.asarray(vars)
        delay = self.received[:, 0]
        zeros = np.zeros(mc_runs)
        return delay, zeros, zeros, zeros


class LegacySampleTestbench:
    def sample(self, values, count):
        assert count == len(values)
        return values[:, 1]


class CsvTestbench:
    def __init__(self, path: Path) -> None:
        self.path = path

    def sample(self, values, count):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["read_delay_s"])
            writer.writeheader()
            for value in values[:, 0]:
                writer.writerow({"read_delay_s": value})
        return self.path


class FailingTestbench:
    def run_mc_simulation(self, **kwargs):
        raise RuntimeError("Xyce exited 1")


class FailOnceTestbench:
    def __init__(self):
        self.calls = 0

    def run_mc_simulation(self, mc_runs=100, vars=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary Xyce error")
        return np.asarray(vars)[:, 0]


class RawMeasurementTestbench:
    def __init__(self, *, raise_after_run=False, values=None):
        self.sim_path = ""
        self.raise_after_run = raise_after_run
        self.values = values

    def run_mc_simulation(self, mc_runs=100, **kwargs):
        root = Path(self.sim_path)
        root.mkdir(parents=True, exist_ok=True)
        for index in range(mc_runs):
            value = self.values[index] if self.values is not None else (index + 1) * 1e-11
            (root / f"read_tb.sp.mt{index}").write_text(
                f"TREAD_TOTAL = {value}\n", encoding="utf-8"
            )
        if self.raise_after_run:
            raise KeyError("legacy CSV aggregation failed")
        zeros = np.zeros(mc_runs)
        return zeros, zeros, zeros, zeros


class SimulationRunnerTests(unittest.TestCase):
    def test_snm_parser_has_no_optional_regex_dependency(self):
        source = (
            Path(__file__).parents[1]
            / "sram_compiler/testbenches/snm.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("from regex import", source)

    def test_native_tuple_and_normalized_to_physical(self):
        with tempfile.TemporaryDirectory() as directory:
            testbench = TupleTestbench()
            runner = SimulationRunner(
                testbench,
                directory,
                input_space="standard_normal",
                nominal=np.array([10.0, 20.0]),
                sigma=np.array([2.0, 4.0]),
            )
            runner.reset_budget(2)
            batch = runner.run_mc_simulation(
                mc_runs=2, vars=np.array([[0.0, 1.0], [1.0, -1.0]])
            )
            np.testing.assert_allclose(testbench.received, [[10.0, 24.0], [12.0, 16.0]])
            np.testing.assert_allclose(batch.values, [10.0, 12.0])
            self.assertEqual(batch.statuses, ("ok", "ok"))
            self.assertTrue(Path(batch.artifacts[0]).exists())
            with Path(batch.artifacts[0]).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["input_1"]), 1.0)
            self.assertEqual(float(rows[0]["physical_1"]), 24.0)

    def test_legacy_sample_and_csv_return(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.array([[1.0, 3.0], [2.0, 4.0]])
            runner = SimulationRunner(LegacySampleTestbench(), Path(directory) / "sample")
            runner.reset_budget(2)
            np.testing.assert_allclose(
                runner.run_mc_simulation(mc_runs=2, vars=values).values,
                [3.0, 4.0],
            )

            csv_path = Path(directory) / "backend.csv"
            csv_runner = SimulationRunner(
                CsvTestbench(csv_path), Path(directory) / "csv", metric="read_delay_s"
            )
            csv_runner.reset_budget(2)
            np.testing.assert_allclose(
                csv_runner.run_mc_simulation(mc_runs=2, vars=values).values,
                [1.0, 2.0],
            )

    def test_failures_are_not_physical_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = SimulationRunner(FailingTestbench(), directory)
            runner.reset_budget(3)
            batch = runner.run_mc_simulation(mc_runs=3, vars=np.ones((3, 2)))
            self.assertTrue(np.isnan(batch.values).all())
            self.assertEqual(batch.simulator_errors, 3)
            self.assertEqual(runner.ledger.simulator_errors, 3)

    def test_strict_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = SimulationRunner(lambda x: x[:, 0], directory)
            runner.reset_budget(2)
            runner.run_mc_simulation(mc_runs=2, vars=np.ones((2, 1)))
            with self.assertRaises(BudgetExceeded):
                runner.run_mc_simulation(mc_runs=1, vars=np.ones((1, 1)))

    def test_retry_is_charged_and_cannot_exceed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FailOnceTestbench()
            runner = SimulationRunner(backend, directory, max_retries=1)
            runner.reset_budget(4)
            batch = runner.run_mc_simulation(mc_runs=2, vars=np.ones((2, 1)))
            self.assertEqual(batch.statuses, ("ok", "ok"))
            self.assertEqual(runner.ledger.charged_calls, 4)
            self.assertEqual(runner.ledger.retry_calls, 2)
            self.assertEqual(runner.ledger.simulator_errors, 2)

    def test_named_metric_uses_native_measurements_for_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = SimulationRunner(
                RawMeasurementTestbench(), directory, metric="TREAD_TOTAL"
            )
            runner.reset_budget(2)
            batch = runner.run_mc_simulation(mc_runs=2, vars=np.ones((2, 1)))
            np.testing.assert_allclose(batch.values, [1e-11, 2e-11])
            self.assertEqual(batch.statuses, ("ok", "ok"))

    def test_native_measurements_recover_post_xyce_aggregation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = SimulationRunner(
                RawMeasurementTestbench(raise_after_run=True),
                directory,
                metric="TREAD_TOTAL",
            )
            runner.reset_budget(2)
            batch = runner.run_mc_simulation(mc_runs=2, vars=np.ones((2, 1)))
            np.testing.assert_allclose(batch.values, [1e-11, 2e-11])
            self.assertEqual(runner.ledger.simulator_errors, 0)

    def test_finite_negative_native_measurement_is_physical_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = SimulationRunner(
                RawMeasurementTestbench(values=[-1.0]),
                directory,
                metric="TREAD_TOTAL",
            )
            runner.reset_budget(1)
            batch = runner.run_mc_simulation(mc_runs=1, vars=np.ones((1, 1)))
            self.assertEqual(batch.statuses, ("ok",))
            self.assertEqual(batch.values[0], -1.0)
            self.assertEqual(runner.ledger.simulator_errors, 0)

    def test_target_cell_adapter_expands_only_selected_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = TupleTestbench()
            adapter = TargetCellTestbenchAdapter(
                backend,
                np.array([10.0, 20.0]),
                num_rows=2,
                num_cols=2,
                target_row=1,
                target_col=0,
            )
            runner = SimulationRunner(adapter, directory)
            runner.reset_budget(1)
            runner.run_mc_simulation(mc_runs=1, vars=np.array([[30.0, 40.0]]))
            np.testing.assert_allclose(
                backend.received,
                [[10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 10.0, 20.0]],
            )


if __name__ == "__main__":
    unittest.main()
