"""Acceptance mechanics: missing evidence fails; timing uses all three decks."""

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sram_compiler.sizing.qualification import (
    Case,
    _cross,
    read_measurements,
    run_case,
    waveform_blocks,
)
from sram_compiler.sizing.timing import TimingConfig, timing_from_measurements


class QualificationTests(unittest.TestCase):
    def test_incomplete_transient_retries_same_case_at_tighter_resolution(self):
        calls = []
        def simulate(case, output_root, xyce, *, timing, timeout, max_step=10e-12):
            calls.append((case, timing, timeout, max_step))
            directory = Path(output_root) / case.name
            directory.mkdir(parents=True, exist_ok=True)
            passed = len(calls) == 2
            result = {'directory': str(directory), 'passed': passed, 'max_time_step': max_step,
                      'signature': str(len(calls)), 'samples': [
                          {'passed': True, 'checks': {'cell_written': True}} if passed else
                          {'passed': False, 'error': 'Incomplete waveform'}]}
            (directory / 'xyce.log').write_text('Time step too small' if not passed else 'Simulation completed')
            (directory / 'result.json').write_text(json.dumps(result))
            return result
        with tempfile.TemporaryDirectory() as temp, patch(
            'sram_compiler.sizing.qualification._run_case_unlocked', side_effect=simulate
        ):
            case = Case(128, 32, samples=10, variation='shared', seed=2026)
            result = run_case(case, temp, 'unused', timing=None, timeout=123)
            self.assertTrue(result['passed'])
            self.assertEqual(calls, [(case, None, 123, 10e-12), (case, None, 123, 5e-12)])
            original = Path(result['original_attempt']['directory']) / 'result.json'
            self.assertFalse(json.loads(original.read_text())['passed'])
            self.assertEqual(result['original_attempt']['sample_errors'], ['Incomplete waveform'])

    def test_electrical_or_nonfinite_failures_cannot_trigger_retry(self):
        failures = [
            [{'passed': False, 'error': 'Nonfinite waveform'}],
            [{'passed': False, 'error': 'Incomplete waveform'},
             {'passed': False, 'checks': {'cell_written': False}}],
        ]
        for samples in failures:
            with self.subTest(samples=samples), tempfile.TemporaryDirectory() as temp:
                case = Case(8, 4)
                directory = Path(temp) / case.name
                directory.mkdir()
                (directory / 'xyce.log').write_text('Time step too small')
                result = {'passed': False, 'samples': samples}
                with patch('sram_compiler.sizing.qualification._run_case_unlocked', return_value=result) as run:
                    self.assertIs(run_case(case, temp, 'unused'), result)
                    self.assertEqual(run.call_count, 1)

    def test_nominal_timestep_failure_preserves_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Case(100, 50)
            original = Path(temp) / case.name
            original.mkdir()
            (original / 'xyce.log').write_text('Time step too small')
            retry = original / 'retry_5ps' / case.name
            retry.mkdir(parents=True)
            failed = {'passed': False, 'error': 'Xyce exit 1; see xyce.log',
                      'simulator_returncode': 1, 'directory': str(original),
                      'signature': 'failed-inputs', 'max_time_step': 10e-12}
            passed = {'passed': True, 'simulator_returncode': 0, 'directory': str(retry)}
            with patch('sram_compiler.sizing.qualification._run_case_unlocked', side_effect=[failed, passed]):
                result = run_case(case, temp, 'unused')
            self.assertTrue(result['passed'])
            self.assertEqual(result['original_attempt']['simulator_returncode'], 1)
            self.assertEqual(result['original_attempt']['signature'], 'failed-inputs')

    def test_identical_case_resumes_cannot_write_concurrently(self):
        active, peak = 0, 0
        guard = threading.Lock()
        def simulate(*args, **kwargs):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(.03)
            with guard:
                active -= 1
            return {'passed': True}
        with tempfile.TemporaryDirectory() as temp, patch(
            'sram_compiler.sizing.qualification._run_case_unlocked', side_effect=simulate
        ), ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(run_case, Case(8, 4), temp, 'unused') for _ in range(2)]
            self.assertTrue(all(job.result()['passed'] for job in jobs))
        self.assertEqual(peak, 1)

    def test_interpolated_crossing_and_missing_crossing(self):
        t = np.array([0., 1., 2., 3.])
        y = np.array([0., 0., 1., 0.])
        self.assertEqual(_cross(t, y, .5, 0, 3), 1.5)
        self.assertEqual(_cross(t, y, .5, 0, 3, False), 2.5)
        with self.assertRaises(ValueError):
            _cross(t, y, 2, 0, 3)

    def test_waveform_sample_resets_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'deck.prn'
            path.write_text('Index TIME V(Q)\n0 0 0\n1 1 1\n0 0 0\n1 1 0\nEnd of Xyce(TM) Simulation\n')
            blocks = list(waveform_blocks(path))
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[1][1][-1, -1], 0)
            path.write_text('Index TIME V(Q)\n0 0\n')
            with self.assertRaises(ValueError):
                list(waveform_blocks(path))

    def test_failed_measure_is_not_a_negative_delay(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'deck.mt0'
            path.write_text('TWRITE_TOTAL = FAILED\nTRESTORE = 2.0e-10\n')
            values = read_measurements(path)
            self.assertIsNone(values['TWRITE_TOTAL'])
            self.assertEqual(values['TRESTORE'], 2e-10)

    def test_timing_uses_slowest_write_and_phase_and_quantizes_up(self):
        read = {'TCLK_WLEN': 0.2e-9, 'TREAD_TOTAL': 0.6e-9, 'TRESTORE': 0.3e-9, 'TCLK_DEC': 0.2e-9}
        ss = {'TCLK_WLEN': 0.2e-9, 'TWRITE_TOTAL': 0.3e-9, 'TRESTORE': 0.4e-9}
        sf = {'TCLK_WLEN': 0.2e-9, 'TWRITE_TOTAL': 0.7e-9, 'TRESTORE': 0.3e-9}
        timing = timing_from_measurements(read, [ss, sf])
        self.assertAlmostEqual(timing.low_write, .9e-9, places=18)
        self.assertGreaterEqual(timing.t_period, 2.25e-9)
        self.assertLess(timing.t_period, 2.3e-9 + 1e-18)
        with self.assertRaises(ValueError):
            timing_from_measurements(read, [ss])
        with self.assertRaises(ValueError):
            TimingConfig(1e-9, 1e-9, .5e-9, .5e-9)
        sf['TCLK_DEC'] = float('nan')
        with self.assertRaises(ValueError):
            timing_from_measurements(read, [ss, sf])


if __name__ == '__main__':
    unittest.main()
