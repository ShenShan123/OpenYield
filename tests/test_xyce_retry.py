"""A numerical retry must not hide failed samples or discard their evidence."""

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from utils.xyce import execute_xyce


class XyceRetryTests(unittest.TestCase):
    def test_timeout_keeps_partial_solver_output_and_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            deck = Path(temp) / 'deck.sp'
            deck.write_text('Circuit\n.END\n')
            failure = subprocess.TimeoutExpired(['Xyce'], 1, output=b'DC still solving', stderr=b'detail')
            with patch('utils.xyce.subprocess.run', side_effect=failure), self.assertRaises(subprocess.TimeoutExpired):
                execute_xyce(deck, ['Xyce', str(deck)], timeout=1)
            log = Path(str(deck) + '.log').read_text()
            self.assertIn('timed out', log)
            self.assertIn('DC still solving', log)

    def test_native_exit_zero_failure_retries_with_the_same_seed_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            deck = Path(temp) / 'deck.sp'
            original = 'Circuit\n.SAMPLING useExpr=true\n.OPTIONS SAMPLES NUMSAMPLES=2\n.END\n'
            deck.write_text(original)
            waveform = Path(str(deck) + '.prn')
            waveform.write_text('partial sample')
            failed = subprocess.CompletedProcess(['Xyce'], 0,
                       'Seeding random number generator with 123\nDC Operating Point Failed', '')
            passed = subprocess.CompletedProcess(['Xyce'], 0, 'complete', '')
            with patch('utils.xyce.subprocess.run', side_effect=[failed, passed]) as run:
                result = execute_xyce(deck, ['Xyce', str(deck)])
            self.assertEqual(result.returncode, 0)
            self.assertEqual(run.call_count, 2)
            self.assertIn('SEARCHMETHOD=2', deck.read_text())
            self.assertIn('SAMPLES SEED=123', deck.read_text())
            self.assertEqual((Path(temp) / 'dcop_attempt/deck.sp').read_text(), original)
            self.assertEqual((Path(temp) / 'dcop_attempt/deck.sp.prn').read_text(), 'partial sample')

    def test_unrecoverable_or_unseeded_failures_are_not_success_or_resampled(self):
        for options in ('.SAMPLING useExpr=true', '.OPTIONS NONLIN SEARCHMETHOD=2'):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temp:
                deck = Path(temp) / 'deck.sp'
                deck.write_text(f'Circuit\n{options}\n.END\n')
                failed = subprocess.CompletedProcess(['Xyce'], 0, 'DC Operating Point Failed', '')
                with patch('utils.xyce.subprocess.run', return_value=failed) as run:
                    result = execute_xyce(deck, ['Xyce', str(deck)])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(run.call_count, 1)

    def test_failed_retry_stays_failed_and_electrical_results_are_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            deck = Path(temp) / 'deck.sp'
            deck.write_text('Circuit\n.END\n')
            failed = subprocess.CompletedProcess(['Xyce'], 1, 'DC Operating Point Failed', '')
            with patch('utils.xyce.subprocess.run', side_effect=[failed, failed]) as run:
                self.assertNotEqual(execute_xyce(deck, ['Xyce', str(deck)]).returncode, 0)
                self.assertEqual(run.call_count, 2)
            electrical = subprocess.CompletedProcess(['Xyce'], 0, 'TREAD_TOTAL = FAILED', '')
            with patch('utils.xyce.subprocess.run', return_value=electrical) as run:
                self.assertEqual(execute_xyce(deck, ['Xyce', str(deck)]).returncode, 0)
                self.assertEqual(run.call_count, 1)
