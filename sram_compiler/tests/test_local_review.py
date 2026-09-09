"""Live inspection must not corrupt a running simulation checkpoint."""

import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sram_compiler.sizing import local_review
from sram_compiler.sizing.local_review import _atomic_json
from sram_compiler.sizing.qualification import Case


class LocalReviewTests(unittest.TestCase):
    def test_concurrent_checkpoint_writers_leave_a_complete_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'review.json'
            records = [{'writer': i, 'payload': str(i) * 10000} for i in range(16)]
            with ThreadPoolExecutor(8) as pool:
                jobs = [pool.submit(_atomic_json, path, record) for record in records]
                for job in jobs:
                    job.result()
            self.assertIn(json.loads(path.read_text()), records)
            self.assertEqual(list(Path(temp).glob('*.tmp')), [])

    def test_calibration_execution_errors_are_not_reported_as_waveform_failures(self):
        passed = {'passed': True, 'measures': [{}], 'samples': [{'checks': {}, 'passed': True}]}
        scenarios = [
            ([{'passed': False, 'error': "Command 'mpiexec' timed out after 1800.0 seconds"},
              passed, passed], 'execution error'),
            ([{'passed': False, 'measures': [{}],
               'samples': [{'checks': {'cell_written': False}, 'passed': False}]},
              passed, passed], 'failed waveform acceptance'),
        ]
        for results, expected in scenarios:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temp, \
                    patch.object(local_review, 'run_case', side_effect=results), \
                    redirect_stdout(io.StringIO()):
                record = local_review.review_architecture(Case(8, 4), Path(temp), 'Xyce', 3, 10)
            self.assertTrue(record['complete'])
            self.assertIn(expected, record['calibration_error'])
            self.assertEqual(record['verification'], [])
        self.assertIn('timed out', scenarios[0][0][0]['error'])


if __name__ == '__main__':
    unittest.main()
