"""Repeated Xyce .PRINT probes must preserve samples or reject conflicting data."""

import io
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

import matplotlib

matplotlib.use('Agg')
import numpy as np

from utils.waveforms import read_prn_with_preprocess, split_blocks
from utils.plotting import process_simulation_data


class DuplicateWaveformTests(unittest.TestCase):
    def test_identical_repeated_probes_preserve_values_and_sample_boundaries(self):
        for indexed in (False, True):
            for axis, times, kind in (
                    ('TIME', [0., 1e-9, 2e-9] * 2, 'tran'),
                    ('{U}', [-.1, 0., .1] * 2, 'dc')):
                with self.subTest(indexed=indexed, kind=kind), tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / 'duplicate.prn'
                    # A distinct far terminal must survive even if its values
                    # happen to equal PRE in this fixture.
                    header = ('Index ' if indexed else '') + axis + ' V(PRE) V(PRE_LINE_FAR) V(PRE) V(Q)\n'
                    rows = ''.join((f'{i} ' if indexed else '') +
                                   f'{when} {i/10} {i/10} {i/10} {1-i/10}\n'
                                   for i, when in enumerate(times))
                    original = header + rows + 'End of Xyce(TM) Simulation\n'
                    path.write_text(original)
                    data, analysis = read_prn_with_preprocess(path)
                    self.assertEqual(analysis, kind)
                    self.assertEqual(list(data.columns), [axis, 'V(PRE)', 'V(PRE_LINE_FAR)', 'V(Q)'])
                    np.testing.assert_array_equal(data['V(PRE)'], np.arange(6)/10)
                    np.testing.assert_array_equal(data['V(Q)'], 1-np.arange(6)/10)
                    self.assertEqual([len(block) for block in split_blocks(data, analysis, 2)], [3, 3])
                    self.assertEqual(path.read_text(), original)

    def test_conflicting_repeated_probe_is_rejected_even_in_later_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'conflict.prn'
            path.write_text('TIME V(PRE) V(PRE)\n'
                            '0 1 1\n1e-9 1 1\n0 1 1\n1e-9 1 0.99\n')
            with self.assertRaisesRegex(ValueError, r'Conflicting or nonfinite duplicate waveform column: V\(PRE\)'):
                read_prn_with_preprocess(path)

    def test_nonfinite_duplicate_values_cannot_be_collapsed(self):
        for value in ('nan', 'inf', '-inf'):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'nonfinite.prn'
                path.write_text(f'TIME V(PRE) V(PRE)\n0 1 1\n1e-9 {value} {value}\n')
                with self.assertRaisesRegex(ValueError, 'nonfinite duplicate waveform column'):
                    read_prn_with_preprocess(path)

    def test_repeated_probes_plot_without_changing_input(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            path = Path(temp) / 'plot.prn'
            original = ('Index TIME V(PRE) V(Q) V(PRE)\n'
                        '0 0 1 0 1\n1 1e-9 0 1 0\n'
                        '0 0 1 0 1\n1 1e-9 0 0.9 0\n')
            path.write_text(original)
            output = Path(temp) / 'plot.png'
            self.assertTrue(process_simulation_data(path, 2, output, ['V(PRE)', 'V(Q)']))
            self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(path.read_text(), original)


if __name__ == '__main__':
    unittest.main()
