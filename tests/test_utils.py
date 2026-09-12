"""Shared utility APIs must preserve simulator data and work without a display."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    generate_mc_statistics, parse_mc_measurements, process_simulation_data,
    read_prn_with_preprocess, split_blocks,
)
from utils.plotting import plot_delay, plot_leak_delay, plot_power, plot_rc_delay


class UtilityTests(unittest.TestCase):
    def test_measurements_preserve_failed_and_missing_sample_positions(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            prefix = Path(temp) / 'sample'
            for index, values in enumerate([
                'TSA = 2e-10\nPAVG = FAILED\n',
                'TSA = -1e-10\nPAVG = 1e-6\n',
                'TSA = 4e-10\nPAVG = 2e-6\n',
            ]):
                Path(f'{prefix}.mt{index}').write_text(values)
            data = parse_mc_measurements(str(prefix), num_runs=4)
            self.assertEqual(list(data.index), [0, 1, 2, 3])
            self.assertTrue(np.isnan(data.loc[0, 'PAVG']))
            self.assertTrue(np.isnan(data.loc[1, 'TSA']))
            self.assertAlmostEqual(data.loc[1, 'PAVG'], 1e-6)
            self.assertTrue(data.loc[3].isna().all())
            self.assertAlmostEqual(generate_mc_statistics(data).loc['TSA', 'mean'], 3e-10, delta=1e-22)

    def test_measurements_keep_zero_and_reject_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            prefix = Path(temp) / 'sample'
            Path(f'{prefix}.mt0').write_text('VWL_PRE_0 = 0\nPAVG = inf\nTSA = -1e-10\n')
            data = parse_mc_measurements(str(prefix), num_runs=1)
            self.assertEqual(data.loc[0, 'VWL_PRE_0'], 0)
            self.assertTrue(np.isnan(data.loc[0, 'PAVG']))
            self.assertTrue(np.isnan(data.loc[0, 'TSA']))

    def test_prn_samples_plot_through_the_legacy_package_api(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), plt.rc_context():
            for indexed in (False, True):
                for axis, values, analysis in (
                    ('TIME', [0, 1e-9, 2e-9] * 2, 'tran'),
                    ('{U}', [-.1, 0, .1] * 2, 'dc'),
                ):
                    with self.subTest(indexed=indexed, analysis=analysis):
                        path = Path(temp) / 'waveform.prn'
                        header = ('Index ' if indexed else '') + axis + ' V(Q) V(QB)\n'
                        rows = ''.join((f'{i} ' if indexed else '') + f'{value} 0.9 0.1\n'
                                       for i, value in enumerate(values))
                        path.write_text(header + rows + 'End of Xyce(TM) Simulation\n')
                        data, kind = read_prn_with_preprocess(path)
                        self.assertEqual(kind, analysis)
                        self.assertEqual([len(block) for block in split_blocks(data, kind, 2)], [3, 3])
                        with self.assertRaises(ValueError):
                            split_blocks(data, kind, 4)
                        output = Path(temp) / 'plots' / 'waveform.png'
                        self.assertTrue(process_simulation_data(path, 2, output, ['V(Q)']))
                        self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_comparison_plots_save_and_close_without_changing_global_style(self):
        values = [1e-9, 2e-9]
        deviations = [1e-11, 2e-11]
        with tempfile.TemporaryDirectory() as temp, patch('matplotlib.pyplot.show') as show:
            original_style = dict(plt.rcParams)
            original_figures = plt.get_fignums()
            for function in (plot_delay, plot_power, plot_rc_delay, plot_leak_delay):
                with self.subTest(function=function.__name__):
                    output = function(['8', '16'], values, deviations, values, deviations,
                                      'Read', 'Write', function.__name__,
                                      output_dir=Path(temp) / 'plots')
                    signature = b'\x89PNG' if function is plot_rc_delay else b'%PDF'
                    self.assertTrue(output.read_bytes().startswith(signature))
                    self.assertEqual(plt.get_fignums(), original_figures)
                    self.assertEqual(dict(plt.rcParams), original_style)
            show.assert_not_called()

    def test_optimizer_plot_imports_remain_usable(self):
        from size_optimization.exp_utils import plot_merit_history, plot_pareto_frontier

        with tempfile.TemporaryDirectory() as temp:
            history, pareto = Path(temp) / 'history.png', Path(temp) / 'pareto.png'
            plot_merit_history([1, 2], 'Example', history)
            plot_pareto_frontier([{'min_snm': .2, 'max_power': 1e-6, 'area': 1e-12}],
                                'Example', pareto)
            for output in (history, pareto):
                self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
