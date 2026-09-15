"""Clock classes are a frozen design specification, never a candidate fit."""

import contextlib
from dataclasses import FrozenInstanceError
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PySpice.Unit import u_ns

from sram_compiler.per_device_mc.run import load_config
from sram_compiler.config_yaml.sweep_config import SWEEP_CONFIGS
from sram_compiler.sizing import load_timing_lookup, resolve_driver_sizes, resolve_timing
from sram_compiler.sizing.table import physical_context
from sram_compiler.sizing.timing import TimingConfig
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench


class TimingLookupTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_boundaries_round_up_and_extrapolate_without_claiming_evidence(self):
        cases = [(1, 1, 4), (32, 4, 4), (33, 4, 4.5), (64, 16, 4.5),
                 (65, 16, 5), (129, 4, 6), (257, 4, 9),
                 (16, 17, 4.5), (48, 20, 4.5), (16, 512, 8),
                 (513, 4, 13.5), (4, 513, 9.15)]
        for rows, cols, ns in cases:
            with self.subTest(rows=rows, cols=cols):
                cfg = load_config(rows, cols, 'TT')
                timing = resolve_timing(cfg, resolve_driver_sizes(cfg))
                self.assertAlmostEqual(timing.t_period / 1e-9, ns)
                self.assertEqual(timing.extrapolated, max(rows, cols) > 512)
                self.assertFalse(timing.qualified)
                self.assertEqual(timing.source, 'lookup')
                self.assertEqual(len(timing.table_sha256), 64)

    def test_period_is_shared_across_pvt_and_physical_modes_except_the_10t_budget(self):
        """Only 10T cells have their own evidenced budget (V2.1.3); PVT and RC never change a class."""
        for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
            for mux in (False, True):
                for rc in (False, True):
                    cfg = load_config(48, 20, 'SF')
                    cfg.global_config.vdd = .9
                    cfg.global_config.temperature = 125
                    sizes = resolve_driver_sizes(cfg, cell_type=cell, mux=mux,
                                                 physical_context=physical_context(rc))
                    timing = resolve_timing(cfg, sizes)
                    separate = cell == 'SRAM_10T_CELL'
                    self.assertAlmostEqual(timing.t_period / 1e-9, 5.5 if separate else 4.5)
                    self.assertEqual(timing.budget, 'SRAM_10T_CELL' if separate else 'shared')

    def test_10t_budget_grows_with_height_above_the_shared_ladder(self):
        """The 10T read port discharges the replica bitline about 1 ps per row slower than 6T and its
        sense path adds about 200 ps: the shared 4 ns class failed 16x16 10T with and without a mux,
        a flat 200 ps failed the 128x8 mux read at 5.5 ns and 32x16 failed a mismatch seed at 4.5 ns,
        so the row budget grows with height and keeps 250 ps at every class bound; at 512 rows the
        10T read-disturb bump must also decay within the storage tolerance, which needs 14 ns."""
        cases = [(8, 4, 5), (16, 16, 5), (32, 16, 5), (33, 16, 5.5), (16, 32, 5),
                 (64, 16, 5.5), (128, 8, 6), (256, 4, 8), (512, 4, 14),
                 (8, 64, 5.5), (8, 128, 6.5), (8, 512, 8.5), (513, 4, 24.5)]
        for rows, cols, ns in cases:
            for mux in (True, False):
                with self.subTest(rows=rows, cols=cols, mux=mux):
                    cfg = load_config(rows, cols, 'SS')
                    sizes = resolve_driver_sizes(cfg, cell_type='SRAM_10T_CELL', mux=mux)
                    timing = resolve_timing(cfg, sizes)
                    self.assertAlmostEqual(timing.t_period / 1e-9, ns)
                    self.assertEqual(timing.budget, 'SRAM_10T_CELL')
                    self.assertEqual(timing.extrapolated, rows > 512)
                    self.assertEqual(timing.table_version, 'v2.1.3-timing-2')
                    shared = resolve_timing(cfg, resolve_driver_sizes(cfg, cell_type='SRAM_6T_CELL', mux=mux))
                    self.assertGreater(timing.t_period, shared.t_period)
                    self.assertEqual(shared.budget, 'shared')

    def test_injected_baseline_survives_candidates_and_rejects_changed_contract(self):
        cfg = load_config(8, 4, 'TT')
        tb = Sram6TCoreMcTestbench(cfg, choose_columnmux=False, variation_mode='nominal')
        timing, sizes = tb.timing_config, tb.driver_sizes
        cfg.sram_6t_cell.pmos_width.value *= 1.5
        cfg.global_config.vdd = .9
        candidate = Sram6TCoreMcTestbench(cfg, choose_columnmux=False, driver_sizes=sizes,
                                        timing_config=timing, corner='SS', temperature=125,
                                        variation_mode='nominal')
        self.assertIs(candidate.timing_config, timing)
        self.assertEqual(candidate.t_period, tb.t_period)
        with self.assertRaises(FrozenInstanceError):
            timing.t_period = 9e-9
        changed_baseline = resolve_driver_sizes(cfg)
        with self.assertRaisesRegex(ValueError, 'driver baseline'):
            timing.validate_for(cfg, changed_baseline)
        cfg.global_config.timing = {'mode': 'lookup', 'margin': .5}
        with self.assertRaisesRegex(ValueError, 'timing options'):
            timing.validate_for(cfg, sizes)

    def test_custom_table_paths_margin_and_manual_override(self):
        cfg = load_config(8, 4, 'TT')
        sizes = resolve_driver_sizes(cfg)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lookup.json'
            table = load_timing_lookup()
            table['row_classes'][0]['half_period_ps'] = 1700
            path.write_text(json.dumps(table))
            cfg.global_config.timing = {'mode': 'lookup', 'lookup': str(path), 'margin': .5}
            custom = resolve_timing(cfg, sizes)
            self.assertAlmostEqual(custom.t_period / 1e-9, 5.1)
            previous = os.getcwd()
            try:
                os.chdir(temp)
                self.assertEqual(load_timing_lookup('sram_compiler/sizing/timing_lookup.json')['version'], 'V2.1.3')
            finally:
                os.chdir(previous)
        cfg.global_config.timing = {'mode': 'fixed', 't_period': 10e-9}
        tb = Sram6TCoreMcTestbench(cfg, choose_columnmux=False, variation_mode='nominal')
        self.assertAlmostEqual(float(tb.t_period), 10e-9)
        self.assertEqual(tb.timing_config.source, 'fixed')
        TimingConfig(4e-9, 0, 0, 0, source='diagnostic').apply(tb)
        self.assertEqual(tb.timing_config.source, 'diagnostic')
        tb.set_timing_parameters(.05 @ u_ns, .05 @ u_ns, 2.5 @ u_ns, 5 @ u_ns, 1 @ u_ns)
        self.assertEqual(tb.timing_config.source, 'manual')
        self.assertAlmostEqual(tb.timing_config.t_period, float(tb.t_period))

    def test_invalid_options_and_malformed_tables_fail_before_generation(self):
        cfg = load_config(8, 4, 'TT')
        sizes = resolve_driver_sizes(cfg)
        for opts in (None, {'mode': 'auto'}, {'typo': 1}, {'t_period': 1e-9},
                     {'margin': True}, {'margin': float('nan')}, {'margin': -.1},
                     {'mode': 'fixed'}, {'mode': 'fixed', 't_period': 0},
                     {'mode': 'fixed', 't_period': True},
                     {'mode': 'fixed', 't_period': 1e-9, 'margin': .25}):
            cfg.global_config.timing = opts
            with self.subTest(options=opts), self.assertRaises(ValueError):
                resolve_timing(cfg, sizes)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'table.json'
            for entry in ({'max_rows': 0, 'half_period_ps': 1},
                          {'max_rows': True, 'half_period_ps': 1},
                          {'max_rows': 32, 'half_period_ps': -1},
                          {'max_rows': 32, 'half_period_ps': 1.5}, None):
                table = load_timing_lookup()
                table['row_classes'][0] = entry
                path.write_text(json.dumps(table))
                with self.assertRaises(ValueError):
                    load_timing_lookup(path)
            # A variant is a separately evidenced budget: one per architecture,
            # on the shared anchors, and never below the shared budget.
            base = load_timing_lookup()
            variant = base['variants'][0]
            relaxed = dict(variant, row_classes=[dict(variant['row_classes'][0], half_period_ps=1500)]
                           + variant['row_classes'][1:])
            shifted = dict(variant, row_classes=[dict(variant['row_classes'][0], max_rows=16)]
                           + variant['row_classes'][1:])
            short = dict(variant, column_classes=variant['column_classes'][:-1])
            for variants in ([dict(variant, cell_type='SRAM_8T_CELL')], [dict(variant, mux=1)],
                             [variant, variant], [variant, dict(variant, mux=True)],
                             [relaxed], [shifted], [short], {'a': variant}, [None]):
                path.write_text(json.dumps(dict(base, variants=variants)))
                with self.subTest(variants=variants), self.assertRaises(ValueError):
                    load_timing_lookup(path)
            # A mux-restricted variant leaves the other mux setting on the shared class.
            cfg.global_config.timing = {'mode': 'lookup', 'lookup': str(path)}
            path.write_text(json.dumps(dict(base, variants=[dict(variant, mux=True)])))
            for mux, ns, budget in ((True, 5., 'SRAM_10T_CELL/mux'), (False, 4., 'shared')):
                timing = resolve_timing(cfg, resolve_driver_sizes(cfg, cell_type='SRAM_10T_CELL', mux=mux))
                self.assertAlmostEqual(timing.t_period / 1e-9, ns)
                self.assertEqual(timing.budget, budget)
            path.write_text(json.dumps(dict(base, variants=[])))
            sizes = resolve_driver_sizes(cfg, cell_type='SRAM_10T_CELL', mux=True)
            self.assertAlmostEqual(resolve_timing(cfg, sizes).t_period / 1e-9, 4.)

    def test_numeric_and_swept_decks_share_clock_and_check_actual_rc_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            for sweep, cell_pin_rc in ((False, False), (False, True), (True, False), (True, True)):
                tb = Sram6TCoreMcTestbench(load_config(4, 4, 'TT'), choose_columnmux=False,
                    variation_mode='nominal', w_rc=True, sweep_cell=sweep, sim_path=temp,
                    interconnect={'cell_pin_rc': cell_pin_rc})
                if sweep:
                    spec = SWEEP_CONFIGS['cell']
                    tb.gen_model_sweep_generic(module_name=spec['name'], param_model_names=spec['model_params'])
                circuit = tb.create_testbench('write', 3, 3)
                simulator = circuit.simulator(simulator='xyce-serial')
                tb.add_meas_and_print(simulator, tb.data_init(), 'write')
                tb.add_analysis(simulator.circuit, 'write', 1)
                deck = str(simulator).upper()
                self.assertAlmostEqual(float(tb.t_period) / 1e-9, 4.)
                self.assertEqual(tb.driver_sizes.precharge_guard_stages, 4)
                local_wl = tb.cell_probe('WL').upper()
                self.assertIn(f'V({local_wl})', deck)
                self.assertTrue(local_wl.endswith(':WL_END' if cell_pin_rc else ':WL3_TAP3'))
                self.assertIn('VACCESS_ERROR_0', deck)
                self.assertIn('VHOLD_ERROR_0', deck)
                line = next(l for l in deck.splitlines() if ' VACCESS_ERROR_0 ' in l)
                self.assertIn('AT=5.8E-09', line)

    def test_retention_covers_deadline_through_next_access_including_last_cycle(self):
        import re

        with tempfile.TemporaryDirectory() as temp:
            tb = Sram6TCoreMcTestbench(load_config(2, 2, 'TT'), choose_columnmux=False,
                variation_mode='nominal', w_rc=True, sim_path=temp)
            circuit = tb.create_testbench('read&write', 1, 1)
            simulator = circuit.simulator(simulator='xyce-serial')
            tb.add_meas_and_print(simulator, tb.data_init(), 'read&write')
            deck = str(simulator).upper()
            for cycle in range(8):
                access = next(l for l in deck.splitlines() if f' VACCESS_ERROR_{cycle} ' in l)
                hold = next(l for l in deck.splitlines() if f' VHOLD_ERROR_{cycle} ' in l)
                deadline = float(re.search(r'AT=(\S+)', access)[1])
                start = float(re.search(r'FROM=(\S+)', hold)[1])
                stop = float(re.search(r'TO=(\S+)', hold)[1])
                self.assertLessEqual(start, deadline)
                self.assertAlmostEqual(stop - deadline, .5 * float(tb.t_period), delta=1e-18)
                self.assertLessEqual(stop, float(tb._analysis_stop('read&write')) + 1e-18)

    def test_late_wrong_missing_or_unretained_data_cannot_be_a_passing_sample(self):
        data = pd.DataFrame({'VACCESS_ERROR_0': [0., .4, np.nan, 0.],
                             'VHOLD_ERROR_0': [0., 0., 0., .3],
                             'VPRE_ACCESS_ERROR_0': [0., 0., 0., 0.],
                             'VRESTORE_ERROR_0': [0., 0., 0., 0.]})
        np.testing.assert_array_equal(Sram6TCoreMcTestbench.access_validity(data, 'read', 1.),
                                      [True, False, False, False])
        self.assertFalse(Sram6TCoreMcTestbench.access_validity(data, 'read&write', 1.).any())
        data.loc[0, 'VRESTORE_ERROR_0'] = .03
        self.assertFalse(Sram6TCoreMcTestbench.access_validity(data, 'read', 1.).any())

    def test_precharge_during_access_rejects_even_correct_retained_data(self):
        data = pd.DataFrame({'VACCESS_ERROR_0': [0., 0.], 'VHOLD_ERROR_0': [0., 0.],
                             'VRESTORE_ERROR_0': [0., 0.], 'VPRE_ACCESS_ERROR_0': [0., .4]})
        np.testing.assert_array_equal(Sram6TCoreMcTestbench.access_validity(data, 'write', 1.),
                                      [True, False])
        self.assertFalse(Sram6TCoreMcTestbench.access_validity(
            data.drop(columns='VPRE_ACCESS_ERROR_0'), 'read', 1.).any())

    def test_cli_period_metadata_and_repeated_runs_preserve_original_evidence(self):
        from sram_compiler.per_device_mc import run
        with tempfile.TemporaryDirectory() as temp:
            argv = ['run', '--rows', '2', '--cols', '2', '--variation-mode', 'nominal',
                    '--output-dir', temp, '--period', '4e-9', '--audit', '--no-waveform']
            with patch('sys.argv', argv):
                args = run.parse_args()
            first, summary = run.generate_deck(args)
            evidence = Path(str(first) + '.mt0')
            evidence.write_text('FAILED original')
            second, repeated = run.generate_deck(args)
            self.assertNotEqual(first.parent, second.parent)
            self.assertEqual(evidence.read_text(), 'FAILED original')
            self.assertEqual(summary['compiler_version'], 'V2.1.3')
            self.assertAlmostEqual(summary['timing']['t_period'], 4e-9)
            self.assertEqual(summary['timing']['source'], 'fixed')
            args.run_xyce = True
            with patch.object(run, 'parse_args', return_value=args), \
                    patch.object(run, 'generate_deck', return_value=(second, repeated)), \
                    patch.object(run, 'run_xyce', side_effect=RuntimeError('solver failed')):
                with self.assertRaisesRegex(RuntimeError, 'solver failed'):
                    run.main()
            saved = json.loads((second.parent / 'summary.json').read_text())
            self.assertIn('solver failed', saved['simulation_error'])
            self.assertIn('timing', saved)
            # A failed run without its solver installation cannot be matched to an MPI stack.
            self.assertIn('xyce', saved)

    def test_cli_interrupted_run_keeps_its_summary_and_solver_identity(self):
        """An interrupted solver run is an incomplete outcome that must stay traceable."""
        from sram_compiler.per_device_mc import run
        with tempfile.TemporaryDirectory() as temp:
            argv = ['run', '--rows', '2', '--cols', '2', '--variation-mode', 'nominal',
                    '--output-dir', temp, '--no-waveform']
            with patch('sys.argv', argv):
                args = run.parse_args()
            deck, summary = run.generate_deck(args)
            args.run_xyce = True
            with patch.object(run, 'parse_args', return_value=args), \
                    patch.object(run, 'generate_deck', return_value=(deck, summary)), \
                    patch.object(run, 'find_xyce', return_value='/opt/xyce/bin/Xyce'), \
                    patch.object(run, 'run_xyce', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run.main()
            saved = json.loads((deck.parent / 'summary.json').read_text())
            self.assertTrue(saved['simulation_error'].startswith('KeyboardInterrupt'))
            self.assertEqual(saved['xyce'], '/opt/xyce/bin/Xyce')
            self.assertNotIn('xyce_exit', saved)

    def test_cli_pvt_overrides_reach_deck_metadata_and_run_identity(self):
        from sram_compiler.per_device_mc import run
        with tempfile.TemporaryDirectory() as temp:
            base = ['run', '--rows', '2', '--cols', '2', '--variation-mode', 'nominal',
                    '--output-dir', temp, '--no-waveform']
            with patch('sys.argv', base):
                default_args = run.parse_args()
            with patch('sys.argv', base + ['--vdd', '0.8', '--temperature', '-40']):
                pvt_args = run.parse_args()
            default_deck, default = run.generate_deck(default_args)
            deck, summary = run.generate_deck(pvt_args)
            self.assertEqual((default['vdd'], default['temperature']), (1.0, 25))
            self.assertEqual((summary['vdd'], summary['temperature']), (0.8, -40.0))
            text = deck.read_text()
            self.assertRegex(text, r'(?m)^VVDD VDD VSS 0\.8V$')
            self.assertRegex(text, r'(?m)^\.options TEMP = -40\.0$')
            # A PVT point must not be filed as a repeated attempt of the default point.
            self.assertNotIn('_attempt', deck.parent.name)
            self.assertNotEqual(default_deck.parent.name.rsplit('_', 1)[1],
                                deck.parent.name.rsplit('_', 1)[1])
            for invalid in (['--vdd', '0'], ['--vdd', 'inf'], ['--temperature', 'nan']):
                with patch('sys.argv', base + invalid), patch('sys.stderr'), \
                        self.assertRaises(SystemExit):
                    run.parse_args()

    def test_cli_rejects_missing_metrics_even_when_voltage_checks_pass(self):
        from sram_compiler.per_device_mc import run

        with tempfile.TemporaryDirectory() as temp:
            argv = ['run', '--rows', '2', '--cols', '2', '--variation-mode', 'nominal',
                    '--output-dir', temp, '--run-xyce', '--no-waveform']
            with patch('sys.argv', argv):
                args = run.parse_args()
            deck, summary = run.generate_deck(args)
            Path(str(deck) + '.mt0').write_text(
                'VWL_PRE_FAR_0 = 0\nVWL_PRE_LOCAL_0 = 0\nVWL_PRE_PEAK_0 = 0\n'
                'VACCESS_ERROR_0 = 0\nVHOLD_ERROR_0 = 0\nVPRE_ACCESS_ERROR_0 = 0\nVRESTORE_ERROR_0 = 0\n'
                'TREAD_TOTAL = FAILED\nPAVG = 1e-6\nPSTC = 1e-7\nPDYN = 9e-7\n')
            with patch.object(run, 'parse_args', return_value=args), \
                    patch.object(run, 'generate_deck', return_value=(deck, summary)), \
                    patch.object(run, 'run_xyce'), self.assertRaisesRegex(RuntimeError, 'metrics'):
                run.main()
            saved = json.loads((deck.parent / 'summary.json').read_text())
            self.assertTrue(saved['access_checked'])
            self.assertFalse(saved['metrics_checked'])

    def test_cli_plot_failure_keeps_passing_measures_and_records_the_error(self):
        """A lost plot is not lost evidence: the .prn and passing measures stand."""
        from sram_compiler.per_device_mc import run

        with tempfile.TemporaryDirectory() as temp:
            argv = ['run', '--rows', '2', '--cols', '2', '--variation-mode', 'nominal',
                    '--output-dir', temp, '--run-xyce']
            with patch('sys.argv', argv):
                args = run.parse_args()
            deck, summary = run.generate_deck(args)
            Path(str(deck) + '.mt0').write_text(
                'VWL_PRE_FAR_0 = 0\nVWL_PRE_LOCAL_0 = 0\nVWL_PRE_PEAK_0 = 0\n'
                'VACCESS_ERROR_0 = 0\nVHOLD_ERROR_0 = 0\nVPRE_ACCESS_ERROR_0 = 0\nVRESTORE_ERROR_0 = 0\n'
                'TREAD_TOTAL = 1.5e-10\nPAVG = 1e-6\nPSTC = 1e-7\nPDYN = 9e-7\n')
            with patch.object(run, 'parse_args', return_value=args), \
                    patch.object(run, 'generate_deck', return_value=(deck, summary)), \
                    patch.object(run, 'run_xyce'), \
                    patch.object(run, 'plot_waveform', side_effect=ValueError('Conflicting duplicate')):
                self.assertEqual(run.main(), 0)
            saved = json.loads((deck.parent / 'summary.json').read_text())
            self.assertTrue(saved['access_checked'])
            self.assertTrue(saved['metrics_checked'])
            self.assertIsNone(saved['waveform_png'])
            self.assertEqual(saved['waveform_error'], 'ValueError: Conflicting duplicate')
            self.assertNotIn('simulation_error', saved)


if __name__ == '__main__':
    unittest.main()
