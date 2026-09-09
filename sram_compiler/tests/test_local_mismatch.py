"""Local mismatch must cover the active topology without changing its devices."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from per_device_mc.netlist import SpiceParseError, _parse_subckts, specialize_netlist
from per_device_mc.run import load_config
from sram_compiler.sizing.campaign import architectures, verification_cases
from sram_compiler.sizing.timing import TimingConfig
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench


def active_devices(deck):
    root = _parse_subckts(deck.splitlines())
    devices = {}

    def visit(scope, path):
        for line in scope.body:
            parts = line.split()
            if not parts:
                continue
            if parts[0].upper().startswith('M'):
                devices[path + '/' + parts[0]] = parts[1:]
            elif parts[0].upper().startswith('X'):
                target = scope.children.get(parts[-1]) or root.children[parts[-1]]
                visit(target, path + '/' + parts[0])
    visit(root, '__top__')
    return devices


class LocalMismatchTests(unittest.TestCase):
    def test_all_active_devices_keep_connectivity_widths_and_independent_models(self):
        for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
            for operation in ('read', 'write'):
                for mux in (False, True):
                    with self.subTest(cell=cell, operation=operation, mux=mux), \
                            tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                        cfg = load_config(2, 2, 'SS')
                        cfg.global_config.sram_cell_type = cell
                        cfg.global_config.sizing = {'mode': 'rules_only'}
                        kwargs = dict(sram_cell_type=cell, choose_columnmux=mux,
                                      corner='SS', real_cell_mode=0, sim_path=temp)
                        nominal = Sram6TCoreMcTestbench(cfg, mc=False, **kwargs)
                        expected = active_devices(str(nominal.create_testbench(operation, 1, 1)))
                        local = Sram6TCoreMcTestbench(cfg, mc_seed=41, **kwargs)
                        circuit = local.create_testbench(operation, 1, 1)
                        actual = active_devices(str(circuit))
                        self.assertEqual(actual.keys(), expected.keys())
                        for path, tokens in actual.items():
                            original = expected[path]
                            self.assertEqual(tokens[:4] + tokens[5:], original[:4] + original[5:])
                            self.assertTrue(tokens[4].startswith('MC_' + original[4] + '_'))
                        self.assertEqual(len({tokens[4] for tokens in actual.values()}), len(actual))
                        with Path(local.variation_summary['audit_file']).open() as stream:
                            audit = list(csv.DictReader(stream))
                        self.assertEqual({row['hier_path'] + '/' + row['mos_name'] for row in audit}, set(actual))
                        self.assertEqual(local.variation_summary['unique_mc_models'], len(actual))
                        array_devices = [path for path in actual if '/X' + cell + '_' in path]
                        self.assertEqual(len(array_devices), (6 if cell == 'SRAM_6T_CELL' else 10) * 4)
                        local.add_analysis(circuit, operation, 1)
                        self.assertIn('.SAMPLING useExpr=true', str(circuit))
                        self.assertIn('numsamples=1 seed=41', str(circuit))

    def test_swept_width_expression_survives_specialization(self):
        cfg = load_config(1, 1, 'TT')
        deck = f'''Width sweep
.include {cfg.global_config.pdk_path_TT}
.subckt pair D G S B
M1 D G S B NMOS_VTG w={{width*2}} l=50n nf=4
.ends pair
Xleft D G 0 0 pair
Xright D G 0 0 pair
.param width=100n
'''
        with tempfile.TemporaryDirectory() as temp:
            result, summary = specialize_netlist(
                deck, base_model_path=Path(cfg.global_config.pdk_path_TT),
                model_output_path=Path(temp) / 'models.spice', mc_runs=2, vth_std=.05)
            self.assertEqual(summary['unique_mc_models'], 2)
            active = active_devices(result)
            self.assertEqual(len(active), 2)
            self.assertTrue(all(tokens[-3:] == ['w={width*2}', 'l=50n', 'nf=4'] for tokens in active.values()))
            with self.assertRaisesRegex(SpiceParseError, 'unresolved'):
                specialize_netlist(deck + 'Xmissing D G 0 0 absent\n',
                    base_model_path=Path(cfg.global_config.pdk_path_TT),
                    model_output_path=Path(temp) / 'bad.spice', mc_runs=1, vth_std=.05)

    def test_nominal_and_shared_are_explicit_comparisons(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            cfg = load_config(2, 2, 'TT')
            for mode, sampled in [('nominal', False), ('shared', True)]:
                tb = Sram6TCoreMcTestbench(cfg, variation_mode=mode, sim_path=temp)
                circuit = tb.create_testbench('write', 1, 1)
                tb.add_analysis(circuit, 'write', 2 if sampled else 1)
                self.assertNotIn('MC_NMOS_', str(circuit))
                self.assertEqual('.SAMPLING' in str(circuit), sampled)

    def test_every_verification_path_uses_full_local_mismatch(self):
        timing = TimingConfig(5e-9, 1e-9, 1e-9, 1e-9)
        for base in architectures([(8, 4), (16, 16), (256, 8), (32, 1)]):
            cases = verification_cases(base, timing)
            self.assertTrue(all(case.variation == 'per-device' and case.real_cell_mode == 0
                                and case.samples == 10 for case in cases))
            self.assertTrue({'read', 'write'} <= {case.operation for case in cases})
            for operation in ('read', 'write'):
                self.assertEqual({case.corner for case in cases if case.operation == operation},
                                 {'TT', 'SS', 'SF', 'FS', 'FF'})
            if base.rows >= 256:
                self.assertTrue(all(case.next_row for case in cases
                                    if case.corner == 'FF' and case.temperature == -40
                                    and case.cell_variant == 'baseline' and case.operation != 'read&write'))
            if base.rows in (16, 256) and not base.mux and not base.w_rc:
                for corner in ('SS', 'SF'):
                    tails = [case for case in cases if case.operation == 'write'
                             and case.corner == corner and case.seed >= 4026]
                    self.assertEqual(sum(case.samples for case in tails), 100)
                    self.assertEqual(len({case.seed for case in tails}), 10)

    def test_local_geometry_sweep_cannot_silently_run_at_mean_or_skip_steps(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            tb = Sram6TCoreMcTestbench(load_config(2, 2, 'TT'), sim_path=temp)
            tb.sweep_cell = True
            with self.assertRaisesRegex(ValueError, 'separate deck per geometry'):
                tb.add_analysis(None, 'read', 2)

    def test_later_exports_preserve_the_earlier_model_artifact(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            tb = Sram6TCoreMcTestbench(load_config(2, 2, 'TT'), sim_path=temp)
            tb.create_testbench('read', 1, 1)
            original = Path(tb.variation_summary['mc_model_file'])
            content = original.read_bytes()
            tb.create_testbench('write', 1, 1)
            self.assertNotEqual(original, Path(tb.variation_summary['mc_model_file']))
            self.assertEqual(original.read_bytes(), content)


if __name__ == '__main__':
    unittest.main()
