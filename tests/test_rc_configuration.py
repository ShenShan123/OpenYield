"""The physical RC settings must reach every circuit and cached extraction."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PySpice.Unit import u_Ohm, u_pF

from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.sizing.table import physical_context
from sram_compiler.subcircuits import sram_cell_add_equivalent as equivalent
from sram_compiler.subcircuits.sram_6t_core import Sram6TCore
from sram_compiler.subcircuits.sram_10t_core import Sram10TCore
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench


class RcConfigurationTests(unittest.TestCase):
    def test_nondefault_values_reach_real_replica_and_peripheral_stubs(self):
        for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
            for mode in ('fixed', 'rules_only'):
                with self.subTest(cell=cell, mode=mode), contextlib.redirect_stdout(io.StringIO()):
                    cfg = load_config(4, 4, 'TT')
                    cfg.global_config.sizing = {'mode': mode}
                    tb = Sram6TCoreTestbench(cfg, sram_cell_type=cell, w_rc=True,
                                            pi_res=777 @ u_Ohm, pi_cap=.007 @ u_pF)
                    deck = str(tb.create_testbench('write', 3, 3))
                    resistors = [line for line in deck.splitlines() if line.startswith('RR_')]
                    capacitors = [line for line in deck.splitlines() if line.startswith('CCg_')]
                    self.assertGreater(len(resistors), 20)
                    self.assertEqual(len(resistors), len(capacitors))
                    self.assertTrue(all(line.endswith('777Ohm') for line in resistors), resistors)
                    self.assertTrue(all(line.endswith('0.007pF') for line in capacitors), capacitors)

    def test_core_forwards_values_with_numeric_and_expression_dimensions(self):
        for core_type in (Sram6TCore, Sram10TCore):
            for sweep in (False, True):
                kwargs = dict(pd_width='pd', pu_width='pu', pg_width='pg', length='len') if sweep else {}
                if core_type is Sram10TCore:
                    kwargs['fd_nmos_model'] = 'NMOS_VTG'
                    if sweep:
                        kwargs['fd_width'] = 'fd'
                with self.subTest(core=core_type.__name__, sweep=sweep), contextlib.redirect_stdout(io.StringIO()):
                    core = core_type(2, 2, 'NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG',
                                     w_rc=True, pi_res=777 @ u_Ohm, pi_cap=.007 @ u_pF, **kwargs)
                    cell = list(core.subcircuits)[0]
                    self.assertEqual(float(cell.pi_res), 777)
                    self.assertEqual(float(cell.pi_cap), 7e-15)

    def test_equivalent_extraction_uses_effective_corner_and_voltage(self):
        seen = []
        original = equivalent._build_tester_from_core

        def build(core, cell_type):
            tester = original(core, cell_type)
            seen.append((tester.corner, tester.config.vdd))
            return tester

        caps = {'caps': dict(c_wl=1e-16, c_bl=1e-16, c_blb=1e-16,
                             c_wl_bl=1e-17, c_wl_blb=1e-17)}
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(equivalent, '_build_tester_from_core', side_effect=build), \
                patch.object(equivalent, '_cached_extraction', side_effect=lambda tester, name, fn: caps if name == 'caps' else 1e12):
            cfg = load_config(4, 4, 'TT')
            tb = Sram6TCoreTestbench(cfg, corner='SS', real_cell_mode=4)
            tb.set_vdd(.9)
            tb.create_testbench('read', 3, 3)
        self.assertEqual(seen, [('SS', .9)])
        self.assertEqual((cfg.global_config.corner, cfg.global_config.vdd), ('TT', 1.0))

    def test_cache_invalidates_when_selected_model_contents_change(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            cfg = load_config(2, 2, 'TT').global_config
            cfg.pdk_path_TT = str(Path(temp) / 'models.spice')
            Path(cfg.pdk_path_TT).write_text('* model revision one\n')
            tester = equivalent.SRAMCellParasiticTester(config=cfg)
            first = equivalent._extraction_key(tester)
            Path(cfg.pdk_path_TT).write_text('* model revision two\n')
            self.assertNotEqual(first, equivalent._extraction_key(tester))
            second = equivalent._extraction_key(tester)
            tester.rise_time *= 2
            self.assertNotEqual(second, equivalent._extraction_key(tester))

    def test_sizing_loads_use_configured_capacitance(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load_config(4, 4, 'TT')
        cfg.global_config.sizing = {'mode': 'rules_only'}
        base = resolve_driver_sizes(cfg, physical_context=physical_context(True, 100, 1e-15))
        larger = resolve_driver_sizes(cfg, physical_context=physical_context(True, 100, 7e-15))
        self.assertGreater(larger.loads.pre_load, base.loads.pre_load)
        self.assertGreater(larger.loads.wl_load, base.loads.wl_load)

    def test_frozen_sizes_reject_changed_physical_rc(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load_config(4, 4, 'TT')
        first = physical_context(True, 100, 1e-15)
        sizes = resolve_driver_sizes(cfg, physical_context=first)
        with self.assertRaisesRegex(ValueError, 'physical context'):
            sizes.validate_for(cfg, 'SRAM_6T_CELL', False, physical_context(True, 200, 1e-15))


if __name__ == '__main__':
    unittest.main()
