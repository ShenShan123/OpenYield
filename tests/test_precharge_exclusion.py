"""Access starts only after the physical far precharge terminal has settled."""
import contextlib
from dataclasses import FrozenInstanceError, replace
import io
import math
import tempfile
import unittest

from sram_compiler.config_yaml.sweep_config import SWEEP_CONFIGS
from sram_compiler.interconnect import resolve_interconnect
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.sizing.table import physical_context
from sram_compiler.subcircuits.time_generate import TIME
from sram_compiler.testbenches.parameter_factor import TIMEFactory
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench


def subcircuit(circuit, name):
    return next(sub for sub in circuit.subcircuits if sub.name == name)


class PrechargeExclusionTests(unittest.TestCase):
    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_common_guard_qualifies_wl_write_and_sense_without_delaying_raw_release(self):
        time = TIMEFactory(num_rows=8, num_cols=4, operation='read&write',
                           replica_precharge_guard=True, precharge_guard_stages=4,
                           precharge_off_guard=True, precharge_off_guard_stages=4).create()
        self.assertEqual(time.NODES[-2:], ['rwl', 'pre_far'])
        self.assertEqual(time['Xaccess_guard'].node_names,
                         ['VDD', 'VSS', 'gated_clk_bar', 'pre_far', 'access_clk_bar'])
        self.assertEqual(time['Xwl_en'].node_names[-2], 'access_clk_bar')
        self.assertEqual(time['Xw_en'].node_names[2], 'access_clk_bar')
        self.assertEqual(time['Xs_en'].node_names[3], 'access_clk_bar')
        guard = subcircuit(time, 'PRECHARGE_OFF_GUARD')
        # The raw request directly inhibits the final gate. The delay never
        # extends a write into the next clock-high phase.
        self.assertEqual(guard['Xrequest_gate'].node_names[2:4],
                         ['request', 'pre_off_ready'])
        self.assertEqual(guard['Xobserve_pre'].node_names[2], 'pre_far')
        self.assertEqual(subcircuit(guard, 'PRECHARGE_OFF_DELAY').stages, 4)
        self.assertEqual(time['Xrwl_precharge_guard'].node_names[2], 'rwl')
        self.assertEqual(time['Xprecharge_guard_delay'].node_names[2], 'rwl_pre_bar')
        self.assertFalse(any(e.name[0].upper() == 'B' for e in guard.elements))

    def test_compiler_observes_replica_precharge_terminal_in_numeric_and_sweep_decks(self):
        for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
            for mux in (False, True):
                for rc in (False, True):
                    for sweep in (False, True):
                        with self.subTest(cell=cell, mux=mux, rc=rc, sweep=sweep), tempfile.TemporaryDirectory() as temp:
                            cfg = load_config(4, 4, 'TT')
                            tb = Sram6TCoreMcTestbench(cfg, sram_cell_type=cell,
                                choose_columnmux=mux, w_rc=rc, variation_mode='nominal',
                                sweep_cell=sweep, real_cell_mode=0, sim_path=temp)
                            if sweep:
                                spec = SWEEP_CONFIGS['cell' if cell == 'SRAM_6T_CELL' else 'cell_10T']
                                tb.gen_model_sweep_generic(module_name=spec['name'], param_model_names=spec['model_params'])
                            circuit = tb.create_testbench('read&write', 3, 3)
                            expected = 'XPRECHARGE_RBL:ENB_end' if rc else 'PRE_line_far'
                            self.assertEqual(circuit['XTIME'].node_names[-1], expected)
                            time = subcircuit(circuit, 'TIME')
                            self.assertEqual(time.NODES[-1], 'pre_far')
                            self.assertTrue(tb.driver_sizes.precharge_off_guard)
                            self.assertEqual(time.precharge_off_guard_stages,
                                             tb.driver_sizes.precharge_off_guard_stages)
                            self.assertEqual(time.access_load, tb.driver_sizes.loads.access_load)
                            self.assertEqual(time.precharge_off_tau, tb.driver_sizes.precharge_off_tau)

    def test_new_observer_and_common_gate_loads_are_frozen_from_actual_input_widths(self):
        for rows, cols in ((8, 4), (8, 512), (512, 4)):
            for rc in (False, True):
                with self.subTest(rows=rows, cols=cols, rc=rc):
                    cfg = load_config(rows, cols, 'TT')
                    sizes = resolve_driver_sizes(cfg, physical_context=physical_context(rc))
                    pre = (cols + 1) * 3 * cfg.precharge.pmos_width.value * sizes.pre / .36e-6
                    if rc:
                        pre += (cols + 1) * 2
                    self.assertAlmostEqual(sizes.loads.pre_load, pre + .5)
                    # WL buffer's first inverter plus write NAND2 and sense
                    # NAND3 request inputs, all normalized by 0.36 um.
                    wl_input = max(1, math.ceil(sizes.loads.wl_load / 24.))
                    self.assertEqual(sizes.loads.access_load, wl_input + 2.5)
                    self.assertEqual(sizes.precharge_off_guard_stages, 4)
                    wire = resolve_interconnect().wl
                    expected_tau = cols * wire.resistance_per_pitch * (
                        sizes.loads.pre_load * .5e-15 + cols * wire.capacitance_per_pitch)
                    if rc:
                        expected_tau += 100 * 1e-15
                    self.assertAlmostEqual(sizes.precharge_off_tau, expected_tau, places=21)
                    with self.assertRaises(FrozenInstanceError):
                        sizes.precharge_off_guard_stages = 2
                    with self.assertRaises(FrozenInstanceError):
                        sizes.loads.access_load = 0

    def test_optional_time_ports_preserve_direct_factory_callers(self):
        time = TIMEFactory(num_rows=4, num_cols=4).create()
        self.assertEqual(time.NODES[-1], 'sa_iso')
        self.assertEqual(time['Xwl_en'].node_names[-2], 'gated_clk_bar')
        for bad in (0, -2, 3, True, 4.0):
            with self.subTest(stages=bad), self.assertRaisesRegex(ValueError, 'positive even'):
                TIME(precharge_off_guard=True, precharge_off_guard_stages=bad)

    def test_physical_settling_rc_precedes_delay_and_resets_without_waiting_for_it(self):
        time = TIMEFactory(precharge_off_guard=True, precharge_off_tau=.6e-9).create()
        guard = subcircuit(time, 'PRECHARGE_OFF_GUARD')
        self.assertEqual(guard['Rsettle'].node_names, ['pre_on', 'pre_on_filtered'])
        self.assertEqual(guard['Csettle'].node_names, ['pre_on_filtered', 'VSS'])
        self.assertAlmostEqual(float(guard['Rsettle'].resistance) * float(guard['Csettle'].capacitance), .6e-9, places=21)
        self.assertEqual(guard['Xsettle_pre'].node_names[2], 'pre_on_filtered')
        self.assertEqual(guard['Xready'].node_names[2:4], ['pre_on', 'pre_on_delayed'])
        for bad in (-1e-9, float('nan'), float('inf')):
            with self.subTest(tau=bad), self.assertRaisesRegex(ValueError, 'settling'):
                TIMEFactory(precharge_off_guard=True, precharge_off_tau=bad).create()

    def test_settling_time_follows_baseline_wire_geometry_without_changing_classes(self):
        cfg = load_config(8, 512, 'TT')
        wire = resolve_interconnect()
        base = resolve_driver_sizes(cfg, physical_context=physical_context(True, interconnect=wire))
        doubled = replace(wire, wl=replace(wire.wl, sheet_resistance_ohm=2 * wire.wl.sheet_resistance_ohm))
        changed = resolve_driver_sizes(cfg, physical_context=physical_context(True, interconnect=doubled))
        local_tau = 100 * 1e-15
        self.assertAlmostEqual(changed.precharge_off_tau - local_tau,
                               2 * (base.precharge_off_tau - local_tau), places=21)
        self.assertEqual(base.loads, changed.loads)
        self.assertEqual((base.pre, base.wd_in, base.wd_out, base.wl_inv, base.wl_nand, base.dec_inv),
                         (changed.pre, changed.wd_in, changed.wd_out, changed.wl_inv, changed.wl_nand, changed.dec_inv))
        self.assertNotEqual(base.key, changed.key)

    def test_stale_or_disabled_guard_snapshots_are_rejected_but_candidates_reuse_baseline(self):
        cfg = load_config(8, 4, 'TT')
        context = physical_context(True)
        sizes = resolve_driver_sizes(cfg, physical_context=context)
        for stale in (
                replace(sizes, precharge_off_guard=False),
                replace(sizes, precharge_off_guard_stages=sizes.precharge_off_guard_stages + 2),
                replace(sizes, precharge_off_tau=0),
                replace(sizes, precharge_off_tau=sizes.precharge_off_tau * 2),
                replace(sizes, loads=replace(sizes.loads, access_load=None)),
                replace(sizes, loads=replace(sizes.loads, access_load=sizes.loads.access_load + 1)),
                replace(sizes, loads=replace(sizes.loads, pre_load=sizes.loads.pre_load - .5))):
            with self.subTest(stale=stale.loads), self.assertRaisesRegex(ValueError, 'guard|observer'):
                stale.validate_for(cfg, 'SRAM_6T_CELL', False, context)
        cfg.sram_6t_cell.pmos_width.value *= 1.5
        cfg.global_config.vdd = .9
        cfg.global_config.temperature = 125
        sizes.validate_for(cfg, 'SRAM_6T_CELL', False, context)


if __name__ == '__main__':
    unittest.main()
