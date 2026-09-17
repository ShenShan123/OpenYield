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
from sram_compiler.sizing.driver_sizing import _RULES
from sram_compiler.sizing.table import physical_context
from sram_compiler.subcircuits.standard_cell import AND2
from sram_compiler.subcircuits.time_generate import TIME_CONTROL
from sram_compiler.testbenches.parameter_factor import TimeControlFactory
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench


def subcircuit(circuit, name):
    return next(sub for sub in circuit.subcircuits if sub.name == name)


class PrechargeExclusionTests(unittest.TestCase):
    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_common_guard_qualifies_wl_write_and_sense_without_delaying_raw_release(self):
        time = TimeControlFactory(num_rows=8, num_cols=4, operation='read&write',
                           replica_precharge_guard=True, precharge_guard_stages=4,
                           precharge_off_guard=True, precharge_off_guard_stages=4).create()
        self.assertEqual(time.NODES[-2:], ['rwl', 'pre_far'])
        self.assertEqual(time['Xaccess_guard'].node_names,
                         ['VDD', 'VSS', 'gated_clk_bar', 'pre_far', 'access_clk_bar', 'pre_off_ready'])
        self.assertEqual(time['Xwl_en'].node_names[2:4], ['access_clk_bar', 's_en_bar'])
        # V2.1.6 write slot: w_en = we_hold & (wl_en | (cs_pre & write_slot)) with
        # write_slot = pre_ready & pre_off_ready & !s_en (V2.1.8) turns the write
        # drivers on in the clock-high phase, once the previous wordline, the
        # physical precharge and the previous sense enable are observed off;
        # PRE is inhibited by the held write request and by either enable.
        self.assertEqual(time['Xw_en'].node_names[2:4], ['we_hold', 'write_window'])
        self.assertEqual(time['Xwrite_slot'].node_names[2:6], ['pre_ready', 'pre_off_ready', 's_en_bar', 'write_slot'])
        self.assertEqual(time['Xselected_slot'].node_names[2:5], ['cs_pre', 'write_slot', 'selected_slot'])
        self.assertEqual(time['Xwrite_window_nor'].node_names[2:5], ['wl_en', 'selected_slot', 'write_window_bar'])
        self.assertEqual(time['Xpre_gate'].node_names[2:6], ['pre_ready', 'we_hold_bar', 'enables_off', 'pre_gate'])
        self.assertEqual(time['Xpre_unbuf'].node_names[2:5], ['clk_buf', 'cs_pre', 'pre_gate'])
        # V2.1.7: only the rising edge of the select is delayed (cs & delayed cs).
        self.assertEqual(time['Xselect_delay'].node_names[2:4], ['cs', 'cs_delayed'])
        self.assertEqual(time['Xselect_gate'].node_names[2:5], ['cs', 'cs_delayed', 'cs_pre'])
        self.assertEqual(subcircuit(time, 'SELECT_DELAY').stages, 8)
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

    def test_access_gates_see_the_write_request_only_through_its_wordline_hold_latch(self):
        """The `we` register updates on the edge that ends an access, before the request falls
        (20 vs 38 ps at FF -40 C), so a raw `we` input pulsed w_en to 0.72 V while a read wordline
        was still on. Every gate must see the request held while wl_en is high, buffered or not."""
        for rows, cols, guard, replica in ((16, 4, False, False), (8, 128, True, False), (8, 4, True, True)):
            with self.subTest(rows=rows, cols=cols, replica=replica):
                time = TimeControlFactory(num_rows=rows, num_cols=cols, operation='read&write',
                                          precharge_off_guard=guard, replica_precharge_guard=replica,
                                          precharge_guard_stages=4 if replica else 0).create()
                request = 'access_clk_bar' if guard else 'gated_clk_bar'
                self.assertEqual(time['Xwe_hold'].node_names,
                                 ['VDD', 'VSS', 'we', 'wl_en_bar', 'we_hold', 'we_hold_bar'])
                # The write slot needs the replica guard (see the no-guard test below).
                window = 'write_window' if replica else 'wl_en'
                self.assertEqual(time['Xw_en'].node_names[2:4], ['we_hold', window])
                self.assertEqual(time['Xs_en'].node_names[2:5], ['rbl_delay', request, 'we_hold_bar'])
                raw = {e.name for e in time.elements if {'we', 'we_bar'} & set(e.node_names)}
                self.assertEqual(raw, {'Xdff_buf1', 'Xwe_hold'})
                self.assertIn('w_en_unbuf' if cols == 128 else 'w_en', time['Xw_en'].node_names)

    def test_select_delay_holds_back_only_the_rising_edge_of_both_clock_high_enables(self):
        """The select is registered on the same edge as the write request and the write data.
        Delayed on its rising edge it keeps the precharge off until the held request can inhibit
        it (V2.1.6 start-up race) and w_en off until new data has passed the write-data hold latch,
        which closes when w_en rises: with the raw select the latch closed only ~25 ps after new
        data settled at an idle -> write edge at FF -40 C (V2.1.7).  The falling edge must follow
        the select at once, or an unselected cycle races the delayed select against the
        wordline-off guard (58 ps at 2x4 FF -40 C with a symmetric delay)."""
        time = TimeControlFactory(num_rows=8, num_cols=4, operation='read&write',
                                  replica_precharge_guard=True, precharge_guard_stages=4,
                                  precharge_off_guard=True).create()
        self.assertEqual(time['Xselect_delay'].node_names[2:4], ['cs', 'cs_delayed'])
        self.assertEqual(subcircuit(time, 'SELECT_DELAY').stages, 8)
        self.assertEqual(time['Xselect_gate'].node_names[2:5], ['cs', 'cs_delayed', 'cs_pre'])
        self.assertIsInstance(subcircuit(time, time['Xselect_gate'].subcircuit_name), AND2)
        # V2.1.8: the select reaches the write enable through the selected slot only.
        readers = {e.name for e in time.elements if 'cs_pre' in e.node_names[2:-1]}
        self.assertEqual(readers, {'Xselected_slot', 'Xpre_unbuf'})
        # Only the access-phase gated clocks and the select delay read the raw select
        # (Xdff_buf is its register).
        raw = {e.name for e in time.elements if 'cs' in e.node_names[2:-1]} - {'Xdff_buf'}
        self.assertEqual(raw, {'Xand2_gated_clk_bar', 'Xand2_gated_clk_buf', 'Xselect_delay', 'Xselect_gate'})

    def test_write_enable_follows_the_wordline_enable_without_the_replica_guard(self):
        """Without the replica-wordline observer the wordline-off signal is wl_en_bar, and a window
        wl_en | wl_en_bar is constantly high: V2.1.6 held w_en on through consecutive writes, so
        the write-data hold latch never reopened for new data, and while the previous wordline was
        still falling.  The precharge-off guard alone does not observe the wordline."""
        for guard in (False, True):
            with self.subTest(precharge_off_guard=guard):
                time = TimeControlFactory(num_rows=8, num_cols=4, operation='write',
                                          precharge_off_guard=guard).create()
                self.assertEqual(time['Xw_en'].node_names[2:4], ['we_hold', 'wl_en'])
                self.assertNotIn('write_window', {n for e in time.elements for n in e.node_names})
                self.assertEqual(time['Xpre_gate'].node_names[2:6], ['wl_en_bar', 'we_hold_bar', 'enables_off', 'pre_gate'])
        # With the observer (settling stages optional) the slot waits for the replica
        # wordline and the previous sense enable; without the precharge-off guard its
        # precharge input is tied high (V2.1.8).
        for stages, off_guard, off, ready in ((0, False, 'rwl_pre_bar', 'VDD'), (4, False, 'pre_ready', 'VDD'),
                                              (0, True, 'rwl_pre_bar', 'pre_off_ready')):
            with self.subTest(stages=stages, precharge_off_guard=off_guard):
                time = TimeControlFactory(num_rows=8, num_cols=4, operation='write', replica_precharge_guard=True,
                                          precharge_guard_stages=stages, precharge_off_guard=off_guard).create()
                self.assertEqual(time['Xwrite_slot'].node_names[2:6], [off, ready, 's_en_bar', 'write_slot'])
                self.assertEqual(time['Xwrite_window_nor'].node_names[2:5], ['wl_en', 'selected_slot', 'write_window_bar'])
                self.assertEqual(time['Xw_en'].node_names[2:4], ['we_hold', 'write_window'])

    def test_read_wordline_ends_at_the_sense_enable_and_the_clock_high_enables_wait_for_the_enables(self):
        """V2.1.8: a read wordline that stays on after the sense enable only discharges the
        bitlines further (the amplifier is isolated from them once it fires) and lengthens the
        restore; the wordline request therefore ends with the sense enable.  The precharge waits
        for both enables and the write slot for the sense enable of the preceding read, so the
        clock-high enables never overlap the previous access; the write enable is not part of the
        slot (the slot feeds it: the loop would oscillate) and ends with the wordline enable, not
        with the deselect (which dropped the drivers at half wordline)."""
        for rows, cols, buffered in ((8, 4, False), (64, 64, True)):
            with self.subTest(rows=rows, cols=cols):
                time = TimeControlFactory(num_rows=rows, num_cols=cols, operation='read&write',
                                          replica_precharge_guard=True, precharge_guard_stages=4,
                                          precharge_off_guard=True, wen_load=400 if buffered else None,
                                          sen_load=200 if buffered else None).create()
                self.assertEqual(time['Xwl_en'].node_names[2:5], ['access_clk_bar', 's_en_bar', 'wl_en'])
                buffer = subcircuit(time, 'WORDLINE_ENABLE_BUFFER')
                self.assertEqual(buffer['Xbuf_nand1'].node_names[2:5], ['A', 'B', 'zb1_node'])
                # The observers read the block's output nodes, never the unbuffered gate outputs.
                self.assertEqual(time['Xinv_s_en_bar'].node_names[2:4], ['s_en', 's_en_bar'])
                self.assertEqual(time['Xenables_off_nor'].node_names[2:5], ['s_en', 'w_en', 'enables_off'])
                self.assertEqual(time['Xs_en'].node_names[-1], 's_en_unbuf' if buffered else 's_en')
                self.assertEqual(time['Xpre_gate'].node_names[2:5], ['pre_ready', 'we_hold_bar', 'enables_off'])
                self.assertEqual(time['Xwrite_slot'].node_names[2:5], ['pre_ready', 'pre_off_ready', 's_en_bar'])
                self.assertNotIn('w_en', time['Xwrite_slot'].node_names)
                self.assertNotIn('cs_pre', time['Xw_en'].node_names)
                readers = {e.name for e in time.elements if 'w_en' in e.node_names[2:-1]}
                self.assertEqual(readers, {'Xenables_off_nor', 'Xsa_iso_nor'} if not buffered else {'Xenables_off_nor'})

    def test_passed_transistor_models_reach_every_control_device(self):
        """The observers, delay chains and every buffer follow the models passed to the block
        (V2.1.6 hard-coded or defaulted them for five block types)."""
        time = TimeControlFactory(nmos_model='NMOS_TEST', pmos_model='PMOS_TEST', num_rows=64, num_cols=64,
                                  operation='read&write', replica_precharge_guard=True, precharge_guard_stages=4,
                                  precharge_off_guard=True, precharge_off_tau=.6e-9, effort_buffers=True,
                                  wen_load=400, sen_load=200, iso_load=200).create()
        text = str(time)
        self.assertNotIn('_VTG', text)
        for name in ('ADDRESS_BUFFER', 'WRITE_ENABLE_BUFFER', 'SENSE_ENABLE_BUFFER', 'SENSE_ISOLATION_BUFFER',
                     'PRECHARGE_BUFFER', 'WORDLINE_ENABLE_BUFFER', 'REPLICA_DELAY_CHAIN', 'PRECHARGE_GUARD_DELAY',
                     'PINV_rwl_precharge'):
            with self.subTest(block=name):
                self.assertIn(f'.subckt {name} ', text)

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
                            self.assertEqual(circuit['XTIME_CONTROL'].node_names[-1], expected)
                            time = subcircuit(circuit, 'TIME_CONTROL')
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
                    # WL request NAND2 input (1.25 units per unit of scale, V2.1.8)
                    # plus the sense NAND3 request input, all normalized by 0.36 um.
                    wl_input = max(1, math.ceil(sizes.loads.wl_load / 24.))
                    self.assertEqual(sizes.loads.access_load, 1.25 * wl_input + 2.5)
                    # s_en_bar and the enables-off NOR load the enables (V2.1.8).
                    senb = max(1, math.ceil(1.25 * (wl_input + 1) / 4.))
                    rc_units = 2 * physical_context(rc)['pi_cap'] / _RULES['unit_inverter_cap_f'] if rc else 0
                    self.assertAlmostEqual(sizes.loads.sen_load,
                                           sizes.loads.num_sa * (.27 / .36 + rc_units) + 3.5 + senb + 1.75)
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
        time = TimeControlFactory(num_rows=4, num_cols=4).create()
        self.assertEqual(time.NODES[-1], 'sa_iso')
        self.assertEqual(time['Xwl_en'].node_names[2], 'gated_clk_bar')
        for bad in (0, -2, 3, True, 4.0):
            with self.subTest(stages=bad), self.assertRaisesRegex(ValueError, 'positive even'):
                TIME_CONTROL(precharge_off_guard=True, precharge_off_guard_stages=bad)

    def test_physical_settling_rc_precedes_delay_and_resets_without_waiting_for_it(self):
        time = TimeControlFactory(precharge_off_guard=True, precharge_off_tau=.6e-9).create()
        guard = subcircuit(time, 'PRECHARGE_OFF_GUARD')
        self.assertEqual(guard['Rsettle'].node_names, ['pre_on', 'pre_on_filtered'])
        self.assertEqual(guard['Csettle'].node_names, ['pre_on_filtered', 'VSS'])
        self.assertAlmostEqual(float(guard['Rsettle'].resistance) * float(guard['Csettle'].capacitance), .6e-9, places=21)
        self.assertEqual(guard['Xsettle_pre'].node_names[2], 'pre_on_filtered')
        self.assertEqual(guard['Xready'].node_names[2:4], ['pre_on', 'pre_on_delayed'])
        for bad in (-1e-9, float('nan'), float('inf')):
            with self.subTest(tau=bad), self.assertRaisesRegex(ValueError, 'settling'):
                TimeControlFactory(precharge_off_guard=True, precharge_off_tau=bad).create()

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
