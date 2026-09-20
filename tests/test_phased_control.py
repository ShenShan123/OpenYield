"""Clock-high accesses finish before the following rising-edge capture."""
import contextlib
import io
import unittest
import hashlib
import tempfile
from pathlib import Path

import pandas as pd

from dataclasses import replace

from sram_compiler.subcircuits.time_generate import TIME_CONTROL
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.sizing.timing import timing_from_measurements
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from tests.spice.phased_access import prepare


class PhasedControlTests(unittest.TestCase):
    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def test_direct_control_gate_stubs_preserve_numeric_and_expression_rc_values(self):
        from PySpice.Unit import u_Ohm, u_pF
        for resistance, capacitance in ((240 @ u_Ohm, .012 @ u_pF),
                                        ('{R_LOCAL}', '{C_LOCAL}')):
            for enabled in (False, True):
                with self.subTest(resistance=str(resistance), enabled=enabled):
                    control = TIME_CONTROL(operation='read&write', w_rc=enabled,
                                           pi_res=resistance, pi_cap=capacitance)
                    gates = [gate for gate in control.subcircuits if gate.name in
                             ('AND2', 'WORDLINE_START_AND', 'WORDLINE_SETUP_AND', 'WRITE_PREPARE_AND')]
                    self.assertEqual(len(gates), 4)
                    for gate in gates:
                        resistors = [element for element in gate.elements if hasattr(element, 'resistance')]
                        capacitors = [element for element in gate.elements if hasattr(element, 'capacitance')]
                        self.assertEqual((len(resistors), len(capacitors)), (8, 8) if enabled else (0, 0))
                        for element in resistors:
                            if isinstance(resistance, str):
                                self.assertEqual(str(element.resistance), resistance)
                            else:
                                self.assertEqual(float(element.resistance), float(resistance))
                        for element in capacitors:
                            if isinstance(capacitance, str):
                                self.assertEqual(str(element.capacitance), capacitance)
                            else:
                                self.assertAlmostEqual(float(element.capacitance), float(capacitance), delta=1e-25)

    def test_tall_setup_tracks_wire_load_without_adding_mos(self):
        from sram_compiler.interconnect import resolve_interconnect
        base = resolve_interconnect(None)
        for rows in (64, 128, 256, 512):
            for factor in (1, 2):
                wires = replace(base, bl=replace(base.bl,
                    capacitance_f_per_m=base.bl.capacitance_f_per_m * factor))
                control = TIME_CONTROL(num_rows=rows, interconnect=wires)
                delay = next(s for s in control.subcircuits if s.name == 'WORDLINE_SETUP_DELAY')
                caps = [e for e in delay.elements if hasattr(e, 'capacitance')]
                self.assertEqual(len(caps), 2 if rows > 64 else 0)
                self.assertEqual(len([e for e in delay.elements if e.name.startswith('X')]), 40)
                for cap in caps:
                    self.assertAlmostEqual(float(cap.capacitance),
                        2 * rows * wires.bl.capacitance_per_pitch, delta=1e-25)
        normal = TIME_CONTROL(num_rows=512, interconnect=base, enable_off_tau=0)
        resistive = TIME_CONTROL(num_rows=512, enable_off_tau=0,
            interconnect=replace(base, bl=replace(base.bl,
                sheet_resistance_ohm=2 * base.bl.sheet_resistance_ohm)))
        self.assertAlmostEqual(float(resistive['Rwordline_setup'].resistance),
                               2 * float(normal['Rwordline_setup'].resistance))

    def test_tall_numeric_and_swept_cells_keep_the_same_decoder_guard(self):
        from sram_compiler.config_yaml.sweep_config import SWEEP_CONFIGS
        loads = []
        with tempfile.TemporaryDirectory() as temp:
            for sweep in (False, True):
                tb = Sram6TCoreMcTestbench(load_config(128, 4, 'TT'),
                    choose_columnmux=False, variation_mode='nominal',
                    w_rc=True, sweep_cell=sweep, sim_path=temp)
                if sweep:
                    spec = SWEEP_CONFIGS['cell']
                    tb.gen_model_sweep_generic(module_name=spec['name'],
                                               param_model_names=spec['model_params'])
                circuit = tb.create_testbench('read&write', 127, 3)
                control = next(s for s in circuit.subcircuits if s.name == 'TIME_CONTROL')
                delay = next(s for s in control.subcircuits if s.name == 'WORDLINE_SETUP_DELAY')
                values = [float(e.capacitance) for e in delay.elements if hasattr(e, 'capacitance')]
                self.assertEqual(len(values), 2)
                self.assertAlmostEqual(float(tb.t_period) / 1e-9, 16.5)
                loads.append(values)
        self.assertEqual(loads[0], loads[1])

    def test_capture_registers_have_no_secondary_hold_latches(self):
        control = TIME_CONTROL(num_rows=16, num_cols=8, operation='read&write',
                               replica_precharge_guard=True, precharge_guard_stages=4,
                               precharge_off_guard=True)
        text = str(control)
        for obsolete in ('HOLD_LATCH', 'DIN_HOLD', 'BUSY_HOLD_DELAY', 'SELECT_DELAY', 'SELECTED_SLOT'):
            self.assertNotIn(obsolete, text)
        self.assertEqual(control['Xaccess_guard'].node_names[2], 'gated_clk_buf')
        self.assertEqual(control['Xpre_unbuf'].node_names[2:5],
                         ['clk_bar', 'pre_ready', 'enables_off'])
        self.assertEqual(control['Xwrite_window_nor'].node_names[2:4],
                         ['access_request', 'wordline_busy'])
        self.assertEqual(control['Xw_en'].node_names[2:5], ['we', 'write_window', 'iso_ready'])
        self.assertEqual(control['Xfar_enables_off'].node_names[2:4], ['sen_far', 'wen_far'])
        self.assertEqual(control['Xenable_off_guard'].node_names[2:6],
                         ['local_enables_off', 'far_enables_off', 'enables_off', 'far_enables_ready'])

    def test_wordline_setup_delays_assertion_but_not_clock_release(self):
        control = TIME_CONTROL(replica_precharge_guard=True, precharge_off_guard=True)
        self.assertEqual(control['Xwordline_setup_gate'].node_names[2:4],
                         ['access_request', 'access_settled'])
        self.assertEqual(control['Xwl_en'].node_names[2:4], ['wordline_request', 'read_done_bar'])
        self.assertEqual(control['Xs_en'].node_names[2:5], ['read_done', 'wordline_idle', 'iso_ready'])
        self.assertEqual(control['Xwordline_start'].node_names[2:4], ['access_request', 'write_ready'])
        self.assertEqual(control['Xwrite_ready_nor'].node_names[2:4], ['we_bar', 'wen_far'])
        self.assertEqual(control['Xisolation_guard'].node_names[2:4], ['VDD', 'iso_far'])

    def test_runtime_rejects_capture_races_and_incompatible_enables_despite_correct_data(self):
        phases = ('ACCESS', 'HOLD', 'PRE_ACCESS', 'WEN_ACCESS', 'RESTORE', 'BOUNDARY', 'ROLE', 'ISO')
        good = pd.DataFrame({f'V{phase}_ERROR_0': [0.] for phase in phases})
        self.assertTrue(Sram6TCoreMcTestbench.access_validity(good, 'write', 1.)[0])
        for phase in ('BOUNDARY', 'ROLE', 'ISO'):
            name = f'V{phase}_ERROR_0'
            for value in (.11, float('nan')):
                bad = good.copy()
                bad.loc[0, name] = value
                with self.subTest(phase=phase, value=value):
                    self.assertFalse(Sram6TCoreMcTestbench.access_validity(bad, 'write', 1.)[0])
            self.assertFalse(Sram6TCoreMcTestbench.access_validity(good.drop(columns=name), 'write', 1.)[0])

    def test_old_control_baselines_cannot_be_injected_into_the_new_architecture(self):
        config = load_config(8, 4, 'TT')
        sizes = resolve_driver_sizes(config)
        for architecture in ('', 'v2.1.10'):
            with self.subTest(architecture=architecture), self.assertRaisesRegex(ValueError, 'control architecture'):
                replace(sizes, control_architecture=architecture).validate_for(config, 'SRAM_6T_CELL', False)
        with self.assertRaisesRegex(ValueError, 'guard'):
            replace(sizes, enable_off_tau=0).validate_for(config, 'SRAM_6T_CELL', False)

    def test_prepared_wordline_request_scales_for_the_actual_frozen_load(self):
        control = TIME_CONTROL(num_rows=512, wl_load=2160, access_load=115.,
                               effort_buffers=True, precharge_off_guard=True,
                               enable_off_tau=.7e-9, isolation_tau=.9e-9)
        self.assertEqual(control.sizing.access_drive_scale, 20)
        gate = next(sub for sub in control.subcircuits if sub.name == 'WORDLINE_SETUP_AND')
        inverter = next(sub for sub in gate.subcircuits if sub.name.startswith('PINV'))
        self.assertAlmostEqual(float(inverter['Mpinv_nmos'].width), 20 * .09e-6, delta=1e-18)
        self.assertAlmostEqual(float(inverter['Mpinv_pmos'].width), 20 * .27e-6, delta=1e-18)
        self.assertEqual(control['Xwordline_setup_delay'].node_names[2], 'wordline_start_filtered')
        self.assertAlmostEqual(float(control['Rwordline_setup'].resistance)
                               * float(control['Cwordline_setup'].capacitance), .7e-9, delta=1e-21)
        isolation = next(sub for sub in control.subcircuits if sub.name == 'ISOLATION_OFF_GUARD')
        self.assertAlmostEqual(isolation.settling_tau, .9e-9, delta=1e-21)

    def test_calibration_includes_recovery_after_writes(self):
        read = dict(TCLK_WLEN=.2e-9, TREAD_TOTAL=1e-9, TRESTORE=.5e-9)
        write = dict(TCLK_WLEN=.3e-9, TWRITE_TOTAL=.4e-9, TRESTORE=2e-9, TWSLOT=.1e-9)
        timing = timing_from_measurements(read, [write, write])
        self.assertAlmostEqual(timing.t_period, 5e-9, delta=1e-18)
        self.assertAlmostEqual(timing.high, 2e-9, delta=1e-18)

    def test_integration_harness_uses_the_requested_corner_in_the_actual_model_cards(self):
        for corner in ('TT', 'SS', 'SF', 'FF', 'FS'):
            for variation in ('nominal', 'per-device'):
                with self.subTest(corner=corner, variation=variation), tempfile.TemporaryDirectory() as temp:
                    config = load_config(2, 2, corner)
                    model = Path(getattr(config.global_config, 'pdk_path_' + corner))
                    path, metadata = prepare(dict(name='corner', rows=2, cols=2, corner=corner,
                                                  variation=variation, operation='read',
                                                  vdd=.9, temperature=125), Path(temp))
                    self.assertEqual(metadata['corner'], corner)
                    self.assertEqual(metadata['temperature'], 125)
                    self.assertEqual(metadata['model_sha256'], hashlib.sha256(model.read_bytes()).hexdigest())
                    if variation == 'nominal':
                        self.assertIn(str(model), path.read_text())
                    else:
                        self.assertEqual(metadata['variation']['base_model_file'], str(model))
                        self.assertEqual(metadata['variation']['base_model_sha256'], metadata['model_sha256'])


if __name__ == '__main__':
    unittest.main()
