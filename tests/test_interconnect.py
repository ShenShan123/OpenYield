"""Distributed wires conserve geometry and keep cells at physical taps."""

import contextlib
import io
import unittest
from unittest.mock import patch

from PySpice.Spice.Netlist import Circuit

from sram_compiler.interconnect import WireRC, resolve_interconnect, add_tapped_line
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.sizing.table import physical_context
from sram_compiler.subcircuits import sram_cell_add_equivalent as equivalent
from sram_compiler.subcircuits.sram_6t_core import Sram6TCore
from sram_compiler.subcircuits.sram_10t_core import Sram10TCore
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench


def wire_config(refinement=1):
    # Illustrative geometry: 1 ohm / 0.1 fF per pitch, not foundry extraction.
    wire = dict(layer='M2', pitch_m=.6e-6, width_m=.1e-6,
                sheet_resistance_ohm=1/6, capacitance_f_per_m=1e-16/.6e-6,
                sections_per_half_pitch=refinement)
    return dict(mode='distributed', wl=wire, bl=dict(wire, layer='M3'))


def build_core(cell_type=Sram6TCore, mode=0, sweep=False, interconnect=None):
    dims = dict(pd_width='pd', pu_width='pu', pg_width='pg', length='length') if sweep else {}
    if cell_type is Sram10TCore:
        dims['fd_nmos_model'] = 'NMOS_VTG'
        if sweep:
            dims['fd_width'] = 'fd'
    return cell_type(4, 4, 'NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG', w_rc=True,
                     real_cell_mode=mode, target_row=3, target_col=3,
                     interconnect=interconnect or wire_config(), **dims)


class WireTests(unittest.TestCase):
    def test_cli_run_identity_includes_wire_configuration(self):
        from sram_compiler.per_device_mc.run import make_run_name
        from types import SimpleNamespace
        args = SimpleNamespace(rows=4, cols=4, operation='read', real_cell_mode=0,
                               variation_mode='nominal', vth_std=.05, corner='TT',
                               pi_res_ohm=100, pi_cap_pf=.001, q_init_val=0, waveform=True, seed=5)
        options = dict(cell_type='SRAM_6T_CELL', target_row=3, target_col=3, mc_runs=1)
        first = make_run_name(args, **options, interconnect=wire_config())
        second = make_run_name(args, **options, interconnect=wire_config(2))
        self.assertNotEqual(first, second)

    def test_pi_sections_conserve_total_resistance_capacitance_and_tap_count(self):
        for count in (1, 4, 32):
            for refinement in (1, 4):
                with self.subTest(count=count, refinement=refinement):
                    wire = resolve_interconnect(wire_config(refinement)).wl
                    circuit = Circuit('wire')
                    taps = add_tapped_line(circuit, 'WL', 'input', count, wire)
                    self.assertEqual(taps, tuple(f'WL_tap{i}' for i in range(count)))
                    resistors = [e for e in circuit.elements if e.name.startswith('Rwire_')]
                    capacitors = [e for e in circuit.elements if e.name.startswith('Cwire_')]
                    self.assertEqual(len(resistors), 2 * count * refinement)
                    self.assertAlmostEqual(sum(float(e.resistance) for e in resistors), count)
                    self.assertAlmostEqual(sum(float(e.capacitance) for e in capacitors) / 1e-16, count)
                    self.assertIn('WL_far', str(circuit))
                    self.assertTrue(all(str(e).split()[2] == 'VSS' for e in capacitors))

    def test_invalid_geometry_and_unknown_options_fail_before_generation(self):
        for field in ('pitch_m', 'width_m', 'sheet_resistance_ohm', 'capacitance_f_per_m'):
            for value in (0, -1, float('nan'), float('inf'), True):
                options = wire_config()
                options['wl'][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    resolve_interconnect(options)
        for options in ({'mode': 'typo'}, {'mode': 'distributed'},
                        dict(wire_config(), unknown=1),
                        dict(wire_config(), cell_pin_rc='false')):
            with self.assertRaises(ValueError):
                resolve_interconnect(options)
        for value in (0, 1.5, True):
            options = wire_config()
            options['bl']['sections_per_half_pitch'] = value
            with self.assertRaises(ValueError):
                resolve_interconnect(options)

    def test_star_default_and_physical_identity(self):
        self.assertFalse(resolve_interconnect(None).distributed)
        self.assertTrue(resolve_interconnect(None).cell_pin_rc)
        distributed = resolve_interconnect(wire_config())
        self.assertFalse(distributed.cell_pin_rc)
        self.assertEqual(distributed, resolve_interconnect(distributed.to_dict()))
        self.assertNotEqual(physical_context(True), physical_context(True, interconnect=distributed))
        changed = wire_config()
        changed['wl']['width_m'] *= 2
        self.assertNotEqual(physical_context(True, interconnect=distributed),
                            physical_context(True, interconnect=changed))


class ArrayWireTests(unittest.TestCase):
    def test_cells_have_unique_taps_in_numeric_and_sweep_paths(self):
        for cell_type in (Sram6TCore, Sram10TCore):
            for sweep in (False, True):
                with self.subTest(cell=cell_type.__name__, sweep=sweep), contextlib.redirect_stdout(io.StringIO()):
                    core = build_core(cell_type, sweep=sweep)
                    cell = list(core.subcircuits)[0]
                    self.assertNotIn('RR_WL_', str(cell))
                    self.assertNotIn('RR_BL_', str(cell))
                    self.assertIn('RR_Q_', str(cell))
                    instances = [str(e).split() for e in core.elements if e.name.startswith('XSRAM')]
                    self.assertEqual(len(instances), 16)
                    for row in range(4):
                        for col in range(4):
                            instance = next(parts for parts in instances if parts[0].endswith(f'_{row}_{col}'))
                            self.assertEqual(instance[3:6], [f'BL{col}_tap{row}', f'BLB{col}_tap{row}', f'WL{row}_tap{col}'])

    def test_optional_pin_stubs_remain_separate_from_distributed_wires(self):
        with contextlib.redirect_stdout(io.StringIO()):
            core = build_core(interconnect=dict(wire_config(), cell_pin_rc=True))
        self.assertIn('RR_WL_', str(list(core.subcircuits)[0]))
        self.assertIn('Rwire_WL', str(core))

    def test_equivalent_write_power_uses_local_wordline_voltage(self):
        caps = {'caps': dict(c_wl=1e-16, c_bl=1e-16, c_blb=1e-16,
                             c_wl_bl=1e-17, c_wl_blb=1e-17)}
        fit = dict(wl_ratios=[0., 1.], avg_currents=[1e-9, 1e-6])
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
                equivalent, '_cached_extraction', side_effect=lambda tester, name, fn: caps if name == 'caps' else fit):
            core = Sram6TCore(4, 4, 'NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG',
                             w_rc=True, real_cell_mode=4, target_row=3, target_col=3,
                             interconnect=wire_config(), write_power_model=True)
        sources = [line for line in core.raw_spice.splitlines() if line.startswith('BIWL_POWER_')]
        self.assertEqual(len(sources), 15)
        self.assertTrue(all('_tap' in line for line in sources))

    def test_equivalent_sweeps_fail_with_actionable_error(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'numeric.*real_cell_mode=0'):
            build_core(mode=4, sweep=True)

    def test_equivalent_modes_preserve_wire_geometry_and_local_coupling(self):
        caps = {'caps': dict(c_wl=1e-16, c_bl=2e-16, c_blb=3e-16, c_wl_bl=1e-17, c_wl_blb=2e-17)}
        for cell_type in (Sram6TCore, Sram10TCore):
            with contextlib.redirect_stdout(io.StringIO()):
                reference = build_core(cell_type)
            wires = [str(e) for e in reference.elements if e.name.startswith(('Rwire_', 'Cwire_'))]
            for mode in (1, 2, 3, 4):
                with self.subTest(cell=cell_type.__name__, mode=mode), contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(equivalent, '_cached_extraction', side_effect=lambda tester, name, fn: caps if name == 'caps' else 1e12):
                    core = build_core(cell_type, mode=mode)
                self.assertEqual(wires, [str(e) for e in core.elements if e.name.startswith(('Rwire_', 'Cwire_'))])
                coupling = [str(e).split() for e in core.elements if e.name.startswith('Ceq_WLBL_')]
                self.assertEqual(len(coupling), core._count_total_unused_cells())
                self.assertTrue(all('_tap' in parts[1] and '_tap' in parts[2] for parts in coupling))

    def test_replica_lengths_taps_and_far_wordline_precharge_guard(self):
        for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
            for mux in (False, True):
                with self.subTest(cell=cell, mux=mux), contextlib.redirect_stdout(io.StringIO()):
                    cfg = load_config(4, 4, 'TT')
                    tb = Sram6TCoreTestbench(cfg, sram_cell_type=cell, choose_columnmux=mux,
                                             w_rc=True, interconnect=wire_config())
                    circuit = tb.create_testbench('read', 3, 3)
                self.assertTrue(tb.driver_sizes.replica_matched)
                self.assertTrue(tb.driver_sizes.replica_precharge_guard)
                replica = next(s for s in circuit.subcircuits if 'replica_column' in s.name)
                self.assertEqual(sum(e.name.startswith('XReplica_CELL') for e in replica.elements), 4)
                rbl = [e for e in replica.elements if e.name.startswith('Rwire_RBL_')]
                self.assertAlmostEqual(sum(float(e.resistance) for e in rbl), 4.)
                rwl = [e for e in circuit.elements if e.name.startswith('Rwire_RWL_')]
                self.assertAlmostEqual(sum(float(e.resistance) for e in rwl), 4.)
                self.assertIn('RWL_far TIME', str(circuit['XTIME']))
                self.assertIn('RWL_tap3', str(circuit[f'X{replica.name}']))
                self.assertTrue(tb.cell_probe('WL').endswith(':WL3_tap3'))

    def test_frozen_baseline_rejects_wire_changes_but_accepts_pvt(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load_config(4, 4, 'TT')
            cfg.global_config.interconnect = wire_config()
            tb = Sram6TCoreTestbench(cfg, w_rc=True, choose_columnmux=False)
            changed = wire_config()
            changed['bl']['pitch_m'] *= 2
            with self.assertRaisesRegex(ValueError, 'physical context'):
                Sram6TCoreTestbench(cfg, w_rc=True, choose_columnmux=False,
                                  interconnect=changed, driver_sizes=tb.driver_sizes)
            other = Sram6TCoreTestbench(cfg, w_rc=True, choose_columnmux=False,
                                      corner='SS', temperature=125, driver_sizes=tb.driver_sizes)
        self.assertEqual(other.driver_sizes, tb.driver_sizes)

    def test_per_device_specialization_retains_wire_nodes_and_local_measurements(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            tb = Sram6TCoreMcTestbench(load_config(4, 4, 'TT'), w_rc=True,
                                      interconnect=wire_config(), sim_path=temp, mc_seed=5)
            circuit = tb.create_testbench('write', 3, 3)
            self.assertIn('MC_NMOS_', str(circuit))
            self.assertIn('Rwire_WL', str(circuit))
            simulator = circuit.simulator(simulator='xyce-serial', temperature=25)
            tb.add_meas_and_print(simulator, tb.data_init(), 'write')
            deck = str(simulator)
            self.assertTrue(f'V({tb.cell_probe("WL")})' in deck)
            self.assertTrue(f'V({tb.cell_probe("BL")})' in deck)


if __name__ == '__main__':
    unittest.main()
