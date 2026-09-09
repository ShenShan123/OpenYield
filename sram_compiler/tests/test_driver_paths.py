"""Structural regression checks for qualification circuit paths."""

import io
import unittest
from contextlib import redirect_stdout

from per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.subcircuits.decoder import DECODER_CASCADE
from sram_compiler.subcircuits.dummy_row_or_column import Dummy_Cell
from sram_compiler.subcircuits.standard_cell import Pinv
from sram_compiler.subcircuits.time_generate import DelayChain, TaperedBuffer
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench


class PathTests(unittest.TestCase):
    def test_wide_buffer_fingers_preserve_total_transistor_width(self):
        with redirect_stdout(io.StringIO()):
            inv = Pinv('NMOS_VTG', 'PMOS_VTG', 20e-6, 30e-6, 50e-9, max_finger_width=2e-6)
            legacy = Pinv('NMOS_VTG', 'PMOS_VTG', 20e-6, 30e-6, 50e-9)
        self.assertEqual(float(inv['Mpinv_nmos'].width), 20e-6)
        self.assertIn('NF=10', str(inv['Mpinv_nmos']))
        self.assertIn('NF=15', str(inv['Mpinv_pmos']))
        self.assertNotIn('NF=', str(legacy))

    def test_delay_chain_odd_counts_and_polarity(self):
        for stages in (1, 5, 9, 11):
            with self.subTest(stages=stages), redirect_stdout(io.StringIO()):
                chain = DelayChain(stages=stages)
                lines = str(chain).splitlines()
                drivers = [line for line in lines if line.startswith('Xdinv')]
                self.assertEqual(len(drivers), stages)
                self.assertIn(' in ', drivers[0])
                self.assertIn(' out ', drivers[-1])
                self.assertEqual(sum(line.startswith('Xdload') for line in lines), 4 * stages)
        for stages in (0, 2, True, 3.5):
            with self.assertRaises(ValueError):
                DelayChain(stages=stages)

    def test_buffer_effort_adds_stages_at_large_loads(self):
        with redirect_stdout(io.StringIO()):
            small = TaperedBuffer('small', 8, effort_based=True)
            large = TaperedBuffer('large', 4096, effort_based=True)
            legacy = TaperedBuffer('legacy', 4096)
        self.assertEqual(small.n_stages, 2)
        self.assertEqual(large.n_stages, 6)
        self.assertEqual(legacy.n_stages, 4)
        self.assertLessEqual(large.drive_scale ** (1 / large.n_stages), 8)
        with redirect_stdout(io.StringIO()):
            iso = TaperedBuffer('iso', 16, effort_based=True, load_units=64)
        self.assertEqual(iso.n_stages, 2)
        with redirect_stdout(io.StringIO()):
            strong = TaperedBuffer('strong', 16, effort_based=True, load_units=64, fall_strength=2)
        final = list(strong.subcircuits)[-1]
        self.assertEqual(float(final.nmos_width), 0.09e-6 * 16 * 2)
        self.assertEqual(float(final.pmos_width), 0.27e-6 * 16)

    def test_dummy_rc_does_not_create_an_unused_qb_island(self):
        with redirect_stdout(io.StringIO()):
            dummy = Dummy_Cell('NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG',
                               .205e-6, .09e-6, .135e-6, 50e-9, w_rc=True)
        self.assertNotIn('RR_QB', str(dummy))
        self.assertIn('RR_WL', str(dummy))

    def test_decoder_scales_only_final_output_inverters(self):
        with redirect_stdout(io.StringIO()):
            decoder = DECODER_CASCADE('NMOS_VTG', 'PMOS_VTG', 'NMOS_VTG', 'PMOS_VTG',
                                      num_rows=64, output_scale=4)
        early = decoder.decoders_by_level[0][0]
        last = decoder.decoders_by_level[-1][0]
        self.assertNotEqual(early.NAME, last.NAME)
        self.assertEqual(float(last.inv_A0.nmos_width), 0.09e-6)
        self.assertIn('w=3.6e-07', str(last.and_for_en[0]))
        self.assertNotIn('w=3.6e-07', str(early.and_for_en[0]))

    def test_matched_replica_and_canonical_read(self):
        with redirect_stdout(io.StringIO()):
            cfg = load_config(8, 4, 'TT')
            cfg.global_config.sizing = {'mode': 'rules_only', 'replica': {'K': 2, 'N': 5}}
            sizes = resolve_driver_sizes(cfg)
            cfg.sram_6t_cell.nmos_width.value[1] *= 0.8
            tb = Sram6TCoreTestbench(cfg, choose_columnmux=False, driver_sizes=sizes)
            deck = str(tb.create_testbench('read', 7, 3))
        self.assertIn('XRWL VDD VSS VDD wl_en RWL WORDLINEDRIVER', deck)
        self.assertEqual(sum(line.startswith('XRWL_LOAD_') for line in deck.splitlines()), 2)
        replica_instance = next(line for line in deck.splitlines() if line.startswith('Xsram_9x1_replica_column '))
        self.assertEqual(replica_instance.split().count('RWL'), 2)
        self.assertEqual(sum(line.startswith('XWRITEDRIVER_') for line in deck.splitlines()), 4)
        self.assertIn('VDD VSS w_en VSS BL3 BLB3 WRITEDRIVER', deck)
        self.assertIn('XREPLICA_WDRV_LOAD VDD VSS VSS VSS RBL RBLB WRITEDRIVER', deck)
        self.assertEqual(sizes.replica_nmos_widths[1], 0.135e-6)

    def test_rc_precharge_waits_for_the_matched_physical_wordline(self):
        with redirect_stdout(io.StringIO()):
            cfg = load_config(8, 4, 'TT')
            cfg.global_config.sizing = {'mode': 'rules_only'}
            tb = Sram6TCoreTestbench(cfg, choose_columnmux=False, w_rc=True)
            circuit = tb.create_testbench('read', 7, 3)
        time = next(block for block in circuit.subcircuits if block.name == 'TIME')
        self.assertTrue(tb.driver_sizes.replica_precharge_guard)
        self.assertEqual(time.NODES[-1], 'rwl')
        self.assertIn('clk_buf cs rwl_pre_bar PRE_UNBUF', str(time))
        self.assertIn('RWL TIME', str(circuit['XTIME']))


if __name__ == '__main__':
    unittest.main()
