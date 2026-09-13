"""Every bitline peripheral block has its own physical wire tap."""
import contextlib
import io
import tempfile
import unittest

from sram_compiler.config_yaml.sweep_config import SWEEP_CONFIGS
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from tests.test_interconnect import wire_config


class DistributedPeripheryTests(unittest.TestCase):
    def test_distinct_peripheral_taps_and_matched_replica_in_numeric_and_sweep_decks(self):
        for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
            for mux in (False, True):
                for rc in (False, True):
                    for sweep in (False, True):
                        with self.subTest(cell=cell, mux=mux, rc=rc, sweep=sweep), \
                                tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                            cfg = load_config(4, 4, 'TT')
                            tb = Sram6TCoreMcTestbench(cfg, sram_cell_type=cell, choose_columnmux=mux,
                                w_rc=rc, variation_mode='nominal', sweep_cell=sweep,
                                interconnect=wire_config(), sim_path=temp)
                            if sweep:
                                spec = SWEEP_CONFIGS['cell' if cell == 'SRAM_6T_CELL' else 'cell_10T']
                                tb.gen_model_sweep_generic(module_name=spec['name'], param_model_names=spec['model_params'])
                            circuit = tb.create_testbench('read&write', 3, 3)
                            elements = {e.name.upper(): str(e).split() for e in circuit.elements}
                            self.assertEqual(elements['XPRECHARGE_3'][3:5], ['BL3_periph_tap0', 'BLB3_periph_tap0'])
                            self.assertEqual(elements['XWRITEDRIVER_3'][-3:-1], ['BL3_periph_tap1', 'BLB3_periph_tap1'])
                            self.assertEqual(elements['XPRECHARGE_RBL'][3:5], ['RBL_periph_tap0', 'RBLB_periph_tap0'])
                            self.assertEqual(elements['XREPLICA_WDRV_LOAD'][-3:-1], ['RBL_periph_tap1', 'RBLB_periph_tap1'])
                            if mux:
                                inputs = elements[(tb.cmux_inst_prefix+'_1').upper()]
                                self.assertIn('BL3_periph_tap2', inputs)
                                self.assertIn('RBL_periph_tap2', elements['XREPLICA_MUX'])
                            else:
                                self.assertEqual(elements[(tb.sa_inst_prefix+'_3').upper()][5:7],
                                                 ['BL3_periph_tap2', 'BLB3_periph_tap2'])
                                self.assertEqual(elements['XREPLICA_SENSEAMP'][5:7], ['RBL_periph_tap2', 'RBLB_periph_tap2'])
                            # The array port cannot still join the three blocks.
                            users = [name for name, fields in elements.items() if name.startswith('X') and 'BL3' in fields[1:-1]]
                            self.assertEqual(users, [tb.arr_inst_prefix.upper()])
                            for prefix in ('BL3', 'RBL'):
                                wires = [e for e in circuit.elements if e.name.startswith(f'Rwire_{prefix}_periph_')]
                                self.assertEqual(len(wires), 6)
                                self.assertAlmostEqual(sum(float(e.resistance) for e in wires), 3.)

    def test_mux_select_groups_have_distinct_column_taps(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            tb = Sram6TCoreMcTestbench(load_config(4, 8, 'TT'), choose_columnmux=True,
                variation_mode='nominal', interconnect=wire_config(), sim_path=temp)
            circuit = tb.create_testbench('read', 3, 7)
        for group in range(4):
            element = next(e for e in circuit.elements if e.name.upper() == (tb.cmux_inst_prefix+f'_{group}').upper())
            fields = str(element).split()
            self.assertEqual(fields[5:7], [f'SEL{i}_line_tap{group*2}' for i in range(2)])


if __name__ == '__main__':
    unittest.main()
