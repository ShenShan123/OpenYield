"""main_sram.py is the entrance: seeded per-device mismatch over the full array, no YAML rewrite."""

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import main_sram

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'sram_compiler' / 'config_yaml'


def yaml_digests():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(CONFIG_DIR.glob('*.yaml'))}


class MainEntranceTests(unittest.TestCase):
    def test_default_entrance_is_seeded_per_device_over_the_full_array(self):
        # The compiler's default is per-device mismatch; the entrance must state it
        # explicitly and seed it, otherwise one sample is a silent random draw.
        self.assertEqual(main_sram.VARIATION_MODE, 'per-device')
        self.assertIsNotNone(main_sram.MC_SEED)
        self.assertEqual(main_sram.REAL_CELL_MODE, 0)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            config = main_sram.configure(2, 2, False, 'TT', main_sram.CELL_6T)
            tb = main_sram.build_testbench(config, temp)
            deck = str(tb.create_testbench('write', 1, 1))
        self.assertEqual(tb.variation_mode, 'per-device')
        self.assertEqual(tb.mc_seed, main_sram.MC_SEED)
        self.assertEqual(tb.real_cell_mode, 0)
        self.assertIn('MC_NMOS_', deck)

    def test_script_settings_override_in_memory_without_rewriting_tracked_yaml(self):
        # The previous entrance rewrote global.yaml and sram_6t_cell.yaml on every
        # run; the settings must reach the netlist through the in-memory config only.
        before = yaml_digests()
        candidate = list(main_sram.CELL_6T)
        candidate[2] = 1.0e-7  # pu_width differs from the tracked YAML default
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            config = main_sram.configure(4, 2, False, 'SS', candidate)
            tb = main_sram.build_testbench(config, temp, variation_mode='nominal')
            deck = str(tb.create_testbench('read', 3, 1))
        self.assertEqual(config.global_config.num_rows, 4)
        self.assertEqual(config.global_config.corner, 'SS')
        self.assertEqual(config.sram_6t_cell.pmos_width.value, 1.0e-7)
        self.assertIn('PMOS_VTG l=5e-08 w=1e-07', deck)  # pull-up carries the override
        self.assertEqual(tb.variation_mode, 'nominal')
        self.assertEqual(yaml_digests(), before)

    def test_interconnect_setting_selects_distributed_wires_in_memory(self):
        # The entrance must reach every compiler feature without editing global.yaml;
        # the default stays the star topology the sizing rules were characterised on.
        self.assertIsNone(main_sram.INTERCONNECT_CONFIG)
        before = yaml_digests()
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            star = main_sram.configure(2, 2, False, 'TT', main_sram.CELL_6T)
            star_deck = str(main_sram.build_testbench(star, temp, variation_mode='nominal')
                            .create_testbench('read', 1, 1))
            config = main_sram.configure(
                2, 2, False, 'TT', main_sram.CELL_6T,
                interconnect_config='sram_compiler/config_yaml/interconnect_example.yaml')
            tb = main_sram.build_testbench(config, temp)
            deck = str(tb.create_testbench('read', 1, 1))
        self.assertNotIn('Rwire_WL', star_deck)
        self.assertTrue(tb.interconnect.distributed)
        self.assertIn('Rwire_WL', deck)
        self.assertIn('MC_NMOS_', deck)  # per-device default is retained
        self.assertEqual(yaml_digests(), before)


if __name__ == '__main__':
    unittest.main()
