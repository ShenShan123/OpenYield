"""Rule and generated-netlist regressions; no Xyce execution required."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.sizing.table import physical_context
from sram_compiler.subcircuits.precharge_and_write_driver import WriteDriver
from sram_compiler.testbenches.parameter_factor import (
    PrechargeFactory,
    WordlineDriverFactory,
    WriteDriverFactory,
)
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench


def config(rows=8, cols=4, mode="rules_only"):
    with redirect_stdout(io.StringIO()):
        result = load_config(rows, cols, "TT")
    result.global_config.sizing = {"mode": mode}
    return result


class ResolverTests(unittest.TestCase):
    def test_floor_load_crossover_and_mux(self):
        for rows, cols, pre, wd_in, wd_out, inv, nand in (
            (8, 4, 0.5, 0.5, 1.5, 1, 1),
            (16, 16, 0.5, 0.5, 1.5, 4, 16 / 15),
            (64, 64, 2, 1, 4, 16, 64 / 15),
            (256, 8, 8, 4, 16, 2, 1),
            (16, 512, 0.5, 0.5, 1.5, 128, 512 / 15),
        ):
            with self.subTest(rows=rows, cols=cols):
                sizes = resolve_driver_sizes(config(rows, cols), mux=True)
                self.assertEqual((sizes.pre, sizes.wd_in, sizes.wd_out, sizes.wl_inv,
                                  sizes.wl_nand), (pre, wd_in, wd_out, inv, nand))
                self.assertEqual(sizes.loads.num_sa, cols // 2)
                self.assertEqual(sizes.source, "rule")

    def test_loads_count_split_gates_replica_and_configured_widths(self):
        cfg = config()
        sizes = resolve_driver_sizes(cfg)
        self.assertAlmostEqual(sizes.loads.pre_load, 5.625)
        self.assertAlmostEqual(sizes.loads.wen_load, 17.0)  # 9 units + 8 hold-buffer units
        self.assertAlmostEqual(sizes.loads.wl_load, 9.0)  # 8 rows + fixed replica NAND
        cfg.precharge.pmos_width.value *= 2
        cfg.write_driver.nmos_width.value *= 2
        cfg.wordline_driver.nmos_width.value[0] *= 2
        cfg.wordline_driver.pmos_width.value[0] *= 2
        changed = resolve_driver_sizes(cfg)
        self.assertAlmostEqual(changed.loads.pre_load, 11.25)
        self.assertAlmostEqual(changed.loads.wen_load, 24.0)
        self.assertAlmostEqual(changed.loads.wl_load, 18.0)
        self.assertAlmostEqual(changed.area_precharge_width, .27e-6, places=15)
        self.assertAlmostEqual(changed.area_wordline_width, .54e-6, places=15)

    def test_fixed_defaults_and_explicit_scales(self):
        cfg = config(16, 16, "fixed")
        sizes = resolve_driver_sizes(cfg)
        self.assertEqual((sizes.pre, sizes.wd_in, sizes.wd_out, sizes.wl_inv,
                          sizes.wl_nand), (1, 1, 1, 4, 2))
        self.assertEqual(sizes.loads.wl_load, 32.0)  # TIME ceil must not see 32 + epsilon.
        overridden = resolve_driver_sizes(cfg, sizing={"mode": "fixed", "fixed_scales": {
            "pre": 3, "wd_in": 0.75, "wd_out": 2, "wl_nand": 5,
        }})
        self.assertEqual((overridden.pre, overridden.wd_in, overridden.wd_out,
                          overridden.wl_nand), (3, 0.75, 2, 5))

    def test_parasitic_factor_changes_load_terms_not_small_array_floor(self):
        small = resolve_driver_sizes(config(), sizing={"mode": "rules_only", "parasitic_factor": 2})
        large = resolve_driver_sizes(config(64, 16), sizing={"mode": "rules_only", "parasitic_factor": 2})
        self.assertEqual(small.wd_out, 1.5)
        self.assertEqual((large.pre, large.wd_out, large.wd_in), (4, 8, 2))

    def test_explicit_rc_caps_are_included_in_control_loads(self):
        cfg = config(16, 16)
        base = resolve_driver_sizes(cfg)
        rc = resolve_driver_sizes(cfg, physical_context=physical_context(True))
        self.assertAlmostEqual(rc.loads.pre_load - base.loads.pre_load, 34)
        self.assertAlmostEqual(rc.loads.wen_load - base.loads.wen_load, 32)
        self.assertAlmostEqual(rc.loads.wl_load - base.loads.wl_load, 54.4)
        self.assertAlmostEqual(rc.loads.sen_load, 47.5)
        self.assertAlmostEqual(rc.loads.iso_load, 96)
        self.assertAlmostEqual(rc.dec_inv, 16 / 15)
        self.assertNotEqual(base.key, rc.key)

    def test_result_and_nested_loads_are_frozen_and_serializable(self):
        sizes = resolve_driver_sizes(config())
        with self.assertRaises(FrozenInstanceError):
            sizes.wd_out = 99
        with self.assertRaises(FrozenInstanceError):
            sizes.loads.pre_load = 99
        self.assertEqual(json.loads(json.dumps(sizes.to_dict()))["source"], "rule")

    def test_fingerprint_tracks_baseline_but_not_run_corner(self):
        cfg = config()
        sizes = resolve_driver_sizes(cfg)
        self.assertEqual(sizes.key, resolve_driver_sizes(cfg).key)
        cfg.global_config.corner = "SF"
        cfg.global_config.vdd = 0.9
        cfg.global_config.temperature = 125
        self.assertEqual(sizes.key, resolve_driver_sizes(cfg).key)
        cfg.sram_6t_cell.pmos_width.upper *= 1.1
        self.assertNotEqual(sizes.key, resolve_driver_sizes(cfg).key)
        sizes.validate_for(cfg, "SRAM_6T_CELL", False)  # Candidate cell may differ.
        cfg.precharge.pmos_width.value *= 1.1
        with self.assertRaisesRegex(ValueError, "peripheral"):
            sizes.validate_for(cfg, "SRAM_6T_CELL", False)

    def test_pdk_fingerprint_is_content_based_and_reuse_detects_changes(self):
        cfg = config()
        sizes = resolve_driver_sizes(cfg)
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "copy.spice"
            model.write_bytes(Path(cfg.global_config.pdk_path_TT).read_bytes())
            cfg.global_config.pdk_path_TT = str(model)
            self.assertEqual(sizes.key, resolve_driver_sizes(cfg).key)
            model.write_text(model.read_text() + "\n* changed PDK\n")
            self.assertNotEqual(sizes.key, resolve_driver_sizes(cfg).key)
            with self.assertRaisesRegex(ValueError, "PDK"):
                sizes.validate_for(cfg, "SRAM_6T_CELL", False)

    def test_invalid_options_and_geometry_fail_explicitly(self):
        for options in ({"mode": "typo"}, {"parasitic_factor": 0},
                        {"wd_floor_margin": float("nan")}, {"k_w": float("inf")},
                        {"pre_min": True}, {"replica": {"N": 2}},
                        {"fixed_scales": {"wd_out": -1}},
                        {"fixed_scales": {"typo": 1}},
                        {"mode": "rules_only", "fixed_scales": {"pre": 1}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_driver_sizes(config(), sizing=options)
        for rows, cols, mux in ((0, 4, False), (True, 4, False), (8, 3, True), (8, 4.5, False)):
            with self.subTest(rows=rows, cols=cols), self.assertRaises(ValueError):
                resolve_driver_sizes(config(rows, cols), mux=mux)


class GeneratorTests(unittest.TestCase):
    def test_write_stack_split_reaches_all_twelve_transistors(self):
        driver = WriteDriverFactory("NMOS_VTG", "PMOS_VTG", scale=0.5, out_scale=1.5).create()
        for number in range(1, 13):
            pmos = number in (1, 3, 5, 6, 9, 10)
            expected = (0.36e-6 if pmos else 0.18e-6) * (0.5 if number <= 4 else 1.5)
            self.assertAlmostEqual(float(driver[f"M{number}"].width), expected, places=15)
        legacy = WriteDriver("NMOS_VTG", "PMOS_VTG", 0.18e-6, 0.36e-6, 50e-9)
        self.assertEqual(legacy["M2"].width, legacy["M12"].width)

    @patch("sram_compiler.testbenches.parameter_factor.read_mos_model_from_param_file",
           return_value={"nmos": 0, "pmos": 0})
    def test_swept_widths_keep_resolved_scales(self, _read_models):
        choices = {"pmos_modle_choices": ["PMOS_VTG"], "nmos_modle_choices": ["NMOS_VTG"]}
        write = WriteDriverFactory("NMOS_VTG", "PMOS_VTG", scale=0.5, out_scale=1.5,
                                   sweep_writedriver=True, **choices).create()
        self.assertIn("w={nmos_width_wrd*0.5}", str(write["M4"]))
        self.assertIn("w={nmos_width_wrd*1.5}", str(write["M7"]))
        pre = PrechargeFactory("PMOS_VTG", scale=2.5, sweep_precharge=True,
                               pmos_modle_choices=["PMOS_VTG"]).create()
        self.assertIn("w={pmos_width_precharge*2.5}", str(pre["M1"]))
        wl = WordlineDriverFactory("NMOS_VTG", "PMOS_VTG", inverter_scale=4,
                                    nand_gate_scale=3, sweep_wordlinedriver=True, **choices).create()
        self.assertIn("w={nmos_width_wld_nandn*3}", str(wl))
        self.assertIn("w={nmos_width_wld_invn*4}", str(wl))

    def test_testbench_reuses_baseline_and_passes_loads_to_time(self):
        cfg = config()
        sizes = resolve_driver_sizes(cfg)
        cfg.sram_6t_cell.pmos_width.value *= 1.2
        with redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as temp:
            tb = Sram6TCoreMcTestbench(cfg, mc=False, choose_columnmux=False,
                                      driver_sizes=sizes, sim_path=temp)
            self.assertIs(tb.driver_sizes, sizes)
            circuit = tb.create_testbench("write", 7, 3)
        subcircuits = {item.name: item for item in circuit.subcircuits}
        self.assertAlmostEqual(float(subcircuits["WRITEDRIVER"]["M7"].width), 0.27e-6, places=15)
        time = subcircuits["TIME"]
        self.assertEqual((time.pre_load, time.wen_load, time.wl_load, time.num_sa),
                         (sizes.loads.pre_load, sizes.loads.wen_load,
                          sizes.loads.wl_load, sizes.loads.num_sa))

    def test_reusing_wrong_architecture_or_mutated_periphery_is_rejected(self):
        cfg = config()
        sizes = resolve_driver_sizes(cfg)
        cfg.global_config.num_rows = 16
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "architecture"):
            Sram6TCoreTestbench(cfg, choose_columnmux=False, driver_sizes=sizes)
        cfg.global_config.num_rows = 8
        with redirect_stdout(io.StringIO()):
            tb = Sram6TCoreTestbench(cfg, choose_columnmux=False, driver_sizes=sizes)
        cfg.write_driver.nmos_width.value *= 2
        with self.assertRaisesRegex(ValueError, "peripheral"):
            tb.create_testbench("write", 7, 3)

    def test_new_precharge_and_wordline_rules_reach_the_generated_mos(self):
        cfg = config(16, 16)
        with redirect_stdout(io.StringIO()):
            tb = Sram6TCoreTestbench(cfg, choose_columnmux=False)
            circuit = tb.create_testbench("read", 15, 15)
        blocks = {item.name: item for item in circuit.subcircuits}
        self.assertAlmostEqual(float(blocks["PRECHARGE"]["M1"].width), 0.135e-6, places=15)
        wl = blocks["WORDLINEDRIVER"]
        self.assertAlmostEqual(float(wl.nand_nmos_width), 0.192e-6, places=15)
        self.assertAlmostEqual(float(wl.inv_nmos_width), 0.36e-6, places=15)

    def test_both_cells_and_mux_generate_read_and_write_decks(self):
        for cell in ("SRAM_6T_CELL", "SRAM_10T_CELL"):
            for mux in (False, True):
                for operation in ("read", "write"):
                    with self.subTest(cell=cell, mux=mux, operation=operation):
                        with redirect_stdout(io.StringIO()):
                            tb = Sram6TCoreTestbench(config(), sram_cell_type=cell,
                                                    choose_columnmux=mux, real_cell_mode=0)
                            deck = str(tb.create_testbench(operation, 7, 3))
                        self.assertIn(".subckt PRECHARGE", deck)
                        self.assertIn(".subckt WORDLINEDRIVER", deck)
                        self.assertEqual(tb.driver_sizes.loads.num_sa, 2 if mux else 4)


if __name__ == "__main__":
    unittest.main()
