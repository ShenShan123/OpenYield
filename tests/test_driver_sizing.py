"""Rule and generated-netlist regressions; no Xyce execution required."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from sram_compiler.interconnect import load_interconnect
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.sizing.driver_sizing import DEFAULT_LOOKUP, interpolate_class
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

    def test_legacy_fixed_mode_and_its_overrides_are_gone(self):
        # V2.0.9 removed the legacy array rules; the replica is always matched and
        # the canonical read path, decoder scaling and effort buffers are on.
        for options in ({"mode": "fixed"}, {"fixed_scales": {"pre": 3}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_driver_sizes(config(16, 16, "lookup"), sizing=options)
        sizes = resolve_driver_sizes(config(16, 16, "lookup"))
        self.assertTrue(sizes.replica_matched and sizes.canonical_read
                        and sizes.effort_buffers and sizes.dec_inv >= 1)
        unmatched = resolve_driver_sizes(config(16, 16, "lookup"),
                                         sizing={"mode": "lookup", "replica": {"matched": False}})
        self.assertEqual(unmatched.loads.wl_load, sizes.loads.wl_load - 2 + 1)  # AND2 = one unit
        self.assertEqual(sizes.loads.wl_load, 34.0)  # TIME ceil must not see 34 + epsilon.

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
                        {"replica": {"matched": "yes"}}, {"scale_decoder": 1},
                        {"mode": "rules_only", "lookup": "x.json"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_driver_sizes(config(), sizing=options)
        for rows, cols, mux in ((0, 4, False), (True, 4, False), (8, 3, True), (8, 4.5, False)):
            with self.subTest(rows=rows, cols=cols), self.assertRaises(ValueError):
                resolve_driver_sizes(config(rows, cols), mux=mux)


class LookupTests(unittest.TestCase):
    """V2.0.9: every array maps to one fixed integer size class; no continuous rule."""

    def test_default_mode_is_lookup_with_the_new_physical_path(self):
        with redirect_stdout(io.StringIO()):
            cfg = load_config(16, 16, "TT")
        self.assertEqual(cfg.global_config.sizing, {"mode": "lookup"})
        sizes = resolve_driver_sizes(cfg)
        self.assertEqual(sizes.source, "lookup")
        self.assertEqual(sizes.size_class, "rows<=32/cols<=16")
        self.assertTrue(sizes.replica_matched and sizes.canonical_read and sizes.effort_buffers)
        self.assertEqual((sizes.replica_k, sizes.dc_stages), (1, 9))
        self.assertEqual(resolve_driver_sizes(cfg, sizing={}).key, sizes.key)  # omitted mode

    def test_classes_are_integers_never_below_the_v205_rule(self):
        # Every class is the V2.0.5 rule at its upper bound, rounded up: a layout
        # library of fixed devices must not be weaker than the screened rules.
        for rows in (1, 2, 8, 16, 32, 33, 64, 100, 128, 256, 512):
            for cols in (1, 4, 5, 8, 16, 32, 50, 64, 128, 256, 512):
                with self.subTest(rows=rows, cols=cols):
                    cfg = config(rows, cols, "lookup")
                    sizes = resolve_driver_sizes(cfg, mux=False)
                    rule = resolve_driver_sizes(cfg, mux=False, sizing={"mode": "rules_only"})
                    for name in ("pre", "wd_in", "wd_out", "wl_inv", "wl_nand", "dec_inv"):
                        value = getattr(sizes, name)
                        self.assertEqual(value, int(value), name)
                        self.assertGreaterEqual(value, getattr(rule, name), name)
                        self.assertLess(value, 2 * getattr(rule, name) + 1, name)

    def test_class_boundaries_and_expected_values(self):
        for rows, cols, expected, size_class in (
            (8, 4, (1, 1, 2, 1, 1, 1), "rows<=32/cols<=4"),
            (32, 32, (1, 1, 2, 8, 3, 1), "rows<=32/cols<=32"),
            (33, 33, (2, 1, 4, 16, 5, 2), "rows<=64/cols<=64"),
            (64, 16, (2, 1, 4, 4, 2, 1), "rows<=64/cols<=16"),
            (256, 8, (8, 4, 16, 2, 1, 1), "rows<=256/cols<=8"),
            (512, 4, (16, 8, 32, 1, 1, 1), "rows<=512/cols<=4"),
            (16, 512, (1, 1, 2, 128, 35, 9), "rows<=32/cols<=512"),
        ):
            with self.subTest(rows=rows, cols=cols):
                sizes = resolve_driver_sizes(config(rows, cols, "lookup"), mux=False)
                self.assertEqual((sizes.pre, sizes.wd_in, sizes.wd_out, sizes.wl_inv,
                                  sizes.wl_nand, sizes.dec_inv), expected)
                self.assertEqual(sizes.size_class, size_class)

    def test_sizes_do_not_depend_on_cell_mux_rc_or_wires(self):
        cfg = config(16, 16, "lookup")
        base = resolve_driver_sizes(cfg, mux=False)
        variants = [
            resolve_driver_sizes(cfg, mux=True),
            resolve_driver_sizes(cfg, cell_type="SRAM_10T_CELL"),
            resolve_driver_sizes(cfg, physical_context=physical_context(True)),
            resolve_driver_sizes(cfg, physical_context=physical_context(
                interconnect=load_interconnect("sram_compiler/config_yaml/interconnect_example.yaml"))),
        ]
        cfg.sram_6t_cell.pmos_width.value *= 1.3
        variants.append(resolve_driver_sizes(cfg))
        for sizes in variants:
            self.assertEqual(
                (sizes.pre, sizes.wd_in, sizes.wd_out, sizes.wl_inv, sizes.wl_nand, sizes.dec_inv),
                (base.pre, base.wd_in, base.wd_out, base.wl_inv, base.wl_nand, base.dec_inv))
        self.assertEqual(variants[2].loads.pre_load - base.loads.pre_load, 34)  # RC load still counted

    def test_unseen_arrays_interpolate_on_the_ladder_and_extrapolate_beyond_it(self):
        # Inside the table an unseen size rounds up to the next anchor; beyond the
        # last anchor the ladder continues with the last anchor ratio (doubling for
        # rows, 35/18 for the NAND), rounded up to integers and flagged.
        inside = resolve_driver_sizes(config(48, 20, "lookup"), mux=False)
        self.assertEqual((inside.pre, inside.wd_out, inside.wl_inv, inside.wl_nand), (2, 4, 8, 3))
        self.assertFalse(inside.extrapolated)
        tall = resolve_driver_sizes(config(1024, 4, "lookup"), mux=False)
        self.assertTrue(tall.extrapolated)
        self.assertEqual((tall.pre, tall.wd_in, tall.wd_out, tall.wl_inv), (32, 16, 64, 1))
        self.assertEqual(tall.size_class, "rows<=1024 (extrapolated)/cols<=4")
        wide = resolve_driver_sizes(config(8, 1024, "lookup"), mux=False)
        self.assertEqual((wide.wl_inv, wide.wl_nand, wide.dec_inv, wide.pre), (256, 69, 17, 1))
        huge = resolve_driver_sizes(config(3000, 700, "lookup"), mux=False)
        self.assertEqual((huge.pre, huge.wd_out, huge.wl_inv, huge.wl_nand), (128, 256, 256, 69))
        self.assertEqual(huge.size_class, "rows<=4096 (extrapolated)/cols<=1024 (extrapolated)")
        rule = resolve_driver_sizes(config(3000, 700, "lookup"), mux=False, sizing={"mode": "rules_only"})
        for name in ("pre", "wd_in", "wd_out", "wl_inv", "wl_nand", "dec_inv"):
            self.assertGreaterEqual(getattr(huge, name), getattr(rule, name), name)
        self.assertNotEqual(tall.key, resolve_driver_sizes(config(600, 4, "lookup"), mux=False).key)
        single = interpolate_class([{"max_rows": 32, "pre": 1, "wd_in": 1, "wd_out": 2}],
                                   "max_rows", ("pre", "wd_in", "wd_out"), 100)
        self.assertEqual((single["max_rows"], single["pre"], single["wd_out"], single["extrapolated"]),
                         (128, 4, 8, True))

    def test_invalid_tables_fail_explicitly(self):
        for options in ({"mode": "rules_only", "lookup": "x.json"},
                        {"mode": "lookup", "lookup": "missing_lookup.json"}):
            with self.subTest(options=options), self.assertRaises((ValueError, OSError)):
                resolve_driver_sizes(config(), sizing=options)
        good = json.loads(Path(DEFAULT_LOOKUP).read_text())
        bad_tables = []
        table = json.loads(json.dumps(good)); table["row_classes"][1]["max_rows"] = 32; bad_tables.append(table)
        table = json.loads(json.dumps(good)); table["column_classes"][0]["wl_nand"] = 0; bad_tables.append(table)
        table = json.loads(json.dumps(good)); table["schema"] = 2; bad_tables.append(table)
        table = json.loads(json.dumps(good)); del table["row_classes"][0]["wd_in"]; bad_tables.append(table)
        with tempfile.TemporaryDirectory() as temp:
            for index, table in enumerate(bad_tables):
                path = Path(temp) / f"bad{index}.json"
                path.write_text(json.dumps(table))
                with self.subTest(index=index), self.assertRaises(ValueError):
                    resolve_driver_sizes(config(), sizing={"mode": "lookup", "lookup": str(path)})

    def test_alternative_table_changes_sizes_and_fingerprint(self):
        good = json.loads(Path(DEFAULT_LOOKUP).read_text())
        good["row_classes"][0]["wd_out"] = 3
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "layout_library.json"
            path.write_text(json.dumps(good))
            default = resolve_driver_sizes(config(mode="lookup"))
            custom = resolve_driver_sizes(config(mode="lookup"), sizing={"mode": "lookup", "lookup": str(path)})
        self.assertEqual((default.wd_out, custom.wd_out), (2, 3))
        self.assertNotEqual(default.key, custom.key)
        self.assertEqual(custom.loads.wen_load, 4 * (2 * 0.18 * 3 + 0.54) / 0.36 + 8)

    def test_lookup_scales_reach_generated_devices_and_time_loads(self):
        cfg = config(16, 16, "lookup")
        sizes = resolve_driver_sizes(cfg, mux=False)
        with redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as temp:
            tb = Sram6TCoreMcTestbench(cfg, mc=False, choose_columnmux=False, sim_path=temp)
            self.assertEqual(tb.driver_sizes.key, sizes.key)
            circuit = tb.create_testbench("write", 15, 15)
        blocks = {item.name: item for item in circuit.subcircuits}
        self.assertAlmostEqual(float(blocks["PRECHARGE"]["M1"].width), 0.27e-6, places=15)
        self.assertAlmostEqual(float(blocks["WRITEDRIVER"]["M2"].width), 0.18e-6, places=15)
        self.assertAlmostEqual(float(blocks["WRITEDRIVER"]["M7"].width), 0.36e-6, places=15)
        wl = blocks["WORDLINEDRIVER"]
        self.assertAlmostEqual(float(wl.nand_nmos_width), 0.36e-6, places=15)
        self.assertAlmostEqual(float(wl.inv_nmos_width), 0.36e-6, places=15)
        time = blocks["TIME"]
        self.assertEqual((time.pre_load, time.wen_load, time.wl_load),
                         (sizes.loads.pre_load, sizes.loads.wen_load, sizes.loads.wl_load))
        self.assertEqual(sizes.loads.wl_load, 34.0)  # 16 rows + replica, NAND class 2


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
