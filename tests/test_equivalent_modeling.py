"""The equivalent array model is a validated simulation input, not a hidden default.

Modes 1-4 are approximations: they must be reachable from YAML, the command line
and the API, they must be recorded in a run's provenance, and they must never be
mistaken for the full-array coverage that qualification evidence requires.
"""

import contextlib
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

from sram_compiler.equivalent_modeling import MODES, EquivalentConfig, resolve_equivalent
from sram_compiler.per_device_mc import run as per_device_run
from sram_compiler.per_device_mc.run import load_config
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench


class ResolveEquivalentTests(unittest.TestCase):
    def test_every_documented_mode_resolves_and_nothing_else_does(self):
        # A silently accepted unknown mode would build an array nobody asked for.
        self.assertEqual(sorted(MODES), [0, 1, 2, 3, 4])
        for mode in MODES:
            self.assertEqual(resolve_equivalent(mode).mode, mode)
            self.assertEqual(resolve_equivalent({'mode': mode}).mode, mode)
        for bad in (5, -1, 1.0, 'cross', {'mode': 9}, {'modes': 1}, [1]):
            with self.assertRaises(ValueError):
                resolve_equivalent(bad)

    def test_only_mode_zero_is_the_full_array_reference(self):
        self.assertFalse(resolve_equivalent(None).approximate)
        self.assertFalse(resolve_equivalent(0).approximate)
        for mode in (1, 2, 3, 4):
            self.assertTrue(resolve_equivalent(mode).approximate)
        self.assertEqual(resolve_equivalent(EquivalentConfig(2)).to_dict(),
                         {'mode': 2, 'approximate': True})

    def test_legacy_boolean_use_equivalent_keeps_its_meaning(self):
        # The core classes accepted True/False before the modes existed.
        self.assertEqual(resolve_equivalent(True).mode, 1)
        self.assertEqual(resolve_equivalent(False).mode, 0)


class EquivalentInputTests(unittest.TestCase):
    def test_tracked_yaml_default_is_the_full_transistor_array(self):
        # Released evidence is full-array; a shipped YAML default of 1-4 would
        # silently turn every run into an approximation.
        config = load_config(4, 4, 'TT')
        self.assertEqual(resolve_equivalent(config.global_config.equivalent).mode, 0)

    def test_testbench_takes_the_yaml_block_and_an_explicit_argument_overrides_it(self):
        config = load_config(4, 4, 'TT')
        config.global_config.equivalent = {'mode': 2}
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            from_yaml = Sram6TCoreMcTestbench(config, variation_mode='nominal',
                                              choose_columnmux=False, sim_path=temp)
            explicit = Sram6TCoreMcTestbench(config, variation_mode='nominal',
                                             choose_columnmux=False, sim_path=temp,
                                             real_cell_mode=0)
        self.assertEqual(from_yaml.equivalent.mode, 2)
        self.assertEqual(from_yaml.real_cell_mode, 2)
        self.assertEqual(explicit.equivalent.mode, 0)
        self.assertEqual(explicit.real_cell_mode, 0)

    def test_command_line_default_follows_the_yaml_and_is_recorded_in_the_summary(self):
        # Provenance must say which array was solved: a per-device sample over an
        # equivalent array covers only the cells that mode keeps real, so a run
        # that leaves the option out must still record the resolved mode.
        with patch.object(sys, 'argv', ['run.py']):
            self.assertIsNone(per_device_run.parse_args().real_cell_mode)
        with patch.object(sys, 'argv', ['run.py', '--real-cell-mode', '3']):
            self.assertEqual(per_device_run.parse_args().real_cell_mode, 3)

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            argv = ['run.py', '--rows', '2', '--cols', '2', '--variation-mode', 'per-device',
                    '--no-waveform', '--output-dir', temp]
            with patch.object(sys, 'argv', argv):
                _, summary = per_device_run.generate_deck(per_device_run.parse_args())
        self.assertEqual(summary['real_cell_mode'], 0)
        self.assertEqual(summary['equivalent'], {'mode': 0, 'approximate': False})
        self.assertIs(summary['full_device_coverage'], True)

    def test_an_equivalent_run_is_never_reported_as_full_device_coverage(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            argv = ['run.py', '--rows', '2', '--cols', '2', '--variation-mode', 'per-device',
                    '--real-cell-mode', '4', '--no-waveform', '--output-dir', temp]
            with patch.object(sys, 'argv', argv):
                _, summary = per_device_run.generate_deck(per_device_run.parse_args())
        self.assertEqual(summary['equivalent'], {'mode': 4, 'approximate': True})
        self.assertIs(summary['full_device_coverage'], False)


if __name__ == '__main__':
    unittest.main()
