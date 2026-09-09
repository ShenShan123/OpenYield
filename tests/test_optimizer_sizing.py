"""The legacy evaluation adapter must retain the baseline and report constraints."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from sram_compiler.per_device_mc.run import load_config
from size_optimization import exp_utils


class OptimizerSizingTests(unittest.TestCase):
    def test_candidates_share_baseline_and_access_violations_are_reported(self):
        seen = []
        def config():
            cfg = load_config(32, 1, 'TT')
            cfg.global_config.sizing = {'mode': 'rules_only'}
            cfg.load_all_configs = lambda **kwargs: None
            return cfg

        class Testbench:
            def __init__(self, cfg, **kwargs):
                seen.append((kwargs['driver_sizes'], cfg.sram_6t_cell.pmos_width.value,
                             cfg.global_config.num_rows, kwargs['choose_columnmux']))
                self.timing_config = None
                self.t_period = 10e-9
            def run_mc_simulation(self, operation, **kwargs):
                if 'snm' in operation:
                    return .1
                return (300e-12 if operation == 'read' else 80e-12), 1e-6, .1e-6, .9e-6

        params = {'pd_width': 205e-9, 'pg_width': 135e-9, 'pu_width': 90e-9, 'length': 50e-9}
        exp_utils._baseline_sizes.cache_clear()
        with redirect_stdout(io.StringIO()), patch.object(exp_utils, 'SRAM_CONFIG', side_effect=config), \
             patch.object(exp_utils, '_load_sram_config_from_yaml', side_effect=config), \
             patch.object(exp_utils, 'Sram6TCoreMcTestbench', Testbench):
            first = exp_utils.evaluate_sram(params)
            second = exp_utils.evaluate_sram(dict(params, pu_width=108e-9))
        self.assertIs(seen[0][0], seen[1][0])
        self.assertEqual([row[1] for row in seen], [90e-9, 108e-9])
        self.assertEqual(seen[1][2:], (32, False))
        self.assertTrue(first[3] and second[3])
        self.assertFalse(second[2]['read_delay_feasible'])
        self.assertTrue(second[2]['write_delay_feasible'])
        self.assertAlmostEqual(second[1][0], .5)
        self.assertEqual(second[1][1], 0)
        self.assertEqual(first[2]['driver_sizes']['key'], second[2]['driver_sizes']['key'])
        exp_utils._baseline_sizes.cache_clear()


if __name__ == '__main__':
    unittest.main()
