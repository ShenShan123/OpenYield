"""Qualification lookup must never promote stale or incomplete evidence."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.sizing.table import (
    current_scoring_version,
    physical_context,
    qualified_timing,
    record_key,
)
from sram_compiler.sizing.timing import TimingConfig
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench


class TableTests(unittest.TestCase):
    def test_auto_falls_back_then_requires_exact_qualified_record(self):
        with redirect_stdout(io.StringIO()):
            cfg = load_config(8, 4, 'TT')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'table.json'
            cfg.global_config.sizing = {'mode': 'auto', 'table': str(path)}
            sizes = resolve_driver_sizes(cfg)
            self.assertEqual(sizes.source, 'rule')
            context = physical_context()
            record = {'sizes_key': sizes.key, 'physical': context, 'qualified': True,
                          'scoring_version': current_scoring_version(),
                          'campaign_complete': True, 'local_mismatch_qualified': True,
                          'variation_policy': 'full-local-v1', 'local_relative_sigma': .05,
                          'driver_sizes': sizes.to_dict(), 'timing': TimingConfig(2.5e-9, 1e-9, .5e-9, .6e-9).to_dict()}
            table = {'schema': 1, 'records': {record_key(sizes.key, context): record}}
            path.write_text(json.dumps(table))
            qualified = resolve_driver_sizes(cfg)
            self.assertEqual(qualified.source, 'table')
            self.assertEqual(qualified.key, sizes.key)
            self.assertEqual(qualified_timing(qualified, path).t_period, 2.5e-9)
            with self.assertRaisesRegex(ValueError, 'physical context'):
                qualified.validate_for(cfg, 'SRAM_6T_CELL', False, physical_context(True))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'periphery is frozen'):
                Sram6TCoreTestbench(cfg, choose_columnmux=False, driver_sizes=qualified, sweep_precharge=True)
            self.assertEqual(resolve_driver_sizes(cfg, physical_context=physical_context(True)).source, 'rule')
            cfg.sram_6t_cell.pmos_width.upper *= 1.1
            self.assertEqual(resolve_driver_sizes(cfg).source, 'rule')
            cfg.sram_6t_cell.pmos_width.upper /= 1.1
            record['local_mismatch_qualified'] = False
            path.write_text(json.dumps(table))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                resolve_driver_sizes(cfg)


if __name__ == '__main__':
    unittest.main()
