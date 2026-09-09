"""Compiler runtime and qualification lookup must not require local development files."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.sizing.table import (
    SCORING_SOURCES, current_scoring_version, physical_context, record_key,
)
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench


class RuntimePackagingTests(unittest.TestCase):
    def test_default_generation_and_scoring_identity_need_no_development_files(self):
        original_open = Path.open
        project_root = Path(__file__).resolve().parents[1]
        development_roots = [project_root / 'dev', project_root / 'tests']

        def runtime_only_open(path, *args, **kwargs):
            if any(path.resolve().is_relative_to(root) for root in development_roots):
                raise AssertionError(f'Runtime opened development file: {path}')
            return original_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch.object(Path, 'open', runtime_only_open):
            self.assertEqual(len(current_scoring_version()), 64)
            tb = Sram6TCoreMcTestbench(load_config(2, 2, 'TT'), real_cell_mode=0,
                                     mc_seed=3, sim_path=temp)
            circuit = tb.create_testbench('read', 1, 1)
            self.assertEqual(tb.variation_mode, 'per-device')
            self.assertIn('MC_NMOS_', str(circuit))

    def test_changed_scoring_manifest_invalidates_the_accepted_identity(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            config = load_config(8, 4, 'TT')
            table = Path(temp) / 'table.json'
            config.global_config.sizing = {'mode': 'auto', 'table': str(table)}
            sizes = resolve_driver_sizes(config)
            context = physical_context()
            record = {
                'sizes_key': sizes.key, 'physical': context, 'qualified': True,
                'scoring_version': current_scoring_version(), 'campaign_complete': True,
                'local_mismatch_qualified': True, 'variation_policy': 'full-local-v1',
                'local_relative_sigma': .05, 'driver_sizes': sizes.to_dict(),
                'timing': {'t_period': 2.5e-9, 'low_read': 1e-9,
                           'low_write': .5e-9, 'high': .6e-9},
            }
            table.write_text(json.dumps({'schema': 1, 'records': {
                record_key(sizes.key, context): record}}))
            self.assertEqual(resolve_driver_sizes(config).source, 'table')
            manifest = json.loads(SCORING_SOURCES.read_text())
            manifest['sources']['qualification.py'] = '0' * 64
            changed = Path(temp) / 'scoring_sources.json'
            changed.write_text(json.dumps(manifest))
            with patch('sram_compiler.sizing.table.SCORING_SOURCES', changed):
                self.assertEqual(resolve_driver_sizes(config).source, 'rule')
