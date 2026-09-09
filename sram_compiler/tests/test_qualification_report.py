"""Incomplete qualification evidence must never produce a published table."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sram_compiler.sizing.report import export_report, mismatch_evidence
from sram_compiler.sizing.table import current_scoring_version


class ReportTests(unittest.TestCase):
    def test_incomplete_campaign_preserves_existing_table(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'campaign.json').write_text(json.dumps({'complete': False}))
            table = root / 'table.json'
            table.write_text('existing qualification records')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                export_report(root, root / 'report', table)
            self.assertEqual(table.read_text(), 'existing qualification records')
            self.assertFalse((root / 'report').exists())

    def test_missing_local_samples_cannot_be_qualified(self):
        with tempfile.TemporaryDirectory() as temp:
            result = mismatch_evidence({'arrays': {}}, Path(temp))
        self.assertFalse(result['passed'])
        self.assertTrue(any('100-sample local ensemble' in error for error in result['errors']))
        self.assertEqual(result['dV_min']['SRAM_6T_CELL'], .3)

    def test_sequence_below_measured_margin_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = {'passed': True, 'checks': {'sense_margin': True},
                      'metrics': {'min_sequence_dv': .31}}
            cfg = {'cell': 'SRAM_6T_CELL', 'variation': 'nominal',
                   'operation': 'read&write', 'period': 3e-9}
            case = {'case': cfg, 'passed': True, 'samples': [sample],
                    'measures': [], 'signature': 'deck-signature', 'sizes_key': 'baseline',
                    'scoring_version': current_scoring_version(), 'directory': str(root)}
            (root / 'result.json').write_text(json.dumps({**case, 'sizing': {'key': 'baseline'}}))
            campaign = {'complete': True, 'arrays': {'sequence': {
                'qualified': True, 'calibration': {'read_SS': case},
                'verification': [case], 'timing': {'t_period': 3e-9},
            }}}
            (root / 'campaign.json').write_text(json.dumps(campaign))
            mismatch = {'passed': True, 'errors': [], 'dV_min': {'SRAM_6T_CELL': .35}}
            table = root / 'table.json'
            with patch('sram_compiler.sizing.report.mismatch_evidence', return_value=mismatch):
                result = export_report(root, root / 'report', table)
            self.assertFalse(result['passed'])
            self.assertFalse(result['arrays'][0]['measured_sensing_margin_pass'])
            self.assertEqual(json.loads(table.read_text())['records'], {})

    def test_write_tail_requires_distinct_batches_and_isolated_architectures(self):
        sample = {'passed': True, 'checks': {'cell_written': True, 'cell_retained': True}}
        def ensemble(seed, mux=False):
            return {'passed': True, 'samples': [sample] * 10, 'case': {
                'cell': 'SRAM_6T_CELL', 'rows': 8, 'cols': 4, 'corner': 'SF',
                'operation': 'write', 'variation': 'per-device', 'cell_variant': 'write_box',
                'mux': mux, 'w_rc': False, 'real_cell_mode': 0, 'seed': seed}}
        def write_result(cases):
            with tempfile.TemporaryDirectory() as temp:
                report = mismatch_evidence({'arrays': {'a': {'verification': cases}}}, Path(temp))
            return next(item for item in report['local_write'] if item['cell'] == 'SRAM_6T_CELL'
                        and item['rows'] == 8 and item['corner'] == 'SF')
        self.assertTrue(write_result([ensemble(seed) for seed in range(4026, 4036)])['passed'])
        self.assertFalse(write_result([ensemble(4026)] * 10)['passed'])
        self.assertFalse(write_result([ensemble(seed, mux=seed > 4030) for seed in range(4026, 4036)])['passed'])


if __name__ == '__main__':
    unittest.main()
