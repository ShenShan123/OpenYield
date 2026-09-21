"""The opt-in waveform scorer fails closed on real electrical safety violations."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.spice.phased_waveforms import (
    PROBE_EVERY_CELL_UPTO, SENSE_MARGIN_FRACTION, probed_rows, score, threshold_overlap)


class PhasedWaveformTests(unittest.TestCase):
    def fixture(self, directory, corrupt=None):
        time = np.arange(0, 19.0001, .05) * 1e-9
        def pulse(start, stop):
            return ((time >= start * 1e-9) & (time < stop * 1e-9)).astype(float)
        def pulses(start, stop):
            return pulse(start, stop) + pulse(start + 9, stop + 9)
        signals = {
            'A0': np.zeros_like(time),
            'WE': np.ones_like(time), 'CS': np.ones_like(time),
            'WL_LOCAL': pulses(4, 7.7), 'WL_FAR': pulses(4, 7.7),
            'PRE_LOCAL': pulses(3, 9), 'WD_EN': pulses(3.7, 8.2),
            'SA_EN': np.zeros_like(time), 'SA_ISO': pulses(3.3, 8.5),
            'DATA': pulse(2.9, 11.9),
            'XARRAY:XCELL_0_0:Q': pulse(4.5, 13.5),
            'XARRAY:XCELL_0_0:QB': 1 - pulse(4.5, 13.5),
            'XARRAY:BL0_far': 1 - pulse(12.7, 18),
            'XARRAY:BLB0_far': 1 - pulse(3.7, 9),
            'RBL': 1 - pulses(3.7, 9),
            'RBL_DELAY': pulses(3.9, 9.5),
            'XTIME_CONTROL:wordline_busy': pulses(4, 8),
            'XTIME_CONTROL:iso_ready': pulses(3.6, 8.6),
            'XTIME_CONTROL:access_request': pulses(3.2, 7.5),
            'XTIME_CONTROL:access_settled': pulses(3.9, 8.0),
            'XTIME_CONTROL:read_done': np.zeros_like(time),
            'XTIME_CONTROL:enables_off': 1 - pulses(3.7, 8.5),
        }
        if corrupt:
            corrupt(time, signals)
        metadata = {
            'period': 9e-9, 'vdd': 1., 'rows': 1, 'cols': 1, 'row': 0,
            'sample_interval': .05e-9, 'analysis_stop': 19e-9,
            'operation': 'write', 'case': {},
            'nodes': {
                'cells': {'0,0': ['XARRAY:XCELL_0_0:Q', 'XARRAY:XCELL_0_0:QB']},
                'wl': {'0': ['WL_LOCAL', 'WL_FAR']},
                'pre': {'0': 'PRE_LOCAL'}, 'wen': {'0': 'WD_EN'},
                'sen': {'0': 'SA_EN'}, 'iso': {'0': 'SA_ISO'},
                'data': {'0': 'DATA'}, 'sense': {},
            },
        }
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        names = list(signals)
        data = np.column_stack([np.arange(len(time)), time, *[signals[n] for n in names]])
        header = 'Index TIME ' + ' '.join(f'V({n})' for n in names)
        np.savetxt(directory / 'deck.sp.prn', data, header=header, comments='',
                   fmt=['%d'] + ['%.12e'] * (len(names) + 1))

    def test_valid_consecutive_writes_release_and_restore_before_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            result = score(directory)
            self.assertTrue(result['passed'], result['failures'])
            self.assertGreater(result['metrics']['min_driver_release_after_wl_ps'], 400)

    def test_recovery_margin_is_reported_and_must_not_go_negative(self):
        """The enable-off to precharge gap is a number, not only a yes/no answer.

        The superseded root-only enable observer left 20.55 ps here at 8x256 SS
        while every exclusion still passed, so a screen that only answers "no
        overlap" cannot show how close it came.
        """
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            result = score(directory)
            self.assertTrue(result['passed'], result['failures'])
            # WD_EN crosses 0.1 VDD at 8.195 ns, PRE crosses 0.9 VDD at 8.955 ns.
            self.assertAlmostEqual(result['metrics']['min_enable_off_before_precharge_ps'], 760, places=6)

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            def corrupt(time, signals):
                # The driver is still enabled after this cycle's precharge fires.
                signals['WD_EN'][np.abs(time - 9.5e-9) < .06e-9] = 1.
            self.fixture(directory, corrupt)
            result = score(directory)
            self.assertFalse(result['passed'])
            self.assertLess(result['metrics']['min_enable_off_before_precharge_ps'], 0.)
            self.assertIn('min_enable_off_before_precharge_ps_nonnegative', result['failures'])

    def test_access_and_recovery_margins_are_reported_and_cannot_be_negative(self):
        """A passing screen must show how much of each phase budget it used.

        V2.2.1 reported only ordering margins; the 8x512 read that met the
        runtime k + 0.7 check while OUT was still on the wrong rail at the
        checker's k + 0.68 deadline is invisible without the output margin.
        """
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            result = score(directory)
            self.assertTrue(result['passed'], result['failures'])
            # Q crosses mid-rail at 4.475 ns and WL_FAR leaves 0.1 VDD at 7.695 ns;
            # BLB0_far / RBL cross 0.9 VDD at 8.995 ns and PRE turns on at
            # 8.955 ns, before the 11.8 ns capture.
            self.assertAlmostEqual(result['metrics']['min_write_wl_after_flip_ps'], 3220, delta=10)
            self.assertAlmostEqual(result['metrics']['min_restore_before_capture_ps'], 2805, delta=10)
            self.assertAlmostEqual(result['metrics']['min_precharge_on_before_capture_ps'], 2845, delta=10)
            self.assertNotIn('min_read_output_margin_ps', result['metrics'])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.read_fixture(directory)
            result = score(directory)
            self.assertTrue(result['passed'], result['failures'])
            # OUT leaves its wrong rail at 5.495 ns, before the 7.12 ns deadline.
            self.assertAlmostEqual(result['metrics']['min_read_output_margin_ps'], 1625, delta=10)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            def late_output(time, signals):
                signals['OUT'][np.abs(time - 7.3e-9) < .06e-9] = 1.
            self.read_fixture(directory, corrupt=late_output)
            result = score(directory)
            self.assertFalse(result['passed'])
            self.assertLess(result['metrics']['min_read_output_margin_ps'], 0.)
            self.assertIn('min_read_output_margin_ps_nonnegative', result['failures'])
            self.assertIn('cycle0_read_output', result['failures'])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            def late_flip(time, signals):
                late = (time >= 4.5e-9) & (time < 7.8e-9)
                signals['XARRAY:XCELL_0_0:Q'][late] = 0.
                signals['XARRAY:XCELL_0_0:QB'][late] = 1.
            self.fixture(directory, late_flip)
            result = score(directory)
            self.assertFalse(result['passed'])
            self.assertLess(result['metrics']['min_write_wl_after_flip_ps'], 0.)
            self.assertIn('min_write_wl_after_flip_ps_nonnegative', result['failures'])

    def test_correct_final_data_does_not_excuse_overlap_or_unfinished_recovery(self):
        hazards = [
            ('PRE_LOCAL', 0., 6., 'pre_exclusion'),
            ('WD_EN', 0., 6., 'driver_covers_wl'),
            ('SA_EN', 1., 6., 'sense_after_wl'),
            ('SA_ISO', 0., 6., 'isolate_before_enable'),
            ('XTIME_CONTROL:wordline_busy', 1., 11.8, 'capture_reset'),
        ]
        for node, value, when, failure in hazards:
            with self.subTest(hazard=node), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                def corrupt(time, signals):
                    signals[node][np.abs(time - when * 1e-9) < .06e-9] = value
                self.fixture(directory, corrupt)
                result = score(directory)
                self.assertFalse(result['passed'])
                self.assertTrue(any(failure in name for name in result['failures']), result['failures'])

    def test_missing_or_truncated_waveforms_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            path = directory / 'deck.sp.prn'
            lines = path.read_text().splitlines()
            path.write_text('\n'.join(lines[:-20]) + '\n')
            self.assertIn('complete', score(directory)['failures'])
            path.write_text(path.read_text().replace('V(WD_EN)', 'V(UNRELATED)'))
            with self.assertRaises(KeyError):
                score(directory)

    def test_forbidden_overlap_immediately_before_capture_is_not_an_unchecked_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            def corrupt(time, signals):
                at_boundary = np.abs(time - 11.75e-9) < .01e-9
                for node in ('WD_EN', 'SA_EN', 'WL_LOCAL', 'WL_FAR'):
                    signals[node][at_boundary] = 1.
            self.fixture(directory, corrupt)
            result = score(directory)
            self.assertFalse(result['passed'])
            self.assertIn('continuous_c0_read_write_exclusion', result['failures'])
            self.assertIn('continuous_c0_pre_exclusion', result['failures'])
            self.assertIn('continuous_c0_isolation', result['failures'])

    def read_fixture(self, directory, idle=False, corrupt=None):
        span = 4 if idle else 2
        time = np.arange(0, 1 + 9 * span + .0001, .05) * 1e-9
        def pulse(start, stop):
            return ((time >= start * 1e-9) & (time < stop * 1e-9)).astype(float)
        def accesses(start, stop):
            return sum(pulse(start + offset, stop + offset)
                       for offset in range(0, 9 * span, 18 if idle else 9))
        zeros, ones = np.zeros_like(time), np.ones_like(time)
        signals = {
            'A0': zeros.copy(), 'WE': zeros.copy(), 'CS': accesses(2.8, 11.8),
            'WL_LOCAL': accesses(4, 5), 'WL_FAR': accesses(4, 5),
            'PRE_LOCAL': sum(pulse(3 + offset, 9 + offset) for offset in range(0, 9 * span, 9)),
            'WD_EN': zeros.copy(), 'SA_EN': accesses(5.3, 7.8),
            'SA_ISO': accesses(4.6, 8), 'DATA': zeros.copy(),
            'XARRAY:XCELL_0_0:Q': zeros.copy(), 'XARRAY:XCELL_0_0:QB': ones.copy(),
            'XARRAY:BL0_far': 1 - accesses(4.2, 9), 'XARRAY:BLB0_far': ones.copy(),
            'RBL': 1 - accesses(4.2, 9), 'RBLB': ones.copy(),
            'RBL_DELAY': accesses(4.6, 9.5),
            'SA_Q': 1 - accesses(4.5, 9.1), 'SA_QB': ones.copy(),
            'OUT': (time < 5.5e-9).astype(float),
            'XTIME_CONTROL:wordline_busy': accesses(4, 5.2),
            'XTIME_CONTROL:iso_ready': accesses(4.8, 8.2),
            'XTIME_CONTROL:access_request': accesses(3.2, 7.5),
            'XTIME_CONTROL:access_settled': accesses(3.9, 8),
            'XTIME_CONTROL:read_done': accesses(4.6, 7.5),
            'XTIME_CONTROL:enables_off': 1 - accesses(5.3, 8.4),
        }
        if corrupt:
            corrupt(time, signals)
        self.fixture(directory)
        metadata = json.loads((directory / 'metadata.json').read_text())
        metadata.update(operation='read', case={'select_every': 2 if idle else 1})
        metadata['analysis_stop'] = (1 + 9 * span) * 1e-9
        metadata['nodes']['sense'] = {'0': ['RBL', 'RBLB']}
        metadata['nodes']['sense_state'] = {'0': ['SA_Q', 'SA_QB']}
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        names = list(signals)
        data = np.column_stack([np.arange(len(time)), time, *[signals[n] for n in names]])
        np.savetxt(directory / 'deck.sp.prn', data,
                   header='Index TIME ' + ' '.join(f'V({n})' for n in names), comments='',
                   fmt=['%d'] + ['%.12e'] * (len(names) + 1))

    def test_reads_and_idle_cycles_have_correct_sense_data_and_quiet_idle_enables(self):
        for idle in (False, True):
            with self.subTest(idle=idle), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.read_fixture(directory, idle)
                result = score(directory)
                self.assertTrue(result['passed'], result['failures'])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            def corrupt(time, signals):
                signals['SA_EN'][np.abs(time - 14e-9) < .06e-9] = 1.
                signals['SA_ISO'][np.abs(time - 14e-9) < .06e-9] = 1.
            self.read_fixture(directory, idle=True, corrupt=corrupt)
            result = score(directory)
            self.assertFalse(result['passed'])
            self.assertTrue(any('cycle1_idle' in name and 'sense_off' in name for name in result['failures']))

    def test_read_and_idle_state_reversals_before_completion_cannot_pass(self):
        for idle, when in ((False, 4.1), (False, 6.), (False, 7.), (False, 14.1), (True, 14.1)):
            with self.subTest(idle=idle, when=when), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                def corrupt(time, signals):
                    changed = abs(time - when * 1e-9) < .1e-9
                    signals['XARRAY:XCELL_0_0:Q'][changed] = 1.
                    signals['XARRAY:XCELL_0_0:QB'][changed] = 0.
                self.read_fixture(directory, idle=idle, corrupt=corrupt)
                result = score(directory)
                self.assertFalse(result['passed'])
                self.assertTrue(any('logical_retention' in name for name in result['failures']))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            def read_disturb(time, signals):
                signals['XARRAY:XCELL_0_0:Q'][abs(time - 4.1e-9) < .1e-9] = .25
            self.read_fixture(directory, corrupt=read_disturb)
            result = score(directory)
            self.assertTrue(result['passed'], result['failures'])

    def test_a_write_must_preserve_unselected_row_state_before_its_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            path = directory / 'deck.sp.prn'
            header = path.read_text().splitlines()[0]
            original = np.loadtxt(path, skiprows=1)
            time = original[:, 1]
            zeros, ones = np.zeros(len(time)), np.ones(len(time))
            data = np.column_stack([original, zeros, zeros, ones])
            header += ' V(WL_OTHER) V(XARRAY:XCELL_1_0:Q) V(XARRAY:XCELL_1_0:QB)'
            metadata = json.loads((directory / 'metadata.json').read_text())
            metadata['rows'] = 2
            metadata['nodes']['wl']['1'] = ['WL_OTHER']
            metadata['nodes']['cells']['1,0'] = ['XARRAY:XCELL_1_0:Q', 'XARRAY:XCELL_1_0:QB']
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            def save():
                np.savetxt(path, data, header=header, comments='',
                           fmt=['%d'] + ['%.12e'] * (data.shape[1] - 1))
            save()
            baseline = score(directory)
            self.assertTrue(baseline['passed'], baseline['failures'])
            changed = abs(time - 6e-9) < .1e-9
            data[changed, -2], data[changed, -1] = 1., 0.
            save()
            result = score(directory)
            self.assertFalse(result['passed'])
            self.assertIn('cycle0_write_cell1,0_logical_retention', result['failures'])

    def test_sense_data_must_hold_after_completion_until_sense_enable_releases(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            def corrupt(time, signals):
                changed = abs(time - 7.4e-9) < .1e-9
                signals['SA_Q'][changed] = 1.
                signals['SA_QB'][changed] = 0.
            self.read_fixture(directory, corrupt=corrupt)
            result = score(directory)
            self.assertFalse(result['passed'])
            self.assertIn('cycle0_read_sense_group0_hold', result['failures'])

    def test_missing_interior_row_cannot_hide_an_unsafe_sample(self):
        for renumber in (False, True):
            with self.subTest(renumber=renumber), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                def corrupt(time, signals):
                    signals['SA_EN'][np.abs(time - 6e-9) < .01e-9] = 1.
                self.fixture(directory, corrupt)
                self.assertFalse(score(directory)['passed'])
                path = directory / 'deck.sp.prn'
                lines = path.read_text().splitlines()
                kept = [line for line in lines[1:] if abs(float(line.split()[1]) - 6e-9) > 1e-15]
                if renumber:
                    kept = [' '.join([str(index), *line.split()[1:]]) for index, line in enumerate(kept)]
                path.write_text('\n'.join([lines[0], *kept]) + '\n')
                result = score(directory)
                self.assertFalse(result['passed'])
                self.assertIn('sample_count', result['failures'])
                self.assertIn('time_grid' if renumber else 'index_contiguous', result['failures'])

    def test_small_output_timestamp_jitter_is_allowed_but_displaced_samples_fail(self):
        for displacement, valid in ((.0008, True), (.02, False)):
            with self.subTest(displacement=displacement), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.fixture(directory)
                path = directory / 'deck.sp.prn'
                lines = path.read_text().splitlines()
                sample = lines[120].split()
                sample[1] = str(float(sample[1]) + displacement * .05e-9)
                lines[120] = ' '.join(sample)
                path.write_text('\n'.join(lines) + '\n')
                result = score(directory)
                self.assertEqual(result['passed'], valid, result['failures'])
                self.assertEqual('time_grid' in result['failures'], not valid)

    def test_negative_interpolated_driver_setup_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            def corrupt(time, signals):
                signals['WD_EN'][(time >= 3.7e-9) & (time < 3.925e-9)] = 0.
                signals['WD_EN'][np.abs(time - 3.95e-9) < .001e-9] = .8
            self.fixture(directory, corrupt)
            result = score(directory)
            self.assertFalse(result['passed'])
            self.assertAlmostEqual(result['metrics']['min_driver_on_before_wl_ps'], -20.)
            self.assertIn('min_driver_on_before_wl_ps_nonnegative', result['failures'])
            self.assertIn('continuous_c0_driver_covers_wl', result['failures'])

    def test_data_must_remain_stable_through_the_interpolated_wordline_tail(self):
        for when, valid in ((7.7, False), (7.75, True)):
            with self.subTest(when=when), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                def corrupt(time, signals):
                    signals['DATA'][np.abs(time - when * 1e-9) < .001e-9] = 0.
                self.fixture(directory, corrupt)
                result = score(directory)
                self.assertEqual(result['passed'], valid, result['failures'])
                self.assertEqual('cycle0_write_c0_data_stable' in result['failures'], not valid)

    def test_partial_final_write_still_requires_stable_data(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            path = directory / 'deck.sp.prn'
            header = path.read_text().splitlines()[0]
            names = header.split()
            original = np.loadtxt(path, skiprows=1)
            time = np.arange(0, 79.3001, .05) * 1e-9
            # Repeat the valid two-write fixture through eight full accesses
            # and a ninth partial one, like the mixed-deck stop at 8.7 T.
            folded = np.where(time < 1e-9, time, 1e-9 + (time - 1e-9) % 18e-9)
            data = np.column_stack([np.arange(len(time)), time,
                                    *[np.interp(folded, original[:, 1], original[:, col])
                                      for col in range(2, len(names))]])
            # Custom patterns retain their eighth captured request/data.
            for node, value in (('DATA', 0.), ('XARRAY:XCELL_0_0:Q', 0.), ('XARRAY:XCELL_0_0:QB', 1.)):
                data[time >= 73e-9, names.index(f'V({node})')] = value
            metadata = json.loads((directory / 'metadata.json').read_text())
            metadata.update(operation='read&write', analysis_stop=79.3e-9,
                            case={'pattern': [{'op': 'write', 'row': 0, 'data': str(1 - k % 2)}
                                              for k in range(8)]})
            metadata['nodes']['sense'] = {'0': ['RBL', 'XARRAY:BL0_far']}
            metadata['nodes']['sense_state'] = {'0': ['XARRAY:XCELL_0_0:Q', 'XARRAY:XCELL_0_0:QB']}
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            def save():
                np.savetxt(path, data, header=header, comments='',
                           fmt=['%d'] + ['%.12e'] * (data.shape[1] - 1))
            save()
            baseline = score(directory)
            self.assertTrue(baseline['passed'], baseline['failures'])
            data[np.argmin(abs(time - 78.1e-9)), names.index('V(DATA)')] = 1.
            save()
            result = score(directory)
            self.assertFalse(result['passed'])
            self.assertIn('cycle8_write_c0_data_stable', result['failures'])
            data[np.argmin(abs(time - 78.1e-9)), names.index('V(DATA)')] = 0.
            data = np.column_stack([data, np.zeros(len(time)), np.zeros(len(time)), np.ones(len(time))])
            header += ' V(WL_OTHER) V(XARRAY:XCELL_1_0:Q) V(XARRAY:XCELL_1_0:QB)'
            metadata['rows'] = 2
            metadata['nodes']['wl']['1'] = ['WL_OTHER']
            metadata['nodes']['cells']['1,0'] = ['XARRAY:XCELL_1_0:Q', 'XARRAY:XCELL_1_0:QB']
            (directory / 'metadata.json').write_text(json.dumps(metadata))
            save()
            baseline = score(directory)
            self.assertTrue(baseline['passed'], baseline['failures'])
            data[(time >= 77e-9) & (time < 78e-9), -3] = 1.
            save()
            self.assertIn('cycle8_write_wl1_0_unselected', score(directory)['failures'])
            data[:, -3] = 0.
            for node in ('WD_EN', 'WL_LOCAL', 'WL_FAR'):
                data[time >= 74.8e-9, names.index(f'V({node})')] = 0.
            data[(time >= 77e-9) & (time < 78e-9), names.index('V(SA_EN)')] = 1.
            save()
            self.assertIn('cycle8_write_c0_sen_quiet', score(directory)['failures'])

    def test_complementary_transitions_cannot_hide_between_sample_overlap(self):
        # At each recorded endpoint only one condition is true; the linear
        # transitions overlap for 80% of the interval.
        time = np.array([0., 1.])
        rising, falling = np.array([0., 1.]), np.array([1., 0.])
        hazards = [
            ((rising, .1, True), (falling, .1, True)),  # WEN / SEN
            ((falling, .9, False), (falling, .1, True)),  # PRE / enable
            ((rising, .9, False), (rising, .1, True)),  # ISO / enable
            ((rising, .1, True), (rising, .9, False)),  # WL / WEN
        ]
        for conditions in hazards:
            with self.subTest(conditions=conditions):
                self.assertTrue(threshold_overlap(time, conditions, 0.))
                self.assertFalse(threshold_overlap(time, conditions, 1.))
        self.assertFalse(threshold_overlap(time, ((rising, .5, True), (falling, .5, True)), 0.))
        self.assertFalse(threshold_overlap(time, ((rising, 1., True),), 0.))
        self.assertFalse(threshold_overlap(time, ((rising, .5, True),), 0., .5))
        self.assertTrue(threshold_overlap(time, ((rising, .5, True),), 0., .6))

    def test_enable_release_requires_precharge_and_isolation_margin_between_samples(self):
        for node, failure in (('PRE_LOCAL', 'pre_exclusion'), ('SA_ISO', 'isolation')):
            with self.subTest(node=node), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                def corrupt(time, signals):
                    signals[node][(time >= 8.2e-9) & (time < 9e-9)] = 0.
                self.fixture(directory, corrupt)
                result = score(directory)
                self.assertFalse(result['passed'])
                self.assertIn(f'continuous_c0_{failure}', result['failures'])

    def test_missing_row_or_cell_probe_contract_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.fixture(directory)
            path = directory / 'metadata.json'
            metadata = json.loads(path.read_text())
            metadata['rows'] = 2
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'Every row'):
                score(directory)
            metadata['nodes']['wl']['1'] = ['WL_LOCAL', 'WL_FAR']
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'cell probes'):
                score(directory)


class ProbeAndBarTests(unittest.TestCase):
    """V2.2.3: what the screen looks at, and what it demands of a read."""

    def test_the_runner_and_the_checker_derive_the_same_probed_rows(self):
        """They are separate modules and the checker fails closed on a missing probe, so a
        rule copied into both would silently reduce coverage the day one copy changed."""
        import inspect

        from tests.spice import phased_access

        source = inspect.getsource(phased_access.prepare)
        self.assertIn('probed_rows(rows, cols, case, tb.target_row)', source)
        self.assertNotIn('rows // 2', source)

    def test_large_arrays_probe_the_neighbours_of_every_addressed_row(self):
        """Through V2.2.2 a 512x4 trace probed 4 of 512 rows: first, middle, last and target.
        A wordline or bitline excursion disturbs the rows next to the accessed one first, and
        those were exactly the rows nobody looked at."""
        self.assertEqual(probed_rows(8, 4, {}, 7), set(range(8)))
        self.assertEqual(probed_rows(32, 32, {}, 31), set(range(32)))
        self.assertEqual(probed_rows(512, 4, {}, 511), {0, 128, 256, 384, 510, 511})
        addressed = probed_rows(256, 256, {'pattern': [{'row': 0}, {'row': 200}]}, 255)
        for row in (0, 1, 199, 200, 201, 254, 255):
            self.assertIn(row, addressed)
        self.assertEqual(probed_rows(128, 128, {'next_row': 64}, 127),
                         {0, 32, 63, 64, 65, 96, 126, 127})
        self.assertEqual(PROBE_EVERY_CELL_UPTO, 1024)

    def test_the_sense_bar_scales_with_the_rail_it_is_measured_against(self):
        """V2.2.2 fixed it at 0.25 V, which is 28 % of a 0.9 V rail but only 23 % of a 1.1 V
        one, so the fast corner was screened to a weaker bar than the slow corner. The fraction
        is the 0.9 V bar, so no point that passed before is relaxed."""
        self.assertAlmostEqual(SENSE_MARGIN_FRACTION * .9, .25, places=2)
        self.assertGreaterEqual(SENSE_MARGIN_FRACTION * .9, .25)
        for vdd, differential, expected in ((.9, .26, True), (.9, .24, False),
                                            (1.1, .29, False), (1.1, .32, True)):
            with self.subTest(vdd=vdd, differential=differential):
                self.assertEqual(differential >= SENSE_MARGIN_FRACTION * vdd, expected)

    def test_the_runtime_deadline_and_the_screen_deadline_are_one_number(self):
        """The V2.2.1 8x512 escape was a read on the wrong rail at the checker's k + 0.68 T
        that the compiler's own cards accepted at k + 0.7 T. One constant, imported by both."""
        from sram_compiler.testbenches.sram_6t_core_MC_testbench import ACCESS_DEADLINE
        from tests.spice import phased_waveforms

        self.assertEqual(ACCESS_DEADLINE, .68)
        self.assertIs(phased_waveforms.ACCESS_DEADLINE, ACCESS_DEADLINE)
        self.assertNotIn('.68', inspect_source(phased_waveforms.score))


class ScreenManifestTests(unittest.TestCase):
    """The case list is a tracked artifact: it must stay runnable and keep its coverage."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent / 'spice'
        cls.cases = json.loads((root / 'v223_cases.json').read_text())
        cls.negative = json.loads((root / 'v223_negative_cases.json').read_text())

    def periods(self, case):
        import contextlib
        import io

        from sram_compiler.per_device_mc.run import load_config
        from sram_compiler.sizing import resolve_driver_sizes, resolve_timing
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = load_config(case.get('rows', 8), case.get('cols', 4), case.get('corner', 'SS'))
            sizes = resolve_driver_sizes(cfg, cell_type=case.get('cell', 'SRAM_6T_CELL'),
                                         mux=case.get('mux', False))
            return resolve_timing(cfg, sizes).t_period

    def test_every_case_names_one_directory_and_resolves_inside_the_envelope(self):
        """The runner makes a directory per case and refuses to reuse one, and a case it
        cannot build wastes whatever the queue already spent."""
        names = [case['name'] for case in self.cases + self.negative]
        self.assertEqual(len(names), len(set(names)))
        for case in self.cases + self.negative:
            with self.subTest(case=case['name']):
                self.assertEqual(Path(case['name']).name, case['name'])
                self.assertIn(case.get('operation', 'read&write'), ('read', 'write', 'read&write'))
                self.assertGreater(self.periods(case), 0)

    def test_arbitrary_patterns_match_the_array_they_run_on(self):
        """apply_pattern rejects a bad pattern only once the deck is already built."""
        for case in self.cases:
            pattern = case.get('pattern')
            if pattern is None:
                continue
            rows, cols = case.get('rows', 8), case.get('cols', 4)
            with self.subTest(case=case['name']):
                self.assertEqual(len(pattern), 8)
                for entry in pattern:
                    self.assertIn(entry['op'], ('read', 'write', 'idle'))
                    self.assertTrue(0 <= entry['row'] < rows)
                    self.assertEqual(len(entry['data']), cols)
                    self.assertFalse(set(entry['data']) - {'0', '1'})

    def test_the_manifest_covers_the_gaps_the_V2_2_2_screen_left(self):
        """Each assertion here is a coverage hole the V2.2.2 record named. They are properties
        of the manifest, so deleting a case to save runtime has to be a deliberate act."""
        def size(case):
            return case.get('rows', 8), case.get('cols', 4)

        # Arrays large in both dimensions: V2.2.2 had none above 32x32.
        joint = [case for case in self.cases if min(size(case)) >= 64]
        self.assertTrue({(256, 256), (128, 128)} <= {size(case) for case in joint})
        self.assertTrue(any(case.get('variation') == 'per-device' for case in joint))

        # Process corner separated from voltage and temperature.
        points = {}
        for case in self.cases:
            points.setdefault(case.get('corner', 'SS'), set()).add(
                (case.get('vdd', .9), case.get('temperature', 125)))
        for corner in ('SS', 'SF', 'FF', 'FS'):
            self.assertGreaterEqual(len(points[corner]), 2, corner)

        # The nominal corner above the 8x4 toy array.
        self.assertTrue(any(case.get('corner') == 'TT' and size(case) != (8, 4)
                            for case in self.cases))

        # Complementary column data above eight columns.
        self.assertTrue(any(case.get('cols', 4) >= 64 and '01' * 2 in entry['data']
                            for case in self.cases for entry in case.get('pattern', ())))

        # More than one mismatch draw at the tall and wide arrays.
        draws = {}
        for case in self.cases:
            if case.get('variation') == 'per-device':
                draws[size(case)] = draws.get(size(case), 0) + 1
        for shape in ((256, 4), (512, 4)):
            self.assertGreaterEqual(draws.get(shape, 0), 2, shape)

    def test_every_negative_control_is_too_short_to_pass(self):
        """A negative control that the contract would accept proves nothing. V2.2.2 had one,
        at one size, and eight of its twelve reported margins never failed anywhere."""
        self.assertGreater(len(self.negative), 1)
        shapes = set()
        for case in self.negative:
            with self.subTest(case=case['name']):
                self.assertIn('period', case)
                self.assertLess(case['period'], self.periods(case))
                shapes.add((case.get('rows', 8), case.get('cols', 4)))
        self.assertGreaterEqual(len(shapes), 4)


def inspect_source(function):
    import inspect
    return inspect.getsource(function)


if __name__ == '__main__':
    unittest.main()
