"""Export V2.0.5 qualification evidence and promote only complete passing records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sram_compiler.sizing.table import (
    current_scoring_version,
    physical_context,
    record_key,
)


def mismatch_evidence(campaign, root):
    offset = []
    errors = []
    for name in ('offset_SS_125_0.9', 'offset_FF_-40_1.0',
                 'offset_SS_125_0.9_rail', 'offset_FF_-40_1.0_rail'):
        path = root / name / 'offset_summary.json'
        if not path.exists():
            errors.append(f'Missing {name}')
            continue
        record = json.loads(path.read_text())
        expected_offset_version = hashlib.sha256(Path(__file__).with_name('offset.py').read_bytes()).hexdigest()
        if record.get('scoring_version') != expected_offset_version:
            errors.append(f'Stale offset scoring: {name}; rerun the cached offset command')
        offset.append(record)
        if not record['passed'] or record['samples'] != 100 or len(record['offsets']) != 100:
            errors.append(f'Incomplete offset ensemble: {name}')
    offset_bound = max((record['conservative_3sigma_offset'] for record in offset
                        if record.get('conservative_3sigma_offset') is not None), default=None)
    groups, write_groups = {}, {}
    for entry in campaign['arrays'].values():
        for case in entry['verification']:
            cfg = case['case']
            if cfg['variation'] != 'per-device':
                continue
            # Keep architecture, operation and ensemble purpose separate. The
            # broad local checks must not inflate the dedicated tail ensembles.
            if cfg.get('mux') or cfg.get('w_rc') or cfg.get('real_cell_mode', 0) != 0:
                continue
            key = (cfg['cell'], cfg['rows'], cfg['cols'], cfg['corner'])
            seed = cfg.get('seed', -1)
            if (cfg.get('operation') == 'write' and cfg.get('cell_variant') == 'write_box'
                    and 4026 <= seed < 4036):
                group = write_groups.setdefault(key, {'samples': 0, 'seeds': set(), 'passed': True})
                group['samples'] += len(case.get('samples', []))
                group['passed'] &= (case['passed'] and seed not in group['seeds']
                                    and len(case.get('samples', [])) == 10
                                    and all(s.get('checks', {}).get('cell_written') is True
                                            and s.get('checks', {}).get('cell_retained') is True
                                            for s in case.get('samples', [])))
                group['seeds'].add(seed)
            if not (cfg.get('operation') == 'read' and cfg.get('cell_variant') == 'read_box'
                    and 3026 <= seed < 3036):
                continue
            group = groups.setdefault(key, {'dv': [], 'skew': [], 'seeds': set(), 'all_waveforms_pass': True})
            group['all_waveforms_pass'] &= seed not in group['seeds'] and len(case.get('samples', [])) == 10
            group['seeds'].add(seed)
            group['all_waveforms_pass'] &= case['passed']
            for sample in case.get('samples', []):
                metrics = sample.get('metrics', {})
                if 'dv_at_sen' in metrics:
                    group['dv'].append(metrics['dv_at_sen'])
                    group['skew'].append(metrics['rwl_minus_wl'])
    summaries, thresholds = [], {}
    for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
        thresholds[cell] = 0.3
        for rows, cols in ((16, 16), (256, 8)):
            for corner in ('SS', 'FF'):
                key = (cell, rows, cols, corner)
                group = groups.get(key)
                if group is None or len(group['dv']) != 100 or len(group['seeds']) != 10 or offset_bound is None:
                    errors.append(f'Missing 100-sample local ensemble: {key}')
                    continue
                mean, sigma = float(np.mean(group['dv'])), float(np.std(group['dv'], ddof=1))
                threshold = offset_bound + 3 * sigma + .1
                passed = group['all_waveforms_pass'] and mean >= threshold
                summaries.append({'cell': cell, 'rows': rows, 'cols': cols, 'corner': corner,
                                  'samples': 100, 'dv_mean': mean, 'dv_sigma': sigma,
                                  'rwl_skew_mean': float(np.mean(group['skew'])),
                                  'rwl_skew_sigma': float(np.std(group['skew'], ddof=1)),
                                  'required_nominal_dv': threshold, 'margin': mean - threshold,
                                  'passed': passed})
                thresholds[cell] = max(thresholds[cell], threshold)
                if not passed:
                    errors.append(f'Local sensing margin/waveform failure: {key}')
    write_summaries = []
    for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
        for rows, cols in ((8, 4), (16, 16), (256, 8)):
            for corner in ('SS', 'SF'):
                key = (cell, rows, cols, corner)
                group = write_groups.get(key)
                passed = bool(group and group['samples'] == 100 and len(group['seeds']) == 10 and group['passed'])
                write_summaries.append({'cell': cell, 'rows': rows, 'cols': cols,
                                        'corner': corner, 'passed': passed,
                                        'samples': group['samples'] if group else 0})
                if not passed:
                    errors.append(f'Missing/failing 100-sample local write ensemble: {key}')
    return {'offset': offset, 'local': summaries, 'local_write': write_summaries,
            'per_sample_dV_min': max(.3, offset_bound + .1) if offset_bound is not None else .3,
            'dV_min': thresholds,
            'passed': not errors, 'errors': errors}


def export_report(output_root, destination, table_path=None):
    root, destination = Path(output_root).resolve(), Path(destination).resolve()
    campaign = json.loads((root / 'campaign.json').read_text())
    if not campaign.get('complete'):
        raise ValueError('Cannot publish an incomplete campaign')
    destination.mkdir(parents=True, exist_ok=True)
    mismatch = mismatch_evidence(campaign, root)
    if not campaign.get('full_matrix') or campaign.get('variation_policy') != 'full-local-v1':
        mismatch['passed'] = False
        mismatch['errors'].append('Full local read/write campaign policy is required for promotion')
    if campaign.get('half_select_qualified') is not True:
        mismatch['passed'] = False
        mismatch['errors'].append('Separate half-select waveform qualification is still required')
    (destination / 'V2.0.5_mismatch.json').write_text(json.dumps(mismatch, indent=2, allow_nan=False))
    rows, failures, records, array_summary = [], [], {}, []
    for name, entry in campaign['arrays'].items():
        array_pass = bool(entry.get('qualified')) and mismatch['passed']
        baseline_key = entry['calibration'].get('read_SS', {}).get('sizes_key')
        baseline_sensing_pass = True
        for stage, cases in [('calibration', list(entry['calibration'].values())), ('verification', entry['verification'])]:
            for case in sorted(cases, key=lambda item: json.dumps(item['case'], sort_keys=True)):
                cfg, samples = case['case'], case.get('samples', [])
                artifact = Path(case['directory']) / 'result.json' if case.get('directory') else None
                metadata = json.loads(artifact.read_text()) if artifact is not None and artifact.exists() else {}
                if case.get('scoring_version') != current_scoring_version():
                    raise ValueError(f'Stale acceptance results for {name}; rerun the cached campaign')
                if metadata and any(metadata.get(key) != case.get(key) for key in
                                    ('signature', 'passed', 'samples', 'measures', 'scoring_version',
                                     'max_time_step', 'original_attempt', 'simulator_returncode', 'execution')):
                    raise ValueError(f'Case evidence changed after the campaign recorded it: {name}')
                if case.get('sizes_key') != metadata.get('sizing', {}).get('key'):
                    raise ValueError(f'Case sizing identity changed after qualification: {name}')
                if stage == 'verification' and not case.get('error'):
                    if (cfg['variation'] != 'per-device' or
                            not metadata.get('variation', {}).get('full_device_coverage') or
                            not metadata.get('simulator_sha256')):
                        array_pass = False
                    if case.get('sizes_key') != baseline_key:
                        raise ValueError(f'Candidate used different periphery from calibration: {name}')
                    if cfg['period'] != entry['timing']['t_period']:
                        raise ValueError(f'Candidate was not tested at the frozen period: {name}')
                failed_checks = sorted({key for sample in samples for key, value in sample.get('checks', {}).items() if not value})
                errors = [sample['error'] for sample in samples if 'error' in sample]
                dv = [sample['metrics'][key] for sample in samples
                      for key in ('dv_at_sen', 'min_sequence_dv') if key in sample.get('metrics', {})]
                if dv:
                    threshold = (mismatch.get('per_sample_dV_min', .3) if cfg['variation'] == 'per-device'
                                 else mismatch['dV_min'][cfg['cell']])
                    baseline_sensing_pass &= min(dv) >= threshold
                measure_key = 'TWRITE_TOTAL' if cfg['operation'] == 'write' else 'TREAD_TOTAL'
                delays = [row[measure_key] for row in case.get('measures', []) if isinstance(row.get(measure_key), (int, float))]
                yaml_pass = [sample['metrics']['yaml_access_spec_pass'] for sample in samples
                             if 'yaml_access_spec_pass' in sample.get('metrics', {})]
                rows.append({**cfg, 'stage': stage, 'samples_observed': len(samples),
                             'baseline_key': metadata.get('sizing', {}).get('key', ''),
                             'deck_signature': metadata.get('signature', ''),
                             'linear_solver': metadata.get('linear_solver', 'Xyce default'),
                             'mpi_ranks': metadata.get('execution', {}).get('mpi_ranks', 1),
                             'sampling_backend': metadata.get('execution', {}).get('sampling_backend', 'Xyce native'),
                             'max_time_step_ps': metadata.get('max_time_step', 10e-12) * 1e12,
                             'retried_incomplete_transient': bool(metadata.get('original_attempt')),
                             'samples_passed': sum(sample['passed'] for sample in samples),
                             'passed': case['passed'], 'expected_rejection': case.get('expected_rejection', False),
                             'failed_checks': ';'.join(failed_checks),
                             'error': ';'.join([case.get('error', ''), *errors]).strip(';'),
                             'worst_access_ps': max(delays) * 1e12 if delays else '',
                             'minimum_sensing_dv': min(dv) if dv else '',
                             'yaml_access_specs_pass': all(yaml_pass) if yaml_pass else ''})
                if not case['passed'] and not case.get('expected_rejection'):
                    failures.append({'array': name, 'case': cfg, 'failed_checks': failed_checks,
                                     'error': case.get('error'), 'sample_errors': errors})
        array_pass &= baseline_sensing_pass
        array_summary.append({'array': name, 'qualified': array_pass,
                              'phase_waveforms_pass': bool(entry.get('qualified')),
                              'measured_sensing_margin_pass': baseline_sensing_pass,
                              'calibration_error': entry.get('calibration_error'),
                              'period_ns': entry.get('timing', {}).get('t_period', 0) * 1e9})
        if array_pass:
            point = entry['calibration']['read_SS']
            result = json.loads((Path(point['directory']) / 'result.json').read_text())
            sizes = result['sizing']
            base = entry['base']
            context = physical_context(base['w_rc'], real_cell_mode=base['real_cell_mode'])
            records[record_key(sizes['key'], context)] = {
                'sizes_key': sizes['key'], 'physical': context, 'driver_sizes': sizes,
                'scoring_version': current_scoring_version(),
                'qualified': True, 'campaign_complete': True, 'local_mismatch_qualified': True,
                'variation_policy': 'full-local-v1', 'local_relative_sigma': .05,
                'timing': entry['timing'], 'dV_min': mismatch['dV_min'][base['cell']],
                'scope': 'Fixed global corners with local mismatch throughout read/write paths',
                'yaml_access_specs_are_separate': True,
            }
    if rows:
        with (destination / 'V2.0.5_cases.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (destination / 'V2.0.5_failures.json').write_text(json.dumps(failures, indent=2))
    if table_path is not None:
        Path(table_path).write_text(json.dumps({'schema': 1, 'version': 'V2.0.5', 'records': records}, indent=2, allow_nan=False))
    qualified = sum(item['qualified'] for item in array_summary)
    failed_yaml = sum(row['yaml_access_specs_pass'] is False for row in rows)
    numerical_retries = sum(row['retried_incomplete_transient'] for row in rows)
    report = {'version': 'V2.0.5', 'complete': True, 'arrays': array_summary,
              'qualified_arrays': qualified, 'array_count': len(array_summary),
              'cases': len(rows), 'samples': sum(row['samples_observed'] for row in rows),
              'failing_cases': len(failures), 'yaml_access_failing_cases': failed_yaml,
              'numerical_retry_cases': numerical_retries,
              'mismatch_pass': mismatch['passed'], 'mismatch_errors': mismatch['errors'],
              'passed': qualified == len(array_summary) and mismatch['passed']}
    (destination / 'V2.0.5_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ['# V2.0.5 qualification report', '',
             f"Status: {'PASS' if report['passed'] else 'FAIL'} — {qualified}/{len(array_summary)} physical configurations qualified.",
             f"Completed {len(rows)} cases and {report['samples']} waveform samples; {len(failures)} cases failed.", '',
             f'{numerical_retries} cases required a retry of the same seeded ensemble at a tighter 5 ps',
             'maximum timestep after Xyce stopped an individual transient early. Original attempts',
             'are retained beside their retries. Electrical and nonfinite-signal failures are not',
             'eligible for this retry; the case ledger records each accepted run\'s actual timestep.', '',
             'The original 200 ps read / 100 ps write access constraints remain separate from the',
             f"phase-based qualification: {failed_yaml} cases violate those original limits. The default",
             'K=1/N=9 replica preserves sensing margin and does not claim compliance with the read limit.', '',
             '## Reproduce', '', '```bash',
             'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m sram_compiler.sizing.campaign --xyce /path/to/Xyce --workers 48',
             'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m sram_compiler.sizing.offset --xyce /path/to/Xyce',
             'python3 -m sram_compiler.sizing.report outputs/qualification/V2.0.5 --table sram_compiler/sizing/sizing_table.json',
             '```', '',
             'Inputs: checked-in circuit YAMLs and FreePDK45 model files. The campaign uses full',
             'transistor arrays at fixed global corners. Every verification sample varies each',
             'MOS independently. Large-array MPI runs materialize local LHS draws into saved',
             'numeric model cards before each transient. Seeds, geometry, settings, pass counts,',
             'failed checks and worst measurements are in [the case ledger](V2.0.5_cases.csv).',
             'Raw decks, waveform files and per-sample checks stay in `outputs/qualification/V2.0.5/`.', '',
             'Runs explicitly use KLU and record the MPI rank count and sampler identity.',
             'The [Xyce solver guidance](https://xyce.sandia.gov/documentation-tutorials/frequently-asked-questions/)',
             'describes parallel device evaluation with KLU. Native MPI random expressions',
             'crashed on this Xyce 7.4 build; saved numeric local draws bypass that path.', '',
             '## Sensing qualification', '',
             'The SA offset ensemble keeps each of 100 independent SA/latch copies fixed over',
             '25 differential trials, at both 0.75 VDD common mode and a precharged-bitline profile',
             'with one input held at VDD. Its bound includes absolute mean, 3 sigma and half the 25 mV',
             'grid step. Per-device replica/cell ensembles contribute 3 sigma of differential variation;',
             'the remaining explicit reserve is 100 mV. Global/corner replica alignment uses ±10 ps;',
             'local skew is measured and evaluated through the sensing margin and settling checks.',
             'See [the mismatch evidence](V2.0.5_mismatch.json).', '',
             '## Configuration results', '', '| Configuration | Period (ns) | Qualified |', '|---|---:|---|']
    lines += [f"| {item['array']} | {item['period_ns']:.2f} | {item['qualified']} |" for item in array_summary]
    lines += ['', 'Failed waveform checks and execution errors are listed in',
              '[the failure ledger](V2.0.5_failures.json). No incomplete or failed record is promoted',
              'to the sizing table. Calibration and MC periods are frozen before candidate evaluation.',
              '', 'This is predictive-model qualification, not post-layout extraction or silicon validation.']
    (destination / 'V2.0.5_report.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--destination', type=Path, default=Path('docs/qualification'))
    parser.add_argument('--table', type=Path)
    args = parser.parse_args()
    result = export_report(args.output_root, args.destination, args.table)
    print(json.dumps({key: value for key, value in result.items() if key != 'arrays'}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
