"""Bounded full-local diagnostic screen and rule-by-rule evidence (no promotion).

This deliberately uses a small ensemble to find failures before the full
qualification campaign. Passing this screen is not tail or yield qualification.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from .campaign import architectures, verification_cases
from .qualification import run_case
from .timing import provisional_period, timing_from_measurements


RULE_CHECKS = {
    'write_output': ('cell_written', 'write_bitline_low', 'write_overlap', 'write_budget'),
    'write_input_and_hold': ('cell_retained',),
    'precharge': ('all_bitlines_restored', 'all_bitlines_equalized', 'restore_budget'),
    'wordline_inverter_and_nand': ('wl_reaches_vdd', 'wordline_budget'),
    'decoder_and_address': ('decoder_budget', 'unselected_wordlines_quiet', 'neighbor_retained'),
    'control_buffers': ('wl_off_before_precharge', 'no_premature_sense',
                        'wl_en_edge_budget', 'wl_en_fall_budget', 'pre_edge_budget',
                        'pre_fall_budget', 'sa_iso_edge_budget', 'sa_iso_fall_budget',
                        's_en_edge_budget', 's_en_fall_budget', 'w_en_edge_budget', 'w_en_fall_budget'),
    'replica_and_sensing': ('sense_differential', 'output_settled', 'read_no_disturb'),
}


def summarize(records):
    """Report observed checks, including gaps; never infer a failing component."""
    families = {name: {'checks_observed': 0, 'checks_failed': 0, 'failures': []}
                for name in RULE_CHECKS}
    samples, failures, execution_errors = 0, [], []
    for entry in records:
        for result in entry.get('verification', []):
            case = result['case']
            if result.get('error'):
                execution_errors.append({'case': case, 'error': result['error']})
            for index, sample in enumerate(result.get('samples', [])):
                samples += 1
                if sample.get('error'):
                    execution_errors.append({'case': case, 'sample': index, 'error': sample['error']})
                failed = [key for key, value in sample.get('checks', {}).items() if not value]
                if failed:
                    failures.append({'case': case, 'sample': index, 'failed_checks': failed,
                                     'metrics': sample.get('metrics', {}), 'directory': result['directory']})
                for family, names in RULE_CHECKS.items():
                    if family.startswith('write_') and case['operation'] != 'write':
                        continue
                    group = families[family]
                    for check, passed in sample.get('checks', {}).items():
                        matches = check in names or (family == 'decoder_and_address' and check.startswith('addr_'))
                        if not matches:
                            continue
                        group['checks_observed'] += 1
                        group['checks_failed'] += not passed
                        if not passed:
                            group['failures'].append({'directory': result['directory'], 'sample': index, 'check': check})
    for group in families.values():
        group['status'] = ('failed observed checks' if group['checks_failed'] else
                           'passes observed checks' if group['checks_observed'] else 'not assessed')
    return {'qualified': False, 'samples_observed': samples, 'rule_families': families,
            'waveform_failures': failures, 'execution_errors': execution_errors,
            'limitations': ['Small diagnostic ensemble; no tail/yield qualification',
                           'Whole-path checks do not isolate an individual driver as the cause',
                           'Separate half-select topology and held-out coefficient validation remain required',
                           'Write-input and hold checks do not establish an independent input-stage sizing floor']}


def _atomic_json(path, value):
    # Live --summarize calls may coincide with the running coordinator.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.stem + '.',
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def review_architecture(base, root, xyce, samples, timeout):
    name = f'{base.cell}_{base.rows}x{base.cols}_mux{int(base.mux)}_rc{int(base.w_rc)}'
    checkpoint = root / (name + '.json')
    record = {'base': asdict(base), 'complete': False, 'calibration': [], 'verification': []}
    period = provisional_period(base.rows, base.cols, base.cell) * (4 if base.w_rc else 1)
    for operation, corner in [('read', 'SS'), ('write', 'SS'), ('write', 'SF')]:
        case = replace(base, operation=operation, corner=corner, variation='nominal', samples=1, period=period)
        result = run_case(case, root / 'cases', xyce, timeout=timeout)
        record['calibration'].append(result)
        _atomic_json(checkpoint, record)
        print(f'calibration {case.name}: {result["passed"]}', flush=True)
    calibration = record['calibration']
    if not all(point['passed'] for point in calibration):
        # A timeout or simulator exit is not waveform evidence; keep them apart.
        errors = [point['error'] for point in calibration if point.get('error')]
        record['calibration_error'] = ('Nominal calibration execution error: ' + '; '.join(errors)
                                       if errors else 'Nominal calibration failed waveform acceptance')
        record['complete'] = True
        _atomic_json(checkpoint, record)
        return record
    timing = timing_from_measurements(calibration[0]['measures'][0],
                                     [point['measures'][0] for point in calibration[1:]])
    record['timing'] = timing.to_dict()
    # Independent diagnostic seeds; never recalibrate for a random sample.
    cases = [replace(case, samples=samples, seed=82026) for case in
             verification_cases(base, timing, full=True, tails=False)]
    record['expected_cases'] = len(cases)
    for case in cases:
        result = run_case(case, root / 'cases', xyce, timing=timing, timeout=timeout)
        record['verification'].append(result)
        _atomic_json(checkpoint, record)
        print(f'local {case.name}: {result["passed"]}', flush=True)
    record['complete'] = True
    _atomic_json(checkpoint, record)
    return record


def run_review(bases, root, xyce, samples, workers, timeout):
    with (root / 'review.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Review already running in {root}') from exc
        _atomic_json(root / 'schedule.json', {'samples_per_case': samples,
                     'workers': workers, 'bases': [asdict(base) for base in bases],
                     'purpose': 'diagnostic screen only', 'seed': 82026})
        records = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(review_architecture, base, root, xyce, samples, timeout)
                       for base in bases]
            for future in as_completed(futures):
                records.append(future.result())
                _atomic_json(root / 'review.json', summarize(records))
        return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='8x4,16x16,64x64,256x8,16x256')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--mpi-ranks', type=int, default=4, help='Cores per large-array Xyce process')
    parser.add_argument('--parallel-min-cells', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=900, help='Seconds per diagnostic sample')
    parser.add_argument('--xyce', default='Xyce')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'outputs/qualification/V2.0.5/review')
    parser.add_argument('--summarize', action='store_true', help='Read completed and partial checkpoints without running simulations')
    parser.add_argument('--rc-only', action='store_true', help='Only the explicit-RC sensitivity architectures (16x16)')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        records = [json.loads(path.read_text()) for path in root.glob('SRAM_*.json')]
    else:
        if min(args.samples, args.workers, args.timeout, args.mpi_ranks, args.parallel_min_cells) <= 0:
            parser.error('Samples, workers and timeout must be positive')
        xyce = shutil.which(args.xyce)
        if xyce is None:
            parser.error('Xyce executable not found')
        sizes = [tuple(map(int, size.lower().split('x'))) for size in args.sizes.split(',')]
        if any(len(size) != 2 or min(size) <= 0 for size in sizes):
            parser.error('Sizes must be positive ROWSxCOLS pairs')
        bases = architectures(sizes, full=False)
        # A small RC sensitivity case first; tall/large RC is in full qualification.
        bases += [replace(base, w_rc=True, parasitic_factor=2) for base in list(bases)
                  if (base.rows, base.cols) == (16, 16)]
        if args.rc_only:
            bases = [base for base in bases if base.w_rc]
            if not bases:
                parser.error('--rc-only needs 16x16 in --sizes')
        bases = [replace(base, mpi_ranks=args.mpi_ranks if base.rows * base.cols >= args.parallel_min_cells else 1)
                 for base in bases]
        import os
        cores = len(os.sched_getaffinity(0))
        if args.workers * args.mpi_ranks > cores:
            parser.error(f'workers × MPI ranks must not exceed the {cores} available cores')
        records = run_review(bases, root, xyce, args.samples, args.workers, args.timeout)
    report = summarize(records)
    report['architectures_observed'] = len(records)
    report['architectures_complete'] = sum(record.get('complete', False) for record in records)
    report['calibration_errors'] = [{'base': record['base'], 'error': record['calibration_error']}
                                     for record in records if record.get('calibration_error')]
    _atomic_json(root / 'review.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in
                     ('rule_families', 'waveform_failures', 'execution_errors')}, indent=2))
    return int(bool(report['waveform_failures'] or report['execution_errors'] or report['calibration_errors']))


if __name__ == '__main__':
    raise SystemExit(main())
