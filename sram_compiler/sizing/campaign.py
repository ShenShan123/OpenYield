"""Run the V2.0.5 calibration and qualification matrix without changing YAML.

    python3 -m sram_compiler.sizing.campaign --xyce /path/to/Xyce --workers 24

Use --sizes 8x4,16x16 for a pilot; omit it for all 27 historical array sizes.
Existing identical simulations are reused, but their waveforms are rescored.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from sram_compiler.sizing.qualification import Case, run_case
from sram_compiler.sizing.timing import provisional_period, timing_from_measurements

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def historical_sizes():
    with (PROJECT_ROOT / 'TIMING_AUTOCONFIG_data.csv').open() as stream:
        return sorted({(int(row['rows']), int(row['cols'])) for row in csv.DictReader(stream)
                       if row['source'] == 'v202_sweep'})


def architectures(sizes, full=True):
    result = []
    for cell in ('SRAM_6T_CELL', 'SRAM_10T_CELL'):
        for rows, cols in sizes:
            for mux in ([False, True] if cols % 2 == 0 else [False]):
                result.append(Case(rows, cols, cell=cell, mux=mux))
    if full:
        for case in list(result):
            if (case.rows, case.cols) in ((16, 16), (64, 64), (256, 8)):
                result.append(replace(case, w_rc=True, parasitic_factor=2.0))
    return result


def verification_cases(base, timing, full=True, tails=True):
    base = replace(base, period=timing.t_period)
    hazard = base.rows >= 256
    base = replace(base, variation='per-device', samples=10)
    cases = [replace(base, operation='read', next_row=hazard)]
    cases += [replace(base, operation='write', corner=corner, cell_variant='write_box',
                      next_row=hazard) for corner in ('SS', 'SF')]
    if full:
        cases += [replace(base, corner=corner, temperature=temperature, vdd=1.0,
                          operation=operation, next_row=hazard) for corner, temperature in
                  [('TT', 25), ('FF', 125), ('FS', 125), ('FF', -40)]
                  for operation in ('read', 'write')]
        cases += [replace(base, operation='read', corner='SF', next_row=hazard)]
        cases += [replace(base, cell_variant='read_box')]
        if (base.rows, base.cols) in ((8, 4), (16, 16), (256, 8)) and not base.w_rc:
            cases += [replace(base, operation='read&write', corner=corner)
                      for corner in ('SS', 'SF')]
        # 100 local samples in ten independent, seeded ten-sample LHS batches.
        # Full transistor arrays: target cell, replica, SA and remaining devices
        # all receive independent model cards (a conservative superset).
        if tails and (base.rows, base.cols) in ((16, 16), (256, 8)) and not base.w_rc and not base.mux:
            for corner, temperature, vdd in [('SS', 125, 0.9), ('FF', -40, 1.0)]:
                cases += [replace(base, corner=corner, temperature=temperature, vdd=vdd,
                                  cell_variant='read_box', variation='per-device', samples=10,
                                  seed=3026 + batch) for batch in range(10)]
        if tails and (base.rows, base.cols) in ((8, 4), (16, 16), (256, 8)) and not base.w_rc and not base.mux:
            for corner in ('SS', 'SF'):
                cases += [replace(base, operation='write', corner=corner,
                                  cell_variant='write_box', samples=10,
                                  seed=4026 + batch) for batch in range(10)]
        if base.rows == 32 and base.cols == 1 and not base.w_rc:
            cases.append(replace(base, operation='write', corner='SF', cell_variant='unwritable'))
    return cases


def _array_key(case):
    return f'{case.cell}_{case.rows}x{case.cols}_mux{int(case.mux)}_rc{int(case.w_rc)}'


def campaign_bases(sizes, full):
    bases = architectures(sizes, full)
    if full and (32, 1) not in sizes:
        bases.extend(architectures([(32, 1)], full=False))
    return bases


def campaign_plan(sizes, full=True, tails=True):
    """Count the work without generating decks or launching a simulator."""
    from sram_compiler.sizing.timing import TimingConfig
    bases = campaign_bases(sizes, full)
    cases = [case for base in bases for case in
             verification_cases(base, TimingConfig(5e-9, 0, 0, 0), full, tails)]
    return {'physical_configurations': len(bases), 'calibration_decks': 3 * len(bases),
            'verification_decks': len(cases),
            'waveform_samples': 3 * len(bases) + sum(case.samples for case in cases),
            'verification_variation': 'per-device', 'full_matrix': full and tails,
            'operations': {op: sum(case.operation == op for case in cases)
                           for op in ('read', 'write', 'read&write')}}


def run_campaign(sizes, output_root, xyce, workers=24, full=True, timeout=21600, tails=True,
                 mpi_ranks=4, parallel_min_cells=1024):
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'campaign.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Campaign already running in {root}') from exc
        return _run_campaign(sizes, root, xyce, workers, full, timeout, tails,
                             mpi_ranks, parallel_min_cells)


def _run_campaign(sizes, output_root, xyce, workers, full, timeout, tails, mpi_ranks, parallel_min_cells):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bases = [replace(base, mpi_ranks=mpi_ranks if base.rows * base.cols >= parallel_min_cells else 1)
             for base in campaign_bases(sizes, full)]
    state: dict[str, dict[str, Any]] = {_array_key(base): {'base': asdict(base), 'calibration': {}, 'verification': []} for base in bases}
    futures, completed, started = {}, 0, time.monotonic()
    pending = deque()
    priority = deque()

    def snapshot(done=False):
        report = {'version': 'V2.0.5', 'variation_policy': 'full-local-v1',
                      'complete': done, 'full_matrix': full and tails,
                      'plan': campaign_plan(sizes, full, tails),
                      'historical_size_count': len(sizes), 'array_count': len(bases), 'completed_cases': completed,
                      'elapsed_seconds': time.monotonic() - started, 'arrays': state}
        checkpoint = output_root / 'campaign.json.tmp'
        checkpoint.write_text(json.dumps(report, indent=2, allow_nan=False))
        checkpoint.replace(output_root / 'campaign.json')

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for base in bases:
            period = provisional_period(base.rows, base.cols, base.cell)
            if base.w_rc:
                # The historical phase fit excludes explicit terminal RC. Use
                # a longer calibration clock, then derive the real period from
                # the measured RC phases; never reuse the no-RC timing fit.
                period *= 4
            for operation, corner in [('read', 'SS'), ('write', 'SS'), ('write', 'SF')]:
                case = replace(base, operation=operation, corner=corner, period=period,
                               variation='nominal', samples=1)
                pending.append((base, 'calibration', case, None))
        snapshot()
        while futures or pending or priority:
            # Keep the executor queue bounded. Once an array is calibrated,
            # exercise its primary MC/TT checks before queuing more large
            # calibration jobs, so common failures are exposed promptly.
            while len(futures) < workers and (priority or pending):
                base, stage, case, timing = (priority if priority else pending).popleft()
                task = pool.submit(run_case, case, output_root / 'cases', xyce,
                                   timing=timing, timeout=timeout)
                futures[task] = (base, stage, case)
            ready, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in ready:
                base, stage, case = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 -- retain failed worker evidence at the process boundary
                    result = {'passed': False, 'error': f'Worker failure: {exc}', 'case': asdict(case)}
                completed += 1
                entry = state[_array_key(base)]
                brief = {key: result[key] for key in
                         ('case', 'passed', 'error', 'directory', 'elapsed_seconds', 'samples', 'measures',
                          'signature', 'scoring_version', 'linear_solver',
                          'max_time_step', 'original_attempt', 'simulator_returncode',
                          'variation', 'simulator_sha256') if key in result}
                brief['execution'] = result.get('execution', {})
                brief['sizes_key'] = result.get('sizing', {}).get('key')
                if stage == 'calibration':
                    entry['calibration'][f'{case.operation}_{case.corner}'] = brief
                    if len(entry['calibration']) == 3:
                        try:
                            cal = entry['calibration']
                            # Calibration must have functional waveforms, not only
                            # positive scalar measures, before deriving a period.
                            for point in cal.values():
                                if point.get('error') or not point.get('samples'):
                                    raise ValueError('Calibration execution/waveform error')
                                sample = point['samples'][0]
                                for key in ('cell_retained', 'cell_written', 'output_settled', 'measures_valid'):
                                    if key in sample.get('checks', {}) and not sample['checks'][key]:
                                        raise ValueError(f'Calibration functional failure: {key}')
                                if sample.get('error'):
                                    raise ValueError(sample['error'])
                            timing = timing_from_measurements(cal['read_SS']['measures'][0],
                                                              [cal['write_SS']['measures'][0], cal['write_SF']['measures'][0]])
                            entry['timing'] = timing.to_dict()
                            cases = verification_cases(base, timing, full, tails)
                            entry['expected_verification_cases'] = len(cases)
                            for index, verify_case in enumerate(cases):
                                (priority if index < 5 else pending).append((base, 'verification', verify_case, timing))
                        except (ValueError, KeyError, TypeError) as exc:
                            entry['calibration_error'] = str(exc)
                else:
                    if case.cell_variant == 'unwritable':
                        brief['expected_rejection'] = not result['passed'] and not result.get('error') and any(
                            sample.get('checks', {}).get('cell_written') is False for sample in result.get('samples', []))
                    entry['verification'].append(brief)
                print(f'{completed}: {stage} {case.name} {"PASS" if result["passed"] else "FAIL"}', flush=True)
            snapshot()
    for entry in state.values():
        checks = entry['verification']
        entry['screening_passed'] = (not entry.get('calibration_error') and all(c['passed'] for c in entry['calibration'].values())
                              and len(checks) == entry.get('expected_verification_cases', -1)
                              and all(case.get('expected_rejection', case['passed']) for case in checks))
        entry['qualified'] = full and tails and entry['screening_passed']
    snapshot(done=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', help='Comma-separated ROWSxCOLS; omit for the full historical matrix')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'outputs/qualification/V2.0.5')
    parser.add_argument('--xyce', default='Xyce')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--mpi-ranks', type=int, default=4, help='Cores per large-array Xyce process')
    parser.add_argument('--parallel-min-cells', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=21600,
                        help='Maximum runtime in seconds per sample (default: six hours)')
    parser.add_argument('--pilot', action='store_true', help='Only the three primary verification decks per array')
    parser.add_argument('--screen', action='store_true', help='All corner/sequence/RC checks, without the 100-sample tail ensembles')
    parser.add_argument('--dry-run', action='store_true', help='Print the case/sample schedule without running Xyce')
    args = parser.parse_args()
    if args.pilot and args.screen:
        parser.error('Choose either --pilot or --screen')
    if min(args.workers, args.timeout, args.mpi_ranks, args.parallel_min_cells) <= 0:
        parser.error('Workers and timeout must be positive')
    sizes = historical_sizes() if not args.sizes else [tuple(map(int, value.lower().split('x'))) for value in args.sizes.split(',')]
    if any(len(size) != 2 or min(size) < 1 for size in sizes):
        parser.error('Sizes must be positive ROWSxCOLS pairs')
    if args.dry_run:
        print(json.dumps(campaign_plan(sizes, not args.pilot, not args.screen), indent=2))
        return 0
    cores = len(os.sched_getaffinity(0))
    if args.workers * args.mpi_ranks > cores:
        parser.error(f'workers × MPI ranks must not exceed the {cores} available cores')
    xyce = shutil.which(args.xyce)
    if xyce is None:
        parser.error('Xyce executable not found')
    state = run_campaign(sizes, args.output_dir, xyce, args.workers, not args.pilot, args.timeout,
                         not args.screen, args.mpi_ranks, args.parallel_min_cells)
    passed = sum(entry['screening_passed'] for entry in state.values())
    print(f'V2.0.5: {passed}/{len(state)} arrays pass waveform screening. '
          f'Release qualification requires the full report. Results: {args.output_dir / "campaign.json"}')
    return 0 if passed == len(state) else 1


if __name__ == '__main__':
    raise SystemExit(main())
