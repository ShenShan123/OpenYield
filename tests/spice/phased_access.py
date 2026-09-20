"""Reproducible V2.2.0 SPICE integration screen. Explicit opt-in; full cells only."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import time

from sram_compiler.per_device_mc.run import load_config
from sram_compiler.sizing.timing import TimingConfig
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from sram_compiler.version import VERSION
from tests.spice.execution import execution_command, execute_local_ensemble, execute
from tests.spice.phased_waveforms import score
from utils.xyce import execute_xyce


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE_PATHS = tuple(sorted({
    path for folder in ('sram_compiler', 'utils', 'tests/spice', 'tran_models')
    for path in (REPOSITORY / folder).rglob('*')
    if path.suffix in ('.py', '.json', '.yaml', '.spice') and '__pycache__' not in path.parts
}))
SOURCE_HASHES = {str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in SOURCE_PATHS}


def check_sources():
    """Workers cache imports; never attribute an old import to changed files."""
    changed = [str(path.relative_to(REPOSITORY)) for path in SOURCE_PATHS
               if hashlib.sha256(path.read_bytes()).hexdigest()
               != SOURCE_HASHES[str(path.relative_to(REPOSITORY))]]
    if changed:
        raise RuntimeError('Sources changed during the campaign: ' + ', '.join(changed))


def apply_pattern(deck, pattern, testbench):
    """Replace external sources only; transistor topology and samples are fixed."""
    rows, cols = testbench.num_rows, testbench.num_cols
    period, vdd = float(testbench.t_period), float(testbench.vdd)
    if len(pattern) != 8:
        raise ValueError('Custom integration patterns have exactly eight cycles')
    for entry in pattern:
        if (entry['op'] not in ('read', 'write', 'idle')
                or not 0 <= entry['row'] < rows
                or len(entry['data']) != cols or set(entry['data']) - {'0', '1'}):
            raise ValueError('Invalid operation, row, or column data in pattern')

    def pwl(name, node, values, initial):
        points, last = [(0., initial)], initial
        for cycle, value in enumerate(values):
            when = 1e-9 + (cycle + .1) * period
            points.extend([(when, last), (when + .01 * period, value)])
            last = value
        points.append((1e-9 + 9 * period, last))
        return f'{name} {node} VSS PWL(' + ' '.join(f'{t:.12g} {v:.12g}' for t, v in points) + ')'

    sources = {
        'VCSB': pwl('VCSB', 'csb', [vdd if x['op'] == 'idle' else 0 for x in pattern], vdd),
        'VWEB': pwl('VWEB', 'web', [0 if x['op'] == 'write' else vdd for x in pattern], vdd),
    }
    for bit in range(max(1, (rows - 1).bit_length())):
        sources[f'VADDR_{bit}'] = pwl(f'VADDR_{bit}', f'A{bit}',
                                     [((x['row'] >> bit) & 1) * vdd for x in pattern], 0)
    for col in range(cols):
        sources[f'VDIN{col}'] = pwl(f'VDIN{col}', f'DIN{col}',
                                  [int(x['data'][col]) * vdd for x in pattern], 0)
    deck = '\n'.join(sources.get(line.split()[0], line) if line.split() else line
                     for line in deck.splitlines()) + '\n'
    # The compiler's WRWR datum expectations do not describe this stimulus.
    # Independent physical waveform checks cover every custom operation.
    return '\n'.join(line for line in deck.splitlines()
                     if not re.match(r'\.meas tran V(?:ACCESS|HOLD|WEN_ACCESS|RESTORE)_ERROR',
                                     line, re.I)) + '\n'


def prepare(case, directory):
    """Build a deck and its probe contract, with explicit nominal/MC provenance."""
    check_sources()
    rows, cols = case.get('rows', 8), case.get('cols', 4)
    operation = case.get('operation', 'read&write')
    with contextlib.redirect_stdout(io.StringIO()):
        config = load_config(rows, cols, case.get('corner', 'SS'))
        config.global_config.vdd = case.get('vdd', .9)
        config.global_config.temperature = case.get('temperature', 125)
        config.global_config.sizing = {'mode': 'lookup'}
        tb = Sram6TCoreMcTestbench(
            config, sram_cell_type=case.get('cell', 'SRAM_6T_CELL'),
            interconnect=case.get('interconnect'),
            choose_columnmux=case.get('mux', False), w_rc=case.get('w_rc', True), real_cell_mode=0,
            corner=config.global_config.corner, temperature=config.global_config.temperature,
            variation_mode=case.get('variation', 'nominal'), mc_seed=case.get('seed', 20260919),
            next_row=case.get('next_row'), select_every=case.get('select_every', 1),
            q_init_val=case.get('background', 0), t_max_step=2e-11, sim_path=str(directory))
        if 'period' in case:
            TimingConfig(case['period'], 0, 0, 0, source='V2.2.0 probe').apply(tb)
        tb.t_step = 5e-12
        circuit = tb.create_testbench(operation, case.get('row', rows - 1), cols - 1)
        simulator = circuit.simulator(simulator='xyce-serial', temperature=tb.temperature,
                                      nominal_temperature=27)
        tb.add_analysis(simulator.circuit, operation, 1)
        tb.add_meas_and_print(simulator, tb.data_init(), operation)
        deck = str(simulator)

    probes = set()
    for line in deck.splitlines():
        if line.upper().startswith('.PRINT TRAN'):
            probes.update(re.findall(r'V\(([^)]+)\)', line, re.I))
    probes.update(('CLK', 'CLK_BUF', 'CS', 'WE', 'WL_EN', 'RWL_far', 'PRE', 'RBL',
                   'RBL_DELAY', 'W_EN', 'S_EN', 'SA_ISO'))
    if operation != 'write':
        probes.add('OUT')
    nodes = {name: {} for name in ('cells', 'wl', 'pre', 'wen', 'sen', 'iso', 'data', 'sense', 'sense_state')}
    checked_rows = {0, rows // 2, rows - 1, tb.target_row, case.get('next_row')}
    checked_rows.update(entry['row'] for entry in case.get('pattern', ()))
    for row in range(rows):
        for col in range(cols):
            # Every cell for small arrays; complete selected/sentinel rows for
            # larger arrays, plus every physical WL endpoint in either case.
            if rows * cols <= 1024 or row in checked_rows:
                q, qb = (f'{tb.cell_inst_prefix}_{row}_{col}:{pin}' for pin in ('Q', 'QB'))
                nodes['cells'][f'{row},{col}'] = [q, qb]
                probes.update((q, qb))
        nodes['wl'][str(row)] = [tb.cell_probe('WL', row, col) for col in sorted({0, cols - 1})]
        probes.update(nodes['wl'][str(row)])
    for col in range(cols):
        group, sense_col = col // tb.mux_in, (col // tb.mux_in) * tb.mux_in
        terminals = {
            'pre': f'{tb.prch_inst_prefix}_{col}:ENB_end' if tb.w_rc else tb.control_tap('PRE', col),
            'wen': f'{tb.wdrv_inst_prefix}_{col}:EN_end' if tb.w_rc else tb.control_tap('w_en', col),
            'sen': f'{tb.sa_inst_prefix}_{group}:EN_end' if tb.w_rc else tb.control_tap('s_en', sense_col),
            'iso': f'{tb.sa_inst_prefix}_{group}:ISO_end' if tb.w_rc else tb.control_tap('sa_iso', sense_col),
        }
        for name, node in terminals.items():
            nodes[name][str(col)] = node
            probes.add(node)
        if operation != 'read':
            node = f'{tb.wdrv_inst_prefix}_{col}:DIN_end' if tb.w_rc else f'DIN_buf{col}'
            nodes['data'][str(col)] = node
            probes.add(node)
        nodes['sense'][str(col)] = [tb.sense_input_probe('IN', col), tb.sense_input_probe('INB', col)]
        nodes['sense_state'][str(col)] = [f'SA_Q{group}', f'SA_QB{group}']
        probes.update(nodes['sense'][str(col)])
        probes.update(nodes['sense_state'][str(col)])
    lines = [line for line in deck.splitlines()
             if not line.upper().startswith('.PRINT TRAN') and line.lower() != '.end']
    deck = '\n'.join(lines) + '\n.PRINT TRAN ' + ' '.join(
        f'V({node})' for node in sorted({node.upper() for node in probes})) + '\n.END\n'
    if 'pattern' in case:
        deck = apply_pattern(deck, case['pattern'], tb)
    path = directory / 'deck.sp'
    path.write_text(deck)
    check_sources()
    metadata = {
        'version': VERSION, 'case': case, 'corner': tb.corner, 'temperature': tb.temperature,
        'period': float(tb.t_period), 'vdd': float(tb.vdd),
        'sample_interval': float(tb.t_step), 'analysis_stop': float(tb._analysis_stop(operation)),
        'nodes': nodes, 'rows': rows, 'cols': cols, 'row': tb.target_row, 'operation': operation,
        'driver_sizes': tb.driver_sizes.to_dict(), 'timing': tb.timing_config.to_dict(),
        'variation': tb.variation_summary,
        'model_sha256': hashlib.sha256(Path(getattr(config.global_config, 'pdk_path_' + tb.corner)).read_bytes()).hexdigest(),
        'sources': SOURCE_HASHES,
    }
    return path, metadata


def run(case, root, xyce):
    name = case['name']
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Case name must be a single directory name')
    directory = Path(root).resolve() / name
    directory.mkdir(parents=True, exist_ok=False)
    path, metadata = prepare(case, directory)
    metadata_path = directory / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    ranks, seed = case.get('mpi_ranks', 1), case.get('seed', 20260919)
    command, execution = execution_command(xyce, path, seed, ranks)
    started = time.time()
    try:
        if ranks > 1:
            with (directory / 'xyce.log').open('w') as log:
                if case.get('variation') == 'per-device':
                    code, execution = execute_local_ensemble(
                        xyce, path, next(directory.glob('models_per_device_*.spice')),
                        seed, 1, ranks, log, case.get('timeout', 3600))
                else:
                    code = execute(command, log, case.get('timeout', 3600))
        else:
            result = execute_xyce(
                path, command, log_path=directory / 'xyce.log', timeout=case.get('timeout', 3600),
                env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            code = result.returncode
    except Exception as exc:
        metadata.update(passed=False, simulation_error=f'{type(exc).__name__}: {exc}',
                        elapsed_seconds=time.time() - started, execution=execution)
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        raise
    metadata.update(returncode=code, elapsed_seconds=time.time() - started, execution=execution,
                    deck_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    solver=subprocess.check_output([xyce, '-v'], text=True).strip())
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    if code:
        print(name, 'SIMULATION FAILED', flush=True)
        return dict(metadata, passed=False)
    try:
        check_sources()
        summary = score(directory)
    except Exception as exc:
        metadata['validation_error'] = f'{type(exc).__name__}: {exc}'
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        raise
    print(name, summary['passed'], summary['failures'][:8], round(metadata['elapsed_seconds'], 1), flush=True)
    return summary


def main():
    import argparse
    import concurrent.futures
    import shutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('v220_cases.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--xyce', default=shutil.which('Xyce'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', action='append', help='Run only these named cases; repeatable')
    args = parser.parse_args()
    if not args.xyce:
        parser.error('Provide --xyce or put Xyce on PATH')
    if args.workers < 1:
        parser.error('--workers must be positive')
    cases = json.loads(args.cases.read_text())
    if args.name:
        missing = set(args.name) - {case['name'] for case in cases}
        if missing:
            parser.error('Unknown case names: ' + ', '.join(sorted(missing)))
        cases = [case for case in cases if case['name'] in args.name]
    failed = False
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, case, args.output, args.xyce): case['name'] for case in cases}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                failed |= not result.get('passed', False)
            except Exception as exc:
                failed = True
                print(futures[future], type(exc).__name__, str(exc), flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
