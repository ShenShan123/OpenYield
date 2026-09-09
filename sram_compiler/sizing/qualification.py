"""Reproducible Xyce qualification cases and waveform acceptance metrics.

All output is case-local. Cached simulation files are reused only for identical
deck and model contents; waveform scoring is repeated on every invocation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySpice.Unit import (
    u_Ohm,  # pyright: ignore[reportAttributeAccessIssue] -- generated unit export
    u_pF,  # pyright: ignore[reportAttributeAccessIssue] -- generated unit export
)

from per_device_mc.run import load_config
from sram_compiler.sizing import resolve_driver_sizes
from sram_compiler.sizing.execution import (
    execute, execution_command, execute_local_ensemble, materialized_sampling_identity,
)
from sram_compiler.sizing.table import current_scoring_version, physical_context
from sram_compiler.sizing.timing import TimingConfig
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench

SCORING_VERSION = current_scoring_version()


@dataclass(frozen=True)
class Case:
    rows: int
    cols: int
    cell: str = 'SRAM_6T_CELL'
    mux: bool = False
    operation: str = 'read'
    corner: str = 'SS'
    temperature: float = 125.0
    vdd: float = 0.9
    period: float = 5e-9
    samples: int = 1
    variation: str = 'per-device'
    seed: int = 2026
    cell_variant: str = 'baseline'
    next_row: bool = False
    w_rc: bool = False
    parasitic_factor: float = 1.0
    replica_k: int = 1
    dc_stages: int = 9
    real_cell_mode: int = 0
    vth_std: float = 0.05
    mpi_ranks: int = 1

    @property
    def name(self):
        identity = asdict(self)
        if self.mpi_ranks == 1:
            identity.pop('mpi_ranks')  # Preserve the serial case cache.
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        return f'{self.cell[5:8]}_{self.rows}x{self.cols}_mux{int(self.mux)}_{self.operation}_{self.corner}_{digest}'


def read_measurements(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r'^([A-Z0-9_]+)\s*=\s*(\S+)', line)
        if match:
            name, value = match.groups()
            try:
                result[name] = float(value)
            except ValueError:
                result[name] = None
    return result


def waveform_blocks(path):
    """Yield samples separately, refusing partial rows and incomplete data."""
    header, rows = None, []
    with Path(path).open() as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'Index':
                new_header = [field.upper() for field in fields]
                if header is not None and new_header != header:
                    raise ValueError('Waveform column schema changed between samples')
                header = new_header
            elif fields[0].isdigit():
                if header is None or len(fields) != len(header):
                    raise ValueError('Truncated waveform row')
                row = [float(value) for value in fields]
                if rows and row[1] < rows[-1][1]:
                    yield header, np.asarray(rows)
                    rows = []
                rows.append(row)
    if rows:
        yield header, np.asarray(rows)


def _cross(t, y, level, start, stop, rising=True):
    mask = ((t[:-1] >= start) & (t[1:] <= stop)
            & ((y[:-1] < level) & (y[1:] >= level) if rising else
               (y[:-1] > level) & (y[1:] <= level)))
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError(f'Missing {"rising" if rising else "falling"} crossing at {level:g}')
    i = indices[0]
    return float(t[i] + (level - y[i]) * (t[i + 1] - t[i]) / (y[i + 1] - y[i]))


def score_waveform(header, data, case, measures, timing=None):
    """Boolean acceptance checks plus numeric evidence. Missing data fails."""
    if case.operation == 'read&write':
        return score_sequence(header, data, case)
    t = data[:, header.index('TIME')]
    if not np.all(np.isfinite(data)):
        raise ValueError('Nonfinite waveform')
    if t[-1] < 1e-9 + 1.95 * case.period:
        raise ValueError('Incomplete waveform')
    def signal(name):
        return data[:, header.index(f'V({name})'.upper())]
    def at(name, when):
        return float(np.interp(when, t, signal(name)))
    def cross(name, level, start, stop, rising=True):
        return _cross(t, signal(name), level, start, stop, rising)
    def window(start, stop):
        indices = (t >= start) & (t <= stop)
        if not np.any(indices):
            raise ValueError('Empty acceptance window')
        return indices

    vdd, period = case.vdd, case.period
    start = 1e-9 + 0.65 * period
    capture = 1e-9 + 1.205 * period
    next_access = 1e-9 + 1.7 * period
    wl = f'WL{case.rows - 1}'
    active = cross(wl, 0.5 * vdd, start, capture)
    wlen = cross('WL_EN', 0.5 * vdd, start, capture)
    wl10 = cross(wl, 0.1 * vdd, start, capture)
    wl90 = cross(wl, 0.9 * vdd, start, capture)
    rwl = cross('RWL', 0.5 * vdd, start, capture)
    release = cross(wl, 0.1 * vdd, capture - 0.04 * period, next_access, False)
    pre_on = cross('PRE', 0.9 * vdd, capture - 0.04 * period, next_access, False)
    # Check restoration at the end of clock-high, while precharge is still on.
    # Write drivers deliberately discharge BL before WL rises, and PRE release
    # injects charge into RBL; sampling immediately before WL confuses these
    # intended access events with a failure to restore in the preceding phase.
    initial_precheck = cross('CLK', 0.5 * vdd, start, capture, False)
    next_precheck = initial_precheck + period
    qs = next(name[2:-1] for name in header if name.endswith(f'_{case.rows - 1}_{case.cols - 1}:Q)'))
    checks, metrics = {}, {}
    metrics.update(wl_slew=wl90 - wl10, twldrv=active - wlen,
                   rwl_minus_wl=rwl - active, pre_after_wl=pre_on - release)
    checks['wl_reaches_vdd'] = wl90 < capture
    checks['wl_off_before_precharge'] = pre_on >= release
    if case.variation != 'per-device':
        # The ±10 ps requirement checks systematic corner/load alignment.
        # Independent device mismatch is measured below and qualified by its
        # three-sigma sensing margin, rather than assumed to be zero skew.
        checks['replica_skew'] = abs(rwl - active) <= 10e-12
    pre_min, eq_max = vdd, 0.0
    for when in (initial_precheck, next_precheck):
        for col in range(case.cols):
            bl, blb = at(f'BL{col}', when), at(f'BLB{col}', when)
            pre_min = min(pre_min, bl, blb)
            eq_max = max(eq_max, abs(bl - blb))
        pre_min = min(pre_min, at('RBL', when), at('RBLB', when))
        eq_max = max(eq_max, abs(at('RBL', when) - at('RBLB', when)))
    metrics.update(precharge_min=pre_min, equalization_max=eq_max)
    checks['all_bitlines_restored'] = pre_min >= 0.98 * vdd
    checks['all_bitlines_equalized'] = eq_max <= 0.005
    before = window(1e-9, wl10 - 2e-12)
    checks['no_premature_sense'] = float(signal('S_EN')[before].max()) < 0.1 * vdd
    access_window = window(wl90, capture - 0.02 * period)
    # Include the capture/release interval: that is precisely where a changed
    # address could raise a neighboring WL before the old one has fallen.
    quiet_window = window(wl90, next_precheck)
    idle_wl_peak = max((float(signal(f'WL{row}')[quiet_window].max())
                       for row in range(case.rows) if row != case.rows - 1), default=0)
    metrics['idle_wl_peak'] = idle_wl_peak
    checks['unselected_wordlines_quiet'] = idle_wl_peak < 0.1 * vdd
    retained_from = release + 20e-12
    if case.operation == 'write':
        wen_release = cross('W_EN', 0.1 * vdd, capture - 0.04 * period, next_access, False)
        retained_from = max(retained_from, wen_release + 20e-12)
        driven = float(signal(f'BLB{case.cols - 1}')[access_window].min())
        retained = float(signal(qs)[window(retained_from, next_precheck)].min())
        metrics.update(driven_bl_min=driven, q_retained_min=retained)
        checks['write_bitline_low'] = driven < 0.1 * vdd
        checks['cell_written'] = at(qs, capture - 0.02 * period) > 0.9 * vdd
        checks['cell_retained'] = retained > 0.9 * vdd
        checks['write_overlap'] = at('W_EN', active) > 0.5 * vdd or cross(
            'W_EN', 0.5 * vdd, start, capture) < capture - 0.1 * period
    else:
        sense = cross('S_EN', 0.5 * vdd, start, capture)
        dv = at(f'BLB{case.cols - 1}', sense) - at(f'BL{case.cols - 1}', sense)
        metrics.update(dv_at_sen=dv, sen_minus_wl=sense - active,
                       out_end=at('OUT', capture - 0.02 * period),
                       cell_read_peak=float(signal(qs)[access_window].max()))
        checks['sense_differential'] = dv >= 0.3
        checks['output_settled'] = metrics['out_end'] < 0.1 * vdd
        checks['read_no_disturb'] = metrics['cell_read_peak'] < 0.5 * vdd
        checks['cell_retained'] = float(signal(qs)[window(retained_from, next_precheck)].max()) < 0.1 * vdd

    if case.next_row:
        nb = f'{qs.rsplit("_", 2)[0]}_{(case.rows - 1) ^ (1 << ((case.rows.bit_length() - 1) // 2))}_{case.cols - 1}:Q'
        if f'V({nb})'.upper() in header:
            checks['neighbor_retained'] = float(signal(nb)[window(retained_from, next_precheck)].max()) < 0.1 * vdd
        else:
            raise ValueError('Missing hazard neighbor waveform')
    required = ['TCLK_WLEN', 'TRESTORE', 'TWLDRV',
                'TREAD_TOTAL' if case.operation == 'read' else 'TWRITE_TOTAL']
    if case.rows > 1:
        required.append('TCLK_DEC')
    checks['measures_valid'] = all(isinstance(measures.get(key), (int, float))
                                   and np.isfinite(measures[key]) and measures[key] >= 0 for key in required)
    if timing is not None and checks['measures_valid']:
        checks['restore_budget'] = measures['TRESTORE'] <= 0.8 * timing.low_read
        checks['decoder_budget'] = measures.get('TCLK_DEC', 0) <= timing.t_period / 2
        if case.operation == 'write':
            checks['write_budget'] = measures['TCLK_WLEN'] + measures['TWRITE_TOTAL'] <= 0.8 * timing.low_read
        else:
            checks['wordline_budget'] = metrics['twldrv'] + metrics['wl_slew'] <= 0.15 * (timing.low_read - measures['TCLK_WLEN'])
    access_key = 'TREAD_TOTAL' if case.operation == 'read' else 'TWRITE_TOTAL'
    if case.corner == 'TT':
        edge_nodes = ['WL_EN', 'PRE', 'SA_ISO', 'S_EN' if case.operation == 'read' else 'W_EN']
        for node in edge_nodes:
            edge10 = cross(node, 0.1 * vdd, start, capture)
            edge90 = cross(node, 0.9 * vdd, start, capture)
            metrics[f'{node.lower()}_rise'] = edge90 - edge10
            checks[f'{node.lower()}_edge_budget'] = edge90 - edge10 <= 40e-12
            fall90 = cross(node, .9 * vdd, capture - .04 * period, next_precheck, False)
            fall10 = cross(node, .1 * vdd, capture - .04 * period, next_precheck, False)
            metrics[f'{node.lower()}_fall'] = fall10 - fall90
            checks[f'{node.lower()}_fall_budget'] = fall10 - fall90 <= 40e-12
        for bit in range(max(1, (case.rows - 1).bit_length())):
            if (case.rows - 1) & (1 << bit):
                node = f'A_DFF{bit}'
                edge10 = cross(node, .1 * vdd, 1e-9 + .15 * period, start)
                edge90 = cross(node, .9 * vdd, 1e-9 + .15 * period, start)
                metrics[f'addr_{bit}_rise'] = edge90 - edge10
                checks[f'addr_{bit}_edge_budget'] = edge90 - edge10 <= 40e-12
    limit = 200e-12 if case.operation == 'read' else 100e-12
    # Report the original access constraints separately: K=1/N=9 intentionally
    # retains margin at the cost of the 200 ps limit; never silently relax it.
    metrics['yaml_access_spec_pass'] = isinstance(measures.get(access_key), (int, float)) and measures[access_key] <= limit
    return {'checks': checks, 'metrics': metrics, 'passed': all(checks.values())}


def score_sequence(header, data, case):
    """Write 1/read 1/write 0/read 0 twice; check each access and release."""
    t = data[:, header.index('TIME')]
    period, vdd = case.period, case.vdd
    if not np.all(np.isfinite(data)):
        raise ValueError('Nonfinite waveform')
    if t[-1] < 1e-9 + 8.45 * period:
        raise ValueError('Incomplete back-to-back sequence')
    def signal(name):
        return data[:, header.index(f'V({name})'.upper())]
    def at(name, when):
        return float(np.interp(when, t, signal(name)))
    def cross(name, level, start, stop, rising=True):
        return _cross(t, signal(name), level, start, stop, rising)
    qs = next(name[2:-1] for name in header if name.endswith(f'_{case.rows - 1}_{case.cols - 1}:Q)'))
    checks, diffs, skews = {}, [], []
    wl = f'WL{case.rows - 1}'
    active_sequence = t >= 1e-9
    checks['unselected_wordlines_quiet'] = all(
        float(signal(f'WL{row}')[active_sequence].max()) < .1 * vdd
        for row in range(case.rows) if row != case.rows - 1)
    for cycle in range(8):
        start = 1e-9 + (cycle + .65) * period
        capture = 1e-9 + (cycle + 1.205) * period
        expected = 1 if (cycle // 2) % 2 == 0 else 0
        active = cross(wl, .5 * vdd, start, capture)
        replica = cross('RWL', .5 * vdd, start, capture)
        skews.append(replica - active)
        if case.variation != 'per-device':
            checks[f'cycle_{cycle}_replica_skew'] = abs(replica - active) <= 10e-12
        q = at(qs, capture - .02 * period)
        checks[f'cycle_{cycle}_cell_data'] = q > .9 * vdd if expected else q < .1 * vdd
        if cycle % 2:
            sense = cross('S_EN', .5 * vdd, start, capture)
            delta = at(f'BLB{case.cols - 1}', sense) - at(f'BL{case.cols - 1}', sense)
            diffs.append(-delta if expected else delta)
            output = at('OUT', capture - .02 * period)
            checks[f'cycle_{cycle}_output_data'] = output > .9 * vdd if expected else output < .1 * vdd
            checks[f'cycle_{cycle}_sense_margin'] = diffs[-1] >= .3
        if cycle < 7:
            next_clock = 1e-9 + (cycle + 1.715) * period
            released = cross(wl, .1 * vdd, capture - .04 * period, next_clock, False)
            if cycle % 2 == 0:
                released = max(released, cross('W_EN', .1 * vdd, capture - .04 * period, next_clock, False))
            mask = (t >= released + 20e-12) & (t <= next_clock)
            if not np.any(mask):
                raise ValueError('Missing sequence retention window')
            checks[f'cycle_{cycle}_retained'] = bool(np.min(signal(qs)[mask]) > .9 * vdd if expected
                                                    else np.max(signal(qs)[mask]) < .1 * vdd)
            checks[f'cycle_{cycle}_restored'] = all(
                at(node, next_clock) >= .98 * vdd for node in
                [*[f'BL{col}' for col in range(case.cols)], *[f'BLB{col}' for col in range(case.cols)], 'RBL', 'RBLB'])
    return {'checks': checks, 'metrics': {'min_sequence_dv': min(diffs),
            'max_sequence_skew': max(abs(value) for value in skews)}, 'passed': all(checks.values())}


def generate_case(case, directory, max_step=10e-12):
    cfg = load_config(case.rows, case.cols, case.corner)
    assert cfg.global_config is not None, 'Global configuration was not loaded'
    cfg.global_config.sram_cell_type = case.cell
    cfg.global_config.vdd = case.vdd
    cfg.global_config.temperature = case.temperature
    cfg.global_config.sizing = {'mode': 'rules_only', 'parasitic_factor': case.parasitic_factor,
                                    'replica': {'K': case.replica_k, 'N': case.dc_stages}}
    sizes = resolve_driver_sizes(cfg, mux=case.mux,
                                 physical_context=physical_context(case.w_rc, real_cell_mode=case.real_cell_mode))
    cell = getattr(cfg, case.cell.lower())
    if case.cell_variant in ('write_box', 'read_box'):
        cell.pmos_width.value = cell.pmos_width.upper
        widths = list(cell.nmos_width.value)
        widths[0] = cell.nmos_width.upper[0] if case.cell_variant == 'write_box' else cell.nmos_width.lower[0]
        widths[1] = cell.nmos_width.lower[1]
        cell.nmos_width.value = widths
    elif case.cell_variant == 'unwritable':
        cell.pmos_width.value *= 4
        cell.nmos_width.value[1] *= 0.25
    elif case.cell_variant != 'baseline':
        raise ValueError(f'Unknown cell variant {case.cell_variant}')
    next_row = ((case.rows - 1) ^ (1 << ((case.rows.bit_length() - 1) // 2))) if case.next_row else None
    tb = Sram6TCoreMcTestbench(
        cfg, sram_cell_type=case.cell, choose_columnmux=case.mux, corner=case.corner,
        variation_mode=case.variation, mc_seed=case.seed, vth_std=case.vth_std,
        w_rc=case.w_rc, real_cell_mode=case.real_cell_mode, driver_sizes=sizes,
        pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
        sim_path=str(directory), next_row=next_row, t_max_step=max_step,
    )
    TimingConfig(case.period, 0, 0, 0, source='qualification').apply(tb)
    tb.t_step = 2e-12
    circuit = tb.create_testbench(case.operation, case.rows - 1, case.cols - 1)
    simulator = circuit.simulator(simulator='xyce-serial', temperature=case.temperature, nominal_temperature=27)
    tb.add_analysis(simulator.circuit, case.operation, case.samples)
    tb.add_meas_and_print(simulator, tb.data_init(), case.operation)
    probes = ['CLK', 'RWL', 'RBLB', 'S_EN']
    probes += [f'BL{col}' for col in range(case.cols)] + [f'BLB{col}' for col in range(case.cols)]
    probes += [f'WL{row}' for row in range(case.rows)]
    simulator.circuit.raw_spice += '.PRINT TRAN ' + ' '.join(f'V({node})' for node in probes) + '\n'
    deck = str(simulator)
    if case.variation not in ('nominal', 'shared', 'per-device'):
        raise ValueError('Unknown variation mode')
    if case.variation == 'nominal' and case.samples != 1:
        raise ValueError('Nominal cases use one sample')
    (directory / 'variation.json').write_text(json.dumps({
        **tb.variation_summary, 'samples': case.samples, 'seed': case.seed,
        'full_device_coverage': case.variation == 'per-device' and case.real_cell_mode == 0,
    }, indent=2))
    return deck, sizes


def run_case(case: Case, output_root, xyce: str, *, timing=None, timeout=21600) -> dict[str, Any]:
    """Serialize identical cases so concurrent resume commands cannot corrupt output."""
    directory = Path(output_root).resolve() / case.name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = _run_case_unlocked(case, output_root, xyce, timing=timing, timeout=timeout)
        samples = result.get('samples', [])
        errors = [sample['error'] for sample in samples if sample.get('error')]
        incomplete = {'Incomplete waveform', 'Incomplete back-to-back sequence'}
        valid_checks = all(all(sample.get('checks', {}).values()) for sample in samples)
        numerical_failure = (not result.get('error') and bool(errors)
                             and all(error in incomplete for error in errors) and valid_checks)
        failed_single_sample = (case.samples == 1 and bool(result.get('simulator_returncode'))
                          and str(result.get('error', '')).startswith('Xyce exit ')
                          and not errors and valid_checks)
        log = directory / 'xyce.log'
        if (numerical_failure or failed_single_sample) and log.exists() and 'Time step too small' in log.read_text():
            # Xyce sampling can return success even when one transient stops
            # early. Retry the same seeded ensemble at a tighter resolution;
            # retain the failed attempt, and never retry failed electrical checks.
            retry = _run_case_unlocked(case, directory / 'retry_5ps', xyce,
                                      timing=timing, timeout=timeout, max_step=5e-12)
            retry['original_attempt'] = {
                'directory': result['directory'], 'signature': result.get('signature'),
                'max_time_step': result['max_time_step'], 'sample_errors': errors,
                'simulator_returncode': result.get('simulator_returncode', 0),
                'error': result.get('error'),
            }
            (Path(retry['directory']) / 'result.json').write_text(json.dumps(retry, indent=2, allow_nan=False))
            return retry
        return result


def _run_case_unlocked(case: Case, output_root, xyce: str, *, timing=None, timeout=21600, max_step=10e-12) -> dict[str, Any]:
    directory = Path(output_root).resolve() / case.name
    directory.mkdir(parents=True, exist_ok=True)
    logging.getLogger('PySpice').setLevel(logging.ERROR)
    start = time.monotonic()
    result: dict[str, Any] = {'case': asdict(case), 'directory': str(directory),
                              'passed': False, 'scoring_version': SCORING_VERSION,
                              'max_time_step': max_step}
    try:
        with (directory / 'generation.log').open('w') as log, redirect_stdout(log):
            deck, sizes = generate_case(case, directory, max_step)
        digest = hashlib.sha256(deck.encode())
        # Only the models this deck includes determine its identity; older
        # immutable model artifacts in the same directory are historical data.
        for include in re.findall(r'^\.include\s+["\']?([^"\'\n]+?)["\']?\s*$', deck, re.I | re.M):
            model = Path(include.strip())
            if not model.is_absolute():
                model = Path(__file__).resolve().parents[2] / model
            digest.update(model.read_bytes())
        digest.update(sizes.pdk_key.encode())
        digest.update(str(case.seed).encode())
        xyce = shutil.which(xyce) or xyce
        xyce_hash = hashlib.sha256(Path(xyce).read_bytes()).hexdigest()
        digest.update(xyce_hash.encode())
        deck_path = directory / 'deck.sp'
        command, execution = execution_command(xyce, deck_path, case.seed, case.mpi_ranks)
        materialized = case.mpi_ranks > 1 and case.variation == 'per-device'
        if materialized:
            execution.update(materialized_sampling_identity())
        if case.mpi_ranks > 1:
            digest.update(json.dumps(execution, sort_keys=True).encode())
        digest.update(b'blas_threads=1;linear_solver=KLU')
        signature = digest.hexdigest()
        result.update(signature=signature, execution=execution, simulator_sha256=xyce_hash,
                      linear_solver='KLU', sizing=sizes.to_dict(),
                      variation=json.loads((directory / 'variation.json').read_text()))
        deck_path.write_text(deck)
        complete = directory / 'simulation_complete.json'
        completion = json.loads(complete.read_text()) if complete.exists() else {}
        cached = (completion.get('signature') == signature
                  and (directory / 'deck.sp.prn').exists()
                  and len(list(directory.glob('deck.sp.mt*'))) == case.samples)
        if not cached:
            # Clear only outputs generated for this exact case, before execution.
            for path in directory.glob('deck.sp.*'):
                path.unlink()
            with (directory / 'xyce.log').open('w') as log:
                if materialized:
                    variation = json.loads((directory / 'variation.json').read_text())
                    returncode, _ = execute_local_ensemble(xyce, deck_path, variation['mc_model_file'],
                                                          case.seed, case.samples, case.mpi_ranks,
                                                          log, timeout)
                else:
                    returncode = execute(command, log, timeout * case.samples)
            # A process exit is evidence of execution, not waveform completion.
            # Retain failed exits too, so resuming can inspect/retry their exact
            # inputs without discarding the original failure.
            completion = {'signature': signature, 'linear_solver': 'KLU',
                          'returncode': returncode, 'execution': execution}
            complete.write_text(json.dumps(completion))
        result.update(sizing=sizes.to_dict(), cached=cached,
                      execution=execution,
                      variation=json.loads((directory / 'variation.json').read_text()),
                      simulator_sha256=xyce_hash,
                      signature=completion['signature'],
                      linear_solver=completion.get('linear_solver', 'Xyce default'),
                      simulator_returncode=completion.get('returncode', 0))
        if materialized:
            result['sampling'] = json.loads((directory / 'samples/sampling.json').read_text())
        if result['simulator_returncode']:
            raise RuntimeError(f"Xyce exit {result['simulator_returncode']}; see xyce.log")
        paths = sorted(directory.glob('deck.sp.mt*'), key=lambda path: int(path.suffix[3:]))
        measures = [read_measurements(path) for path in paths]
        scored = []
        for index, (header, data) in enumerate(waveform_blocks(directory / 'deck.sp.prn')):
            if index >= len(measures):
                raise ValueError('Missing sample measures')
            try:
                scored.append(score_waveform(header, data, case, measures[index], timing))
            except (ValueError, StopIteration) as exc:
                scored.append({'passed': False, 'error': str(exc)})
        if len(scored) != case.samples or len(measures) != case.samples:
            raise ValueError(f'Incomplete samples: {len(scored)} waveforms, {len(measures)} measures; expected {case.samples}')
        result.update(passed=all(sample['passed'] for sample in scored), samples=scored, measures=measures)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result['error'] = str(exc)
    result['elapsed_seconds'] = time.monotonic() - start
    (directory / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
