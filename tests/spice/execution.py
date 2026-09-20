"""Xyce multicore execution with a matching MPI launcher and bounded lifetime."""

from __future__ import annotations

import hashlib
import os
import signal
import shutil
import subprocess
from pathlib import Path

from sram_compiler.per_device_mc import sampling


def execution_command(xyce, deck_path, seed, ranks=1, linear_solver='KLU'):
    """Xyce command; ``linear_solver=None`` leaves the choice to Xyce (recorded as such)."""
    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
        raise ValueError('MPI ranks must be a positive integer')
    command = [str(xyce), *(['-linsolv', str(linear_solver)] if linear_solver else []),
               '-randseed', str(seed), '-o', str(deck_path), str(deck_path)]
    metadata = {'mpi_ranks': ranks, 'blas_threads_per_rank': 1,
                'linear_solver': str(linear_solver) if linear_solver else 'Xyce default'}
    if ranks > 1:
        capabilities = subprocess.check_output([str(xyce), '-capabilities'], text=True, timeout=30)
        if 'Parallel with MPI' not in capabilities:
            raise ValueError('Multicore execution requires an MPI-enabled Xyce build')
        # A launcher from unrelated EDA tools on PATH may have a different MPI ABI.
        launcher = Path(xyce).resolve().parent / 'mpiexec'
        if not launcher.is_file() or not os.access(launcher, os.X_OK):
            raise ValueError(f'Matching MPI launcher not found beside Xyce: {launcher}')
        command = [str(launcher), '-n', str(ranks), *command]
        metadata.update(mpi_launcher=str(launcher.resolve()),
                        mpi_launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest())
    return command, metadata


def execute(command, log, timeout):
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1')
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                          env=environment, start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            # subprocess.run only kills the launcher on timeout. Terminate its
            # process group as well so MPI ranks cannot keep consuming cores.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise


def materialized_sampling_identity():
    return {'sampling_backend': sampling.SAMPLER_VERSION,
            'sampler_sha256': hashlib.sha256(Path(sampling.__file__).read_bytes()).hexdigest()}


LINE_SEARCH = '.OPTIONS NONLIN SEARCHMETHOD=2'


def operating_point_fallbacks(ranks, linear_solver):
    """(ranks, solver, extra option) ladder for a sample whose DC operating point failed.

    Newton with line search converged every sample of the reproduced 16x16 and
    32x32 decks that plain Newton failed with KLU, so it is tried first with the
    same solver and ranks; the default solver on the same ranks follows, and the
    serial rungs come last because they are the expensive ones on large decks
    (a serial 16x256 sample exceeded two hours).
    """
    ladder = [(ranks, linear_solver, LINE_SEARCH), (ranks, None, ''),
              (1, linear_solver, LINE_SEARCH), (1, None, '')]
    seen = {(ranks, linear_solver, '')}
    result = []
    for step in ladder:
        if step not in seen:
            seen.add(step)
            result.append(step)
    return result


def deck_with_max_step(deck_path, max_step, name='deck_tighter_step.sp'):
    """Copy of a materialized sample deck with a tighter ``.TRAN`` maximum step."""
    deck_path = Path(deck_path)
    lines = deck_path.read_text().split('\n')
    for index, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0].upper() == '.TRAN' and len(fields) >= 5:
            fields[4] = f'{max_step:.4e}'
            lines[index] = ' '.join(fields)
            break
    else:
        raise ValueError(f'No .TRAN line with a maximum step in {deck_path}')
    target = deck_path.with_name(name)
    target.write_text('\n'.join(lines))
    return target


def deck_with_option(deck_path, option, name='deck_fallback.sp'):
    """Copy of a materialized sample deck with one extra option line (the original is untouched)."""
    deck_path = Path(deck_path)
    text = deck_path.read_text()
    anchor = '.OPTIONS MEASURE MEASFAIL=1\n'
    if anchor in text:
        text = text.replace(anchor, anchor + option + '\n', 1)
    else:
        text = text.replace('\n.end', f'\n{option}\n.end', 1)
    target = deck_path.with_name(name)
    target.write_text(text)
    return target


def execute_local_ensemble(xyce, deck_path, model_path, seed, samples, ranks, log, timeout,
                           linear_solver='KLU', serial_fallback=True):
    """Keep independent draws in numeric cards; run each sample on MPI cores.

    Xyce 7.4 fails the DC operating point of some sampled circuits with a given
    solver and rank count while another converges (the same 64x16 sample failed
    with KLU on four ranks and on one rank, and converged with the default
    solver). A failed sample is rerun alone through the ladder of
    ``operating_point_fallbacks`` with the same deck, cards and seed; the
    manifest records every step taken per sample.
    """
    deck_path = Path(deck_path)
    manifest = sampling.materialize_decks(deck_path.read_text(), model_path,
                                          deck_path.parent / 'samples', seed, samples)
    manifest['operating_point_fallbacks'] = {}
    manifest['timestep_retries'] = {}
    waveform = Path(str(deck_path) + '.prn')
    log_path = Path(getattr(log, 'name', '')) if isinstance(getattr(log, 'name', None), str) else None

    def log_since(offset):
        return '' if offset is None else log_path.read_text(errors='replace')[offset:]

    def operating_point_failed(offset):
        return 'DC Operating Point Failed' in log_since(offset)

    with waveform.open('wb') as combined:
        for entry in manifest['decks']:
            sample_deck = Path(entry['deck'])
            for old in sample_deck.parent.glob('deck.sp.*'):
                old.unlink()
            command, _ = execution_command(xyce, sample_deck, seed, ranks, linear_solver)
            log.write(f"\nMaterialized local sample {entry['sample']}: {sample_deck}\n")
            log.flush()
            offset = log_path.stat().st_size if log_path is not None and log_path.exists() else None
            code = execute(command, log, timeout)
            executed = sample_deck
            if code and serial_fallback and operating_point_failed(offset):
                for fallback_ranks, fallback_solver, option in operating_point_fallbacks(ranks, linear_solver):
                    label = (f"{fallback_ranks} rank(s), {fallback_solver or 'Xyce default'} solver"
                             + (f", {option}" if option else ''))
                    log.write(f"\nOperating-point fallback for sample {entry['sample']}: {label}\n")
                    log.flush()
                    manifest['operating_point_fallbacks'].setdefault(str(entry['sample']), []).append(label)
                    for old in sample_deck.parent.glob('deck*.sp.*'):
                        old.unlink()
                    executed = deck_with_option(sample_deck, option) if option else sample_deck
                    command, _ = execution_command(xyce, executed, seed, fallback_ranks, fallback_solver)
                    offset = log_path.stat().st_size if log_path is not None else None
                    code = execute(command, log, timeout)
                    if not code or not operating_point_failed(offset):
                        break
            if code and serial_fallback and 'Time step too small' in log_since(offset):
                # Same numerical retry as the native path, for this sample only.
                log.write(f"\nTighter-step retry for sample {entry['sample']}: 5 ps maximum step\n")
                log.flush()
                manifest['timestep_retries'][str(entry['sample'])] = 5e-12
                for old in sample_deck.parent.glob('deck*.sp.*'):
                    old.unlink()
                executed = deck_with_max_step(executed, 5e-12)
                command, _ = execution_command(xyce, executed, seed, ranks, linear_solver)
                code = execute(command, log, timeout)
            if code:
                return code, manifest
            measure = Path(str(executed) + '.mt0')
            shutil.copyfile(measure, Path(str(deck_path) + f".mt{entry['sample']}"))
            with Path(str(executed) + '.prn').open('rb') as source:
                shutil.copyfileobj(source, combined)
            combined.flush()
    return 0, manifest
