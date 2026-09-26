"""Xyce multicore execution with a matching MPI launcher and bounded lifetime."""

from __future__ import annotations

import hashlib
import os
import re
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
    """Copy of a materialized sample deck with one extra option line (the original is untouched).

    The line goes last, before ``.END``: Xyce keeps only the last ``.OPTIONS``
    line of a package, so a fallback placed before a deck's own
    ``.OPTIONS NONLIN`` line (from a case's ``xyce_options``) would be discarded
    and the fallback would repeat the attempt that failed.
    """
    deck_path = Path(deck_path)
    text, count = re.subn(r'(?im)^\s*\.END\s*$', lambda m: option + '\n' + m.group(0).strip(),
                          deck_path.read_text(), count=1)
    if count != 1:
        raise ValueError(f'No .END line in {deck_path}')
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


# V2.2.4: a seeded DC operating point for the largest nominal decks. Plain
# Newton + GMIN stepping spent 13 h in the operating point of a 256x128 deck
# and 92 h without converging at 256x256; MOSFET homotopy converged some
# decks in minutes and failed others after hours, with no pattern in the .IC
# set. A stimulus-free UIC transient settles every node, Xyce saves the result
# as a .NODESET guess, and the deck's own operating point starts from it. The
# .IC values are part of the guess (Xyce refuses .IC with .NODESET), so the
# saved solution is checked against them: at 64x64 and 128x128 it equals the
# plain .IC operating point within 0.42 mV on every node voltage. It solved
# 256x128 in 4 min; at 256x256 Newton had not converged after 5.6 h (open).
SETTLE_STOP = 0.9e-9        # the first cycle starts at 1 ns; no stimulus moves before it
SEED_TOLERANCE = 0.01       # volts: worst .IC node deviation accepted in the seeded solution
OPERATING_POINT_FILE = 'operating_point.txt'
_END = re.compile(r'(?im)^\s*\.END\s*$')
_NODESET = re.compile(r'\.NODESET V\((.+)\) = (\S+)$')


def ic_values(text):
    """{NODE: volts} of every ``.IC`` card in a deck."""
    values = {}
    for line in text.splitlines():
        if line[:4].lower() == '.ic ':
            for item in line.split()[1:]:
                match = re.fullmatch(r'V\((.+)\)=([-+0-9.eE]+)V?', item, re.I)
                if not match:
                    raise ValueError(f'Unparsed .IC item {item!r}')
                values[match.group(1).upper()] = float(match.group(2))
    return values


def settle_deck(deck_path, directory):
    """UIC transient over the stimulus-free window that saves every node as a guess."""
    lines = []
    for line in Path(deck_path).read_text().splitlines():
        key = line.split()[0].upper() if line.split() else ''
        if key.startswith(('.PRINT', '.MEAS')):
            continue
        if key == '.TRAN':
            line = f'.TRAN 5e-12 {SETTLE_STOP:.4e} 0 5e-12 UIC'
        lines.append(line)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text, count = _END.subn(f'.PRINT TRAN V(PRE)\n.SAVE TYPE=NODESET FILE={directory / "nodeset.txt"} '
                            f'TIME={SETTLE_STOP:.4e}\n.END', '\n'.join(lines), count=1)
    if count != 1:
        raise ValueError(f'No .END line in {deck_path}')
    target = directory / 'deck.sp'
    target.write_text(text + '\n')
    return target


def seed_operating_point(deck_path, nodeset_path):
    """Replace the deck's .IC with the settled guess (holding the .IC values) in place.

    Returns the .IC values the saved operating point must reproduce.
    """
    deck_path = Path(deck_path)
    text = deck_path.read_text()
    ic = ic_values(text)
    guess, seen = [], set()
    for line in Path(nodeset_path).read_text().splitlines():
        match = _NODESET.match(line.strip())
        if match:
            name = match.group(1).upper()
            seen.add(name)
            guess.append(f'.NODESET V({match.group(1)}) = {ic.get(name, match.group(2))}')
    guess += [f'.NODESET V({name}) = {value}' for name, value in ic.items() if name not in seen]
    merged = deck_path.with_name('nodeset.sp')
    merged.write_text('\n'.join(guess) + '\n')
    text = '\n'.join(line for line in text.splitlines() if line[:4].lower() != '.ic ')
    text, count = _END.subn(f'.INCLUDE {merged}\n.SAVE TYPE=NODESET '
                            f'FILE={deck_path.with_name(OPERATING_POINT_FILE)} TIME=0\n.END', text, count=1)
    if count != 1:
        raise ValueError(f'No .END line in {deck_path}')
    deck_path.write_text(text + '\n')
    return ic


def operating_point_deviation(ic, saved_path):
    """(volts, node) of the .IC node farthest from its value in a saved operating point."""
    saved = {}
    for line in Path(saved_path).read_text().splitlines():
        match = _NODESET.match(line.strip())
        if match:
            saved[match.group(1).upper()] = float(match.group(2))
    missing = sorted(set(ic) - set(saved))
    if missing:
        raise ValueError(f'{len(missing)} .IC nodes absent from {saved_path}: {missing[:5]}')
    return max(((abs(saved[name] - value), name) for name, value in ic.items()), default=(0.0, None))
