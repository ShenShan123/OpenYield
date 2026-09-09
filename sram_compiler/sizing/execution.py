"""Xyce multicore execution with a matching MPI launcher and bounded lifetime."""

from __future__ import annotations

import hashlib
import os
import signal
import shutil
import subprocess
from pathlib import Path

from per_device_mc import sampling


def execution_command(xyce, deck_path, seed, ranks=1):
    if isinstance(ranks, bool) or not isinstance(ranks, int) or ranks < 1:
        raise ValueError('MPI ranks must be a positive integer')
    command = [str(xyce), '-linsolv', 'KLU', '-randseed', str(seed),
               '-o', str(deck_path), str(deck_path)]
    metadata = {'mpi_ranks': ranks, 'blas_threads_per_rank': 1, 'linear_solver': 'KLU'}
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


def execute_local_ensemble(xyce, deck_path, model_path, seed, samples, ranks, log, timeout):
    """Keep independent draws in numeric cards; run each sample on MPI cores."""
    deck_path = Path(deck_path)
    manifest = sampling.materialize_decks(deck_path.read_text(), model_path,
                                          deck_path.parent / 'samples', seed, samples)
    waveform = Path(str(deck_path) + '.prn')
    with waveform.open('wb') as combined:
        for entry in manifest['decks']:
            sample_deck = Path(entry['deck'])
            for old in sample_deck.parent.glob('deck.sp.*'):
                old.unlink()
            command, _ = execution_command(xyce, sample_deck, seed, ranks)
            log.write(f"\nMaterialized local sample {entry['sample']}: {sample_deck}\n")
            log.flush()
            code = execute(command, log, timeout)
            if code:
                return code, manifest
            measure = Path(str(sample_deck) + '.mt0')
            shutil.copyfile(measure, Path(str(deck_path) + f".mt{entry['sample']}"))
            with Path(str(sample_deck) + '.prn').open('rb') as source:
                shutil.copyfileobj(source, combined)
            combined.flush()
    return 0, manifest
