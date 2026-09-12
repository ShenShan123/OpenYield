"""Preserve failed Xyce operating points and retry the same sampled circuit."""

from pathlib import Path
import re
import shutil
import subprocess


def execute_xyce(deck_path, command, *, log_path=None, cwd=None, timeout=None, env=None):
    """One Newton line-search retry for a failed DC operating point.

    Xyce sampling can report exit zero after a failed operating point. Treat
    that as failure too, since subsequent samples may have corrupted states.
    Keep the first attempt and preserve the native sampling seed on retry.
    Electrical failures are handled by the caller and never trigger retries.
    """
    deck_path = Path(deck_path)
    log_path = Path(log_path) if log_path is not None else Path(str(deck_path) + '.log')

    def run():
        try:
            result = subprocess.run(command, cwd=cwd, env=env, timeout=timeout,
                                    capture_output=True, text=True, check=False)
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
            log_path.write_text(f'Xyce timed out after {timeout} seconds\n'
                                + decoded(exc.stdout) + '\n' + decoded(exc.stderr))
            raise
        log_path.write_text(result.stdout + '\n' + result.stderr)
        if 'DC Operating Point Failed' in result.stdout + result.stderr:
            result.returncode = result.returncode or 1
        return result

    result = run()
    output = result.stdout + result.stderr
    if 'DC Operating Point Failed' not in output:
        return result
    deck = deck_path.read_text()
    if re.search(r'(?im)^\s*\.OPTIONS\s+NONLIN\b[^\n]*\bSEARCHMETHOD\s*=\s*2\b', deck):
        return result
    options = ['.OPTIONS NONLIN SEARCHMETHOD=2']
    if re.search(r'(?im)^\s*\.SAMPLING\b', deck):
        if not re.search(r'(?im)^\s*\.OPTIONS\s+SAMPLES\b[^\n]*\bSEED\s*=\s*\d+', deck):
            seed = re.search(r'Seeding random number generator with\s+(\d+)', output)
            if seed is None or int(seed[1]) <= 0:
                # An unseeded retry would silently replace the failed sample.
                return result
            options.append(f'.OPTIONS SAMPLES SEED={seed[1]}')
    amended, count = re.subn(r'(?im)^\s*\.END\s*$', '\n'.join(options) + '\n.END', deck, count=1)
    if count != 1:
        return result

    attempt = deck_path.parent / 'dcop_attempt'
    suffix = 1
    while attempt.exists():
        attempt = deck_path.parent / f'dcop_attempt_{suffix}'
        suffix += 1
    attempt.mkdir()
    files = {deck_path, log_path, *deck_path.parent.glob(deck_path.name + '.*')}
    for path in files:
        if path.is_file():
            shutil.copy2(path, attempt / path.name)
    # The first attempt stopped part-way, so it can leave per-sample measure
    # files the retry never rewrites.  Parsers index those files by sample
    # number, so a surviving one would report the failed attempt's value as a
    # retry result.  The copies above keep the evidence.
    for pattern in ('.mt*', '.ms*', '.prn', '.prn.*'):
        for path in deck_path.parent.glob(deck_path.name + pattern):
            if path.is_file():
                path.unlink()
    deck_path.write_text(amended)
    result = run()
    log_path.write_text(f'Operating-point retry with Newton line search; original attempt: {attempt}\n'
                        + '\n'.join(options) + '\n' + log_path.read_text())
    return result
