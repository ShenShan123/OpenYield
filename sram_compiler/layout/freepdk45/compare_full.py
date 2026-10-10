"""Prepare matched complete-macro estimated/post-layout transient decks.

The native input is a nominal 8x8 6T read&write compiler deck at 1 V, TT,
25 C, 9 ns. This four-cycle diagnostic contains W1/R1/W0/R0 and ends at
38.795 ns, including final recovery and ending before the fifth capture.
It does not qualify the full release waveform screen.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import time

from sram_compiler.calibrated_rc import apply_to_deck, initial_state, validate_source
from sram_compiler.parasitics import read_dspf, spice_value
from utils.xyce import execute_xyce
from .reduce_rc import reduce
from .run import sha256


def prepare(native_deck, profile_path, extraction_dir, output_dir):
    """Keep original stimulus and print probes; share complete logical ICs."""
    native_deck, profile_path = Path(native_deck), Path(profile_path)
    extraction_dir, output_dir = Path(extraction_dir), Path(output_dir)
    content = profile_path.read_bytes()
    profile = json.loads(gzip.decompress(content) if profile_path.suffix == '.gz' else content)
    if profile.get('schema') == 2:
        from sram_compiler.per_net_rc import assemble
        assemble(profile, profile_path.parent)
    validate_source(native_deck.read_text(), profile)
    receipt = profile['extraction_receipt']
    dspf_path = extraction_dir / 'xrc/full_macro.dspf'
    if sha256(dspf_path) != receipt['dspf_sha256']:
        raise ValueError('Post-layout extraction differs from the calibration reference')
    if receipt['drc_violations'] != 0 or receipt['lvs'] != 'CORRECT':
        raise ValueError('Physical verification must pass')
    text = native_deck.read_text()
    clock = re.search(r'(?im)^VCLK\s+clk\s+VSS.*?PULSE\(([^)]+)\)', text)
    if clock is None or abs(spice_value(clock[1].split()[-1].rstrip('Ss'))-9e-9) > 1e-18:
        raise ValueError('Complete macro diagnostic requires the native 9 ns clock')
    conditions = {name: spice_value(value.rstrip('Vv')) for name, value in
                  re.findall(r'(?i)\bV\(([^)]+)\)\s*=\s*([^\s]+)', text)}
    text = '\n'.join(line for line in text.splitlines()
                     if not line.lower().startswith(('.ic ', '.measure ', '.meas '))) + '\n'
    text, count = re.subn(r'(?im)^\.TRAN .*$', '.TRAN 5e-12 38.795e-9 0 20e-12 UIC', text)
    if count != 1:
        raise ValueError('Comparison needs one transient analysis')
    # Solver selection does not change the physical network.
    text = re.sub(r'(?im)^\.OPTIONS\s+LINSOL[^\n]*\n?', '', text)
    text = text.replace('\n.END', '\n.OPTIONS LINSOL TYPE=KLU\n.END')
    full = copy.deepcopy(profile)
    full.update(reduce(read_dspf(dspf_path), 0))
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = dict(source_deck_sha256=sha256(native_deck),
                      profile_sha256=sha256(profile_path),
                      source_dspf_sha256=sha256(dspf_path),
                      complete_cycles=['write1', 'read1', 'write0', 'read0'],
                      period_s=9e-9, stop_s=38.795e-9, output_step_s=5e-12,
                      maximum_integration_step_s=20e-12,
                      initialization='common complete schematic logic state, UIC; released throughout transient',
                      full_release_screen=False)
    for kind, network in [('estimated', profile), ('post', full)]:
        directory = output_dir / kind
        directory.mkdir()
        deck = apply_to_deck(text, network)
        states = initial_state(network, conditions)
        entries = '\n'.join(f'.IC V({node})={value:.15g}' for node, value in states.items())
        deck = deck.replace('\n.END', '\n' + entries + '\n.END')
        (directory / 'deck.sp').write_text(deck)
        (directory / 'provenance.json').write_text(json.dumps(
            dict(provenance, model=kind, deck_sha256=sha256(directory/'deck.sp')),
            indent=2) + '\n')
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--extraction-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--xyce', default='Xyce')
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args(argv)
    output = prepare(args.deck, args.profile, args.extraction_dir, args.output_dir)
    if args.execute:
        solver = shutil.which(args.xyce)
        if solver is None:
            raise ValueError(f'Xyce not found: {args.xyce}')
        def run(kind):
            directory = (output / kind).resolve()
            deck = directory / 'deck.sp'
            started = time.monotonic()
            result = execute_xyce(deck, [solver, str(deck)],
                                  log_path=directory/'xyce.log', timeout=args.timeout,
                                  env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
            receipt = dict(returncode=result.returncode,
                           elapsed_seconds=time.monotonic()-started,
                           solver_sha256=sha256(solver), waveform_validated=False)
            (directory/'execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
            return result.returncode
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(run, ['estimated', 'post']))
        if any(codes):
            raise RuntimeError('Comparison simulation failed; preserved output is unqualified')
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
