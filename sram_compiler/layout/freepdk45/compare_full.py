"""Prepare matched complete-macro estimated/post-layout transient decks.

The native input is a nominal 6T read&write compiler deck matching the layout.
This four-cycle diagnostic contains W1/R1/W0/R0 and includes final recovery,
ending 5 ps before the fifth capture (38.795 ns with the original 9 ns clock).
It does not qualify the full release waveform screen.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

from sram_compiler.calibrated_rc import apply_to_deck, initial_state, validate_source, read_initial_conditions
from sram_compiler.parasitics import read_dspf, spice_value
from utils.xyce import execute_xyce
from .reduce_rc import reduce
from .run import sha256


def comparison_vdd(deck, scope):
    """Use the native DC supply and reject a stale voltage-scoped profile."""
    supply = re.search(r'(?im)^VVDD\s+VDD\s+(?:VSS|0)\s+(?:DC\s+)?(\S+)\s*$', deck)
    if supply is None:
        raise ValueError('Complete macro comparison requires a constant native VDD supply')
    vdd = spice_value(supply[1].rstrip('Vv'))
    if vdd <= 0:
        raise ValueError('Native VDD must be positive')
    if 'vdd' in scope:
        reference_vdd = float(scope['vdd'])
        if not math.isfinite(reference_vdd) or abs(reference_vdd-vdd)>1e-12:
            raise ValueError('Native VDD differs from the calibration profile scope')
    return vdd


def probe_alias_states(deck, states):
    """Give unloaded observation aliases their physical node's initial value.

    An alias connected only through its probe resistor has zero current and
    must share that terminal voltage. Leaving it at UIC's implicit zero adds
    an artificial startup discontinuity that can affect the error estimate.
    Aliases carrying testbench device current retain their existing policy.
    """
    counts = Counter(token.upper() for line in deck.splitlines()
                     if not line.lstrip().startswith('*') for token in line.split())
    physical = {name.upper(): value for name, value in states.items()}
    existing = {name.upper(): value for name, value in read_initial_conditions(deck).items()}
    aliases = {}
    for line in deck.splitlines():
        tokens = line.split()
        if not tokens or not tokens[0].upper().startswith('RPHYSICAL_PROBE_'):
            continue
        name, terminal = tokens[1:3]
        if counts[name.upper()] != 1 or terminal.upper() not in physical:
            continue
        value = physical[terminal.upper()]
        if name.upper() in existing and existing[name.upper()] != value:
            raise ValueError('Observation alias initial state differs from its physical terminal')
        aliases[name] = value
    return aliases


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
    vdd = comparison_vdd(text, profile['scope'])
    clock = re.search(r'(?im)^VCLK\s+clk\s+VSS.*?PULSE\(([^)]+)\)', text)
    if clock is None:
        raise ValueError('Complete macro diagnostic requires a native periodic clock')
    period = spice_value(clock[1].split()[-1].rstrip('Ss'))
    if period <= 0:
        raise ValueError('Clock period must be positive')
    stop = 1e-9 + 4.2*period - 5e-12
    conditions = read_initial_conditions(text)
    text = '\n'.join(line for line in text.splitlines()
                     if not line.lower().startswith(('.ic ', '.measure ', '.meas '))) + '\n'
    analysis = ('.TRAN 5e-12 38.795e-9 0 20e-12 UIC' if abs(period-9e-9)<1e-18
                else f'.TRAN 5e-12 {stop:.15g} 0 20e-12 UIC')
    text, count = re.subn(r'(?im)^\.TRAN .*$', analysis, text)
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
                      rows=profile['scope']['rows'], cols=profile['scope']['cols'],
                      period_s=period, stop_s=stop, output_step_s=5e-12,
                      vdd=vdd,
                      maximum_integration_step_s=20e-12,
                      initialization='common complete schematic logic state, UIC; released throughout transient',
                      full_release_screen=False)
    for kind, network in [('estimated', profile), ('post', full)]:
        directory = output_dir / kind
        directory.mkdir()
        deck = apply_to_deck(text, network)
        states = initial_state(network, conditions, vdd=vdd)
        aliases = probe_alias_states(deck, states)
        states.update(aliases)
        entries = '\n'.join(f'.IC V({node})={value:.15g}' for node, value in states.items())
        deck = deck.replace('\n.END', '\n' + entries + '\n.END')
        (directory / 'deck.sp').write_text(deck)
        (directory / 'provenance.json').write_text(json.dumps(
            dict(provenance, model=kind, deck_sha256=sha256(directory/'deck.sp'),
                 unloaded_probe_alias_initial_conditions=len(aliases)),
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
