"""Compare macro waveforms with ideal controls and the exact coupled DSPF.

TIME_CONTROL is outside this experiment. A pass verifies only this slow 8x8
nominal sequence; it is not a compiler timing screen or yield evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess

from .analyze import verified_dspf
from .export import MACRO_INSTANCES, flat_devices, parse, instance_parts
from .run import sha256

SEQUENCE = [('write', 5, 0xA5), ('write', 2, 0x5A), ('read', 5, 0xA5),
            ('read', 2, 0x5A), ('write', 5, 0x0F), ('read', 5, 0x0F), ('read', 2, 0x5A)]
SLOT = 4e-9
SLEW = .05e-9


def stimulus():
    steps = {p: [(0., 0.)] for p in ['A0', 'A1', 'A2', 'WL_EN', 'PRE', 'W_EN', 'S_EN', 'SA_ISO']
             + [f'DIN{i}' for i in range(8)]}
    for k, (op, row, byte) in enumerate(SEQUENCE):
        t = (k + 1) * SLOT
        for bit in range(3):
            steps[f'A{bit}'].append((t, float(row >> bit & 1)))
        steps['PRE'] += [(t + .2e-9, 1.), (t + 3.4e-9, 0.)]
        steps['WL_EN'] += [(t + .6e-9, 1.), (t + (2.6e-9 if op == 'write' else 1.8e-9), 0.)]
        if op == 'write':
            for bit in range(8):
                steps[f'DIN{bit}'].append((t, float(byte >> bit & 1)))
            steps['W_EN'] += [(t + .4e-9, 1.), (t + 2.8e-9, 0.)]
        else:
            steps['SA_ISO'] += [(t + 1.6e-9, 1.), (t + 3.2e-9, 0.)]
            steps['S_EN'] += [(t + 1.7e-9, 1.), (t + 3.2e-9, 0.)]
    result = {}
    for pin, points in steps.items():
        values = ['0 0']
        current = 0.
        for time, value in sorted(points):
            if value != current:
                values += [f'{time:.12g} {current:g}', f'{time + SLEW:.12g} {value:g}']
                current = value
        result[pin] = 'PWL(' + ' '.join(values) + ')'
    return result


def flatten_pre_layout(deck, output):
    """Retain every macro R/C and MOS in the actual pre-layout compiler deck."""
    root = parse(Path(deck).read_text().splitlines())
    ports = ['VDD', 'VSS', 'A0', 'A1', 'A2', 'WL_EN', 'PRE', 'S_EN', 'SA_ISO', 'W_EN']
    ports += [f'{p}{i}' for p in ('DIN', 'Q', 'QB') for i in range(8)] + ['RBL', 'RWL']
    rename = dict(wl_en='WL_EN', w_en='W_EN', s_en='S_EN', sa_iso='SA_ISO')
    rename.update({f'{old}{i}': f'{new}{i}' for old, new in
                   [('A_dff', 'A'), ('DIN_dff', 'DIN'), ('SA_Q', 'Q'), ('SA_QB', 'QB')] for i in range(8)})
    lines = [f'.subckt SRAM_8X8 {" ".join(ports)}']
    pins_by_instance = {}

    def emit(scope, tokens, connection, path):
        key = tokens[0][0].upper()

        def net(name):
            return connection.get(name, path + '/' + name if path else rename.get(name, name))

        if key == 'X':
            pins, ref = instance_parts(tokens)
            child = scope.resolve(ref)
            childpath = path + '/' + tokens[0] if path else tokens[0]
            mapping = {p: net(n) for p, n in zip(child.pins, pins)}
            for line in child.lines:
                emit(child, line, mapping, childpath)
        elif key in ('M', 'R', 'C'):
            count = 4 if key == 'M' else 2
            actual = [net(n) for n in tokens[1:1 + count]]
            name = key + (path + '/' if path else '') + tokens[0]
            lines.append(' '.join([name, *actual, *tokens[1 + count:]]))
            if key == 'M':
                pins_by_instance[path + '/' + tokens[0]] = dict(zip('dgsb', actual))
        else:
            raise ValueError(f'Unsupported macro element: {tokens[0]}')

    for tokens in root.lines:
        if any(re.fullmatch(pattern, tokens[0]) for pattern in MACRO_INSTANCES) or tokens[0].startswith(('Rwire_', 'Cwire_')):
            emit(root, tokens, {}, '')
    lines += ['.ends SRAM_8X8', '']
    Path(output).write_text('\n'.join(lines))
    return ports, pins_by_instance


def probes(pins):
    # The physical cell terminal, not a renamed lumped WL or Q net.
    result = {}
    for row in range(8):
        result[f'wl{row}'] = pins[f'XSRAM_6T_CORE_8x8/XSRAM_6T_CELL_{row}_7/MPGL']['g']
    for row in (2, 5):
        for col in range(8):
            result[f'cell{row}_{col}'] = pins[f'XSRAM_6T_CORE_8x8/XSRAM_6T_CELL_{row}_{col}/MPDL']['d']
    for col in range(8):
        for side, device in [('bl', 'MPGL'), ('blb', 'MPGR')]:
            result[f'{side}{col}'] = pins[f'XSRAM_6T_CORE_8x8/XSRAM_6T_CELL_7_{col}/{device}']['d']
    return result


def evaluate(records):
    failures, checks = [], 0

    def check(ok, description):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(description)

    def sample(time):
        return min(records, key=lambda r: abs(r['TIME'] - time))

    written = {}
    for slot, (op, address, byte) in enumerate(SEQUENCE):
        start = (slot + 1) * SLOT
        active = sample(start + 1.4e-9)
        for row in range(8):
            voltage = active[f'wl{row}']
            check(voltage > .9 if row == address else voltage < .1,
                  f'slot {slot}: WL{row} selection ({voltage:g} V)')
        if op == 'write':
            written[address] = byte
        if op == 'read':
            sensed = sample(start + 3e-9)
            for col in range(8):
                voltage = sensed[f'out{col}']
                check(voltage > .9 if byte >> col & 1 else voltage < .1,
                      f'slot {slot}: sense Q{col} ({voltage:g} V)')
        end = sample(start + 3.9e-9)
        for row, expected in written.items():
            for col in range(8):
                voltage = end[f'cell{row}_{col}']
                check(voltage > .9 if expected >> col & 1 else voltage < .1,
                      f'slot {slot}: retained cell {row},{col} ({voltage:g} V)')
        for row in range(8):
            check(end[f'wl{row}'] < .1, f'slot {slot}: WL{row} recovery')
        for col in range(8):
            for side in ('bl', 'blb'):
                check(end[f'{side}{col}'] > .9, f'slot {slot}: {side}{col} precharge recovery')
    return dict(passed=not failures, checks=checks, failures=failures)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--macro-dir', type=Path, required=True)
    parser.add_argument('--pre-layout-deck', type=Path, required=True)
    parser.add_argument('--candidate-deck', type=Path)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--xyce', default='Xyce')
    parser.add_argument('--timeout', type=float, default=2400., help='seconds per solver run')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    for path in [args.pre_layout_deck, args.models, args.candidate_deck]:
        if path is not None and not path.is_file():
            parser.error(f'Input file not found: {path}')
    if args.candidate_deck:
        inventories = []
        for path in (args.pre_layout_deck, args.candidate_deck):
            tree = parse(path.read_text().splitlines())
            selected = [line for line in tree.lines
                        if any(re.fullmatch(pattern, line[0]) for pattern in MACRO_INSTANCES)]
            inventories.append(flat_devices(selected, tree))
        if inventories[0] != inventories[1]:
            raise ValueError('Candidate changes MOS widths, lengths, models or device counts')
    post, receipt = verified_dspf(args.macro_dir, 'macro')
    if sha256(args.pre_layout_deck) != receipt['source_deck_sha256']:
        raise ValueError('Pre-layout deck differs from the extracted layout source')
    xyce = shutil.which(args.xyce)
    if xyce is None:
        parser.error('Xyce executable not found')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; previous waveforms are preserved')
    output.mkdir(parents=True, exist_ok=True)
    modelpath = args.models.resolve()
    audit = dict(scope='nominal 8x8 macro, ideal 4ns controls, no TIME_CONTROL',
                 full_compiler_screen=False, defaults_promotable=False,
                 xyce_sha256=sha256(xyce), model_sha256=sha256(modelpath),
                 extraction_receipt=receipt, results={},
                 solver='KLU', initialization='DC NODESET Q=0,QB=1 for all 64 cells; released for transient')
    variants = [('pre_layout', args.pre_layout_deck), ('post_layout', None)]
    if args.candidate_deck:
        variants.append(('candidate', args.candidate_deck))
    for label, source in variants:
        directory = output / label
        directory.mkdir()
        netlist = directory / 'macro.cir'
        if source is not None:
            ports, pins = flatten_pre_layout(source, netlist)
        else:
            shutil.copyfile(args.macro_dir / 'xrc/macro.dspf', netlist)
            ports = post.ports
            pins = {}
            for net in post.nets.values():
                for node, instance, terminal in net.pins:
                    pins.setdefault(instance, {})[terminal] = node
        channels = probes(pins)
        channels.update({f'out{i}': None for i in range(8)})
        expressions = [f'V(XDUT:{node})' if node is not None else f'V(Q{key[3:]})'
                       for key, node in channels.items()]
        nodesets = []
        for row in range(8):
            for col in range(8):
                prefix = f'XSRAM_6T_CORE_8x8/XSRAM_6T_CELL_{row}_{col}'
                for device, value in [('MPDL', 0), ('MPDR', 1)]:
                    nodesets.append(f'.NODESET V(XDUT:{pins[prefix + "/" + device]["d"]})={value}')
        deck = [f'* {label}: macro waveform comparison', f'.include "{modelpath}"',
                f'.include "{netlist}"', 'VVDD VDD 0 1', 'VVSS VSS 0 0',
                *[f'V{p} {p} 0 {src}' for p, src in stimulus().items()],
                f'XDUT {" ".join(ports)} SRAM_8X8', '.options DEVICE TEMP=25',
                '.OPTIONS LINSOL TYPE=KLU', *nodesets,
                '.TRAN 5p 32n 0 5p', '.PRINT TRAN FORMAT=CSV ' + ' '.join(expressions), '.end', '']
        path = directory / 'tb.cir'
        path.write_text('\n'.join(deck))
        with (directory / 'xyce.log').open('w') as stream:
            root = Path(__file__).resolve().parents[3]
            try:
                result = subprocess.run([xyce, str(path)], cwd=root, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                audit['results'][label] = dict(passed=False, failures=['Xyce timeout; see xyce.log'],
                                               deck_sha256=sha256(path), timeout_seconds=args.timeout)
                continue
        audit['results'][label] = dict(deck_sha256=sha256(path), source_deck_sha256=sha256(source) if source else None,
                                      solver_exit_code=result.returncode)
        if result.returncode:
            audit['results'][label].update(passed=False, failures=['Xyce failed; see xyce.log'])
            continue
        with Path(str(path) + '.csv').open() as stream:
            reader = csv.reader(stream)
            next(reader)
            # Xyce preserves .PRINT expression order; semantic channel names
            # keep source and extracted MOS-terminal probes comparable.
            records = [dict(zip(['TIME', *channels], map(float, row))) for row in reader]
        audit['results'][label].update(evaluate(records))
        sampled = []
        for k in range(len(SEQUENCE)):
            for offset in (1.4e-9, 3e-9, 3.9e-9):
                sampled.append(min(records, key=lambda r: abs(r['TIME'] - (k + 1) * SLOT - offset)))
        (directory / 'sampled_waveforms.json').write_text(json.dumps(sampled, indent=2) + '\n')
    path = output / 'comparison.json'
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    print(path)
    return 0 if all(r.get('passed') for r in audit['results'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
