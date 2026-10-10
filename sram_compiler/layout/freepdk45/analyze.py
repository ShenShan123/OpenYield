"""Fit the array wire slope using actual mirrored contact locations.

The YAML is a diagnostic reduction, not a defaults promotion. Capacitances
include coupling to quiet neighbours; dynamic coupling and pin stubs remain
in the coupled DSPF reference and require independent waveform comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

from sram_compiler.parasitics import read_dspf
from .run import sha256
from . import bitcell


def fit(samples):
    """Least-squares R = stub + pitch_index * R_per_pitch."""
    xmean = statistics.mean(x for x, _ in samples)
    ymean = statistics.mean(y for _, y in samples)
    denominator = sum((x - xmean) ** 2 for x, _ in samples)
    if not denominator:
        raise ValueError('At least two distinct physical tap positions are required')
    slope = sum((x - xmean) * (y - ymean) for x, y in samples) / denominator
    intercept = ymean - slope * xmean
    residual = max(abs(y - intercept - slope * x) for x, y in samples)
    if slope <= 0 or intercept < 0:
        raise ValueError('Nonphysical wire fit; inspect the terminal map')
    return dict(resistance_per_pitch_ohm=slope, port_and_pin_stub_ohm=intercept,
                max_residual_ohm=residual)


def analyze_array(dspf):
    result = {}
    for kind, names, pitch in (
        ('wl', [f'WL{i}' for i in range(8)], bitcell.W * 1e-9),
        ('bl', [f'{polarity}{i}' for polarity in ('BL', 'BLB') for i in range(8)], bitcell.H * 1e-9),
    ):
        records = []
        for name in names:
            net = dspf.nets[name]
            if len(net.ports) != 1:
                raise ValueError(f'{name}: exactly one physical source port required')
            positions = []
            for node, instance, terminal in net.pins:
                match = re.fullmatch(r'XSRAM_6T_CELL_(\d+)_(\d+)/MPG([LR])', instance)
                if not match:
                    continue
                row, col, side = match.groups()
                row, col = int(row), int(col)
                if kind == 'wl' and terminal == 'g':
                    # PG_L contacts at the left column boundary; PG_R at the
                    # right boundary. Odd columns mirror these positions.
                    position = col + (int(side == 'R') ^ (col % 2))
                elif kind == 'bl' and terminal == 'd':
                    # PG_L drain is on the upper cell boundary; PG_R is on the
                    # lower. Odd rows mirror. BL spans 6 contact pitches;
                    # BLB spans 8, so one hard-coded span would bias the fit.
                    position = row + (int(side == 'L') ^ (row % 2))
                else:
                    continue
                positions.append((node, position))
            expected = 16 if kind == 'wl' else 8
            if len(positions) != expected:
                raise ValueError(f'{name}: {len(positions)} cell terminals, expected {expected}')
            resistance = net.effective_resistances(net.ports[0], [n for n, _ in positions])
            samples = [(position, resistance[node]) for node, position in positions]
            records.append(dict(net=name, capacitance_per_pitch_f=net.total_capacitance_f / 8,
                                physical_span_pitches=max(x for x, _ in samples) - min(x for x, _ in samples),
                                fit=fit(samples)))
        resistance = statistics.mean(n['fit']['resistance_per_pitch_ohm'] for n in records)
        capacitance = statistics.mean(n['capacitance_per_pitch_f'] for n in records)
        result[kind] = dict(pitch_m=pitch, width_m=70e-9,
                            resistance_per_pitch_ohm=resistance,
                            capacitance_per_pitch_f=capacitance,
                            capacitance_per_pitch_min_f=min(n['capacitance_per_pitch_f'] for n in records),
                            capacitance_per_pitch_max_f=max(n['capacitance_per_pitch_f'] for n in records),
                            effective_sheet_resistance_ohm=resistance * 70e-9 / pitch,
                            capacitance_f_per_m=capacitance / pitch,
                            nets=records)
    return result


def verified_dspf(directory, scope):
    directory = Path(directory)
    receipt = json.loads((directory / 'receipt.json').read_text())
    path = directory / 'xrc/macro.dspf'
    if (receipt.get('scope') != scope or receipt.get('status') != 'macro_extracted'
            or receipt.get('calibre_drc_violations') != 0 or receipt.get('calibre_lvs') != 'CORRECT'
            or receipt.get('extraction') != 'distributed R+C+CC'):
        raise ValueError(f'{directory}: a successful coupled {scope} extraction is required')
    if sha256(path) != receipt.get('dspf_sha256'):
        raise ValueError(f'{directory}: DSPF hash differs from the extraction receipt')
    if sha256(directory / 'schematic.sp') != receipt.get('schematic_sha256'):
        raise ValueError(f'{directory}: schematic hash differs from the extraction receipt')
    gds = directory / ('array.gds' if scope == 'array' else 'sram8x8.gds')
    if sha256(gds) != receipt.get('layout_sha256'):
        raise ValueError(f'{directory}: GDS hash differs from the extraction receipt')
    return read_dspf(path), receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--array-dir', type=Path, required=True)
    parser.add_argument('--macro-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    array, array_receipt = verified_dspf(args.array_dir, 'array')
    macro, macro_receipt = verified_dspf(args.macro_dir, 'macro')
    if (array_receipt['source_deck_sha256'] != macro_receipt['source_deck_sha256']
            or array_receipt['rule_sha256'] != macro_receipt['rule_sha256']):
        raise ValueError('Array and macro must use the same schematic source and PDK rules')
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; previous analysis evidence is preserved')
    output.mkdir(parents=True, exist_ok=True)
    fit_result = analyze_array(array)
    report = dict(status='diagnostic_candidate', defaults_promotable=False,
                  scope='FreePDK45 8x8 6T, nominal, mode 0, no TIME_CONTROL',
                  capacitance_policy='NET incident total / 8; quiet-aggressor coupling approximation',
                  resistance_policy='conductance solve, then R versus physical contact position fit',
                  array_receipt=array_receipt, macro_receipt=macro_receipt,
                  array=fit_result, macro_capacitance=macro.capacitance_inventory(),
                  controller_layout=False, clock_wire_calibrated=False,
                  local_pin_stubs_calibrated=False, waveform_validation='not_run',
                  uncovered=['TIME_CONTROL layout and intentional passives', 'DATA_REGISTER CLK',
                             '10T geometry', 'column-mux geometry', 'array-size scaling',
                             'dynamic coupling and local pin-stub back-annotation'])
    # Record every extracted network; the WL/BL reduction cannot calibrate all
    # these physically different routes by assigning them the same wire class.
    report['macro_nets'] = {name: dict(capacitance_f=net.total_capacitance_f,
                                     resistors=len(net.resistors), terminals=len(net.pins))
                            for name, net in macro.nets.items()}
    report['macro_port_networks'] = {}
    for name in ('PRE', 'W_EN', 'S_EN', 'SA_ISO', 'WL_EN', 'A0', 'A1', 'A2', 'RBL', 'RWL'):
        net = macro.nets[name]
        if len(net.ports) != 1 or not net.pins:
            raise ValueError(f'{name}: expected one port and physical device terminals')
        resistance = net.effective_resistances(net.ports[0], [node for node, _, _ in net.pins])
        near = min(resistance, key=resistance.get)
        far = max(resistance, key=resistance.get)
        report['macro_port_networks'][name] = dict(
            capacitance_f=net.total_capacitance_f, source=net.ports[0],
            nearest_terminal=near, nearest_terminal_resistance_ohm=resistance[near],
            farthest_terminal=far, farthest_terminal_resistance_ohm=resistance[far],
            interpretation='port-to-terminal effective R includes access metal/contact/poly; not wire R per pitch')
    (output / 'analysis.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    import yaml
    profile = dict(mode='distributed', cell_pin_rc=False)
    for kind, layer in [('wl', 'M2-FreePDK45-8x8-effective'), ('bl', 'M3-FreePDK45-8x8-effective')]:
        row = fit_result[kind]
        profile[kind] = dict(layer=layer, pitch_m=row['pitch_m'], width_m=row['width_m'],
                             sheet_resistance_ohm=row['effective_sheet_resistance_ohm'],
                             capacitance_f_per_m=row['capacitance_f_per_m'], sections_per_half_pitch=1)
    header = ('# Diagnostic FreePDK45 8x8 6T array reduction from Calibre R+C+CC.\n'
              '# Quiet-aggressor C includes coupling; port/pin stub R is excluded.\n'
              '# Not an all-route calibration: the compiler reuses WL/BL for\n'
              '# controls, decoder, periphery and clock. Do not promote defaults.\n'
              '# Evidence: analysis.json in this directory.\n')
    (output / 'interconnect_candidate.yaml').write_text(header + yaml.safe_dump(profile, sort_keys=False))
    print(output / 'analysis.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
