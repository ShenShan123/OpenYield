"""Measure driver loads in a verified bank DSPF; no default sizing promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from sram_compiler.parasitics import read_dspf
from .dense_views import expand


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(layout_dir: Path, extraction_dir: Path) -> dict[str, Any]:
    """Include coupling incident on either net and actual gate widths once.

    Capacitance is extracted metal, separate from bias-dependent MOS input
    charge. Resistance uses a conductance solve, including parallel paths.
    Historical receipts certify their artifacts, irrespective of newer code.
    """
    manifest = json.loads((layout_dir / 'layout.json').read_text())
    receipt = json.loads((extraction_dir / 'receipt.json').read_text())
    if (receipt.get('status') != 'extracted' or receipt.get('drc_violations') != 0
            or receipt.get('lvs') != 'CORRECT'
            or receipt.get('extraction') != 'distributed R+C+CC'):
        raise ValueError('Driver audit requires passing DRC, LVS and coupled extraction')
    paths = {'layout_sha256': layout_dir / 'full_macro.gds',
             'source_sha256': layout_dir / 'source.sp',
             'dspf_sha256': extraction_dir / 'xrc/full_macro.dspf'}
    for key, path in paths.items():
        if digest(path) != receipt.get(key):
            raise ValueError(f'{key}: artifact differs from its physical receipt')
        if key != 'dspf_sha256' and digest(path) != manifest.get(key):
            raise ValueError(f'{key}: artifact differs from its layout manifest')
    dspf = read_dspf(paths['dspf_sha256'])
    pin_locations, fingers = expand(manifest)
    devices = {(f['device'] + (f"@{f['finger'] + 1}" if f['finger'] else '')).upper(): f
               for f in fingers}
    owner: dict[str, str] = {}
    for name, net in dspf.nets.items():
        for node in net.ports + [p[0] for p in net.pins]:
            owner[node] = name
        for a, b, _ in net.resistors:
            owner[a] = owner[b] = name
    capacitance = {name: [0., 0.] for name in dspf.nets}
    for net in dspf.nets.values():
        for a, b, value in net.capacitors:
            na, nb = owner.get(a, a), owner.get(b, b)
            # Supply coupling is a quiet-rail load. Internal capacitance
            # between nodes of one net does not load its common-mode edge.
            if na == nb:
                continue
            for name, other in ((na, nb), (nb, na)):
                if name in capacitance:
                    capacitance[name][0 if other.upper() in ('0', 'VSS', 'VDD') else 1] += value
    records = []
    for name, net in dspf.nets.items():
        short = name.rsplit('/', 1)[-1].upper()
        if short in ('PRE', 'W_EN', 'S_EN', 'SA_ISO', 'WL_EN', 'CLK_BUF', 'RBL', 'RWL'):
            family = short.lower()
        elif re.fullmatch(r'(WL|BLB?|DEC_WL|A_DFF|DIN_DFF|DIN_BUF)\d+', short):
            family = re.sub(r'\d+$', '', short).lower()
        else:
            continue
        gate_width = {'nmos': 0., 'pmos': 0.}
        gates: list[str] = []
        drivers: list[str] = []
        for node, instance, terminal in net.pins:
            device = devices.get(instance.upper())
            if device is None:
                raise ValueError(f'{instance}: extracted MOS has no source finger')
            if terminal.lower() == 'g':
                gate_width['pmos' if device['model'].upper().startswith('P') else 'nmos'] += device['width_nm']
                gates.append(node)
            elif terminal.lower() == 'd':
                # Storage/sense nodes are loads, not drivers of the global bus.
                if any(role in instance.upper() for role in
                       ('XWL_DRV_', 'XWRITEDRIVER_', 'XPRECHARGE_', 'XDIN_BUF_',
                        'XTIME_CONTROL/', 'XDECODER/', 'XRWL/')):
                    drivers.append(node)
        cell_pins = [(node, pin_locations[f"{devices[instance.upper()]['device']}@{devices[instance.upper()]['finger']}:{terminal.lower()}"][:2])
                     for node, instance, terminal in net.pins
                     if '/XSRAM_6T_CELL_' in instance.upper()
                     and terminal.lower() == ('g' if family == 'wl' else 'd')
                     and family in ('wl', 'bl', 'blb')]
        targets = list(dict.fromkeys(gates + [n for n, _ in cell_pins]))
        origin = drivers[0] if drivers else (net.ports[0] if net.ports else None)
        resistance = None
        wire_fit = None
        if origin is not None and targets:
            solved = net.effective_resistances(origin, targets)
            resistance = max(solved.values())
            if cell_pins:
                samples = [(point[0 if family == 'wl' else 1], solved[node]) for node, point in cell_pins]
                xm = sum(x for x, _ in samples) / len(samples)
                ym = sum(y for _, y in samples) / len(samples)
                denom = sum((x-xm)**2 for x, _ in samples)
                slope = sum((x-xm)*(y-ym) for x, y in samples) / denom
                wire_fit = dict(resistance_ohm_per_um=slope*1000.,
                                maximum_residual_ohm=max(abs(y-ym-slope*(x-xm)) for x, y in samples))
        quiet, coupling = capacitance[name]
        records.append(dict(net=name, family=family, gate_width_nm=gate_width,
                            gate_terminals=len(targets), driver_terminal=origin,
                            far_gate_resistance_ohm=resistance,
                            quiet_rail_capacitance_f=quiet, coupling_capacitance_f=coupling,
                            quiet_neighbor_load_f=quiet + coupling,
                            opposite_switching_load_f=quiet + 2 * coupling,
                            annotated_capacitance_f=net.total_capacitance_f))
        records[-1]['wire_fit'] = wire_fit
    return dict(schema=1, rows=manifest['rows'], cols=manifest['cols'],
                physical_receipt_sha256=digest(extraction_dir / 'receipt.json'),
                **{k: digest(p) for k, p in paths.items()},
                model_and_sizing=manifest.get('compiler_provenance'),
                method='coupled capacitor incidence and terminal conductance solve',
                mos_capacitance='not included; requires model input-charge characterization',
                scaling='measured size only; larger arrays require new extraction',
                defaults_promotable=False, drivers=records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout-dir', type=Path, required=True)
    parser.add_argument('--extraction-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.layout_dir, args.extraction_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'driver_loads.json').write_text(json.dumps(result, indent=2) + '\n')
    print(args.output_dir / 'driver_loads.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
