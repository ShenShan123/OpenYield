"""Flatten the complete nominal macro while preserving intentional passives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .export import Nets, parse, instance_parts, WIRE_R, WIRE_C
from sram_compiler.parasitics import spice_value


@dataclass
class FullSource:
    ports: list
    mos: list
    passives: list
    aliases: dict
    instances: dict


def flatten(deck):
    root = parse(Path(deck).read_text().splitlines())
    nets, devices, passives, aliases, instances = Nets(), [], [], {}, {}

    def visit(scope, connections, path):
        def net(name):
            value = connections.get(name, path + '/' + name if path else name)
            return value.replace(':', '/')

        for formal, actual in connections.items():
            aliases[path + '/' + formal] = actual.replace(':', '/')
            nets.union(path + '/' + formal, actual.replace(':', '/'))
        for tokens in scope.lines:
            kind = tokens[0][0].upper()
            if kind == 'X':
                pins, ref = instance_parts(tokens)
                child = scope.resolve(ref)
                childpath = path + '/' + tokens[0] if path else tokens[0]
                mapping = dict(zip(child.pins, [net(n) for n in pins]))
                instances[childpath] = dict(subcircuit=ref, pins=mapping)
                visit(child, mapping, childpath)
            elif kind == 'M':
                if not path:
                    # Testbench startup clamps are stimulus infrastructure.
                    if tokens[0].startswith('MCSINIT_'):
                        continue
                    raise ValueError(f'Unexpected top-level MOS: {tokens[0]}')
                params = dict(p.lower().split('=', 1) for p in tokens[6:])
                name = path + '/' + tokens[0]
                device = dict(name=name, nodes=[net(n) for n in tokens[1:5]], model=tokens[5],
                              width_m=spice_value(params['w']), length_m=spice_value(params['l']),
                              fingers=int(params.get('nf', 1)))
                devices.append(device)
            elif kind == 'R':
                a, b = net(tokens[1]), net(tokens[2])
                if WIRE_R.match(tokens[0]):
                    nets.union(a, b)
                else:
                    passives.append(dict(name=path + '/' + tokens[0], kind='R', nodes=[a, b],
                                         value=spice_value(tokens[3])))
            elif kind == 'C':
                if not WIRE_C.match(tokens[0]):
                    passives.append(dict(name=path + '/' + tokens[0], kind='C',
                                         nodes=[net(tokens[1]), net(tokens[2])], value=spice_value(tokens[3])))
            elif not path:
                # All independent/controlled sources belong to the testbench.
                continue
            else:
                raise ValueError(f'Unsupported element in macro: {tokens[0]}')

    visit(root, {}, '')
    ports = ['VDD', 'VSS', 'clk', 'csb', 'web', 'A0', 'A1', 'A2']
    ports += [f'DIN{i}' for i in range(8)] + ['OUT', 'OUT_B']
    ports += [f'SA_Q{i}' for i in range(8)] + [f'SA_QB{i}' for i in range(8)]
    for device in devices:
        for n in device['nodes']:
            nets.find(n)
    for item in passives:
        for n in item['nodes']:
            nets.find(n)
    canonical = {}
    for group in nets.groups().values():
        # An external pin must remain externally addressable. Otherwise prefer
        # the source's short global name over arbitrary union-find roots.
        preferred = [n for n in group if n in ports]
        rep = preferred[0] if preferred else min(group, key=lambda n: (n.count('/'), len(n), n))
        for n in group:
            canonical[n] = rep
    for device in devices:
        device['original_nodes'] = list(device['nodes'])
        device['nodes'] = [canonical[n] for n in device['nodes']]
    for item in passives:
        item['nodes'] = [canonical[n] for n in item['nodes']]
    aliases = dict(canonical)
    for item in instances.values():
        item['pins'] = {k: canonical.get(v, v) for k, v in item['pins'].items()}
    return FullSource(ports, devices, passives, aliases, instances)


def render(source, name='OPENYIELD_8X8', filter_passives=False):
    lines = [f'.subckt {name} ' + ' '.join(source.ports)]
    for device in source.mos:
        lines.append('M' + device['name'] + ' ' + ' '.join(device['nodes']) +
                     f' {device["model"]} W={device["width_m"]:.15g} L={device["length_m"]:.15g}' +
                     (f' NF={device["fingers"]}' if device['fingers'] > 1 else ''))
    for item in source.passives:
        if not filter_passives:
            lines.append(item['kind'] + item['name'] + ' ' + ' '.join(item['nodes']) + f' {item["value"]:.15g}')
    lines += [f'.ends {name}', '']
    return '\n'.join(lines)
