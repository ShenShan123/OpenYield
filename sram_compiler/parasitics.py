"""Read flat Calibre DSPF networks without discarding coupling or pin stubs.

This module does not infer a calibrated wire from a net's total R/C. Pin
locations, layout coverage and transient comparisons are separate inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
import re


_SUFFIXES = {'t': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3, 'm': 1e-3,
             'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15}


def spice_value(value):
    """Convert a numeric SPICE value, including suffixes and D exponents."""
    match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)'
                         r'(meg|[tgkmunpf])?(?:ohm|farad|f)?', value, re.I)
    if not match:
        raise ValueError(f'Unsupported numeric SPICE value: {value}')
    number, suffix = match.groups()
    result = float(number.lower().replace('d', 'e')) * _SUFFIXES.get((suffix or '').lower(), 1.)
    if not isfinite(result) or result < 0:
        raise ValueError(f'Parasitic value must be finite and nonnegative: {value}')
    return result


def logical_lines(text):
    """Join both SPICE '+' statements and DSPF '*+' annotation continuations."""
    result = []
    for raw in text.splitlines():
        line = raw.strip()
        prefix = '*+' if line.startswith('*+') else '+' if line.startswith('+') else None
        if prefix:
            if not result:
                raise ValueError('Continuation without a preceding statement')
            result[-1] += ' ' + line[len(prefix):].strip()
        elif line:
            result.append(line)
    return result


@dataclass
class DspfNet:
    name: str
    total_capacitance_f: float
    ports: list[str] = field(default_factory=list)
    # (node name, source instance, terminal)
    pins: list[tuple[str, str, str]] = field(default_factory=list)
    resistors: list[tuple[str, str, float]] = field(default_factory=list)
    capacitors: list[tuple[str, str, float]] = field(default_factory=list)

    def effective_resistances(self, source, targets):
        """DC driving-point resistance from source to each terminal.

        Solve the conductance network, including parallel paths. A shortest
        resistor path is not the electrical resistance of a mesh. SciPy is
        loaded only for this optional analysis, not for deck generation.
        Missing/disconnected terminals fail instead of disappearing from a fit.
        """
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu

        if any(r <= 0 for _, _, r in self.resistors):
            raise ValueError(f'{self.name}: zero-ohm edges must be merged before solving')
        graph = {}
        for a, b, _ in self.resistors:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        seen, pending = {source}, [source]
        while pending:
            for node in graph.get(pending.pop(), ()):
                if node not in seen:
                    seen.add(node)
                    pending.append(node)
        missing = set(targets) - seen
        if missing:
            raise ValueError(f'{self.name}: disconnected terminals: {sorted(missing)}')
        index = {node: i for i, node in enumerate(sorted(seen - {source}))}
        if not index:
            return {node: 0. for node in targets}
        rows, cols, values = [], [], []
        for a, b, resistance in self.resistors:
            if a not in seen or a == b:
                continue
            conductance = 1. / resistance
            for node, other in ((a, b), (b, a)):
                if node == source:
                    continue
                i = index[node]
                rows.append(i); cols.append(i); values.append(conductance)
                if other != source:
                    rows.append(i); cols.append(index[other]); values.append(-conductance)
        matrix = coo_matrix((values, (rows, cols)), shape=(len(index), len(index))).tocsc()
        solver = splu(matrix)
        result = {}
        for node in dict.fromkeys(targets):
            if node == source:
                result[node] = 0.
                continue
            excitation = np.zeros(len(index))
            excitation[index[node]] = 1.
            result[node] = float(solver.solve(excitation)[index[node]])
            if not isfinite(result[node]) or result[node] < 0:
                raise ValueError(f'{self.name}: invalid conductance solution for {node}')
        return result


@dataclass
class Dspf:
    name: str
    ports: tuple[str, ...]
    ground: str
    nets: dict[str, DspfNet]
    statements: tuple[str, ...]

    def capacitance_inventory(self):
        """Classify actual capacitor edges, counting each edge once.

        NET totals are extractor annotations and may use a coupling convention;
        retain them separately rather than treating them as shunt capacitance.
        """
        shunt = coupling = 0.
        for net in self.nets.values():
            for a, b, value in net.capacitors:
                if self.ground in (a, b):
                    shunt += value
                else:
                    coupling += value
        return dict(shunt_f=shunt, coupling_f=coupling,
                    annotated_total_f=sum(n.total_capacitance_f for n in self.nets.values()))


def read_dspf(path):
    """Read one flat, explicitly valued DSPF subcircuit; reject unsupported decks."""
    lines = logical_lines(Path(path).read_text())
    name = ground = None
    ports = ()
    nets, current, names = {}, None, set()
    closed = False
    for line in lines:
        tokens = line.split()
        key = tokens[0].lower()
        if key == '.subckt':
            if name is not None:
                raise ValueError('Only one flat DSPF subcircuit is supported')
            name, ports = tokens[1], tuple(tokens[2:])
        elif key == '.ends':
            closed = True
        elif key == '*|ground_net':
            ground = tokens[1]
        elif key == '*|net':
            net_name = tokens[1]
            if net_name in nets:
                raise ValueError(f'Duplicate DSPF net: {net_name}')
            current = DspfNet(net_name, spice_value(tokens[2]))
            nets[net_name] = current
        elif key in ('*|p', '*|i'):
            if current is None:
                raise ValueError('DSPF pin annotation without a net')
            annotation = line.split('(', 1)[1].rsplit(')', 1)[0].split()
            if key == '*|p':
                current.ports.append(annotation[0])
            else:
                current.pins.append(tuple(annotation[:3]))
        elif not key.startswith(('*', '.')):
            if name is None or closed:
                raise ValueError('Element outside DSPF subcircuit')
            if key in names:
                raise ValueError(f'Duplicate DSPF element: {tokens[0]}')
            names.add(key)
            if key[0] in 'rc':
                if current is None or len(tokens) != 4:
                    raise ValueError(f'Unsupported parasitic statement: {line}')
                item = (tokens[1], tokens[2], spice_value(tokens[3]))
                getattr(current, 'resistors' if key[0] == 'r' else 'capacitors').append(item)
            elif key[0] != 'm':
                raise ValueError(f'Unsupported DSPF element: {tokens[0]}')
    if name is None or ground is None or not nets or not closed:
        raise ValueError('Incomplete DSPF: subcircuit, ground, nets and .ends required')
    return Dspf(name, ports, ground, nets, tuple(lines))
