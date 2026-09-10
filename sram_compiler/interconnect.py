"""Physical array interconnect, independent of transistor sizing policy.

WL drivers sit at the left edge (column zero); all bitline periphery sits at
the row-zero edge. Cell taps sit at half-pitch centers. Distributed geometry
must be supplied explicitly: the compiler has no extracted metal defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping

from PySpice.Unit import u_Ohm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mapping(value):
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, '__dict__'):
        return vars(value).copy()
    raise ValueError('Interconnect settings must be a mapping')


def _positive(name, value):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite and positive')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and positive') from exc
    if not isfinite(result) or result <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return result


def _count(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


@dataclass(frozen=True)
class WireRC:
    layer: str
    pitch_m: float
    width_m: float
    sheet_resistance_ohm: float
    capacitance_f_per_m: float
    sections_per_half_pitch: int = 1

    def __post_init__(self):
        if not isinstance(self.layer, str) or not self.layer.strip():
            raise ValueError('Wire layer must be a nonempty technology identifier')
        for name in ('pitch_m', 'width_m', 'sheet_resistance_ohm', 'capacitance_f_per_m'):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        _count('sections_per_half_pitch', self.sections_per_half_pitch)

    @property
    def resistance_per_pitch(self):
        return self.sheet_resistance_ohm * self.pitch_m / self.width_m

    @property
    def capacitance_per_pitch(self):
        return self.capacitance_f_per_m * self.pitch_m


@dataclass(frozen=True)
class InterconnectConfig:
    mode: str = 'star'
    # None resolves to the documented default: local cell WL/BL/BLB stubs stay
    # with the star topology and are omitted once distributed wires carry them.
    cell_pin_rc: bool | None = None
    wl: WireRC | None = None
    bl: WireRC | None = None

    def __post_init__(self):
        if self.mode not in ('star', 'distributed'):
            raise ValueError('Interconnect mode must be star or distributed')
        if self.cell_pin_rc is None:
            object.__setattr__(self, 'cell_pin_rc', not self.distributed)
        if not isinstance(self.cell_pin_rc, bool):
            raise ValueError('cell_pin_rc must be boolean')
        if self.distributed and not all(isinstance(wire, WireRC) for wire in (self.wl, self.bl)):
            raise ValueError('Distributed interconnect requires wl and bl wire geometry')
        if not self.distributed and (self.wl is not None or self.bl is not None):
            raise ValueError('Wire geometry requires distributed mode')

    @property
    def distributed(self):
        return self.mode == 'distributed'

    def to_dict(self):
        return asdict(self)


def resolve_interconnect(options=None):
    if options is None:
        return InterconnectConfig()
    if isinstance(options, InterconnectConfig):
        return options
    values = _mapping(options)
    unknown = values.keys() - {'mode', 'cell_pin_rc', 'wl', 'bl'}
    if unknown:
        raise ValueError(f'Unknown interconnect options: {sorted(unknown)}')
    for key in ('wl', 'bl'):
        if values.get(key) is not None and not isinstance(values[key], WireRC):
            try:
                values[key] = WireRC(**_mapping(values[key]))
            except TypeError as exc:
                raise ValueError(f'Invalid {key} wire geometry: {exc}') from exc
    return InterconnectConfig(**values)


def load_interconnect(path):
    """Read an interconnect YAML mapping; relative paths resolve from the project root.

    The mapping is returned unresolved so callers can store it on
    ``global_config.interconnect`` exactly as a YAML ``interconnect:`` block.
    """
    import yaml

    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, Mapping):
        raise ValueError(f'Interconnect file must contain a mapping: {path}')
    resolve_interconnect(values)
    return dict(values)


def add_tapped_line(circuit, prefix, start, count, wire, ground='VSS'):
    """Add count pitches, each with a centered cell tap and symmetric pi RC.

    Refining sections changes discretization while conserving total R and C.
    The far endpoint includes the last half pitch beyond the final cell tap.
    """
    _count('tap count', count)
    steps = 2 * wire.sections_per_half_pitch
    resistance = wire.resistance_per_pitch / steps
    capacitance = wire.capacitance_per_pitch / steps
    caps = {}
    taps = tuple(f'{prefix}_tap{i}' for i in range(count))
    previous = start
    for index in range(count * steps):
        pitch, position = divmod(index + 1, steps)
        if index + 1 == count * steps:
            node = f'{prefix}_far'
        elif position == wire.sections_per_half_pitch:
            node = taps[pitch]
        else:
            node = f'{prefix}_wire{index}'
        circuit.R(f'wire_{prefix}_{index}', previous, node, resistance @ u_Ohm)
        caps[previous] = caps.get(previous, 0.) + capacitance / 2
        caps[node] = caps.get(node, 0.) + capacitance / 2
        previous = node
    for index, (node, value) in enumerate(caps.items()):
        circuit.C(f'wire_{prefix}_{index}', node, ground, value)
    return taps


def cell_wire_nodes(core, row, col):
    if core.interconnect.distributed:
        return f'BL{col}_tap{row}', f'BLB{col}_tap{row}', f'WL{row}_tap{col}'
    return f'BL{col}', f'BLB{col}', f'WL{row}'


def add_array_wires(core):
    config = core.interconnect
    if not config.distributed:
        return
    for row in range(core.num_rows):
        add_tapped_line(core, f'WL{row}', f'WL{row}', core.num_cols, config.wl)
    for col in range(core.num_cols):
        for name in ('BL', 'BLB'):
            add_tapped_line(core, f'{name}{col}', f'{name}{col}', core.num_rows, config.bl)
