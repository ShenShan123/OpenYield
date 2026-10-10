"""Validated TIME_CONTROL settings, independent of netlist topology."""
from __future__ import annotations

from collections.abc import Mapping
from math import ceil, isfinite
from pathlib import Path
from types import MappingProxyType

import yaml

DEFAULT_PATH = Path(__file__).with_name('time_control.yaml')


class TimeControlConfig(Mapping):
    """Immutable nested settings; defaults cannot leak between constructions."""
    __slots__ = ('_data', '_validated')
    def __init__(self, data, *, _validated=False):
        def freeze(value):
            if isinstance(value, Mapping):
                return TimeControlConfig(value)
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            return value
        self._data = MappingProxyType({key: freeze(value) for key, value in data.items()})
        self._validated = _validated

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def to_dict(self):
        def thaw(value):
            if isinstance(value, TimeControlConfig):
                return value.to_dict()
            if isinstance(value, tuple):
                return [thaw(item) for item in value]
            return value
        return {key: thaw(value) for key, value in self._data.items()}

    def widths(self, role):
        pair = self.devices[role]
        return pair.nmos_width_m, pair.pmos_width_m

    def physical_width(self, width):
        """Round total width up so every folded finger fits the process grid."""
        width = float(width)
        if not isfinite(width) or width <= 0:
            raise ValueError('Controller transistor width must be finite and positive')
        width = max(width, self.devices.minimum_width_m)
        fingers = max(1, ceil(round(width / self.devices.max_finger_width_m, 12)))
        per_finger = ceil(round(width / fingers / self.devices.width_grid_m, 12))
        return fingers * per_finger * self.devices.width_grid_m


def _read(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, Mapping) or set(data) != {'TIME_CONTROL'}:
        raise ValueError('TIME_CONTROL YAML must contain exactly one TIME_CONTROL mapping')
    return data['TIME_CONTROL']


_SCHEMA = _read(DEFAULT_PATH)


def _validate(data, schema, path='TIME_CONTROL'):
    if schema is None:
        if data is None:
            return None
        if (isinstance(data, bool) or not isinstance(data, (int, float))
                or not isfinite(data) or data < 0):
            raise ValueError(f'{path}: expected optional finite nonnegative number')
        if not path.endswith(('_tau', '.din_hold_load')) and data == 0:
            raise ValueError(f'{path}: load must be positive')
        if path.endswith('.num_sa') and not isinstance(data, int):
            raise ValueError(f'{path}: expected optional integer')
        return data
    if isinstance(schema, Mapping):
        if not isinstance(data, Mapping) or set(data) != set(schema):
            raise ValueError(f'{path}: missing or unknown settings')
        return {key: _validate(data[key], value, path + '.' + key) for key, value in schema.items()}
    if isinstance(schema, list):
        if not isinstance(data, (list, tuple)) or len(data) != len(schema):
            raise ValueError(f'{path}: expected {len(schema)} stages')
        return [_validate(item, template, f'{path}[{i}]') for i, (item, template) in enumerate(zip(data, schema))]
    if isinstance(schema, bool):
        if type(data) is not bool:
            raise ValueError(f'{path}: expected boolean')
    elif isinstance(schema, str):
        if not isinstance(data, str) or not data.strip():
            raise ValueError(f'{path}: expected nonempty string')
        if path.endswith('.operation') and data not in ('read', 'write', 'read&write'):
            raise ValueError(f'{path}: expected read, write or read&write')
    else:
        if isinstance(data, bool) or not isinstance(data, (int, float)) or not isfinite(data):
            raise ValueError(f'{path}: expected finite numeric value')
        if isinstance(schema, int) and not isinstance(data, int):
            raise ValueError(f'{path}: expected integer')
        minimum = 0 if schema == 0 else 1 if isinstance(schema, int) else 0
        if data < minimum or (schema != 0 and data == 0):
            raise ValueError(f'{path}: expected positive value (zero only for optional delay)')
    return data


def resolve_time_control_config(value=None):
    if value is None:
        return DEFAULT_CONFIG
    if isinstance(value, TimeControlConfig):
        if value._validated:
            return value
        value = value.to_dict()
    data = _validate(value, _SCHEMA)
    devices = data['devices']
    if devices['minimum_width_m'] > devices['max_finger_width_m']:
        raise ValueError('Controller minimum width exceeds maximum finger width')
    grid_count = devices['max_finger_width_m'] / devices['width_grid_m']
    if abs(grid_count-round(grid_count)) > 1e-9:
        raise ValueError('Maximum controller finger width must fit the width grid')
    defaults, delays, guards, buffers = (data[k] for k in ('defaults', 'delays', 'guards', 'buffers'))
    if len(data['devices']['clock_buffer_stages']) != 4:
        raise ValueError('TIME_CONTROL clock topology requires exactly four configured stages')
    if len(buffers['clock_stage_scale_exponents']) != 4:
        raise ValueError('TIME_CONTROL clock scaling requires exactly four configured exponents')
    if defaults['dc_stages'] % 2 != 1:
        raise ValueError('TIME_CONTROL.defaults.dc_stages must be odd')
    for count in (defaults['precharge_guard_stages'], defaults['precharge_off_guard_stages'],
                  delays['unit_stages'], delays['wordline_setup_stages'], guards['isolation_stages'],
                  guards['enable_off_stages'], buffers['short_stages'], buffers['long_stages']):
        if count % 2:
            raise ValueError('TIME_CONTROL non-inverting delay/buffer stage counts must be even')
    if defaults['precharge_off_guard_stages'] < 2 or delays['unit_stages'] < 2:
        raise ValueError('TIME_CONTROL settling delays require at least two stages')
    if defaults['precharge_guard_stages'] and not defaults['replica_precharge_guard']:
        raise ValueError('TIME_CONTROL precharge delay requires the replica guard')
    return TimeControlConfig(data, _validated=True)


DEFAULT_CONFIG = resolve_time_control_config(_SCHEMA)


def load_time_control_config(path=DEFAULT_PATH):
    path = Path(path)
    if not path.is_absolute():
        path = DEFAULT_PATH.parents[2] / path
    return resolve_time_control_config(_read(path))
