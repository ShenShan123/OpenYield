"""Frozen array-class clocks and historical measured-phase calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite, log2
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class TimingConfig:
    t_period: float
    low_read: float
    low_write: float
    high: float
    margin: float = 0.25
    source: str = "calibrated"

    def __post_init__(self):
        if not isfinite(self.t_period) or self.t_period <= 0:
            raise ValueError('Clock period must be finite and positive')
        if any(not isfinite(value) or value < 0 for value in
               (self.low_read, self.low_write, self.high, self.margin)):
            raise ValueError('Timing phases and margin must be finite and nonnegative')
        minimum = 2 * max(self.low_read, self.low_write, self.high) * (1 + self.margin)
        if self.t_period + 1e-18 < minimum:
            raise ValueError('Clock period does not cover its declared phase margin')

    def to_dict(self):
        return asdict(self)

    def apply(self, testbench):
        from PySpice.Unit import (
            u_ns,  # pyright: ignore[reportAttributeAccessIssue] -- generated unit export
        )
        period = (self.t_period / 1e-9) @ u_ns
        testbench.set_timing_parameters(
            t_rise=0.01 * period, t_fall=0.01 * period, t_pulse=0.5 * period,
            t_period=period, t_delay=1 @ u_ns,
        )
        testbench.timing_config = self


@dataclass(frozen=True)
class ArrayTiming(TimingConfig):
    """A timing specification bound to the frozen driver baseline.

    Keep TimingConfig's historical calibration/qualification format unchanged.
    Lookup budgets are design settings, not qualification records.
    """
    driver_key: str = ""
    options_key: str = ""
    table_sha256: str = ""
    table_version: str = ""
    size_class: str = ""
    extrapolated: bool = False
    qualified: bool = False
    budget: str = "shared"

    def validate_for(self, config, driver_sizes):
        if driver_sizes.key != self.driver_key:
            raise ValueError('Frozen timing belongs to a different driver baseline')
        if _options_key(_timing_options(config)) != self.options_key:
            raise ValueError('Frozen timing requires the baseline timing options')


DEFAULT_TIMING_LOOKUP = Path(__file__).with_name('timing_lookup.json')


def _timing_options(config):
    options = getattr(config.global_config, 'timing', {'mode': 'lookup'})
    if not isinstance(options, dict):
        raise ValueError('timing options must be a mapping')
    if options.keys() - {'mode', 'lookup', 'margin', 't_period'}:
        raise ValueError('Unsupported timing options')
    mode = options.get('mode', 'lookup')
    if mode not in ('lookup', 'fixed'):
        raise ValueError('timing.mode must be lookup or fixed')
    if mode == 'lookup' and 't_period' in options:
        raise ValueError('timing.t_period requires timing.mode=fixed')
    if mode == 'fixed' and options.keys() & {'lookup', 'margin'}:
        raise ValueError('Fixed timing accepts only mode and t_period')
    return options


def _options_key(options):
    return hashlib.sha256(json.dumps(options, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_timing_lookup(path=None):
    """Load class phase budgets in integer ps, resolving paths from the project."""
    path = DEFAULT_TIMING_LOOKUP if path is None else Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    raw = path.read_bytes()
    table = json.loads(raw)
    if not isinstance(table, dict) or table.get('schema') != 1:
        raise ValueError('Unsupported timing lookup schema')
    _validate_ladders(table)
    variants = table.get('variants', [])
    if not isinstance(variants, list):
        raise ValueError('Timing lookup variants must be a list')
    seen = set()
    for variant in variants:
        if (not isinstance(variant, dict) or variant.get('cell_type') not in _CELL_TYPES
                or not isinstance(variant.get('mux', False), bool)):
            raise ValueError('Timing lookup variants need a supported cell_type and, if given, a boolean mux')
        # One entry per cell type; an entry without mux covers both mux settings.
        if variant['cell_type'] in seen:
            raise ValueError('Duplicate timing lookup variant')
        seen.add(variant['cell_type'])
        _validate_ladders(variant)
        # A variant is a separately evidenced budget for one architecture; it
        # keeps the shared anchors and may only add to the shared budget.
        for group, bound in _BOUNDS.items():
            shared = table[group]
            if (len(variant[group]) != len(shared)
                    or any(entry[bound] != base[bound] or entry['half_period_ps'] < base['half_period_ps']
                           for entry, base in zip(variant[group], shared))):
                raise ValueError('Timing lookup variants must keep the shared anchors and never relax a budget')
    table['sha256'] = hashlib.sha256(raw).hexdigest()
    return table


_CELL_TYPES = ('SRAM_6T_CELL', 'SRAM_10T_CELL')
_BOUNDS = {'row_classes': 'max_rows', 'column_classes': 'max_cols'}


def _validate_ladders(table):
    for group, bound in _BOUNDS.items():
        entries = table.get(group)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f'Timing lookup needs a non-empty {group}')
        previous_bound = previous_budget = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f'Invalid timing {group} entry')
            limit, budget = entry.get(bound), entry.get('half_period_ps')
            if (type(limit) is not int or limit <= previous_bound or
                    type(budget) is not int or budget <= 0 or budget < previous_budget):
                raise ValueError('Timing class bounds must increase and positive budgets must not decrease')
            previous_bound, previous_budget = limit, budget
        if len(entries) > 1 and entries[-1]['half_period_ps'] <= entries[-2]['half_period_ps']:
            raise ValueError('The final timing budget must grow for geometric extrapolation')


def _budget_ladders(table, cell_type, mux):
    """The variant ladders for this architecture, or the shared ladders."""
    for variant in table.get('variants', []):
        if variant['cell_type'] == cell_type and variant.get('mux', mux) == mux:
            if 'mux' not in variant:
                return variant, cell_type
            return variant, f'{cell_type}/mux' if mux else f'{cell_type}/nomux'
    return table, 'shared'


def resolve_timing(config, driver_sizes, context=None):
    """Select one period by array class, without simulation or PVT/cell refitting.

    Like driver sizing, unseen dimensions round up and sizes beyond the last
    class extrapolate the ladder. Use integer ps until conversion to seconds.
    A table variant keyed by cell type and optionally mux (V2.1.3: 10T cells
    with or without a column mux) replaces the shared ladders for that
    architecture only.
    Exact qualified driver records retain their measured timing.
    """
    driver_sizes.validate_for(config, driver_sizes.cell_type, driver_sizes.mux, context)
    options = _timing_options(config)
    if options.get('mode', 'lookup') == 'fixed':
        period = options.get('t_period')
        if isinstance(period, bool):
            raise ValueError('Fixed clock period must be finite and positive')
        try:
            period = float(period)
        except (TypeError, ValueError) as exc:
            raise ValueError('Fixed timing requires t_period in seconds') from exc
        return ArrayTiming(period, 0, 0, 0, margin=0, source='fixed',
                           driver_key=driver_sizes.key, options_key=_options_key(options))
    if driver_sizes.source == 'table':
        from .table import qualified_timing
        sizing = config.global_config.sizing
        table_path = sizing.get('table') if isinstance(sizing, dict) else getattr(sizing, 'table', None)
        timing = qualified_timing(driver_sizes, table_path, context)
        if timing is None:
            raise ValueError('Qualified timing record is stale or unavailable for this physical context')
        return timing
    from .driver_sizing import interpolate_class
    table = load_timing_lookup(options.get('lookup'))
    margin = options.get('margin', .25)
    if isinstance(margin, bool) or not isinstance(margin, (float, int)) or not isfinite(margin) or margin < 0:
        raise ValueError('Timing margin must be finite and nonnegative')
    ladders, budget = _budget_ladders(table, driver_sizes.cell_type, driver_sizes.mux)
    row = interpolate_class(ladders['row_classes'], 'max_rows', ('half_period_ps',), driver_sizes.rows)
    col = interpolate_class(ladders['column_classes'], 'max_cols', ('half_period_ps',), driver_sizes.cols)
    half_ps = max(row['half_period_ps'], col['half_period_ps'])
    period = ceil(2 * half_ps * (1 + margin) / 50) * 50e-12
    return ArrayTiming(period, half_ps * 1e-12, half_ps * 1e-12, half_ps * 1e-12,
                       margin=margin, source='lookup', driver_key=driver_sizes.key,
                       options_key=_options_key(options), table_sha256=table['sha256'],
                       table_version=table.get('lookup_version', ''),
                       size_class=f"rows<={row['max_rows']}/cols<={col['max_cols']}",
                       extrapolated=row['extrapolated'] or col['extrapolated'], budget=budget)


def timing_from_measurements(read, writes, *, margin=0.25):
    """Use SS read and both SS/SF writes; never recalibrate from MC candidates."""
    def positive(metrics, key):
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Missing/failed timing calibration measurement {key}') from exc
        if not isfinite(value) or value < 0:
            raise ValueError(f"Invalid timing calibration measurement {key}: {value}")
        return value

    if not isfinite(margin) or margin < 0:
        raise ValueError("Timing margin must be finite and nonnegative")
    if len(writes) != 2:
        raise ValueError("Timing calibration requires SS and SF write phases")
    low_read = positive(read, 'TCLK_WLEN') + positive(read, 'TREAD_TOTAL')
    low_write = max(positive(m, 'TCLK_WLEN') + positive(m, 'TWRITE_TOTAL') for m in writes)
    high = max(max(positive(m, 'TRESTORE'), positive(m, 'TCLK_DEC') if 'TCLK_DEC' in m else 0)
               for m in [read, *writes])
    if not isfinite(high) or max(low_read, low_write, high) <= 0:
        raise ValueError("Invalid calibration phases")
    period = ceil(2 * max(low_read, low_write, high) * (1 + margin) / 50e-12) * 50e-12
    return TimingConfig(period, low_read, low_write, high, margin)


def provisional_period(rows, cols, cell_type):
    """Long calibration clock only: historical phase model plus a factor of two.

    The old fitted coefficients do not qualify the new periphery. Every shipped
    timing record must come from the measured phases above, then pass MC checks.
    """
    read_low_ps = (392 + 1.20 * rows + 9.33 * log2(cols) if cell_type == 'SRAM_10T_CELL'
                   else 390 + 0.946 * rows + 9.06 * log2(cols)) * 2.31
    write_high_ps = (188 + 0.41 * cols + 10.7 * log2(max(rows, 2))) * 2.24
    return max(5e-9, ceil(5 * max(read_low_ps, write_high_ps) / 50) * 50e-12)
