"""Frozen array-class clocks and historical measured-phase calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING
from math import ceil, isfinite, log2
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class TimingConfig:
    """Phase budgets, retaining the historical JSON field names.

    Legacy low_read/low_write describe clock-high access and high describes
    clock-low recovery. For self-timed control, cycle_budget is capture to
    the recovered state before margin; access_time is the rounded absolute
    data deadline after the capture crossing, including margin.
    """
    t_period: float
    low_read: float
    low_write: float
    high: float
    margin: float = 0.25
    source: str = "calibrated"
    cycle_budget: float = 0.0
    access_time: float = 0.0

    def __post_init__(self):
        if not isfinite(self.t_period) or self.t_period <= 0:
            raise ValueError('Clock period must be finite and positive')
        if any(not isfinite(value) or value < 0 for value in
               (self.low_read, self.low_write, self.high, self.margin, self.cycle_budget, self.access_time)):
            raise ValueError('Timing phases and margin must be finite and nonnegative')
        if self.cycle_budget and not self.access_time:
            raise ValueError('A self-timed cycle needs an absolute access deadline')
        if self.access_time > self.t_period:
            raise ValueError('Access deadline exceeds the clock period')
        minimum = ((self.cycle_budget if self.cycle_budget else
                    2 * max(self.low_read, self.low_write, self.high)) * (1 + self.margin))
        if self.t_period + 1e-18 < minimum:
            raise ValueError('Clock period does not cover its declared phase margin')

    def to_dict(self):
        return asdict(self)

    def apply(self, testbench):
        from PySpice.Unit import (
            u_ns,  # pyright: ignore[reportAttributeAccessIssue] -- generated unit export
        )
        period = (self.t_period / 1e-9) @ u_ns
        duty = getattr(testbench, 'clock_duty', 0.5)
        self_timed = getattr(getattr(testbench, 'driver_sizes', None), 'access_speedup', False)
        if self_timed and not .01 < duty < .99:
            raise ValueError('Clock duty must leave room for the 1% rising and falling edges')
        pulse = (duty - .01 if self_timed else duty) * period
        testbench.set_timing_parameters(
            t_rise=0.01 * period, t_fall=0.01 * period,
            t_pulse=pulse,
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
    if not isinstance(table, dict) or table.get('schema') not in (1, 2):
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
    if table['schema'] == 2:
        _validate_completion_ladders(table.get('self_timed'))
    _validate_envelope(table)
    table['sha256'] = hashlib.sha256(raw).hexdigest()
    return table


_CELL_TYPES = ('SRAM_6T_CELL', 'SRAM_10T_CELL')
_BOUNDS = {'row_classes': 'max_rows', 'column_classes': 'max_cols'}


def _completion_time(budget_ps: float | Decimal, margin: float) -> float:
    """Round decimal budgets upward without adding a tick at a binary boundary."""
    ticks = (Decimal(str(budget_ps)) * (1 + Decimal(str(margin))) / 50).to_integral_value(
        rounding=ROUND_CEILING)
    return int(ticks) * 50e-12


def _validate_completion_ladders(block: object) -> None:
    if (not isinstance(block, dict) or block.get('cell_type') != 'SRAM_6T_CELL'
            or block.get('mux') is not True or type(block.get('compact_control')) is not bool
            or not isinstance(block.get('control_architecture'), str) or not block['control_architecture']):
        raise ValueError('Self-timed budgets need a 2:1-mux 6T control architecture')
    margin = block.get('margin', .10)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not isfinite(margin) or margin < 0:
        raise ValueError('Self-timed margin must be finite and nonnegative')
    for group, bound in _BOUNDS.items():
        entries = block.get(group)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f'Self-timed budgets need {group}')
        previous = (0, 0, 0)
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f'Invalid self-timed {group} entry')
            values = tuple(entry.get(name) for name in (bound, 'access_ps', 'cycle_ps'))
            if (any(type(value) is not int or value <= 0 for value in values)
                    or values[0] <= previous[0] or values[1] < previous[1] or values[2] < previous[2]
                    or values[1] > values[2]
                    or entry.get('evidence_status') not in ('postlayout', 'schematic', 'extrapolated')):
                raise ValueError('Self-timed bounds must increase and completion budgets must be positive and monotonic')
            previous = values
    grid = block.get('joint_classes')
    if grid is not None:
        rows=[entry['max_rows'] for entry in block['row_classes']]
        cols=[entry['max_cols'] for entry in block['column_classes']]
        if not isinstance(grid,list) or len(grid)!=len(rows)*len(cols):
            raise ValueError('Self-timed joint classes must cover the full row/column grid')
        expected={(r,c) for r in rows for c in cols}
        entries={}
        for entry in grid:
            if not isinstance(entry,dict):raise ValueError('Invalid self-timed joint class')
            key=(entry.get('max_rows'),entry.get('max_cols'))
            if (any(type(value) is not int for value in key) or key not in expected or key in entries
                    or any(type(entry.get(name)) is not int or entry[name]<=0 for name in ('access_ps','cycle_ps'))
                    or entry['access_ps']>entry['cycle_ps']
                    or entry.get('evidence_status') not in ('postlayout','schematic','extrapolated')):
                raise ValueError('Invalid or duplicate self-timed joint class')
            entries[key]=entry
        for i,r in enumerate(rows):
            for j,c in enumerate(cols):
                earlier=([(rows[i-1],c)] if i else [])+([(r,cols[j-1])] if j else [])
                if any(entries[r,c][name]<entries[key][name] for key in earlier for name in ('access_ps','cycle_ps')):
                    raise ValueError('Self-timed joint budgets must not decrease with dimension')
        for group,bound,other in (('row_classes','max_rows',cols[0]),('column_classes','max_cols',rows[0])):
            for entry in block[group]:
                key=(entry[bound],other) if group=='row_classes' else (other,entry[bound])
                if any(entry[name]!=entries[key][name] for name in ('access_ps','cycle_ps')):
                    raise ValueError('Self-timed joint grid must match its single-dimension anchors')


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


def _validate_envelope(table):
    """The supported array envelope, which the ladders must reach but not exceed.

    A size beyond it is rejected rather than extrapolated: the geometric
    continuation carries no evidence, and V2.2.3 fixes the envelope at the
    largest array the compiler is built for (512 rows by 256 columns).
    """
    envelope = table.get('supported_envelope')
    if not isinstance(envelope, dict) or envelope.keys() != set(_BOUNDS.values()):
        raise ValueError('Timing lookup needs a supported_envelope with max_rows and max_cols')
    for group, bound in _BOUNDS.items():
        limit = envelope[bound]
        if type(limit) is not int or limit <= 0:
            raise ValueError('Supported envelope bounds must be positive integers')
        ladders_to_check = [table] + table.get('variants', [])
        if table.get('schema') == 2:
            ladders_to_check.append(table['self_timed'])
        for ladders in ladders_to_check:
            if ladders[group][-1][bound] != limit:
                raise ValueError(f'The last {group} anchor must equal the supported envelope')


def _budget_ladders(table, cell_type, mux):
    """The variant ladders for this architecture, or the shared ladders."""
    for variant in table.get('variants', []):
        if variant['cell_type'] == cell_type and variant.get('mux', mux) == mux:
            if 'mux' not in variant:
                return variant, cell_type
            return variant, f'{cell_type}/mux' if mux else f'{cell_type}/nomux'
    return table, 'shared'


def require_envelope(rows, cols, table=None, *, fixed_diagnostic: bool = False):
    """Allow explicitly timed physical diagnostics through 512x512."""
    if fixed_diagnostic and rows <= 512 and cols <= 512:
        return
    envelope = (load_timing_lookup() if table is None else table)['supported_envelope']
    if rows > envelope['max_rows'] or cols > envelope['max_cols']:
        raise ValueError(
            f"{rows}x{cols} is outside the supported array envelope "
            f"of {envelope['max_rows']} rows by {envelope['max_cols']} columns")


def _half_period_ps(ladders: dict, rows: int, cols: int,
                    budget_field: str = 'half_period_ps') -> tuple[int, dict, dict]:
    from .driver_sizing import interpolate_class
    row = interpolate_class(ladders['row_classes'], 'max_rows', (budget_field,), rows)
    col = interpolate_class(ladders['column_classes'], 'max_cols', (budget_field,), cols)
    # V2.2.3: bitline height and wordline width cost time separately, so an
    # array large in both dimensions pays for both. A dimension's excess is its
    # budget above the budget of its own first class, which is what the
    # smallest array of this architecture already pays. The larger of the two
    # budgets carries that base once and the smaller excess is added on top.
    # A dimension inside its first class has no excess, so every size screened
    # before V2.2.3 keeps exactly the maximum-rule period it was screened at.
    joint_ps = min(row[budget_field] - ladders['row_classes'][0][budget_field],
                   col[budget_field] - ladders['column_classes'][0][budget_field])
    return max(row[budget_field], col[budget_field]) + joint_ps, row, col


def _validate_completion_baseline(block: dict, config: Any, sizes: Any) -> None:
    baseline = block.get('baseline')
    if baseline is None:
        return
    from .driver_sizing import load_lookup, lookup_classes
    options = config.global_config.sizing
    options = options if isinstance(options, dict) else vars(options)
    if (sizes.source != 'lookup' or options.keys() -
            {'mode', 'lookup', 'compact_control', 'access_speedup', 'replica'}):
        raise ValueError('Self-timed budgets require the frozen lookup driver baseline; use fixed diagnostic timing')
    lookup = load_lookup(options.get('lookup'))
    if (lookup['sha256'] != baseline.get('sizing_lookup_sha256')
            or sizes.lookup_sha256 != baseline.get('sizing_lookup_sha256')):
        raise ValueError('Self-timed budgets belong to a different sizing lookup')
    if sizes.sizing_policy_key != baseline.get('sizing_policy_key'):
        raise ValueError('Self-timed budgets require the frozen buffer and decoder sizing policy')
    if (sizes.peripheral_key != baseline.get('peripheral_key')
            or sizes.pdk_key != baseline.get('pdk_key')):
        raise ValueError('Self-timed budgets require the baseline peripheral configuration and PDK')
    if sizes.physical_key not in baseline.get('physical_keys', []):
        raise ValueError('Self-timed budgets require the baseline physical RC profile')
    row, _ = lookup_classes(lookup, sizes.rows, sizes.cols)
    reference = row['self_timed']['replica']
    if (sizes.replica_k, sizes.dc_stages) != (min(reference['K'], sizes.rows, sizes.cols), reference['N']):
        raise ValueError('Self-timed budgets require the measured replica timing reference')


def resolve_timing(config, driver_sizes, context=None):
    """Select one period by array class, without simulation or PVT/cell refitting.

    Unseen dimensions round up to the next anchor; a dimension beyond the
    supported envelope is rejected by automatic timing. An explicit fixed
    period permits diagnostic layout sources through 512x512, without timing
    qualification. Use integer ps until
    conversion to seconds. A table variant keyed by cell type and optionally
    mux (V2.1.3: 10T cells with or without a column mux; V2.1.4: 6T cells with
    a column mux) replaces the shared ladders for that architecture only.
    Exact qualified driver records retain their measured timing.
    """
    driver_sizes.validate_for(config, driver_sizes.cell_type, driver_sizes.mux, context)
    options = _timing_options(config)
    table = load_timing_lookup(options.get('lookup'))
    require_envelope(driver_sizes.rows, driver_sizes.cols, table,
                     fixed_diagnostic=options.get('mode') == 'fixed')
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
    completion = table.get('self_timed') if driver_sizes.access_speedup and table['schema'] == 2 else None
    if driver_sizes.access_speedup and completion is None:
        raise ValueError('Self-timed automatic clocks require a completion-budget table; use fixed diagnostic timing')
    margin = options.get('margin', completion.get('margin', .10) if completion else .25)
    if isinstance(margin, bool) or not isinstance(margin, (float, int)) or not isfinite(margin) or margin < 0:
        raise ValueError('Timing margin must be finite and nonnegative')
    if completion:
        if (driver_sizes.control_architecture.split(':')[0] != completion['control_architecture']
                or driver_sizes.compact_control != completion['compact_control']):
            raise ValueError('Self-timed budgets belong to a different control architecture')
        _validate_completion_baseline(completion, config, driver_sizes)
        access_ps, row, col = _half_period_ps(completion, driver_sizes.rows, driver_sizes.cols, 'access_ps')
        cycle_ps = _half_period_ps(completion, driver_sizes.rows, driver_sizes.cols, 'cycle_ps')[0]
        joint=next((entry for entry in completion.get('joint_classes',[]) if
                    (entry['max_rows'],entry['max_cols'])==(row['max_rows'],col['max_cols'])),None)
        if joint is not None:
            access_ps,cycle_ps=joint['access_ps'],joint['cycle_ps']
        if access_ps > cycle_ps:
            raise ValueError('Joint access budget exceeds the joint cycle budget')
        period = _completion_time(cycle_ps, margin)
        access_time = _completion_time(access_ps, margin)
        return ArrayTiming(period, 0, 0, 0, margin=margin, source='lookup',
                           cycle_budget=cycle_ps * 1e-12, access_time=access_time,
                           driver_key=driver_sizes.key, options_key=_options_key(options),
                           table_sha256=table['sha256'], table_version=table.get('lookup_version', ''),
                           size_class=f"rows<={row['max_rows']}/cols<={col['max_cols']}",
                           extrapolated=any(entry.get('evidence_status') == 'extrapolated' for entry in
                                            ((joint,) if joint is not None else (row,col))),
                           budget='6T-self-timed')
    ladders, budget = _budget_ladders(table, driver_sizes.cell_type, driver_sizes.mux)
    half_ps, row, col = _half_period_ps(ladders, driver_sizes.rows, driver_sizes.cols)
    if ladders is not table:
        # V2.2.4: a variant may only add time. Its excess is measured from its
        # own first class, so a variant that raises only its early classes has
        # a smaller joint excess and, without this floor, a shorter clock than
        # the shared ladder it left (e.g. 128x128 at 19.5 against 20.5 ns).
        half_ps = max(half_ps, _half_period_ps(table, driver_sizes.rows, driver_sizes.cols)[0])
    period = ceil(2 * half_ps * (1 + margin) / 50) * 50e-12
    return ArrayTiming(period, half_ps * 1e-12, half_ps * 1e-12, half_ps * 1e-12,
                       margin=margin, source='lookup', driver_key=driver_sizes.key,
                       options_key=_options_key(options), table_sha256=table['sha256'],
                       table_version=table.get('lookup_version', ''),
                       size_class=f"rows<={row['max_rows']}/cols<={col['max_cols']}",
                       extrapolated=row['extrapolated'] or col['extrapolated'], budget=budget)


def timing_from_measurements(read: dict, writes: Sequence[dict], *,
                             margin: float | None = None, self_timed: bool = False) -> TimingConfig:
    """Use SS read and both SS/SF writes; never recalibrate from MC candidates."""
    def positive(metrics, key):
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Missing/failed timing calibration measurement {key}') from exc
        if not isfinite(value) or value < 0:
            raise ValueError(f"Invalid timing calibration measurement {key}: {value}")
        return value

    margin = (.10 if self_timed else .25) if margin is None else margin
    if not isfinite(margin) or margin < 0:
        raise ValueError("Timing margin must be finite and nonnegative")
    if self_timed:
        if not writes:
            raise ValueError('Self-timed calibration requires write completion measurements')
        access = positive(read, 'TACCESS')
        cycle = max(positive(metrics, 'TCYCLE') for metrics in [read, *writes])
        if access <= 0 or cycle < access:
            raise ValueError('Invalid self-timed completion measurements')
        return TimingConfig(_completion_time(Decimal(str(cycle)) * Decimal('1e12'), margin),
                            0, 0, 0, margin, source='measured-self-timed', cycle_budget=cycle,
                            access_time=_completion_time(Decimal(str(access)) * Decimal('1e12'), margin))
    if len(writes) != 2:
        raise ValueError("Timing calibration requires SS and SF write phases")
    low_read = positive(read, 'TCLK_WLEN') + positive(read, 'TREAD_TOTAL')
    low_write = max(positive(m, 'TCLK_WLEN') + positive(m, 'TWRITE_TOTAL') for m in writes)
    # V2.2.0 restores after every operation in clock-low. Keep the legacy
    # field names for serialized calibration records, but measure recovery.
    high = max(max(positive(m, 'TRESTORE'),
                   positive(m, 'TCLK_DEC') if 'TCLK_DEC' in m else 0)
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
