"""Fixed driver size classes (V2.0.9), legacy rules and qualification lookup.

Resolve once from the baseline, then pass the same result to each cell candidate.
This module does not run Xyce. The default ``lookup`` mode reads integer size
classes from ``sizing_lookup.json`` and extrapolates the class ladder for arrays
beyond the table; ``rules_only`` keeps the V2.0.5 continuous rules the classes
were derived from, and ``auto`` consults the qualified table before falling back
to those rules. The legacy ``fixed`` array rules were removed in V2.0.9.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import Any

_RULES = json.loads(Path(__file__).with_name('sizing_rules.json').read_text())
RULE_VERSION = _RULES['rule_version']
CONTROL_ARCHITECTURE = 'v2.2.8-digital-guards-high-access'
COMPACT_CONTROL_ARCHITECTURE = 'v2.2.8-compact-digital-guards-high-access'
SELF_TIMED_CONTROL_ARCHITECTURE = 'v2.2.9-self-timed-completion-1'
DEFAULT_LOOKUP = Path(__file__).with_name('sizing_lookup.json')
_MODES = ("lookup", "rules_only", "auto")
_PERIPHERALS = (
    "precharge", "write_driver", "wordline_driver", "decoder", "column_mux", "senseamp",
)
_ROW_SCALES = ("pre", "wd_in", "wd_out")
_COLUMN_SCALES = ("wl_inv", "wl_nand", "dec_inv")
PRECHARGE_OFF_GUARD_STAGES = 4
# Fixed digital delays of TIME_CONTROL when a class table does not set them:
# the wordline setup chain (row class, `setup_stages`) and the far-end
# settling guards (column class, `guard_stages`: precharge-off, isolation,
# replica-wordline-off and, when a table sets it, the enables-off guard; the
# fixed default keeps the enables-off guard at its two stages).
DEFAULT_STAGES = dict(setup_stages=8, guard_stages=PRECHARGE_OFF_GUARD_STAGES, enable_off_stages=2,
                      read_completion_stages=8, write_completion_stages=8)
_COMPLETION_STAGES = ('read_completion_stages', 'write_completion_stages')
_GUARD_PATHS = ('precharge_off_stages','isolation_stages','wordline_off_stages','write_drive_stages')
_STAGE_FIELDS = {"row_classes": "setup_stages", "column_classes": "guard_stages"}
# A column class may cap the bitline-driver classes: every column adds its
# precharge and write-driver gates to one control bar, so a wide array's
# restore and write completion is limited by the bar, not the bitline.
_COLUMN_CAPS = {"pre_cap": "pre", "wd_cap": "wd_out"}
_JOINT_SIZING_FIELDS = (*_ROW_SCALES, 'delay_driver_scale', 'delay_loads_per_stage')


@dataclass(frozen=True)
class DriverLoads:
    """TIME_CONTROL input loads: inverter units except wl_load, which uses NAND units."""

    pre_load: float
    wen_load: float
    wl_load: float
    num_sa: int
    wenb_scale: int
    sen_load: float | None = None
    iso_load: float | None = None
    sen_effort: float = 8.0
    access_load: float | None = None
    observer_load_units: float | None = None
    far_enable_observer_load_units: float | None = None


@dataclass(frozen=True)
class DriverSizes:
    rows: int
    cols: int
    cell_type: str
    mux: bool
    pre: float
    wd_in: float
    wd_out: float
    wl_inv: float
    wl_nand: float
    loads: DriverLoads
    source: str
    key: str
    peripheral_key: str
    pdk_key: str
    dec_inv: float = 1.0
    replica_matched: bool = True
    replica_k: int = 1
    dc_stages: int = 9
    effort_buffers: bool = False
    canonical_read: bool = True
    replica_precharge_guard: bool = True
    replica_nmos_models: tuple = ()
    replica_nmos_widths: tuple = ()
    replica_pmos_model: str = ""
    replica_pmos_width: float = 0.0
    replica_length: float = 0.0
    qualified_context_key: str = ""
    area_precharge_width: float = 0.0
    area_wordline_width: float = 0.0
    area_senseamp_width: float = 0.0
    rule_version: str = RULE_VERSION
    physical_key: str = ""
    size_class: str = ""
    extrapolated: bool = False
    precharge_guard_stages: int = 4
    precharge_off_guard: bool = True
    precharge_off_guard_stages: int = PRECHARGE_OFF_GUARD_STAGES
    precharge_off_tau: float = 0.0
    control_architecture: str = ""
    enable_off_tau: float = 0.0
    isolation_tau: float = 0.0
    compact_control: bool = False
    wordline_setup_stages: int = DEFAULT_STAGES['setup_stages']
    isolation_guard_stages: int = DEFAULT_STAGES['guard_stages']
    enable_off_guard_stages: int = DEFAULT_STAGES['enable_off_stages']
    access_speedup: bool = False
    read_completion_stages: int = 8
    write_completion_stages: int = 8
    lookup_sha256: str = ''
    sizing_policy_key: str = ''
    write_drive_guard_stages: int = 4
    delay_driver_scale: float | None = None
    delay_loads_per_stage: int | None = None

    def to_dict(self):
        """JSON-ready experiment metadata, including the result's evidence source."""
        return asdict(self)

    def validate_for(self, sram_config, cell_type, mux, context=None):
        """Allow cell candidates, but reject changed geometry or periphery."""
        architecture = COMPACT_CONTROL_ARCHITECTURE if self.compact_control else CONTROL_ARCHITECTURE
        if self.access_speedup:
            architecture = SELF_TIMED_CONTROL_ARCHITECTURE + (':compact' if self.compact_control else ':schematic')
        if self.control_architecture != architecture:
            raise ValueError("Frozen driver sizes belong to a different control architecture")
        cfg = sram_config.global_config
        options = _mapping(getattr(cfg, 'sizing', {}))
        if self.source == 'lookup':
            lookup = load_lookup(options.get('lookup'))
            if self.lookup_sha256 != lookup['sha256']:
                raise ValueError('Frozen driver sizes require the current sizing lookup')
            row, column = lookup_classes(lookup, self.rows, self.cols)
            profile = _joint_sizing(lookup, row, column, self.access_speedup and self.compact_control)
            scale = profile.get('delay_driver_scale')
            if (self.delay_driver_scale != (float(scale) if scale is not None else None)
                    or self.delay_loads_per_stage != profile.get('delay_loads_per_stage')):
                raise ValueError('Frozen driver sizes require the current delay sizing profile')
        requested_speedup = _self_timed_control(options, cell_type, mux)
        if requested_speedup != self.access_speedup:
            raise ValueError('Frozen driver sizes belong to a different control architecture')
        if 'compact_control' in options:
            if options['compact_control'] != self.compact_control:
                raise ValueError('Frozen driver sizes belong to a different control architecture')
        if (cfg.num_rows, cfg.num_cols, cell_type, mux) != (
            self.rows, self.cols, self.cell_type, self.mux
        ):
            raise ValueError("Frozen driver sizes belong to a different array architecture")
        if _digest(_peripheral_inputs(sram_config)) != self.peripheral_key:
            raise ValueError("Frozen driver sizes require the baseline peripheral configuration")
        if _digest(_pdk_inputs(cfg)) != self.pdk_key:
            raise ValueError("Frozen driver sizes require the baseline PDK model contents")
        # Archived snapshots can still be read, but the current topology must
        # not silently add an observer or common gate to an older load budget.
        expected_access = 1.25 * max(1, ceil(self.loads.wl_load / (24.0 if self.effort_buffers else 32.0))) + 2.5
        stages = _stage_policy(options, cfg.num_rows, cfg.num_cols, self.access_speedup)
        if self.access_speedup and any(getattr(self,name) != stages[name] for name in _COMPLETION_STAGES):
            raise ValueError('Frozen driver sizes require the current completion timing references')
        if (self.precharge_off_guard is not True
                or self.replica_precharge_guard is not True
                or type(self.precharge_off_guard_stages) is not int
                or self.precharge_off_guard_stages != stages['precharge_off_stages']
                or self.isolation_guard_stages != stages['isolation_stages']
                or self.precharge_guard_stages != options.get('precharge_guard_stages',stages['wordline_off_stages'])
                or (self.access_speedup and self.write_drive_guard_stages != stages['write_drive_stages'])
                or self.enable_off_guard_stages != stages['enable_off_stages']
                or self.wordline_setup_stages != stages['setup_stages']
                or not isfinite(self.precharge_off_tau) or self.precharge_off_tau <= 0
                or not isfinite(self.enable_off_tau) or self.enable_off_tau <= 0
                or not isfinite(self.isolation_tau) or self.isolation_tau <= 0
                or self.loads.access_load != expected_access):
            raise ValueError('Frozen driver sizes require the current wordline-off guards and access load')
        if context is not None and _digest(context) != self.physical_key:
            raise ValueError('Frozen driver sizes belong to a different physical context')
        if context is not None:
            if (self.loads.observer_load_units is None
                    or self.loads.far_enable_observer_load_units is None):
                raise ValueError('Frozen driver loads predate the configured physical observers')
            rc_units = (float(context.get('pi_cap', _RULES['peripheral_rc_cap_f']))
                        / _RULES['unit_inverter_cap_f'] if context.get('w_rc', False) else 0.0)
            expected_pre = ((self.cols + 1) * (3 * float(sram_config.precharge.pmos_width.value)
                            * self.pre / .36e-6 + rc_units) + self.loads.observer_load_units)
            if (not isfinite(self.loads.pre_load)
                    or abs(self.loads.pre_load - expected_pre) > 1e-10 * max(1.0, expected_pre)):
                raise ValueError('Frozen precharge load must include the far-PRE observer')
            expected_tau = _precharge_off_tau(self.cols, self.loads.pre_load, context)
            if abs(self.precharge_off_tau - expected_tau) > 1e-10 * expected_tau:
                raise ValueError('Frozen precharge-off guard requires the baseline wire settling time')
            expected_enable_tau = _enable_off_tau(self.cols, self.loads, context)
            if abs(self.enable_off_tau - expected_enable_tau) > 1e-10 * expected_enable_tau:
                raise ValueError('Frozen enable-off guard requires the baseline wire settling time')
            expected_iso_tau = _control_settling_tau(self.cols, self.loads.iso_load + self.loads.observer_load_units, context, 'isolation_control')
            if abs(self.isolation_tau - expected_iso_tau) > 1e-10 * expected_iso_tau:
                raise ValueError('Frozen isolation guard requires the baseline wire settling time')
        if self.source == 'table' and context is not None:
            from .table import record_key
            if self.qualified_context_key != record_key(self.key, context):
                raise ValueError('Qualified driver sizes belong to a different physical context')


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _precharge_off_tau(cols, pre_load, context, route='precharge_control'):
    """Conservative wire/load settling scale, fixed before any PVT/cell sample.

    Use the same inverter-capacitance estimate as the driver-load rules. This
    is a geometry-based design delay, not extracted metal or a qualification.
    """
    from sram_compiler.interconnect import resolve_interconnect
    wire = resolve_interconnect(context.get('interconnect')).wire(route)
    resistance = cols * wire.resistance_per_pitch
    capacitance = pre_load * _RULES['unit_inverter_cap_f'] + cols * wire.capacitance_per_pitch
    local_tau = (float(context.get('pi_res', 100.0))
                 * float(context.get('pi_cap', _RULES['peripheral_rc_cap_f']))
                 if context.get('w_rc', False) else 0.0)
    return resistance * capacitance + local_tau


def _enable_off_tau(cols, loads, context):
    """Freeze recovery settling from the slower distributed enable line.

    Sense EN has two local RC sections. Four local time constants also cover
    their falling tail after the far observer switches near mid-rail.
    """
    from sram_compiler.config_yaml.time_control_config import DEFAULT_CONFIG
    far_load = loads.far_enable_observer_load_units
    if far_load is None:
        far_load = sum(DEFAULT_CONFIG.widths('far_enables_nor')) / DEFAULT_CONFIG.sizing.unit_input_width_m
    return max(_control_settling_tau(cols, loads.wen_load+far_load, context, 'write_enable_control'),
               _control_settling_tau(cols, loads.sen_load+far_load, context, 'sense_enable_control'))


def _control_settling_tau(cols, load, context, route='precharge_control'):
    """Wire plus a conservative allowance for two local RC sections."""
    tau = _precharge_off_tau(cols, load, context, route)
    if context.get('w_rc', False):
        tau += 3 * float(context.get('pi_res', 100.0)) * float(
            context.get('pi_cap', _RULES['peripheral_rc_cap_f']))
    return tau


def _circuit_inputs(config):
    # Descriptions and sample vectors do not change baseline qualification.
    return {
        name: {field: getattr(param, field) for field in ("value", "lower", "upper", "choices")}
        for name, param in config.parameters.items()
    }


def _peripheral_inputs(sram_config):
    from sram_compiler.config_yaml.time_control_config import resolve_time_control_config
    values = {name: _circuit_inputs(getattr(sram_config, name)) for name in _PERIPHERALS}
    values['time_control'] = resolve_time_control_config(getattr(sram_config, 'time_control', None)).to_dict()
    return values


def _pdk_inputs(cfg):
    return {
        corner: hashlib.sha256(Path(getattr(cfg, f"pdk_path_{corner}")).read_bytes()).hexdigest()
        for corner in ("TT", "FF", "SS", "FS", "SF")
    }


def _positive(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")  # noqa: TRY004 -- normalized configuration errors
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _mapping(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise ValueError("sizing options must be a mapping")


def _class_scale(entry, name, where):
    value = entry.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f"{where} needs a finite positive {name}")
    return float(value)


def _replica_reference(value: object, where: str) -> dict:
    if not isinstance(value, dict) or value.keys() - {'K', 'N'}:
        raise ValueError(f'{where} replica accepts only K and N')
    for name, count in value.items():
        if type(count) is not int or count < 1 or (name == 'N' and count % 2 == 0):
            raise ValueError(f'{where} replica needs positive integer K and positive odd N')
    return value


def _self_timed_control(options: dict, cell_type: str, mux: bool, table: dict | None = None) -> bool:
    if 'access_speedup' in options:
        if type(options['access_speedup']) is not bool:
            raise ValueError('access_speedup must be boolean')
        return options['access_speedup']
    if cell_type != 'SRAM_6T_CELL' or not mux or options.get('mode', 'lookup') != 'lookup':
        return False
    table = load_lookup(options.get('lookup')) if table is None else table
    return table.get('self_timed_control_architecture') == SELF_TIMED_CONTROL_ARCHITECTURE


def load_lookup(path=None):
    """Read and validate the driver size-class table (project-relative paths allowed)."""
    path = DEFAULT_LOOKUP if path is None else Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    table = json.loads(path.read_text())
    if table.get("schema") != 1:
        raise ValueError(f"Unsupported driver size lookup schema: {path}")
    architecture = table.get('self_timed_control_architecture')
    if architecture is not None and architecture != SELF_TIMED_CONTROL_ARCHITECTURE:
        raise ValueError('Driver lookup belongs to a different self-timed control architecture')
    for group, bound, names in (("row_classes", "max_rows", _ROW_SCALES),
                                ("column_classes", "max_cols", _COLUMN_SCALES)):
        classes = table.get(group)
        if not isinstance(classes, list) or not classes:
            raise ValueError(f"Driver size lookup {path} needs a non-empty {group} list")
        previous = 0
        for entry in classes:
            limit = entry.get(bound)
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= previous:
                raise ValueError(f"{group} of {path} must have strictly increasing positive {bound}")
            previous = limit
            if architecture and group == 'row_classes' and 'self_timed' not in entry:
                raise ValueError('Default self-timed control needs references for every row class')
            for name in names:
                _class_scale(entry, name, f"{group} {bound}={limit} of {path}")
            if 'self_timed' in entry:
                policy=entry['self_timed']
                allowed=set(_COMPLETION_STAGES) | ({'replica','setup_stages'} if group=='row_classes' else {'guard_stages'})
                if not isinstance(policy,dict) or policy.keys()-allowed:
                    raise ValueError('Unsupported self-timed timing reference')
                if group=='row_classes':
                    reference=_replica_reference(policy.get('replica',{}), f'{group} {bound}={limit}')
                    if reference.get('K',1)>limit:
                        raise ValueError('Replica K must not exceed its row-class bound')
                for name in ('setup_stages','guard_stages',*_COMPLETION_STAGES):
                    if name in policy:
                        count=policy[name]
                        minimum=8 if name in _COMPLETION_STAGES else 2
                        if type(count) is not int or count<minimum or count%2:
                            raise ValueError(f'Self-timed {name} must be an even integer of at least {minimum}')
            stage_name = _STAGE_FIELDS[group]
            if stage_name in entry:
                stages = entry[stage_name]
                if isinstance(stages, bool) or not isinstance(stages, int) or stages < 2 or stages % 2:
                    raise ValueError(f"{group} {bound}={limit} of {path} needs a positive even {stage_name}")
            if group == "column_classes":
                for cap in _COLUMN_CAPS:
                    if cap in entry:
                        _class_scale(entry, cap, f"{group} {bound}={limit} of {path}")
    _replica_reference(table.get('replica', {}), str(path))
    if 'self_timed_joint_references' in table:
        references=table['self_timed_joint_references'];seen=set()
        row_bounds={row['max_rows'] for row in table['row_classes']}
        col_bounds={col['max_cols'] for col in table['column_classes']}
        if not isinstance(references,list) or not references:
            raise ValueError('Self-timed joint references must be a nonempty list')
        for entry in references:
            allowed={'max_rows','max_cols','guard_stages','enable_off_stages',*_GUARD_PATHS,*_COMPLETION_STAGES}
            if not isinstance(entry,dict) or entry.keys()-allowed:
                raise ValueError('Unsupported self-timed joint timing reference')
            pair=(entry.get('max_rows'),entry.get('max_cols'))
            if (any(type(value) is not int for value in pair) or pair[0] not in row_bounds
                    or pair[1] not in col_bounds or pair in seen):
                raise ValueError('Self-timed joint references require unique row/column anchors')
            seen.add(pair)
            counts=[entry[name] for name in ('guard_stages','enable_off_stages',*_GUARD_PATHS,*_COMPLETION_STAGES) if name in entry]
            if not counts or any(type(value) is not int or value<2 or value%2 for value in counts):
                raise ValueError('Self-timed joint guard stages must be positive even integers')
            if any(entry.get(name,8)<8 for name in _COMPLETION_STAGES):
                raise ValueError('Self-timed joint completion stages must be at least eight')
    if 'self_timed_joint_sizing' in table:
        profiles = table['self_timed_joint_sizing']
        if not isinstance(profiles, list) or not profiles:
            raise ValueError('Self-timed joint sizing must be a non-empty list')
        row_bounds = {r['max_rows'] for r in table['row_classes']}
        col_bounds = {c['max_cols'] for c in table['column_classes']}
        seen = set()
        for profile in profiles:
            allowed = {'max_rows', 'max_cols', *_JOINT_SIZING_FIELDS}
            if not isinstance(profile, dict) or profile.keys() - allowed:
                raise ValueError('Unsupported self-timed joint sizing field')
            pair = (profile.get('max_rows'), profile.get('max_cols'))
            if (any(type(value) is not int for value in pair) or pair[0] not in row_bounds
                    or pair[1] not in col_bounds or pair in seen):
                raise ValueError('Self-timed joint sizing requires unique row/column anchors')
            seen.add(pair)
            if not any(name in profile for name in _JOINT_SIZING_FIELDS):
                raise ValueError('Self-timed joint sizing needs a sizing override')
            for name in (*_ROW_SCALES, 'delay_driver_scale'):
                if name in profile:
                    _positive(name, profile[name])
            delay_fields = ('delay_driver_scale', 'delay_loads_per_stage')
            if any(name in profile for name in delay_fields):
                if not all(name in profile for name in delay_fields):
                    raise ValueError('Delay sizing needs both driver scale and load count')
                count = profile['delay_loads_per_stage']
                if type(count) is not int or count < 1:
                    raise ValueError('Delay load count must be a positive integer')
    table["path"] = str(path)
    table["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return table


def _joint_sizing(lookup: dict | None, row_class: dict | None, col_class: dict | None,
                  enabled: bool) -> dict:
    if not enabled or lookup is None or row_class['extrapolated'] or col_class['extrapolated']:
        return {}
    pair = (row_class['max_rows'], col_class['max_cols'])
    return next((profile for profile in lookup.get('self_timed_joint_sizing', [])
                 if (profile['max_rows'], profile['max_cols']) == pair), {})


def interpolate_class(classes, bound, names, size):
    """Fixed-size interpolation on the class ladder, in log space of the array size.

    Inside the table the next anchor at or above ``size`` is used (round-up
    interpolation keeps every array on one of the tabulated fixed sizes, never
    below the rule the anchors were derived from). Beyond the last anchor the
    ladder continues geometrically: bounds and sizes grow by the ratio of the
    last two anchors (doubling for a single-anchor table), rounded up to
    integers, and the result is marked ``extrapolated``. Extrapolated sizes
    carry no waveform evidence; add measured anchors to the table instead of
    relying on them.
    """
    for entry in classes:
        if size <= entry[bound]:
            return dict(entry, extrapolated=False)
    last = classes[-1]
    previous = classes[-2] if len(classes) > 1 else None
    def ratio(name):
        if previous is None or previous[name] <= 0:
            return 2.0
        return max(1.0, last[name] / previous[name])
    steps = 1
    while last[bound] * ratio(bound) ** steps < size:
        steps += 1
    entry = {bound: ceil(last[bound] * ratio(bound) ** steps - 1e-9), "extrapolated": True}
    for name in names:
        entry[name] = float(ceil(last[name] * ratio(name) ** steps - 1e-9))
    return entry


def lookup_classes(table, rows, cols):
    """Row and column classes for an array; both may be extrapolated beyond the table."""
    return (interpolate_class(table["row_classes"], "max_rows", _ROW_SCALES, rows),
            interpolate_class(table["column_classes"], "max_cols", _COLUMN_SCALES, cols))


def _stage_policy(options: dict, rows: int, cols: int, access_speedup: bool = False) -> dict:
    """TIME_CONTROL fixed-delay stage counts for an array: the class table's
    `setup_stages` (row class) and `guard_stages` (column class) when the
    lookup sets them, otherwise the fixed defaults. Self-timed completion
    takes the larger row/column allowance, with the screened eight-stage
    floor. An extrapolated class keeps the last anchor's counts."""
    stages = dict(DEFAULT_STAGES)
    if options.get("mode", "lookup") == "lookup":
        table = load_lookup(options.get("lookup"))
        row_class, col_class = lookup_classes(table, rows, cols)
        for entry, name, classes in ((row_class, "setup_stages", table["row_classes"]),
                                     (col_class, "guard_stages", table["column_classes"])):
            source = classes[-1] if entry.get("extrapolated") else entry
            if name in source:
                stages[name] = int(source[name])
                if name == "guard_stages":
                    stages["enable_off_stages"] = int(source[name])
            if name=='setup_stages' and access_speedup:
                stages[name]=source.get('self_timed',{}).get(name,stages[name])
            if name=='guard_stages' and access_speedup and name in source.get('self_timed',{}):
                stages[name]=source['self_timed'][name]
                stages['enable_off_stages']=stages[name]
            if access_speedup:
                for field in _COMPLETION_STAGES:
                    stages[field]=max(stages[field],source.get('self_timed',{}).get(field,8))
        for field in _GUARD_PATHS:stages[field]=stages['guard_stages']
        if access_speedup:
            pair=(table['row_classes'][-1]['max_rows'] if row_class.get('extrapolated') else row_class['max_rows'],
                  table['column_classes'][-1]['max_cols'] if col_class.get('extrapolated') else col_class['max_cols'])
            for reference in table.get('self_timed_joint_references',[]):
                if (reference['max_rows'],reference['max_cols'])==pair:
                    stages['guard_stages']=reference.get('guard_stages',stages['guard_stages'])
                    stages['enable_off_stages']=reference.get('enable_off_stages',
                        reference.get('guard_stages',stages['enable_off_stages']))
                    for field in _GUARD_PATHS:stages[field]=reference.get(field,stages['guard_stages'])
                    for field in _COMPLETION_STAGES:
                        stages[field]=max(stages[field],reference.get(field,8))
    for field in _GUARD_PATHS:stages.setdefault(field,stages['guard_stages'])
    return stages


def resolve_driver_sizes(sram_config, *, cell_type=None, mux=None, sizing=None, physical_context=None):
    """Return immutable scales/loads without mutating configuration or simulating.

    ``lookup`` (default) takes integer size classes from ``sizing_lookup.json``,
    selected by row and column class only, so every array configuration maps to
    one of a small set of fixed driver sizes; arrays beyond the table continue
    the class ladder and are marked extrapolated. ``rules_only`` evaluates the
    V2.0.5 continuous rules the classes were derived from. ``auto`` uses an
    exact qualified-table match or falls back to those unverified rules.
    The result's key describes the baseline, not any later candidate cell/PVT.
    """
    cfg = sram_config.global_config
    from .table import physical_context as make_context
    from sram_compiler.interconnect import resolve_interconnect
    context = (make_context(interconnect=getattr(cfg, 'interconnect', None))
               if physical_context is None else dict(_mapping(physical_context)))
    wire = resolve_interconnect(context.get('interconnect', getattr(cfg, 'interconnect', None)))
    context['interconnect'] = wire.to_dict()
    rows, cols = cfg.num_rows, cfg.num_cols
    for name, value in (("num_rows", rows), ("num_cols", cols)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    cell_type = cfg.sram_cell_type if cell_type is None else cell_type
    if cell_type not in ("SRAM_6T_CELL", "SRAM_10T_CELL"):
        raise ValueError(f"Unsupported cell type: {cell_type}")
    mux = cfg.choose_columnmux if mux is None else mux
    if not isinstance(mux, bool):
        raise ValueError("mux must be a boolean")  # noqa: TRY004 -- normalized configuration errors
    if mux and cols % 2:
        raise ValueError("Column mux requires an even number of columns (fan-in 2)")
    options = dict(_mapping(getattr(cfg, "sizing", {}) if sizing is None else sizing))
    unknown = options.keys() - {
        "mode", "parasitic_factor", "wd_floor_margin", "k_w", "pre_min",
        "replica", "scale_decoder", "effort_buffers", "canonical_read",
        "table", "lookup", "precharge_guard_stages", "compact_control", "access_speedup",
    }
    if unknown:
        raise ValueError(f"Unsupported sizing options: {sorted(unknown)}")
    mode = options.get("mode", "lookup")
    if mode not in _MODES:
        raise ValueError("sizing.mode must be lookup, rules_only or auto")
    requested_mode = mode
    mode = 'rules_only' if mode == 'auto' else mode
    lookup = row_class = col_class = None
    if mode == "lookup":
        lookup = load_lookup(options.get("lookup"))
        row_class, col_class = lookup_classes(lookup, rows, cols)
    elif "lookup" in options:
        raise ValueError("sizing.lookup requires sizing.mode=lookup")
    p = _positive("parasitic_factor", options.get("parasitic_factor", 1.0))
    margin = _positive("wd_floor_margin", options.get("wd_floor_margin", _RULES['wd_floor_margin']))
    k_w = _positive("k_w", options.get("k_w", _RULES['k_w']))
    pre_min = _positive("pre_min", options.get("pre_min", _RULES['pre_min']))
    access_speedup = _self_timed_control(options, cell_type, mux, lookup)
    replica = dict(_mapping(options.get("replica", {})))
    if replica.keys() - {"K", "N", "matched"}:
        raise ValueError("replica accepts only K, N and matched")
    replica_default = dict(lookup.get('replica', {})) if lookup is not None else {}
    if row_class is not None and access_speedup is True:
        reference = lookup['row_classes'][-1] if row_class.get('extrapolated') else row_class
        replica_default.update(reference.get('self_timed', {}).get('replica', {}))
    default_k = replica_default.get('K', 1)
    if access_speedup is True:
        default_k = min(default_k, rows, cols)
    replica_k = replica.get("K", default_k)
    dc_stages = replica.get("N", replica_default.get("N", 9))
    for name, value in (("replica.K", replica_k), ("replica.N", dc_stages)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if replica_k > rows or dc_stages % 2 != 1:
        raise ValueError("replica K must not exceed rows; N must be odd")
    replica_matched = replica.get("matched", True)
    scale_decoder = options.get("scale_decoder", True)
    effort_buffers = options.get("effort_buffers", True)
    canonical_read = options.get("canonical_read", True)
    compact_control = options.get('compact_control', False)
    if any(not isinstance(value, bool) for value in
           (replica_matched, scale_decoder, effort_buffers, canonical_read, compact_control, access_speedup)):
        raise ValueError("replica.matched and circuit feature switches must be boolean")
    if access_speedup and cell_type != 'SRAM_6T_CELL':
        raise ValueError('Parallel access has not been screened for 10T cells')
    if access_speedup and not mux:
        raise ValueError('Self-timed access requires the standard 2:1 column mux')
    architecture = COMPACT_CONTROL_ARCHITECTURE if compact_control else CONTROL_ARCHITECTURE
    if access_speedup:
        architecture = SELF_TIMED_CONTROL_ARCHITECTURE + (':compact' if compact_control else ':schematic')
    if not replica_matched or not canonical_read:
        raise ValueError('Distributed wiring requires a matched replica and canonical read')
    if replica_k > cols:
        raise ValueError("Matched replica K must not exceed columns (wordline gate load)")
    joint_sizing = _joint_sizing(lookup, row_class, col_class, access_speedup and compact_control)
    if mode == "lookup":
        scales = {name: float(row_class[name]) for name in ("pre", "wd_out", "wd_in")}
        scales.update({name: float(joint_sizing[name]) for name in _ROW_SCALES if name in joint_sizing})
        scales.update({name: float(col_class[name]) for name in ("wl_inv", "wl_nand")})
        # Column caps on the bitline drivers (control-bar load); the write
        # input stage follows the capped output stage.
        source = lookup["column_classes"][-1] if col_class.get("extrapolated") else col_class
        for cap, name in _COLUMN_CAPS.items():
            if cap in source and scales[name] > float(source[cap]):
                if name == "wd_out":
                    scales["wd_in"] = max(0.5, scales["wd_in"] * float(source[cap]) / scales[name])
                scales[name] = float(source[cap])
    else:
        wd_out = max(_RULES['wd_floor'][cell_type] * margin, k_w * p * rows)
        scales = {'pre': max(pre_min, p * rows / _RULES['pre_rows_divisor']), 'wd_out': wd_out,
                      'wd_in': max(_RULES['wd_input_floor'], wd_out / _RULES['wd_input_divisor']),
                      'wl_inv': max(1.0, cols / _RULES['wl_inv_cols_divisor']),
                      'wl_nand': max(1.0, cols / _RULES['wl_nand_cols_divisor'])}

    pre = sram_config.precharge
    wd = sram_config.write_driver
    wl = sram_config.wordline_driver
    pre_width = _positive("precharge width", pre.pmos_width.value) * scales["pre"]
    wn = _positive("write NMOS width", wd.nmos_width.value)
    wp = _positive("write PMOS width", wd.pmos_width.value)
    nand_gate = (_positive("WL NAND NMOS width", wl.nmos_width.value[0])
                 + _positive("WL NAND PMOS width", wl.pmos_width.value[0]))
    # Normalize unit conversion noise before TIME_CONTROL applies ceil(load / 32).
    # The nominal 0.18 + 0.27 um gate must be exactly one unit, not 1 + epsilon.
    nand_units = round(nand_gate / 0.45e-6, 12)
    rc_input_units = (_positive('pi_cap', context.get('pi_cap', _RULES['peripheral_rc_cap_f'])) / _RULES['unit_inverter_cap_f']
                      if context.get('w_rc', False) else 0.0)
    stages = _stage_policy(options, rows, cols, access_speedup)
    guard_stages = options.get('precharge_guard_stages', stages['wordline_off_stages'])
    if type(guard_stages) is not int or guard_stages < 0 or guard_stages % 2:
        raise ValueError('precharge_guard_stages must be a nonnegative even integer')
    # Wordline-driver A/B and sense-amplifier EN/ISO each have two RC sections;
    # precharge and write-driver enables have one. TIME_CONTROL has no RC wrapper.
    rc_wl_units = 2 * rc_input_units / 1.25
    rc_sa_units = 2 * rc_input_units
    num_sa = cols // (2 if mux else 1) + 1
    sa = sram_config.senseamp
    sa_n_units = round(_positive('SA NMOS width', sa.nmos_width.value) / 0.36e-6, 12)
    sa_iso_units = round(2 * (4 / 3) * _positive('SA PMOS width', sa.pmos_width.value) / 0.36e-6, 12)
    # Retain the V2.1.10 write-enable buffer capacity as a conservative floor.
    # The latch-enable inverter is gone in V2.2.0; its former allowance more
    # than covers the new write-ready observer. wenb_scale is legacy metadata,
    # not an instantiated enable line. Peripheral driver classes stay fixed.
    latch_scale = max(1.0, scales["wd_in"])
    wenb_scale = max(1, ceil(2 * 0.45 * cols * latch_scale / 0.36 / 8.0))
    wl_load = (rows * scales["wl_nand"] * nand_units
               + scales["wl_nand"] * nand_units + (rows + 1) * rc_wl_units)
    wl_en_scale = max(1, ceil(wl_load / (24.0 if effort_buffers else 32.0)))
    # V2.1.8: s_en_bar drives the wordline-request NAND2 and the write-slot
    # NAND3 inputs (1.25 units each); the enables-off NOR adds 1.75 units to
    # each enable.  Keep these in step with resolve_control_sizing().
    senb_scale = max(1, ceil(1.25 * (wl_en_scale + 1) / 4.0))
    from sram_compiler.config_yaml.time_control_config import resolve_time_control_config
    controller = resolve_time_control_config(getattr(sram_config, 'time_control', None))
    observer_units = sum(controller.physical_width(w) for w in controller.widths('observer')) / controller.sizing.unit_input_width_m
    far_enable_units = sum(controller.widths('far_enables_nor')) / controller.sizing.unit_input_width_m
    loads = DriverLoads(
        # The access guard observes far PRE with its configured inverter.
        pre_load=(cols + 1) * 3 * pre_width / 0.36e-6 + (cols + 1) * rc_input_units + observer_units,
        # The replica write driver is enabled with the real ones (write slot).
        wen_load=(cols + 1) * (2 * wn * scales["wd_out"] + (wn + wp) * scales["wd_in"])
        / 0.36e-6 + (0 if compact_control else 4 * wenb_scale) + (cols + 1) * rc_input_units + 1.75,
        # The matched replica driver adds one NAND2 with the same local RC.
        wl_load=wl_load,
        num_sa=num_sa,
        wenb_scale=wenb_scale,
        sen_load=num_sa * (sa_n_units + rc_sa_units) + 3.5 + (0 if compact_control else senb_scale) + 1.75,
        iso_load=num_sa * (sa_iso_units + rc_sa_units),
        sen_effort=3.0 if rc_input_units else 4.0,
        # Wordline-request NAND2 input (1.25 units per unit of wl_en scale) plus
        # the sense NAND3 request input.
        access_load=1.25 * wl_en_scale + 2.5,
        observer_load_units=observer_units,
        far_enable_observer_load_units=far_enable_units,
    )
    periphery = _peripheral_inputs(sram_config)
    pdk = _pdk_inputs(cfg)
    cell = getattr(sram_config, cell_type.lower())
    decoder_units = round((float(sram_config.decoder.nmos_width.value[1])
                           + float(sram_config.decoder.pmos_width.value[1])) / 0.36e-6, 12)
    decoder_units = _positive('decoder inverter gate units', decoder_units)
    if not scale_decoder:
        dec_inv = 1.0
    elif mode == "lookup":
        # A fixed class per column range; the RC stub load is covered by rounding up.
        dec_inv = float(col_class['dec_inv'])
    else:
        dec_inv = max(1.0, scales["wl_nand"] * nand_units / (_RULES['decoder_divisor'] * decoder_units)
                      + 2 * rc_input_units / (5 * decoder_units))
    path_options: dict[str, Any] = {
        'dec_inv': dec_inv,
        'replica_matched': replica_matched, 'replica_k': replica_k, 'dc_stages': dc_stages,
        'effort_buffers': effort_buffers, 'canonical_read': canonical_read,
        'replica_precharge_guard': True,
        'precharge_guard_stages': guard_stages,
        'precharge_off_guard': True, 'precharge_off_guard_stages': stages['precharge_off_stages'],
        'wordline_setup_stages': stages['setup_stages'],
        'isolation_guard_stages': stages['isolation_stages'], 'enable_off_guard_stages': stages['enable_off_stages'],
        'precharge_off_tau': _precharge_off_tau(cols, loads.pre_load, context),
        'enable_off_tau': _enable_off_tau(cols, loads, context),
        'isolation_tau': _control_settling_tau(cols, loads.iso_load + observer_units, context, 'isolation_control'),
        'replica_nmos_models': tuple(cell.nmos_model.value),
        'replica_nmos_widths': tuple(cell.nmos_width.value),
        'replica_pmos_model': cell.pmos_model.value,
        'replica_pmos_width': cell.pmos_width.value, 'replica_length': cell.length.value,
    }
    if compact_control:
        path_options['compact_control'] = True
    if access_speedup:
        path_options['access_speedup'] = True
        path_options.update({name:stages[name] for name in _COMPLETION_STAGES})
        path_options['write_drive_guard_stages']=stages['write_drive_stages']
    if 'delay_driver_scale' in joint_sizing:
        path_options.update(delay_driver_scale=float(joint_sizing['delay_driver_scale']),
                            delay_loads_per_stage=joint_sizing['delay_loads_per_stage'])
    baseline = {
        "version": RULE_VERSION, "control_architecture": architecture, "cell_type": cell_type,
        "cell": _circuit_inputs(getattr(sram_config, cell_type.lower())),
        "periphery": periphery, "pdk": pdk, "rows": rows, "cols": cols, "mux": mux,
        "mode": mode, "parasitic_factor": p, "wd_floor_margin": margin,
        "k_w": k_w, "pre_min": pre_min, "scales": scales,
        "path_options": path_options,
        "rules": _RULES,
        "rc_input_units": rc_input_units,
        "physical": context,
    }
    size_class, extrapolated = "", False
    if lookup is not None:
        extrapolated = bool(row_class["extrapolated"] or col_class["extrapolated"])
        size_class = "/".join(f"{name}<={entry[bound]}{' (extrapolated)' if entry['extrapolated'] else ''}"
                              for name, bound, entry in (("rows", "max_rows", row_class),
                                                         ("cols", "max_cols", col_class)))
        baseline["lookup"] = {"lookup_version": lookup.get("lookup_version"), "sha256": lookup["sha256"],
                              "row_class": row_class, "column_class": col_class}
    result = DriverSizes(
        rows=rows, cols=cols, cell_type=cell_type, mux=mux, **scales, loads=loads,
        control_architecture=architecture,
        source="lookup" if mode == "lookup" else "rule",
        size_class=size_class, extrapolated=extrapolated,
        key=_digest(baseline), peripheral_key=_digest(periphery), pdk_key=_digest(pdk),
        physical_key=_digest(context),
        lookup_sha256=lookup['sha256'] if lookup is not None else '',
        sizing_policy_key=_digest({'mode':mode,'parasitic_factor':p,'wd_floor_margin':margin,
            'k_w':k_w,'pre_min':pre_min,'scale_decoder':scale_decoder,'effort_buffers':effort_buffers,
            'canonical_read':canonical_read,'replica_matched':replica_matched,
            'compact_control':compact_control,'access_speedup':access_speedup,
            'precharge_guard_override':options.get('precharge_guard_stages')}),
        area_precharge_width=pre_width,
        area_wordline_width=max(
            _positive('WL NAND NMOS width', wl.nmos_width.value[0]) * scales['wl_nand'],
            _positive('WL NAND PMOS width', wl.pmos_width.value[0]) * scales['wl_nand'],
            _positive('WL inverter NMOS width', wl.nmos_width.value[1]) * scales['wl_inv'],
            _positive('WL inverter PMOS width', wl.pmos_width.value[1]) * scales['wl_inv'],
        ),
        area_senseamp_width=max(_positive('SA NMOS width', sram_config.senseamp.nmos_width.value),
                               _positive('SA PMOS width', sram_config.senseamp.pmos_width.value)),
        **path_options,
    )
    if requested_mode == 'auto':
        from dataclasses import replace

        from .table import lookup_record, record_key
        record = lookup_record(result, options.get('table'), context)
        if record is not None:
            result = replace(result, source='table', qualified_context_key=record_key(result.key, record['physical']))
    return result
