"""Driver rules and exact qualification lookup for the V2.0.5 proposal.

Resolve once from the baseline, then pass the same result to each cell candidate.
This module does not run Xyce. Rule predictions remain unverified; only an exact
qualified-table match supplies evidence for the baseline physical configuration.
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
_PERIPHERALS = (
    "precharge", "write_driver", "wordline_driver", "decoder", "column_mux", "senseamp",
)
_SCALES = ("pre", "wd_in", "wd_out", "wl_inv", "wl_nand")


@dataclass(frozen=True)
class DriverLoads:
    """TIME input loads: inverter units except wl_load, which uses NAND units."""

    pre_load: float
    wen_load: float
    wl_load: float
    num_sa: int
    wenb_scale: int
    sen_load: float | None = None
    iso_load: float | None = None
    sen_effort: float = 8.0


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
    replica_matched: bool = False
    replica_k: int = 1
    dc_stages: int = 9
    effort_buffers: bool = False
    canonical_read: bool = False
    replica_precharge_guard: bool = False
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

    def to_dict(self):
        """JSON-ready experiment metadata, including the result's evidence source."""
        return asdict(self)

    def validate_for(self, sram_config, cell_type, mux, context=None):
        """Allow cell candidates, but reject changed geometry or periphery."""
        cfg = sram_config.global_config
        if (cfg.num_rows, cfg.num_cols, cell_type, mux) != (
            self.rows, self.cols, self.cell_type, self.mux
        ):
            raise ValueError("Frozen driver sizes belong to a different array architecture")
        if _digest(_peripheral_inputs(sram_config)) != self.peripheral_key:
            raise ValueError("Frozen driver sizes require the baseline peripheral configuration")
        if _digest(_pdk_inputs(cfg)) != self.pdk_key:
            raise ValueError("Frozen driver sizes require the baseline PDK model contents")
        if self.source == 'table' and context is not None:
            from .table import record_key
            if self.qualified_context_key != record_key(self.key, context):
                raise ValueError('Qualified driver sizes belong to a different physical context')


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _circuit_inputs(config):
    # Descriptions and sample vectors do not change baseline qualification.
    return {
        name: {field: getattr(param, field) for field in ("value", "lower", "upper", "choices")}
        for name, param in config.parameters.items()
    }


def _peripheral_inputs(sram_config):
    return {name: _circuit_inputs(getattr(sram_config, name)) for name in _PERIPHERALS}


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


def resolve_driver_sizes(sram_config, *, cell_type=None, mux=None, sizing=None, physical_context=None):
    """Return immutable scales/loads without mutating configuration or simulating.

    ``fixed`` preserves legacy numeric array rules, optionally overridden through
    ``fixed_scales``. ``rules_only`` opts into the proposed floors and load rules.
    ``auto`` uses an exact qualified-table match or falls back to unverified rules.
    The result's key describes the baseline, not any later candidate cell/PVT.
    """
    cfg = sram_config.global_config
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
        "mode", "parasitic_factor", "wd_floor_margin", "k_w", "pre_min", "fixed_scales",
        "replica", "scale_decoder", "effort_buffers", "canonical_read",
        "table",
    }
    if unknown:
        raise ValueError(f"Unsupported sizing options: {sorted(unknown)}")
    mode = options.get("mode", "fixed")
    if mode not in ("fixed", "rules_only", "auto"):
        raise ValueError("sizing.mode must be fixed, rules_only or auto")
    requested_mode = mode
    mode = 'rules_only' if mode == 'auto' else mode
    p = _positive("parasitic_factor", options.get("parasitic_factor", 1.0))
    margin = _positive("wd_floor_margin", options.get("wd_floor_margin", _RULES['wd_floor_margin']))
    k_w = _positive("k_w", options.get("k_w", _RULES['k_w']))
    pre_min = _positive("pre_min", options.get("pre_min", _RULES['pre_min']))
    replica = dict(_mapping(options.get("replica", {})))
    if replica.keys() - {"K", "N", "matched"}:
        raise ValueError("replica accepts only K, N and matched")
    replica_k, dc_stages = replica.get("K", 1), replica.get("N", 9)
    for name, value in (("replica.K", replica_k), ("replica.N", dc_stages)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if replica_k > rows + 1 or dc_stages % 2 != 1:
        raise ValueError("replica K must not exceed rows + 1; N must be odd")
    replica_matched = replica.get("matched", mode != "fixed")
    scale_decoder = options.get("scale_decoder", mode != "fixed")
    effort_buffers = options.get("effort_buffers", mode != "fixed")
    canonical_read = options.get("canonical_read", mode != "fixed")
    if any(not isinstance(value, bool) for value in
           (replica_matched, scale_decoder, effort_buffers, canonical_read)):
        raise ValueError("replica.matched and circuit feature switches must be boolean")
    if replica_matched and replica_k > cols:
        raise ValueError("Matched replica K must not exceed columns (wordline gate load)")
    wl_inv = max(1.0, cols / 4.0)
    if mode == "rules_only":
        if "fixed_scales" in options:
            raise ValueError("fixed_scales requires sizing.mode=fixed")
        wd_out = max(_RULES['wd_floor'][cell_type] * margin, k_w * p * rows)
        scales = {'pre': max(pre_min, p * rows / _RULES['pre_rows_divisor']), 'wd_out': wd_out,
                      'wd_in': max(_RULES['wd_input_floor'], wd_out / _RULES['wd_input_divisor']),
                      'wl_inv': max(1.0, cols / _RULES['wl_inv_cols_divisor']),
                      'wl_nand': max(1.0, cols / _RULES['wl_nand_cols_divisor'])}
    else:
        scales = {'pre': max(0.5, rows / 16.0), 'wd_out': max(8, rows) / 16.0,
                      'wd_in': max(8, rows) / 16.0, 'wl_inv': wl_inv, 'wl_nand': wl_inv ** 0.5}
        overrides = dict(_mapping(options.get("fixed_scales", {})))
        if overrides.keys() - set(_SCALES):
            raise ValueError(f"Unknown fixed scales: {sorted(overrides.keys() - set(_SCALES))}")
        scales.update({name: _positive(name, value) for name, value in overrides.items()})

    pre = sram_config.precharge
    wd = sram_config.write_driver
    wl = sram_config.wordline_driver
    pre_width = _positive("precharge width", pre.pmos_width.value) * scales["pre"]
    wn = _positive("write NMOS width", wd.nmos_width.value)
    wp = _positive("write PMOS width", wd.pmos_width.value)
    nand_gate = (_positive("WL NAND NMOS width", wl.nmos_width.value[0])
                 + _positive("WL NAND PMOS width", wl.pmos_width.value[0]))
    # Normalize unit conversion noise before TIME applies ceil(load / 32).
    # The nominal 0.18 + 0.27 um gate must be exactly one unit, not 1 + epsilon.
    nand_units = round(nand_gate / 0.45e-6, 12)
    context = {} if physical_context is None else _mapping(physical_context)
    rc_input_units = (_RULES['peripheral_rc_cap_f'] / _RULES['unit_inverter_cap_f']
                      if mode != 'fixed' and context.get('w_rc', False) else 0.0)
    # Wordline-driver A and B each have two RC sections; other peripheral
    # enables have one. TIME itself has no optional RC wrapper in this compiler.
    rc_wl_units = 2 * rc_input_units / 1.25
    num_sa = cols // (2 if mux else 1)
    sa = sram_config.senseamp
    sa_n_units = round(_positive('SA NMOS width', sa.nmos_width.value) / 0.36e-6, 12)
    sa_iso_units = round(2 * (4 / 3) * _positive('SA PMOS width', sa.pmos_width.value) / 0.36e-6, 12)
    wenb_scale = max(1, ceil(2 * 0.45 * cols / 0.36 / 8.0))
    loads = DriverLoads(
        pre_load=(cols + 1) * 3 * pre_width / 0.36e-6 + (cols + 1) * rc_input_units,
        wen_load=cols * (2 * wn * scales["wd_out"] + (wn + wp) * scales["wd_in"])
        / 0.36e-6 + 4 * wenb_scale + cols * rc_input_units,
        # Include the fixed replica AND2 in new rules. Legacy fixed mode keeps
        # its historical load accounting for reproducible default decks.
        wl_load=rows * scales["wl_nand"] * nand_units
        + (scales["wl_nand"] * nand_units if replica_matched else
           (1.0 if mode == "rules_only" else 0.0))
        + (rows + int(replica_matched)) * rc_wl_units,
        num_sa=num_sa,
        wenb_scale=wenb_scale,
        sen_load=num_sa * (sa_n_units + rc_input_units) + 3.5 if mode != 'fixed' else None,
        iso_load=num_sa * (sa_iso_units + rc_input_units) if mode != 'fixed' else None,
        sen_effort=3.0 if rc_input_units else (4.0 if mode != 'fixed' else 8.0),
    )
    periphery = _peripheral_inputs(sram_config)
    pdk = _pdk_inputs(cfg)
    cell = getattr(sram_config, cell_type.lower())
    decoder_units = round((float(sram_config.decoder.nmos_width.value[1])
                           + float(sram_config.decoder.pmos_width.value[1])) / 0.36e-6, 12)
    decoder_units = _positive('decoder inverter gate units', decoder_units)
    path_options: dict[str, Any] = {
        'dec_inv': max(1.0, scales["wl_nand"] * nand_units / (_RULES['decoder_divisor'] * decoder_units)
                       + 2 * rc_input_units / (5 * decoder_units)) if scale_decoder else 1.0,
        'replica_matched': replica_matched, 'replica_k': replica_k, 'dc_stages': dc_stages,
        'effort_buffers': effort_buffers, 'canonical_read': canonical_read,
        'replica_precharge_guard': bool(rc_input_units and replica_matched),
        'replica_nmos_models': tuple(cell.nmos_model.value),
        'replica_nmos_widths': tuple(cell.nmos_width.value),
        'replica_pmos_model': cell.pmos_model.value,
        'replica_pmos_width': cell.pmos_width.value, 'replica_length': cell.length.value,
    }
    baseline = {
        "version": RULE_VERSION, "cell_type": cell_type,
        "cell": _circuit_inputs(getattr(sram_config, cell_type.lower())),
        "periphery": periphery, "pdk": pdk, "rows": rows, "cols": cols, "mux": mux,
        "mode": mode, "parasitic_factor": p, "wd_floor_margin": margin,
        "k_w": k_w, "pre_min": pre_min, "scales": scales,
        "path_options": path_options,
        "rules": _RULES,
        "rc_input_units": rc_input_units,
    }
    result = DriverSizes(
        rows=rows, cols=cols, cell_type=cell_type, mux=mux, **scales, loads=loads,
        source="rule" if mode == "rules_only" else "fixed",
        key=_digest(baseline), peripheral_key=_digest(periphery), pdk_key=_digest(pdk),
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
        record = lookup_record(result, options.get('table'), physical_context)
        if record is not None:
            result = replace(result, source='table', qualified_context_key=record_key(result.key, record['physical']))
    return result
