"""Variation policy: independent per-device mismatch is the default."""
from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sram_compiler.config_yaml.config import SRAM_CONFIG

VARIATION_MODES = ("nominal", "shared", "custom", "per-device")
DEFAULT_VARIATION_MODE = "per-device"


def resolve_mc_runs(
    requested: int | None,
    variation_mode: str,
    custom_vars: list[list[float]] | None,
) -> int:
    if variation_mode not in VARIATION_MODES:
        raise ValueError(f'Unknown variation_mode: {variation_mode}')
    if requested is not None and (isinstance(requested, bool)
                                 or not isinstance(requested, Integral)
                                 or requested <= 0):
        raise ValueError("mc_runs must be a positive integer")
    if variation_mode == "nominal":
        if requested not in (None, 1):
            raise ValueError("nominal variation uses exactly one run")
        return 1
    if variation_mode == "custom" and not custom_vars:
        raise ValueError(
            "custom variation requires process_parameters.vars in the cell YAML"
        )
    if variation_mode == "custom":
        sample_count = len(custom_vars)
        if requested is not None and requested != sample_count:
            raise ValueError(
                "custom variation requires mc_runs to match the "
                f"{sample_count} rows in process_parameters.vars"
            )
        return sample_count
    if requested is not None:
        return int(requested)
    return 100


def get_custom_vars(config: SRAM_CONFIG, cell_type: str) -> list[list[float]] | None:
    cell_config = (
        config.sram_6t_cell if cell_type == "SRAM_6T_CELL" else config.sram_10t_cell
    )
    values = getattr(cell_config.process_parameters, "vars", None)
    if not values:
        return None
    if all(isinstance(value, (int, float)) for value in values):
        return [[float(value) for value in values]]
    if not all(isinstance(row, (list, tuple)) for row in values):
        raise ValueError("process_parameters.vars must be numeric rows")
    return [[float(value) for value in row] for row in values]
