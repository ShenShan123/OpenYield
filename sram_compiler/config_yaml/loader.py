"""Load compiler YAML and apply experiment overrides without rewriting inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .config import SRAM_CONFIG
from sram_compiler.interconnect import load_interconnect

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(rows: int, cols: int, corner: str) -> SRAM_CONFIG:
    config = SRAM_CONFIG()
    config.load_all_configs(
        global_file=str(PROJECT_ROOT / "sram_compiler/config_yaml/global.yaml"),
        circuit_configs={
            "SRAM_6T_CELL": str(
                PROJECT_ROOT / "sram_compiler/config_yaml/sram_6t_cell.yaml"
            ),
            "SRAM_10T_CELL": str(
                PROJECT_ROOT / "sram_compiler/config_yaml/sram_10t_cell.yaml"
            ),
            "WORDLINEDRIVER": str(
                PROJECT_ROOT / "sram_compiler/config_yaml/wordline_driver.yaml"
            ),
            "PRECHARGE": str(PROJECT_ROOT / "sram_compiler/config_yaml/precharge.yaml"),
            "COLUMNMUX": str(PROJECT_ROOT / "sram_compiler/config_yaml/mux.yaml"),
            "SENSEAMP": str(PROJECT_ROOT / "sram_compiler/config_yaml/sa.yaml"),
            "WRITEDRIVER": str(
                PROJECT_ROOT / "sram_compiler/config_yaml/write_driver.yaml"
            ),
            "DECODER": str(PROJECT_ROOT / "sram_compiler/config_yaml/decoder.yaml"),
        },
    )
    config.global_config.num_rows = rows
    config.global_config.num_cols = cols
    config.global_config.corner = corner
    for name in ("TT", "FF", "SS", "FS", "SF"):
        key = f"pdk_path_{name}"
        path = Path(getattr(config.global_config, key)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        setattr(config.global_config, key, str(path.resolve()))
    return config


def configure(rows: int, cols: int, choose_columnmux: bool, corner: str,
              cell_6t: Optional[Sequence[Any]] = None,
              interconnect_config: Optional[str] = None, *,
              config: Optional[SRAM_CONFIG] = None) -> SRAM_CONFIG:
    """Load the tracked YAML files in memory and apply the script settings."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (rows, cols)):
        raise ValueError('ARRAY rows and columns must be positive integers')
    if not isinstance(choose_columnmux, bool):
        raise ValueError('ARRAY column_mux must be a boolean')
    config = load_config(rows, cols, corner) if config is None else config
    config.global_config.choose_columnmux = choose_columnmux
    if interconnect_config is not None:
        config.global_config.interconnect = load_interconnect(interconnect_config)
    if cell_6t is not None and config.global_config.sram_cell_type == "SRAM_6T_CELL":
        if len(cell_6t) != 7:
            raise ValueError("CELL_6T needs [pd_width, pg_width, pu_width, length, "
                             "pd_model, pg_model, pu_model]")
        pd_width, pg_width, pu_width, length, pd_model, pg_model, pu_model = cell_6t
        if any(not np.isfinite(float(value)) or float(value) <= 0
               for value in (pd_width, pg_width, pu_width, length)):
            raise ValueError('CELL_6T widths and length must be finite and positive metres')
        cell = config.sram_6t_cell
        cell.nmos_width.value = [float(pd_width), float(pg_width)]
        cell.pmos_width.value = float(pu_width)
        cell.length.value = float(length)
        cell.nmos_model.value = [str(pd_model), str(pg_model)]
        cell.pmos_model.value = str(pu_model)
    return config
