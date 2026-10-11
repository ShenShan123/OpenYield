#!/usr/bin/env python3
"""Unified OpenYield entrance for simulation, layout and sizing experiments.

Edit the settings below or override them from the command line. Per-device
vth0/u0/voff mismatch is the default for every retained MOS. Run from the root:

    python main_sram.py --help
    python main_sram.py simulate --rows 8 --cols 4 --no-run-xyce
    python main_sram.py simulate --operation read --mc-runs 10 --run-xyce
    python main_sram.py layout --help
    python main_sram.py experiment --dry-run

Both settings-only runs and CLI simulations use one deck and result workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from sram_compiler.config_yaml.loader import configure, load_config
from sram_compiler.experiments import dispatch
# Preserve the script's Python helper API while sharing the implementation.
from sram_compiler.experiments.simulation import bitcell_area, build_testbench, write_deck
from sram_compiler.testbenches.sram_6t_core_MC_testbench import full_device_coverage
from sram_compiler.variation import get_custom_vars, resolve_mc_runs
from sram_compiler.version import VERSION

_PROJECT_ROOT = str(Path(__file__).resolve().parent)

# ============================ Settings ============================
# Array architecture: [num_rows, num_cols, choose_columnmux]
ARRAY: List[Any] = [16, 16, False]
CORNER = "TT"                   # PDK corner: TT / FF / SS / FS / SF
# 6T cell, applied when global.yaml selects SRAM_6T_CELL (widths/length in metres):
# [pd_width, pg_width, pu_width, length, pd_model, pg_model, pu_model]
CELL_6T: List[Any] = [2.05e-7, 1.35e-7, 9.0e-8, 50.0e-9, "NMOS_VTG", "NMOS_VTG", "PMOS_VTG"]
OPERATION = "write"             # read | write | read&write | hold_snm | read_snm | write_snm
VARIATION_MODE = "per-device"   # per-device (default) | nominal | shared | custom
MC_RUNS: Optional[int] = None   # None: YAML count for sampled modes; custom: table rows
MC_SEED: Optional[int] = 20260711  # Xyce seed; None draws a new seed every run
RUN_XYCE = True                 # False: generate the deck without simulating it
# Equivalent array model; None keeps the `equivalent` block of global.yaml
# (0: full transistor array, complete local coverage; 1-4: equivalent cells for
# the unused array, an approximation). See sram_compiler/equivalent_modeling/.
REAL_CELL_MODE: Optional[int] = None
W_RC = True                     # Local storage/peripheral series RC (100 ohm / 1 fF)
# Interconnect YAML mapping applied in memory (see docs/design/DISTRIBUTED_RC_MODEL.md),
# e.g. "sram_compiler/config_yaml/interconnect_example.yaml"; None keeps global.yaml's
# `interconnect` settings (independent per-net RC for the qualified 8x8 reference;
# separate illustrative route settings outside that scope).
INTERCONNECT_CONFIG: Optional[str] = None
TARGET: Optional[Sequence[int]] = None  # (row, col); None: last row, last column
# ==================================================================


def main(argv: Sequence[str] = ()) -> int:
    """Apply script settings as defaults; explicit CLI options take precedence."""
    rows, cols, column_mux = ARRAY
    target_row, target_col = (None, None) if TARGET is None else TARGET
    return dispatch.main(argv, simulation_defaults={
        'rows': rows, 'cols': cols, 'column_mux': column_mux,
        'corner': CORNER, 'cell_6t': CELL_6T, 'operation': OPERATION,
        'variation_mode': VARIATION_MODE, 'mc_runs': MC_RUNS, 'seed': MC_SEED,
        'run_xyce': RUN_XYCE, 'real_cell_mode': REAL_CELL_MODE, 'w_rc': W_RC,
        'interconnect_config': INTERCONNECT_CONFIG,
        'target_row': target_row, 'target_col': target_col,
        'output_dir': Path(_PROJECT_ROOT) / 'outputs/main_sram',
    })


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
