#!/usr/bin/env python3
"""OpenYield main entrance: generate one SRAM array deck and simulate it with Xyce.

Per-device local mismatch is the default variation model. The PDK corner stays
fixed while every retained MOS receives independent ``vth0``, ``u0`` and ``voff``
draws, and the fixed ``MC_SEED`` makes the sample reproducible. Configuration is
read from the tracked YAML files in memory; the settings below override the
array geometry and 6T cell sizes without rewriting those files.

Edit the settings block, then run from the repository root inside the
``openyield`` conda environment (Xyce must be on PATH):

    python main_sram.py

For batch or scripted generation use ``python -m sram_compiler.per_device_mc.run``,
which exposes the same defaults as command-line options.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, List, Optional, Sequence

import numpy as np
from PySpice.Unit import u_Ohm, u_pF

from sram_compiler.config_yaml.config import SRAM_CONFIG
from sram_compiler.per_device_mc.run import get_custom_vars, load_config, resolve_mc_runs
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from utils import estimate_bitcell_area  # type: ignore

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================ Settings ============================
# Array architecture: [num_rows, num_cols, choose_columnmux]
ARRAY: List[Any] = [16, 16, False]
CORNER = "TT"                   # PDK corner: TT / FF / SS / FS / SF
# 6T cell, applied when global.yaml selects SRAM_6T_CELL (widths/length in metres):
# [pd_width, pg_width, pu_width, length, pd_model, pg_model, pu_model]
CELL_6T: List[Any] = [2.05e-7, 1.35e-7, 9.0e-8, 50.0e-9, "NMOS_VTG", "NMOS_VTG", "PMOS_VTG"]
OPERATION = "write"             # read | write | read&write | hold_snm | read_snm | write_snm
VARIATION_MODE = "per-device"   # per-device (default) | nominal | shared | custom
MC_RUNS: Optional[int] = None   # None: monte_carlo_runs from global.yaml
MC_SEED: Optional[int] = 20260711  # Xyce sampling seed; None draws a new seed every run
REAL_CELL_MODE = 0              # 0: full transistor array (complete local coverage);
                                # 1-4: equivalent cells for the unused array (approximation)
W_RC = True                     # Add the per-pin RC stubs (100 ohm / 1 fF)
TARGET: Optional[Sequence[int]] = None  # (row, col); None: last row, last column
# ==================================================================

_TRANSIENT_OPERATIONS = ("read", "write", "read&write")


def configure(rows: int, cols: int, choose_columnmux: bool, corner: str,
              cell_6t: Optional[Sequence[Any]] = None) -> SRAM_CONFIG:
    """Load the tracked YAML files in memory and apply the script settings."""
    config = load_config(rows, cols, corner)
    config.global_config.choose_columnmux = bool(choose_columnmux)
    if cell_6t is not None and config.global_config.sram_cell_type == "SRAM_6T_CELL":
        if len(cell_6t) != 7:
            raise ValueError("CELL_6T needs [pd_width, pg_width, pu_width, length, "
                             "pd_model, pg_model, pu_model]")
        pd_width, pg_width, pu_width, length, pd_model, pg_model, pu_model = cell_6t
        cell = config.sram_6t_cell
        cell.nmos_width.value = [float(pd_width), float(pg_width)]
        cell.pmos_width.value = float(pu_width)
        cell.length.value = float(length)
        cell.nmos_model.value = [str(pd_model), str(pg_model)]
        cell.pmos_model.value = str(pu_model)
    return config


def bitcell_area(config: SRAM_CONFIG) -> float:
    cell_type = config.global_config.sram_cell_type
    cell = config.sram_6t_cell if cell_type == "SRAM_6T_CELL" else config.sram_10t_cell
    extra = {} if cell_type == "SRAM_6T_CELL" else {
        "w_fd": cell.nmos_width.value[2], "cell_type": cell_type}
    return estimate_bitcell_area(
        w_access=cell.nmos_width.value[1],
        w_pd=cell.nmos_width.value[0],
        w_pu=cell.pmos_width.value,
        l_transistor=cell.length.value,
        **extra,
    )


def build_testbench(config: SRAM_CONFIG, sim_path: str, *,
                    variation_mode: str = VARIATION_MODE,
                    mc_seed: Optional[int] = MC_SEED,
                    real_cell_mode: int = REAL_CELL_MODE,
                    w_rc: bool = W_RC) -> Sram6TCoreMcTestbench:
    """Build the Monte Carlo testbench with the per-device default made explicit."""
    return Sram6TCoreMcTestbench(
        config,
        sram_cell_type=config.global_config.sram_cell_type,
        w_rc=w_rc,
        pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
        vth_std=0.05,  # relative sigma of vth0/u0/voff for shared and per-device modes
        mc=variation_mode in ("shared", "per-device"),
        custom_mc=variation_mode == "custom",
        variation_mode=variation_mode,
        mc_seed=mc_seed,
        sweep_cell=False,
        sweep_precharge=False,
        sweep_senseamp=False,
        sweep_wordlinedriver=False,
        sweep_columnmux=False,
        sweep_writedriver=False,
        sweep_decoder=False,
        corner=config.global_config.corner,
        choose_columnmux=bool(config.global_config.choose_columnmux),
        real_cell_mode=real_cell_mode,
        q_init_val=0,
        sim_path=sim_path,
    )


def _mean(values: Any) -> float:
    return float(np.mean(np.asarray(values, dtype=float).ravel()))


def main() -> None:
    rows, cols, choose_columnmux = ARRAY
    config = configure(int(rows), int(cols), bool(choose_columnmux), CORNER, CELL_6T)
    cell_type = config.global_config.sram_cell_type
    custom_vars = get_custom_vars(config, cell_type) if VARIATION_MODE == "custom" else None
    requested = int(config.global_config.monte_carlo_runs) if MC_RUNS is None else int(MC_RUNS)
    mc_runs = resolve_mc_runs(requested, VARIATION_MODE, custom_vars)
    target_row, target_col = (rows - 1, cols - 1) if TARGET is None else TARGET

    suffix = "6t" if cell_type == "SRAM_6T_CELL" else "10t"
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_path = os.path.join(_PROJECT_ROOT, "sim1", f"{time_str}_mc_{suffix}")
    os.makedirs(sim_path, exist_ok=True)

    area = bitcell_area(config)
    print(f"Estimated {suffix.upper()} SRAM Cell Area: {area*1e12:.2f} µm²")
    print(f"[INPUT] array: num_rows={rows}, num_cols={cols}, choose_columnmux={choose_columnmux}, "
          f"corner={CORNER}, cell={cell_type}, target=({target_row}, {target_col})")
    if cell_type == "SRAM_6T_CELL":
        print(f"[INPUT] sram6tcell_param: pd_width={CELL_6T[0]*1e9:.1f} nm, pg_width={CELL_6T[1]*1e9:.1f} nm, "
              f"pu_width={CELL_6T[2]*1e9:.1f} nm, length={CELL_6T[3]*1e9:.1f} nm, "
              f"pd_model={CELL_6T[4]}, pg_model={CELL_6T[5]}, pu_model={CELL_6T[6]}")
    print(f"[INPUT] variation: mode={VARIATION_MODE}, mc_runs={mc_runs}, seed={MC_SEED}, "
          f"real_cell_mode={REAL_CELL_MODE}, w_rc={W_RC}, operation={OPERATION}")

    print(f"===== {suffix.upper()} SRAM Array Monte Carlo Simulation ({VARIATION_MODE}) =====")
    testbench = build_testbench(config, sim_path)
    temperature = config.global_config.temperature

    if OPERATION in _TRANSIENT_OPERATIONS:
        delay, pavg, pstc, pdyn = testbench.run_mc_simulation(
            operation=OPERATION, target_row=target_row, target_col=target_col,
            mc_runs=mc_runs, temperature=temperature, vars=custom_vars,
        )
        y = np.array([_mean(delay), _mean(np.asarray(pstc) + np.asarray(pdyn)), area])
        print(f"[OUTPUT] mean of {mc_runs} sample(s): y[0]=Delay({y[0]*1e9:.3f} ns), "
              f"y[1]=Power({y[1]*1e6:.2f} uW), y[2]=Area({y[2]*1e12:.2f} µm²)")
    elif OPERATION in ("hold_snm", "write_snm", "read_snm"):
        snm = testbench.run_mc_simulation(
            operation=OPERATION, target_row=target_row, target_col=target_col,
            mc_runs=mc_runs, temperature=temperature, vars=custom_vars,
        )
        print(f"[OUTPUT] {OPERATION}: {snm}")
    else:
        raise ValueError(f"Unknown OPERATION: {OPERATION}")

    print(f"[DEBUG] Monte Carlo simulation completed; outputs in {sim_path}")


if __name__ == "__main__":
    main()
