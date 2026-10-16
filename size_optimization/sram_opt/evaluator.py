"""Nominal V2.2.6 circuit adapter for the SramOpt joint design space."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


Evaluation = Tuple[Sequence[float], Sequence[float], Optional[Dict[str, Any]], bool]


def system_metrics(
    params: Dict[str, Any], raw: Dict[str, float], area: float
) -> Dict[str, float]:
    """Apply SramOpt's 64-bit macro scaling after validating raw measures."""
    required = (
        "hold_snm", "read_snm", "write_snm", "read_delay", "write_delay",
        "read_pstc", "read_pdyn", "write_pstc", "write_pdyn",
    )
    if not all(name in raw and math.isfinite(raw[name]) for name in required):
        raise ValueError("SPICE returned missing or non-finite measurements")
    for name in ("read_delay", "write_delay"):
        if raw[name] <= 0.0:
            raise ValueError(f"invalid raw {name}: {raw[name]}")
    if any(raw[name] < 0.0 for name in required if name.endswith(("pstc", "pdyn"))):
        raise ValueError("SPICE returned negative power measurements")
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("area must be positive and finite")

    arrays, cols = int(params["num_arrays"]), int(params["cols"])
    log_arrays = math.ceil(math.log2(arrays)) if arrays > 1 else 0
    cs_delay = 4.167213500e-11 * log_arrays
    mux_levels = math.log2(max(1, cols // 64))
    width_factor = max(1, 64 // cols)
    read_delay = (
        raw["read_delay"] + cs_delay + (log_arrays + mux_levels) * 1.8e-10
    ) * width_factor
    write_delay = (raw["write_delay"] + cs_delay) * width_factor
    read_power = raw["read_pstc"] * arrays + raw["read_pdyn"] * width_factor
    write_power = raw["write_pstc"] * arrays + raw["write_pdyn"] * width_factor
    if arrays > 1:
        read_power += ((arrays - 1) * log_arrays + mux_levels) * 0.1e-6
    min_snm = min(raw["hold_snm"], raw["read_snm"], raw["write_snm"])
    return {
        **raw,
        "raw_read_delay": raw["read_delay"],
        "raw_write_delay": raw["write_delay"],
        "read_delay": read_delay,
        "write_delay": write_delay,
        "max_delay": max(read_delay, write_delay),
        "read_power": read_power,
        "write_power": write_power,
        "max_power": max(read_power, write_power),
        "min_snm": min_snm,
        "area": area,
        "single_array_area": area / arrays,
    }


class NominalCircuitEvaluator:
    """Use the current compiler, frozen sizing/timing, and nominal cross mode."""

    def __init__(self, output_dir: Path, *, seed: int = 42) -> None:
        self.output_dir = output_dir.resolve()
        self.seed = int(seed)
        self.iteration = 0

    def __call__(self, params: Dict[str, Any]) -> Evaluation:
        from sram_compiler.per_device_mc.run import load_config
        from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
        from size_optimization.exp_utils import apply_params_to_sram_config
        from utils import estimate_scaled_array_area

        rows, cols = int(params["rows"]), int(params["cols"])
        arrays = int(params["num_arrays"])
        self.iteration += 1
        path = self.output_dir / f"iter_{self.iteration:04d}"
        path.mkdir(parents=True, exist_ok=True)
        config = load_config(rows, cols, "TT")
        apply_params_to_sram_config(config, params)
        temperature = float(config.global_config.temperature)
        testbench = Sram6TCoreMcTestbench(
            config,
            sram_cell_type="SRAM_6T_CELL",
            corner="TT",
            choose_columnmux=False,
            real_cell_mode=1,
            variation_mode="nominal",
            mc=False,
            mc_seed=self.seed,
            q_init_val=0,
            sim_path=str(path),
            enable_waveform=False,
        )

        def scalar(value: Any) -> float:
            values = np.asarray(value, dtype=float).reshape(-1)
            if len(values) != 1:
                raise ValueError("nominal evaluation must return exactly one sample")
            return float(values[0])

        try:
            raw = {}
            for operation in ("hold_snm", "read_snm", "write_snm"):
                raw[operation] = scalar(testbench.run_mc_simulation(
                    operation, rows - 1, cols - 1, mc_runs=1, temperature=temperature
                ))
            for operation in ("read", "write"):
                delay, _, static, dynamic = testbench.run_mc_simulation(
                    operation, rows - 1, cols - 1, mc_runs=1, temperature=temperature
                )
                raw[f"{operation}_delay"] = scalar(delay)
                raw[f"{operation}_pstc"] = scalar(static)
                raw[f"{operation}_pdyn"] = scalar(dynamic)
            area = estimate_scaled_array_area(
                rows, cols, arrays,
                w_access=params["pg_width"], w_pd=params["pd_width"],
                w_pu=params["pu_width"], l_transistor=params["length"],
                sa_max_width=max(params["sa_p_width"], params["sa_n_width"]),
                wld_max_width=max(params[name] for name in (
                    "wld_nand_p_width", "wld_inv_p_width",
                    "wld_nand_n_width", "wld_inv_n_width",
                )),
                prc_max_width=params["prc_p_width"],
            )
            result = system_metrics(params, raw, float(area))
            result.update(
                real_cell_mode=1, variation_mode="nominal", corner="TT",
                seed=self.seed, waveform_validated=False,
                full_array_yield_validated=False,
            )
            return (
                [-result["min_snm"], result["max_power"], area, result["max_delay"]],
                [0.0], result, True,
            )
        except (ValueError, RuntimeError, OSError, TypeError) as exc:
            print(f"SramOpt nominal circuit evaluation failed: {exc}")
            return [0.0, 10.0, 5e-6, 5e-8], [1.0], None, False
