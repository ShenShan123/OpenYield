#!/usr/bin/env python3
"""Measure the equivalent array model against the full transistor array.

Mode 0 is the reference; every other mode is scored as a relative error against
mode 0 of the same array, operation and PVT, with every other input (wires,
local RC, driver classes, clock, corner, supply, temperature and variation)
held fixed.  This is an accuracy and runtime measurement, not qualification
evidence: a passing row says the approximation tracks the reference for these
metrics, nothing about waveform correctness.

Run from the repository root with Xyce on PATH (parasitic extraction and the
array simulation both call it):

    python3 -m sram_compiler.equivalent_modeling.compare --sizes 16x16
    python3 -m sram_compiler.equivalent_modeling.compare \
        --sizes 16x16,32x32 --modes 0,1,4 --operations read,write --plot
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PySpice.Unit import u_Ohm, u_pF  # noqa: E402

from sram_compiler.equivalent_modeling import MODES, resolve_equivalent  # noqa: E402
from sram_compiler.interconnect import resolve_interconnect  # noqa: E402
from sram_compiler.per_device_mc.run import load_config  # noqa: E402
from sram_compiler.testbenches.sram_6t_core_MC_testbench import (  # noqa: E402
    Sram6TCoreMcTestbench,
)
from sram_compiler.version import VERSION  # noqa: E402

OPERATIONS = ("read", "write")
METRICS = ("delay", "pavg", "pstc", "pdyn")


def parse_sizes(text: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in text.split(","):
        rows, _, cols = item.strip().lower().partition("x")
        if not cols or not rows.isdigit() or not cols.isdigit():
            raise argparse.ArgumentTypeError(f"--sizes takes ROWSxCOLS entries, got {item!r}")
        sizes.append((int(rows), int(cols)))
    if not sizes:
        raise argparse.ArgumentTypeError("--sizes must list at least one array")
    return sizes


def parse_modes(text: str) -> list[int]:
    modes = [int(item) for item in text.split(",") if item.strip()]
    for mode in modes:
        resolve_equivalent(mode)
    if modes[0] != 0:
        raise argparse.ArgumentTypeError("--modes must start with the reference mode 0")
    return modes


def parse_operations(text: str) -> list[str]:
    operations = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(operations) - set(OPERATIONS))
    if unknown:
        raise argparse.ArgumentTypeError(f"--operations supports {OPERATIONS}, got {unknown}")
    return operations


def mean(values: Any) -> float:
    return float(np.mean(np.asarray(values, dtype=float).ravel()))


def run_case(args: argparse.Namespace, rows: int, cols: int, mode: int,
             operations: Sequence[str], sim_root: Path) -> dict[str, Any]:
    """Simulate one (array, mode) point and return its metrics and runtime."""
    config = load_config(rows, cols, args.corner)
    config.global_config.vdd = args.vdd
    config.global_config.temperature = args.temperature
    config.global_config.choose_columnmux = args.mux
    cell_type = config.global_config.sram_cell_type
    target_row = rows - 1 if args.target_row is None else args.target_row
    target_col = cols - 1 if args.target_col is None else args.target_col
    sim_path = sim_root / f"{rows}x{cols}_mode{mode}"
    sim_path.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "compiler_version": VERSION, "cell_type": cell_type,
        "num_rows": rows, "num_cols": cols, "real_cell_mode": mode,
        "target_row": target_row, "target_col": target_col,
        "corner": args.corner, "vdd": args.vdd, "temperature": args.temperature,
        "mux": args.mux, "variation": args.variation, "seed": args.seed,
        "mc_runs": args.mc_runs,
    }
    started = time.monotonic()
    # The compiler prints construction progress; keep the table readable.
    with contextlib.redirect_stdout(io.StringIO() if args.quiet else sys.stdout):
        testbench = Sram6TCoreMcTestbench(
            config, sram_cell_type=cell_type,
            w_rc=args.w_rc, pi_res=args.pi_res_ohm @ u_Ohm, pi_cap=args.pi_cap_pf @ u_pF,
            vth_std=0.05, variation_mode=args.variation, mc_seed=args.seed,
            corner=args.corner, choose_columnmux=args.mux, temperature=args.temperature,
            real_cell_mode=mode, q_init_val=0, sim_path=str(sim_path),
        )
        record["t_period"] = float(testbench.t_period)
        record["equivalent"] = testbench.equivalent.to_dict()
        record["interconnect"] = resolve_interconnect(testbench.interconnect).to_dict()
        for operation in operations:
            delay, pavg, pstc, pdyn = testbench.run_mc_simulation(
                operation=operation, target_row=target_row, target_col=target_col,
                mc_runs=args.mc_runs, temperature=args.temperature, vars=None,
            )
            prefix = operation[0]
            for name, values in zip(METRICS, (delay, pavg, pstc, pdyn)):
                record[f"{prefix}_{name}"] = mean(values)
    record["time_usage_s"] = time.monotonic() - started
    return record


def relative_errors(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Relative error of every mode against mode 0 of the same array."""
    rows = []
    for (num_rows, num_cols), group in frame.groupby(["num_rows", "num_cols"], sort=False):
        reference = group[group["real_cell_mode"] == 0]
        if reference.empty:
            raise RuntimeError(f"{num_rows}x{num_cols} has no mode 0 reference")
        base = reference.iloc[0]
        for _, row in group[group["real_cell_mode"] != 0].iterrows():
            entry = {"num_rows": num_rows, "num_cols": num_cols,
                     "real_cell_mode": int(row["real_cell_mode"])}
            for column in columns:
                denominator = float(base[column])
                entry[column] = ((float(row[column]) - denominator) / denominator
                                 if denominator else float("nan"))
            rows.append(entry)
    return pd.DataFrame(rows)


def plot(diff: pd.DataFrame, columns: Sequence[str], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{int(r.num_rows)}x{int(r.num_cols)}\nmode {int(r.real_cell_mode)}"
              for r in diff.itertuples()]
    figure, axes = plt.subplots(figsize=(1.1 * len(labels) + 4, 4.5))
    width = 0.8 / len(columns)
    positions = np.arange(len(labels))
    for index, column in enumerate(columns):
        axes.bar(positions + index * width, 100 * diff[column].to_numpy(dtype=float),
                 width=width, label=column)
    axes.set_xticks(positions + 0.4 - width / 2)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylabel("relative error against the full transistor array (%)")
    axes.axhline(0, color="black", linewidth=0.8)
    axes.legend(fontsize=8, ncol=2)
    axes.set_title("Equivalent array model accuracy")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("16x16"),
                        help="comma separated ROWSxCOLS arrays (default: 16x16)")
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("0,1,2,3,4"),
                        help="equivalent modes, the reference mode 0 first; "
                             + "; ".join(f"{k}={v}" for k, v in MODES.items()))
    parser.add_argument("--operations", type=parse_operations,
                        default=parse_operations("read,write"))
    parser.add_argument("--corner", choices=("TT", "FF", "SS", "FS", "SF"), default="TT")
    parser.add_argument("--vdd", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=25.0)
    parser.add_argument("--mux", action="store_true", help="insert the 2:1 column mux")
    parser.add_argument("--variation", choices=("nominal", "per-device"), default="nominal",
                        help="nominal keeps every mode on the same deterministic models; "
                             "per-device mismatch only covers the cells a mode keeps real")
    parser.add_argument("--mc-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--target-row", type=int)
    parser.add_argument("--target-col", type=int)
    parser.add_argument("--w-rc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pi-res-ohm", type=float, default=100.0)
    parser.add_argument("--pi-cap-pf", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/equivalent_modeling")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True,
                        help="hide the compiler's construction output")
    args = parser.parse_args(argv)
    if args.mc_runs <= 0:
        parser.error("--mc-runs must be positive")
    if args.variation == "nominal" and args.mc_runs != 1:
        parser.error("nominal variation uses exactly one run")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    columns = [f"{operation[0]}_{metric}"
               for operation in args.operations for metric in METRICS]

    records: list[dict[str, Any]] = []
    for rows, cols in args.sizes:
        for mode in args.modes:
            record = run_case(args, rows, cols, mode, args.operations, run_dir / "sim")
            records.append(record)
            frame = pd.DataFrame(records)
            frame.to_csv(run_dir / "result.csv", index=False)
            print(json.dumps({key: record[key] for key in
                              ("num_rows", "num_cols", "real_cell_mode", "time_usage_s", *columns)},
                             default=float), flush=True)

    frame = pd.DataFrame(records)
    diff = relative_errors(frame, [*columns, "time_usage_s"])
    diff.to_csv(run_dir / "result_diff.csv", index=False)
    if args.plot:
        plot(diff, columns, run_dir / "result_diff.png")
    (run_dir / "settings.json").write_text(
        json.dumps({**vars(args), "output_dir": str(args.output_dir),
                    "compiler_version": VERSION,
                    "xyce": os.environ.get("XYCE", "Xyce")},
                   default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(diff.to_string(index=False))
    print(f"[DEBUG] results in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
