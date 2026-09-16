#!/usr/bin/env python3
"""Generate and optionally run OpenYield decks with selectable variation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySpice.Unit import u_Ohm, u_pF  # type: ignore  # noqa: E402

from sram_compiler.config_yaml.config import SRAM_CONFIG  # type: ignore  # noqa: E402
from sram_compiler.equivalent_modeling import resolve_equivalent
from sram_compiler.interconnect import load_interconnect, resolve_interconnect
from sram_compiler.sizing import resolve_driver_sizes, resolve_timing
from sram_compiler.sizing.table import physical_context
from sram_compiler.version import VERSION
from sram_compiler.testbenches.sram_6t_core_MC_testbench import (  # type: ignore  # noqa: E402
    Sram6TCoreMcTestbench,
)


VARIATION_MODES = ("nominal", "shared", "custom", "per-device")
OPERATIONS = (
    "read",
    "write",
    "read&write",
    "hold_snm",
    "read_snm",
    "write_snm",
)
SNM_OPERATIONS = frozenset({"hold_snm", "read_snm", "write_snm"})


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


def resolve_mc_runs(
    requested: int | None,
    variation_mode: str,
    custom_vars: list[list[float]] | None,
) -> int:
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
        if requested <= 0:
            raise ValueError("mc_runs must be positive")
        return requested
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


def make_run_name(
    args: argparse.Namespace,
    *,
    cell_type: str,
    target_row: int,
    target_col: int,
    mc_runs: int,
    interconnect=None,
    timing=None,
    vdd=None,
    temperature=None,
) -> str:
    settings = {
        "compiler_version": VERSION,
        "interconnect": resolve_interconnect(interconnect).to_dict(),
        "cell_type": cell_type,
        "rows": args.rows,
        "cols": args.cols,
        "target_row": target_row,
        "target_col": target_col,
        "operation": args.operation,
        "real_cell_mode": args.real_cell_mode,
        "variation_mode": args.variation_mode,
        "mc_runs": mc_runs,
        "vth_std": args.vth_std,
        "corner": args.corner,
        "pi_res_ohm": args.pi_res_ohm,
        "pi_cap_pf": args.pi_cap_pf,
        "q_init_val": args.q_init_val,
        "waveform": args.waveform,
        "seed": args.seed,
        "timing": timing,
        # Another supply or temperature is another configuration, not a repeated attempt.
        "vdd": vdd,
        "temperature": temperature,
    }
    digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    cell_label = "6t" if cell_type == "SRAM_6T_CELL" else "10t"
    operation_label = args.operation.replace("&", "-and-")
    return (
        f"{operation_label}_{args.rows}x{args.cols}_r{target_row}c{target_col}_"
        f"{cell_label}_mode{args.real_cell_mode}_{args.variation_mode}_{args.corner}_{digest}"
    )


def waveform_columns(summary: dict[str, Any]) -> list[str]:
    operation = str(summary["operation"])
    if operation in SNM_OPERATIONS:
        return ["{U}", "V(V1)", "V(V2)"]
    if operation == "read&write":
        return ["V(CLK)", "V(DIN0)", "V(DIN_DFF0)", "V(OUT)"]

    rows = int(summary["rows"])
    cols = int(summary["cols"])
    target_row = int(summary["target_row"])
    target_col = int(summary["target_col"])
    cell_name = "SRAM_6T_CELL" if summary["cell_type"] == "SRAM_6T_CELL" else "SRAM_10T_CELL"
    core_name = "SRAM_6T_CORE" if cell_name == "SRAM_6T_CELL" else "SRAM_10T_CORE"
    cell_prefix = f"X{core_name}_{rows}X{cols}:X{cell_name}_{target_row}_{target_col}"
    columns = [
        f"V(WL{target_row})",
        f"V(BL{target_col})",
        f"V(BLB{target_col})",
        f"V({cell_prefix}:Q)",
        f"V({cell_prefix}:QB)",
    ]
    return ["V(S_EN)", *columns] if operation == "read" else ["V(WE)", *columns]


def plot_waveform(deck_path: Path, summary: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from utils import process_simulation_data  # type: ignore

    prn_path = Path(f"{deck_path}.prn")
    if not prn_path.is_file():
        raise FileNotFoundError(f"Xyce waveform output was not generated: {prn_path}")
    output_path = Path(summary["run_dir"]) / "waveform.png"
    process_simulation_data(
        prn_path=prn_path,
        num_mc=int(summary["mc_runs"]),
        output=output_path,
        selected_columns=waveform_columns(summary),
    )
    return output_path


def generate_deck(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    target_row = args.rows - 1 if args.target_row is None else args.target_row
    target_col = args.cols - 1 if args.target_col is None else args.target_col

    if not 0 <= target_row < args.rows:
        raise ValueError(f"target_row must be in [0, {args.rows - 1}]")
    if not 0 <= target_col < args.cols:
        raise ValueError(f"target_col must be in [0, {args.cols - 1}]")

    config = load_config(args.rows, args.cols, args.corner)
    if getattr(args, 'vdd', None) is not None:
        config.global_config.vdd = args.vdd
    if getattr(args, 'temperature', None) is not None:
        config.global_config.temperature = args.temperature
    if getattr(args, 'period', None) is not None:
        config.global_config.timing = {'mode': 'fixed', 't_period': args.period}
    if getattr(args, 'timing_lookup', None) is not None:
        config.global_config.timing = {'mode': 'lookup', 'lookup': str(args.timing_lookup)}
    wire_file = getattr(args, 'interconnect_config', None)
    if wire_file is not None:
        config.global_config.interconnect = load_interconnect(wire_file)
    interconnect = resolve_interconnect(config.global_config.interconnect)
    # None keeps the global.yaml `equivalent` block; the option overrides it.
    equivalent = resolve_equivalent(config.global_config.equivalent
                                    if args.real_cell_mode is None else args.real_cell_mode)
    args.real_cell_mode = equivalent.mode
    cell_type = config.global_config.sram_cell_type
    custom_vars = (
        get_custom_vars(config, cell_type) if args.variation_mode == "custom" else None
    )
    mc_runs = resolve_mc_runs(args.mc_runs, args.variation_mode, custom_vars)
    context = physical_context(True, args.pi_res_ohm, args.pi_cap_pf * 1e-12,
                               args.real_cell_mode, interconnect)
    sizes = resolve_driver_sizes(config, physical_context=context)
    timing = resolve_timing(config, sizes, context)
    run_name = make_run_name(
        args,
        cell_type=cell_type,
        target_row=target_row,
        target_col=target_col,
        mc_runs=mc_runs,
        interconnect=interconnect,
        timing=timing.to_dict(),
        vdd=float(config.global_config.vdd),
        temperature=float(config.global_config.temperature),
    )
    run_root = args.output_dir.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / run_name
    # Repeating a diagnostic must not erase the failed evidence it investigates.
    attempt = 1
    while True:
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            run_dir = run_root / f'{run_name}_attempt{attempt}'
            attempt += 1
    is_shared = args.variation_mode == "shared"
    is_custom = args.variation_mode == "custom"
    sample_count = 1 if args.variation_mode == "nominal" else mc_runs

    testbench = Sram6TCoreMcTestbench(
        config,
        sram_cell_type=cell_type,
        w_rc=True,
        pi_res=args.pi_res_ohm @ u_Ohm,
        pi_cap=args.pi_cap_pf @ u_pF,
        vth_std=args.vth_std,
        mc=is_shared,
        custom_mc=is_custom,
        variation_mode=args.variation_mode,
        mc_seed=args.seed,
        sweep_cell=False,
        sweep_precharge=False,
        sweep_senseamp=False,
        sweep_wordlinedriver=False,
        sweep_columnmux=False,
        sweep_writedriver=False,
        sweep_decoder=False,
        corner=args.corner,
        choose_columnmux=bool(config.global_config.choose_columnmux),
        real_cell_mode=args.real_cell_mode,
        q_init_val=args.q_init_val,
        sim_path=str(run_dir),
        enable_waveform=args.waveform,
        interconnect=interconnect,
        driver_sizes=sizes,
        timing_config=timing,
    )
    circuit = testbench.create_testbench(args.operation, target_row, target_col)
    temperature = config.global_config.temperature
    simulator = circuit.simulator(simulator='xyce-serial', temperature=temperature, nominal_temperature=27)
    testbench.add_analysis(simulator.circuit, args.operation, sample_count)
    testbench.add_meas_and_print(simulator, testbench.data_init(), args.operation)
    if is_custom:
        testbench.gen_process_params(
            simulator.circuit,
            args.operation,
            num_mc=mc_runs,
            vars=custom_vars,
        )

    deck_text = str(simulator)
    if not args.waveform:
        deck_text = (
            "\n".join(
                line
                for line in deck_text.splitlines()
                if not line.lstrip().lower().startswith(".print")
            )
            + "\n"
        )
    deck_path = run_dir / "deck.sp"
    variation_summary = testbench.variation_summary

    deck_path.write_text(deck_text, encoding="utf-8")
    summary = {
        "compiler_version": VERSION,
        "interconnect": interconnect.to_dict(),
        "deck": str(deck_path),
        "run_dir": str(run_dir),
        "rows": args.rows,
        "cols": args.cols,
        "target_row": target_row,
        "target_col": target_col,
        "operation": args.operation,
        "real_cell_mode": args.real_cell_mode,
        "equivalent": equivalent.to_dict(),
        "variation_mode": args.variation_mode,
        "mc_runs": mc_runs,
        "corner": args.corner,
        "vdd": float(testbench.vdd),
        "temperature": temperature,
        "model_sha256": hashlib.sha256(Path(getattr(
            config.global_config, f'pdk_path_{args.corner}')).read_bytes()).hexdigest(),
        "cell_type": cell_type,
        "full_device_coverage": args.real_cell_mode == 0 and args.variation_mode == "per-device",
        "seed": args.seed,
        "driver_sizes": testbench.driver_sizes.to_dict(),
        "timing": testbench.timing_config.to_dict(),
        **variation_summary,
    }
    return deck_path, summary


def find_xyce(command: str) -> str:
    candidate = Path(command).expanduser()
    if candidate.parent != Path("."):
        if candidate.is_file():
            return str(candidate.resolve())
        raise FileNotFoundError(f"Xyce executable does not exist: {candidate}")
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"Xyce executable was not found on PATH: {command}")
    return resolved


def run_xyce(deck_path: Path, command: str, seed: int) -> None:
    from utils.xyce import execute_xyce

    if seed <= 0:
        raise ValueError("seed must be positive")
    result = execute_xyce(
        deck_path,
        [
            find_xyce(command),
            "-randseed",
            str(seed),
            "-o",
            str(deck_path),
            str(deck_path),
        ],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Xyce failed with exit code {result.returncode}: {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--target-row", type=int)
    parser.add_argument("--target-col", type=int)
    parser.add_argument("--operation", choices=OPERATIONS, default="read")
    parser.add_argument("--real-cell-mode", type=int, choices=range(5), default=None,
                        help="equivalent array model; default: the global.yaml `equivalent` block "
                             "(0 = full transistor array, 1-4 replace unused cells, an approximation)")
    parser.add_argument(
        "--variation-mode", choices=VARIATION_MODES, default="per-device"
    )
    parser.add_argument("--mc-runs", type=int)
    parser.add_argument("--vth-std", type=float, default=0.05)
    parser.add_argument(
        "--corner", choices=("TT", "FF", "SS", "FS", "SF"), default="TT"
    )
    parser.add_argument("--pi-res-ohm", type=float, default=100.0)
    parser.add_argument("--pi-cap-pf", type=float, default=0.001)
    parser.add_argument("--interconnect-config", type=Path,
                        help="YAML interconnect mapping; relative paths resolve from the project root")
    parser.add_argument("--q-init-val", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/per_device_mc",
        help="root directory for deterministic per-configuration run directories",
    )
    parser.add_argument("--audit", action="store_true")
    parser.add_argument(
        "--waveform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain .PRINT lines and waveform output",
    )
    parser.add_argument("--run-xyce", action="store_true")
    parser.add_argument("--xyce", default="Xyce")
    parser.add_argument("--seed", type=int, default=20260711)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--period", type=float, help="fixed diagnostic clock period in seconds")
    timing.add_argument("--timing-lookup", type=Path, help="alternative timing class JSON table")
    parser.add_argument("--vdd", type=float, help="supply voltage in volts (default: global.yaml)")
    parser.add_argument("--temperature", type=float,
                        help="temperature in Celsius (default: global.yaml)")
    args = parser.parse_args()
    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive")
    if args.vth_std < 0:
        parser.error("--vth-std must be non-negative")
    if args.vdd is not None and not (math.isfinite(args.vdd) and args.vdd > 0):
        parser.error("--vdd must be finite and positive")
    if args.temperature is not None and not math.isfinite(args.temperature):
        parser.error("--temperature must be finite")
    if args.operation in SNM_OPERATIONS and not args.waveform:
        parser.error("SNM operations require waveform output; omit --no-waveform")
    return args


def main() -> int:
    args = parse_args()
    deck_path, summary = generate_deck(args)
    rejection: Exception | None = None
    if args.run_xyce:
        # Solver/MPI stacks differ between environments; keep the installation
        # even when the run fails before producing measurements.
        try:
            summary["xyce"] = find_xyce(args.xyce)
        except FileNotFoundError:
            summary["xyce"] = None
    try:
        if args.run_xyce:
            run_xyce(deck_path, args.xyce, args.seed)
            summary["xyce_exit"] = 0
            if summary['operation'] not in SNM_OPERATIONS:
                from utils.measurements import parse_mc_measurements
                measurements = parse_mc_measurements(str(deck_path), num_runs=summary['mc_runs'])
                measurements.to_csv(Path(str(deck_path) + '.data.csv'))
                try:
                    Sram6TCoreMcTestbench.validate_distributed_precharge(
                        measurements, summary['operation'], summary['vdd'])
                except RuntimeError as exc:
                    # An unsafe release is evidence, not an aborted run: keep the
                    # deck, seed, model and driver-size provenance of the sample
                    # that has to be investigated before re-raising below.
                    summary['precharge_release_checked'] = False
                    summary['precharge_release_error'] = str(exc)
                    rejection = exc
                else:
                    summary['precharge_release_checked'] = True
            if summary['operation'] not in SNM_OPERATIONS:
                valid = Sram6TCoreMcTestbench.access_validity(
                    measurements, summary['operation'], summary['vdd'])
                summary['access_checked'] = bool(len(valid) and valid.all())
                if not summary['access_checked'] and rejection is None:
                    rejection = RuntimeError('SRAM access/precharge, frozen data deadline, retention or restore check failed')
                delay_name = {'read': 'TREAD_TOTAL', 'write': 'TWRITE_TOTAL',
                              'read&write': 'TVOUT_PERIOD'}[summary['operation']]
                metrics = measurements.reindex(columns=[delay_name, 'PAVG', 'PSTC', 'PDYN'])
                summary['metrics_checked'] = bool(len(metrics) and metrics.notna().to_numpy().all())
                if not summary['metrics_checked'] and rejection is None:
                    rejection = RuntimeError('SRAM metrics are missing or FAILED in the requested ensemble')
            if args.waveform:
                try:
                    summary["waveform_png"] = str(plot_waveform(deck_path, summary))
                except Exception as exc:
                    # The .prn stays on disk; a plot that cannot be drawn is
                    # neither a solver nor a measurement failure and must not
                    # turn passing measures into a rejected run.
                    summary["waveform_png"] = None
                    summary["waveform_error"] = f'{type(exc).__name__}: {exc}'
                    print(f'warning: waveform plot failed: {summary["waveform_error"]}',
                          file=sys.stderr)
    except Exception as exc:
        summary['simulation_error'] = f'{type(exc).__name__}: {exc}'
        rejection = rejection or exc
    except BaseException as exc:
        # An interrupted or killed solver run is an incomplete outcome, not a
        # missing one: keep its deck, seed and Xyce provenance before leaving.
        summary['simulation_error'] = f'{type(exc).__name__}: {exc}'
        (Path(summary["run_dir"]) / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    if args.audit or rejection is not None or 'waveform_error' in summary:
        (Path(summary["run_dir"]) / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if rejection is not None:
        raise rejection
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
