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
from typing import Any, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySpice.Unit import u_Ohm, u_pF  # type: ignore  # noqa: E402

from sram_compiler.config_yaml.config import SRAM_CONFIG
from sram_compiler.config_yaml.loader import configure, load_config
from sram_compiler.variation import (
    DEFAULT_VARIATION_MODE, VARIATION_MODES, get_custom_vars, resolve_mc_runs,
)
from sram_compiler.equivalent_modeling import resolve_equivalent
from sram_compiler.interconnect import resolve_interconnect
from sram_compiler.sizing import resolve_driver_sizes, resolve_timing
from sram_compiler.sizing.table import physical_context
from sram_compiler.version import VERSION
from sram_compiler.testbenches.sram_6t_core_MC_testbench import (  # type: ignore  # noqa: E402
    Sram6TCoreMcTestbench,
    full_device_coverage,
)


OPERATIONS = (
    "read",
    "write",
    "read&write",
    "hold_snm",
    "read_snm",
    "write_snm",
)
SNM_OPERATIONS = frozenset({"hold_snm", "read_snm", "write_snm"})


def bitcell_area(config: SRAM_CONFIG) -> float:
    cell_type = config.global_config.sram_cell_type
    cell = config.sram_6t_cell if cell_type == "SRAM_6T_CELL" else config.sram_10t_cell
    extra = {} if cell_type == "SRAM_6T_CELL" else {
        "w_fd": cell.nmos_width.value[2], "cell_type": cell_type}
    from utils import estimate_bitcell_area

    return estimate_bitcell_area(
        w_access=cell.nmos_width.value[1],
        w_pd=cell.nmos_width.value[0],
        w_pu=cell.pmos_width.value,
        l_transistor=cell.length.value,
        **extra,
    )


def cell_parameters(config: SRAM_CONFIG) -> dict[str, Any]:
    cell = config.sram_6t_cell if config.global_config.sram_cell_type == 'SRAM_6T_CELL' else config.sram_10t_cell
    return {key: getattr(cell, key).value
            for key in ('nmos_width', 'pmos_width', 'length', 'nmos_model', 'pmos_model')}


def build_testbench(config: SRAM_CONFIG, sim_path: str, *,
                    variation_mode: str = DEFAULT_VARIATION_MODE,
                    mc_seed: Optional[int] = 20260711,
                    real_cell_mode: Optional[int] = None,
                    w_rc: bool = True, pi_res_ohm: float = 100.0,
                    pi_cap_pf: float = 0.001, vth_std: float = 0.05,
                    q_init_val: int = 0, enable_waveform: bool = True,
                    interconnect=None, driver_sizes=None,
                    timing_config=None) -> Sram6TCoreMcTestbench:
    """Build the Monte Carlo testbench with the per-device default made explicit."""
    return Sram6TCoreMcTestbench(
        config,
        sram_cell_type=config.global_config.sram_cell_type,
        w_rc=w_rc,
        pi_res=pi_res_ohm @ u_Ohm, pi_cap=pi_cap_pf @ u_pF,
        vth_std=vth_std,
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
        q_init_val=q_init_val,
        sim_path=sim_path,
        enable_waveform=enable_waveform,
        interconnect=interconnect, driver_sizes=driver_sizes,
        timing_config=timing_config,
    )


def write_deck(testbench: Sram6TCoreMcTestbench, operation: str,
               target_row: int, target_col: int, mc_runs: int,
               custom_vars: Any, *, waveform: bool = True) -> Path:
    """Export the same sampled circuit and measures used by a Xyce run."""
    circuit = testbench.create_testbench(operation, target_row, target_col)
    simulator = circuit.simulator(simulator='xyce-serial',
                                  temperature=testbench.temperature,
                                  nominal_temperature=27)
    testbench.add_analysis(simulator.circuit, operation, mc_runs)
    testbench.add_meas_and_print(simulator, testbench.data_init(), operation)
    if custom_vars is not None:
        testbench.gen_process_params(simulator.circuit, operation,
                                     num_mc=mc_runs, vars=custom_vars)
    deck_path = Path(testbench.sim_path) / 'deck.sp'
    deck_text = str(simulator)
    if not waveform:
        deck_text = '\n'.join(line for line in deck_text.splitlines()
                              if not line.lstrip().lower().startswith('.print')) + '\n'
    deck_path.write_text(deck_text, encoding='utf-8')
    model_path = Path(getattr(testbench.sram_config.global_config,
                              f'pdk_path_{testbench.corner}'))
    summary = {
        'compiler_version': VERSION,
        'run_dir': str(Path(testbench.sim_path)),
        'deck': str(deck_path), 'operation': operation,
        'rows': testbench.num_rows, 'cols': testbench.num_cols,
        'target_row': target_row, 'target_col': target_col,
        'cell_type': testbench.sram_cell_type, 'corner': testbench.corner,
        'temperature': testbench.temperature, 'vdd': float(testbench.vdd),
        'variation_mode': testbench.variation_mode, 'mc_runs': mc_runs,
        'full_device_coverage': full_device_coverage(
            testbench.variation_mode, testbench.real_cell_mode, operation),
        'seed': testbench.mc_seed, 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'equivalent': testbench.equivalent.to_dict(),
        'real_cell_mode': testbench.real_cell_mode,
        'time_control': testbench.sram_config.time_control.to_dict(),
        'local_rc': {'enabled': testbench.w_rc,
                     'resistance_ohm': float(testbench.pi_res),
                     'capacitance_f': float(testbench.pi_cap)},
        'bitcell_area_m2': bitcell_area(testbench.sram_config),
        'cell_parameters': cell_parameters(testbench.sram_config),
        'column_mux': testbench.choose_columnmux,
        'interconnect': testbench.interconnect.to_dict(),
        'driver_sizes': testbench.driver_sizes.to_dict(),
        'timing': testbench.timing_config.to_dict(),
        **testbench.variation_summary,
    }
    (Path(testbench.sim_path) / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return deck_path


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
    control_config=None,
    cell_config=None,
    choose_columnmux=None,
    model_sha256=None,
) -> str:
    settings = {
        "compiler_version": VERSION,
        "time_control": control_config,
        "interconnect": resolve_interconnect(interconnect).to_dict(),
        "cell_type": cell_type,
        "cell_config": cell_config,
        "choose_columnmux": choose_columnmux,
        "model_sha256": model_sha256,
        "w_rc": getattr(args, 'w_rc', True),
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
    if getattr(args, 'cell_type', None) is not None:
        config.global_config.sram_cell_type = args.cell_type
    mux = getattr(args, 'column_mux', None)
    config = configure(args.rows, args.cols,
                       config.global_config.choose_columnmux if mux is None else mux,
                       args.corner, getattr(args, 'cell_6t', None),
                       getattr(args, 'interconnect_config', None), config=config)
    if getattr(args, 'vdd', None) is not None:
        config.global_config.vdd = args.vdd
    if getattr(args, 'temperature', None) is not None:
        config.global_config.temperature = args.temperature
    if getattr(args, 'period', None) is not None:
        config.global_config.timing = {'mode': 'fixed', 't_period': args.period}
    if getattr(args, 'timing_lookup', None) is not None:
        config.global_config.timing = {'mode': 'lookup', 'lookup': str(args.timing_lookup)}
    if getattr(args, 'sizing_lookup', None) is not None:
        config.global_config.sizing = dict(config.global_config.sizing,
                                          mode='lookup', lookup=str(args.sizing_lookup))
    if getattr(args, 'compact_control', False):
        config.global_config.sizing = dict(config.global_config.sizing, compact_control=True)
    interconnect = resolve_interconnect(config.global_config.interconnect)
    # None keeps the global.yaml `equivalent` block; the option overrides it.
    equivalent = resolve_equivalent(config.global_config.equivalent
                                    if args.real_cell_mode is None else args.real_cell_mode)
    args.real_cell_mode = equivalent.mode
    cell_type = config.global_config.sram_cell_type
    custom_vars = (
        get_custom_vars(config, cell_type) if args.variation_mode == "custom" else None
    )
    requested = args.mc_runs
    if requested is None and args.variation_mode in ('shared', 'per-device'):
        requested = config.global_config.monte_carlo_runs
    mc_runs = resolve_mc_runs(requested, args.variation_mode, custom_vars)
    context = physical_context(getattr(args, 'w_rc', True), args.pi_res_ohm, args.pi_cap_pf * 1e-12,
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
        control_config=config.time_control.to_dict(),
        cell_config=cell_parameters(config),
        choose_columnmux=config.global_config.choose_columnmux,
        model_sha256=hashlib.sha256(Path(getattr(
            config.global_config, f'pdk_path_{args.corner}')).read_bytes()).hexdigest(),
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
    testbench = build_testbench(
        config, str(run_dir), variation_mode=args.variation_mode,
        mc_seed=args.seed, real_cell_mode=args.real_cell_mode,
        w_rc=getattr(args, 'w_rc', True), pi_res_ohm=args.pi_res_ohm,
        pi_cap_pf=args.pi_cap_pf, vth_std=args.vth_std,
        q_init_val=args.q_init_val, enable_waveform=args.waveform,
        interconnect=interconnect, driver_sizes=sizes, timing_config=timing,
    )
    deck_path = write_deck(testbench, args.operation, target_row, target_col,
                           mc_runs, custom_vars, waveform=args.waveform)
    summary = json.loads(deck_path.with_name('summary.json').read_text())
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


def run_xyce(deck_path: Path, command: str, seed: Optional[int]) -> None:
    from utils.xyce import execute_xyce

    if seed is not None and seed <= 0:
        raise ValueError("seed must be positive or None")
    command_args = [find_xyce(command)]
    if seed is not None:
        command_args.extend(['-randseed', str(seed)])
    command_args.extend(['-o', str(deck_path), str(deck_path)])
    result = execute_xyce(deck_path, command_args, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Xyce failed with exit code {result.returncode}: {detail}")


def parse_seed(value: str) -> Optional[int]:
    """An explicit random seed request omits Xyce's fixed seed."""
    return None if value.lower() in ('random', 'none') else int(value)


def parse_args(argv: Optional[Sequence[str]] = None, *, defaults=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, prog="main_sram.py simulate")
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
        default=PROJECT_ROOT / "outputs/main_sram",
        help="root directory for deterministic per-configuration run directories",
    )
    parser.add_argument("--audit", action="store_true",
                        help="compatibility flag; per-device model audits are always saved")
    parser.add_argument(
        "--waveform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain .PRINT lines and waveform output",
    )
    parser.add_argument("--run-xyce", action=argparse.BooleanOptionalAction, default=False,
                        help="execute Xyce; --no-run-xyce generates decks only")
    parser.add_argument("--xyce", default="Xyce")
    parser.add_argument("--seed", type=parse_seed, default=20260711,
                        help="positive integer for repeatable draws, or random")
    parser.add_argument('--sizing-lookup', type=Path, help='explicit diagnostic driver class JSON')
    parser.add_argument('--compact-control', action='store_true',
                        help='use the simplified controller and remove obsolete load allowances')
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--period", type=float, help="fixed diagnostic clock period in seconds")
    timing.add_argument("--timing-lookup", type=Path, help="alternative timing class JSON table")
    parser.add_argument("--vdd", type=float, help="supply voltage in volts (default: global.yaml)")
    parser.add_argument("--temperature", type=float,
                        help="temperature in Celsius (default: global.yaml)")
    parser.add_argument('--cell-type', choices=('SRAM_6T_CELL', 'SRAM_10T_CELL'),
                        help='cell architecture; default: global.yaml')
    parser.add_argument('--cell-6t', nargs=7, metavar=('PD_W', 'PG_W', 'PU_W', 'L',
                                                   'PD_MODEL', 'PG_MODEL', 'PU_MODEL'),
                        help='6T widths/length in SI metres and model names')
    parser.add_argument('--column-mux', action=argparse.BooleanOptionalAction, default=None,
                        help='override the global.yaml column-mux setting')
    parser.add_argument('--w-rc', action=argparse.BooleanOptionalAction, default=True,
                        help='local device stubs, separate from distributed interconnect')
    if defaults:
        parser.set_defaults(**defaults)
        # argparse converts string defaults with type=int. Script settings must
        # retain their types so invalid MC_RUNS values fail the shared resolver.
        parser.set_defaults(mc_runs=None)
    args = parser.parse_args(argv)
    if defaults and args.mc_runs is None:
        args.mc_runs = defaults.get('mc_runs')
    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive")
    if not math.isfinite(args.vth_std) or args.vth_std < 0:
        parser.error("--vth-std must be finite and non-negative")
    if args.seed is not None and args.seed <= 0:
        parser.error("--seed must be positive")
    if args.vdd is not None and not (math.isfinite(args.vdd) and args.vdd > 0):
        parser.error("--vdd must be finite and positive")
    if args.temperature is not None and not math.isfinite(args.temperature):
        parser.error("--temperature must be finite")
    if args.operation in SNM_OPERATIONS and not args.waveform:
        parser.error("SNM operations require waveform output; omit --no-waveform")
    for option, value in (('pi-res-ohm', args.pi_res_ohm), ('pi-cap-pf', args.pi_cap_pf)):
        if not math.isfinite(value) or value < 0:
            parser.error(f'--{option} must be finite and non-negative')
    if args.period is not None and (not math.isfinite(args.period) or args.period <= 0):
        parser.error('--period must be finite and positive')
    return args


def main(argv: Optional[Sequence[str]] = None, *, defaults=None) -> int:
    args = parse_args() if argv is None and defaults is None else parse_args(argv, defaults=defaults)
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
                from utils.measurements import parse_mc_measurements, generate_mc_statistics
                measurements = parse_mc_measurements(str(deck_path), num_runs=summary['mc_runs'])
                measurements.to_csv(Path(str(deck_path) + '.data.csv'))
                generate_mc_statistics(measurements).to_csv(Path(str(deck_path) + '.stats.csv'))
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
                summary['metrics_checked'] = bool(
                    len(metrics) == summary['mc_runs'] and np.isfinite(metrics.to_numpy()).all())
                if not summary['metrics_checked'] and rejection is None:
                    rejection = RuntimeError('SRAM metrics are missing or FAILED in the requested ensemble')
                if summary['metrics_checked']:
                    summary['mean_metrics'] = {name: float(values.mean())
                                               for name, values in metrics.items()}
            else:
                from sram_compiler.testbenches.snm import process_xyce_montecarlo_prn
                snm_data, _ = process_xyce_montecarlo_prn(
                    prn_path=str(deck_path) + '.prn', metric_name=summary['operation'],
                    operation=summary['operation'], vdd=summary['vdd'])
                values = snm_data[summary['operation']].to_numpy(dtype=float)
                summary['metrics_checked'] = bool(
                    len(values) == summary['mc_runs'] and np.isfinite(values).all())
                if summary['metrics_checked']:
                    summary['mean_metrics'] = {summary['operation']: float(values.mean())}
                else:
                    rejection = RuntimeError('SNM samples are missing or undefined in the requested ensemble')
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
