"""Command-line validation runner for unified OpenYield estimators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from yield_estimation import (
    EXPERIMENTAL_ALGORITHMS,
    STABLE_ALGORITHMS,
    TargetCellTestbenchAdapter,
    YieldEstimator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "578d53d502d6418970447c1f34accfe3d0b6b957"
DESIGN_POINT = {
    "pd_width_m": 105.920881e-9,
    "pg_width_m": 90.531661e-9,
    "pu_width_m": 93.899314e-9,
    "length_m": 34.915390e-9,
    "pd_model": "NMOS_VTL",
    "pg_model": "NMOS_VTL",
    "pu_model": "PMOS_VTL",
}
MOS_ORDER_6T = ("PGL", "PGR", "PDL", "PUL", "PDR", "PUR")
PARAM_ORDER = ("vth0", "u0", "voff")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sram_config(
    rows: int, cols: int, corner: str, pdk_path: Path | None = None
):
    from sram_compiler.config_yaml.config import SRAM_CONFIG

    config_dir = PROJECT_ROOT / "sram_compiler" / "config_yaml"
    config = SRAM_CONFIG()
    config.load_all_configs(
        global_file=str(config_dir / "global.yaml"),
        circuit_configs={
            "SRAM_6T_CELL": str(config_dir / "sram_6t_cell.yaml"),
            "WORDLINEDRIVER": str(config_dir / "wordline_driver.yaml"),
            "PRECHARGE": str(config_dir / "precharge.yaml"),
            "COLUMNMUX": str(config_dir / "mux.yaml"),
            "SENSEAMP": str(config_dir / "sa.yaml"),
            "WRITEDRIVER": str(config_dir / "write_driver.yaml"),
            "DECODER": str(config_dir / "decoder.yaml"),
        },
    )
    config.global_config.num_rows = rows
    config.global_config.num_cols = cols
    config.global_config.corner = corner
    config.global_config.choose_columnmux = False
    resolved_pdk = pdk_path or PROJECT_ROOT / "tran_models" / f"models_{corner}.spice"
    if not resolved_pdk.is_file():
        raise FileNotFoundError(f"PDK model file does not exist: {resolved_pdk}")
    setattr(config.global_config, f"pdk_path_{corner}", str(resolved_pdk.resolve()))
    cell = config.sram_6t_cell
    cell.nmos_width.value = [DESIGN_POINT["pd_width_m"], DESIGN_POINT["pg_width_m"]]
    cell.pmos_width.value = DESIGN_POINT["pu_width_m"]
    cell.length.value = DESIGN_POINT["length_m"]
    cell.nmos_model.value = [DESIGN_POINT["pd_model"], DESIGN_POINT["pg_model"]]
    cell.pmos_model.value = DESIGN_POINT["pu_model"]
    return config


def _one_cell_nominal(config, corner: str) -> np.ndarray:
    from utils import parse_spice_models

    models = parse_spice_models(getattr(config.global_config, f"pdk_path_{corner}"))
    cell = config.sram_6t_cell
    model_for_mos = {
        "PGL": cell.nmos_model.value[1],
        "PGR": cell.nmos_model.value[1],
        "PDL": cell.nmos_model.value[0],
        "PDR": cell.nmos_model.value[0],
        "PUL": cell.pmos_model.value,
        "PUR": cell.pmos_model.value,
    }
    return np.asarray(
        [
            float(models[model_for_mos[mos]]["parameters"][parameter])
            for mos in MOS_ORDER_6T
            for parameter in PARAM_ORDER
        ],
        dtype=float,
    )


def make_xyce_testbench(
    rows: int,
    cols: int,
    corner: str,
    run_root: Path,
    *,
    pdk_path: Path | None = None,
    std_ratio: float = 0.05,
    clock_period_ns: float = 5.0,
):
    from PySpice.Unit import u_Ohm, u_pF, u_ns
    from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench

    config = _load_sram_config(rows, cols, corner, pdk_path)
    testbench = Sram6TCoreMcTestbench(
        config,
        sram_cell_type="SRAM_6T_CELL",
        w_rc=False,
        pi_res=100 @ u_Ohm,
        pi_cap=0.001 @ u_pF,
        vth_std=std_ratio,
        custom_mc=True,
        sweep_cell=False,
        sweep_precharge=False,
        sweep_senseamp=False,
        sweep_wordlinedriver=False,
        sweep_columnmux=False,
        sweep_writedriver=False,
        sweep_decoder=False,
        corner=corner,
        choose_columnmux=False,
        q_init_val=0,
        sim_path=str(run_root / "bootstrap_sim"),
        enable_waveform=False,
    )
    period = clock_period_ns @ u_ns
    testbench.set_timing_parameters(
        0.01 * period,
        0.01 * period,
        6 @ u_ns,
        period,
        1 @ u_ns,
    )
    return testbench, _one_cell_nominal(config, corner)


def make_analytic_model(reference_probability: float):
    # P(Z > threshold) uses the same standardized 18-D input contract. The
    # threshold below gives approximately the SRAM reference probability.
    normal_threshold = 1.1462

    def model(samples: np.ndarray) -> np.ndarray:
        return samples[:, 0]

    return model, normal_threshold


def run_validation(args: argparse.Namespace):
    run_root = args.output / args.algorithm.lower()
    if args.backend == "xyce":
        resolved_pdk = (
            args.pdk_path or PROJECT_ROOT / "tran_models" / f"models_{args.corner}.spice"
        ).resolve()
        model, nominal_cell = make_xyce_testbench(
            args.rows,
            args.cols,
            args.corner,
            args.output / "_bootstrap" / args.algorithm.lower(),
            pdk_path=resolved_pdk,
            std_ratio=args.std_ratio,
            clock_period_ns=args.clock_period_ns,
        )
        if args.vary == "target":
            model = TargetCellTestbenchAdapter(
                model,
                nominal_cell,
                num_rows=args.rows,
                num_cols=args.cols,
                target_row=args.target_row,
                target_col=args.target_col,
            )
            physical_nominal = nominal_cell
        else:
            physical_nominal = np.tile(nominal_cell, args.rows * args.cols)
        physical_sigma = np.abs(physical_nominal) * args.std_ratio
        if np.any(physical_sigma == 0):
            raise ValueError("PDK nominal contains zero-valued varied parameters")
        mean = np.zeros(physical_nominal.size, dtype=float)
        covariance = np.eye(mean.size, dtype=float)
        threshold = args.threshold
        pdk_sha256 = _sha256(resolved_pdk)
    else:
        model, threshold = make_analytic_model(args.reference_probability)
        dimension = 18 if args.vary == "target" else args.rows * args.cols * 18
        mean = np.zeros(dimension, dtype=float)
        covariance = np.eye(mean.size, dtype=float)
        physical_nominal = None
        physical_sigma = None
        resolved_pdk = None
        pdk_sha256 = None

    conditions = None
    if args.algorithm == "BIBD":
        conditions = [
            {"name": "nominal", "threshold": threshold, "temperature": 27},
            {"name": "hot", "threshold": threshold, "temperature": 85},
        ]
    estimator = YieldEstimator(
        model=model,
        algorithm_choice=args.algorithm,
        basic_params={
            "mean": mean,
            "covariance": covariance,
            "threshold": threshold,
            "seed": args.seed,
            "failure_direction": "greater",
        },
        algo_params={
            "pilot_fraction": args.pilot_fraction,
            "defensive_ratio": args.defensive_ratio,
            "proposal_scale": args.proposal_scale,
            "max_components": args.max_components,
            "batch_size": args.batch_size,
            **({"mode": args.acs_mode} if args.algorithm == "ACS" else {}),
            **(
                {"surrogate_backend": args.fusis_surrogate_backend}
                if args.algorithm == "FUSIS"
                else {}
            ),
            **(
                {"flow_backend": args.opt_flow_backend}
                if args.algorithm == "OPT"
                else {}
            ),
            "failure_if_nonpositive": args.backend == "xyce",
            "metadata": {
                "source_commit": SOURCE_COMMIT,
                "backend": args.backend,
                "rows": args.rows,
                "cols": args.cols,
                "vary": args.vary,
                "dimension": int(mean.size),
                "std_ratio": args.std_ratio,
                "clock_period_ns": args.clock_period_ns,
                "pdk_path": str(resolved_pdk) if resolved_pdk else None,
                "pdk_sha256": pdk_sha256,
                "reference_probability": args.reference_probability,
                "reference_count": 99400,
                "design_point": DESIGN_POINT if args.backend == "xyce" else None,
            },
            **({"conditions": conditions} if conditions else {}),
        },
        spice_params={
            "run_root": run_root,
            "operation": "read",
            "target_row": args.target_row,
            "target_col": args.target_col,
            "temperature": args.temperature,
            "metric": "TREAD_TOTAL" if args.backend == "xyce" else 0,
            "input_space": "standard_normal" if args.backend == "xyce" else "physical",
            "nominal": physical_nominal,
            "sigma": physical_sigma,
            "max_retries": args.max_retries,
            "quiet": True,
            "pdk_path": resolved_pdk,
            "pdk_sha256": pdk_sha256,
            "rows": args.rows,
            "cols": args.cols,
            "vary": args.vary,
            "std_ratio": args.std_ratio,
            "clock_period_ns": args.clock_period_ns,
        },
    )
    result = estimator.run(max_num=args.budget)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    choices = STABLE_ALGORITHMS + EXPERIMENTAL_ALGORITHMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=choices, required=True)
    parser.add_argument("--backend", choices=("analytic", "xyce"), default="analytic")
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--target-row", type=int, default=3)
    parser.add_argument("--target-col", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=27.0)
    parser.add_argument("--corner", default="TT")
    parser.add_argument("--pdk-path", type=Path)
    parser.add_argument("--vary", choices=("target", "all"), default="target")
    parser.add_argument("--std-ratio", type=float, default=0.05)
    parser.add_argument("--clock-period-ns", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=6.8869e-11)
    parser.add_argument("--reference-probability", type=float, default=12495 / 99400)
    parser.add_argument("--pilot-fraction", type=float, default=0.4)
    parser.add_argument("--defensive-ratio", type=float, default=0.1)
    parser.add_argument("--proposal-scale", type=float, default=1.0)
    parser.add_argument("--max-components", type=int, default=64)
    parser.add_argument(
        "--acs-mode", choices=("original", "improved"), default="original"
    )
    parser.add_argument(
        "--fusis-surrogate-backend",
        choices=("auto", "deep_kernel_svm", "rbf_svm", "numpy_rbf"),
        default="auto",
    )
    parser.add_argument(
        "--opt-flow-backend",
        choices=("auto", "nflows", "gaussian"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-retries", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_validation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
