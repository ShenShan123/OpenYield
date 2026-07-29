"""Small, shared output contract for every optimization algorithm."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DESIGN_COLUMNS = [
    "topology",
    "rows",
    "cols",
    "num_arrays",
    "pu_width",
    "pd_width",
    "pg_width",
    "cell_length",
    "sa_p_width",
    "sa_n_width",
    "sa_length",
    "wld_nand_p_width",
    "wld_inv_p_width",
    "wld_nand_n_width",
    "wld_inv_n_width",
    "wld_length",
    "prc_p_width",
    "prc_length",
    "pu_model",
    "pd_model",
    "pg_model",
    "sa_p_model",
    "sa_n_model",
    "wld_nand_p_model",
    "wld_inv_p_model",
    "wld_nand_n_model",
    "wld_inv_n_model",
    "prc_p_model",
    "fd_present",
    "fd_width",
    "fd_model",
]

RAW_METRIC_COLUMNS = [
    "raw_hold_snm",
    "raw_read_snm",
    "raw_write_snm",
    "raw_raw_read_delay",
    "raw_raw_write_delay",
    "raw_read_pstc",
    "raw_read_pdyn",
    "raw_write_pstc",
    "raw_write_pdyn",
    "raw_single_array_area",
]

SYSTEM_METRIC_COLUMNS = [
    "hold_snm",
    "read_snm",
    "write_snm",
    "read_delay",
    "write_delay",
    "read_power",
    "write_power",
    "area",
    "min_snm",
    "max_delay",
    "max_power",
    "power_delay_product",
]

PUBLIC_COLUMNS = [
    "algorithm",
    "evaluation",
    "stage",
    "constraint_valid",
    *DESIGN_COLUMNS,
    *RAW_METRIC_COLUMNS,
    *SYSTEM_METRIC_COLUMNS,
]


def create_run_directory(output_base: Path | str) -> tuple[Path, Path, Path]:
    """Create the only two data directories exposed by an optimizer run."""

    output_dir = Path(output_base).expanduser() / time.strftime("%Y%m%d_%H%M%S")
    evaluations_dir = output_dir / "evaluations"
    fronts_dir = output_dir / "pareto_fronts"
    evaluations_dir.mkdir(parents=True, exist_ok=False)
    fronts_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, evaluations_dir, fronts_dir


def _stage_series(df: pd.DataFrame) -> pd.Series:
    if "stage" in df:
        stage = df["stage"]
    elif "phase" in df:
        stage = df["phase"]
    else:
        stage = pd.Series("search", index=df.index)
    stage = stage.fillna("search").astype(str).str.lower()
    stage = stage.replace(
        {
            "initialization": "search",
            "optimization": "search",
            "bayesian_optimization": "search",
            "gradient_step": "refine",
            "hard_audit": "refine",
        }
    )
    stage = stage.mask(stage.str.contains("coarse", case=False, na=False), "coarse")
    stage = stage.mask(stage.str.contains("refine", case=False, na=False), "refine")
    return stage


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def public_evaluations(
    df: pd.DataFrame,
    algorithm: str,
    *,
    expected_rows: int | None = None,
    renumber: bool = True,
) -> pd.DataFrame:
    """Return the same compact CSV schema for every algorithm."""

    result = df.copy().reset_index(drop=True)
    if expected_rows is not None and len(result) != int(expected_rows):
        raise RuntimeError(
            f"{algorithm} produced {len(result)} evaluation rows; "
            f"expected exactly {expected_rows}."
        )

    result["algorithm"] = str(algorithm).upper()
    if renumber or "evaluation" not in result:
        result["evaluation"] = np.arange(1, len(result) + 1, dtype=int)
    else:
        result["evaluation"] = pd.to_numeric(
            result["evaluation"], errors="coerce"
        ).astype("Int64")
    result["stage"] = _stage_series(result)
    if "constraint_valid" not in result:
        if "is_feasible" in result:
            result["constraint_valid"] = result["is_feasible"]
        else:
            result["constraint_valid"] = True
    result["constraint_valid"] = _boolean_series(result["constraint_valid"])

    for column in PUBLIC_COLUMNS:
        if column not in result:
            result[column] = np.nan
    return result.loc[:, PUBLIC_COLUMNS]


def public_pareto_front(
    df: pd.DataFrame,
    algorithm: str,
    *,
    objective_columns: Iterable[str],
) -> pd.DataFrame:
    """Format a front and assert that every configured objective is present."""

    missing = [name for name in objective_columns if name not in df.columns]
    if missing:
        raise ValueError(
            f"{algorithm} Pareto front is missing configured objectives: {missing}"
        )
    return public_evaluations(df, algorithm, renumber=False)
