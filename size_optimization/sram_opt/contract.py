"""The data-free 26-feature design contract used by SramOpt."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


# Three architecture fields plus 23 jointly varied bitcell/peripheral fields.
# Column-mux fields are fixed by the V2.2.6 configuration and are not features.
FEATURE_COLUMNS = (
    "rows",
    "cols",
    "num_arrays",
    "pg_width",
    "pd_width",
    "pu_width",
    "length",
    "nmos_model_name",
    "pmos_model_name",
    "sa_p_width",
    "sa_n_width",
    "sa_length",
    "sa_p_model",
    "sa_n_model",
    "prc_p_width",
    "prc_length",
    "prc_p_model",
    "wld_nand_p_width",
    "wld_inv_p_width",
    "wld_nand_n_width",
    "wld_inv_n_width",
    "wld_length",
    "wld_nand_p_model",
    "wld_inv_p_model",
    "wld_nand_n_model",
    "wld_inv_n_model",
)

CATEGORICAL_COLUMNS = (
    "nmos_model_name",
    "pmos_model_name",
    "sa_p_model",
    "sa_n_model",
    "prc_p_model",
    "wld_nand_p_model",
    "wld_inv_p_model",
    "wld_nand_n_model",
    "wld_inv_n_model",
)

NUMERIC_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
)


def normalize_design(design: Mapping[str, Any]) -> Dict[str, Any]:
    """Return one canonical SramOpt feature row."""
    row = dict(design)
    pd_model = row.get("pd_model_name", row.get("nmos_model_name"))
    pg_model = row.get("pg_model_name", row.get("nmos_model_name"))
    if pd_model is None or pg_model is None:
        raise ValueError("design is missing the shared NMOS model")
    if str(pd_model) != str(pg_model):
        raise ValueError("SramOpt requires PD and PG to share one NMOS model")
    row["nmos_model_name"] = str(pd_model)
    row["pd_model_name"] = str(pd_model)
    row["pg_model_name"] = str(pd_model)

    missing = [column for column in FEATURE_COLUMNS if column not in row]
    if missing:
        raise ValueError("design is missing SramOpt features: " + ", ".join(missing))
    return {column: row[column] for column in FEATURE_COLUMNS}


def feature_rows(designs: Iterable[Mapping[str, Any]]):
    """Build a pandas frame lazily so basic checks do not require pandas."""
    import pandas as pd

    return pd.DataFrame(
        [normalize_design(design) for design in designs],
        columns=list(FEATURE_COLUMNS),
    )
