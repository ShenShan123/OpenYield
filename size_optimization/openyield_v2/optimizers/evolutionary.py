#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evolutionary multi-objective baselines for the shared 6T/10T search space."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


# =============================================================================
# 项目路径与工具库
# =============================================================================
def setup_project_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,
        script_dir.parent,
        script_dir.parents[2],
        Path.cwd(),
        Path.cwd().parent,
    ]
    for path in candidates:
        try:
            path = path.resolve()
        except Exception:
            continue
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return script_dir


SCRIPT_DIR = setup_project_path()

# 必须复用用户的工具库。
if __package__:
    from ..surrogate_utils import (  # noqa: E402
        seed_set,
        feature_engineering,
        DataLoader,
        TabPFNSurrogate,
        MultiTargetSurrogateManager,
        process_raw_to_system_metrics,
    )
else:
    from surrogate_utils import (  # type: ignore[no-redef]  # noqa: E402
        seed_set,
        feature_engineering,
        DataLoader,
        TabPFNSurrogate,
        MultiTargetSurrogateManager,
        process_raw_to_system_metrics,
    )
if __package__:
    from ..optimization_spec import (  # noqa: E402
        feasible_mask as shared_feasible_mask,
        load_problem_spec,
        pareto_front_from_evaluations,
    )
    from ..output_schema import (  # noqa: E402
        create_run_directory,
        public_evaluations,
        public_pareto_front,
    )
else:
    from optimization_spec import (  # type: ignore[no-redef]  # noqa: E402
        feasible_mask as shared_feasible_mask,
        load_problem_spec,
        pareto_front_from_evaluations,
    )
    from output_schema import (  # type: ignore[no-redef]  # noqa: E402
        create_run_directory,
        public_evaluations,
        public_pareto_front,
    )

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 默认配置：直接执行不需要传任何 -- 参数
# =============================================================================
SEED = 33
TOTAL_KB = 32
OUTPUT_COLS = 16

PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_6T_DATASET = str(PACKAGE_ROOT / "datasets" / "train_6t.csv")
DEFAULT_10T_DATASET = str(PACKAGE_ROOT / "datasets" / "train_10t.csv")
DEFAULT_OUTPUT_DIR = str(
    PACKAGE_ROOT / "runs" / "optimization" / "evolutionary"
)
DEFAULT_TEST_SIZE = 0.05
DEFAULT_DEVICE = "auto"
DEFAULT_MAX_TRAIN_PER_TOPOLOGY = 250  # 0 表示不限制
DEFAULT_MAX_TEST_PER_TOPOLOGY = 0   # 0 表示不限制
DEFAULT_BALANCE_TOPOLOGIES = False
DEFAULT_VERBOSE_LIBRARY_TRAINING = False


# =============================================================================
# 输入变量与目标
# =============================================================================
SHARED_CONT_FEATURES = [
    "pu_width", "pd_width", "pg_width", "cell_length",
    "sa_p_width", "sa_n_width", "sa_length",
    "wld_nand_p_width", "wld_inv_p_width",
    "wld_nand_n_width", "wld_inv_n_width", "wld_length",
    "prc_p_width", "prc_length",
]

TOPOLOGY_CONT_FEATURES = ["fd_present", "fd_width"]
DERIVED_CONT_FEATURES = ["aspect_ratio", "log_rows", "log_cols"]
ARCH_FEATURES = ["rows", "cols"]

SHARED_CAT_FEATURES = [
    "pu_model", "pd_model", "pg_model",
    "sa_p_model", "sa_n_model",
    "wld_nand_p_model", "wld_inv_p_model",
    "wld_nand_n_model", "wld_inv_n_model",
    "prc_p_model",
]
TOPOLOGY_CAT_FEATURES = ["topology", "fd_model"]

RAW_TARGETS = [
    "hold_snm", "read_snm", "write_snm",
    "raw_read_delay", "raw_write_delay",
    "read_pstc", "read_pdyn", "write_pstc", "write_pdyn",
    "single_array_area",
]

SYSTEM_METRICS = [
    "hold_snm", "read_snm", "write_snm",
    "read_delay", "write_delay",
    "read_power", "write_power", "area",
]

OBJECTIVE_METRICS = ["min_snm", "max_delay", "max_power", "area"]
OBJECTIVE_PREFERENCES = {
    "min_snm": "max",
    "max_delay": "min",
    "max_power": "min",
    "area": "min",
}

METRIC_UNITS = {
    "hold_snm": "V",
    "read_snm": "V",
    "write_snm": "V",
    "min_snm": "V",
    "raw_read_delay": "s",
    "raw_write_delay": "s",
    "read_delay": "s",
    "write_delay": "s",
    "max_delay": "s",
    "read_pstc": "W",
    "read_pdyn": "W",
    "write_pstc": "W",
    "write_pdyn": "W",
    "read_power": "W",
    "write_power": "W",
    "max_power": "W",
    "single_array_area": "m^2",
    "area": "m^2",
}


# =============================================================================
# 数据统一与清理
# =============================================================================
def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def harmonize_topology_dataframe(raw_df: pd.DataFrame, topology: str) -> pd.DataFrame:
    """把 6T / 10T 映射到同一个 pooled_union 输入表。"""
    topology = str(topology).upper()
    if topology not in {"6T", "10T"}:
        raise ValueError(f"Unsupported topology: {topology}")

    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df["topology"] = topology
    df["source_row"] = np.arange(len(df), dtype=int)

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().str.lower() == "ok"]

    for flag_col in ["valid_snm", "valid_delay", "valid_design"]:
        if flag_col in df.columns:
            df = df[_as_bool_series(df[flag_col])]

    if topology == "6T":
        # DataLoader 会 dropna，因此不能用 NaN 表示不存在。
        df["fd_present"] = 0.0
        df["fd_width"] = 0.0
        df["fd_model"] = "NOT_APPLICABLE"
    else:
        df["fd_present"] = 1.0

    required = (
        SHARED_CONT_FEATURES
        + TOPOLOGY_CONT_FEATURES
        + ARCH_FEATURES
        + SHARED_CAT_FEATURES
        + TOPOLOGY_CAT_FEATURES
        + RAW_TARGETS
    )
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{topology} dataset is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    numeric_cols = (
        SHARED_CONT_FEATURES
        + TOPOLOGY_CONT_FEATURES
        + ARCH_FEATURES
        + RAW_TARGETS
    )
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    categorical_cols = SHARED_CAT_FEATURES + TOPOLOGY_CAT_FEATURES
    for column in categorical_cols:
        df[column] = df[column].where(df[column].notna(), np.nan)
        df[column] = df[column].astype("string").str.strip()

    df = df.dropna(subset=required).copy()

    positive_cols = (
        SHARED_CONT_FEATURES
        + ARCH_FEATURES
        + [
            "hold_snm", "read_snm", "write_snm",
            "raw_read_delay", "raw_write_delay",
            "single_array_area",
        ]
    )
    for column in positive_cols:
        df = df[df[column] > 0]

    if topology == "10T":
        df = df[df["fd_width"] > 0]

    df["sample_uid"] = topology + ":" + df["source_row"].astype(str)
    return df.reset_index(drop=True)


def _architecture_stratify_key(df: pd.DataFrame) -> Optional[pd.Series]:
    key = df["rows"].astype(str) + "x" + df["cols"].astype(str)
    counts = key.value_counts()
    if len(counts) > 1 and int(counts.min()) >= 2:
        return key
    return None


def split_one_topology(
    df: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stratify = _architecture_stratify_key(df)
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        warnings.warn(
            "Architecture-stratified split is not feasible; using random split.",
            RuntimeWarning,
        )
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def limit_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if int(max_rows) <= 0 or len(df) <= int(max_rows):
        return df.reset_index(drop=True)
    return df.sample(n=int(max_rows), random_state=seed).reset_index(drop=True)


def balance_pooled_training_data(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = df["topology"].value_counts()
    if len(counts) < 2:
        return df.reset_index(drop=True)
    n = int(counts.min())
    pieces = []
    for index, (_, group) in enumerate(df.groupby("topology", sort=True)):
        pieces.append(group.sample(n=n, random_state=seed + index))
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


# =============================================================================
# 固定的 pooled_union 特征配置
# =============================================================================
def build_pooled_union_schema(train_csv: str) -> Dict[str, Any]:
    return {
        "cont_features": (
            SHARED_CONT_FEATURES
            + TOPOLOGY_CONT_FEATURES
            + DERIVED_CONT_FEATURES
        ),
        "arch_features": ARCH_FEATURES,
        "cat_features": SHARED_CAT_FEATURES + TOPOLOGY_CAT_FEATURES,
        "targets": RAW_TARGETS,
        "filter_positive_targets": False,
        "filepath": train_csv,
    }


def print_input_schema_summary() -> None:
    print("\n" + "=" * 108)
    print("[1/6] 联合模型输入定义（pooled_union）")
    print("=" * 108)
    print(f"公共连续变量 ({len(SHARED_CONT_FEATURES)}):")
    print("  " + ", ".join(SHARED_CONT_FEATURES))
    print(f"拓扑专用数值变量 ({len(TOPOLOGY_CONT_FEATURES)}):")
    print("  " + ", ".join(TOPOLOGY_CONT_FEATURES))
    print(f"架构及派生变量 ({len(ARCH_FEATURES) + len(DERIVED_CONT_FEATURES)}):")
    print("  " + ", ".join(ARCH_FEATURES + DERIVED_CONT_FEATURES))
    print(f"分类变量 ({len(SHARED_CAT_FEATURES) + len(TOPOLOGY_CAT_FEATURES)}，随后独热编码):")
    print("  " + ", ".join(SHARED_CAT_FEATURES + TOPOLOGY_CAT_FEATURES))
    print("6T 的 FD 占位:  fd_present=0, fd_width=0, fd_model=NOT_APPLICABLE")
    print("10T 的 FD 输入:  fd_present=1, fd_width=实际值, fd_model=实际值")


# =============================================================================
# 训练与预测：完全复用用户工具库
# =============================================================================
def add_objective_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["min_snm"] = df[["hold_snm", "read_snm", "write_snm"]].min(axis=1)
    df["max_delay"] = df[["read_delay", "write_delay"]].max(axis=1)
    df["max_power"] = df[["read_power", "write_power"]].max(axis=1)
    return df


def convert_raw_to_system_metrics(
    raw_df: pd.DataFrame,
    design_df: pd.DataFrame,
) -> pd.DataFrame:
    processed = process_raw_to_system_metrics(
        raw_df_metrics=raw_df[RAW_TARGETS].reset_index(drop=True),
        rows_array=design_df["rows"].to_numpy(),
        cols_array=design_df["cols"].to_numpy(),
        total_KB=TOTAL_KB,
        output_cols=OUTPUT_COLS,
    )
    return add_objective_columns(processed)



# =============================================================================
# 联合拓扑优化配置：后续更换目标时主要修改这里
# =============================================================================
# direction: "min" 表示最小化；"max" 表示最大化。
# source 必须是 add_optimization_metric_columns() 生成的列名。
#
# 自适应规则：
#   - pymoo 的 n_obj、全局 Pareto 筛选和参考方向会自动适配；
OPTIMIZATION_OBJECTIVES = [
    {
        "name": "max_power",
        "source": "max_power",
        "direction": "min",
        "label": "max_power",
        "unit": "W",
    },
    {
        "name": "max_delay",
        "source": "max_delay",
        "direction": "min",
        "label": "max_delay",
        "unit": "s",
    },
    {
        "name": "area",
        "source": "area",
        "direction": "min",
        "label": "area",
        "unit": "m^2",
    },
    {
        "name": "min_snm",
        "source": "min_snm",
        "direction": "max",
        "label": "min_snm",
        "unit": "V",
    },
]

# 注意：这只是二维展示方式，不会改变上面的四目标优化定义。
PROBLEM_CONSTRAINTS: List[Dict[str, Any]] = []
ENFORCE_PHYSICAL_VALIDITY = True


def configure_problem(problem_config: str | Path) -> None:
    global OPTIMIZATION_OBJECTIVES
    global PROBLEM_CONSTRAINTS
    global ENFORCE_PHYSICAL_VALIDITY
    (
        OPTIMIZATION_OBJECTIVES,
        PROBLEM_CONSTRAINTS,
        ENFORCE_PHYSICAL_VALIDITY,
    ) = load_problem_spec(problem_config)


DEFAULT_MAX_EVALS = 1000

# 每代种群数量
DEFAULT_POP_SIZE = 30

# 简单物理有效性约束开关：
# 后续可以在 check_prediction_constraints() 中扩展
ENABLE_SIMPLE_SIGN_CONSTRAINT = True


# 可直接加入 UNSGA3、CTAEA；参考方向数量会根据目标数和种群规模自动适配。
DEFAULT_ALGORITHMS = "NSGA2,SPEA2,UNSGA3,CTAEA"
BASELINE_ALGORITHMS = {"NSGA2", "SPEA2", "UNSGA3", "CTAEA"}
DEFAULT_BOUNDS_LOWER_Q = 0.01
DEFAULT_BOUNDS_UPPER_Q = 0.99

# 真正由优化器搜索的公共连续变量。派生变量由 feature_engineering 自动生成。
OPTIMIZED_CONT_FEATURES = SHARED_CONT_FEATURES.copy()


# =============================================================================
# pymoo imports
# =============================================================================
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import (
    MixedVariableSampling,
    MixedVariableMating,
    MixedVariableDuplicateElimination,
)
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.util.ref_dirs import get_reference_directions


# =============================================================================
# 训练 pooled_union 模型
# =============================================================================
def train_pooled_union_model(
    train_df: pd.DataFrame,
    *,
    output_dir: Path,
    device: str,
    verbose_library_training: bool,
) -> Tuple[DataLoader, MultiTargetSurrogateManager]:
    """使用用户的 DataLoader / MultiTargetSurrogateManager 训练统一代理。"""
    with tempfile.TemporaryDirectory(prefix="openyield_train_") as temp_dir:
        train_csv = Path(temp_dir) / "train_context.csv"
        train_df.to_csv(train_csv, index=False)
        dataloader = DataLoader(build_pooled_union_schema(str(train_csv)))
        dataloader.add_feature_engineering(feature_engineering)
        X_train, y_train, _ = dataloader.load_and_preprocess()

    print("\n" + "=" * 108)
    print("[3/7] pooled_union TabPFN 训练")
    print("=" * 108)
    print(f"训练上下文: {X_train.shape[0]} 条样本 × {X_train.shape[1]} 个独热编码后特征")

    manager = MultiTargetSurrogateManager(target_names=RAW_TARGETS, device=device)
    if verbose_library_training:
        manager.fit_all(X_train, y_train)
    else:
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                manager.fit_all(X_train, y_train)
        except Exception:
            print(captured.getvalue())
            raise

    shared_id = id(manager.shared_regressor)
    child_ids = {
        id(model._regressor)
        for model in manager.models.values()
        if isinstance(model, TabPFNSurrogate)
    }
    if child_ids != {shared_id}:
        raise RuntimeError(
            "并非所有目标共用同一个 TabPFNRegressor："
            f"manager={shared_id}, children={child_ids}"
        )
    print(f"训练完成；10个目标共享 regressor id={shared_id}")
    return dataloader, manager


# =============================================================================
# 设计空间构建
# =============================================================================
def build_continuous_bounds(
    train_df: pd.DataFrame,
    *,
    lower_q: float,
    upper_q: float,
    min_span: float = 1e-12,
) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in OPTIMIZED_CONT_FEATURES:
        values = pd.to_numeric(train_df[col], errors="coerce").dropna().to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError(f"连续变量没有有效数据: {col}")
        low = float(np.quantile(values, lower_q))
        high = float(np.quantile(values, upper_q))
        if high <= low:
            center = float(np.median(values))
            delta = max(abs(center) * 0.05, min_span)
            low, high = center - delta, center + delta
        low = max(low, min_span)
        high = max(high, low + min_span)
        bounds[col] = (low, high)

    # fd_width 仅从10T训练数据中确定范围。
    fd_values = pd.to_numeric(
        train_df.loc[train_df["topology"] == "10T", "fd_width"],
        errors="coerce",
    ).dropna().to_numpy(float)
    fd_values = fd_values[np.isfinite(fd_values) & (fd_values > 0)]
    if len(fd_values) == 0:
        raise ValueError("10T训练数据中没有有效 fd_width。")
    fd_low = float(np.quantile(fd_values, lower_q))
    fd_high = float(np.quantile(fd_values, upper_q))
    if fd_high <= fd_low:
        center = float(np.median(fd_values))
        delta = max(abs(center) * 0.05, min_span)
        fd_low, fd_high = center - delta, center + delta
    bounds["fd_width"] = (max(fd_low, min_span), max(fd_high, fd_low + min_span))
    return bounds


def build_categorical_choices(train_df: pd.DataFrame) -> Dict[str, List[str]]:
    choices: Dict[str, List[str]] = {}
    for col in SHARED_CAT_FEATURES:
        vals = sorted(train_df[col].dropna().astype(str).unique().tolist())
        if not vals:
            raise ValueError(f"分类变量没有候选值: {col}")
        choices[col] = vals

    fd_vals = sorted(
        train_df.loc[train_df["topology"] == "10T", "fd_model"]
        .dropna().astype(str).unique().tolist()
    )
    fd_vals = [v for v in fd_vals if v != "NOT_APPLICABLE"]
    if not fd_vals:
        raise ValueError("10T训练数据中没有 fd_model 候选值。")
    choices["fd_model"] = fd_vals
    return choices


def build_architecture_templates(train_df: pd.DataFrame) -> List[Dict[str, int]]:
    """
    只保留6T/10T共同支持的架构。

    目的：
    避免10T独有架构（例如64x16）造成拓扑比较不公平。
    """
    if "topology" not in train_df.columns:
        raise ValueError("缺少 topology 列，无法计算共同架构。")

    arch_sets = []
    for topo in ["6T", "10T"]:
        part = train_df[train_df["topology"] == topo]
        arch_sets.append(
            set(
                zip(
                    part["rows"].astype(int),
                    part["cols"].astype(int)
                )
            )
        )

    common_arch = sorted(list(arch_sets[0].intersection(arch_sets[1])))

    if len(common_arch) == 0:
        raise RuntimeError("6T/10T没有共同架构，无法进行公平联合优化。")

    total_bits = TOTAL_KB * 1024 * 8

    result = []
    for rows, cols in common_arch:
        result.append(
            {
                "rows": int(rows),
                "cols": int(cols),
                "num_arrays": int(
                    max(
                        math.ceil(total_bits / (rows * cols)),
                        1,
                    )
                ),
            }
        )

    print(f"共同架构数量: {len(result)}")
    print(f"共同架构: {[(x['rows'], x['cols']) for x in result]}")

    return result


def print_design_space(
    bounds: Dict[str, Tuple[float, float]],
    cat_choices: Dict[str, List[str]],
    architectures: List[Dict[str, int]],
) -> None:
    print("\n" + "=" * 108)
    print("[4/7] 联合搜索空间")
    print("=" * 108)
    print("自由拓扑变量: topology_id ∈ {0: 6T, 1: 10T}")
    print(f"架构模板数量: {len(architectures)}")
    print(f"公共连续尺寸变量: {len(OPTIMIZED_CONT_FEATURES)}")
    print(f"公共器件类型变量: {len(SHARED_CAT_FEATURES)}")
    print("10T专用变量: fd_width, fd_model；当 topology=6T 时自动覆盖为0和NOT_APPLICABLE。")
    print("连续变量边界采用联合训练数据的分位数范围：")
    for name, (low, high) in bounds.items():
        print(f"  {name:23s}: {low:.6e} ~ {high:.6e}")
    print("分类候选：")
    for name, vals in cat_choices.items():
        print(f"  {name:23s}: {vals}")


# =============================================================================
# 目标定义
# =============================================================================
def add_optimization_metric_columns(system_df: pd.DataFrame) -> pd.DataFrame:
    df = add_objective_columns(system_df)
    # max_power × max_delay 的量纲是 J。
    df["power_delay_product"] = df["max_power"] * df["max_delay"]

    # min_snm 越大越好，与 inv_snm 越小越好完全等价。
    min_snm = df["min_snm"].to_numpy(dtype=float)
    valid = np.isfinite(min_snm) & (min_snm > 0)
    inv_snm = np.full(len(df), np.nan, dtype=float)
    inv_snm[valid] = 1.0 / min_snm[valid]
    df["inv_snm"] = inv_snm
    return df


def objective_matrix_for_configs(
    metrics_df: pd.DataFrame,
    objective_configs: Sequence[Dict[str, Any]],
) -> np.ndarray:
    """把任意一组目标统一转换成 pymoo 使用的“全部最小化”矩阵。"""
    if len(objective_configs) == 0:
        raise ValueError("objective_configs 不能为空。")

    columns: List[np.ndarray] = []
    for cfg in objective_configs:
        source = str(cfg["source"])
        if source not in metrics_df.columns:
            raise KeyError(
                f"目标列不存在: {source}；当前可用列为 {list(metrics_df.columns)}"
            )
        values = metrics_df[source].to_numpy(dtype=float)
        direction = str(cfg["direction"]).lower()
        if direction == "max":
            values = -values
        elif direction != "min":
            raise ValueError(f"未知优化方向: {cfg['direction']}")
        columns.append(values)
    return np.column_stack(columns)


def objective_matrix_for_pymoo(metrics_df: pd.DataFrame) -> np.ndarray:
    """完整优化目标矩阵；列数自动等于 OPTIMIZATION_OBJECTIVES 的长度。"""
    return objective_matrix_for_configs(metrics_df, OPTIMIZATION_OBJECTIVES)


def is_pareto_mask(values: np.ndarray) -> np.ndarray:
    """
    返回非支配点掩码。

    参数 values 必须已经统一成全部最小化方向。
    完全相同的目标点会同时保留，以免误删不同拓扑或不同算法的等价设计。
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Pareto 输入必须是二维矩阵，实际 shape={values.shape}")

    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=bool)

    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominated_by_other = np.any(
            np.all(values <= values[i], axis=1)
            & np.any(values < values[i], axis=1)
        )
        if dominated_by_other:
            keep[i] = False
    return keep


def validate_objective_configuration() -> None:
    """Validate the configured multi-objective problem."""
    n_obj = len(OPTIMIZATION_OBJECTIVES)
    if n_obj < 2:
        raise ValueError(
            "OPTIMIZATION_OBJECTIVES 至少需要包含 2 个目标，"
            f"实际为 {n_obj} 个。"
        )

    required_keys = {"name", "source", "direction", "label", "unit"}
    names: List[str] = []
    sources: List[str] = []
    for index, cfg in enumerate(OPTIMIZATION_OBJECTIVES, start=1):
        missing = required_keys.difference(cfg.keys())
        if missing:
            raise ValueError(f"第 {index} 个目标缺少字段: {sorted(missing)}")
        direction = str(cfg["direction"]).lower()
        if direction not in {"min", "max"}:
            raise ValueError(
                f"第 {index} 个目标 direction 必须是 min 或 max，实际为 {direction}"
            )
        names.append(str(cfg["name"]))
        sources.append(str(cfg["source"]))

    if len(set(names)) != len(names):
        raise ValueError(f"目标 name 不能重复: {names}")
    if len(set(sources)) != len(sources):
        raise ValueError(f"目标 source 不能重复: {sources}")


def check_prediction_constraints(metrics_df: pd.DataFrame) -> np.ndarray:
    """
    返回 True 表示候选设计满足全部约束。
    """

    constraints = PROBLEM_CONSTRAINTS if ENABLE_SIMPLE_SIGN_CONSTRAINT else []
    return shared_feasible_mask(
        metrics_df,
        constraints,
        enforce_physical_validity=ENFORCE_PHYSICAL_VALIDITY,
    )

# =============================================================================
# 联合混合变量问题
# =============================================================================
class JointTopologySRAMProblem(Problem):
    def __init__(
        self,
        *,
        surrogate_manager: MultiTargetSurrogateManager,
        dataloader: DataLoader,
        continuous_bounds: Dict[str, Tuple[float, float]],
        categorical_choices: Dict[str, List[str]],
        architecture_templates: List[Dict[str, int]],
        algorithm_name: str,
        evaluation_budget: Optional[int] = None,
    ):
        self.surrogate_manager = surrogate_manager
        self.dataloader = dataloader
        self.continuous_bounds = continuous_bounds
        self.categorical_choices = categorical_choices
        self.architecture_templates = architecture_templates
        self.algorithm_name = algorithm_name
        self.evaluation_budget = (
            None if evaluation_budget is None else int(evaluation_budget)
        )
        self.eval_counter = 0
        self.evaluation_records: List[pd.DataFrame] = []

        variables: Dict[str, Any] = {
            "topology_id": Integer(bounds=(0, 1)),
            "arch_id": Integer(bounds=(0, len(architecture_templates) - 1)),
        }
        for name in OPTIMIZED_CONT_FEATURES:
            variables[name] = Real(bounds=continuous_bounds[name])
        variables["fd_width"] = Real(bounds=continuous_bounds["fd_width"])
        for col in SHARED_CAT_FEATURES:
            variables[f"{col}_id"] = Integer(
                bounds=(0, len(categorical_choices[col]) - 1)
            )
        variables["fd_model_id"] = Integer(
            bounds=(0, len(categorical_choices["fd_model"]) - 1)
        )

        super().__init__(
            vars=variables,
            n_obj=len(OPTIMIZATION_OBJECTIVES),
            n_ieq_constr=1,
        )

    def build_candidate_dataframe(self, x) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for xi in x:
            topology = "10T" if int(xi["topology_id"]) == 1 else "6T"
            arch = self.architecture_templates[int(xi["arch_id"])]
            row: Dict[str, Any] = {
                "topology": topology,
                "rows": int(arch["rows"]),
                "cols": int(arch["cols"]),
                "num_arrays": int(arch["num_arrays"]),
            }
            for name in OPTIMIZED_CONT_FEATURES:
                row[name] = float(xi[name])
            for col in SHARED_CAT_FEATURES:
                row[col] = self.categorical_choices[col][int(xi[f"{col}_id"])]

            if topology == "10T":
                row["fd_present"] = 1.0
                row["fd_width"] = float(xi["fd_width"])
                row["fd_model"] = self.categorical_choices["fd_model"][
                    int(xi["fd_model_id"])
                ]
            else:
                # 6T时FD变量不参与物理设计，统一规范化为固定占位值。
                row["fd_present"] = 0.0
                row["fd_width"] = 0.0
                row["fd_model"] = "NOT_APPLICABLE"
            records.append(row)
        return pd.DataFrame(records)

    def predict_metrics(self, candidate_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = self.dataloader.transform_features(candidate_df)
        raw_pred = pd.DataFrame(
            self.surrogate_manager.predict(X, return_std=False)
        )[RAW_TARGETS]
        system_pred = process_raw_to_system_metrics(
            raw_df_metrics=raw_pred,
            rows_array=candidate_df["rows"].to_numpy(),
            cols_array=candidate_df["cols"].to_numpy(),
            total_KB=TOTAL_KB,
            output_cols=OUTPUT_COLS,
        )
        system_pred = add_optimization_metric_columns(system_pred)
        return raw_pred.reset_index(drop=True), system_pred.reset_index(drop=True)

    def _evaluate(self, x, out, *args, **kwargs):
        candidates = self.build_candidate_dataframe(x).reset_index(drop=True)
        n_requested = len(candidates)
        remaining = (
            n_requested
            if self.evaluation_budget is None
            else max(self.evaluation_budget - self.eval_counter, 0)
        )
        n = min(n_requested, remaining)
        F = np.full((n_requested, len(OPTIMIZATION_OBJECTIVES)), 1.0e30)
        G = np.ones((n_requested, 1), dtype=float)
        if n == 0:
            out["F"] = F
            out["G"] = G
            return

        evaluated = candidates.iloc[:n].reset_index(drop=True)
        raw_pred, metrics = self.predict_metrics(evaluated)
        F[:n] = objective_matrix_for_pymoo(metrics)
        valid_mask = check_prediction_constraints(metrics)
        G[:n, 0] = (~valid_mask).astype(float)
        out["F"] = F
        out["G"] = G

        start = self.eval_counter + 1
        self.eval_counter += n
        record = pd.concat(
            [evaluated, raw_pred.add_prefix("raw_"), metrics], axis=1
        )
        record.insert(0, "evaluation", np.arange(start, start + n))
        record.insert(0, "algorithm", self.algorithm_name)
        record.insert(2, "constraint_valid", valid_mask.astype(bool))
        self.evaluation_records.append(record)

    def evaluation_history(self) -> pd.DataFrame:
        if not self.evaluation_records:
            return pd.DataFrame()
        return pd.concat(self.evaluation_records, ignore_index=True)


class BudgetTraceCallback(Callback):
    def __init__(self):
        super().__init__()
        self.records: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def notify(self, algorithm):
        self.records.append(
            {
                "generation": int(algorithm.n_gen),
                "n_eval": int(algorithm.evaluator.n_eval),
                "elapsed_seconds": float(time.time() - self.start_time),
            }
        )


# =============================================================================
# 算法构建
# =============================================================================
def build_algorithm(name: str, *, pop_size: int, n_obj: int, seed: int):
    name = name.upper()
    duplicate = MixedVariableDuplicateElimination()
    sampling = MixedVariableSampling()
    mating = MixedVariableMating(eliminate_duplicates=duplicate)

    if name == "NSGA2":
        return NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            mating=mating,
            eliminate_duplicates=duplicate,
        )
    if name == "SPEA2":
        return SPEA2(
            pop_size=pop_size,
            sampling=sampling,
            mating=mating,
            eliminate_duplicates=duplicate,
        )
    if name in {"UNSGA3", "CTAEA"}:
        ref_dirs = get_reference_directions(
            "energy",
            n_obj,
            pop_size,
            seed=seed,
        )
        if len(ref_dirs) != pop_size:
            raise RuntimeError(
                f"{name} created {len(ref_dirs)} reference directions; "
                f"expected exactly pop_size={pop_size}."
            )
        if name == "UNSGA3":
            return UNSGA3(
                ref_dirs=ref_dirs,
                pop_size=pop_size,
                sampling=sampling,
                mating=mating,
                eliminate_duplicates=duplicate,
            )
        return CTAEA(
            ref_dirs=ref_dirs,
            sampling=sampling,
            mating=mating,
            eliminate_duplicates=duplicate,
        )
    raise ValueError(f"不支持的算法: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="6T/10T topology+sizing joint multi-objective optimization using pooled_union TabPFN."
    )
    parser.add_argument("--data-6t", default=DEFAULT_6T_DATASET)
    parser.add_argument("--data-10t", default=DEFAULT_10T_DATASET)
    parser.add_argument(
        "--problem-config",
        default=PACKAGE_ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE)
    parser.add_argument("--max-train-per-topology", type=int, default=DEFAULT_MAX_TRAIN_PER_TOPOLOGY)
    parser.add_argument("--balance-topologies", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose-library-training", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS)
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--bounds-lower-q", type=float, default=DEFAULT_BOUNDS_LOWER_Q)
    parser.add_argument("--bounds-upper-q", type=float, default=DEFAULT_BOUNDS_UPPER_Q)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA，但 torch.cuda.is_available() 为 False。")
    return requested


def validate_args(args: argparse.Namespace) -> None:
    validate_objective_configuration()
    for attr in ["data_6t", "data_10t"]:
        path = Path(getattr(args, attr))
        if not path.exists():
            raise FileNotFoundError(f"数据集不存在: {path}")
    if not (0 < args.test_size < 1):
        raise ValueError("test_size必须在(0,1)之间。")
    if args.max_evals <= 0 or args.pop_size <= 1:
        raise ValueError("max_evals和pop_size必须为正。")
    if args.max_evals < args.pop_size:
        raise ValueError("max_evals must be at least pop_size.")
    if args.max_evals % args.pop_size:
        raise ValueError(
            "For an exact evolutionary budget, max_evals must be divisible "
            "by pop_size; partial generations are not comparable."
        )
    if not (0 <= args.bounds_lower_q < args.bounds_upper_q <= 1):
        raise ValueError("边界分位数必须满足 0 <= lower < upper <= 1。")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    configure_problem(args.problem_config)
    validate_args(args)
    seed_set(args.seed)
    device = resolve_device(args.device)
    output_dir, evals_dir, fronts_dir = create_run_directory(args.output_dir)

    algorithms = [x.strip().upper() for x in args.algorithms.split(",") if x.strip()]
    unsupported = sorted(set(algorithms).difference(BASELINE_ALGORITHMS))
    if unsupported:
        raise ValueError(
            "Evolutionary baselines are limited to NSGA2, SPEA2, UNSGA3 and "
            f"CTAEA. Unsupported: {unsupported}"
        )
    print("=" * 108)
    print("实验1：6T/10T联合搜索——拓扑作为真正自由变量")
    print("=" * 108)
    print(f"设备: {device}")
    print(f"算法: {algorithms}")
    print(f"每个算法代理评价预算: {args.max_evals} 个候选设计")
    print(f"种群规模: {args.pop_size}")
    print("架构策略: 仅使用6T/10T共同架构")
    print(f"优化目标数: {len(OPTIMIZATION_OBJECTIVES)}")
    for objective_index, cfg in enumerate(OPTIMIZATION_OBJECTIVES, start=1):
        print(
            f"目标{objective_index}: {cfg['direction']} {cfg['source']} "
            f"[{cfg['label']}, {cfg['unit']}]"
        )
    print("绘图数量: 每个算法1张 + 所有算法合并1张；所有图均不设置标题。")
    print("合并图中: 算法=颜色+线型，拓扑=点形(6T圆点、10T三角形)。")
    print("Pareto front: feasible non-dominated designs from all recorded evaluations.")
    print("拓扑、公共尺寸、器件类型、架构以及10T的FD变量在同一种群中联合搜索。")
    print(f"输出目录: {output_dir}")

    raw_6t = pd.read_csv(args.data_6t)
    raw_10t = pd.read_csv(args.data_10t)
    df_6t = harmonize_topology_dataframe(raw_6t, "6T")
    df_10t = harmonize_topology_dataframe(raw_10t, "10T")
    train_6t, test_6t = split_one_topology(
        df_6t, test_size=args.test_size, seed=args.seed
    )
    train_10t, test_10t = split_one_topology(
        df_10t, test_size=args.test_size, seed=args.seed + 1
    )
    train_6t = limit_rows(train_6t, args.max_train_per_topology, args.seed)
    train_10t = limit_rows(train_10t, args.max_train_per_topology, args.seed + 1)
    pooled_train = pd.concat([train_6t, train_10t], ignore_index=True)
    if args.balance_topologies:
        pooled_train = balance_pooled_training_data(pooled_train, args.seed)
    pooled_train = pooled_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print("\n" + "=" * 108)
    print("[1/7] 数据")
    print("=" * 108)
    print(f"6T有效/训练: {len(df_6t)} / {len(train_6t)}")
    print(f"10T有效/训练: {len(df_10t)} / {len(train_10t)}")
    print(f"联合TabPFN上下文: {len(pooled_train)}")

    print_input_schema_summary()
    dataloader, manager = train_pooled_union_model(
        pooled_train,
        output_dir=output_dir,
        device=device,
        verbose_library_training=args.verbose_library_training,
    )

    bounds = build_continuous_bounds(
        pooled_train,
        lower_q=args.bounds_lower_q,
        upper_q=args.bounds_upper_q,
    )
    cat_choices = build_categorical_choices(pooled_train)
    architectures = build_architecture_templates(pooled_train)
    print_design_space(bounds, cat_choices, architectures)

    config = vars(args).copy()
    config["resolved_device"] = device
    config["algorithms_resolved"] = algorithms
    config["objectives"] = OPTIMIZATION_OBJECTIVES
    config["n_objectives"] = len(OPTIMIZATION_OBJECTIVES)
    config["front_source"] = (
        "all recorded feasible evaluations; no extra surrogate queries"
    )
    config["output_contract"] = (
        "evaluations/<algorithm>.csv contains one row per design query; "
        "pareto_fronts/<algorithm>.csv uses every configured objective"
    )
    config["budget_definition"] = (
        "one candidate design predicted for all raw targets = one evaluation"
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 108)
    print("[5/7] 多算法联合优化")
    print("=" * 108)
    all_fronts: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for algorithm_name in algorithms:
        print(f"\n--- {algorithm_name}: budget={args.max_evals} ---")
        problem = JointTopologySRAMProblem(
            surrogate_manager=manager,
            dataloader=dataloader,
            continuous_bounds=bounds,
            categorical_choices=cat_choices,
            architecture_templates=architectures,
            algorithm_name=algorithm_name,
            evaluation_budget=args.max_evals,
        )
        manager.reset_query_count()
        algorithm = build_algorithm(
            algorithm_name,
            pop_size=args.pop_size,
            n_obj=len(OPTIMIZATION_OBJECTIVES),
            seed=args.seed,
        )
        trace = BudgetTraceCallback()
        start = time.time()
        minimize(
            problem,
            algorithm,
            termination=("n_eval", args.max_evals),
            seed=args.seed,
            callback=trace,
            verbose=False,
        )
        elapsed = time.time() - start

        evaluations = problem.evaluation_history()
        tabpfn_queries = manager.get_query_count()
        if tabpfn_queries != args.max_evals:
            raise RuntimeError(
                f"{algorithm_name} used {tabpfn_queries} TabPFN design queries; "
                f"expected exactly {args.max_evals}."
            )
        evaluations = public_evaluations(
            evaluations, algorithm_name, expected_rows=args.max_evals
        )
        evaluations.to_csv(evals_dir / f"{algorithm_name}.csv", index=False)

        front = pareto_front_from_evaluations(
            evaluations,
            OPTIMIZATION_OBJECTIVES,
        )
        if front.empty:
            print(f"{algorithm_name}: no feasible Pareto point was found.")

        front = public_pareto_front(
            front,
            algorithm_name,
            objective_columns=[cfg["source"] for cfg in OPTIMIZATION_OBJECTIVES],
        )
        front.to_csv(fronts_dir / f"{algorithm_name}.csv", index=False)

        all_fronts.append(front)
        summary_rows.append(
            {
                "algorithm": algorithm_name,
                "evaluations": int(len(evaluations)),
                "feasible_evaluations": int(
                    evaluations["constraint_valid"].sum()
                ),
                "pareto_front_size": int(len(front)),
                "tabpfn_design_queries": int(tabpfn_queries),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"完成: eval={len(evaluations)}, full-front={len(front)}, "
            f"time={elapsed:.1f}s"
        )

    if not all_fronts:
        raise RuntimeError("所有算法都没有产生有效Pareto前沿。")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "algorithm_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"Evaluations: {evals_dir}")
    print(f"Pareto fronts: {fronts_dir}")

if __name__ == "__main__":
    main()
