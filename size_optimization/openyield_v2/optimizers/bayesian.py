#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bayesian multi-objective baselines for the shared 6T/10T search space."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


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
        pareto_front_from_evaluations as shared_pareto_front,
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
        pareto_front_from_evaluations as shared_pareto_front,
    )
    from output_schema import (  # type: ignore[no-redef]  # noqa: E402
        create_run_directory,
        public_evaluations,
        public_pareto_front,
    )

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 用户最常修改的配置
# =============================================================================
SEED = 33
TOTAL_KB = 32
OUTPUT_COLS = 16

PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_6T_DATASET = str(PACKAGE_ROOT / "datasets" / "train_6t.csv")
DEFAULT_10T_DATASET = str(PACKAGE_ROOT / "datasets" / "train_10t.csv")
DEFAULT_OUTPUT_DIR = str(
    PACKAGE_ROOT / "runs" / "optimization" / "bayesian"
)

# 数据和代理模型配置：与第二段 EA 代码一致。
DEFAULT_TEST_SIZE = 0.05
DEFAULT_DEVICE = "auto"
DEFAULT_MAX_TRAIN_PER_TOPOLOGY = 250  # 0 表示不限制
DEFAULT_BALANCE_TOPOLOGIES = False
DEFAULT_VERBOSE_LIBRARY_TRAINING = False

# BO 预算配置。
DEFAULT_MAX_EVALS = 1000
DEFAULT_INIT_SAMPLES = 100
DEFAULT_BATCH_SIZE = 70
DEFAULT_CANDIDATE_POOL = 8192
DEFAULT_ALGORITHMS = "GPBO,PAREGO,MACE"

# GP 与采集函数配置。
DEFAULT_GP_MAX_TRAIN = 600  # 0 表示使用全部已评价点
DEFAULT_GP_NOISE = 1.0e-6
DEFAULT_LCB_BETA = 2.0
DEFAULT_PAREGO_RHO = 0.05
DEFAULT_DIVERSITY_RADIUS = 0.08

# 搜索边界采用联合训练数据的分位数。
DEFAULT_BOUNDS_LOWER_Q = 0.01
DEFAULT_BOUNDS_UPPER_Q = 0.99

# 简单约束开关，与第二段 EA 代码一致。
ENABLE_SIMPLE_SIGN_CONSTRAINT = True

# 绘图配置。


# =============================================================================
# 输入变量与原始目标
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

OPTIMIZED_CONT_FEATURES = SHARED_CONT_FEATURES.copy()


# =============================================================================
# =============================================================================
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


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def harmonize_topology_dataframe(raw_df: pd.DataFrame, topology: str) -> pd.DataFrame:
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
        # DataLoader 会 dropna，不能用 NaN 表示不存在的 FD。
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
    from sklearn.model_selection import train_test_split

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
# pooled_union TabPFN
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
    print("[2/7] 联合模型输入定义（pooled_union）")
    print("=" * 108)
    print(f"公共连续变量 ({len(SHARED_CONT_FEATURES)}):")
    print("  " + ", ".join(SHARED_CONT_FEATURES))
    print(f"拓扑专用数值变量 ({len(TOPOLOGY_CONT_FEATURES)}):")
    print("  " + ", ".join(TOPOLOGY_CONT_FEATURES))
    print(f"架构及派生变量 ({len(ARCH_FEATURES) + len(DERIVED_CONT_FEATURES)}):")
    print("  " + ", ".join(ARCH_FEATURES + DERIVED_CONT_FEATURES))
    print(f"分类变量 ({len(SHARED_CAT_FEATURES) + len(TOPOLOGY_CAT_FEATURES)}，独热编码):")
    print("  " + ", ".join(SHARED_CAT_FEATURES + TOPOLOGY_CAT_FEATURES))
    print("6T FD占位: fd_present=0, fd_width=0, fd_model=NOT_APPLICABLE")
    print("10T FD输入: fd_present=1, fd_width=实际值, fd_model=实际模型")


def train_pooled_union_model(
    train_df: pd.DataFrame,
    *,
    output_dir: Path,
    device: str,
    verbose_library_training: bool,
) -> Tuple[DataLoader, MultiTargetSurrogateManager]:
    temp_context = tempfile.TemporaryDirectory(prefix="openyield_train_")
    train_csv = Path(temp_context.name) / "train_context.csv"
    train_df.to_csv(train_csv, index=False)

    dataloader = DataLoader(build_pooled_union_schema(str(train_csv)))
    dataloader.add_feature_engineering(feature_engineering)

    # surrogate_utils.DataLoader 的接口。
    X_train, y_train, _ = dataloader.load_and_preprocess()
    temp_context.cleanup()

    print("\n" + "=" * 108)
    print("[3/7] pooled_union TabPFN 训练")
    print("=" * 108)
    print(f"训练上下文: {X_train.shape[0]} 条样本 × {X_train.shape[1]} 个独热编码后特征")

    manager = MultiTargetSurrogateManager(
        target_names=RAW_TARGETS,
        device=device,
    )

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
# 搜索空间构建：共同架构、正确的 FD 范围、无重复 topology 变量
# =============================================================================
def build_continuous_bounds(
    train_df: pd.DataFrame,
    *,
    lower_q: float,
    upper_q: float,
    min_span: float = 1.0e-12,
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

    # fd_width 只能由 10T 的正值数据确定；不能把 6T 占位零值混入边界。
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
    bounds["fd_width"] = (
        max(fd_low, min_span),
        max(fd_high, fd_low + min_span),
    )
    return bounds


def build_categorical_choices(train_df: pd.DataFrame) -> Dict[str, List[str]]:
    choices: Dict[str, List[str]] = {}
    for col in SHARED_CAT_FEATURES:
        vals = sorted(train_df[col].dropna().astype(str).unique().tolist())
        if not vals:
            raise ValueError(f"分类变量没有候选值: {col}")
        choices[col] = vals

    # 10T 不允许选择 6T 的 NOT_APPLICABLE 占位模型。
    fd_vals = sorted(
        train_df.loc[train_df["topology"] == "10T", "fd_model"]
        .dropna().astype(str).unique().tolist()
    )
    fd_vals = [value for value in fd_vals if value != "NOT_APPLICABLE"]
    if not fd_vals:
        raise ValueError("10T训练数据中没有 fd_model 候选值。")
    choices["fd_model"] = fd_vals
    return choices


def build_architecture_templates(train_df: pd.DataFrame) -> List[Dict[str, int]]:
    """只保留 6T 和 10T 都支持的 rows × cols 架构。"""
    if "topology" not in train_df.columns:
        raise ValueError("缺少 topology 列，无法计算共同架构。")

    architecture_sets = []
    for topology in ["6T", "10T"]:
        part = train_df[train_df["topology"] == topology]
        architecture_sets.append(
            set(zip(part["rows"].astype(int), part["cols"].astype(int)))
        )

    common_architectures = sorted(
        architecture_sets[0].intersection(architecture_sets[1])
    )
    if not common_architectures:
        raise RuntimeError("6T/10T没有共同架构，无法进行公平联合优化。")

    total_bits = TOTAL_KB * 1024 * 8
    templates: List[Dict[str, int]] = []
    for rows, cols in common_architectures:
        templates.append(
            {
                "rows": int(rows),
                "cols": int(cols),
                "num_arrays": int(max(math.ceil(total_bits / (rows * cols)), 1)),
            }
        )

    print(f"共同架构数量: {len(templates)}")
    print(f"共同架构: {[(x['rows'], x['cols']) for x in templates]}")
    return templates


def print_design_space(
    bounds: Dict[str, Tuple[float, float]],
    categorical_choices: Dict[str, List[str]],
    architectures: List[Dict[str, int]],
) -> None:
    print("\n" + "=" * 108)
    print("[4/7] 联合 BO 搜索空间")
    print("=" * 108)
    print("自由拓扑变量: topology_id ∈ {0: 6T, 1: 10T}")
    print(f"共同架构模板数量: {len(architectures)}")
    print(f"公共连续尺寸变量: {len(OPTIMIZED_CONT_FEATURES)}")
    print(f"公共器件类型变量: {len(SHARED_CAT_FEATURES)}")
    print("10T专用变量: fd_width, fd_model；6T时强制覆盖为0和NOT_APPLICABLE。")
    print("连续变量范围：")
    for name, (low, high) in bounds.items():
        print(f"  {name:23s}: {low:.6e} ~ {high:.6e}")
    print("分类候选：")
    for name, values in categorical_choices.items():
        print(f"  {name:23s}: {values}")


class JointSearchSpace:
    """
    把混合变量编码到 [0,1]^D，供 sklearn GP 使用。

    topology 只编码一次。分类变量通过区间映射成类别编号；解码后的真实
    DataFrame 仍使用与 EA 代码完全相同的物理变量和类别字符串。
    """

    def __init__(
        self,
        *,
        continuous_bounds: Dict[str, Tuple[float, float]],
        categorical_choices: Dict[str, List[str]],
        architecture_templates: List[Dict[str, int]],
    ) -> None:
        self.continuous_bounds = continuous_bounds
        self.categorical_choices = categorical_choices
        self.architecture_templates = architecture_templates

        self.unit_feature_names = (
            ["topology_id", "arch_id"]
            + OPTIMIZED_CONT_FEATURES
            + ["fd_width"]
            + [f"{name}_id" for name in SHARED_CAT_FEATURES]
            + ["fd_model_id"]
        )
        self.dim = len(self.unit_feature_names)

    @staticmethod
    def _decode_index(unit_value: float, count: int) -> int:
        if count <= 0:
            raise ValueError("离散候选数必须为正。")
        value = float(np.clip(unit_value, 0.0, np.nextafter(1.0, 0.0)))
        return min(int(math.floor(value * count)), count - 1)

    def sample_lhs(self, n: int, seed: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, self.dim), dtype=float)
        sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
        return np.asarray(sampler.random(n=n), dtype=float)

    def sample_random(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.empty((0, self.dim), dtype=float)
        return rng.random((n, self.dim), dtype=float)

    def decode_one(self, unit_vector: np.ndarray) -> Dict[str, Any]:
        u = np.asarray(unit_vector, dtype=float).reshape(-1)
        if len(u) != self.dim:
            raise ValueError(f"单位向量维度错误: expected={self.dim}, actual={len(u)}")

        pointer = 0
        topology_id = self._decode_index(u[pointer], 2)
        topology = "10T" if topology_id == 1 else "6T"
        pointer += 1

        arch_id = self._decode_index(u[pointer], len(self.architecture_templates))
        architecture = self.architecture_templates[arch_id]
        pointer += 1

        row: Dict[str, Any] = {
            "topology": topology,
            "rows": int(architecture["rows"]),
            "cols": int(architecture["cols"]),
            "num_arrays": int(architecture["num_arrays"]),
        }

        for name in OPTIMIZED_CONT_FEATURES:
            low, high = self.continuous_bounds[name]
            row[name] = float(low + np.clip(u[pointer], 0.0, 1.0) * (high - low))
            pointer += 1

        fd_low, fd_high = self.continuous_bounds["fd_width"]
        proposed_fd_width = float(
            fd_low + np.clip(u[pointer], 0.0, 1.0) * (fd_high - fd_low)
        )
        pointer += 1

        for name in SHARED_CAT_FEATURES:
            values = self.categorical_choices[name]
            category_id = self._decode_index(u[pointer], len(values))
            row[name] = values[category_id]
            pointer += 1

        fd_values = self.categorical_choices["fd_model"]
        fd_model_id = self._decode_index(u[pointer], len(fd_values))
        proposed_fd_model = fd_values[fd_model_id]
        pointer += 1

        if pointer != self.dim:
            raise RuntimeError(f"解码指针错误: pointer={pointer}, dim={self.dim}")

        if topology == "10T":
            row["fd_present"] = 1.0
            row["fd_width"] = proposed_fd_width
            row["fd_model"] = proposed_fd_model
        else:
            row["fd_present"] = 0.0
            row["fd_width"] = 0.0
            row["fd_model"] = "NOT_APPLICABLE"
        return row

    def decode_many(self, unit_matrix: np.ndarray) -> pd.DataFrame:
        matrix = np.asarray(unit_matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return pd.DataFrame([self.decode_one(row) for row in matrix])


# =============================================================================
# raw -> system 指标、目标方向和约束
# =============================================================================
def add_optimization_metric_columns(system_df: pd.DataFrame) -> pd.DataFrame:
    df = system_df.copy()
    df["min_snm"] = df[["hold_snm", "read_snm", "write_snm"]].min(axis=1)
    df["max_delay"] = df[["read_delay", "write_delay"]].max(axis=1)
    df["max_power"] = df[["read_power", "write_power"]].max(axis=1)
    df["power_delay_product"] = df["max_power"] * df["max_delay"]

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
    if not objective_configs:
        raise ValueError("objective_configs 不能为空。")

    columns: List[np.ndarray] = []
    for config in objective_configs:
        source = str(config["source"])
        if source not in metrics_df.columns:
            raise KeyError(f"目标列不存在: {source}")
        values = metrics_df[source].to_numpy(dtype=float)
        direction = str(config["direction"]).lower()
        if direction == "max":
            values = -values
        elif direction != "min":
            raise ValueError(f"未知目标方向: {config['direction']}")
        columns.append(values)
    return np.column_stack(columns)


def objective_matrix(metrics_df: pd.DataFrame) -> np.ndarray:
    return objective_matrix_for_configs(metrics_df, OPTIMIZATION_OBJECTIVES)


def check_prediction_constraints(metrics_df: pd.DataFrame) -> np.ndarray:
    constraints = PROBLEM_CONSTRAINTS if ENABLE_SIMPLE_SIGN_CONSTRAINT else []
    return shared_feasible_mask(
        metrics_df,
        constraints,
        enforce_physical_validity=ENFORCE_PHYSICAL_VALIDITY,
    )


def validate_objective_configuration() -> None:
    n_obj = len(OPTIMIZATION_OBJECTIVES)
    if n_obj < 2:
        raise ValueError(
            "OPTIMIZATION_OBJECTIVES 至少需要包含2个目标，"
            f"实际为{n_obj}个。"
        )

    required_keys = {"name", "source", "direction", "label", "unit"}
    names: List[str] = []
    sources: List[str] = []
    for index, config in enumerate(OPTIMIZATION_OBJECTIVES, start=1):
        missing = required_keys.difference(config.keys())
        if missing:
            raise ValueError(f"第{index}个目标缺少字段: {sorted(missing)}")
        direction = str(config["direction"]).lower()
        if direction not in {"min", "max"}:
            raise ValueError(f"第{index}个目标 direction 必须是 min 或 max。")
        names.append(str(config["name"]))
        sources.append(str(config["source"]))
    if len(set(names)) != len(names) or len(set(sources)) != len(sources):
        raise ValueError("优化目标 name/source 不能重复。")


class TabPFNOracle:
    def __init__(
        self,
        *,
        dataloader: DataLoader,
        surrogate_manager: MultiTargetSurrogateManager,
    ) -> None:
        self.dataloader = dataloader
        self.surrogate_manager = surrogate_manager

    def evaluate(
        self,
        candidate_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        candidate_df = candidate_df.reset_index(drop=True)
        X = self.dataloader.transform_features(candidate_df)
        raw_pred = pd.DataFrame(
            self.surrogate_manager.predict(X, return_std=False)
        )[RAW_TARGETS].reset_index(drop=True)

        system_pred = process_raw_to_system_metrics(
            raw_df_metrics=raw_pred,
            rows_array=candidate_df["rows"].to_numpy(),
            cols_array=candidate_df["cols"].to_numpy(),
            total_KB=TOTAL_KB,
            output_cols=OUTPUT_COLS,
        )
        metrics = add_optimization_metric_columns(system_pred).reset_index(drop=True)
        F = objective_matrix(metrics)
        valid = check_prediction_constraints(metrics)
        valid &= np.all(np.isfinite(F), axis=1)

        records = pd.concat(
            [candidate_df, raw_pred.add_prefix("raw_"), metrics],
            axis=1,
        )
        return records, metrics, F, valid


# =============================================================================
# Pareto、标量化和 GP 采集函数
# =============================================================================
def non_dominated_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Pareto输入必须为二维矩阵，实际shape={values.shape}")
    mask = np.zeros(len(values), dtype=bool)
    if len(values) == 0:
        return mask
    finite = np.all(np.isfinite(values), axis=1)
    finite_indices = np.flatnonzero(finite)
    if len(finite_indices) == 0:
        return mask
    front = NonDominatedSorting().do(
        values[finite_indices],
        only_non_dominated_front=True,
    )
    mask[finite_indices[np.asarray(front, dtype=int)]] = True
    return mask


def normalize_objective_matrix(
    values: np.ndarray,
    feasible_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    把所有最小化目标缩放到近似 [0,1]。

    如果已经出现可行点，归一化上下界只由可行点确定，并把不可行点统一
    放到劣侧；如果尚无可行点，则暂时利用全部有限点建立 GP，避免 BO
    在初始化阶段因没有可行点而无法继续。
    """
    values = np.asarray(values, dtype=float)
    feasible_mask = np.asarray(feasible_mask, dtype=bool)
    finite = np.all(np.isfinite(values), axis=1)
    has_feasible = bool(np.any(feasible_mask & finite))
    reference_mask = (feasible_mask & finite) if has_feasible else finite
    if not np.any(reference_mask):
        raise RuntimeError("所有已评价点的目标均为非有限值，BO无法建立内部GP。")

    lower = np.min(values[reference_mask], axis=0)
    upper = np.max(values[reference_mask], axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    normalized = (values - lower) / span

    if has_feasible:
        normalized[~(feasible_mask & finite), :] = 1.25
    else:
        normalized[~finite, :] = 1.25

    normalized = np.nan_to_num(normalized, nan=1.25, posinf=1.25, neginf=1.25)
    return normalized, lower, upper, has_feasible


def sample_weight(n_objectives: int, rng: np.random.Generator) -> np.ndarray:
    weight = rng.dirichlet(np.ones(n_objectives, dtype=float))
    weight = np.maximum(weight, 1.0e-8)
    return weight / np.sum(weight)


def linear_scalarization(normalized_F: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.sum(normalized_F * weight.reshape(1, -1), axis=1)


def augmented_tchebycheff(
    normalized_F: np.ndarray,
    weight: np.ndarray,
    rho: float,
) -> np.ndarray:
    weighted = normalized_F * weight.reshape(1, -1)
    return np.max(weighted, axis=1) + float(rho) * np.sum(weighted, axis=1)


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best: float,
) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1.0e-12)
    improvement = float(best) - mean
    z_value = improvement / std
    return improvement * norm.cdf(z_value) + std * norm.pdf(z_value)


def probability_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best: float,
) -> np.ndarray:
    std = np.maximum(np.asarray(std, dtype=float), 1.0e-12)
    z_value = (float(best) - np.asarray(mean, dtype=float)) / std
    return norm.cdf(z_value)


def minmax_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = np.zeros(len(values), dtype=float)
    if not np.any(finite):
        return result
    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))
    if high <= low + 1.0e-15:
        result[finite] = 0.5
    else:
        result[finite] = (values[finite] - low) / (high - low)
    result[~finite] = 0.0
    return result


def select_gp_training_subset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_train: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """保留一半当前优值点，再随机保留一半探索点。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if max_train <= 0 or len(X) <= max_train:
        return X, y

    n_best = max(1, max_train // 2)
    order = np.argsort(y, kind="mergesort")
    best_indices = order[:n_best]
    remaining_indices = order[n_best:]
    n_random = max_train - len(best_indices)
    if n_random > 0:
        random_indices = rng.choice(
            remaining_indices,
            size=min(n_random, len(remaining_indices)),
            replace=False,
        )
        selected = np.concatenate([best_indices, random_indices])
    else:
        selected = best_indices
    selected = np.unique(selected)
    return X[selected], y[selected]


class ScalarGaussianProcess:
    def __init__(self, *, noise: float, seed: int) -> None:
        kernel = (
            ConstantKernel(1.0, (1.0e-3, 1.0e3))
            * Matern(length_scale=0.2, length_scale_bounds=(1.0e-3, 1.0e2), nu=2.5)
            + WhiteKernel(
                noise_level=max(float(noise), 1.0e-12),
                noise_level_bounds=(1.0e-12, 1.0e-1),
            )
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1.0e-10,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, std = self.model.predict(np.asarray(X, dtype=float), return_std=True)
        return np.asarray(mean, dtype=float), np.maximum(np.asarray(std, dtype=float), 1.0e-12)


def greedy_diverse_topk(
    score_to_maximize: np.ndarray,
    candidates: np.ndarray,
    *,
    batch_size: int,
    diversity_radius: float,
) -> np.ndarray:
    """按采集值选点，并用局部惩罚避免同一批次重复/扎堆。"""
    score = minmax_1d(score_to_maximize)
    candidates = np.asarray(candidates, dtype=float)
    n_select = min(int(batch_size), len(candidates))
    if n_select <= 0:
        return np.empty(0, dtype=int)

    selected: List[int] = []
    available = np.ones(len(candidates), dtype=bool)
    radius = max(float(diversity_radius), 1.0e-8)

    for _ in range(n_select):
        work = score.copy()
        if selected:
            selected_X = candidates[np.asarray(selected, dtype=int)]
            diff = candidates[:, None, :] - selected_X[None, :, :]
            rms_distance = np.sqrt(np.mean(diff * diff, axis=2))
            min_distance = np.min(rms_distance, axis=1)
            local_penalty = 1.0 - np.exp(-0.5 * (min_distance / radius) ** 2)
            work = work * local_penalty
        work[~available] = -np.inf
        index = int(np.argmax(work))
        if not np.isfinite(work[index]):
            remaining = np.flatnonzero(available)
            if len(remaining) == 0:
                break
            index = int(remaining[0])
        selected.append(index)
        available[index] = False
    return np.asarray(selected, dtype=int)


def select_mace_batch(
    candidate_X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    best: float,
    beta: float,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    MACE：把 LCB、-EI、-PI 作为三个最小化采集目标，先得到采集空间
    的非支配层，再从前几层中用设计空间最大最小距离选择批次。
    """
    lcb = mean - float(beta) * std
    ei = expected_improvement(mean, std, best)
    pi = probability_improvement(mean, std, best)
    acquisition_matrix = np.column_stack([lcb, -ei, -pi])

    target_pool_size = min(
        len(candidate_X),
        max(int(batch_size) * 30, int(batch_size)),
    )
    fronts = NonDominatedSorting().do(
        acquisition_matrix,
        n_stop_if_ranked=target_pool_size,
    )
    pool_parts: List[np.ndarray] = []
    ranked = 0
    for front in fronts:
        front = np.asarray(front, dtype=int)
        if len(front) == 0:
            continue
        pool_parts.append(front)
        ranked += len(front)
        if ranked >= target_pool_size:
            break
    if pool_parts:
        pool = np.concatenate(pool_parts)
    else:
        pool = np.arange(len(candidate_X), dtype=int)

    pool_X = candidate_X[pool]
    pool_A = acquisition_matrix[pool]
    normalized_A = np.column_stack([minmax_1d(pool_A[:, j]) for j in range(3)])

    n_select = min(int(batch_size), len(pool))
    selected_local: List[int] = []
    available = np.ones(len(pool), dtype=bool)
    if n_select > 0:
        # 第一个点取三种采集函数综合最优位置。
        first = int(np.argmin(np.sum(normalized_A, axis=1)))
        selected_local.append(first)
        available[first] = False

    while len(selected_local) < n_select:
        selected_X = pool_X[np.asarray(selected_local, dtype=int)]
        design_diff = pool_X[:, None, :] - selected_X[None, :, :]
        design_distance = np.sqrt(np.mean(design_diff * design_diff, axis=2))
        min_design_distance = np.min(design_distance, axis=1)

        selected_A = normalized_A[np.asarray(selected_local, dtype=int)]
        acquisition_diff = normalized_A[:, None, :] - selected_A[None, :, :]
        acquisition_distance = np.sqrt(np.mean(acquisition_diff * acquisition_diff, axis=2))
        min_acquisition_distance = np.min(acquisition_distance, axis=1)

        diversity = 0.5 * minmax_1d(min_design_distance) + 0.5 * minmax_1d(
            min_acquisition_distance
        )
        diversity[~available] = -np.inf
        next_index = int(np.argmax(diversity))
        if not np.isfinite(diversity[next_index]):
            remaining = np.flatnonzero(available)
            if len(remaining) == 0:
                break
            next_index = int(remaining[0])
        selected_local.append(next_index)
        available[next_index] = False

    selected_global = pool[np.asarray(selected_local, dtype=int)]
    details = {
        "mace_candidate_pf_pool": int(len(pool)),
        "mace_first_front": int(len(fronts[0])) if len(fronts) > 0 else 0,
    }
    return selected_global, details


# =============================================================================
# 多目标贝叶斯优化器
# =============================================================================
class BayesianOptimizer:
    def __init__(
        self,
        *,
        method: str,
        oracle: TabPFNOracle,
        search_space: JointSearchSpace,
        max_evals: int,
        init_samples: int,
        batch_size: int,
        candidate_pool: int,
        gp_max_train: int,
        gp_noise: float,
        lcb_beta: float,
        parego_rho: float,
        diversity_radius: float,
        seed: int,
    ) -> None:
        self.method = str(method).upper()
        if self.method not in {"GPBO", "PAREGO", "MACE"}:
            raise ValueError(f"不支持的 BO 算法: {method}")
        self.oracle = oracle
        self.search_space = search_space
        self.max_evals = int(max_evals)
        self.init_samples = min(int(init_samples), self.max_evals)
        self.batch_size = int(batch_size)
        self.candidate_pool = int(candidate_pool)
        self.gp_max_train = int(gp_max_train)
        self.gp_noise = float(gp_noise)
        self.lcb_beta = float(lcb_beta)
        self.parego_rho = float(parego_rho)
        self.diversity_radius = float(diversity_radius)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.X_observed: List[np.ndarray] = []
        self.F_observed: List[np.ndarray] = []
        self.feasible_observed: List[bool] = []
        self.record_parts: List[pd.DataFrame] = []
        self.history: List[Dict[str, Any]] = []
        self.eval_count = 0
        self.start_time = time.time()

    def _evaluate(
        self,
        unit_X: np.ndarray,
        *,
        iteration: int,
        phase: str,
    ) -> None:
        unit_X = np.asarray(unit_X, dtype=float)
        if unit_X.ndim == 1:
            unit_X = unit_X.reshape(1, -1)
        if len(unit_X) == 0:
            return

        designs = self.search_space.decode_many(unit_X)
        records, _, F, feasible = self.oracle.evaluate(designs)

        n = len(records)
        start = self.eval_count + 1
        self.eval_count += n
        records.insert(0, "algorithm", self.method)
        records.insert(1, "evaluation", np.arange(start, start + n, dtype=int))
        records.insert(2, "iteration", int(iteration))
        records.insert(3, "phase", str(phase))
        records.insert(4, "constraint_valid", feasible.astype(bool))
        self.record_parts.append(records)

        self.X_observed.extend(unit_X.copy())
        self.F_observed.extend(np.asarray(F, dtype=float).copy())
        self.feasible_observed.extend(feasible.astype(bool).tolist())

    def _current_front_size(self) -> int:
        if not self.F_observed:
            return 0
        F = np.asarray(self.F_observed, dtype=float)
        feasible = np.asarray(self.feasible_observed, dtype=bool)
        finite = np.all(np.isfinite(F), axis=1)
        indices = np.flatnonzero(feasible & finite)
        if len(indices) == 0:
            return 0
        return int(np.sum(non_dominated_mask(F[indices])))

    def _record_history(
        self,
        *,
        iteration: int,
        batch_evals: int,
        weight: Optional[np.ndarray],
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        feasible = np.asarray(self.feasible_observed, dtype=bool)
        row: Dict[str, Any] = {
            "algorithm": self.method,
            "iteration": int(iteration),
            "batch_evals": int(batch_evals),
            "cumulative_evals": int(self.eval_count),
            "feasible_evals": int(np.sum(feasible)),
            "current_pareto_size": self._current_front_size(),
            "elapsed_seconds": float(time.time() - self.start_time),
            "scalarization_weight": (
                json.dumps(weight.tolist()) if weight is not None else ""
            ),
        }
        if details:
            row.update(details)
        self.history.append(row)

    def _select_next_batch(
        self,
        *,
        batch_size: int,
        iteration: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        X_observed = np.asarray(self.X_observed, dtype=float)
        F_observed = np.asarray(self.F_observed, dtype=float)
        feasible = np.asarray(self.feasible_observed, dtype=bool)
        normalized_F, _, _, has_feasible = normalize_objective_matrix(
            F_observed,
            feasible,
        )

        weight = sample_weight(normalized_F.shape[1], self.rng)
        if self.method == "GPBO":
            scalar_y = linear_scalarization(normalized_F, weight)
        else:
            scalar_y = augmented_tchebycheff(
                normalized_F,
                weight,
                self.parego_rho,
            )

        X_gp, y_gp = select_gp_training_subset(
            X_observed,
            scalar_y,
            max_train=self.gp_max_train,
            rng=self.rng,
        )
        gp = ScalarGaussianProcess(
            noise=self.gp_noise,
            seed=self.seed + iteration,
        )
        gp.fit(X_gp, y_gp)

        pool_size = max(self.candidate_pool, batch_size)
        candidate_X = self.search_space.sample_random(pool_size, self.rng)
        mean, std = gp.predict(candidate_X)
        best = float(np.min(scalar_y[feasible])) if np.any(feasible) else float(np.min(scalar_y))

        details: Dict[str, Any] = {
            "gp_train_size": int(len(X_gp)),
            "candidate_pool": int(len(candidate_X)),
            "had_feasible_before_batch": bool(has_feasible),
            "scalar_best": best,
        }

        if self.method == "GPBO":
            lcb = mean - self.lcb_beta * std
            selected = greedy_diverse_topk(
                -lcb,
                candidate_X,
                batch_size=batch_size,
                diversity_radius=self.diversity_radius,
            )
        elif self.method == "PAREGO":
            ei = expected_improvement(mean, std, best)
            selected = greedy_diverse_topk(
                ei,
                candidate_X,
                batch_size=batch_size,
                diversity_radius=self.diversity_radius,
            )
        else:
            selected, mace_details = select_mace_batch(
                candidate_X,
                mean,
                std,
                best=best,
                beta=self.lcb_beta,
                batch_size=batch_size,
            )
            details.update(mace_details)

        if len(np.unique(selected)) != len(selected):
            raise RuntimeError("同一 BO 批次出现重复候选索引。")
        return candidate_X[selected], weight, details

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("\n" + "=" * 108)
        print(f"Bayesian Optimization: {self.method}")
        print("=" * 108)

        initial_X = self.search_space.sample_lhs(self.init_samples, self.seed)
        self._evaluate(initial_X, iteration=0, phase="initialization")
        self._record_history(
            iteration=0,
            batch_evals=len(initial_X),
            weight=None,
        )
        print(
            f"初始化: {self.eval_count}/{self.max_evals}, "
            f"可行={sum(self.feasible_observed)}, PF={self._current_front_size()}"
        )

        iteration = 0
        while self.eval_count < self.max_evals:
            iteration += 1
            current_batch_size = min(
                self.batch_size,
                self.max_evals - self.eval_count,
            )
            next_X, weight, details = self._select_next_batch(
                batch_size=current_batch_size,
                iteration=iteration,
            )
            self._evaluate(next_X, iteration=iteration, phase="bayesian_optimization")
            self._record_history(
                iteration=iteration,
                batch_evals=len(next_X),
                weight=weight,
                details=details,
            )
            print(
                f"迭代{iteration:03d}: {self.eval_count}/{self.max_evals}, "
                f"可行={sum(self.feasible_observed)}, "
                f"PF={self._current_front_size()}, "
                f"weight={np.round(weight, 3).tolist()}"
            )

        evaluated = pd.concat(self.record_parts, ignore_index=True)
        pareto = shared_pareto_front(evaluated, OPTIMIZATION_OBJECTIVES)

        history_df = pd.DataFrame(self.history)
        return evaluated, pareto, history_df


# =============================================================================
# Result formatting
# =============================================================================
def topology_counts(df: pd.DataFrame) -> Tuple[int, int]:
    if df.empty or "topology" not in df.columns:
        return 0, 0
    counts = df["topology"].astype(str).value_counts()
    return int(counts.get("6T", 0)), int(counts.get("10T", 0))


# =============================================================================
# 命令行与主程序
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="6T/10T topology+sizing joint multi-objective Bayesian optimization."
    )
    parser.add_argument("--data-6t", default=DEFAULT_6T_DATASET)
    parser.add_argument("--data-10t", default=DEFAULT_10T_DATASET)
    parser.add_argument(
        "--problem-config",
        default=PACKAGE_ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE)
    parser.add_argument(
        "--max-train-per-topology",
        type=int,
        default=DEFAULT_MAX_TRAIN_PER_TOPOLOGY,
    )
    parser.add_argument(
        "--balance-topologies",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BALANCE_TOPOLOGIES,
    )
    parser.add_argument(
        "--verbose-library-training",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_VERBOSE_LIBRARY_TRAINING,
    )
    parser.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS)
    parser.add_argument("--init-samples", type=int, default=DEFAULT_INIT_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--gp-max-train", type=int, default=DEFAULT_GP_MAX_TRAIN)
    parser.add_argument("--gp-noise", type=float, default=DEFAULT_GP_NOISE)
    parser.add_argument("--lcb-beta", type=float, default=DEFAULT_LCB_BETA)
    parser.add_argument("--parego-rho", type=float, default=DEFAULT_PAREGO_RHO)
    parser.add_argument(
        "--diversity-radius",
        type=float,
        default=DEFAULT_DIVERSITY_RADIUS,
    )
    parser.add_argument("--bounds-lower-q", type=float, default=DEFAULT_BOUNDS_LOWER_Q)
    parser.add_argument("--bounds-upper-q", type=float, default=DEFAULT_BOUNDS_UPPER_Q)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA，但 torch.cuda.is_available() 为 False。")
    return requested


def normalize_algorithm_names(raw: str) -> List[str]:
    aliases = {
        "GPBO": "GPBO",
        "PAREGO": "PAREGO",
        "PAR-EGO": "PAREGO",
        "MACE": "MACE",
    }
    algorithms: List[str] = []
    for item in str(raw).split(","):
        key = item.strip().upper()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"不支持的算法: {item}; 可选 GPBO, PAREGO, MACE")
        resolved = aliases[key]
        if resolved not in algorithms:
            algorithms.append(resolved)
    if not algorithms:
        raise ValueError("至少需要指定一个 BO 算法。")
    return algorithms


def validate_args(args: argparse.Namespace) -> None:
    validate_objective_configuration()
    for attribute in ["data_6t", "data_10t"]:
        path = Path(getattr(args, attribute))
        if not path.exists():
            raise FileNotFoundError(f"数据集不存在: {path}")
    if not (0 < args.test_size < 1):
        raise ValueError("test_size必须在(0,1)之间。")
    if args.max_evals <= 0:
        raise ValueError("max_evals必须为正。")
    if args.init_samples <= 1:
        raise ValueError("init_samples至少为2。")
    if args.init_samples > args.max_evals:
        raise ValueError("init_samples不能大于max_evals。")
    if args.batch_size <= 0 or args.candidate_pool <= 0:
        raise ValueError("batch_size和candidate_pool必须为正。")
    if args.candidate_pool < args.batch_size:
        raise ValueError("candidate_pool must be at least batch_size.")
    if args.gp_max_train != 0 and args.gp_max_train < 2:
        raise ValueError("gp_max_train must be 0 or at least 2.")
    if args.gp_noise <= 0 or args.lcb_beta <= 0:
        raise ValueError("gp_noise和lcb_beta必须为正。")
    if args.parego_rho < 0 or args.diversity_radius <= 0:
        raise ValueError("parego_rho必须非负，diversity_radius必须为正。")
    if not (0 <= args.bounds_lower_q < args.bounds_upper_q <= 1):
        raise ValueError("边界分位数必须满足 0 <= lower < upper <= 1。")


def main() -> None:
    args = parse_args()
    configure_problem(args.problem_config)
    validate_args(args)
    algorithms = normalize_algorithm_names(args.algorithms)
    seed_set(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    output_dir, evaluations_dir, fronts_dir = create_run_directory(args.output_dir)

    print("=" * 108)
    print("实验1：6T/10T联合搜索——多目标贝叶斯优化对齐版")
    print("=" * 108)
    print(f"设备: {device}")
    print(f"算法: {algorithms}")
    print(f"每个算法 TabPFN oracle 评价预算: {args.max_evals}")
    print(f"初始化/批次/候选池: {args.init_samples}/{args.batch_size}/{args.candidate_pool}")
    print(f"GP最大训练点数: {args.gp_max_train}（0表示全部）")
    print("架构策略: 仅使用6T/10T共同架构")
    print("最终结果口径: BO全部已查询可行点中的非支配解（BO没有最后一代）")
    print(f"优化目标数: {len(OPTIMIZATION_OBJECTIVES)}")
    for index, config in enumerate(OPTIMIZATION_OBJECTIVES, start=1):
        print(
            f"目标{index}: {config['direction']} {config['source']} "
            f"[{config['label']}, {config['unit']}]"
        )
    print(f"输出目录: {output_dir}")

    raw_6t = pd.read_csv(args.data_6t)
    raw_10t = pd.read_csv(args.data_10t)
    df_6t = harmonize_topology_dataframe(raw_6t, "6T")
    df_10t = harmonize_topology_dataframe(raw_10t, "10T")
    train_6t, test_6t = split_one_topology(
        df_6t,
        test_size=args.test_size,
        seed=args.seed,
    )
    train_10t, test_10t = split_one_topology(
        df_10t,
        test_size=args.test_size,
        seed=args.seed + 1,
    )
    train_6t = limit_rows(train_6t, args.max_train_per_topology, args.seed)
    train_10t = limit_rows(train_10t, args.max_train_per_topology, args.seed + 1)
    pooled_train = pd.concat([train_6t, train_10t], ignore_index=True)
    if args.balance_topologies:
        pooled_train = balance_pooled_training_data(pooled_train, args.seed)
    pooled_train = pooled_train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

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
    categorical_choices = build_categorical_choices(pooled_train)
    architectures = build_architecture_templates(pooled_train)
    print_design_space(bounds, categorical_choices, architectures)

    search_space = JointSearchSpace(
        continuous_bounds=bounds,
        categorical_choices=categorical_choices,
        architecture_templates=architectures,
    )
    oracle = TabPFNOracle(
        dataloader=dataloader,
        surrogate_manager=manager,
    )
    print(f"BO单位超立方体维度: {search_space.dim}")

    config = vars(args).copy()
    config["resolved_device"] = device
    config["algorithms_resolved"] = algorithms
    config["optimization_objectives"] = OPTIMIZATION_OBJECTIVES
    config["unit_feature_names"] = search_space.unit_feature_names
    config["continuous_bounds"] = {
        key: [float(value[0]), float(value[1])]
        for key, value in bounds.items()
    }
    config["categorical_choices"] = categorical_choices
    config["architecture_templates"] = architectures
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 108)
    print("[5/7] 运行贝叶斯优化")
    print("=" * 108)

    all_fronts: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for algorithm_index, algorithm_name in enumerate(algorithms):
        algorithm_start = time.time()
        manager.reset_query_count()
        optimizer = BayesianOptimizer(
            method=algorithm_name,
            oracle=oracle,
            search_space=search_space,
            max_evals=args.max_evals,
            init_samples=args.init_samples,
            batch_size=args.batch_size,
            candidate_pool=args.candidate_pool,
            gp_max_train=args.gp_max_train,
            gp_noise=args.gp_noise,
            lcb_beta=args.lcb_beta,
            parego_rho=args.parego_rho,
            diversity_radius=args.diversity_radius,
            # 所有算法使用同一初始 seed，因此 LHS 初始化设计相同。
            seed=args.seed,
        )
        evaluated, front, history = optimizer.run()
        tabpfn_queries = manager.get_query_count()
        if tabpfn_queries != args.max_evals:
            raise RuntimeError(
                f"{algorithm_name} used {tabpfn_queries} TabPFN design queries; "
                f"expected exactly {args.max_evals}."
            )
        runtime_seconds = float(time.time() - algorithm_start)

        evaluated = public_evaluations(
            evaluated, algorithm_name, expected_rows=args.max_evals
        )
        evaluated.to_csv(evaluations_dir / f"{algorithm_name}.csv", index=False)
        front = public_pareto_front(
            front,
            algorithm_name,
            objective_columns=[cfg["source"] for cfg in OPTIMIZATION_OBJECTIVES],
        )
        front.to_csv(fronts_dir / f"{algorithm_name}.csv", index=False)

        if not front.empty:
            all_fronts.append(front)

        feasible_count = int(evaluated["constraint_valid"].astype(bool).sum())
        front_6t, front_10t = topology_counts(front)
        summary_rows.append(
            {
                "algorithm": algorithm_name,
                "evaluations": int(len(evaluated)),
                "feasible_evaluations": feasible_count,
                "pareto_front_size": int(len(front)),
                "tabpfn_design_queries": int(tabpfn_queries),
                "elapsed_seconds": runtime_seconds,
            }
        )
        print(
            f"{algorithm_name}完成: evaluations={len(evaluated)}, "
            f"feasible={feasible_count}, PF={len(front)}, "
            f"6T/10T={front_6t}/{front_10t}, time={runtime_seconds:.2f}s"
        )

    print("\n" + "=" * 108)
    print("[6/7] 汇总")
    print("=" * 108)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "algorithm_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"Evaluations: {evaluations_dir}")
    print(f"Pareto fronts: {fronts_dir}")
    return

if __name__ == "__main__":
    main()
