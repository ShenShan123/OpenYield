from __future__ import annotations

"""
SRAM Circuit Optimization Utilities
SRAM电路优化工具函数

This file contains common utilities for SRAM circuit optimization algorithms.
该文件包含SRAM电路优化算法的通用工具函数。
"""

import numpy as np
import os
try:
    import torch
except ImportError:
    torch = None
import random
import time
import csv
import json
import traceback
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Any, Union, Tuple
from abc import ABC, abstractmethod
from config import SRAM_CONFIG

# Import SRAM simulation modules
# 导入SRAM仿真模块
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from utils import estimate_bitcell_area, estimate_total_area, estimate_array_area, estimate_scaled_array_area


def seed_set(seed):
    """
    Fix the random seed for reproducibility
    固定随机种子以确保结果可重现
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def create_directories():
    """
    Create necessary directories for simulation and results
    创建仿真和结果所需的目录
    """
    # Create directories under sim/opt/
    # 在sim/opt/下创建目录
    Path("sim/opt/plots").mkdir(exist_ok=True, parents=True)
    Path("sim/opt/results").mkdir(exist_ok=True, parents=True)
    Path("sim").mkdir(exist_ok=True, parents=True)


def get_default_initial_params():
    """
    Get default initial parameters for SRAM optimization
    获取SRAM优化的默认初始参数
    """
    length = 50e-9
    return {"nmos_model_name": "NMOS_VTG", "pmos_model_name": "PMOS_VTG", "pd_width": 0.205e-6, "pu_width": 0.09e-6, "pg_width": 0.135e-6, "length": length, "length_nm": length * 1e9}


def _load_sram_config_from_yaml():
    """
    Load full SRAM_CONFIG from YAML files.
    从 YAML 文件加载完整 SRAM_CONFIG。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    sram_config = SRAM_CONFIG()
    sram_config.load_all_configs(
        global_file=os.path.join(project_root, "sram_compiler/config_yaml/global.yaml"),
        circuit_configs={
            "SRAM_6T_CELL": os.path.join(project_root, "sram_compiler/config_yaml/sram_6t_cell.yaml"),
            "WORDLINEDRIVER": os.path.join(project_root, "sram_compiler/config_yaml/wordline_driver.yaml"),
            "PRECHARGE": os.path.join(project_root, "sram_compiler/config_yaml/precharge.yaml"),
            "COLUMNMUX": os.path.join(project_root, "sram_compiler/config_yaml/mux.yaml"),
            "SENSEAMP": os.path.join(project_root, "sram_compiler/config_yaml/sa.yaml"),
            "WRITEDRIVER": os.path.join(project_root, "sram_compiler/config_yaml/write_driver.yaml"),
            "DECODER": os.path.join(project_root, "sram_compiler/config_yaml/decoder.yaml"),
        },
    )
    return sram_config


def get_params_from_yaml():
    """
    Load transistor parameters from SRAM YAML config so scripts honor YAML edits
    从 SRAM 的 YAML 配置中加载晶体管参数，使脚本遵循 YAML 修改
    """
    sram_config = _load_sram_config_from_yaml()

    # Extract values from loaded config
    pu_val = float(getattr(sram_config.sram_6t_cell.pmos_width, "value", 0.0))
    nmos_vals = getattr(sram_config.sram_6t_cell.nmos_width, "value", [0.0, 0.0])
    if isinstance(nmos_vals, (list, tuple)):
        pd_val = float(nmos_vals[0])
        pg_val = float(nmos_vals[1])
    else:
        pd_val = float(nmos_vals)
        pg_val = float(nmos_vals)

    length_val = float(getattr(sram_config.sram_6t_cell.length, "value", 0.0))
    pmos_model_val = getattr(sram_config.sram_6t_cell.pmos_model, "value", "PMOS_VTG")
    nmos_model_val = getattr(sram_config.sram_6t_cell.nmos_model, "value", "NMOS_VTG")
    if isinstance(nmos_model_val, (list, tuple)):
        nmos_model_name = str(nmos_model_val[0])
    else:
        nmos_model_name = str(nmos_model_val)

    return {
        "nmos_model_name": nmos_model_name,
        "pmos_model_name": str(pmos_model_val),
        "pd_width": pd_val,
        "pg_width": pg_val,
        "pu_width": pu_val,
        "length": length_val,
        "length_nm": length_val * 1e9,
    }


def get_peripheral_params_from_yaml():
    """
    Load peripheral circuit parameters from YAML configs.
    从 YAML 配置加载外围电路参数（column mux, sense amp, wordline driver）。
    """
    sram_config = _load_sram_config_from_yaml()
    params = {}

    # --- Column Mux ---
    mux = sram_config.column_mux
    params["mux_p_width"] = float(getattr(mux.pmos_width, "value", 0.135e-6))
    params["mux_n_width"] = float(getattr(mux.nmos_width, "value", 0.135e-6))
    params["mux_length"] = float(getattr(mux.length, "value", 50e-9))
    params["mux_p_model"] = str(getattr(mux.pmos_model, "value", "PMOS_VTG"))
    params["mux_n_model"] = str(getattr(mux.nmos_model, "value", "NMOS_VTG"))

    # --- Sense Amp ---
    sa = sram_config.senseamp
    params["sa_p_width"] = float(getattr(sa.pmos_width, "value", 0.54e-6))
    params["sa_n_width"] = float(getattr(sa.nmos_width, "value", 0.27e-6))
    params["sa_length"] = float(getattr(sa.length, "value", 50e-9))
    params["sa_p_model"] = str(getattr(sa.pmos_model, "value", "PMOS_VTG"))
    params["sa_n_model"] = str(getattr(sa.nmos_model, "value", "NMOS_VTG"))

    # --- Wordline Driver ---
    wld = sram_config.wordline_driver
    wld_pmos_vals = getattr(wld.pmos_width, "value", [0.27e-6, 0.27e-6])
    if isinstance(wld_pmos_vals, (list, tuple)):
        params["wld_nand_p_width"] = float(wld_pmos_vals[0])
        params["wld_inv_p_width"] = float(wld_pmos_vals[1])
    else:
        params["wld_nand_p_width"] = float(wld_pmos_vals)
        params["wld_inv_p_width"] = float(wld_pmos_vals)

    wld_nmos_vals = getattr(wld.nmos_width, "value", [0.18e-6, 0.09e-6])
    if isinstance(wld_nmos_vals, (list, tuple)):
        params["wld_nand_n_width"] = float(wld_nmos_vals[0])
        params["wld_inv_n_width"] = float(wld_nmos_vals[1])
    else:
        params["wld_nand_n_width"] = float(wld_nmos_vals)
        params["wld_inv_n_width"] = float(wld_nmos_vals)

    params["wld_length"] = float(getattr(wld.length, "value", 50e-9))

    wld_pmos_model_vals = getattr(wld.pmos_model, "value", ["PMOS_VTG", "PMOS_VTG"])
    if isinstance(wld_pmos_model_vals, (list, tuple)):
        params["wld_nand_p_model"] = str(wld_pmos_model_vals[0])
        params["wld_inv_p_model"] = str(wld_pmos_model_vals[1])
    else:
        params["wld_nand_p_model"] = str(wld_pmos_model_vals)
        params["wld_inv_p_model"] = str(wld_pmos_model_vals)

    wld_nmos_model_vals = getattr(wld.nmos_model, "value", ["NMOS_VTG", "NMOS_VTG"])
    if isinstance(wld_nmos_model_vals, (list, tuple)):
        params["wld_nand_n_model"] = str(wld_nmos_model_vals[0])
        params["wld_inv_n_model"] = str(wld_nmos_model_vals[1])
    else:
        params["wld_nand_n_model"] = str(wld_nmos_model_vals)
        params["wld_inv_n_model"] = str(wld_nmos_model_vals)

    return params


def get_composite_initial_params():
    """
    Get full initial parameter dict (bitcell + peripheral circuits) from YAML.
    从 YAML 获取完整初始参数字典（bitcell + 外围电路）。
    """
    bc = get_params_from_yaml()
    periph = get_peripheral_params_from_yaml()
    bc.update(periph)
    return bc


def estimate_scaled_total_area(params, num_rows=32, num_cols=1, num_arrays=1):
    """Pitch-scaling area model: 列/行间距随 bitcell 尺寸缩放，外围电路随 SA/WLD/Precharge 宽度缩放。返回 m²。"""
    sa_max_w  = max(params.get("sa_p_width", 0.54e-6),
                    params.get("sa_n_width", 0.27e-6))
    wld_max_w = max(params.get("wld_nand_p_width", 0.27e-6),
                    params.get("wld_inv_p_width",  0.27e-6),
                    params.get("wld_nand_n_width", 0.18e-6),
                    params.get("wld_inv_n_width",  0.09e-6))
    prc_max_w = params.get("prc_p_width", 0.27e-6)
    return estimate_scaled_array_area(
        num_rows, num_cols, num_arrays,
        w_access=params["pg_width"],
        w_pd=params["pd_width"],
        w_pu=params["pu_width"],
        l_transistor=params["length"],
        sa_max_width=sa_max_w,
        wld_max_width=wld_max_w,
        prc_max_width=prc_max_w,
    )


def get_default_normalized_vector(parameter_space, rows=16, cols=16):
    def _index_value(index, length):
        if length <= 1:
            return 0.0
        value = (index + 0.01) / float(length)
        if value >= 1.0:
            value = (length - 0.01) / float(length)
        if value < 0.0:
            value = 0.0
        return float(value)

    if hasattr(parameter_space, "base_space") and hasattr(parameter_space, "row_choices") and hasattr(parameter_space, "column_choices"):
        row_idx = 0
        col_idx = 0
        if rows in parameter_space.row_choices:
            row_idx = parameter_space.row_choices.index(rows)
        if cols in parameter_space.column_choices:
            col_idx = parameter_space.column_choices.index(cols)
        prefix = [_index_value(row_idx, len(parameter_space.row_choices)), _index_value(col_idx, len(parameter_space.column_choices))]
        return prefix + get_default_normalized_vector(parameter_space.base_space, rows=rows, cols=cols)

    if not hasattr(parameter_space, "param_info"):
        return []

    x = []
    for param_name, info in parameter_space.param_info.items():
        if info["type"] == "continuous_list":
            default = info["default"]
            for i in range(info["count"]):
                default_val = default[i] if isinstance(default, (list, tuple)) else default
                lower = info["lower"][i] if isinstance(info["lower"], (list, tuple)) else info["lower"]
                upper = info["upper"][i] if isinstance(info["upper"], (list, tuple)) else info["upper"]
                if float(upper) == float(lower):
                    x.append(0.0)
                else:
                    x.append(float((float(default_val) - float(lower)) / (float(upper) - float(lower))))
        elif info["type"] == "continuous_scalar":
            default_val = info["default"][0] if isinstance(info["default"], (list, tuple)) else info["default"]
            lower = info["lower"]
            upper = info["upper"]
            if float(upper) == float(lower):
                x.append(0.0)
            else:
                x.append(float((float(default_val) - float(lower)) / (float(upper) - float(lower))))
        elif info["type"] == "categorical_list":
            defaults = info["default"]
            for i in range(info["count"]):
                default_val = defaults[i] if isinstance(defaults, (list, tuple)) else defaults
                choices = info["choices"]
                idx = choices.index(default_val) if default_val in choices else 0
                x.append(_index_value(idx, len(choices)))
    return x


def get_default_transistor_features(parameter_space):
    base_space = parameter_space.base_space if hasattr(parameter_space, "base_space") else parameter_space
    default_vector = get_default_normalized_vector(base_space)
    default_vector = list(default_vector) if default_vector else []
    while len(default_vector) < 4:
        default_vector.append(0.0)

    nmos_idx = 0
    pmos_idx = 0
    dim_idx = 0
    if hasattr(base_space, "param_info"):
        for param_name, info in base_space.param_info.items():
            if info["type"] == "continuous_list":
                dim_idx += info["count"]
            elif info["type"] == "continuous_scalar":
                dim_idx += 1
            elif info["type"] == "categorical_list":
                choices = info.get("choices", [])
                if param_name == "nmos_model":
                    if choices and dim_idx < len(default_vector):
                        idx = int(default_vector[dim_idx] * len(choices))
                        nmos_idx = max(0, min(idx, len(choices) - 1))
                if param_name == "pmos_model":
                    if choices and dim_idx < len(default_vector):
                        idx = int(default_vector[dim_idx] * len(choices))
                        pmos_idx = max(0, min(idx, len(choices) - 1))
                dim_idx += info["count"]

    return default_vector[:4], nmos_idx, pmos_idx


def get_default_fallback_result():
    """
    Get default fallback result when initial simulation fails
    获取初始仿真失败时的默认后备结果
    """
    return {"hold_snm": {"success": True, "snm": 0.30173446708423357}, "read_snm": {"success": True, "snm": 0.12591724230394877}, "write_snm": {"success": True, "snm": 0.3732610863628419}, "read": {"success": True, "delay": 2.0883543988703797e-10, "power": 4.024476625792127e-05}, "write": {"success": True, "delay": 6.086260190977158e-11, "power": 3.975272388991992e-05}}


def run_initial_evaluation(params):
    """
    Run initial parameter evaluation
    运行初始参数评估
    """
    objectives, constraints, result, success = evaluate_sram(params)

    if success:
        formatted_result = {"hold_snm": {"success": True, "snm": result["hold_snm"]}, "read_snm": {"success": True, "snm": result["read_snm"]}, "write_snm": {"success": True, "snm": result["write_snm"]}, "read": {"success": True, "delay": result["read_delay"], "power": abs(result["read_power"])}, "write": {"success": True, "delay": result["write_delay"], "power": abs(result["write_power"])}}
        return formatted_result, params
    else:
        return get_default_fallback_result(), params


# 通用配置支持
# Universal Configuration Support
class ConfigLoader:
    """
    配置文件加载器
    Configuration file loader
    """

    def __init__(self, config_path: str = "config_sram.yaml"):
        """
        初始化配置加载器
        Initialize configuration loader
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: 配置文件 {self.config_path} 不存在，使用默认配置")
            return self._get_default_config()
        except Exception as e:
            print(f"Warning: 配置文件加载失败 {e}，使用默认配置")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {"sim_params": {"vdd": 1.0, "temperature": 27, "num_rows": 32, "num_cols": 1, "monte_carlo_runs": 1, "timeout": 120}, "pdk": {"path": "tran_models/models_TT.spice"}, "subcircuit": {"name": "SRAM_6T_Cell", "parameter_space": {"parameters": {"pmos_width": {"type": "continuous list", "names": ["pu"], "upper": [1.35e-7], "lower": [4.5e-8], "default": [9e-8]}, "nmos_width": {"type": "continuous list", "names": ["pd", "pg"], "upper": [3.075e-7, 2.025e-7], "lower": [1.025e-7, 6.75e-8], "default": [2.05e-7, 1.35e-7]}, "length": {"type": "continuous value", "names": "l", "upper": 1e-7, "lower": 3e-8, "default": 50e-9}, "nmos_model": {"type": "categorical list", "names": ["pd", "pg"], "choices": ["NMOS_VTL", "NMOS_VTG", "NMOS_VTH"], "default": ["NMOS_VTG", "NMOS_VTG"]}, "pmos_model": {"type": "categorical list", "names": ["pu"], "choices": ["PMOS_VTL", "PMOS_VTG", "PMOS_VTH"], "default": ["PMOS_VTG"]}}}}}

    def get_global_params(self) -> Dict:
        """获取全局参数"""
        return {"sim_params": self.config.get("sim_params", {}), "pdk": self.config.get("pdk", {})}

    def get_subcircuit_config(self) -> Dict:
        """获取子电路配置"""
        return self.config.get("subcircuit", {})


class ModifiedSRAMParameterSpace:
    """
    SRAM Parameter Space class to handle multi-dimensional tensors
    SRAM参数空间类，支持处理多维张量和配置文件
    """

    def __init__(self, config_path: str = "config_sram.yaml"):
        """
        初始化参数空间
        Initialize parameter space
        """
        self.config_loader = ConfigLoader(config_path)
        self.algorithm_min_length = 70e-9
        self.param_config = self.config_loader.get_subcircuit_config().get("parameter_space", {}).get("parameters", {})
        self.param_info = self._parse_parameters()
        self.bounds = self._create_bounds()

        # 为了兼容原始代码，保留原始属性
        self._setup_legacy_attributes()

    def _parse_parameters(self) -> Dict:
        """解析参数配置"""
        param_info = {}

        for param_name, param_config in self.param_config.items():
            param_type = param_config["type"]

            if param_type == "continuous list":
                names = param_config["names"]
                upper = param_config["upper"]
                lower = param_config["lower"]
                default = param_config["default"]

                param_info[param_name] = {"type": "continuous_list", "names": names, "upper": upper, "lower": lower, "default": default, "other_default": param_config.get("other_default", 0.0), "count": len(names)}

            elif param_type == "continuous value":
                upper = float(param_config["upper"])
                lower = float(param_config["lower"])
                default = float(param_config["default"]) if not isinstance(param_config["default"], (list, tuple)) else float(param_config["default"][0])
                if param_name == "length":
                    lower = min(upper, max(lower, self.algorithm_min_length))
                    default = min(upper, max(default, lower))
                param_info[param_name] = {"type": "continuous_scalar", "names": param_config["names"], "upper": upper, "lower": lower, "default": default, "count": 1}

            elif param_type == "categorical list":
                names = param_config["names"]
                choices = param_config["choices"]
                default = param_config["default"]

                param_info[param_name] = {"type": "categorical_list", "names": names, "choices": choices, "default": default, "count": len(names)}

        return param_info

    def _create_bounds(self) -> torch.Tensor:
        """创建参数边界"""
        bounds = []

        for param_name, info in self.param_info.items():
            if info["type"] == "continuous_list":
                for i in range(info["count"]):
                    bounds.append([0.0, 1.0])  # 标准化到[0,1]
            elif info["type"] == "continuous_scalar":
                bounds.append([0.0, 1.0])  # 标准化到[0,1]
            elif info["type"] == "categorical_list":
                for i in range(info["count"]):
                    bounds.append([0.0, 1.0])  # 标准化到[0,1]，后续映射到离散选择

        return torch.tensor(bounds).T  # shape: (2, total_dims)

    def _setup_legacy_attributes(self):
        """设置兼容原始代码的属性"""
        # 为了兼容原始代码，从配置中提取基础参数
        nmos_width_config = self.param_info.get("nmos_width", {})
        pmos_width_config = self.param_info.get("pmos_width", {})
        length_config = self.param_info.get("length", {})

        # 设置默认值
        if nmos_width_config:
            nmos_defaults = nmos_width_config["default"]
            if isinstance(nmos_defaults, (list, tuple)):
                pd_default = nmos_defaults[0] if "pd" in nmos_width_config["names"] else 2.05e-7
                pg_default = nmos_defaults[1] if len(nmos_defaults) > 1 else 1.35e-7
            else:
                pd_default = pg_default = float(nmos_defaults)
        else:
            pd_default, pg_default = 2.05e-7, 1.35e-7

    def get_physical_bounds(self) -> List:
        nmos_width_config = self.param_info.get("nmos_width", {})
        pmos_width_config = self.param_info.get("pmos_width", {})
        length_config = self.param_info.get("length", {})
        nmos_model_config = self.param_info.get("nmos_model", {})
        pmos_model_config = self.param_info.get("pmos_model", {})

        def _range_from_config(cfg, index=0):
            if not cfg:
                return (0.0, 0.0)
            lower = cfg.get("lower", 0.0)
            upper = cfg.get("upper", 0.0)
            if isinstance(lower, (list, tuple)):
                lower = lower[index] if index < len(lower) else lower[0]
            if isinstance(upper, (list, tuple)):
                upper = upper[index] if index < len(upper) else upper[0]
            return (float(lower), float(upper))

        pu_range = _range_from_config(pmos_width_config, 0)
        pd_range = _range_from_config(nmos_width_config, 0)
        pg_range = _range_from_config(nmos_width_config, 1)
        length_range = _range_from_config(length_config, 0)

        nmos_choices = nmos_model_config.get("choices", ["NMOS_VTL", "NMOS_VTG", "NMOS_VTH"]) if nmos_model_config else ["NMOS_VTL", "NMOS_VTG", "NMOS_VTH"]
        pmos_choices = pmos_model_config.get("choices", ["PMOS_VTL", "PMOS_VTG", "PMOS_VTH"]) if pmos_model_config else ["PMOS_VTL", "PMOS_VTG", "PMOS_VTH"]

        return [pu_range, pd_range, pg_range, length_range, nmos_choices, pmos_choices]

        if pmos_width_config:
            pmos_defaults = pmos_width_config["default"]
            if isinstance(pmos_defaults, (list, tuple)):
                pu_default = pmos_defaults[0] if pmos_defaults else 9e-8
            else:
                pu_default = float(pmos_defaults)
        else:
            pu_default = 9e-8

        if length_config:
            length_default = length_config["default"]
            # 确保length_default是标量值，如果是列表则取第一个元素
            if isinstance(length_default, (list, tuple)):
                length_default = length_default[0]
        else:
            length_default = 50e-9

        self.base_pd_width = pd_default
        self.base_pu_width = pu_default
        self.base_pg_width = pg_default
        self.base_length_nm = float(length_default) * 1e9

    def convert_params(self, x):
        """
        Convert parameters based on configuration
        根据配置转换参数
        """
        # 确保x是1D张量
        if isinstance(x, torch.Tensor):
            if x.dim() > 1:
                x = x.flatten()
        else:
            x = torch.tensor(x, dtype=torch.float32)

        params = {}
        dim_idx = 0

        for param_name, info in self.param_info.items():
            if info["type"] == "continuous_list":
                # 连续型列表参数
                for i, name in enumerate(info["names"]):
                    # 从[0,1]映射到实际范围
                    val = info["lower"][i] + x[dim_idx] * (info["upper"][i] - info["lower"][i])

                    if param_name == "nmos_width":
                        if name == "pd":
                            params["pd_width"] = float(val)
                        elif name == "pg":
                            params["pg_width"] = float(val)
                    elif param_name == "pmos_width":
                        if name == "pu":
                            params["pu_width"] = float(val)

                    dim_idx += 1

            elif info["type"] == "continuous_scalar" or info["type"] == "continuous value":
                # 连续型标量参数
                upper = float(info["upper"])
                lower = float(info["lower"])
                val = lower + x[dim_idx] * (upper - lower)

                if param_name == "length":
                    params["length"] = float(val)
                    params["length_nm"] = float(val * 1e9)

                dim_idx += 1

            elif info["type"] == "categorical_list":
                # 分类型列表参数
                for i, name in enumerate(info["names"]):
                    val = float(x[dim_idx])
                    if val > 1.0:
                        choice_idx = int(val)
                    else:
                        choice_idx = int(val * len(info["choices"]))
                    if choice_idx < 0:
                        choice_idx = 0
                    if choice_idx >= len(info["choices"]):
                        choice_idx = len(info["choices"]) - 1

                    if param_name == "nmos_model":
                        params["nmos_model_name"] = info["choices"][choice_idx]
                    elif param_name == "pmos_model":
                        params["pmos_model_name"] = info["choices"][choice_idx]

                    dim_idx += 1

        return params

    def print_params(self, params):
        """Print parameter information"""
        print("Parameters / 参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")


# ============================================================================
# 外围电路参数定义（从YAML动态构建，自动过滤常量）
# Peripheral circuit parameter definitions (dynamically built from YAML,
# automatically filtering out constants where upper == lower)
# ============================================================================

# 硬编码的默认定义作为后备 / Hardcoded defaults as fallback
_MUX_PARAM_DEFS_FALLBACK = [
    {"key": "mux_p_width", "type": "continuous", "lower": 1.0e-7, "upper": 1.0e-6, "default": 0.135e-6},
    {"key": "mux_n_width", "type": "continuous", "lower": 1.0e-7, "upper": 1.0e-6, "default": 0.135e-6},
    {"key": "mux_length",  "type": "continuous", "lower": 5.0e-8, "upper": 1.5e-7, "default": 50e-9},
    {"key": "mux_p_model", "type": "categorical", "choices": ["PMOS_VTL", "PMOS_VTG", "PMOS_VTH"], "default": "PMOS_VTG"},
    {"key": "mux_n_model", "type": "categorical", "choices": ["NMOS_VTL", "NMOS_VTG", "NMOS_VTH"], "default": "NMOS_VTG"},
]
_SA_PARAM_DEFS_FALLBACK = [
    {"key": "sa_p_width", "type": "continuous", "lower": 9.0e-8, "upper": 1.0e-6, "default": 0.54e-6},
    {"key": "sa_n_width", "type": "continuous", "lower": 9.0e-8, "upper": 1.0e-6, "default": 0.27e-6},
    {"key": "sa_length",  "type": "continuous", "lower": 5.0e-8, "upper": 1.5e-7, "default": 50e-9},
    {"key": "sa_p_model", "type": "categorical", "choices": ["PMOS_VTL", "PMOS_VTG", "PMOS_VTH"], "default": "PMOS_VTG"},
    {"key": "sa_n_model", "type": "categorical", "choices": ["NMOS_VTL", "NMOS_VTG", "NMOS_VTH"], "default": "NMOS_VTG"},
]
_WLD_PARAM_DEFS_FALLBACK = [
    {"key": "wld_nand_p_width", "type": "continuous", "lower": 9.0e-8, "upper": 1.0e-6, "default": 0.27e-6},
    {"key": "wld_inv_p_width",  "type": "continuous", "lower": 9.0e-8, "upper": 1.0e-6, "default": 0.27e-6},
    {"key": "wld_nand_n_width", "type": "continuous", "lower": 9.0e-8, "upper": 1.0e-6, "default": 0.18e-6},
    {"key": "wld_inv_n_width",  "type": "continuous", "lower": 9.0e-8, "upper": 1.0e-6, "default": 0.09e-6},
    {"key": "wld_length",       "type": "continuous", "lower": 5.0e-8, "upper": 1.5e-7, "default": 50e-9},
    {"key": "wld_nand_p_model", "type": "categorical", "choices": ["PMOS_VTL", "PMOS_VTG", "PMOS_VTH"], "default": "PMOS_VTG"},
    {"key": "wld_inv_p_model",  "type": "categorical", "choices": ["PMOS_VTL", "PMOS_VTG", "PMOS_VTH"], "default": "PMOS_VTG"},
    {"key": "wld_nand_n_model", "type": "categorical", "choices": ["NMOS_VTL", "NMOS_VTG", "NMOS_VTH"], "default": "NMOS_VTG"},
    {"key": "wld_inv_n_model",  "type": "categorical", "choices": ["NMOS_VTL", "NMOS_VTG", "NMOS_VTH"], "default": "NMOS_VTG"},
]


def _build_scalar_def(param, key, ptype):
    """从 Parameter 对象构建单个 scalar 参数定义。"""
    if ptype == "continuous":
        return {"key": key, "type": "continuous",
                "lower": float(param.lower), "upper": float(param.upper),
                "default": float(param.value)}
    else:
        return {"key": key, "type": "categorical",
                "choices": list(param.choices), "default": str(param.value)}


def _build_list_def(param, index, key, ptype):
    """从 Parameter 对象的列表元素构建单个参数定义。"""
    if ptype == "continuous":
        return {"key": key, "type": "continuous",
                "lower": float(param.lower[index]), "upper": float(param.upper[index]),
                "default": float(param.value[index])}
    else:
        val = param.value[index] if isinstance(param.value, list) else param.value
        return {"key": key, "type": "categorical",
                "choices": list(param.choices), "default": str(val)}


def _is_constant_def(d):
    """判断参数定义是否为常量（上下限相同或仅1个选项）。"""
    if d["type"] == "continuous":
        return d["lower"] == d["upper"]
    elif d["type"] == "categorical":
        return len(d["choices"]) <= 1
    return False


def _build_peripheral_defs_from_yaml():
    """
    从YAML配置文件动态构建外围电路参数定义，自动过滤上下限相同的常量。
    Build peripheral param defs from YAML configs, filtering out constants
    where upper == lower (continuous) or single-choice (categorical).

    Returns:
        (variable_defs, constant_defs): 变量定义列表和常量定义列表
    """
    sram_config = _load_sram_config_from_yaml()
    all_defs = []

    # --- Column Mux (scalar params) ---
    mux = sram_config.column_mux
    for param_name, key, ptype in [
        ("pmos_width", "mux_p_width", "continuous"),
        ("nmos_width", "mux_n_width", "continuous"),
        ("length",     "mux_length",  "continuous"),
        ("pmos_model", "mux_p_model", "categorical"),
        ("nmos_model", "mux_n_model", "categorical"),
    ]:
        all_defs.append(_build_scalar_def(mux.get_parameter(param_name), key, ptype))

    # --- Sense Amp (scalar params) ---
    sa = sram_config.senseamp
    for param_name, key, ptype in [
        ("pmos_width", "sa_p_width", "continuous"),
        ("nmos_width", "sa_n_width", "continuous"),
        ("length",     "sa_length",  "continuous"),
        ("pmos_model", "sa_p_model", "categorical"),
        ("nmos_model", "sa_n_model", "categorical"),
    ]:
        all_defs.append(_build_scalar_def(sa.get_parameter(param_name), key, ptype))

    # --- Wordline Driver (list + scalar params) ---
    wld = sram_config.wordline_driver
    # pmos_width: list [nand_p, inv_p]
    pw = wld.get_parameter("pmos_width")
    for i, key in enumerate(["wld_nand_p_width", "wld_inv_p_width"]):
        all_defs.append(_build_list_def(pw, i, key, "continuous"))
    # nmos_width: list [nand_n, inv_n]
    nw = wld.get_parameter("nmos_width")
    for i, key in enumerate(["wld_nand_n_width", "wld_inv_n_width"]):
        all_defs.append(_build_list_def(nw, i, key, "continuous"))
    # length: scalar
    all_defs.append(_build_scalar_def(wld.get_parameter("length"), "wld_length", "continuous"))
    # pmos_model: list [nand_p, inv_p]
    pm = wld.get_parameter("pmos_model")
    for i, key in enumerate(["wld_nand_p_model", "wld_inv_p_model"]):
        all_defs.append(_build_list_def(pm, i, key, "categorical"))
    # nmos_model: list [nand_n, inv_n]
    nm = wld.get_parameter("nmos_model")
    for i, key in enumerate(["wld_nand_n_model", "wld_inv_n_model"]):
        all_defs.append(_build_list_def(nm, i, key, "categorical"))

    # --- Precharge (scalar params, PMOS only) ---
    prc = sram_config.precharge
    for param_name, key, ptype in [
        ("pmos_width", "prc_p_width", "continuous"),
        ("length",     "prc_length",  "continuous"),
        ("pmos_model", "prc_p_model", "categorical"),
    ]:
        all_defs.append(_build_scalar_def(prc.get_parameter(param_name), key, ptype))

    # 分离变量和常量 / Separate variables from constants
    variable_defs = [d for d in all_defs if not _is_constant_def(d)]
    constant_defs = [d for d in all_defs if _is_constant_def(d)]

    if constant_defs:
        const_keys = [d["key"] for d in constant_defs]
        print(f"[PeripheralDefs] 以下参数上下限相同，视为常量，不纳入优化: {const_keys}")

    return variable_defs, constant_defs


# 尝试从YAML动态构建；失败则使用硬编码后备
# Try to build from YAML; fall back to hardcoded defaults on failure
try:
    PERIPHERAL_PARAM_DEFS, PERIPHERAL_CONSTANT_DEFS = _build_peripheral_defs_from_yaml()
except Exception as _e:
    print(f"[PeripheralDefs] 从YAML构建失败 ({_e})，使用硬编码默认定义")
    PERIPHERAL_PARAM_DEFS = _MUX_PARAM_DEFS_FALLBACK + _SA_PARAM_DEFS_FALLBACK + _WLD_PARAM_DEFS_FALLBACK
    PERIPHERAL_CONSTANT_DEFS = []

# ---- WLD 参数已加入优化变量（不再固定为常量）----
# WLD parameters are now included in the optimization search space.

# ---- 降维：将 Column MUX 参数从优化变量移入常量（固定为 YAML 默认值）----
_MUX_PREFIX = "mux_"
_mux_vars = [d for d in PERIPHERAL_PARAM_DEFS if d["key"].startswith(_MUX_PREFIX)]
PERIPHERAL_PARAM_DEFS = [d for d in PERIPHERAL_PARAM_DEFS if not d["key"].startswith(_MUX_PREFIX)]
PERIPHERAL_CONSTANT_DEFS = PERIPHERAL_CONSTANT_DEFS + _mux_vars
if _mux_vars:
    print(f"[PeripheralDefs] MUX 降维: 将 {[d['key'] for d in _mux_vars]} 固定为默认值，不参与优化")

# 外围电路参数 flat key 列表（仅变量，用于优化维度）
PERIPHERAL_CONTINUOUS_KEYS = [d["key"] for d in PERIPHERAL_PARAM_DEFS if d["type"] == "continuous"]
PERIPHERAL_CATEGORICAL_KEYS = [d["key"] for d in PERIPHERAL_PARAM_DEFS if d["type"] == "categorical"]
# ALL_KEYS 包含变量+常量，用于 CSV 记录（确保 CSV 列完整）
PERIPHERAL_ALL_KEYS = [d["key"] for d in PERIPHERAL_PARAM_DEFS] + [d["key"] for d in PERIPHERAL_CONSTANT_DEFS]
# 常量 key 列表（仅用于注入固定值）
PERIPHERAL_CONSTANT_KEYS = [d["key"] for d in PERIPHERAL_CONSTANT_DEFS]


class CompositeSRAMParameterSpace:
    """
    Composite parameter space that includes bitcell + peripheral circuits.
    复合参数空间，包含 bitcell + 外围电路（column mux, sense amp, wordline driver）。

    Normalizes all parameters to [0,1] range. The first N dimensions are bitcell
    parameters (same as ModifiedSRAMParameterSpace), followed by peripheral dims.
    """

    def __init__(self, config_path: str = "config_sram.yaml"):
        # Bitcell base space
        self.bitcell_space = ModifiedSRAMParameterSpace(config_path)
        self.bitcell_dim = self.bitcell_space.bounds.shape[1]

        # Build peripheral parameter info (仅变量维度参与优化)
        self.peripheral_defs = PERIPHERAL_PARAM_DEFS          # 变量参数
        self.peripheral_constant_defs = PERIPHERAL_CONSTANT_DEFS  # 常量参数
        self.peripheral_dim = len(self.peripheral_defs)

        # Total dimension (仅变量参数参与优化)
        self.total_dim = self.bitcell_dim + self.peripheral_dim

        # Build combined bounds: all [0, 1]
        lower = torch.zeros(self.total_dim)
        upper = torch.ones(self.total_dim)
        self.bounds = torch.stack([lower, upper])  # shape: (2, total_dim)

        # Expose param_info for compatibility with get_default_normalized_vector
        self.param_info = self.bitcell_space.param_info

        # Legacy attributes
        self._setup_legacy_attributes()

        if self.peripheral_constant_defs:
            print(f"[CompositeSRAMParameterSpace] 优化维度={self.total_dim} "
                  f"(bitcell={self.bitcell_dim}, peripheral变量={self.peripheral_dim}, "
                  f"peripheral常量={len(self.peripheral_constant_defs)})")

    def _setup_legacy_attributes(self):
        """Forward legacy attributes from the bitcell space."""
        for attr in ("base_pd_width", "base_pu_width", "base_pg_width", "base_length_nm"):
            if hasattr(self.bitcell_space, attr):
                setattr(self, attr, getattr(self.bitcell_space, attr))

    def convert_params(self, x):
        """
        Convert normalized [0,1] vector to actual parameter dict.
        前 bitcell_dim 维 -> bitcell 参数，后 peripheral_dim 维 -> 外围电路变量参数。
        常量参数自动以固定值注入（不占优化维度）。
        """
        if isinstance(x, torch.Tensor):
            if x.dim() > 1:
                x = x.flatten()
        else:
            x = torch.tensor(x, dtype=torch.float32)

        # --- Bitcell params ---
        bc_x = x[:self.bitcell_dim]
        params = self.bitcell_space.convert_params(bc_x)

        # --- Peripheral variable params (仅变量参数从优化向量取值) ---
        periph_x = x[self.bitcell_dim:]
        for i, pdef in enumerate(self.peripheral_defs):
            if i < len(periph_x):
                val = float(periph_x[i])
            else:
                val = 0.5  # fallback to midpoint

            if pdef["type"] == "continuous":
                actual = pdef["lower"] + val * (pdef["upper"] - pdef["lower"])
                params[pdef["key"]] = float(actual)
            elif pdef["type"] == "categorical":
                choices = pdef["choices"]
                if val > 1.0:
                    idx = int(val)
                else:
                    idx = int(val * len(choices))
                idx = max(0, min(idx, len(choices) - 1))
                params[pdef["key"]] = choices[idx]

        # --- Peripheral constant params (常量参数直接注入默认值) ---
        for cdef in self.peripheral_constant_defs:
            params[cdef["key"]] = cdef["default"]

        return params

    def get_default_normalized_vector(self):
        """
        Return default normalized vector for the composite space.
        返回复合空间的默认归一化向量。
        """
        # Bitcell defaults
        bc_defaults = get_default_normalized_vector(self.bitcell_space)

        # Peripheral defaults
        periph_defaults = []
        for pdef in self.peripheral_defs:
            if pdef["type"] == "continuous":
                rng = pdef["upper"] - pdef["lower"]
                if rng > 0:
                    periph_defaults.append((pdef["default"] - pdef["lower"]) / rng)
                else:
                    periph_defaults.append(0.5)
            elif pdef["type"] == "categorical":
                choices = pdef["choices"]
                if pdef["default"] in choices:
                    idx = choices.index(pdef["default"])
                else:
                    idx = 0
                if len(choices) > 1:
                    periph_defaults.append((idx + 0.01) / float(len(choices)))
                else:
                    periph_defaults.append(0.0)

        return list(bc_defaults) + periph_defaults

    def get_physical_bounds(self):
        """Forward to bitcell space for legacy compatibility."""
        return self.bitcell_space.get_physical_bounds()

    def print_params(self, params):
        """Print all parameters."""
        print("Parameters / 参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")


def apply_params_to_sram_config(sram_config, params):
    """
    Apply both bitcell and peripheral parameters to a loaded SRAM_CONFIG instance.
    将 bitcell 和外围电路参数写入已加载的 SRAM_CONFIG 实例。

    This function is the single point of truth for writing optimized parameters
    back into the SRAM configuration before simulation.
    """
    # --- Bitcell (sram_6t_cell) ---
    if "pu_width" in params:
        sram_config.sram_6t_cell.pmos_width.value = float(params["pu_width"])
    if "pd_width" in params:
        sram_config.sram_6t_cell.nmos_width.value[0] = float(params["pd_width"])
    if "pg_width" in params:
        sram_config.sram_6t_cell.nmos_width.value[1] = float(params["pg_width"])
    if "length" in params:
        sram_config.sram_6t_cell.length.value = float(params["length"])
    if params.get("pmos_model_name"):
        pmos_model = params["pmos_model_name"]
        if pmos_model in sram_config.sram_6t_cell.pmos_model.choices:
            sram_config.sram_6t_cell.pmos_model.value = pmos_model
    if params.get("nmos_model_name"):
        nmos_model = params["nmos_model_name"]
        if nmos_model in sram_config.sram_6t_cell.nmos_model.choices:
            sram_config.sram_6t_cell.nmos_model.value[0] = nmos_model
            sram_config.sram_6t_cell.nmos_model.value[1] = nmos_model

    # --- Column Mux ---
    if "mux_p_width" in params:
        sram_config.column_mux.pmos_width.value = float(params["mux_p_width"])
    if "mux_n_width" in params:
        sram_config.column_mux.nmos_width.value = float(params["mux_n_width"])
    if "mux_length" in params:
        sram_config.column_mux.length.value = float(params["mux_length"])
    if params.get("mux_p_model"):
        if params["mux_p_model"] in sram_config.column_mux.pmos_model.choices:
            sram_config.column_mux.pmos_model.value = params["mux_p_model"]
    if params.get("mux_n_model"):
        if params["mux_n_model"] in sram_config.column_mux.nmos_model.choices:
            sram_config.column_mux.nmos_model.value = params["mux_n_model"]

    # --- Sense Amplifier ---
    if "sa_p_width" in params:
        sram_config.senseamp.pmos_width.value = float(params["sa_p_width"])
    if "sa_n_width" in params:
        sram_config.senseamp.nmos_width.value = float(params["sa_n_width"])
    if "sa_length" in params:
        sram_config.senseamp.length.value = float(params["sa_length"])
    if params.get("sa_p_model"):
        if params["sa_p_model"] in sram_config.senseamp.pmos_model.choices:
            sram_config.senseamp.pmos_model.value = params["sa_p_model"]
    if params.get("sa_n_model"):
        if params["sa_n_model"] in sram_config.senseamp.nmos_model.choices:
            sram_config.senseamp.nmos_model.value = params["sa_n_model"]

    # --- Wordline Driver ---
    if "wld_nand_p_width" in params:
        wld_pvals = sram_config.wordline_driver.pmos_width.value
        if isinstance(wld_pvals, list):
            wld_pvals[0] = float(params["wld_nand_p_width"])
        else:
            sram_config.wordline_driver.pmos_width.value = float(params["wld_nand_p_width"])
    if "wld_inv_p_width" in params:
        wld_pvals = sram_config.wordline_driver.pmos_width.value
        if isinstance(wld_pvals, list) and len(wld_pvals) > 1:
            wld_pvals[1] = float(params["wld_inv_p_width"])
    if "wld_nand_n_width" in params:
        wld_nvals = sram_config.wordline_driver.nmos_width.value
        if isinstance(wld_nvals, list):
            wld_nvals[0] = float(params["wld_nand_n_width"])
        else:
            sram_config.wordline_driver.nmos_width.value = float(params["wld_nand_n_width"])
    if "wld_inv_n_width" in params:
        wld_nvals = sram_config.wordline_driver.nmos_width.value
        if isinstance(wld_nvals, list) and len(wld_nvals) > 1:
            wld_nvals[1] = float(params["wld_inv_n_width"])
    if "wld_length" in params:
        sram_config.wordline_driver.length.value = float(params["wld_length"])
    if params.get("wld_nand_p_model"):
        wld_pm = sram_config.wordline_driver.pmos_model.value
        if isinstance(wld_pm, list):
            if params["wld_nand_p_model"] in sram_config.wordline_driver.pmos_model.choices:
                wld_pm[0] = params["wld_nand_p_model"]
    if params.get("wld_inv_p_model"):
        wld_pm = sram_config.wordline_driver.pmos_model.value
        if isinstance(wld_pm, list) and len(wld_pm) > 1:
            if params["wld_inv_p_model"] in sram_config.wordline_driver.pmos_model.choices:
                wld_pm[1] = params["wld_inv_p_model"]
    if params.get("wld_nand_n_model"):
        wld_nm = sram_config.wordline_driver.nmos_model.value
        if isinstance(wld_nm, list):
            if params["wld_nand_n_model"] in sram_config.wordline_driver.nmos_model.choices:
                wld_nm[0] = params["wld_nand_n_model"]
    if params.get("wld_inv_n_model"):
        wld_nm = sram_config.wordline_driver.nmos_model.value
        if isinstance(wld_nm, list) and len(wld_nm) > 1:
            if params["wld_inv_n_model"] in sram_config.wordline_driver.nmos_model.choices:
                wld_nm[1] = params["wld_inv_n_model"]

    # --- Precharge ---
    if "prc_p_width" in params:
        sram_config.precharge.pmos_width.value = float(params["prc_p_width"])
    if "prc_length" in params:
        sram_config.precharge.length.value = float(params["prc_length"])
    if params.get("prc_p_model"):
        if params["prc_p_model"] in sram_config.precharge.pmos_model.choices:
            sram_config.precharge.pmos_model.value = params["prc_p_model"]


def collect_peripheral_param_columns(params):
    """
    Extract peripheral parameter key-value pairs for CSV output.
    提取外围电路参数的 key-value 对用于 CSV 输出。
    Returns (fieldnames, row_dict).
    """
    fieldnames = list(PERIPHERAL_ALL_KEYS)
    row_dict = {k: params.get(k, "") for k in PERIPHERAL_ALL_KEYS}
    return fieldnames, row_dict


def evaluate_sram(params, timeout=120):
    """
    Execute SRAM evaluation with given parameters
    使用给定参数执行SRAM评估
    """
    # 获取项目根目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    sram_config = SRAM_CONFIG()
    sram_config.load_all_configs(global_file=os.path.join(project_root, "sram_compiler/config_yaml/global.yaml"), circuit_configs={"SRAM_6T_CELL": os.path.join(project_root, "sram_compiler/config_yaml/sram_6t_cell.yaml"), "WORDLINEDRIVER": os.path.join(project_root, "sram_compiler/config_yaml/wordline_driver.yaml"), "PRECHARGE": os.path.join(project_root, "sram_compiler/config_yaml/precharge.yaml"), "COLUMNMUX": os.path.join(project_root, "sram_compiler/config_yaml/mux.yaml"), "SENSEAMP": os.path.join(project_root, "sram_compiler/config_yaml/sa.yaml"), "WRITEDRIVER": os.path.join(project_root, "sram_compiler/config_yaml/write_driver.yaml"), "DECODER": os.path.join(project_root, "sram_compiler/config_yaml/decoder.yaml")})

    # Apply peripheral params if present
    apply_params_to_sram_config(sram_config, params)

    try:
        print(f"Starting SRAM evaluation with parameters: {params}")
        print(f"开始使用参数进行SRAM评估: {params}")
        start_time = time.time()

        # Set simulation parameters
        # 设置仿真参数
        vdd = 1.0
        pdk_path = os.path.join(project_root, "tran_models/models_TT.spice")
        num_rows = 32
        num_cols = 1
        num_mc = 1

        area = estimate_scaled_total_area(params, num_rows=num_rows, num_cols=num_cols, num_arrays=1)
        print(f"Estimated 6T SRAM cell area: {area*1e12:.2f} µm²")
        print(f"估算的6T SRAM单元面积: {area*1e12:.2f} µm²")

        # 优化场景下直接使用固定参数值（不用 sweep 模式，否则需要 param_sweep_models.data 文件）
        mc_testbench = Sram6TCoreMcTestbench(
            sram_config,
            sram_cell_type="SRAM_6T_CELL",
            w_rc=True,  # Whether add RC to nets
            pi_res=100 @ u_Ohm,
            pi_cap=0.001 @ u_pF,
            vth_std=0.05,  # Process parameter variation is a percentage of its value in model lib
            custom_mc=False,  # Use your own process params?
            sweep_cell=False,        # 优化时参数由 sram_config 直接设置，无需 sweep
            sweep_precharge=False,
            sweep_senseamp=False,
            sweep_wordlinedriver=False,
            sweep_columnmux=False,
            sweep_writedriver=False,
            sweep_decoder=False,
            corner="TT",  # or FF or SS or FS or SF
            q_init_val=0,
            sim_path="sim",
        )

        print("Starting SRAM Monte Carlo simulation...")
        print("开始SRAM蒙特卡洛仿真...")

        try:
            # Run simulation directly
            # 直接运行仿真
            hold_snm = mc_testbench.run_mc_simulation(
                operation="hold_snm",
                target_row=num_rows - 1,
                target_col=num_cols - 1,
                mc_runs=num_mc,
                vars=None,
            )

            read_snm = mc_testbench.run_mc_simulation(
                operation="read_snm",
                target_row=num_rows - 1,
                target_col=num_cols - 1,
                mc_runs=num_mc,
                vars=None,
            )

            write_snm = mc_testbench.run_mc_simulation(
                operation="write_snm",
                target_row=num_rows - 1,
                target_col=num_cols - 1,
                mc_runs=num_mc,
                vars=None,
            )

            w_delay, w_pavg, w_pstc, w_pdyn = mc_testbench.run_mc_simulation(
                operation="write",
                target_row=num_rows - 1,
                target_col=num_cols - 1,
                mc_runs=num_mc,
                vars=None,
            )

            r_delay, r_pavg, r_pstc, r_pdyn = mc_testbench.run_mc_simulation(
                operation="read",
                target_row=num_rows - 1,
                target_col=num_cols - 1,
                mc_runs=num_mc,
                vars=None,
            )

            # Process simulation results
            # 处理仿真结果
            hold_snm_val = float(hold_snm[0]) if isinstance(hold_snm, np.ndarray) else float(hold_snm)
            read_snm_val = float(read_snm[0]) if isinstance(read_snm, np.ndarray) else float(read_snm)
            write_snm_val = float(write_snm[0]) if isinstance(write_snm, np.ndarray) else float(write_snm)
            min_snm = min(hold_snm_val, read_snm_val, write_snm_val)

            read_power = float(r_pavg[0]) if isinstance(r_pavg, np.ndarray) else float(r_pavg)
            write_power = float(w_pavg[0]) if isinstance(w_pavg, np.ndarray) else float(w_pavg)

            # Take absolute value of power
            # 取功耗的绝对值
            read_power = abs(read_power)
            write_power = abs(write_power)
            max_power = max(read_power, write_power)

            read_delay = float(r_delay[0]) if isinstance(r_delay, np.ndarray) else float(r_delay)
            write_delay = float(w_delay[0]) if isinstance(w_delay, np.ndarray) else float(w_delay)

            read_delay_feasible = True
            write_delay_feasible = True
            max_delay = max(read_delay, write_delay)
            if min_snm > 0 and max_power > 0 and area > 0 and max_delay > 0:
                merit = np.log10(min_snm / (max_power * np.sqrt(area) * max_delay))
            else:
                merit = -10.0  # Penalty value for invalid cases

            # Construct objectives (for multi-objective optimization)
            # 构建目标函数（用于多目标优化）
            # 4目标，全部最小化: [-SNM, power, area, delay]
            objectives = [-min_snm, max_power, area, max_delay]

            constraints = [0.0, 0.0]

            # Construct detailed results
            # 构建详细结果
            result = {"hold_snm": hold_snm_val, "read_snm": read_snm_val, "write_snm": write_snm_val, "min_snm": min_snm, "read_power": read_power, "write_power": write_power, "max_power": max_power, "read_delay": read_delay, "write_delay": write_delay, "max_delay": max_delay, "area": area, "merit": merit, "read_delay_feasible": read_delay_feasible, "write_delay_feasible": write_delay_feasible}

            end_time = time.time()
            print(f"Simulation completed successfully! Time taken: {end_time - start_time:.2f} seconds")
            print(f"仿真成功完成！用时: {end_time - start_time:.2f} 秒")
            print(f"SNM = {min_snm:.6f}, Power = {max_power:.6e}, Area = {area*1e12:.2f} µm², Merit = {merit:.6e}")

            return objectives, constraints, result, True

        except Exception as e:
            print(f"Error occurred during simulation: {str(e)}")
            print(f"仿真过程中发生错误: {str(e)}")
            traceback.print_exc()
            # 4目标惩罚值 (all minimized): [-SNM=0, power=10W, area=1e-3m², delay=10µs]
            objectives = [0.0, 10.0, 5e-6, 5e-8]
            constraints = [1.0, 1.0]  # Constraint violation
            return objectives, constraints, None, False

    except Exception as e:
        print(f"Error occurred during overall evaluation: {str(e)}")
        print(f"整体评估过程中发生错误: {str(e)}")
        traceback.print_exc()
        # 4目标惩罚值 (all minimized): [-SNM=0, power=10W, area=1e-3m², delay=10µs]
        objectives = [0.0, 10.0, 5e-6, 5e-8]
        constraints = [1.0, 1.0]  # Constraint violation
        return objectives, constraints, None, False


class BaseOptimizer:
    """
    Base optimizer class
    基础优化器类
    """

    def __init__(self, parameter_space, algorithm_name, num_objectives=4, num_constraints=2, initial_result=None, initial_params=None):
        """
        Initialize base optimizer
        初始化基础优化器
        """
        self.parameter_space = parameter_space
        self.algorithm_name = algorithm_name
        self.bounds = parameter_space.bounds
        self.dim = self.bounds.shape[1]
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints

        # Record all evaluation results
        # 记录所有评估结果
        self.all_x = []
        self.all_objectives = []
        self.all_constraints = []
        self.all_results = []
        self.all_success = []
        self.pareto_front = []

        # Initialize Merit tracking
        # 初始化Merit跟踪
        self.all_merit = []
        self.best_merit = float("-inf")
        self.best_params = None
        self.best_result = None

        # Set initial feasible point Merit and results
        # 设置初始可行点Merit和结果
        if initial_result:
            initial_min_snm = min(initial_result["hold_snm"]["snm"], initial_result["read_snm"]["snm"], initial_result["write_snm"]["snm"])
            initial_max_power = max(initial_result["read"]["power"], initial_result["write"]["power"])
            initial_max_delay = max(initial_result["read"]["delay"], initial_result["write"]["delay"])

            if initial_params:
                initial_area = estimate_scaled_total_area(initial_params)

                self.initial_merit = np.log10(initial_min_snm / (initial_max_power * np.sqrt(initial_area) * initial_max_delay))
                self.best_merit = self.initial_merit
                print(f"Initial Merit: {self.initial_merit:.6e}")

        # Initialize history data
        # 初始化历史数据
        self.iteration_history = []
        self.best_history = []

    def observe(self, x, objectives, constraints, result, success, iteration, source):
        """
        Record observation results
        记录观测结果
        """
        self.all_x.append(x)
        self.all_objectives.append(objectives)
        self.all_constraints.append(constraints)
        self.all_results.append(result)
        self.all_success.append(success)

        if success and result:
            merit = result["merit"]
            self.all_merit.append(merit)

            # Update best result
            # 更新最佳结果
            if merit > self.best_merit:
                self.best_merit = merit
                self.best_params = self.parameter_space.convert_params(torch.tensor(x, dtype=torch.float32))
                self.best_result = result

            # Record iteration data
            # 记录迭代数据
            self.iteration_history.append({"iteration": iteration, "merit": merit, "objectives": objectives, "constraints": constraints, "success": success, "source": source})

            # Record best Merit history
            # 记录最佳Merit历史
            self.best_history.append(self.best_merit)


class OptimizationLogger:
    """
    Optimization logger class
    优化日志记录器类
    """

    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.log_data = []

    def log_iteration(self, iteration, merit, objectives, constraints, success):
        """
        Log iteration data
        记录迭代数据
        """
        self.log_data.append({"iteration": iteration, "merit": merit, "objectives": objectives, "constraints": constraints, "success": success})

    def save_log(self, filename):
        """
        Save log to file
        保存日志到文件
        """
        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["iteration", "merit", "objectives", "constraints", "success"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.log_data:
                writer.writerow(row)


def save_pareto_front(pareto_front, filename):
    """
    Save Pareto front to file
    保存Pareto前沿到文件
    """
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["min_snm", "max_power", "area", "merit"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for point in pareto_front:
            writer.writerow(point)


def save_best_result(best_result, algorithm_name, filename):
    """
    Save best result to file
    保存最佳结果到文件
    """
    result_data = {"algorithm": algorithm_name, "best_merit": best_result["merit"] if best_result else None, "best_params": best_result["params"] if best_result else None, "best_result": best_result["result"] if best_result else None}

    with open(filename, "w") as jsonfile:
        json.dump(result_data, jsonfile, indent=2, default=str)


def plot_merit_history(merit_history, algorithm_name, filename):
    """
    Plot Merit function history
    绘制Merit函数历史
    """
    plt.figure(figsize=(10, 6))
    plt.plot(merit_history, "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Merit")
    plt.title(f"{algorithm_name} Optimization: Merit vs Iteration")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pareto_frontier(pareto_front, algorithm_name, filename):
    """
    Plot Pareto frontier
    绘制Pareto前沿
    """
    if len(pareto_front) == 0:
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    snm_vals = [p["min_snm"] for p in pareto_front]
    power_vals = [p["max_power"] for p in pareto_front]
    area_vals = [p["area"] for p in pareto_front]

    ax.scatter(snm_vals, power_vals, area_vals, c="red", s=50)
    ax.set_xlabel("Min SNM (V)")
    ax.set_ylabel("Max Power (W)")
    ax.set_zlabel("Area (m²)")
    ax.set_title(f"{algorithm_name} Pareto Frontier")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def update_pareto_front(pareto_front, objectives, result):
    """
    Update Pareto front
    更新Pareto前沿
    """
    new_point = {"min_snm": objectives[0], "max_power": -objectives[1], "area": -objectives[2], "merit": result["merit"]}  # Convert back from negative  # Convert back from negative

    # Simple Pareto front update (can be optimized)
    # 简单的Pareto前沿更新（可以优化）
    pareto_front.append(new_point)
    return pareto_front


def save_optimization_history(history, algorithm_name, filename):
    """
    Save optimization history
    保存优化历史
    """
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["iteration", "merit", "min_snm", "max_power", "area", "success"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for entry in history:
            row = {"iteration": entry["iteration"], "merit": entry["merit"], "min_snm": entry["objectives"][0] if entry["success"] else None, "max_power": -entry["objectives"][1] if entry["success"] else None, "area": -entry["objectives"][2] if entry["success"] else None, "success": entry["success"]}
            writer.writerow(row)
