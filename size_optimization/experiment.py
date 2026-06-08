"""
Two-Stage SRAM Optimization Program
二阶段SRAM优化程序

This program implements a two-stage optimization approach for 32KB (262144-bit) SRAM arrays:
该程序实现了32KB（262144 位）SRAM阵列的二阶段优化方法：

Stage 1: Architecture Configuration Optimization (SA or SMAC)
第一阶段：架构配置优化（SA 或 SMAC）
- Find optimal row/column counts and number of arrays
- Constraints: rows ≤ 512, columns ≤ 512, total capacity = 262144 bits (32KB)

Stage 2: Transistor Parameter Optimization
第二阶段：晶体管参数优化
- Optimize transistor parameters for the best architecture from Stage 1
- Supported algorithms: SA, PSO, SMAC, CBO, RoSE_Opt, CMA-ES, MOEAD, MOBO, NSGA-II, tSS-BO, CPN

Joint Optimization Mode: optimize architecture + transistor sizing simultaneously
联合优化模式：同时优化架构与晶体管尺寸
"""

import os
import sys
import time
import math
import numpy as np
import torch
import random
import warnings
import csv
from pathlib import Path
import traceback
from datetime import datetime
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROW_CHOICES = [16, 32, 64, 128, 256, 512]
COLUMN_CHOICES = [16, 32, 64, 128, 256, 512]
TOTAL_BITS = 262144
output_cols = 64

# Import SMAC for Stage 1 optimization
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace
import ConfigSpace.hyperparameters as CSH

SMAC_AVAILABLE = True

# Import path handling
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import utilities
from size_optimization.exp_utils import (
    seed_set, create_directories, ModifiedSRAMParameterSpace, get_params_from_yaml,
    CompositeSRAMParameterSpace, apply_params_to_sram_config, collect_peripheral_param_columns,
    PERIPHERAL_ALL_KEYS, get_composite_initial_params,
)
from utils import estimate_total_area, estimate_bitcell_area, estimate_array_area, estimate_scaled_array_area

# Import SRAM simulation modules
# 导入SRAM仿真模块
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from config import SRAM_CONFIG


class ArchitectureConfigurationSpace:
    """
    Architecture configuration space for 32KB (262144-bit) SRAM
    32KB（262144 位）SRAM的架构配置空间
    """

    def __init__(self):
        self.total_bits = TOTAL_BITS
        self.row_choices = ROW_CHOICES
        self.column_choices = COLUMN_CHOICES
        self.max_rows = max(self.row_choices)
        self.max_cols = max(self.column_choices)
        self.min_rows = min(self.row_choices)
        self.min_cols = min(self.column_choices)
        self.valid_configs = self._generate_valid_configurations()

    def _generate_valid_configurations(self) -> List[Dict]:
        """
        Generate all valid architecture configurations
        生成所有有效的架构配置
        行列最小值不能小于2
        """
        valid_configs = []

        for rows in self.row_choices:
            for cols in self.column_choices:
                array_capacity = rows * cols

                # Check if total capacity can be achieved with integer number of arrays
                # 检查是否可以用整数个阵列实现总容量
                if self.total_bits % array_capacity == 0:
                    num_arrays = self.total_bits // array_capacity

                    config = {"rows": rows, "cols": cols, "num_arrays": num_arrays, "array_capacity": array_capacity, "total_capacity": self.total_bits}
                    valid_configs.append(config)

        print(f"Generated {len(valid_configs)} valid configurations (rows in {self.row_choices}, cols in {self.column_choices})")
        print(f"生成了 {len(valid_configs)} 个有效配置 (行={self.row_choices}, 列={self.column_choices})")

        return valid_configs

    def get_configuration(self, index: int) -> Dict:
        """
        Get configuration by index
        根据索引获取配置
        """
        if 0 <= index < len(self.valid_configs):
            return self.valid_configs[index]
        else:
            raise IndexError(f"Configuration index {index} out of range [0, {len(self.valid_configs)-1}]")

    def get_num_configurations(self) -> int:
        """
        Get total number of valid configurations
        获取有效配置的总数
        """
        return len(self.valid_configs)


class JointParameterSpace:
    def __init__(self, arch_space: ArchitectureConfigurationSpace, base_space: ModifiedSRAMParameterSpace):
        self.arch_space = arch_space
        self.base_space = base_space
        self.row_choices = arch_space.row_choices
        self.column_choices = arch_space.column_choices
        prefix = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        self.bounds = torch.cat([prefix, base_space.bounds], dim=1)

    def convert_params(self, x):
        if isinstance(x, torch.Tensor):
            if x.dim() > 1:
                x = x.flatten()
        else:
            x = torch.tensor(x, dtype=torch.float32)
        row_idx = int(float(x[0]) * len(self.row_choices))
        col_idx = int(float(x[1]) * len(self.column_choices))
        row_idx = max(0, min(row_idx, len(self.row_choices) - 1))
        col_idx = max(0, min(col_idx, len(self.column_choices) - 1))
        rows = self.row_choices[row_idx]
        cols = self.column_choices[col_idx]
        params = self.base_space.convert_params(x[2:])
        params["rows"] = rows
        params["cols"] = cols
        return params

    @property
    def dim(self):
        return int(self.bounds.shape[1])

    def print_params(self, params):
        return self.base_space.print_params(params)


def evaluate_sram_with_config(params, num_rows, num_cols, num_arrays, timeout=120, *, stage_label="stage_eval", iteration_index=None, temperature=None, corner="TT", gen_unused_cells=True):
    """
    Execute SRAM evaluation with given parameters and specific row-column configuration
    使用给定参数和特定行列配置执行SRAM评估

    This function is copied from demo_joint_optimization.py to make two_stage_optimization.py
    independent of demo_joint_optimization.py
    此函数从demo_joint_optimization.py复制而来，使two_stage_optimization.py不依赖demo_joint_optimization.py
    """
    # 获取项目根目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录为 size_optimization/ 的父目录，动态计算无需硬编码
    project_root = os.path.dirname(current_dir)

    # 依据阶段标签确定仿真输出目录：sim/stage1 或 sim/stage2
    sim_subdir = "stage1" if str(stage_label).startswith("stage1") else "stage2" if str(stage_label).startswith("stage2") else str(stage_label)
    if str(stage_label).startswith("joint"):
        stage_parts = str(stage_label).split("/", 1)
        if len(stage_parts) > 1 and stage_parts[1]:
            sim_subdir = str(Path(sim_subdir) / stage_parts[1])
        sim_subdir = str(Path(sim_subdir) / f"r{num_rows}_c{num_cols}_a{num_arrays}")
    # 输出到绝对路径 /OpenYield/size_optimization/sim/<stage>
    sim_dir_path = Path(project_root) / "size_optimization" / "sim" / sim_subdir
    sim_dir_path.mkdir(parents=True, exist_ok=True)

    sram_config = SRAM_CONFIG()
    sram_config.load_all_configs(global_file=os.path.join(project_root, "sram_compiler/config_yaml/global.yaml"), circuit_configs={"SRAM_6T_CELL": os.path.join(project_root, "sram_compiler/config_yaml/sram_6t_cell.yaml"), "WORDLINEDRIVER": os.path.join(project_root, "sram_compiler/config_yaml/wordline_driver.yaml"), "PRECHARGE": os.path.join(project_root, "sram_compiler/config_yaml/precharge.yaml"), "COLUMNMUX": os.path.join(project_root, "sram_compiler/config_yaml/mux.yaml"), "SENSEAMP": os.path.join(project_root, "sram_compiler/config_yaml/sa.yaml"), "WRITEDRIVER": os.path.join(project_root, "sram_compiler/config_yaml/write_driver.yaml"), "DECODER": os.path.join(project_root, "sram_compiler/config_yaml/decoder.yaml")})

    # 修复：global.yaml 中 pdk_path_* 是相对路径，从 size_optimization/ 运行时会找不到文件。
    # 加载完 config 后立即将所有 pdk_path_* 转为绝对路径。
    for corner_suffix in ("TT", "FF", "SS", "FS", "SF"):
        attr = f"pdk_path_{corner_suffix}"
        rel = getattr(sram_config.global_config, attr, None)
        if rel and not os.path.isabs(rel):
            setattr(sram_config.global_config, attr, os.path.join(project_root, rel))

    try:
        print(f"Starting SRAM evaluation with {num_rows}x{num_cols} and fixed {output_cols}-bit output config")
        print(f"开始使用{num_rows}x{num_cols}, 固定{output_cols}位输出配置进行SRAM评估")

        def _write_and_load_params_csv():
            """将当前参数写入CSV并读取最新一行返回。"""
            # 将参数追踪CSV写入阶段化目录（与网表输出一致）
            sim_dir = sim_dir_path
            sim_dir.mkdir(parents=True, exist_ok=True)

            headers = [
                "timestamp",
                "stage",
                "iteration",
                "rows",
                "cols",
                "pu_width",
                "pd_width",
                "pg_width",
                "length",
                "pmos_model_name",
                "nmos_model_name",
            ] + list(PERIPHERAL_ALL_KEYS)
            timestamp = datetime.now().isoformat(timespec="seconds")
            row = [
                timestamp,
                stage_label,
                "" if iteration_index is None else iteration_index,
                num_rows,
                num_cols,
                params.get("pu_width", 0.0),
                params.get("pd_width", 0.0),
                params.get("pg_width", 0.0),
                params.get("length", 0.0),
                params.get("pmos_model_name", ""),
                params.get("nmos_model_name", ""),
            ] + [params.get(k, "") for k in PERIPHERAL_ALL_KEYS]

            history_path = sim_dir / "transistor_params_history.csv"
            write_header = not history_path.exists()
            with open(history_path, "a", newline="", encoding="utf-8") as f_hist:
                writer = csv.writer(f_hist)
                if write_header:
                    writer.writerow(headers)
                writer.writerow(row)

            latest_path = sim_dir / "transistor_params_latest.csv"
            with open(latest_path, "w", newline="", encoding="utf-8") as f_latest:
                writer = csv.writer(f_latest)
                writer.writerow(headers)
                writer.writerow(row)

            with open(latest_path, "r", newline="", encoding="utf-8") as f_latest:
                reader = csv.DictReader(f_latest)
                latest_row = next(reader)

            def _to_float(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return value

            result_dict = {
                "pu_width": _to_float(latest_row["pu_width"]),
                "pd_width": _to_float(latest_row["pd_width"]),
                "pg_width": _to_float(latest_row["pg_width"]),
                "length": _to_float(latest_row["length"]),
                "pmos_model_name": latest_row["pmos_model_name"],
                "nmos_model_name": latest_row["nmos_model_name"],
            }
            # Also read peripheral params from CSV if present
            for k in PERIPHERAL_ALL_KEYS:
                if k in latest_row and latest_row[k] != "":
                    result_dict[k] = _to_float(latest_row[k])
            return result_dict

        latest_params = _write_and_load_params_csv()
        params.update({k: v for k, v in latest_params.items() if v != ""})

        # CRITICAL: Update the SRAM configuration with the specified row/column counts
        # 关键：使用指定的行列数更新SRAM配置
        sram_config.global_config.num_rows = num_rows
        sram_config.global_config.num_cols = num_cols
        if temperature is None:
            temperature = getattr(sram_config.global_config, "temperature", 27)

        # CRITICAL: Update ALL transistor parameters (bitcell + peripheral) in sram_config
        # 关键：在创建testbench之前更新sram_config中所有晶体管参数（bitcell + 外围电路）
        apply_params_to_sram_config(sram_config, params)

        print(f"Updated SRAM config with parameters: pu={float(params['pu_width'])*1e9:.2f}nm, "
              f"pd={float(params['pd_width'])*1e9:.2f}nm, pg={float(params['pg_width'])*1e9:.2f}nm, "
              f"length={float(params['length'])*1e9:.2f}nm")
        print(f"已更新SRAM配置参数: pu={float(params['pu_width'])*1e9:.2f}nm, "
              f"pd={float(params['pd_width'])*1e9:.2f}nm, pg={float(params['pg_width'])*1e9:.2f}nm, "
              f"length={float(params['length'])*1e9:.2f}nm")
        # Print peripheral params if present
        if any(k in params for k in PERIPHERAL_ALL_KEYS):
            periph_info = ", ".join(f"{k}={params[k]}" for k in PERIPHERAL_ALL_KEYS if k in params)
            print(f"Peripheral params: {periph_info}")
            print(f"外围电路参数: {periph_info}")

        start_time = time.time()

        # Set simulation parameters
        # 设置仿真参数
        vdd = 1.0
        pdk_path = os.path.join(project_root, "tran_models/models_TT.spice")
        num_mc = 1

        # ── Pitch-scaling area model ──────────────────────────────────────
        # 列间距随 bitcell 宽度缩放，行间距随 bitcell 高度缩放
        # 外围电路开销随 SA/WLD/Precharge 晶体管宽度缩放
        sa_max_w  = max(params.get("sa_p_width", 0.54e-6),
                        params.get("sa_n_width", 0.27e-6))
        wld_max_w = max(params.get("wld_nand_p_width", 0.27e-6),
                        params.get("wld_inv_p_width",  0.27e-6),
                        params.get("wld_nand_n_width", 0.18e-6),
                        params.get("wld_inv_n_width",  0.09e-6))
        prc_max_w = params.get("prc_p_width", 0.27e-6)
        area = estimate_scaled_array_area(
            num_rows, num_cols, num_arrays,
            w_access=params.get("pg_width", 0.135e-6),
            w_pd=params.get("pd_width", 0.205e-6),
            w_pu=params.get("pu_width", 0.09e-6),
            l_transistor=params.get("length", 50e-9),
            sa_max_width=sa_max_w,
            wld_max_width=wld_max_w,
            prc_max_width=prc_max_w,
        )
        single_array_area = area / num_arrays

        mc_testbench = Sram6TCoreMcTestbench(
            sram_config,
            sram_cell_type="SRAM_6T_CELL",
            w_rc=True,
            pi_res=100 @ u_Ohm,
            pi_cap=0.001 @ u_pF,
            vth_std=0.05,
            custom_mc=False,
            sweep_cell=False,
            sweep_precharge=False,
            sweep_senseamp=False,
            sweep_wordlinedriver=False,
            sweep_columnmux=False,
            sweep_writedriver=False,
            sweep_decoder=False,
            corner=corner,
            choose_columnmux=False,  # 电路无MUX，多阵列延迟/功耗由惩罚项补偿
            use_equivalent=not gen_unused_cells,
            q_init_val=0,
            sim_path=str(sim_dir_path),
            enable_mc=False,
        )

        

        try:
            # Run simulation directly
            # 直接运行仿真
            if stage_label == "stage1":
                print("Running stage 1 simulation...")
            else:
                print("Running stage 2 simulation...")
                hold_snm = mc_testbench.run_mc_simulation(
                    operation="hold_snm",
                    target_row=num_rows - 1,
                    target_col=num_cols - 1,
                    mc_runs=num_mc,
                    temperature=temperature,
                    vars=None,
                )

                read_snm = mc_testbench.run_mc_simulation(
                    operation="read_snm",
                    target_row=num_rows - 1,
                    target_col=num_cols - 1,
                    mc_runs=num_mc,
                    temperature=temperature,
                    vars=None,
                )

                write_snm = mc_testbench.run_mc_simulation(
                    operation="write_snm",
                    target_row=num_rows - 1,
                    target_col=num_cols - 1,
                    mc_runs=num_mc,
                    temperature=temperature,
                    vars=None,
                )

            w_delay, w_pavg, w_pstc, w_pdyn = mc_testbench.run_mc_simulation(
                operation="write",
                target_row=num_rows - 1,
                target_col=num_cols - 1,
                mc_runs=num_mc,
                temperature=temperature,
                vars=None,
            )

            r_delay, r_pavg, r_pstc, r_pdyn = mc_testbench.run_mc_simulation(
                operation="read",
                target_row=num_rows - 1,
                target_col=num_cols - 1,
                mc_runs=num_mc,
                temperature=temperature,
                vars=None,
            )

            

            # Process simulation results
            # 处理仿真结果
            def get_float_value(sim_result):
                return float(sim_result[0]) if isinstance(sim_result, np.ndarray) else float(sim_result)

            if stage_label == "stage1":
                hold_snm_val = None
                read_snm_val = None
                write_snm_val = None
                min_snm = None
            else:
                hold_snm_val = get_float_value(hold_snm)
                read_snm_val = get_float_value(read_snm)
                write_snm_val = get_float_value(write_snm)
                min_snm = min(hold_snm_val, read_snm_val, write_snm_val)

            # 提取原始延迟标量
            raw_read_delay = float(r_delay[0]) if isinstance(r_delay, np.ndarray) else float(r_delay)
            raw_write_delay = float(w_delay[0]) if isinstance(w_delay, np.ndarray) else float(w_delay)

            # 计算片选信号延迟（以阵列数的对数上取整为解码级数）
            num_arrays_log2 = math.ceil(math.log2(num_arrays)) if num_arrays > 1 else 0
            cs_delay_adder = 4.167213500e-11 * num_arrays_log2
            mux_delay = 1.8e-10

            read_delay = raw_read_delay + cs_delay_adder + num_arrays_log2 * mux_delay + math.log2(max(1, num_cols // output_cols)) * mux_delay
            write_delay = raw_write_delay + cs_delay_adder

            if num_cols < output_cols:
                factor = output_cols // num_cols
                write_delay *= factor
                read_delay *= factor

            mux_power = 0.1e-6
            read_power = get_float_value(r_pstc) * num_arrays + get_float_value(r_pdyn)
            write_power = get_float_value(w_pstc) * num_arrays + get_float_value(w_pdyn)

            if num_arrays > 1:
                read_power += (num_arrays - 1) * num_arrays_log2 * mux_power + math.log2(max(1, num_cols // output_cols)) * mux_power

            if num_cols < output_cols:
                factor = output_cols // num_cols
                read_power += get_float_value(r_pdyn) * (factor - 1)
                write_power += get_float_value(w_pdyn) * (factor - 1)

            max_power = max(read_power, write_power)

            # If any delay equals -1, treat this evaluation as invalid and discard the solution
            # 如果任一延迟为 -1，则视为无效评估并舍弃该解
            if (read_delay == -1) or (write_delay == -1):
                print("Invalid delay detected (read/write = -1). Discarding this solution.")
                print("检测到无效延迟（读/写为 -1）。舍弃该解。")
                # 4目标惩罚值 (all minimized): [-SNM=0, power=10W, area=1e-3m², delay=10µs]
                objectives = [0.0, 10.0, 5e-6, 5e-8]
                constraints = [1.0, 1.0]  # Constraint violation
                return objectives, constraints, None, False

            read_delay_feasible = True
            write_delay_feasible = True

            # Calculate FoM (Figure of Merit)
            # 计算FoM（品质因数）
            # Stage 1: FoM = log10(1 / (max_power * sqrt(area) * max_delay))
            # 阶段1: FoM = log10(1 / (max_power * sqrt(area) * max_delay))
            # Stage 2: FoM = log10(min_snm / (max_power * max_delay))
            # 阶段2: FoM = log10(min_snm / (max_power * max_delay))
            max_delay = max(read_delay, write_delay)
            if stage_label == "stage1":
                if max_power > 0 and area > 0 and max_delay > 0:
                    fom = np.log10(1 / (max_power * np.sqrt(area) * max_delay))
                else:
                    fom = -10.0  # Penalty value for invalid cases

                # Construct objectives (for multi-objective optimization)
                # 构建目标函数（用于多目标优化）
                objectives = [-max_power, -area]  # Maximize SNM, minimize power and area

            else:
                # Stage 2 FoM no longer includes area; uses min_snm numerator
                # 阶段2 FoM 不再包含面积项；使用 min_snm 作为分子
                if min_snm > 0 and max_power > 0 and max_delay > 0:
                    fom = np.log10(min_snm / (max_power * max_delay))
                else:
                    fom = -10.0  # Penalty value for invalid cases

                # Construct objectives (for multi-objective optimization)
                # 构建目标函数（用于多目标优化）
                # Stage 2: 4目标全部最小化 [-SNM, power, area, delay]
                objectives = [-min_snm, max_power, area, max_delay]

            # Construct detailed results
            # 构建详细结果
            result = {"hold_snm": hold_snm_val, "read_snm": read_snm_val, "write_snm": write_snm_val, "min_snm": min_snm, "read_power": read_power, "write_power": write_power, "max_power": max_power, "read_delay": read_delay, "write_delay": write_delay, "max_delay": max_delay, "area": area, "single_array_area": single_array_area, "fom": fom, "read_delay_feasible": read_delay_feasible, "write_delay_feasible": write_delay_feasible}

            # 补充阵列规模信息，供外部保存
            result.update(
                {
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "capacity": num_rows * num_cols,
                    "aspect_ratio": (num_rows / num_cols) if num_cols else None,
                }
            )

            end_time = time.time()
            print(f"Simulation completed successfully! Time taken: {end_time - start_time:.2f} seconds")
            print(f"仿真成功完成！用时: {end_time - start_time:.2f} 秒")

            constraints = [0, 0]  # Constraint violation

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
        print(f"Error in testbench setup: {str(e)}")
        print(f"测试平台设置错误: {str(e)}")
        traceback.print_exc()
        # 4目标惩罚值 (all minimized): [-SNM=0, power=10W, area=1e-3m², delay=10µs]
        objectives = [0.0, 10.0, 5e-6, 5e-8]
        constraints = [1.0, 1.0]  # Constraint violation
        return objectives, constraints, None, False


class SAOptimizer:
    """
    Simulated Annealing Optimizer
    模拟退火优化器
    """

    def __init__(self, evaluation_function, bounds=None, maximize=True):
        """
        Initialize SA optimizer
        初始化SA优化器

        Args:
            evaluation_function: Function to evaluate solutions
            bounds: Parameter bounds (for continuous optimization)
            maximize: Whether to maximize (True) or minimize (False) the objective
        """
        self.evaluation_function = evaluation_function
        self.bounds = bounds
        self.maximize = maximize
        self.best_solution = None
        self.best_value = float("-inf") if maximize else float("inf")
        self.history = []

    def _accept_probability(self, old_value, new_value, temperature):
        """
        Calculate acceptance probability for SA
        计算SA的接受概率
        """
        if self.maximize:
            if new_value > old_value:
                return 1.0
            else:
                return np.exp((new_value - old_value) / temperature)
        else:
            if new_value < old_value:
                return 1.0
            else:
                return np.exp((old_value - new_value) / temperature)

    def optimize_discrete(self, num_solutions, max_iter=100, T_max=1000, T_min=1e-7):
        """
        Optimize discrete solution space (for Stage 1)
        优化离散解空间（用于第一阶段）
        """
        print(f"Starting SA optimization for discrete space with {num_solutions} solutions")
        print(f"开始对包含{num_solutions}个解的离散空间进行SA优化")

        # Initialize with random solution
        current_solution = random.randint(0, num_solutions - 1)
        current_value, current_result = self.evaluation_function(current_solution)

        # Track best solution
        self.best_solution = current_solution
        self.best_value = current_value
        self.best_result = current_result

        # Temperature schedule
        T = T_max
        alpha = (T_min / T_max) ** (1.0 / max_iter)

        print(f"Initial solution: {current_solution}, value: {current_value:.6f}")
        print(f"初始解: {current_solution}, 值: {current_value:.6f}")

        for iteration in range(max_iter):
            # Generate neighbor solution
            neighbor_solution = random.randint(0, num_solutions - 1)
            neighbor_value, neighbor_result = self.evaluation_function(neighbor_solution)

            # Accept or reject
            accept_prob = self._accept_probability(current_value, neighbor_value, T)
            random_val = random.random()
            accepted = random_val < accept_prob

            if accepted:
                current_solution = neighbor_solution
                current_value = neighbor_value
                current_result = neighbor_result

                # Update best solution
                if (self.maximize and current_value > self.best_value) or (not self.maximize and current_value < self.best_value):
                    self.best_solution = current_solution
                    self.best_value = current_value
                    self.best_result = current_result
                    print(f"Iteration {iteration}: New best solution {self.best_solution}, value: {self.best_value:.6f}")
                    print(f"迭代 {iteration}: 新的最佳解 {self.best_solution}, 值: {self.best_value:.6f}")

            # Record history
            self.history.append({"iteration": iteration, "current_solution": current_solution, "current_value": current_value, "best_value": self.best_value, "temperature": T, "accepted": accepted})

            # Cool down
            T *= alpha

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: T={T:.2e}, current_value={current_value:.6f}, best_value={self.best_value:.6f}")
                print(f"迭代 {iteration}: T={T:.2e}, 当前值={current_value:.6f}, 最佳值={self.best_value:.6f}")

        return self.best_solution, self.best_value, self.best_result

    def optimize_continuous(self, initial_solution, max_iter=100, T_max=1000, T_min=1e-7, arch_dims=0, arch_sigma=0.35):
        """
        Optimize continuous solution space (for Stage 2)
        优化连续解空间（用于第二阶段）
        arch_dims: 前 arch_dims 维为架构维度，使用更大的扰动步长 arch_sigma 以充分探索离散架构空间
        """
        print(f"Starting SA optimization for continuous space")
        print(f"开始连续空间的SA优化")

        # Initialize
        current_solution = np.array(initial_solution)
        current_value, current_result = self.evaluation_function(current_solution)

        # Track best solution
        self.best_solution = current_solution.copy()
        self.best_value = current_value
        self.best_result = current_result

        # Temperature schedule
        T = T_max
        alpha = (T_min / T_max) ** (1.0 / max_iter)

        print(f"Initial value: {current_value:.6f}")
        print(f"初始值: {current_value:.6f}")

        for iteration in range(max_iter):
            # Generate neighbor solution
            # 架构维度（前 arch_dims 维）使用更大扰动，确保能跨越离散槽边界（1/6≈0.167）
            perturbation = np.random.normal(0, 0.1, size=current_solution.shape)
            if arch_dims > 0:
                perturbation[:arch_dims] = np.random.normal(0, arch_sigma, size=arch_dims)
            neighbor_solution = current_solution + perturbation

            # Apply bounds if specified
            if self.bounds is not None:
                neighbor_solution = np.clip(neighbor_solution, self.bounds[0], self.bounds[1])

            neighbor_value, neighbor_result = self.evaluation_function(neighbor_solution)

            # Accept or reject
            accept_prob = self._accept_probability(current_value, neighbor_value, T)
            random_val = random.random()
            accepted = random_val < accept_prob

            if accepted:
                current_solution = neighbor_solution.copy()
                current_value = neighbor_value
                current_result = neighbor_result

                # Update best solution
                if (self.maximize and current_value > self.best_value) or (not self.maximize and current_value < self.best_value):
                    self.best_solution = current_solution.copy()
                    self.best_value = current_value
                    self.best_result = current_result
                    print(f"Iteration {iteration}: New best value: {self.best_value:.6f}")
                    print(f"迭代 {iteration}: 新的最佳值: {self.best_value:.6f}")

            # Record history
            self.history.append({"iteration": iteration, "current_value": current_value, "best_value": self.best_value, "temperature": T, "accepted": accepted})

            # Cool down
            T *= alpha

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: T={T:.2e}, current_value={current_value:.6f}, best_value={self.best_value:.6f}")
                print(f"迭代 {iteration}: T={T:.2e}, 当前值={current_value:.6f}, 最佳值={self.best_value:.6f}")

        return self.best_solution, self.best_value, self.best_result


class SMACStage1Optimizer:
    """
    SMAC optimizer for Stage 1 discrete optimization (architecture configuration)
    阶段1离散优化（架构配置）的SMAC优化器
    """

    def __init__(self, evaluation_function, num_configurations, maximize=True):
        """
        Initialize SMAC optimizer for discrete space
        初始化离散空间的SMAC优化器

        Args:
            evaluation_function: Function to evaluate configuration index -> (fom, result)
            num_configurations: Total number of valid configurations
            maximize: Whether to maximize (True) or minimize (False) the objective
        """
        if not SMAC_AVAILABLE:
            raise ImportError("SMAC is not available. Please install with: pip install smac ConfigSpace")

        self.evaluation_function = evaluation_function
        self.num_configurations = num_configurations
        self.maximize = maximize
        self.best_solution = None
        self.best_value = float("-inf") if maximize else float("inf")
        self.best_result = None
        self.history = []
        self.evaluation_count = 0

        # Create configuration space with single integer hyperparameter
        # 创建包含单个整数超参数的配置空间
        self.cs = ConfigurationSpace()
        # Use UniformIntegerHyperparameter with positional arguments
        # 使用UniformIntegerHyperparameter，参数为位置参数
        hp = CSH.UniformIntegerHyperparameter("config_index", 0, num_configurations - 1, default_value=num_configurations // 2)  # lower bound  # upper bound
        self.cs.add_hyperparameter(hp)

    def smac_objective(self, config: Configuration, seed: int = 0) -> float:
        """
        Objective function for SMAC (discrete optimization)
        SMAC的目标函数（离散优化）
        """
        # Extract configuration index
        # 提取配置索引
        config_index = config["config_index"]
        config_index = int(config_index)

        # Ensure index is within valid range
        # 确保索引在有效范围内
        config_index = max(0, min(config_index, self.num_configurations - 1))

        print(f"\n===== SMAC Stage 1 评估 {self.evaluation_count + 1} =====")
        print(f"评估配置索引: {config_index}/{self.num_configurations - 1}")

        # Evaluate configuration
        # 评估配置
        start_time = time.time()
        fom, result = self.evaluation_function(config_index)
        end_time = time.time()

        print(f"评估用时: {end_time - start_time:.2f} 秒")
        print(f"FoM: {fom:.6e}")

        # Record history
        # 记录历史
        self.history.append({"iteration": self.evaluation_count, "config_index": config_index, "fom": fom, "result": result})

        # Update best solution
        # 更新最佳解
        if (self.maximize and fom > self.best_value) or (not self.maximize and fom < self.best_value):
            self.best_solution = config_index
            self.best_value = fom
            self.best_result = result
            print(f"*** 发现新的最优解: 索引={config_index}, FoM={fom:.6e} ***")

        self.evaluation_count += 1

        # SMAC minimizes, so return negative FoM if maximizing
        # SMAC是最小化，所以如果最大化则返回负FoM
        if self.maximize:
            return -fom
        else:
            return fom

    def optimize_discrete(self, max_iter=100):
        """
        Run SMAC optimization for discrete space
        运行离散空间的SMAC优化
        """
        print(f"Starting SMAC optimization for discrete space with {self.num_configurations} configurations")
        print(f"开始对包含{self.num_configurations}个配置的离散空间进行SMAC优化")

        # Create SMAC scenario
        # 创建SMAC场景
        scenario = Scenario(
            configspace=self.cs,
            deterministic=True,
            n_trials=max_iter,
            seed=42,
        )

        # Initialize SMAC
        # 初始化SMAC
        smac = HyperparameterOptimizationFacade(
            scenario,
            self.smac_objective,
            dask_client=None,  # No parallel execution
        )

        # Run optimization
        # 运行优化
        print("开始SMAC优化循环...")
        incumbent = smac.optimize()

        # Extract best configuration index
        # 提取最佳配置索引
        best_config_index = int(incumbent["config_index"])
        best_config_index = max(0, min(best_config_index, self.num_configurations - 1))

        print(f"\nSMAC优化完成，共进行{self.evaluation_count}次评估")
        print(f"最终配置索引: {best_config_index}")

        # Use best solution found during optimization
        # 使用优化过程中找到的最佳解
        if self.best_solution is not None:
            return self.best_solution, self.best_value, self.best_result
        else:
            # Fallback: evaluate the incumbent configuration
            # 后备方案：评估最终配置
            fom, result = self.evaluation_function(best_config_index)
            return best_config_index, fom, result


class TwoStageOptimizer:
    """
    Two-stage SRAM optimization
    二阶段SRAM优化
    """

    def __init__(self, seed=42, gen_unused_cells=True):
        """
        Initialize two-stage optimizer
        初始化二阶段优化器
        """
        # Set random seed
        seed_set(seed)
        self.seed = seed

        # Create directories
        create_directories()

        self.gen_unused_cells = gen_unused_cells

        # Initialize architecture configuration space
        self.arch_space = ArchitectureConfigurationSpace()

        # Initialize transistor parameter space (includes bitcell + peripheral circuits)
        self.param_space = CompositeSRAMParameterSpace()

        # Results storage
        self.stage1_results = {}
        self.stage2_results = {}
        self.joint_results = {}

        # Iteration history storage
        # 迭代历史存储
        self.stage1_iteration_history = []  # 阶段1每次迭代的详细信息
        self.stage2_iteration_history = []  # 阶段2每次迭代的详细信息
        self.joint_iteration_history = []
        self.joint_csv_tag = None

        print("Two-stage optimizer initialized")
        print("二阶段优化器已初始化")

    def _build_iteration_record(self, iteration_num, params, result, success, objectives, constraints, rows, cols, num_arrays):
        """Build a standardized iteration record dict for CSV logging."""
        record = {
            "iteration": iteration_num,
            "fom": (result.get("fom", -1e9) if success and result else -1e9),
            "success": success,
            "rows": rows,
            "cols": cols,
            "num_arrays": num_arrays,
        }
        if params:
            record.update({
                "pg_width": params.get("pg_width", 0),
                "pd_width": params.get("pd_width", 0),
                "pu_width": params.get("pu_width", 0),
                "length": params.get("length", 0),
                "nmos_model_name": params.get("nmos_model_name", ""),
                "pmos_model_name": params.get("pmos_model_name", ""),
            })
            for pk in PERIPHERAL_ALL_KEYS:
                record[pk] = params.get(pk, "")
        if success and result:
            record.update({
                "min_snm": result.get("min_snm", 0),
                "hold_snm": result.get("hold_snm", 0),
                "read_snm": result.get("read_snm", 0),
                "write_snm": result.get("write_snm", 0),
                "read_delay": result.get("read_delay", 0),
                "write_delay": result.get("write_delay", 0),
                "max_delay": max(result.get("read_delay", 0), result.get("write_delay", 0)),
                "read_power": result.get("read_power", 0),
                "write_power": result.get("write_power", 0),
                "max_power": result.get("max_power", 0),
                "total_power": result.get("total_power", 0),
                "single_array_area": result.get("single_array_area", 0),
                "total_area": result.get("area", 0),
            })
        if objectives:
            for i, v in enumerate(objectives[:4]):
                record[f"objective_{i}"] = v
        if constraints:
            for i, v in enumerate(constraints[:2]):
                record[f"constraint_{i}"] = v
        return record

    def _append_iteration_csv(self, stage: str, record: Dict[str, Any], label: str = None):
        algorithm_dir = getattr(self, "current_algorithm", None) or "unknown"
        base_dir = Path(current_dir) / "experiment" / algorithm_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        if label == "joint" and self.joint_csv_tag:
            filename = f"{stage}_iterations_live_joint_{self.joint_csv_tag}.csv"
        else:
            filename = f"{stage}_iterations_live.csv" if not label else f"{stage}_iterations_live_{label}.csv"
        path = base_dir / filename

        stage1_fields = [
            "seed",
            "iteration",
            "config_index",
            "rows",
            "cols",
            "num_arrays",
            "array_capacity",
            "fom",
            "min_snm",
            "hold_snm",
            "read_snm",
            "write_snm",
            "read_delay",
            "write_delay",
            "max_delay",
            "read_power",
            "write_power",
            "max_power",
            "total_power",
            "single_array_area",
            "total_area",
            "success",
        ]
        stage2_fields = [
            "seed",
            "iteration",
            "fom",
            "success",
            "rows",
            "cols",
            "num_arrays",
            "pg_width",
            "pd_width",
            "pu_width",
            "length",
            "nmos_model_name",
            "pmos_model_name",
        ] + list(PERIPHERAL_ALL_KEYS) + [
            "min_snm",
            "hold_snm",
            "read_snm",
            "write_snm",
            "read_delay",
            "write_delay",
            "max_delay",
            "read_power",
            "write_power",
            "max_power",
            "total_power",
            "single_array_area",
            "total_area",
            "objective_0",
            "objective_1",
            "objective_2",
            "objective_3",
            "constraint_0",
            "constraint_1",
        ]

        fields = stage1_fields if stage == "stage1" else stage2_fields
        write_header = not path.exists()
        if not record.get("success", False):
            return
        key_fields = ["pg_width", "pd_width", "pu_width", "length", "nmos_model_name", "pmos_model_name", "fom", "max_delay", "max_power"]
        last = None
        if path.exists():
            try:
                with open(path, "r", newline="", encoding="utf-8") as fr:
                    reader = csv.DictReader(fr)
                    for r in reader:
                        last = r
            except Exception:
                last = None
        dedupe = False
        if last is not None:
            try:
                dedupe = all(str(record.get(k, "")) == str(last.get(k, "")) for k in key_fields if k in fields)
            except Exception:
                dedupe = False
        if dedupe:
            return
        record = dict(record)
        record.setdefault("seed", getattr(self, "seed", None))
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            row = {k: record.get(k, "") for k in fields}
            writer.writerow(row)

    def _calculate_timeout(self, rows, cols, base_timeout=120):
        """
        Calculate dynamic timeout based on configuration size
        根据配置大小计算动态超时时间

        Args:
            rows: Number of rows
            cols: Number of columns
            base_timeout: Base timeout in seconds (default 120)

        Returns:
            Calculated timeout in seconds
        """
        # Calculate array size
        # 计算阵列大小
        array_size = rows * cols

        # Timeout scales with array size
        # For small arrays (< 1000 bits): use base timeout
        # For medium arrays (1000-10000 bits): scale by 1.5x
        # For large arrays (> 10000 bits): scale by 2.0x or more
        # 超时时间随阵列大小缩放
        # 小阵列 (< 1000 bits): 使用基础超时
        # 中等阵列 (1000-10000 bits): 缩放1.5倍
        # 大阵列 (> 10000 bits): 缩放2.0倍或更多

        if array_size < 1000:
            timeout = base_timeout
        elif array_size < 10000:
            # Linear scaling: 1.0 to 1.5x
            # 线性缩放：1.0到1.5倍
            scale = 1.0 + 0.5 * (array_size - 1000) / 9000
            timeout = int(base_timeout * scale)
        else:
            # For very large arrays, use more aggressive scaling
            # 对于非常大的阵列，使用更激进的缩放
            # Scale factor: 1.5 + (size - 10000) / 20000, capped at 3.0x
            # 缩放因子：1.5 + (大小 - 10000) / 20000，上限为3.0倍
            scale = min(1.5 + (array_size - 10000) / 20000, 3.0)
            timeout = int(base_timeout * scale)

        # Ensure minimum timeout
        # 确保最小超时时间
        timeout = max(timeout, base_timeout)

        return timeout

    def _evaluate_architecture(self, config_index: int) -> Tuple[float, Dict]:
        """
        Evaluate architecture configuration with fixed transistor parameters
        使用固定晶体管参数评估架构配置
        """
        try:
            # Get configuration
            config = self.arch_space.get_configuration(config_index)

            # Use default transistor parameters for Stage 1
            default_params = {"nmos_model_name": "NMOS_VTG", "pmos_model_name": "PMOS_VTG", "pd_width": 0.205e-6, "pu_width": 0.09e-6, "pg_width": 0.135e-6, "length": 50e-9, "length_nm": 50}

            print(f"Evaluating config {config_index}: {config['rows']}x{config['cols']} x{config['num_arrays']} arrays")
            print(f"评估配置 {config_index}: {config['rows']}x{config['cols']} x{config['num_arrays']} 个阵列")

            timeout = None

            # Evaluate single array performance
            iteration_num = len(self.stage1_iteration_history)
            objectives, constraints, result, success = evaluate_sram_with_config(default_params, config["rows"], config["cols"], config["num_arrays"], timeout=timeout, stage_label="stage1", iteration_index=iteration_num, gen_unused_cells=self.gen_unused_cells)

            if success and result:
                # Scale metrics by number of arrays
                max_power = result["max_power"]
                total_area = result["area"]

                # Stage1 FoM (no SNM simulation): FoM = log10(1 / (max_power * sqrt(area) * max_delay))
                # Stage1 品质因数（无 SNM 仿真）: FoM = log10(1 / (max_power * sqrt(area) * max_delay))
                max_delay = max(result["read_delay"], result["write_delay"])
                if max_power > 0 and total_area > 0 and max_delay > 0:
                    fom = np.log10(1 / (max_power * np.sqrt(total_area) * max_delay))
                else:
                    fom = -10.0  # Penalty value for invalid cases

                evaluation_result = {"config": config, "single_array_result": result, "total_power": max_power, "total_area": total_area, "max_delay": max_delay, "fom": fom, "success": True}

                # Record iteration history
                # 记录迭代历史
                iteration_record = {"iteration": iteration_num, "config_index": config_index, "rows": config["rows"], "cols": config["cols"], "num_arrays": config["num_arrays"], "array_capacity": config["array_capacity"], "fom": fom, "min_snm": result["min_snm"], "hold_snm": result["hold_snm"], "read_snm": result["read_snm"], "write_snm": result["write_snm"], "read_delay": result["read_delay"], "write_delay": result["write_delay"], "max_delay": max_delay, "read_power": result["read_power"], "write_power": result["write_power"], "max_power": result["max_power"], "total_power": max_power, "single_array_area": result.get("single_array_area", total_area), "total_area": total_area, "success": True}
                self.stage1_iteration_history.append(iteration_record)
                self._append_iteration_csv("stage1", iteration_record)

                print(f"Config {config_index} FoM: {fom:.6e}, max_delay: {max_delay:.2e}s")
                print(f"配置 {config_index} FoM: {fom:.6e}, 最大延迟: {max_delay:.2e}s")

                return fom, evaluation_result
            else:
                # Record failed iteration
                # 记录失败的迭代
                iteration_record = {"iteration": iteration_num, "config_index": config_index, "rows": config["rows"], "cols": config["cols"], "num_arrays": config["num_arrays"], "array_capacity": config["array_capacity"], "fom": -1e9, "success": False}
                self.stage1_iteration_history.append(iteration_record)
                self._append_iteration_csv("stage1", iteration_record)

                print(f"Config {config_index} evaluation failed")
                print(f"配置 {config_index} 评估失败")
                return -1e9, {"config": config, "success": False}

        except Exception as e:
            print(f"Error evaluating config {config_index}: {str(e)}")
            print(f"评估配置 {config_index} 时出错: {str(e)}")
            # Record error iteration
            try:
                config = self.arch_space.get_configuration(config_index)
                iteration_record = {"iteration": len(self.stage1_iteration_history), "config_index": config_index, "rows": config["rows"], "cols": config["cols"], "num_arrays": config.get("num_arrays", 0), "array_capacity": config.get("array_capacity", 0), "fom": -1e9, "success": False, "error": str(e)}
                self.stage1_iteration_history.append(iteration_record)
                self._append_iteration_csv("stage1", iteration_record)
            except:
                pass
            return -1e9, {"config": self.arch_space.get_configuration(config_index), "success": False}

    def _evaluate_transistor_params(self, params_normalized: np.ndarray) -> Tuple[float, Dict]:
        """
        Evaluate transistor parameters for the best architecture from Stage 1
        为第一阶段的最佳架构评估晶体管参数
        """
        try:
            # Convert normalized parameters to actual parameters
            params_tensor = torch.tensor(params_normalized, dtype=torch.float32)
            params = self.param_space.convert_params(params_tensor)

            # Get best architecture from Stage 1
            best_config = self.stage1_results["best_config"]

            print(f"Evaluating transistor params for {best_config['rows']}x{best_config['cols']} configuration")
            print(f"为 {best_config['rows']}x{best_config['cols']} 配置评估晶体管参数")

            # Evaluate single array performance
            objectives, constraints, result, success = evaluate_sram_with_config(params, best_config["rows"], best_config["cols"], best_config["num_arrays"], stage_label="stage2_pre", gen_unused_cells=self.gen_unused_cells)

            if success and result:
                # Scale metrics by number of arrays
                max_power = result["max_power"]
                total_area = result["area"]

                # FoM (Figure of Merit) for Stage 2 pre-evaluation
                # 阶段2预评估的FoM
                # FoM = log10(min_snm / (max_power * max_delay))
                max_delay = max(result["read_delay"], result["write_delay"])
                if result["min_snm"] > 0 and max_power > 0 and max_delay > 0:
                    fom = np.log10(result["min_snm"] / (max_power * max_delay))
                else:
                    fom = -10.0  # Penalty value for invalid cases

                evaluation_result = {"params": params, "single_array_result": result, "total_power": max_power, "total_area": total_area, "max_delay": max_delay, "fom": fom, "success": True}

                return fom, evaluation_result
            else:
                return -1e9, {"params": params, "success": False}

        except Exception as e:
            print(f"Error evaluating transistor params: {str(e)}")
            print(f"评估晶体管参数时出错: {str(e)}")
            return -1e9, {"success": False}

    def run_stage1_optimization(self, max_iter=400, algorithm="SA"):
        """
        Run Stage 1: Architecture configuration optimization
        运行第一阶段：架构配置优化

        Args:
            max_iter: Maximum iterations
            algorithm: Algorithm to use ('SA' or 'SMAC' for discrete optimization)
        """
        print("=" * 60)
        print(f"STAGE 1: Architecture Configuration Optimization ({algorithm})")
        print(f"第一阶段：架构配置优化 ({algorithm})")
        print("=" * 60)

        self.current_algorithm = algorithm
        # Clear iteration history for new run
        # 清除迭代历史以开始新的运行
        self.stage1_iteration_history = []

        num_configs = self.arch_space.get_num_configurations()

        if algorithm == "SA":
            # Create SA optimizer for discrete space
            sa_optimizer = SAOptimizer(evaluation_function=self._evaluate_architecture, maximize=True)

            # Run optimization
            best_config_index, best_fom, best_result = sa_optimizer.optimize_discrete(num_solutions=num_configs, max_iter=max_iter)

            # Store Stage 1 results
            self.stage1_results = {"best_config_index": best_config_index, "best_config": self.arch_space.get_configuration(best_config_index), "best_fom": best_fom, "best_result": best_result, "optimization_history": sa_optimizer.history, "algorithm": algorithm}
        elif algorithm == "SMAC":
            # Create SMAC optimizer for discrete space
            smac_optimizer = SMACStage1Optimizer(evaluation_function=self._evaluate_architecture, num_configurations=num_configs, maximize=True)

            # Run optimization
            best_config_index, best_fom, best_result = smac_optimizer.optimize_discrete(max_iter=max_iter)

            # Store Stage 1 results
            self.stage1_results = {"best_config_index": best_config_index, "best_config": self.arch_space.get_configuration(best_config_index), "best_fom": best_fom, "best_result": best_result, "optimization_history": smac_optimizer.history, "algorithm": algorithm}
        else:
            raise ValueError(f"Unsupported algorithm for Stage 1: {algorithm}. Stage 1 requires discrete optimization (currently only SA and SMAC are supported).")

        print(f"\nStage 1 completed!")
        print(f"第一阶段完成！")
        print(f"Best configuration: {self.stage1_results['best_config']}")
        print(f"最佳配置: {self.stage1_results['best_config']}")
        print(f"Best FoM: {self.stage1_results['best_fom']:.6e}")
        print(f"最佳FoM: {self.stage1_results['best_fom']:.6e}")

        # 在阶段1完成后，生成功耗-延时散点与帕累托前沿图
        try:
            output_png = str(Path(current_dir) / "new_result" / "power_delay_pareto_stage1.png")
            self._plot_power_delay_pareto_stage1(output_png)
        except Exception as e:
            print(f"Warning: Failed to plot Stage 1 power-delay Pareto frontier: {e}")

        return self.stage1_results

    def run_stage2_optimization(self, max_iter=400, algorithm="SA", config_file=None):
        """
        Run Stage 2: Transistor parameter optimization
        运行第二阶段：晶体管参数优化

        Args:
            max_iter: Maximum iterations
            algorithm: Algorithm to use ('SA', 'PSO', 'SMAC', 'CBO', 'RoSE_Opt', 'MOEAD')
            config_file: Configuration file path
        """
        print("=" * 60)
        print(f"STAGE 2: Transistor Parameter Optimization ({algorithm})")
        print(f"第二阶段：晶体管参数优化 ({algorithm})")
        print("=" * 60)

        if not self.stage1_results:
            raise ValueError("Stage 1 must be completed before Stage 2")

        self.current_algorithm = algorithm
        # Clear iteration history for new run
        # 清除迭代历史以开始新的运行
        self.stage2_iteration_history = []

        # Get best architecture from Stage 1
        best_config = self.stage1_results["best_config"]

        # Track iteration count for stage 2
        # 跟踪阶段2的迭代计数
        stage2_iteration_counter = {"count": 0}

        # Create stage 2 evaluation function (fixed architecture, optimize transistor params)
        def stage2_eval_fn(x, timeout=None):
            """评估函数：固定架构，优化晶体管参数"""
            import numpy as _np

            # Handle both dict and array inputs (RoSE_Opt may pass dict, others pass array)
            # 处理字典和数组两种输入（RoSE_Opt可能传递字典，其他算法传递数组）
            if isinstance(x, dict):
                # If x is already a parameter dict, use it directly
                # 如果x已经是参数字典，直接使用
                params = x
            else:
                # If x is an array, convert it to parameters
                # 如果x是数组，将其转换为参数
                xv = _np.array(x, dtype=float)
                params_tensor = torch.tensor(xv, dtype=torch.float32)
                params = self.param_space.convert_params(params_tensor)

            # Calculate dynamic timeout if not provided
            # 如果未提供，计算动态超时时间
            if timeout is None:
                timeout = self._calculate_timeout(best_config["rows"], best_config["cols"])

            # Evaluate with fixed architecture
            objectives, constraints, result, success = evaluate_sram_with_config(
                params,
                best_config["rows"],
                best_config["cols"],
                best_config["num_arrays"],
                timeout=timeout,
                stage_label="stage2",
                iteration_index=stage2_iteration_counter["count"],
                gen_unused_cells=self.gen_unused_cells,
            )

            # Calculate FoM for Stage 2
            fom = None
            if success and result:
                max_power = result["max_power"]
                total_area = result["area"]
                max_delay = max(result["read_delay"], result["write_delay"])
                # Stage 2 FoM: log10(min_snm / (max_power * max_delay))
                if result["min_snm"] > 0 and max_power > 0 and max_delay > 0:
                    fom = np.log10(result["min_snm"] / (max_power * max_delay))
                else:
                    fom = -10.0  # Penalty value for invalid cases

                if result:
                    result["fom"] = fom
                    result["total_power"] = max_power
                    result["total_area"] = total_area
                    result["max_delay"] = max_delay
                    result["num_rows"] = best_config["rows"]
                    result["num_cols"] = best_config["cols"]
                    result["num_arrays"] = best_config["num_arrays"]
                    # 提供给下游优化器的统一指标字段
                    result["merit"] = fom

            # Record iteration history
            # 记录迭代历史
            iteration_num = stage2_iteration_counter["count"]
            stage2_iteration_counter["count"] += 1

            iteration_record = self._build_iteration_record(
                iteration_num, params, result, success, objectives, constraints,
                best_config["rows"], best_config["cols"], best_config["num_arrays"],
            )
            self.stage2_iteration_history.append(iteration_record)
            self._append_iteration_csv("stage2", iteration_record)

            return objectives, constraints, result, success

        # Create problem interface (similar to demo_joint_optimization.py)
        param_space = CompositeSRAMParameterSpace(config_file or "config_sram.yaml")
        problem = (param_space, stage2_eval_fn, None)

        # Run optimization based on algorithm
        ret = None
        try:
            if algorithm == "SA":
                from size_optimization import demo_sa

                # Pass iteration count to SA so it runs with the requested max_iter (e.g., 30)
                # 将迭代次数传递给SA，使其按请求的max_iter运行（例如30次）
                ret = demo_sa.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "PSO":
                from size_optimization import demo_pso

                ret = demo_pso.main(config_file or "config_sram.yaml", problem=problem)
            elif algorithm == "SMAC":
                from size_optimization import demo_smac

                ret = demo_smac.main(config_file or "config_sram.yaml", problem=problem)
            elif algorithm == "CBO":
                from size_optimization import demo_cbo

                ret = demo_cbo.main(config_file or "config_sram.yaml", problem=problem)
            elif algorithm == "RoSE_Opt":
                from size_optimization import demo_roseopt

                ret = demo_roseopt.main(config_file or "config_sram.yaml", problem=problem)
            elif algorithm == "MOEAD":
                from size_optimization import demo_moead

                circuit_mode = 1 if self.gen_unused_cells else 2
                ret = demo_moead.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter, circuit_mode=circuit_mode)
            elif algorithm == "MOBO":
                from size_optimization import demo_mobo

                circuit_mode = 1 if self.gen_unused_cells else 2
                ret = demo_mobo.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter, circuit_mode=circuit_mode)
            elif algorithm == "NSGA-II":
                from size_optimization import demo_nsgaii

                circuit_mode = 1 if self.gen_unused_cells else 2
                ret = demo_nsgaii.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter, circuit_mode=circuit_mode)
            elif algorithm == "tSS-BO":
                from size_optimization import demo_tssbo

                ret = demo_tssbo.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "CMA-ES":
                from size_optimization import demo_cmaes

                ret = demo_cmaes.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "CPN":
                from size_optimization import demo_cpn

                ret = demo_cpn.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
        except Exception as e:
            print(f"Error running {algorithm}: {e}")
            traceback.print_exc()
            raise

        # Parse results
        best_params = ret.get("params") if isinstance(ret, dict) else None
        best_result = ret.get("result") if isinstance(ret, dict) else None
        best_fom = ret.get("fom") if isinstance(ret, dict) else None

        # If FoM not directly in result, calculate it (Stage 2 formula)
        if best_fom is None and best_result:
            max_power = best_result.get("max_power", 0)
            total_area = best_result.get("area", 0)
            max_delay = max(best_result.get("read_delay", 0), best_result.get("write_delay", 0))
            min_snm = best_result.get("min_snm", 0)
            # Stage 2 FoM: log10(min_snm / (max_power * max_delay))
            if min_snm > 0 and max_power > 0 and max_delay > 0:
                best_fom = np.log10(min_snm / (max_power * max_delay))
            else:
                best_fom = -10.0

        # Store Stage 2 results
        self.stage2_results = {"best_params": best_params, "best_fom": best_fom, "best_result": best_result, "algorithm": algorithm, "iteration": ret.get("iteration") if isinstance(ret, dict) else None}

        print(f"\nStage 2 completed!")
        print(f"第二阶段完成！")
        if best_fom:
            print(f"Best FoM: {best_fom:.6e}")
            print(f"最佳FoM: {best_fom:.6e}")

        return self.stage2_results

    def _select_candidate_architectures(self, num_candidates: int = 3) -> List[Dict]:
        """
        从阶段1迭代历史的功耗-延时数据中选取候选架构配置。
        策略：基于极小帕累托前沿，选择低功耗点、低延时点以及若干中间代表点。
        返回包含 rows/cols/num_arrays/array_capacity 的配置字典列表。
        """
        # 收集有效点 (power, delay, record)
        valid_records = [rec for rec in self.stage1_iteration_history if rec.get("success") and (rec.get("total_power", 0) > 0) and (rec.get("max_delay", 0) > 0)]
        if not valid_records:
            print("No valid Stage 1 records for candidate selection.")
            return []

        points = [(float(rec["total_power"]), float(rec["max_delay"])) for rec in valid_records]
        frontier = self._compute_pareto_frontier(points)
        if not frontier:
            print("No Pareto frontier found; falling back to best-FoM records.")
            # 退化处理：按FoM排序取前num_candidates
            sorted_by_fom = sorted(valid_records, key=lambda r: r.get("fom", -1e9), reverse=True)
            chosen = sorted_by_fom[: max(1, num_candidates)]
            return [{"rows": c.get("rows"), "cols": c.get("cols"), "num_arrays": c.get("num_arrays"), "array_capacity": c.get("array_capacity")} for c in chosen]

        # 将前沿点映射到对应的记录（首个匹配）
        frontier_records: List[Dict] = []
        for pwr, dly in frontier:
            for rec in valid_records:
                if abs(float(rec["total_power"]) - pwr) < 1e-12 and abs(float(rec["max_delay"]) - dly) < 1e-12:
                    frontier_records.append(rec)
                    break

        if not frontier_records:
            print("Pareto frontier records empty after mapping; using valid records fallback.")
            frontier_records = valid_records

        # 选择低功耗、低延时及中间点
        n = len(frontier_records)
        if n <= num_candidates:
            chosen_records = frontier_records
        else:
            # 等距采样索引（包含首尾）
            import numpy as _np

            indices = list(map(lambda x: int(round(x)), _np.linspace(0, n - 1, num_candidates)))
            chosen_records = [frontier_records[i] for i in indices]

        # 构造配置字典列表
        candidates = []
        seen = set()
        for c in chosen_records:
            key = (c.get("rows"), c.get("cols"), c.get("num_arrays"))
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"rows": c.get("rows"), "cols": c.get("cols"), "num_arrays": c.get("num_arrays"), "array_capacity": c.get("array_capacity")})
        print(f"Selected {len(candidates)} candidate architectures from Stage 1 Pareto frontier.")
        for i, cfg in enumerate(candidates, 1):
            print(f"  [{i}] rows={cfg['rows']}, cols={cfg['cols']}, arrays={cfg['num_arrays']}")
        return candidates

    def _plot_power_delay_pareto_from_records(self, records: List[Dict], output_png_path: str, title: str):
        """
        从给定记录列表绘制功耗-延时散点及极小帕累托前沿，保存到指定路径。
        """
        try:
            points: List[Tuple[float, float]] = []
            for rec in records:
                try:
                    if rec.get("success") and ("total_power" in rec) and ("max_delay" in rec):
                        pwr = float(rec["total_power"])
                        dly = float(rec["max_delay"])
                        if pwr > 0 and dly > 0:
                            points.append((pwr, dly))
                except Exception:
                    continue

            if not points:
                print(f"No valid points to plot Pareto frontier for {title}.")
                return

            frontier = self._compute_pareto_frontier(points)

            plt.figure(figsize=(8, 6))
            p_all = [p for p, d in points]
            d_all = [d for p, d in points]
            plt.scatter(p_all, d_all, s=24, c="tab:blue", alpha=0.6, label="All iterations")
            if frontier:
                p_f = [p for p, d in frontier]
                d_f = [d for p, d in frontier]
                plt.plot(p_f, d_f, "-o", color="tab:red", linewidth=2, markersize=4, label="Pareto frontier")
            plt.xlabel("Total Power (W)")
            plt.ylabel("Max Delay (s)")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()

            out_path = Path(output_png_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_path), dpi=200)
            plt.close()
            print(f"Stage 2 Pareto plot saved: {out_path}")
        except Exception as e:
            print(f"Error while plotting Stage 2 Pareto frontier for {title}: {e}")

    def run_stage2_optimization_for_config(self, fixed_config: Dict, max_iter=400, algorithm="SA", config_file=None, label: str = None) -> Dict:
        """
        在指定架构配置下运行阶段2尺寸优化，并返回结果与本次迭代历史。
        同时生成该配置的功耗-延时帕累托图。
        """
        self.current_algorithm = algorithm
        # 局部迭代历史（不污染全局 self.stage2_iteration_history）
        local_history: List[Dict] = []
        stage2_iteration_counter = {"count": 0}

        if label is None:
            label = f"r{fixed_config['rows']}_c{fixed_config['cols']}_a{fixed_config['num_arrays']}"

        def stage2_eval_fn_local(x, timeout=None):
            # 参数转换
            if isinstance(x, dict):
                params = x
            else:
                xv = np.array(x, dtype=float)
                params_tensor = torch.tensor(xv, dtype=torch.float32)
                params = self.param_space.convert_params(params_tensor)

            if timeout is None:
                timeout = self._calculate_timeout(fixed_config["rows"], fixed_config["cols"])

            objectives, constraints, result, success = evaluate_sram_with_config(
                params,
                fixed_config["rows"],
                fixed_config["cols"],
                fixed_config["num_arrays"],
                timeout=timeout,
                stage_label="stage2",
                iteration_index=stage2_iteration_counter["count"],
                gen_unused_cells=self.gen_unused_cells,
            )

            fom = None
            if success and result:
                max_power = result["max_power"]
                total_area = result["area"]
                max_delay = max(result["read_delay"], result["write_delay"])
                # Stage 2 FoM: log10(min_snm / (max_power * max_delay))
                if result["min_snm"] > 0 and max_power > 0 and max_delay > 0:
                    fom = np.log10(result["min_snm"] / (max_power * max_delay))
                else:
                    fom = -10.0
                result["fom"] = fom
                result["total_power"] = max_power
                result["total_area"] = total_area
                result["max_delay"] = max_delay
                result["num_rows"] = fixed_config["rows"]
                result["num_cols"] = fixed_config["cols"]
                result["num_arrays"] = fixed_config["num_arrays"]
                result["merit"] = fom

            iteration_num = stage2_iteration_counter["count"]
            stage2_iteration_counter["count"] += 1

            iteration_record = self._build_iteration_record(
                iteration_num, params, result, success, objectives, constraints,
                fixed_config["rows"], fixed_config["cols"], fixed_config["num_arrays"],
            )
            local_history.append(iteration_record)
            self._append_iteration_csv("stage2", iteration_record, label=label)
            return objectives, constraints, result, success

        # 问题接口
        param_space = CompositeSRAMParameterSpace(config_file or "config_sram.yaml")
        problem = (param_space, stage2_eval_fn_local, None)

        # 运行优化
        ret = None
        try:
            if algorithm == "SA":
                from size_optimization import demo_sa

                ret = demo_sa.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "PSO":
                from size_optimization import demo_pso

                ret = demo_pso.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "SMAC":
                from size_optimization import demo_smac

                ret = demo_smac.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "CBO":
                from size_optimization import demo_cbo

                ret = demo_cbo.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "RoSE_Opt":
                from size_optimization import demo_roseopt

                ret = demo_roseopt.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "MOEAD":
                from size_optimization import demo_moead

                circuit_mode = 1 if self.gen_unused_cells else 2
                ret = demo_moead.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter, circuit_mode=circuit_mode)
            elif algorithm == "tSS-BO":
                from size_optimization import demo_tssbo

                ret = demo_tssbo.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "CMA-ES":
                from size_optimization import demo_cmaes

                ret = demo_cmaes.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            elif algorithm == "CPN":
                from size_optimization import demo_cpn

                ret = demo_cpn.main(config_file or "config_sram.yaml", problem=problem, max_iter=max_iter)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
        except Exception as e:
            print(f"Error running {algorithm} for fixed config {label}: {e}")
            traceback.print_exc()
            raise

        # 解析结果
        best_params = ret.get("params") if isinstance(ret, dict) else None
        best_result = ret.get("result") if isinstance(ret, dict) else None
        best_fom = ret.get("fom") if isinstance(ret, dict) else None
        if best_fom is None and best_result:
            max_power = best_result.get("max_power", 0)
            total_area = best_result.get("area", 0)

            max_delay = max(best_result.get("read_delay", 0), best_result.get("write_delay", 0))
            min_snm = best_result.get("min_snm", 0)
            # Stage 2 FoM: log10(min_snm / (max_power * max_delay))
            if min_snm > 0 and max_power > 0 and max_delay > 0:
                best_fom = np.log10(min_snm / (max_power * max_delay))
            else:
                best_fom = -10.0

        # 绘制该架构的阶段2帕累托图
        out_dir = Path(current_dir) / "new_result"
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"power_delay_pareto_stage2_{label}.png"
        title = f"Stage 2 Pareto (rows={fixed_config['rows']}, cols={fixed_config['cols']}, arrays={fixed_config['num_arrays']})"
        self._plot_power_delay_pareto_from_records(local_history, str(png_path), title)

        # 在指定输出目录写入每次迭代的CSV（包含架构与算法信息）
        try:
            out_dir = Path(current_dir) / "experiment"
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = out_dir / f"iterations_{label}_{algorithm}_{timestamp}.csv"

            # 聚合字段名
            all_fields = set(["architecture", "algorithm"])  # 额外字段
            for rec in local_history:
                all_fields.update(rec.keys())
            fieldnames = sorted(list(all_fields))

            # 写入记录
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in local_history:
                    row = {k: rec.get(k, "") for k in fieldnames}
                    row["architecture"] = f"{fixed_config['rows']}x{fixed_config['cols']}"
                    row["algorithm"] = algorithm
                    writer.writerow(row)
            print(f"Stage 2 iteration CSV saved to: {csv_path}")
        except Exception as e:
            print(f"Warning: failed to save iteration CSV: {e}")

        return {"best_params": best_params, "best_result": best_result, "best_fom": best_fom, "history": local_history, "label": label, "config": fixed_config}

    def run_stage2_for_candidate_architectures(self, num_candidates: int = 3, max_iter=400, algorithm="SA", config_file=None) -> List[Dict]:
        """
        基于阶段1的帕累托前沿选择若干候选架构，分别运行阶段2优化，并为每个候选输出一张功耗-延时帕累托图。
        返回每个候选架构的阶段2优化摘要列表。
        """
        if not self.stage1_results:
            raise ValueError("Stage 1 must be completed before selecting candidates for Stage 2.")
        candidates = self._select_candidate_architectures(num_candidates=num_candidates)
        summaries: List[Dict] = []
        for i, cfg in enumerate(candidates, 1):
            label = f"cand{i}_r{cfg['rows']}_c{cfg['cols']}_a{cfg['num_arrays']}"
            print(f"\nRunning Stage 2 for candidate [{i}] rows={cfg['rows']}, cols={cfg['cols']}, arrays={cfg['num_arrays']}")
            summ = self.run_stage2_optimization_for_config(fixed_config=cfg, max_iter=max_iter, algorithm=algorithm, config_file=config_file, label=label)
            summaries.append(summ)
        print(f"\nCompleted Stage 2 for {len(summaries)} candidate architectures.")
        return summaries

    def _build_fixed_joint_vector(self, fixed_params: dict) -> np.ndarray:
        """
        Encode a params dict (with 'rows', 'cols', transistor params) into the
        normalized joint vector [row_x, col_x, *bitcell_x, *peripheral_x] used
        by SA's optimize_continuous.
        将包含 rows/cols 的参数字典编码为 SA 连续优化所需的归一化联合向量。
        """
        row_choices = self.arch_space.row_choices
        col_choices = self.arch_space.column_choices
        rows = fixed_params.get("rows", row_choices[0])
        cols = fixed_params.get("cols", col_choices[0])
        row_idx = row_choices.index(rows) if rows in row_choices else 0
        col_idx = col_choices.index(cols) if cols in col_choices else 0
        x_row = (row_idx + 0.5) / len(row_choices)
        x_col = (col_idx + 0.5) / len(col_choices)

        # --- Bitcell dimensions ---
        bc_x = []
        for param_name, info in self.param_space.bitcell_space.param_info.items():
            if info["type"] == "continuous_list":
                key_map = {}
                if param_name == "nmos_width":
                    key_map = {"pg": "pg_width", "pd": "pd_width"}
                elif param_name == "pmos_width":
                    key_map = {"pu": "pu_width"}
                for i, name in enumerate(info["names"]):
                    key = key_map.get(name)
                    val = float(fixed_params[key]) if (key and key in fixed_params) else float(info["lower"][i])
                    x_norm = (val - info["lower"][i]) / (info["upper"][i] - info["lower"][i])
                    bc_x.append(float(np.clip(x_norm, 0.0, 1.0)))
            elif info["type"] in ("continuous_scalar", "continuous value"):
                if param_name == "length":
                    val = float(fixed_params.get("length", info["lower"]))
                    lo, hi = float(info["lower"]), float(info["upper"])
                    x_norm = (val - lo) / (hi - lo) if hi > lo else 0.5
                    bc_x.append(float(np.clip(x_norm, 0.0, 1.0)))
                else:
                    bc_x.append(0.5)
            elif info["type"] == "categorical_list":
                key_map = {}
                if param_name == "nmos_model":
                    key_map = {n: "nmos_model_name" for n in info["names"]}
                elif param_name == "pmos_model":
                    key_map = {n: "pmos_model_name" for n in info["names"]}
                choices = info["choices"]
                for name in info["names"]:
                    key = key_map.get(name)
                    val = fixed_params.get(key, choices[0]) if key else choices[0]
                    idx = choices.index(val) if val in choices else 0
                    bc_x.append((idx + 0.01) / len(choices))

        # --- Peripheral variable dimensions ---
        periph_x = []
        for pdef in self.param_space.peripheral_defs:
            key = pdef["key"]
            if pdef["type"] == "continuous":
                val = float(fixed_params.get(key, pdef["default"]))
                rng = pdef["upper"] - pdef["lower"]
                x_norm = (val - pdef["lower"]) / rng if rng > 0 else 0.5
                periph_x.append(float(np.clip(x_norm, 0.0, 1.0)))
            elif pdef["type"] == "categorical":
                choices = pdef["choices"]
                val = fixed_params.get(key, pdef["default"])
                idx = choices.index(val) if val in choices else 0
                periph_x.append((idx + 0.01) / len(choices))

        return np.array([x_row, x_col] + bc_x + periph_x, dtype=float)

    def _decode_joint_vector(self, x):
        x = np.array(x, dtype=float)
        if x.size < 2:
            return None
        row_count = len(self.arch_space.row_choices)
        col_count = len(self.arch_space.column_choices)
        row_idx = int(x[0] * row_count)
        col_idx = int(x[1] * col_count)
        row_idx = max(0, min(row_idx, row_count - 1))
        col_idx = max(0, min(col_idx, col_count - 1))
        rows = self.arch_space.row_choices[row_idx]
        cols = self.arch_space.column_choices[col_idx]
        array_capacity = rows * cols
        if self.arch_space.total_bits % array_capacity != 0:
            return None
        num_arrays = self.arch_space.total_bits // array_capacity
        params_tensor = torch.tensor(x[2:], dtype=torch.float32)
        params = self.param_space.convert_params(params_tensor)
        return rows, cols, num_arrays, params, array_capacity, row_idx, col_idx

    def run_joint_optimization(self, max_iter=200, algorithm="SA"):
        print("=" * 60)
        print("JOINT SRAM OPTIMIZATION")
        print("架构与晶体管尺寸联合优化")
        print("=" * 60)

        self.current_algorithm = algorithm
        seed_offsets = {"SA": 1, "PSO": 2, "SMAC": 3, "CBO": 4, "RoSE_Opt": 5, "MOEAD": 6}
        seed_set(self.seed + seed_offsets.get(algorithm, 0))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.joint_csv_tag = f"{algorithm}_{max_iter}_seed{self.seed}_{timestamp}"
        self.joint_iteration_history = []
        iteration_counter = {"count": 0}
        joint_param_space = JointParameterSpace(self.arch_space, self.param_space)

        # 统一起始点：所有算法从相同初始配置出发，确保 FoM 可比
        # Unified starting point for all algorithms to ensure FoM comparability
        FIXED_INITIAL_PARAMS = {
            "rows": 16, "cols": 16,
            "pg_width": 1.35e-7, "pd_width": 2.05e-7, "pu_width": 9e-8, "length": 5e-8,
            "nmos_model_name": "NMOS_VTG", "pmos_model_name": "PMOS_VTG",
            "sa_p_width": 5.4e-7, "sa_n_width": 2.7e-7, "sa_length": 5e-8,
        }

        def _evaluate_joint(x):
            decoded = None
            params = None
            if isinstance(x, dict):
                params = dict(x)
                rows = params.get("rows", self.arch_space.row_choices[0])
                cols = params.get("cols", self.arch_space.column_choices[0])
                array_capacity = rows * cols
                if self.arch_space.total_bits % array_capacity == 0:
                    num_arrays = self.arch_space.total_bits // array_capacity
                    decoded = (rows, cols, num_arrays, params, array_capacity, None, None)
            else:
                decoded = self._decode_joint_vector(x)
            iteration_num = iteration_counter["count"]
            iteration_counter["count"] += 1
            if decoded is None:
                record = {"iteration": iteration_num, "fom": -1e9, "success": False}
                self.joint_iteration_history.append(record)
                self._append_iteration_csv("stage2", record, label="joint")
                return -1e9, None, None, None, False, None, params

            rows, cols, num_arrays, params, array_capacity, row_idx, col_idx = decoded
            timeout = self._calculate_timeout(rows, cols)
            objectives, constraints, result, success = evaluate_sram_with_config(
                params,
                rows,
                cols,
                num_arrays,
                timeout=timeout,
                stage_label=f"joint/{algorithm}",
                iteration_index=iteration_num,
                gen_unused_cells=self.gen_unused_cells,
            )
            fom = -1e9
            if success and result:
                total_power = result["max_power"]
                max_delay = max(result["read_delay"], result["write_delay"])
                area = result.get("area", 0)
                if result["min_snm"] > 0 and total_power > 0 and area > 0 and max_delay > 0:
                    fom = np.log10(result["min_snm"] / (total_power * np.sqrt(area) * max_delay))
                else:
                    fom = -10.0
                result["total_power"] = total_power
                result["total_area"] = area
                result["fom"] = fom
                result["merit"] = fom
                result["num_rows"] = rows
                result["num_cols"] = cols
                result["num_arrays"] = num_arrays

            record = self._build_iteration_record(
                iteration_num, params, result, success, objectives, constraints, rows, cols, num_arrays,
            )
            self.joint_iteration_history.append(record)
            self._append_iteration_csv("stage2", record, label="joint")

            config = {"rows": rows, "cols": cols, "num_arrays": num_arrays, "array_capacity": array_capacity}
            return fom, objectives, constraints, result, success, config, params

        def joint_eval_fn(x):
            fom, objectives, constraints, result, success, config, params = _evaluate_joint(x)
            return fom, {"config": config, "params": params, "result": result, "success": success}

        def joint_eval_backend(x):
            fom, objectives, constraints, result, success, config, params = _evaluate_joint(x)
            if objectives is None:
                objectives = [fom, 0.0, 0.0]
            if constraints is None:
                constraints = [1.0, 1.0]
            if success and result:
                if "merit" not in result:
                    result["merit"] = fom
                if "fom" not in result:
                    result["fom"] = fom
            return objectives, constraints, result, success

        if algorithm == "SA":
            param_dim = int(self.param_space.bounds.shape[1])
            total_dim = param_dim + 2
            bounds = (np.zeros(total_dim), np.ones(total_dim))
            # 将固定起始点编码为归一化向量，SA 从该点开始探索（iter 0 = 固定起始点）
            # Encode fixed starting point; SA explores from there (iter 0 = fixed point)
            initial_solution = self._build_fixed_joint_vector(FIXED_INITIAL_PARAMS)

            sa_optimizer = SAOptimizer(evaluation_function=joint_eval_fn, bounds=bounds, maximize=True)
            best_solution, best_fom, best_result = sa_optimizer.optimize_continuous(
                initial_solution, max_iter=max_iter, arch_dims=2, arch_sigma=0.35
            )

            best_config = None
            best_params = None
            decoded_best = self._decode_joint_vector(best_solution)
            if decoded_best is not None:
                rows, cols, num_arrays, params, array_capacity, row_idx, col_idx = decoded_best
                best_config = {"rows": rows, "cols": cols, "num_arrays": num_arrays, "array_capacity": array_capacity}
                best_params = params

            self.joint_results = {
                "best_config": best_config,
                "best_params": best_params,
                "best_fom": best_fom,
                "best_result": best_result,
                "history": sa_optimizer.history,
                "algorithm": algorithm,
            }
            return self.joint_results

        # 非 SA 算法：预评估固定起始点为 iter 0，再由算法从 iter 1 开始自主探索
        # Non-SA algorithms: pre-evaluate fixed starting point as iter 0
        print("Pre-evaluating fixed initial point (iter 0) for FoM comparability...")
        print("预评估固定起始点为第0次迭代，确保各算法FoM可比...")
        _evaluate_joint(FIXED_INITIAL_PARAMS)

        problem = (joint_param_space, joint_eval_backend, None)
        ret = None
        if algorithm == "PSO":
            from size_optimization import demo_pso

            ret = demo_pso.main("config_sram.yaml", problem=problem, max_iter=max_iter)
        elif algorithm == "SMAC":
            from size_optimization import demo_smac

            ret = demo_smac.main("config_sram.yaml", problem=problem, max_iter=max_iter)
        elif algorithm == "CBO":
            from size_optimization import demo_cbo

            ret = demo_cbo.main("config_sram.yaml", problem=problem, max_iter=max_iter)
        elif algorithm == "RoSE_Opt":
            from size_optimization import demo_roseopt

            ret = demo_roseopt.main("config_sram.yaml", problem=problem, max_iter=max_iter)
        elif algorithm == "MOEAD":
            from size_optimization import demo_moead

            circuit_mode = 1 if self.gen_unused_cells else 2
            ret = demo_moead.main("config_sram.yaml", problem=problem, max_iter=max_iter, circuit_mode=circuit_mode)
        elif algorithm == "MOBO":
            from size_optimization import demo_mobo

            circuit_mode = 1 if self.gen_unused_cells else 2
            ret = demo_mobo.main("config_sram.yaml", problem=problem, max_iter=max_iter, circuit_mode=circuit_mode)
        elif algorithm == "NSGA-II":
            from size_optimization import demo_nsgaii

            circuit_mode = 1 if self.gen_unused_cells else 2
            ret = demo_nsgaii.main("config_sram.yaml", problem=problem, max_iter=max_iter, circuit_mode=circuit_mode)
        elif algorithm == "tSS-BO":
            from size_optimization import demo_tssbo

            ret = demo_tssbo.main("config_sram.yaml", problem=problem, max_iter=max_iter)
        elif algorithm == "CMA-ES":
            from size_optimization import demo_cmaes

            ret = demo_cmaes.main("config_sram.yaml", problem=problem, max_iter=max_iter)
        elif algorithm == "CPN":
            from size_optimization import demo_cpn

            ret = demo_cpn.main("config_sram.yaml", problem=problem, max_iter=max_iter)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Parse results - handle both single-objective and multi-objective return formats
        if isinstance(ret, dict) and ret.get('is_multiobjective') and ret.get('pareto_solutions'):
            # Multi-objective algorithms (MOEAD / NSGA-II) return Pareto solutions
            best_fom = -1e9
            best_params = None
            best_result = None
            for sol in ret['pareto_solutions']:
                # Try nested result dict (MOEAD format)
                r = sol.get('result')
                if r is None:
                    # NSGA-II flat format: build result from top-level keys
                    r = {
                        'min_snm': sol.get('min_snm', 0),
                        'max_power': sol.get('max_power', 0),
                        'area': sol.get('area', 0),
                        'max_delay': sol.get('max_delay', 0),
                        'read_delay': sol.get('max_delay', 0),
                        'write_delay': sol.get('max_delay', 0),
                    }
                min_snm = r.get('min_snm', 0)
                max_power = r.get('max_power', 0)
                area = r.get('area', 0)
                max_delay = max(r.get('read_delay', 0), r.get('write_delay', 0))
                if min_snm > 0 and max_power > 0 and area > 0 and max_delay > 0:
                    fom_val = np.log10(min_snm / (max_power * np.sqrt(area) * max_delay))
                    if fom_val > best_fom:
                        best_fom = fom_val
                        best_result = r
                        best_params = sol.get('params')
                        if best_params is None:
                            # NSGA-II flat format: build params from top-level keys
                            best_params = {k: sol.get(k) for k in ['rows', 'cols', 'pu_width', 'pd_width', 'pg_width', 'length', 'nmos_model_name', 'pmos_model_name']}
            if best_fom <= -1e9:
                best_fom = None
        else:
            # Single-objective algorithms return params/merit/result directly
            best_params = ret.get("params") if isinstance(ret, dict) else None
            best_result = ret.get("result") if isinstance(ret, dict) else None
            best_fom = ret.get("merit") if isinstance(ret, dict) else None

        best_config = None
        if isinstance(best_params, dict) and ("rows" in best_params) and ("cols" in best_params):
            rows = best_params["rows"]
            cols = best_params["cols"]
            array_capacity = rows * cols
            if self.arch_space.total_bits % array_capacity == 0:
                num_arrays = self.arch_space.total_bits // array_capacity
                best_config = {"rows": rows, "cols": cols, "num_arrays": num_arrays, "array_capacity": array_capacity}

        self.joint_results = {
            "best_config": best_config,
            "best_params": best_params,
            "best_fom": best_fom,
            "best_result": best_result,
            "algorithm": algorithm,
        }
        return self.joint_results

    def _compute_pareto_frontier(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        计算二维点集（功耗-延时）的极小帕累托前沿。
        参数点为 (power, delay)，均越小越优。
        返回按 power 升序排序的前沿点集。
        """
        if not points:
            return []
        # 去重并按功耗升序、延时升序排序
        unique = list({(float(p), float(d)) for p, d in points})
        sorted_pts = sorted(unique, key=lambda x: (x[0], x[1]))
        frontier: List[Tuple[float, float]] = []
        min_delay = float("inf")
        for pwr, dly in sorted_pts:
            if dly <= min_delay:
                frontier.append((pwr, dly))
                min_delay = dly
        return frontier

    def _plot_power_delay_pareto_stage1(self, output_png_path: str):
        """
        绘制阶段1（架构优化）的功耗-延时散点与帕累托前沿，并保存到指定路径。
        目标函数参考：FOM = log10(1/ (max_power * sqrt(total_area) * max_delay))（最大化）。
        此处帕累托前沿按功耗与延时的极小解计算。
        """
        try:
            # 从迭代历史中提取有效数据
            points: List[Tuple[float, float]] = []
            for rec in getattr(self, "stage1_iteration_history", []):
                if rec.get("success") and ("total_power" in rec) and ("max_delay" in rec):
                    try:
                        pwr = float(rec["total_power"])
                        dly = float(rec["max_delay"])
                    except Exception:
                        continue
                    if pwr > 0 and dly > 0:
                        points.append((pwr, dly))

            if not points:
                print("No valid Stage 1 points to plot Pareto frontier.")
                return

            frontier = self._compute_pareto_frontier(points)

            # 绘图
            plt.figure(figsize=(8, 6))
            p_all = [p for p, d in points]
            d_all = [d for p, d in points]
            plt.scatter(p_all, d_all, s=24, c="tab:blue", alpha=0.6, label="All configs")

            if frontier:
                p_f = [p for p, d in frontier]
                d_f = [d for p, d in frontier]
                plt.plot(p_f, d_f, "-o", color="tab:red", linewidth=2, markersize=4, label="Pareto frontier")

            plt.xlabel("Total Power (W)")
            plt.ylabel("Max Delay (s)")
            plt.title("Stage 1 Power-Delay Pareto Frontier")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")
            plt.tight_layout()

            out_path = Path(output_png_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_path), dpi=200)
            plt.close()
            print(f"Stage 1 Pareto plot saved: {out_path}")
        except Exception as e:
            print(f"Error while plotting Stage 1 Pareto frontier: {e}")


def main():
    """
    Main function to run two-stage optimization
    运行二阶段优化的主函数
    """
    print("\n" + "=" * 60)
    print("Two-Stage SRAM Optimization")
    print("二阶段SRAM优化")
    print("=" * 60)

    mode_raw = input("Select mode (1=joint, 2=two-stage-fixed-configs, default=1): ").strip()
    mode = mode_raw if mode_raw in {"1", "2"} else "1"

    if mode == "2":
        print(f"Total capacity in use: total_bits={TOTAL_BITS}.")

        algorithm_raw = input("Select stage2 algorithm first (1=SA, 2=PSO, 3=RoSE_Opt, 4=CBO, 5=SMAC, default=1): ").strip()
        algorithm_map = {"1": "SA", "2": "PSO", "3": "RoSE_Opt", "4": "CBO", "5": "SMAC"}
        if algorithm_raw in algorithm_map:
            algorithm = algorithm_map[algorithm_raw]
        else:
            upper = algorithm_raw.replace("-", "_").upper()
            if upper in {"SA", "PSO", "SMAC", "CBO"}:
                algorithm = upper
            elif upper in {"ROSE_OPT", "ROSEOPT"}:
                algorithm = "RoSE_Opt"
            else:
                algorithm = "SA"
                print("Invalid algorithm input. Default to SA.")

        iter_raw = input("Enter iteration count (default=400): ").strip()
        try:
            max_iter = int(iter_raw) if iter_raw else 400
            if max_iter <= 0:
                raise ValueError
        except ValueError:
            max_iter = 400
            print("Invalid iteration input. Default to 400.")

        seed_raw = input("Enter seed (default=42): ").strip()
        try:
            seed = int(seed_raw) if seed_raw else 42
        except ValueError:
            seed = 42
            print("Invalid seed input. Default to 42.")

        fixed_configs = [
            {"rows": 128, "cols": 32, "num_arrays": TOTAL_BITS // (128 * 32), "array_capacity": 128 * 32},
            {"rows": 64, "cols": 32, "num_arrays": TOTAL_BITS // (64 * 32), "array_capacity": 64 * 32},
            {"rows": 32, "cols": 32, "num_arrays": TOTAL_BITS // (32 * 32), "array_capacity": 32 * 32},
            {"rows": 32, "cols": 16, "num_arrays": TOTAL_BITS // (32 * 16), "array_capacity": 32 * 16},
            {"rows": 16, "cols": 16, "num_arrays": TOTAL_BITS // (16 * 16), "array_capacity": 16 * 16},
        ]
        selected_raw = input("Select fixed config index (1-5) or 0 for all (default=0): ").strip()
        try:
            selected_index = int(selected_raw) if selected_raw else 0
        except ValueError:
            selected_index = 0
        if selected_index < 0 or selected_index > len(fixed_configs):
            selected_index = 0
        if selected_index == 0:
            run_configs = fixed_configs
        else:
            run_configs = [fixed_configs[selected_index - 1]]

        print("Using real 6T cell + peripheral circuits.")
        print(f"Stage2 algorithm: {algorithm}")
        print(f"Stage2 iterations: {max_iter}")
        optimizer = TwoStageOptimizer(seed=seed, gen_unused_cells=True)
        summaries = []
        for idx, cfg in enumerate(run_configs, 1):
            label = f"fixed{idx}_r{cfg['rows']}_c{cfg['cols']}_a{cfg['num_arrays']}"
            print(f"\nRunning Stage 2 for {cfg['rows']}x{cfg['cols']} x{cfg['num_arrays']} arrays")
            summary = optimizer.run_stage2_optimization_for_config(fixed_config=cfg, max_iter=max_iter, algorithm=algorithm, config_file="config_sram.yaml", label=label)
            summaries.append(summary)
        return summaries

    circuit_mode_raw = input("Select circuit mode (1=real, 2=equivalent, default=1): ").strip()
    try:
        circuit_mode = int(circuit_mode_raw) if circuit_mode_raw else 1
        if circuit_mode not in (1, 2):
            raise ValueError
    except Exception:
        circuit_mode = 1
        print("Invalid input. Default to 1.")
    gen_unused_cells = circuit_mode == 1
    if gen_unused_cells:
        print("Using real 6T cell + peripheral circuits.")
    else:
        print("Using equivalent 6T cell + peripheral circuits.")

    base_seed_input = input("Enter base seed (default=42): ").strip()
    try:
        base_seed = int(base_seed_input) if base_seed_input else 42
    except ValueError:
        base_seed = 42
        print("Invalid input. Default to 42.")

    num_seeds_input = input("Enter number of seeds (default=1): ").strip()
    try:
        num_seeds = int(num_seeds_input) if num_seeds_input else 1
        if num_seeds < 1:
            raise ValueError
    except ValueError:
        num_seeds = 1
        print("Invalid input. Default to 1.")

    max_iter_input = input("Enter the maximum number of iterations for joint optimization (default=40): ").strip()
    try:
        max_iter = int(max_iter_input) if max_iter_input else 40
    except ValueError:
        max_iter = 40
        print("Invalid input. Default to 40.")

    print(f"Maximum number of iterations for joint optimization: {max_iter}")

    algorithm_raw = input("Select algorithm for joint optimization (1=SA, 2=PSO, 3=SMAC, 4=CBO, 5=RoSE_Opt, 6=MOEAD, 7=MOBO, 8=NSGA-II, 9=tSS-BO, 10=CMA-ES, 11=CPN, default=1): ").strip()
    algorithm_norm = algorithm_raw.replace("-", "_").strip()
    algorithm_map = {"1": "SA", "2": "PSO", "3": "SMAC", "4": "CBO", "5": "RoSE_Opt", "6": "MOEAD", "7": "MOBO", "8": "NSGA-II", "9": "tSS-BO", "10": "CMA-ES", "11": "CPN"}
    if not algorithm_norm:
        algorithm = "SA"
    elif algorithm_norm in algorithm_map:
        algorithm = algorithm_map[algorithm_norm]
    else:
        upper = algorithm_norm.upper()
        if upper in {"SA", "PSO", "SMAC", "CBO"}:
            algorithm = upper
        elif upper in {"ROSE_OPT", "ROSEOPT"}:
            algorithm = "RoSE_Opt"
        elif upper in {"MOEAD"}:
            algorithm = "MOEAD"
        elif upper in {"MOBO"}:
            algorithm = "MOBO"
        elif upper in {"NSGA_II", "NSGAII", "NSGA-II"}:
            algorithm = "NSGA-II"
        elif upper in {"TSS_BO", "TSSBO", "TSS-BO"}:
            algorithm = "tSS-BO"
        elif upper in {"CMA_ES", "CMAES", "CMA-ES"}:
            algorithm = "CMA-ES"
        elif upper in {"CPN", "TABPFN"}:
            algorithm = "CPN"
        else:
            algorithm = "SA"
            print("Invalid algorithm input. Default to SA.")

    print(f"Joint optimization algorithm: {algorithm}")

    results = []
    for seed_offset in range(num_seeds):
        seed_value = base_seed + seed_offset
        print(f"\nRunning joint optimization with seed={seed_value} ({seed_offset + 1}/{num_seeds})")
        optimizer = TwoStageOptimizer(seed=seed_value, gen_unused_cells=gen_unused_cells)
        run_result = optimizer.run_joint_optimization(max_iter=max_iter, algorithm=algorithm)
        results.append(run_result)

    return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    main()
