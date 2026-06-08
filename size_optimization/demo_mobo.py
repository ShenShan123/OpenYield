"""
SRAM Circuit Optimization using MOBO (Multi-Objective Bayesian Optimization)
使用MOBO算法的SRAM电路多目标优化
优化目标: SNM(最大化), 功耗(最小化), 延迟(最小化), 面积(最小化)
8维输入: row_idx, col_idx, pu_width, pd_width, pg_width, length, nmos_model_idx, pmos_model_idx
"""

import os
import sys
import time
import numpy as np
import warnings
import csv
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

# Tee类：同时输出到终端和文件
class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Import path handling
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import utilities from exp_utils
from size_optimization.exp_utils import (
    seed_set, create_directories, evaluate_sram, ModifiedSRAMParameterSpace,
    OptimizationLogger, get_default_normalized_vector
)
from utils import estimate_bitcell_area

# Import MOBO module
# 假设 MOBO 代码在 size_optimization/MOBO 下
# from size_optimization.MOBO.bayesian_optimizer import BayesianOptimizer
sys.path.append(os.path.join(project_root, 'size_optimization', 'MOBO'))
from bayesian_optimizer import BayesianOptimizer
# Problem 类通常在算法模块中定义，如果 MOBO 没有自带 Problem 类，我们需要自己定义一个简单的
# 检查发现 MOBO 的 BayesianOptimizer 接受一个 problem 对象，该对象需要有 num_of_variables, objectives, variables_range 等属性
# 我们可以复用 moead.py 中的 Problem 定义，或者在这里定义一个简单的类

class Problem:
    def __init__(self, objectives, num_of_variables, variables_range, variables_type=None, expand=True, same_range=False, default_features=None):
        self.objectives = objectives
        self.num_of_variables = num_of_variables
        self.variables_range = variables_range
        self.variables_type = variables_type
        self.expand = expand
        self.same_range = same_range
        self.default_features = default_features
        self._first_individual = True
        
    def generate_individual(self):
        # 简单模拟 MOEAD 中的 Individual
        class Individual:
            def __init__(self, features):
                self.features = features

        if self._first_individual and self.default_features is not None:
            self._first_individual = False
            features = list(self.default_features)
        else:
            features = []
            for i in range(self.num_of_variables):
                lo, hi = self.variables_range[i]
                if self.variables_type and self.variables_type[i] == 'int':
                    features.append(np.random.randint(lo, hi + 1))
                else:
                    features.append(np.random.uniform(lo, hi))
        return Individual(features)
        
    @property
    def num_of_objectives(self):
        return len(self.objectives)


def main(config_path="config_sram.yaml", problem=None, max_iter=None, circuit_mode=1):
    """
    Main function for MOBO joint optimization
    MOBO联合优化主函数
    
    Parameters:
        config_path: Path to SRAM configuration file
        problem: Problem object tuple (parameter_space, eval_fn, constraints_fn) from experiment.py (not used directly here)
        max_iter: Maximum number of iterations
        circuit_mode: Circuit simulation mode
                     1 = Real Circuit
                     2 = Equivalent Model
    """
    # 设置日志文件
    # log_dir = Path(project_root) / "results" / "mobo"
    # log_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = Path(project_root) / "size_optimization" / "experiment" / "MOBO"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保基础 sim 目录存在
    create_directories()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # MOBO特定参数
    n_iterations = max_iter if max_iter is not None else 50
    n_initial = min(20, n_iterations // 2) # 初始采样
    
    # Circuit mode configuration
    gen_unused_cells = (circuit_mode == 1)
    
    # ---- 检测是否由 experiment.py 传入外部 problem ----
    _external_problem = problem
    _use_external = (_external_problem is not None
                     and isinstance(_external_problem, (tuple, list))
                     and len(_external_problem) >= 2)

    import torch

    print("=" * 70)
    print("SRAM Joint Multi-Objective Optimization using MOBO")
    print("SRAM联合多目标优化 - 使用MOBO算法")
    print("4目标: SNM, Power, Delay, Area")
    print("=" * 70)

    if _use_external:
        # ---- 外部参数空间模式（含外围电路） ----
        _ext_param_space = _external_problem[0]
        _ext_eval_fn = _external_problem[1]
        _total_dim = int(_ext_param_space.bounds.shape[1])

        variables_range = [
            (float(_ext_param_space.bounds[0, i]), float(_ext_param_space.bounds[1, i]))
            for i in range(_total_dim)
        ]
        variables_type = ['float'] * _total_dim
        default_features = [
            (float(_ext_param_space.bounds[0, i]) + float(_ext_param_space.bounds[1, i])) / 2.0
            for i in range(_total_dim)
        ]

        # 全量缓存: key -> (4-obj list, result dict, success bool)
        _ext_results_cache = {}

        def _eval_ext_full(*args):
            x = list(args)
            key = tuple(round(v, 10) for v in x)
            if key in _ext_results_cache:
                return _ext_results_cache[key]
            objectives_ret, constraints_ret, result_ret, success_ret = _ext_eval_fn(x)
            if success_ret and result_ret:
                _min_snm = result_ret.get('min_snm', 0.0)
                _max_power = result_ret.get('max_power', 1e-3)
                _area = result_ret.get('area', 0.0)
                _max_delay = max(result_ret.get('read_delay', 0), result_ret.get('write_delay', 0))
                objs = [-_min_snm, _max_power, _area, _max_delay]
            else:
                objs = [0.0, 10.0, 5e-6, 5e-8]
            _ext_results_cache[key] = (objs, result_ret, success_ret)
            return (objs, result_ret, success_ret)

        _ext_obj_cache = {'key': None, 'objs': None}

        def _eval_ext_objs(*args):
            x = list(args)
            key = tuple(round(v, 10) for v in x)
            if _ext_obj_cache['key'] == key:
                return _ext_obj_cache['objs']
            objs, _, _ = _eval_ext_full(*args)
            _ext_obj_cache['key'] = key
            _ext_obj_cache['objs'] = objs
            return objs

        f1_snm   = lambda *a: _eval_ext_objs(*a)[0]
        f2_power = lambda *a: _eval_ext_objs(*a)[1]
        f3_area  = lambda *a: _eval_ext_objs(*a)[2]
        f4_delay = lambda *a: _eval_ext_objs(*a)[3]

        print(f"\n使用外部参数空间（包含外围电路优化），维度: {_total_dim}")

    else:
        # ---- 原始内部8维模式 ----
        from size_optimization.experiment import ArchitectureConfigurationSpace

        arch_space = ArchitectureConfigurationSpace()
        param_space = ModifiedSRAMParameterSpace(config_path)
        default_vector = get_default_normalized_vector(param_space)

        def _model_index(param_name, fallback=0):
            dim_idx = 0
            for name, info in param_space.param_info.items():
                if info["type"] == "continuous_list":
                    dim_idx += info["count"]
                elif info["type"] == "continuous_scalar":
                    dim_idx += 1
                elif info["type"] == "categorical_list":
                    if name == param_name:
                        choices = info.get("choices", [])
                        if not choices or dim_idx >= len(default_vector):
                            return fallback
                        idx = int(default_vector[dim_idx] * len(choices))
                        return max(0, min(idx, len(choices) - 1))
                    dim_idx += info["count"]
            return fallback

        nmos_idx = _model_index("nmos_model", 0)
        pmos_idx = _model_index("pmos_model", 0)
        default_features = [
            0, 0,
            default_vector[0] if len(default_vector) > 0 else 0.0,
            default_vector[1] if len(default_vector) > 1 else 0.0,
            default_vector[2] if len(default_vector) > 2 else 0.0,
            default_vector[3] if len(default_vector) > 3 else 0.0,
            nmos_idx, pmos_idx,
        ]

        # 创建评估缓存
        evaluation_counter = {'count': 0}
        iteration_csv_tag = f"MOBO_{n_iterations}_{timestamp}"
        iteration_csv_path = experiment_dir / f"stage2_iterations_live_joint_{iteration_csv_tag}.csv"
        iteration_csv_fields = [
            "iteration", "fom", "success", "rows", "cols", "num_arrays",
            "pg_width", "pd_width", "pu_width", "length",
            "nmos_model_name", "pmos_model_name",
            "min_snm", "hold_snm", "read_snm", "write_snm",
            "read_delay", "write_delay", "max_delay",
            "read_power", "write_power", "max_power", "total_power",
            "single_array_area", "total_area",
            "objective_0", "objective_1", "objective_2", "objective_3",
            "constraint_0", "constraint_1",
        ]

        def _append_iteration_csv(record):
            if not record.get("success", False):
                return
            write_header = not iteration_csv_path.exists()
            with open(iteration_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=iteration_csv_fields)
                if write_header:
                    writer.writeheader()
                writer.writerow({k: record.get(k, "") for k in iteration_csv_fields})

        def evaluate_joint(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx):
            x_norm = torch.tensor([
                pu_width_norm, pd_width_norm, pg_width_norm, length_norm,
                nmos_idx, nmos_idx, pmos_idx
            ], dtype=torch.float32)
            params = param_space.convert_params(x_norm)

            rows = arch_space.row_choices[int(row_idx)]
            cols = arch_space.column_choices[int(col_idx)]
            num_arrays = arch_space.total_bits // (rows * cols)

            from size_optimization.experiment import evaluate_sram_with_config
            params_dict = {
                'pu_width': params['pu_width'], 'pd_width': params['pd_width'],
                'pg_width': params['pg_width'], 'length': params['length'],
                'nmos_model_name': params['nmos_model_name'],
                'pmos_model_name': params['pmos_model_name'],
            }
            objectives, constraints, result, success = evaluate_sram_with_config(
                params_dict, num_rows=rows, num_cols=cols, num_arrays=num_arrays,
                stage_label="joint/MOBO", gen_unused_cells=gen_unused_cells
            )

            iteration_num = evaluation_counter["count"]
            evaluation_counter["count"] += 1

            fom = -10.0
            if success and result:
                _md = max(result.get("read_delay", 0), result.get("write_delay", 0))
                _a = result.get("area", 0)
                _ms = result.get("min_snm", 0)
                _mp = result.get("max_power", 0)
                if _ms > 0 and _mp > 0 and _a > 0 and _md > 0:
                    fom = np.log10(_ms / (_mp * np.sqrt(_a) * _md))

            record = {
                "iteration": iteration_num, "fom": fom, "success": success,
                "rows": rows, "cols": cols, "num_arrays": num_arrays,
                "pg_width": params.get("pg_width"), "pd_width": params.get("pd_width"),
                "pu_width": params.get("pu_width"), "length": params.get("length"),
                "nmos_model_name": params.get("nmos_model_name"),
                "pmos_model_name": params.get("pmos_model_name"),
            }
            if success and result:
                total_power = result.get("total_power", result.get("max_power", 0))
                single_area = result.get("area", 0)
                total_area = result.get("total_area", single_area * num_arrays if single_area else 0)
                record.update({
                    "min_snm": result.get("min_snm", 0), "hold_snm": result.get("hold_snm", 0),
                    "read_snm": result.get("read_snm", 0), "write_snm": result.get("write_snm", 0),
                    "read_delay": result.get("read_delay", 0), "write_delay": result.get("write_delay", 0),
                    "max_delay": max(result.get("read_delay", 0), result.get("write_delay", 0)),
                    "read_power": result.get("read_power", 0), "write_power": result.get("write_power", 0),
                    "max_power": result.get("max_power", 0), "total_power": total_power,
                    "single_array_area": single_area, "total_area": total_area,
                })
            if objectives:
                for oi in range(min(4, len(objectives))):
                    record[f"objective_{oi}"] = objectives[oi]
            if constraints:
                for ci in range(min(2, len(constraints))):
                    record[f"constraint_{ci}"] = constraints[ci]

            _append_iteration_csv(record)
            return objectives, constraints, result, success

        # 缓存机制
        last_eval_params = {'params': None, 'results': None}

        def get_objectives_for_params(row_idx, col_idx, pu_w, pd_w, pg_w, l_n, nm_i, pm_i):
            params_key = (int(row_idx), int(col_idx), pu_w, pd_w, pg_w, l_n, int(nm_i), int(pm_i))
            if last_eval_params['params'] == params_key:
                return last_eval_params['results']
            objectives, constraints, result, success = evaluate_joint(
                row_idx, col_idx, pu_w, pd_w, pg_w, l_n, nm_i, pm_i
            )
            if success and result:
                results = [
                    -result.get('min_snm', 0.0),
                    result.get('max_power', 1e-3),
                    result.get('total_area', result.get('area', 0)),
                    max(result.get('read_delay', 0), result.get('write_delay', 0)),
                ]
            else:
                results = [0.0, 10.0, 5e-6, 5e-8]
            last_eval_params['params'] = params_key
            last_eval_params['results'] = results
            return results

        f1_snm   = lambda *a: get_objectives_for_params(*a)[0]
        f2_power = lambda *a: get_objectives_for_params(*a)[1]
        f3_area  = lambda *a: get_objectives_for_params(*a)[2]
        f4_delay = lambda *a: get_objectives_for_params(*a)[3]

        variables_range = [
            (0, len(arch_space.row_choices) - 1),
            (0, len(arch_space.column_choices) - 1),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            (0, 2), (0, 2),
        ]
        variables_type = ['int', 'int', 'float', 'float', 'float', 'float', 'int', 'int']

    # ---- 公共部分: 创建 Problem, 运行 MOBO, 提取 Pareto ----
    mobo_problem = Problem(
        objectives=[f1_snm, f2_power, f3_area, f4_delay],
        num_of_variables=len(variables_range),
        variables_range=variables_range,
        variables_type=variables_type,
        expand=True,
        same_range=False,
        default_features=default_features,
    )

    mobo = BayesianOptimizer(
        problem=mobo_problem,
        n_initial=n_initial,
        n_iterations=n_iterations - n_initial if n_iterations > n_initial else 0,
        n_candidates_per_iter=1000,
        mc_samples=50,
    )

    print(f"\n开始 MOBO 优化...")
    pareto_individuals = mobo.optimize()

    # ---- 保存 Pareto CSV ----
    output_dir = experiment_dir
    pareto_solutions = []

    for idx, ind in enumerate(pareto_individuals):
        state = ind.features

        if _use_external:
            params = _ext_param_space.convert_params(torch.tensor(state, dtype=torch.float32))
            rows = params.get('rows', 0)
            cols = params.get('cols', 0)
            _cache_key = tuple(round(float(v), 10) for v in state)
            if _cache_key in _ext_results_cache:
                _, result, _ = _ext_results_cache[_cache_key]
            else:
                _, result, _ = _eval_ext_full(*state)
        else:
            row_idx = int(state[0])
            col_idx = int(state[1])
            rows = arch_space.row_choices[row_idx]
            cols = arch_space.column_choices[col_idx]
            x_tensor = torch.tensor(
                [state[2], state[3], state[4], state[5], state[6], state[6], state[7]],
                dtype=torch.float32,
            )
            params = param_space.convert_params(x_tensor)
            _, _, result, _ = evaluate_joint(
                state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]
            )

        min_snm = result.get('min_snm', 0) if result else 0
        max_power = result.get('max_power', 0) if result else 0
        area = result.get('total_area', result.get('area', 0)) if result else 0
        max_delay = result.get('max_delay', 0) if result else 0

        sol = {
            'solution_id': idx + 1,
            'rows': rows, 'cols': cols,
            'pg_width': params.get("pg_width"),
            'pd_width': params.get("pd_width"),
            'pu_width': params.get("pu_width"),
            'length': params.get("length"),
            'nmos_model_name': params.get("nmos_model_name"),
            'pmos_model_name': params.get("pmos_model_name"),
            'min_snm': min_snm, 'max_power': max_power,
            'area': area, 'max_delay': max_delay,
        }
        pareto_solutions.append(sol)

    csv_path = output_dir / f"pareto_solutions_{timestamp}.csv"
    fieldnames = [
        "solution_id", "rows", "cols",
        "pg_width", "pd_width", "pu_width", "length",
        "nmos_model_name", "pmos_model_name",
        "min_snm", "max_power", "area", "max_delay",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in pareto_solutions:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    print(f"MOBO pareto CSV saved to: {csv_path}")

    return {
        'algorithm': 'MOBO',
        'is_multiobjective': True,
        'pareto_solutions': pareto_solutions,
        'num_solutions': len(pareto_solutions),
        'iteration': n_iterations,
    }

if __name__ == "__main__":
    import argparse
    SEED = 1
    seed_set(seed=SEED)
    parser = argparse.ArgumentParser(description='MOBO多目标优化')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--circuit_mode', type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    main(max_iter=args.max_iter, circuit_mode=args.circuit_mode)
