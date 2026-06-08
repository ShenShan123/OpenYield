"""
SRAM Circuit Optimization using MOEAD Algorithm
使用MOEAD算法的SRAM电路多目标优化
优化目标: SNM(最大化), 功耗(最小化), 面积(最小化), 延时(最小化)
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
if project_root not in sys.path:
    sys.path.append(project_root)
# moead subpackage lives under size_optimization/
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import utilities from exp_utils
from size_optimization.exp_utils import (
    seed_set, create_directories, evaluate_sram, ModifiedSRAMParameterSpace,
    OptimizationLogger, get_default_transistor_features,
    get_peripheral_params_from_yaml, PERIPHERAL_ALL_KEYS,
)
from utils import estimate_bitcell_area

# Import MOEAD module
from moead import MOEAD, Problem


def main(config_path="config_sram.yaml", problem=None, max_iter=None, circuit_mode=1, h=17, T_size=None):
    """
    Main function for MOEAD joint optimization
    MOEAD联合优化主函数
    
    Parameters:
        config_path: Path to SRAM configuration file
        problem: Problem object tuple (parameter_space, eval_fn, constraints_fn) from experiment.py
        max_iter: Maximum number of iterations (maps to max_gen for MOEAD)
        circuit_mode: Circuit simulation mode
                     1 = 真实电路 (Real Circuit, 使用SPICE仿真, 慢但准确)
                     2 = 等效模型 (Equivalent Model, 使用代理模型, 快但近似)
        h: 权重向量划分数 (控制Pareto前沿解的数量)
        T_size: 邻域大小 (如果为None则根据h自动设置)
    
    Note: This function performs joint optimization of both architecture and transistor parameters
    注意: 此函数同时优化架构参数和晶体管参数
    """
    # Save external problem before local variable shadows it
    _external_problem = problem
    _use_external = (_external_problem is not None
                     and isinstance(_external_problem, (tuple, list))
                     and len(_external_problem) >= 2)

    # 设置日志文件
    log_dir = Path(project_root) / "results" / "moead"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"moead_output_{timestamp}.txt"
    
    # 保存原始stdout并设置Tee输出
    original_stdout = sys.stdout
    log_fileobj = open(log_file, 'w', encoding='utf-8')
    sys.stdout = Tee(original_stdout, log_fileobj)
    
    # MOEAD特定参数
    max_gen = max_iter if max_iter is not None else 800
    if T_size is None:
        # 根据h自动设置T_size
        if h <= 2:
            T_size = 2
        elif h <= 4:
            T_size = 3
        elif h <= 9:
            T_size = 5
        else:
            T_size = 20
    
    # Circuit mode configuration
    # 电路模式配置
    gen_unused_cells = (circuit_mode == 1)
    
    print("=" * 70)
    print("SRAM Joint Multi-Objective Optimization using MOEAD")
    print("SRAM联合多目标优化 - 使用MOEAD算法")
    print("8输入: row_idx, col_idx + 6个晶体管参数")
    print("4目标: SNM, Power, Area, Delay")
    print("=" * 70)
    print(f"\n算法参数:")
    print(f"  最大迭代次数: {max_gen}")
    print(f"  权重向量划分数 h: {h}")
    print(f"  邻域大小 T_size: {T_size}")
    print(f"  预计Pareto前沿解数量: {(h+1)*(h+2)*(h+3)//6}")
    print(f"\n电路仿真模式:")
    if gen_unused_cells:
        print(f"  模式: 真实电路 (Real Circuit)")
        print(f"  特点: SPICE仿真, 慢但准确")
    else:
        print(f"  模式: 等效模型 (Equivalent Model)")
        print(f"  特点: 代理模型, 快速近似")

    
    # Import architecture configuration space from experiment.py
    from size_optimization.experiment import ArchitectureConfigurationSpace, ModifiedSRAMParameterSpace
    
    arch_space = ArchitectureConfigurationSpace()
    param_space = ModifiedSRAMParameterSpace(config_path)
    
    print(f"\n联合优化配置:")
    print(f"  架构选择: {len(arch_space.row_choices)} rows × {len(arch_space.column_choices)} cols = {arch_space.get_num_configurations()} 组合")
    print(f"  Row选项: {arch_space.row_choices}")
    print(f"  Column选项: {arch_space.column_choices}")
    print(f"  总容量: {arch_space.total_bits} bits (32KB)")
    print(f"  晶体管参数: 6维连续+离散空间")
    
    # 创建评估缓存,避免同一参数组合重复仿真
    evaluation_cache = {}
    evaluation_counter = {'count': 0, 'total': '?'}  # 评估计数器，先用?占位
    iteration_csv_tag = f"MOEAD_{max_gen}_h{h}_{timestamp}"
    experiment_dir = Path(project_root) / "size_optimization" / "experiment" / "MOEAD"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    iteration_csv_path = experiment_dir / f"stage2_iterations_live_joint_{iteration_csv_tag}.csv"
    iteration_csv_fields = [
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
    # 加载外围电路默认参数，用于每次评估
    _peripheral_defaults = get_peripheral_params_from_yaml()

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
        """
        联合优化评估函数: 8维输入 → 4个输出
        输入: 2个架构索引 + 4个标准化参数[0,1] + 2个模型索引
        输出: (objectives, constraints, result, success)
        """
        # 将[0,1]标准化参数映射到物理范围
        # convert_params期望: [pu, pd, pg, length, nmos_pd_idx, nmos_pg_idx, pmos_pu_idx]
        # 我们简化为pd和pg共用同一个nmos模型索引
        import torch
        x_norm = torch.tensor([
            pu_width_norm,    # pmos_width[pu]
            pd_width_norm,    # nmos_width[pd]
            pg_width_norm,    # nmos_width[pg]
            length_norm,      # length
            nmos_idx,         # nmos_model[pd] 索引
            nmos_idx,         # nmos_model[pg] 索引（与pd共用）
            pmos_idx          # pmos_model[pu] 索引
        ], dtype=torch.float32)
        params = param_space.convert_params(x_norm)
        
        # convert_params已经返回了所有需要的参数，包括nmos_model_name和pmos_model_name
        pu_width = params['pu_width']
        pd_width = params['pd_width']
        pg_width = params['pg_width']
        length = params['length']
        nmos_model_name = params['nmos_model_name']
        pmos_model_name = params['pmos_model_name']
        
        # 解码架构参数
        rows = arch_space.row_choices[int(row_idx)]
        cols = arch_space.column_choices[int(col_idx)]
        array_capacity = rows * cols
        num_arrays = arch_space.total_bits // array_capacity
        
        # 调用实际评估
        from size_optimization.experiment import evaluate_sram_with_config
        
        params_dict = dict(_peripheral_defaults)
        params_dict.update({
            'pu_width': pu_width,
            'pd_width': pd_width,
            'pg_width': pg_width,
            'length': length,
            'nmos_model_name': nmos_model_name,  # 直接使用convert_params返回的模型名
            'pmos_model_name': pmos_model_name   # 直接使用convert_params返回的模型名
        })
        
        objectives, constraints, result, success = evaluate_sram_with_config(
            params_dict,
            num_rows=rows,
            num_cols=cols,
            num_arrays=num_arrays,
            stage_label="joint/MOEAD",
            gen_unused_cells=gen_unused_cells
        )
        iteration_num = evaluation_counter["count"]
        evaluation_counter["count"] += 1
        fom = -10.0
        if success and result:
            max_delay = max(result.get("read_delay", 0), result.get("write_delay", 0))
            area = result.get("area", 0)
            min_snm = result.get("min_snm", 0)
            max_power = result.get("max_power", 0)
            if min_snm > 0 and max_power > 0 and area > 0 and max_delay > 0:
                fom = np.log10(min_snm / (max_power * np.sqrt(area) * max_delay))
        record = {
            "iteration": iteration_num,
            "fom": fom,
            "success": success,
            "rows": rows,
            "cols": cols,
            "num_arrays": num_arrays,
            "pg_width": params.get("pg_width"),
            "pd_width": params.get("pd_width"),
            "pu_width": params.get("pu_width"),
            "length": params.get("length"),
            "nmos_model_name": params.get("nmos_model_name"),
            "pmos_model_name": params.get("pmos_model_name"),
        }
        # 添加外围电路参数到记录
        for pk in PERIPHERAL_ALL_KEYS:
            record[pk] = params_dict.get(pk, "")
        if success and result:
            record.update(
                {
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
                    "total_power": result.get("total_power", result.get("max_power", 0)),
                    "single_array_area": result.get("area", 0),
                    "total_area": result.get("total_area", result.get("area", 0)),
                }
            )
        if objectives:
            record["objective_0"] = objectives[0] if len(objectives) > 0 else 0
            record["objective_1"] = objectives[1] if len(objectives) > 1 else 0
            record["objective_2"] = objectives[2] if len(objectives) > 2 else 0
            record["objective_3"] = objectives[3] if len(objectives) > 3 else 0
        if constraints:
            record["constraint_0"] = constraints[0] if len(constraints) > 0 else 0
            record["constraint_1"] = constraints[1] if len(constraints) > 1 else 0
        _append_iteration_csv(record)

        return objectives, constraints, result, success
    
    # Create unified objective function that returns all 4 objectives in one evaluation
    # 创建统一的目标函数，在一次评估中返回所有4个目标值
    
    def evaluate_multi_objectives(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx):
        """
        联合优化统一评估函数: 一次仿真返回4个目标值（输入为标准化参数）
        返回: [obj1_snm, obj2_power, obj3_area, obj4_delay]
        """
        objectives, constraints, result, success = evaluate_joint(
            row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx
        )
        
        if success and result:
            # 从result中提取4个目标值（字段名与experiment.py保持一致）
            min_snm = result.get('min_snm', 0.0)
            max_power = result.get('max_power', 1e-3)  # max_power相当于total_power
            area = result.get('area', 0.0)
            max_delay = result.get('max_delay', 1e-9)  # 直接使用result中的max_delay
            
            # 4个目标 (都是最小化)
            obj1 = -min_snm      # 目标1: 最小化负SNM = 最大化SNM
            obj2 = max_power     # 目标2: 最小化功耗（max_power）
            obj3 = area          # 目标3: 最小化面积
            obj4 = max_delay     # 目标4: 最小化延时（max_delay）
            
            return [obj1, obj2, obj3, obj4]
        else:
            # Penalty values for failed evaluation:
            # [-min_snm, max_power, area, max_delay] all minimized
            # SNM=0(worst), power=1mW(bad >>1-100uW), area=1e-3m2(bad >>1e-8m2), delay=10us(bad >>1-5ns)
            return [0.0, 10.0, 5e-6, 5e-8]
    
    # 使用闭包缓存评估结果，避免同一参数重复评估
    last_eval_params = {'params': None, 'results': None}
    
    def get_objectives_for_params(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx):
        """获取或计算8维参数对应的4个目标值（输入为标准化参数）"""
        params_key = (int(row_idx), int(col_idx), pu_width_norm, pd_width_norm, pg_width_norm, length_norm, int(nmos_idx), int(pmos_idx))
        
        # 如果参数相同，直接返回缓存的结果
        if last_eval_params['params'] == params_key:
            return last_eval_params['results']
        
        # 否则重新评估
        results = evaluate_multi_objectives(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx)
        last_eval_params['params'] = params_key
        last_eval_params['results'] = results
        return results
    
    def f1_snm(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx):
        """目标1: 最小化 -SNM (即最大化SNM)"""
        return get_objectives_for_params(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx)[0]
    
    def f2_power(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx):
        """目标2: 最小化 Power"""
        return get_objectives_for_params(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx)[1]
    
    def f3_area(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx):
        """目标3: 最小化 Area"""
        return get_objectives_for_params(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx)[2]
    
    def f4_delay(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx):
        """目标4: 最小化 Delay"""
        return get_objectives_for_params(row_idx, col_idx, pu_width_norm, pd_width_norm, pg_width_norm, length_norm, nmos_idx, pmos_idx)[3]
    
    # Define parameter ranges for joint optimization (8维输入)
    # 使用[0,1]标准化空间（与SA/PSO/CBO等算法一致），评估时映射到物理范围
    # 2 architecture discrete variables + 4 normalized continuous + 2 discrete model indices
    
    variables_range = [
        (0, len(arch_space.row_choices) - 1),     # row_idx: 0-5 (离散)
        (0, len(arch_space.column_choices) - 1),  # col_idx: 0-5 (离散)
        (0.0, 1.0),  # pu_width_norm: [0,1]标准化空间
        (0.0, 1.0),  # pd_width_norm: [0,1]标准化空间
        (0.0, 1.0),  # pg_width_norm: [0,1]标准化空间
        (0.0, 1.0),  # length_norm: [0,1]标准化空间
        (0, 2),      # nmos_model_idx: 0=VTL, 1=VTG, 2=VTH
        (0, 2)       # pmos_model_idx: 0=VTL, 1=VTG, 2=VTH
    ]
    
    # 获取物理参数范围用于输出信息
    physical_bounds = param_space.get_physical_bounds()
    print(f"\n参数空间配置（使用[0,1]标准化，与其他算法一致）:")
    
    # 处理physical_bounds中可能的非数值类型
    try:
        pu_lower, pu_upper = float(physical_bounds[0][0]), float(physical_bounds[0][1])
        pd_lower, pd_upper = float(physical_bounds[1][0]), float(physical_bounds[1][1])
        pg_lower, pg_upper = float(physical_bounds[2][0]), float(physical_bounds[2][1])
        length_lower, length_upper = float(physical_bounds[3][0]), float(physical_bounds[3][1])
        
        print(f"  pu_width: [0,1] → [{pu_lower*1e9:.1f}nm, {pu_upper*1e9:.1f}nm]")
        print(f"  pd_width: [0,1] → [{pd_lower*1e9:.1f}nm, {pd_upper*1e9:.1f}nm]")
        print(f"  pg_width: [0,1] → [{pg_lower*1e9:.1f}nm, {pg_upper*1e9:.1f}nm]")
        print(f"  length: [0,1] → [{length_lower*1e9:.1f}nm, {length_upper*1e9:.1f}nm]")
        print(f"  nmos_model: 索引 {physical_bounds[4]}")
        print(f"  pmos_model: 索引 {physical_bounds[5]}")
    except (TypeError, ValueError) as e:
        print(f"  (参数范围获取失败: {e})")
    
    variables_type = ['int', 'int', 'float', 'float', 'float', 'float', 'int', 'int']
    default_vector, nmos_idx, pmos_idx = get_default_transistor_features(param_space)
    row_idx = arch_space.row_choices.index(16) if 16 in arch_space.row_choices else 0
    col_idx = arch_space.column_choices.index(16) if 16 in arch_space.column_choices else 0
    default_features = [
        row_idx,
        col_idx,
        default_vector[0],
        default_vector[1],
        default_vector[2],
        default_vector[3],
        nmos_idx,
        pmos_idx,
    ]
    
    # 当外部problem提供时, 使用外部参数空间和评估函数(含外围电路参数)
    if _use_external:
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

        f1_snm = lambda *a: _eval_ext_objs(*a)[0]
        f2_power = lambda *a: _eval_ext_objs(*a)[1]
        f3_area = lambda *a: _eval_ext_objs(*a)[2]
        f4_delay = lambda *a: _eval_ext_objs(*a)[3]

        print(f"\n使用外部参数空间（包含外围电路优化），维度: {_total_dim}")

    # Create Problem with 4 objectives
    moead_problem = Problem(
        objectives=[f1_snm, f2_power, f3_area, f4_delay],
        num_of_variables=len(variables_range),
        variables_range=variables_range,
        variables_type=variables_type,
        expand=True,
        same_range=False,
        default_features=default_features
    )
    
    print(f"\n优化问题定义:")
    print(f"  变量数: {moead_problem.num_of_variables}")
    print(f"  目标数: {moead_problem.num_of_objectives}")
    print(f"  目标1: 最小化 -SNM (最大化SNM)")
    print(f"  目标2: 最小化 Power")
    print(f"  目标3: 最小化 Area")
    print(f"  目标4: 最小化 Delay")
    
    # Create MOEAD optimizer
    moead = MOEAD(
        problem=moead_problem,
        h=h,
        max_gen=max_gen,
        T_size=T_size,
        name=f'sram_joint_4obj_h{h}'
    )
    
    # 计算种群大小
    # 4目标: Pop_size = C(h+3, 3) = (h+1)×(h+2)×(h+3) / 6
    m = moead_problem.num_of_objectives  # 4个目标
    Pop_size = (h + 1) * (h + 2) * (h + 3) // 6
    
    # 计算预期总评估次数
    # MOEAD评估次数 = 初始种群 + 每代进化
    # = Pop_size + max_gen × Pop_size = Pop_size × (1 + max_gen)
    expected_total = Pop_size * (1 + max_gen)
    evaluation_counter['total'] = expected_total  # 添加total字段供evaluate_once使用
    
    print(f"{'='*70}")
    print(f"MOEAD优化器初始化")
    print(f"问题: sram_joint_4obj_h{h}")
    print(f"变量维度: {moead_problem.num_of_variables}维联合搜索空间")
    print(f"目标数量: {m} (4目标Pareto优化)")
    print(f"权重向量分割数H: {h}")
    print(f"计算得种群大小: {Pop_size} = ({h}+1)×({h}+2)×({h}+3)/6")
    print(f"最大迭代次数: {max_gen}")
    print(f"邻域大小: {T_size}")
    print(f"{'='*70}")
    
    # Run optimization
    print(f"\n开始联合优化...")
    print(f"  初始化中...")
    print(f"  种群大小: {Pop_size}")
    print(f"  迭代次数: {max_gen}")
    print(f"  邻域大小: {T_size}")
    print(f"  说明: 4目标优化时,h={h}生成{Pop_size}个权重向量")
    print(f"  预计总评估次数: {expected_total} = {Pop_size} (初始) + {max_gen} × {Pop_size} (进化)")
    print(f"  注意: 联合优化同时搜索架构空间({arch_space.get_num_configurations()}种)和晶体管参数空间")
    print()
    
    moead.run()
    
    # Save results
    output_dir = Path(current_dir) / 'results' / 'moead'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    moead.save_results(str(output_dir / 'pareto_solutions_4obj.txt'))
    
    # Print best solutions
    states, objectives = moead.get_pareto_front() 
    
    print("\n" + "=" * 70)
    print("Pareto前沿最优解 (前5个):")
    print("=" * 70)
    print(f"总评估次数: {len(evaluation_cache)} (已缓存避免重复)")
    
    for i, (state, obj_vals) in enumerate(zip(states[:5], objectives[:5])):
        nmos_models = ["NMOS_VTL", "NMOS_VTG", "NMOS_VTH"]
        pmos_models = ["PMOS_VTL", "PMOS_VTG", "PMOS_VTH"]
        
        # 将state映射回物理参数（state是标准化值[0,1]）
        import torch
        x_norm = torch.tensor([
            state[2], state[3], state[4], state[5],  # 4个晶体管连续参数
            state[6], state[6],  # nmos_model用同一个索引
            state[7]             # pmos_model
        ], dtype=torch.float32)
        params = param_space.convert_params(x_norm)
        
        # 获取架构配置
        rows = arch_space.row_choices[int(state[0])]
        cols = arch_space.column_choices[int(state[1])]
        
        # 获取目标值
        if isinstance(obj_vals, (list, tuple, np.ndarray)):
            obj1, obj2, obj3, obj4 = obj_vals[0], obj_vals[1], obj_vals[2], obj_vals[3]
        else:
            obj1, obj2, obj3, obj4 = 0, 0, 0, 0
        
        print(f"\n解 {i+1}:")
        print(f"  架构: {rows}x{cols}")
        print(f"  pu_width: {params['pu_width']*1e9:.2f} nm")
        print(f"  pd_width: {params['pd_width']*1e9:.2f} nm")
        print(f"  pg_width: {params['pg_width']*1e9:.2f} nm")
        print(f"  length: {params['length']*1e9:.2f} nm")
        print(f"  NMOS模型: {params.get('nmos_model_name', nmos_models[int(state[6])])}")
        print(f"  PMOS模型: {params.get('pmos_model_name', pmos_models[int(state[7])])}")
        print(f"  SNM: {-obj1:.6f} V  (最大化SNM)")
        print(f"  Power: {obj2:.6e} W")
        print(f"  Area: {obj3*1e12:.2f} µm²")
        print(f"  Delay: {obj4:.6e} s")
    
    print("\n" + "=" * 70)
    print(f"优化完成! 共找到 {len(states)} 个Pareto最优解")
    print(f"实际评估次数: {evaluation_counter['count']}")
    print(f"缓存命中: {len(evaluation_cache)}个不同参数")
    print(f"Pareto解数据已保存到: {output_dir / 'pareto_solutions_4obj.txt'}")
    print(f"完整日志已保存到: {log_file}")
    print("=" * 70)
    
    # 恢复stdout并关闭日志文件
    sys.stdout = original_stdout
    log_fileobj.close()
    
    # 返回Pareto前沿解集（多目标优化原生结果）
    # MOEAD返回多个不同权衡的解，而不是单一"最优"解
    if len(states) > 0 and len(objectives) > 0:
        # 构建所有Pareto解的详细信息
        pareto_solutions = []
        import torch
        
        for idx, (state, obj) in enumerate(zip(states, objectives)):
            if _use_external:
                # 外部参数空间: state 为全维向量
                params = _ext_param_space.convert_params(torch.tensor(state, dtype=torch.float32))
                rows = params.get('rows', 0)
                cols = params.get('cols', 0)
                # 从缓存获取结果
                _cache_key = tuple(round(float(v), 10) for v in state)
                if _cache_key in _ext_results_cache:
                    _, result, _ = _ext_results_cache[_cache_key]
                else:
                    _, result, _ = _eval_ext_full(*state)
            else:
                # 内部8维: 原有解码逻辑
                rows = arch_space.row_choices[int(state[0])]
                cols = arch_space.column_choices[int(state[1])]
                x_tensor = torch.tensor([state[2], state[3], state[4], state[5], state[6], state[7]], dtype=torch.float32)
                params = param_space.convert_params(x_tensor)
                params['rows'] = rows
                params['cols'] = cols
                params['row_idx'] = int(state[0])
                params['col_idx'] = int(state[1])
                _, _, result, _ = evaluate_joint(
                    state[0], state[1], state[2], state[3],
                    state[4], state[5], state[6], state[7]
                )
            
            # 从result中提取实际字段值，确保与experiment.py一致
            min_snm = result.get('min_snm', 0) if result else 0
            max_power = result.get('max_power', 0) if result else 0
            area = result.get('area', 0) if result else 0
            max_delay = result.get('max_delay', 0) if result else 0
            
            pareto_solutions.append({
                'solution_id': idx + 1,
                'params': params,
                'objectives': {
                    'min_snm': min_snm,        # 使用result中的实际字段名
                    'max_power': max_power,    # 使用max_power（相当于total_power）
                    'area': area,
                    'max_delay': max_delay     # 使用max_delay
                },
                'result': result,
                'state': state.tolist() if hasattr(state, 'tolist') else state
            })
        
        experiment_dir = Path(project_root) / "size_optimization" / "experiment" / "MOEAD"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        csv_path = experiment_dir / f"pareto_solutions_{timestamp}.csv"
        records = []
        for sol in pareto_solutions:
            params = sol.get("params") or {}
            objectives = sol.get("objectives") or {}
            rows = params.get("rows")
            cols = params.get("cols")
            num_arrays = None
            if rows and cols and arch_space.total_bits % (rows * cols) == 0:
                num_arrays = arch_space.total_bits // (rows * cols)
            rec_dict = {
                    "solution_id": sol.get("solution_id"),
                    "rows": rows,
                    "cols": cols,
                    "num_arrays": num_arrays,
                    "pg_width": params.get("pg_width"),
                    "pd_width": params.get("pd_width"),
                    "pu_width": params.get("pu_width"),
                    "length": params.get("length"),
                    "nmos_model_name": params.get("nmos_model_name"),
                    "pmos_model_name": params.get("pmos_model_name"),
                }
            # 添加外围电路参数
            for pk in PERIPHERAL_ALL_KEYS:
                rec_dict[pk] = _peripheral_defaults.get(pk, "")
            rec_dict.update({
                    "min_snm": objectives.get("min_snm"),
                    "max_power": objectives.get("max_power"),
                    "area": objectives.get("area"),
                    "max_delay": objectives.get("max_delay"),
                })
            records.append(rec_dict)
        fieldnames = [
            "solution_id",
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
            "max_power",
            "area",
            "max_delay",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        print(f"MOEAD pareto CSV saved to: {csv_path}")

        # Pick best solution by FoM for experiment.py compatibility
        _best_sol = None
        _best_fom = -1e9
        for _s in pareto_solutions:
            _r = _s.get('result')
            if _r:
                _snm = _r.get('min_snm', 0)
                _pwr = _r.get('max_power', 0)
                _ar = _r.get('area', 0)
                _dl = max(_r.get('read_delay', 0), _r.get('write_delay', 0))
                if _snm > 0 and _pwr > 0 and _ar > 0 and _dl > 0:
                    _f = np.log10(_snm / (_pwr * np.sqrt(_ar) * _dl))
                    if _f > _best_fom:
                        _best_fom = _f
                        _best_sol = _s

        return {
            'algorithm': 'MOEAD',
            'is_multiobjective': True,
            'pareto_solutions': pareto_solutions,
            'num_solutions': len(pareto_solutions),
            'iteration': max_gen,
            'moead': moead,
            'params': _best_sol.get('params') if _best_sol else None,
            'merit': _best_fom if _best_fom > -1e9 else None,
            'result': _best_sol.get('result') if _best_sol else None,
        }
    else:
        experiment_dir = Path(project_root) / "size_optimization" / "experiment" / "MOEAD"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        csv_path = experiment_dir / f"pareto_solutions_{timestamp}.csv"
        fieldnames = [
            "solution_id",
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
            "max_power",
            "area",
            "max_delay",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"MOEAD pareto CSV saved to: {csv_path}")
        return {
            'algorithm': 'MOEAD',
            'is_multiobjective': True,
            'pareto_solutions': [],
            'num_solutions': 0,
            'iteration': max_gen,
            'moead': moead,
            'note': '优化失败，未找到有效解'
        }


if __name__ == "__main__":
    import argparse
    
    # Set random seed for reproducibility
    SEED = 1
    seed_set(seed=SEED)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MOEAD多目标优化 - 支持真实电路和等效模型')
    parser.add_argument('--max_iter', type=int, default=50, help='最大迭代次数 (默认: 50)')
    parser.add_argument('--h', type=int, default=2, help='权重向量划分数 (默认: 2, 快速测试)')
    parser.add_argument('--circuit_mode', type=int, default=1, choices=[1, 2], 
                       help='电路模式: 1=真实电路(慢但准确), 2=等效模型(快速近似), 默认=1')
    
    args = parser.parse_args()
    
    # Run MOEAD optimization
    # 推荐参数:
    #   快速测试: --h=2 --max_iter=10 --circuit_mode=2 
    #   标准运行: --h=3 --max_iter=50 --circuit_mode=2
    #   生产环境: --h=3 --max_iter=100 --circuit_mode=1 
    
    print(f"\n运行参数:")
    print(f"  max_iter = {args.max_iter}")
    print(f"  h = {args.h}")
    print(f"  circuit_mode = {args.circuit_mode} ({'真实电路' if args.circuit_mode == 1 else '等效模型'})")
    print(f"  预计Pareto前沿解数量 ≈ {(args.h+1)*(args.h+2)*(args.h+3)//6}")
    
    moead = main(max_iter=args.max_iter, circuit_mode=args.circuit_mode, h=args.h)
