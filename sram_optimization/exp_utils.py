"""
SRAM Circuit Optimization Utilities
SRAM电路优化工具函数

This file contains common utilities for SRAM circuit optimization algorithms.
该文件包含SRAM电路优化算法的通用工具函数。
"""

import numpy as np
import os
import torch
import random
import time
import csv
import json
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import signal

# Import SRAM simulation modules
# 导入SRAM仿真模块
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from utils import estimate_bitcell_area


def seed_set(seed):
    """
    Fix the random seed for reproducibility
    固定随机种子以确保结果可重现
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
    return {
        'nmos_model_name': 'NMOS_VTG',
        'pmos_model_name': 'PMOS_VTG',
        'pd_width': 0.205e-6,
        'pu_width': 0.09e-6,
        'pg_width': 0.135e-6,
        'length': 50e-9,
        'length_nm': 50
    }


def get_default_fallback_result():
    """
    Get default fallback result when initial simulation fails
    获取初始仿真失败时的默认后备结果
    """
    return {
        'hold_snm': {'success': True, 'snm': 0.30173446708423357},
        'read_snm': {'success': True, 'snm': 0.12591724230394877},
        'write_snm': {'success': True, 'snm': 0.3732610863628419},
        'read': {'success': True, 'delay': 2.0883543988703797e-10, 'power': 4.024476625792127e-05},
        'write': {'success': True, 'delay': 6.086260190977158e-11, 'power': 3.975272388991992e-05}
    }


def process_initial_result(initial_result, success):
    """
    Process initial simulation result into optimizer-compatible format
    将初始仿真结果处理为优化器兼容格式
    """
    if not success:
        print("Warning: Initial point simulation failed, using default values as initial point")
        print("警告：初始点仿真失败，使用默认值作为初始点")
        return get_default_fallback_result()
    else:
        # Convert to optimizer-compatible format
        # 转换为优化器兼容格式
        formatted_result = {
            'hold_snm': {'success': True, 'snm': initial_result['hold_snm']},
            'read_snm': {'success': True, 'snm': initial_result['read_snm']},
            'write_snm': {'success': True, 'snm': initial_result['write_snm']},
            'read': {'success': True, 'delay': initial_result['read_delay'], 
                     'power': abs(initial_result['read_power'])},  # Take absolute value
            'write': {'success': True, 'delay': initial_result['write_delay'], 
                      'power': abs(initial_result['write_power'])}  # Take absolute value
        }
        
        print(f"Initial point simulation successful!")
        print(f"初始点仿真成功！")
        print(f"SNM: Hold={formatted_result['hold_snm']['snm']:.4f}, Read={formatted_result['read_snm']['snm']:.4f}, Write={formatted_result['write_snm']['snm']:.4f}")
        print(f"Delay: Read={formatted_result['read']['delay']*1e12:.2f}ps, Write={formatted_result['write']['delay']*1e12:.2f}ps")
        print(f"Power: Read={formatted_result['read']['power']:.2e}W, Write={formatted_result['write']['power']:.2e}W")
        
        return formatted_result


def run_initial_evaluation(parameter_space, initial_params):
    """
    Run initial point evaluation and return processed results
    运行初始点评估并返回处理后的结果
    """
    print("Running initial point simulation to get baseline performance...")
    print("运行初始点仿真以获得基准性能...")
    
    # Use evaluate_sram function to evaluate initial parameters
    # 使用evaluate_sram函数评估初始参数
    objectives, constraints, initial_result, success = evaluate_sram(initial_params)
    
    # Process and return results
    # 处理并返回结果
    processed_result = process_initial_result(initial_result, success)
    return processed_result, initial_params


def evaluate_sram(params, timeout=120):
    """
    Execute SRAM evaluation with given parameters
    使用给定参数执行SRAM评估
    """
    try:
        print(f"Starting SRAM evaluation with parameters: {params}")
        print(f"开始使用参数进行SRAM评估: {params}")
        start_time = time.time()
        
        # Set simulation parameters
        # 设置仿真参数
        vdd = 1.0
        pdk_path = 'model_lib/models.spice'
        num_rows = 32
        num_cols = 1
        num_mc = 1
        
        # Calculate SRAM cell area
        # 计算SRAM单元面积
        area = estimate_bitcell_area(
            w_access=params['pg_width'],
            w_pd=params['pd_width'],
            w_pu=params['pu_width'],
            l_transistor=params['length']
        )
        print(f"Estimated 6T SRAM cell area: {area*1e12:.2f} µm²")
        print(f"估算的6T SRAM单元面积: {area*1e12:.2f} µm²")
        
        # Create testbench
        # 创建测试平台
        mc_testbench = Sram6TCoreMcTestbench(
            vdd,
            pdk_path,
            params['nmos_model_name'],
            params['pmos_model_name'],
            num_rows=num_rows,
            num_cols=num_cols,
            pd_width=params['pd_width'],
            pu_width=params['pu_width'],
            pg_width=params['pg_width'],
            length=params['length'],
            w_rc=False,
            pi_res=10 @ u_Ohm,
            pi_cap=0.001 @ u_pF,
            custom_mc=False,
            q_init_val=0,
            sim_path='sim',
        )
        
        print("Starting SRAM Monte Carlo simulation...")
        print("开始SRAM蒙特卡洛仿真...")
        
        # Set timeout handling
        # 设置超时处理
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Simulation timeout")
        
        # Set timeout
        # 设置超时
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            # Run simulation directly
            # 直接运行仿真
            hold_snm = mc_testbench.run_mc_simulation(
                operation='hold_snm', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, 
                vars=None,
            )
            
            read_snm = mc_testbench.run_mc_simulation(
                operation='read_snm', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, 
                vars=None,
            )
            
            write_snm = mc_testbench.run_mc_simulation(
                operation='write_snm', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, 
                vars=None,
            )
            
            w_delay, w_pavg = mc_testbench.run_mc_simulation(
                operation='write', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, 
                vars=None,
            )
            
            r_delay, r_pavg = mc_testbench.run_mc_simulation(
                operation='read', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, 
                vars=None,
            )
            
            # Cancel timeout
            # 取消超时
            signal.alarm(0)
            
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
            
            # Constraint conditions - only check maximum delay
            # 约束条件 - 仅检查最大延迟
            max_delay_constraint = 5e-10   # 500ps
            read_delay_feasible = read_delay < max_delay_constraint
            write_delay_feasible = write_delay < max_delay_constraint
            
            # Calculate comprehensive metric Merit
            # 计算综合指标Merit
            merit = np.log10(min_snm / (max_power * (area**0.5)))
            
            # Set objective function values - added area objective
            # 设置目标函数值 - 添加面积目标
            objectives = [
                -float(min_snm),      # Minimize negative min SNM (equivalent to maximizing min SNM)
                float(max_power),     # Minimize maximum power
                float(area)           # Minimize area
            ]
            
            # Constraints - only check maximum delay
            # 约束条件 - 仅检查最大延迟
            constraints = [
                float(read_delay - max_delay_constraint),   # read_delay < 500ps
                float(write_delay - max_delay_constraint),  # write_delay < 500ps
            ]
            
            # Create result object
            # 创建结果对象
            result = {
                'hold_snm': hold_snm_val,
                'read_snm': read_snm_val,
                'write_snm': write_snm_val,
                'min_snm': min_snm,
                'read_power': read_power,
                'write_power': write_power,
                'max_power': max_power,
                'area': area,
                'read_delay': read_delay,
                'write_delay': write_delay,
                'read_delay_feasible': read_delay_feasible,
                'write_delay_feasible': write_delay_feasible,
                'merit': merit
            }
            
            end_time = time.time()
            print(f"Simulation completed successfully! Time taken: {end_time - start_time:.2f} seconds")
            print(f"仿真成功完成！用时: {end_time - start_time:.2f} 秒")
            print(f"SNM = {min_snm:.6f}, Power = {max_power:.6e}, Area = {area*1e12:.2f} µm², Merit = {merit:.6e}")
            print(f"约束状态: 读延迟 {'满足' if read_delay_feasible else '违反'}, 写延迟 {'满足' if write_delay_feasible else '违反'}")
            
            return objectives, constraints, result, True
            
        except TimeoutError:
            signal.alarm(0)  # Cancel timeout
            print(f"Simulation timeout (exceeded {timeout} seconds)")
            print(f"仿真超时 (超过 {timeout} 秒)")
            objectives = [10.0, 1e-3, 1e-10]  # Penalty objective function
            constraints = [1.0, 1.0]  # Constraint violation
            return objectives, constraints, None, False
            
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            print(f"Error occurred during simulation: {str(e)}")
            print(f"仿真过程中发生错误: {str(e)}")
            traceback.print_exc()
            objectives = [10.0, 1e-3, 1e-10]  # Penalty objective function
            constraints = [1.0, 1.0]  # Constraint violation
            return objectives, constraints, None, False
            
    except Exception as e:
        print(f"Error occurred during overall evaluation: {str(e)}")
        print(f"整体评估过程中发生错误: {str(e)}")
        traceback.print_exc()
        objectives = [10.0, 1e-3, 1e-10]  # Penalty objective function
        constraints = [1.0, 1.0]  # Constraint violation
        return objectives, constraints, None, False


class ModifiedSRAMParameterSpace:
    """
    Modified SRAM Parameter Space class to handle multi-dimensional tensors
    修改的SRAM参数空间类，支持处理多维张量
    """
    
    def __init__(self):
        # Import the original SRAMParameterSpace to inherit its properties
        # 导入原始的SRAMParameterSpace以继承其属性
        from sram_cbo import SRAMParameterSpace
        base_space = SRAMParameterSpace()
        
        # Copy all attributes from base class
        # 从基类复制所有属性
        for attr in dir(base_space):
            if not attr.startswith('_'):
                setattr(self, attr, getattr(base_space, attr))
    
    def convert_params(self, x):
        """
        Convert normalized parameters to actual SRAM parameters, supporting multi-dimensional tensor input
        将归一化参数转换为实际SRAM参数，支持多维张量输入
        """
        # Ensure x is numpy array and one-dimensional
        # 确保x是numpy数组且为一维
        if isinstance(x, torch.Tensor):
            if x.ndim > 1:
                # If x is 2D or higher dimensional tensor, take the first row
                # 如果x是2D或更高维张量，取第一行
                x = x[0].cpu().numpy()
            else:
                x = x.cpu().numpy()
        elif isinstance(x, np.ndarray) and x.ndim > 1:
            # If x is 2D or higher dimensional numpy array, take the first row
            # 如果x是2D或更高维numpy数组，取第一行
            x = x[0]

        # Print original normalized parameter values
        # 打印原始归一化参数值
        print(f"Normalized parameter values: {x.tolist()}")
        print(f"归一化参数值: {x.tolist()}")

        # Discrete parameters (model selection) - properly handle discrete variables
        # 离散参数（模型选择） - 正确处理离散变量
        # Divide [0,1] interval into 3 parts, avoiding boundary issues
        # 将[0,1]区间分为3部分，避免边界问题
        nmos_idx = min(2, max(0, int(x[0] * 3)))
        pmos_idx = min(2, max(0, int(x[1] * 3)))
        nmos_model = self.nmos_model_map[nmos_idx]
        pmos_model = self.pmos_model_map[pmos_idx]

        print(f"NMOS index: {nmos_idx}, PMOS index: {pmos_idx}")
        print(f"NMOS索引: {nmos_idx}, PMOS索引: {pmos_idx}")

        # Continuous parameters (transistor sizes) - ensure conversion to native Python float
        # 连续参数（晶体管尺寸） - 确保转换为原生Python浮点数
        pd_width = float(x[2] * (self.pd_width_max - self.pd_width_min) + self.pd_width_min)
        pu_width = float(x[3] * (self.pu_width_max - self.pu_width_min) + self.pu_width_min)
        pg_width = float(x[4] * (self.pg_width_max - self.pg_width_min) + self.pg_width_min)

        # Discrete parameter (length) - improved mapping method
        # 离散参数（长度） - 改进的映射方法
        # Use finer quantization, avoid always falling on boundaries
        # 使用更细的量化，避免总是落在边界上
        length_range = self.length_max_nm - self.length_min_nm + 1
        length_nm = int(self.length_min_nm + x[5] * length_range)
        length_nm = max(self.length_min_nm, min(self.length_max_nm, length_nm))  # Ensure within range
        length = length_nm * 1e-9  # Convert to meters

        return {
            'nmos_model_name': nmos_model,
            'pmos_model_name': pmos_model,
            'pd_width': pd_width,
            'pu_width': pu_width,
            'pg_width': pg_width,
            'length': length,
            'length_nm': length_nm
        }

    def random_params(self):
        """
        Generate completely random parameters, ensuring diverse values
        生成完全随机的参数，确保值的多样性
        """
        x = torch.zeros(6)

        # Random selection of transistor models - use uniform distribution in [0,1] range, handle mapping when converting to models
        # 随机选择晶体管模型 - 在[0,1]范围内使用均匀分布，转换为模型时处理映射
        x[0] = random.random()  # Uniform distribution between 0-1
        x[1] = random.random()  # Uniform distribution between 0-1

        # Randomly generate width parameters (use uniform distribution)
        # 随机生成宽度参数（使用均匀分布）
        x[2] = random.random()  # Uniform distribution between 0-1
        x[3] = random.random()
        x[4] = random.random()

        # Randomly generate length parameter (use uniform distribution)
        # 随机生成长度参数（使用均匀分布）
        x[5] = random.random()

        # Print randomly generated normalized parameters
        # 打印随机生成的归一化参数
        print(f"Randomly generated normalized parameters: {x.tolist()}")
        print(f"随机生成的归一化参数: {x.tolist()}")

        return x


class BaseOptimizer:
    """
    Base optimizer class with common functionality for all optimization algorithms
    包含所有优化算法通用功能的基础优化器类
    """
    
    def __init__(self, parameter_space, algorithm_name, num_objectives=3, num_constraints=2, 
                 initial_result=None, initial_params=None):
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

        # Initialize training data
        # 初始化训练数据
        self.train_x = torch.zeros((0, self.dim))
        self.train_obj = torch.zeros((0, self.num_objectives))
        self.train_con = torch.zeros((0, self.num_constraints))

        # Initialize best tracking
        # 初始化最佳跟踪
        self._initialize_best_tracking(initial_result, initial_params)

        # Initialize logger
        # 初始化日志记录器
        self.logger = OptimizationLogger(algorithm_name)

        # If there's an initial feasible point, record it in CSV
        # 如果有初始可行点，将其记录到CSV
        if initial_result and hasattr(self, 'best_merit'):
            initial_min_snm = min(
                initial_result['hold_snm']['snm'],
                initial_result['read_snm']['snm'],
                initial_result['write_snm']['snm']
            )
            initial_max_power = max(
                initial_result['read']['power'],
                initial_result['write']['power']
            )
            
            if initial_params:
                initial_area = estimate_bitcell_area(
                    w_access=initial_params['pg_width'],
                    w_pd=initial_params['pd_width'],
                    w_pu=initial_params['pu_width'],
                    l_transistor=initial_params['length']
                )
            else:
                initial_area = estimate_bitcell_area(
                    w_access=self.parameter_space.base_pg_width,
                    w_pd=self.parameter_space.base_pd_width,
                    w_pu=self.parameter_space.base_pu_width,
                    l_transistor=self.parameter_space.base_length_nm * 1e-9
                )
            
            self.logger.write_merit_result(0, self.best_merit, initial_min_snm, initial_max_power, initial_area, True)

        # Reference point (for Pareto frontier)
        # 参考点（用于帕累托前沿）
        self.ref_point = torch.tensor([10.0, 1e-3, 1e-10])

        # Initialize historical data
        # 初始化历史数据
        self.iteration_history = []
        self.best_history = []

    def _initialize_best_tracking(self, initial_result, initial_params):
        """
        Initialize best tracking with initial result
        使用初始结果初始化最佳跟踪
        """
        if initial_result:
            # Extract values from initial results
            # 从初始结果中提取值
            initial_min_snm = min(
                initial_result['hold_snm']['snm'],
                initial_result['read_snm']['snm'],
                initial_result['write_snm']['snm']
            )
            initial_max_power = max(
                initial_result['read']['power'],
                initial_result['write']['power']
            )
            
            # Calculate area using initial parameters
            # 使用初始参数计算面积
            if initial_params:
                initial_area = estimate_bitcell_area(
                    w_access=initial_params['pg_width'],
                    w_pd=initial_params['pd_width'],
                    w_pu=initial_params['pu_width'],
                    l_transistor=initial_params['length']
                )
            else:
                initial_area = estimate_bitcell_area(
                    w_access=self.parameter_space.base_pg_width,
                    w_pd=self.parameter_space.base_pd_width,
                    w_pu=self.parameter_space.base_pu_width,
                    l_transistor=self.parameter_space.base_length_nm * 1e-9
                )
            
            # Calculate initial Merit value
            # 计算初始Merit值
            initial_merit = np.log10(initial_min_snm / (initial_max_power * (initial_area**0.5)))

            # Set initial optimal Merit and related results
            # 设置初始最优Merit和相关结果
            self.best_merit = initial_merit

            # Create parameter object representing initial configuration
            # 创建表示初始配置的参数对象
            if initial_params:
                self.best_merit_params = initial_params.copy()
            else:
                self.best_merit_params = {
                    'nmos_model_name': 'NMOS_VTG',
                    'pmos_model_name': 'PMOS_VTG',
                    'pd_width': self.parameter_space.base_pd_width,
                    'pu_width': self.parameter_space.base_pu_width,
                    'pg_width': self.parameter_space.base_pg_width,
                    'length': self.parameter_space.base_length_nm * 1e-9,
                    'length_nm': self.parameter_space.base_length_nm
                }

            # Create initial result object
            # 创建初始结果对象
            self.best_merit_result = {
                'hold_snm': initial_result['hold_snm']['snm'],
                'read_snm': initial_result['read_snm']['snm'],
                'write_snm': initial_result['write_snm']['snm'],
                'min_snm': initial_min_snm,
                'read_power': initial_result['read']['power'],
                'write_power': initial_result['write']['power'],
                'max_power': initial_max_power,
                'area': initial_area,
                'read_delay': initial_result['read']['delay'],
                'write_delay': initial_result['write']['delay'],
                'read_delay_feasible': initial_result['read']['delay'] < 5e-10,
                'write_delay_feasible': initial_result['write']['delay'] < 5e-10,
                'merit': initial_merit
            }

            self.best_merit_iteration = 0
            print(
                f"[Initial feasible point] Merit = {initial_merit:.6e}, Min SNM = {initial_min_snm:.6f}, "
                f"Max power = {initial_max_power:.6e}, Area = {initial_area*1e12:.2f} µm²"
            )
            print(
                f"[初始可行点] Merit = {initial_merit:.6e}, 最小SNM = {initial_min_snm:.6f}, "
                f"最大功耗 = {initial_max_power:.6e}, 面积 = {initial_area*1e12:.2f} µm²"
            )
        else:
            # If no initial result provided, use default values
            # 如果未提供初始结果，使用默认值
            self.best_merit = float('-inf')
            self.best_merit_params = None
            self.best_merit_result = None
            self.best_merit_iteration = -1

    def observe(self, x, objectives, constraints, result, success, iteration, opt_type=None):
        """
        Record observation results and update model
        记录观察结果并更新模型
        """
        if opt_type is None:
            opt_type = self.algorithm_name

        # Print debug information
        # 打印调试信息
        print(f"Observing new point - success: {success}")
        print(f"观察新点 - 成功: {success}")

        # Ensure x is numpy array, not tensor
        # 确保x是numpy数组，而非张量
        if isinstance(x, torch.Tensor):
            self.all_x.append(x.detach().cpu().numpy())
        else:
            self.all_x.append(np.array(x))

        # Ensure objectives is regular Python list
        # 确保objectives是常规Python列表
        self.all_objectives.append([float(obj) if isinstance(obj, torch.Tensor) else obj for obj in objectives])
        self.all_constraints.append([float(con) if isinstance(con, torch.Tensor) else con for con in constraints])
        self.all_results.append(result)
        self.all_success.append(success)

        # Convert parameters and write to CSV
        # 转换参数并写入CSV
        params = self.parameter_space.convert_params(
            x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32))
        self.logger.write_csv_result(iteration, opt_type, params, success, objectives, constraints, result)

        # Update training data
        # 更新训练数据
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x_tensor = x.unsqueeze(0)
        obj_tensor = torch.tensor([[float(obj) for obj in objectives]], dtype=torch.float32)
        con_tensor = torch.tensor([[float(con) for con in constraints]], dtype=torch.float32)

        self.train_x = torch.cat([self.train_x, x_tensor])
        self.train_obj = torch.cat([self.train_obj, obj_tensor])
        self.train_con = torch.cat([self.train_con, con_tensor])

        # Update Merit tracking
        # 更新Merit跟踪
        if success and result is not None:
            merit = result['merit']
            min_snm = result['min_snm']
            max_power = result['max_power']
            area = result['area']

            # Save current Merit result
            # 保存当前Merit结果
            self.all_merit.append(merit)

            # Check if it's the best Merit - add constraint condition check
            # 检查是否为最佳Merit - 添加约束条件检查
            is_best = False
            if merit > self.best_merit and all(c <= 0 for c in constraints):
                self.best_merit = merit
                self.best_merit_params = params
                self.best_merit_result = result
                self.best_merit_iteration = iteration
                is_best = True
                print(
                    f"[New best Merit!] Iteration {iteration}: Merit = {merit:.6e}, "
                    f"Min SNM = {min_snm:.6f}, Max power = {max_power:.6e}, Area = {area*1e12:.2f} µm²"
                )
                print(
                    f"[新的最佳Merit!] 迭代 {iteration}: Merit = {merit:.6e}, "
                    f"最小SNM = {min_snm:.6f}, 最大功耗 = {max_power:.6e}, 面积 = {area*1e12:.2f} µm²"
                )

            # Write Merit tracking file
            # 写入Merit跟踪文件
            self.logger.write_merit_result(iteration, merit, min_snm, max_power, area, is_best)

            # Track best history
            # 跟踪最佳历史
            if not self.best_history or merit > max(self.best_history):
                self.best_history.append(merit)
            else:
                self.best_history.append(max(self.best_history))
        else:
            # For failed simulations, record an invalid Merit
            # 对于失败的仿真，记录无效Merit
            self.all_merit.append(float('-inf'))
            self.logger.write_merit_result(iteration, "N/A", "N/A", "N/A", "N/A", False)

            # For failed simulations, maintain previous best value in best_history
            # 对于失败的仿真，在best_history中保持先前的最佳值
            if self.best_history:
                self.best_history.append(self.best_history[-1])
            else:
                self.best_history.append(float('-inf'))

        # Update Pareto front (only consider points that satisfy constraints)
        # 更新帕累托前沿（只考虑满足约束的点）
        if result is not None and all(c <= 0 for c in constraints):
            self.pareto_front = update_pareto_front(
                self.pareto_front,
                {
                    'params': params,
                    'objectives': {
                        'min_snm': -float(objectives[0]),  # Convert back to original value (positive SNM)
                        'max_power': float(objectives[1]),
                        'area': float(objectives[2])
                    },
                    'raw_result': result
                }
            )

        # Record current iteration to history
        # 将当前迭代记录到历史
        self.iteration_history.append({
            'iteration': iteration,
            'params': {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in params.items() if k != 'length'},
            'objectives': objectives if success else None,
            'constraints': constraints if success else None,
            'merit': merit if success and result else None,
            'success': success
        })

    def get_best_points(self):
        """
        Get best points on Pareto front
        获取帕累托前沿上的最佳点
        """
        return self.pareto_front

    def get_best_merit(self):
        """
        Get best Merit and corresponding parameters
        获取最佳Merit及其对应参数
        """
        return {
            'merit': self.best_merit,
            'params': self.best_merit_params,
            'result': self.best_merit_result,
            'iteration': self.best_merit_iteration
        }

    def save_results(self):
        """
        Save optimization results
        保存优化结果
        """
        # Save Pareto front
        # 保存帕累托前沿
        save_pareto_front(self.pareto_front, self.algorithm_name)

        # Save best Merit result
        # 保存最佳Merit结果
        best_merit = self.get_best_merit()
        save_best_result(best_merit, self.algorithm_name)

        # Save iteration history
        # 保存迭代历史
        serializable_history = {
            "iteration_history": self.iteration_history,
            "best_history": self.best_history,
            "all_merit": [float(m) if not isinstance(m, str) else m for m in self.all_merit]
        }
        save_optimization_history(serializable_history, self.algorithm_name)

        # Plot and save Merit history curve and Pareto front plots
        # 绘制并保存Merit历史曲线和帕累托前沿图
        plot_merit_history(self.best_history, self.algorithm_name)
        plot_pareto_frontier(self.pareto_front, self.algorithm_name)

    def print_final_results(self):
        """
        Print final optimization results
        打印最终优化结果
        """
        print(f"\n===== {self.algorithm_name} Optimization Best Results =====")
        print(f"\n===== {self.algorithm_name}优化最佳结果 =====")
        
        if self.best_merit_params is not None:
            print(f"Best Merit: {self.best_merit:.6e}")
            print(f"Iteration count: {self.best_merit_iteration}")
            print("Best parameters:")
            print("最佳参数:")
            self.parameter_space.print_params(self.best_merit_params)
            print(f"Min SNM: {self.best_merit_result['min_snm']:.6f}")
            print(f"Max power: {self.best_merit_result['max_power']:.6e}")
            print(f"Area: {self.best_merit_result['area']*1e12:.2f} µm²")
            print(f"最小SNM: {self.best_merit_result['min_snm']:.6f}")
            print(f"最大功耗: {self.best_merit_result['max_power']:.6e}")
            print(f"面积: {self.best_merit_result['area']*1e12:.2f} µm²")
        else:
            print("No valid solution found")
            print("未找到有效解")


class OptimizationLogger:
    """
    Unified logger for all optimization algorithms
    所有优化算法的统一日志记录器
    """
    
    def __init__(self, algorithm_name, save_dir="sim/opt/results"):
        """
        Initialize the optimization logger
        初始化优化日志记录器
        """
        self.algorithm_name = algorithm_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # CSV files
        # CSV文件
        self.csv_file = self.save_dir / f"sram_{algorithm_name.lower()}_optimization_results.csv"
        self.merit_csv_file = self.save_dir / f"sram_{algorithm_name.lower()}_merit_results.csv"
        
        # Initialize CSV headers
        # 初始化CSV头部
        self.write_csv_header()
        self.write_merit_csv_header()
        
        # Best values tracking
        # 最佳值跟踪
        self.current_best_merit = float('-inf')
        self.current_best_min_snm = None
        self.current_best_max_power = None
        self.current_best_area = None
    
    def write_csv_header(self):
        """
        Create CSV file and write header
        创建CSV文件并写入头部
        """
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Iteration',
                'Type',
                'NMOS_Model',
                'PMOS_Model',
                'PD_Width',
                'PU_Width',
                'PG_Width',
                'Length_nm',
                'Success',
                'Hold_SNM',
                'Read_SNM',
                'Write_SNM',
                'Min_SNM',
                'Read_Power',
                'Write_Power',
                'Max_Power',
                'Area',
                'Read_Delay',
                'Write_Delay',
                'Read_Delay_Feasible',
                'Write_Delay_Feasible',
                'Merit'
            ])
    
    def write_merit_csv_header(self):
        """
        Create Merit tracking CSV file and write header
        创建Merit跟踪CSV文件并写入头部
        """
        with open(self.merit_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Iteration',
                'Merit',
                'Min_SNM',
                'Max_Power',
                'Area'
            ])
    
    def write_csv_result(self, iteration, opt_type, params, success, objectives, constraints, result=None):
        """
        Write results to CSV file
        将结果写入CSV文件
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if success and result is not None:
                row = [
                    iteration,
                    opt_type,
                    params['nmos_model_name'],
                    params['pmos_model_name'],
                    params['pd_width'],
                    params['pu_width'],
                    params['pg_width'],
                    params['length_nm'],
                    success,
                    result['hold_snm'],
                    result['read_snm'],
                    result['write_snm'],
                    result['min_snm'],
                    result['read_power'],
                    result['write_power'],
                    result['max_power'],
                    result['area'],
                    result['read_delay'],
                    result['write_delay'],
                    result['read_delay_feasible'],
                    result['write_delay_feasible'],
                    result['merit']
                ]
            else:
                row = [
                    iteration,
                    opt_type,
                    params['nmos_model_name'],
                    params['pmos_model_name'],
                    params['pd_width'],
                    params['pu_width'],
                    params['pg_width'],
                    params['length_nm'],
                    success,
                    "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
                ]
            writer.writerow(row)
    
    def write_merit_result(self, iteration, merit, min_snm, max_power, area, is_best):
        """
        Write Merit results to CSV file
        将Merit结果写入CSV文件
        """
        # Adjust iteration number to start from 1
        # 调整迭代编号从1开始
        adjusted_iteration = iteration + 1

        # If it's a new best value, update stored best values
        # 如果是新的最佳值，更新存储的最佳值
        if is_best:
            self.current_best_merit = merit
            self.current_best_min_snm = min_snm
            self.current_best_max_power = max_power
            self.current_best_area = area

        # Always write current best value
        # 总是写入当前最佳值
        with open(self.merit_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                adjusted_iteration,
                self.current_best_merit if hasattr(self, 'current_best_merit') else merit,
                self.current_best_min_snm if hasattr(self, 'current_best_min_snm') else min_snm,
                self.current_best_max_power if hasattr(self, 'current_best_max_power') else max_power,
                self.current_best_area if hasattr(self, 'current_best_area') else area
            ])


def save_pareto_front(pareto_front, algorithm_name, save_dir="sim/opt/results"):
    """
    Save Pareto front to CSV file
    将帕累托前沿保存到CSV文件
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    pareto_file = save_path / f"pareto_front_{algorithm_name.lower()}.csv"
    with open(pareto_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'NMOS_Model',
            'PMOS_Model',
            'PD_Width',
            'PU_Width',
            'PG_Width',
            'Length_nm',
            'Min_SNM',
            'Max_Power',
            'Area',
            'Merit',
            'Satisfies_Constraints'
        ])

        for point in pareto_front:
            params = point['params']
            obj = point['objectives']
            result = point['raw_result']

            # Check if constraints are satisfied
            # 检查约束是否满足
            satisfies_constraints = (
                result['read_delay_feasible'] and
                result['write_delay_feasible']
            )

            writer.writerow([
                params['nmos_model_name'],
                params['pmos_model_name'],
                params['pd_width'],
                params['pu_width'],
                params['pg_width'],
                params['length_nm'],
                obj['min_snm'],
                obj['max_power'],
                obj['area'],
                result['merit'],
                satisfies_constraints
            ])
    print(f"Pareto front saved to {pareto_file}")
    print(f"帕累托前沿已保存到 {pareto_file}")


def save_best_result(best_result, algorithm_name, save_dir="sim/opt/results"):
    """
    Save best result to CSV file
    将最佳结果保存到CSV文件
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    best_file = save_path / f"best_merit_result_{algorithm_name.lower()}.csv"
    if best_result['params'] is not None:
        with open(best_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Iteration',
                'Merit',
                'NMOS_Model',
                'PMOS_Model',
                'PD_Width',
                'PU_Width',
                'PG_Width',
                'Length_nm',
                'Min_SNM',
                'Max_Power',
                'Area',
                'Read_Delay',
                'Write_Delay'
            ])

            writer.writerow([
                best_result['iteration'],
                best_result['merit'],
                best_result['params']['nmos_model_name'],
                best_result['params']['pmos_model_name'],
                best_result['params']['pd_width'],
                best_result['params']['pu_width'],
                best_result['params']['pg_width'],
                best_result['params']['length_nm'],
                best_result['result']['min_snm'],
                best_result['result']['max_power'],
                best_result['result']['area'],
                best_result['result']['read_delay'],
                best_result['result']['write_delay']
            ])
        print(f"Best result saved to {best_file}")
        print(f"最佳结果已保存到 {best_file}")


def plot_merit_history(best_history, algorithm_name, save_dir="sim/opt/plots"):
    """
    Plot Merit history curve
    绘制Merit历史曲线
    """
    if not best_history:
        print("Not enough historical data for plotting")
        print("没有足够的历史数据用于绘图")
        return

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(10, 6))
    
    # Set line color based on algorithm
    # 根据算法设置线条颜色
    color_map = {
        'pso': 'b-',
        'sa': 'r-', 
        'cbo': 'g-',
        'rose_opt': 'm-',
        'smac': 'c-'
    }
    color = color_map.get(algorithm_name.lower(), 'k-')
    
    plt.plot(range(1, len(best_history) + 1), best_history, color, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Figure of Merit', fontsize=12)
    plt.title(f'{algorithm_name.upper()} Optimization: Best Merit vs Iteration', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add best value marker
    # 添加最佳值标记
    if best_history:
        best_iter = np.argmax(best_history) + 1
        best_merit = max(best_history)
        plt.plot(best_iter, best_merit, 'ro', markersize=8)
        plt.annotate(f'Best Merit: {best_merit:.6e}',
                     xy=(best_iter, best_merit),
                     xytext=(best_iter + 1, best_merit),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=10)

    plt.tight_layout()
    plot_file = save_path / f"{algorithm_name.lower()}_merit_history.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Merit history curve saved to {plot_file}")
    print(f"Merit历史曲线已保存到 {plot_file}")


def plot_pareto_frontier(pareto_front, algorithm_name, save_dir="sim/opt/plots"):
    """
    Plot 2D and 3D views of Pareto front
    绘制帕累托前沿的2D和3D视图
    """
    if len(pareto_front) == 0:
        print("No Pareto front points available for plotting")
        print("没有帕累托前沿点可用于绘图")
        return
        
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
        
    # Extract three objective values from Pareto front points
    # 从帕累托前沿点提取三个目标值
    snm_values = []
    power_values = []
    area_values = []
    
    for point in pareto_front:
        obj = point['objectives']
        snm_values.append(obj['min_snm'])
        power_values.append(obj['max_power'])
        area_values.append(obj['area'])
    
    # Create three 2D plots: SNM vs Power, SNM vs Area, Power vs Area
    # 创建三个2D图：SNM vs 功耗, SNM vs 面积, 功耗 vs 面积
    plt.figure(figsize=(15, 5))
    
    # 1. SNM vs Power
    plt.subplot(1, 3, 1)
    plt.scatter(snm_values, power_values, c='blue', marker='o')
    plt.title('SNM vs Power Pareto Frontier')
    plt.xlabel('Min SNM')
    plt.ylabel('Max Power (W)')
    plt.grid(True)
    
    # 2. SNM vs Area
    plt.subplot(1, 3, 2)
    plt.scatter(snm_values, [a*1e12 for a in area_values], c='red', marker='o')  # Convert to µm²
    plt.title('SNM vs Area Pareto Frontier')
    plt.xlabel('Min SNM')
    plt.ylabel('Area (µm²)')
    plt.grid(True)
    
    # 3. Power vs Area
    plt.subplot(1, 3, 3)
    plt.scatter(power_values, [a*1e12 for a in area_values], c='green', marker='o')  # Convert to µm²
    plt.title('Power vs Area Pareto Frontier')
    plt.xlabel('Max Power (W)')
    plt.ylabel('Area (µm²)')
    plt.grid(True)
    
    plt.tight_layout()
    plot_2d_file = save_path / f"pareto_frontiers_2d_{algorithm_name.lower()}.png"
    plt.savefig(plot_2d_file, dpi=300)
    plt.close()
    
    # Plot three-dimensional scatter plot
    # 绘制三维散点图
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(snm_values, power_values, [a*1e12 for a in area_values], 
                c=[p['raw_result']['merit'] for p in pareto_front],
                cmap='viridis', marker='o')
                
        ax.set_xlabel('Min SNM')
        ax.set_ylabel('Max Power (W)')
        ax.set_zlabel('Area (µm²)')
        ax.set_title('3D Pareto Frontier')
        
        # Add color bar, representing Merit values
        # 添加颜色条，表示Merit值
        cbar = plt.colorbar(scatter)
        cbar.set_label('Merit Value')
        
        plot_3d_file = save_path / f"pareto_frontier_3d_{algorithm_name.lower()}.png"
        plt.savefig(plot_3d_file, dpi=300)
        plt.close()
        
        print(f"Pareto frontier plots saved")
        print(f"帕累托前沿图已保存")
        
    except Exception as e:
        print(f"Error plotting 3D Pareto front: {e}")
        print(f"绘制3D帕累托前沿时出错: {e}")


def update_pareto_front(pareto_front, new_point):
    """
    Update Pareto front with new point
    用新点更新帕累托前沿
    """
    # If Pareto front is empty, directly add new point
    # 如果帕累托前沿为空，直接添加新点
    if len(pareto_front) == 0:
        return [new_point]

    # Check if new point is dominated by existing front
    # 检查新点是否被现有前沿支配
    is_dominated = False

    # Points to remove from front
    # 要从前沿移除的点
    to_remove = []

    for i, point in enumerate(pareto_front):
        # Check if new point dominates existing point
        # 检查新点是否支配现有点
        if dominates(new_point, point):
            to_remove.append(i)
        # Check if existing point dominates new point
        # 检查现有点是否支配新点
        elif dominates(point, new_point):
            is_dominated = True
            break

    # If new point is not dominated, add to front
    # 如果新点未被支配，添加到前沿
    if not is_dominated:
        # Remove dominated points
        # 移除被支配的点
        updated_front = [point for i, point in enumerate(pareto_front) if i not in to_remove]
        updated_front.append(new_point)
        return updated_front

    return pareto_front


def dominates(point1, point2):
    """
    Check if point1 dominates point2 for three-objective case
    检查point1是否在三目标情况下支配point2
    """
    # Higher SNM is better, lower power is better, lower area is better
    # SNM越高越好，功耗越低越好，面积越小越好
    obj1 = point1['objectives']
    obj2 = point2['objectives']

    at_least_as_good = (
        (obj1['min_snm'] >= obj2['min_snm']) and      # Higher SNM is better
        (obj1['max_power'] <= obj2['max_power']) and  # Lower power is better
        (obj1['area'] <= obj2['area'])                # Lower area is better
    )

    strictly_better = (
        (obj1['min_snm'] > obj2['min_snm']) or       # Higher SNM is better
        (obj1['max_power'] < obj2['max_power']) or   # Lower power is better
        (obj1['area'] < obj2['area'])                # Lower area is better
    )

    return at_least_as_good and strictly_better


def save_optimization_history(history, algorithm_name, save_dir="sim/opt/results"):
    """
    Save optimization history to JSON file
    将优化历史保存到JSON文件
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    history_file = save_path / f"{algorithm_name.lower()}_history.json"
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Optimization history saved to {history_file}")
        print(f"优化历史已保存到 {history_file}")
    except Exception as e:
        print(f"Error saving historical data: {e}")
        print(f"保存历史数据时出错: {e}")
        traceback.print_exc()


def create_standard_optimizer_main(algorithm_name, optimizer_class, **kwargs):
    """
    Create a standardized main function for optimization algorithms
    为优化算法创建标准化的主函数
    """
    def main():
        print(f"===== SRAM {algorithm_name} Optimization =====")
        print(f"===== SRAM {algorithm_name}优化 =====")

        # Setup paths and create directories
        # 设置路径并创建目录
        create_directories()

        # Create parameter space
        # 创建参数空间
        parameter_space = ModifiedSRAMParameterSpace()

        # Get initial parameters and run evaluation
        # 获取初始参数并运行评估
        initial_params = get_default_initial_params()
        initial_result, initial_params = run_initial_evaluation(parameter_space, initial_params)

        # Create optimizer
        # 创建优化器
        optimizer = optimizer_class(
            parameter_space,
            initial_result=initial_result,
            initial_params=initial_params,
            **kwargs
        )

        # Run optimization
        # 运行优化
        best_result = optimizer.run_optimization(max_iter=400)

        # Print final results
        # 打印最终结果
        optimizer.print_final_results()

        return optimizer

    return main
