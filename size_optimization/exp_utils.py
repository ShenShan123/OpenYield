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
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import signal
from typing import Dict, List, Any, Union, Tuple
from abc import ABC, abstractmethod
from config import SRAM_CONFIG
# Import SRAM simulation modules
# 导入SRAM仿真模块
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
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


def run_initial_evaluation(params):
    """
    Run initial parameter evaluation
    运行初始参数评估
    """
    objectives, constraints, result, success = evaluate_sram(params)
    
    if success:
        formatted_result = {
            'hold_snm': {'success': True, 'snm': result['hold_snm']},
            'read_snm': {'success': True, 'snm': result['read_snm']},
            'write_snm': {'success': True, 'snm': result['write_snm']},
            'read': {'success': True, 'delay': result['read_delay'], 
                     'power': abs(result['read_power'])},
            'write': {'success': True, 'delay': result['write_delay'], 
                      'power': abs(result['write_power'])}
        }
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
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: 配置文件 {self.config_path} 不存在，使用默认配置")
            return self._get_default_config()
        except Exception as e:
            print(f"Warning: 配置文件加载失败 {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'sim_params': {
                'vdd': 1.0,
                'temperature': 27,
                'num_rows': 32,
                'num_cols': 1,
                'monte_carlo_runs': 1,
                'timeout': 120
            },
            'pdk': {
                'path': 'tran_models/models_TT.spice'
            },
            'subcircuit': {
                'name': 'SRAM_6T_Cell',
                'parameter_space': {
                    'parameters': {
                        'pmos_width': {
                            'type': 'continuous list',
                            'names': ['pu'],
                            'upper': [1.35e-7],
                            'lower': [4.5e-8],
                            'default': [9e-8]
                        },
                        'nmos_width': {
                            'type': 'continuous list',
                            'names': ['pd', 'pg'],
                            'upper': [3.075e-7, 2.025e-7],
                            'lower': [1.025e-7, 6.75e-8],
                            'default': [2.05e-7, 1.35e-7]
                        },
                        'length': {
                            'type': 'continuous value',
                            'names': 'l',
                            'upper': 1e-7,
                            'lower': 3e-8,
                            'default': 50e-9
                        },
                        'nmos_model': {
                            'type': 'categorical list',
                            'names': ['pd', 'pg'],
                            'choices': ['NMOS_VTL', 'NMOS_VTG', 'NMOS_VTH'],
                            'default': ['NMOS_VTG', 'NMOS_VTG']
                        },
                        'pmos_model': {
                            'type': 'categorical list',
                            'names': ['pu'],
                            'choices': ['PMOS_VTL', 'PMOS_VTG', 'PMOS_VTH'],
                            'default': ['PMOS_VTG']
                        }
                    }
                }
            }
        }
    
    def get_global_params(self) -> Dict:
        """获取全局参数"""
        return {
            'sim_params': self.config.get('sim_params', {}),
            'pdk': self.config.get('pdk', {})
        }
    
    def get_subcircuit_config(self) -> Dict:
        """获取子电路配置"""
        return self.config.get('subcircuit', {})


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
        self.param_config = self.config_loader.get_subcircuit_config().get('parameter_space', {}).get('parameters', {})
        self.param_info = self._parse_parameters()
        self.bounds = self._create_bounds()
        
        # 为了兼容原始代码，保留原始属性
        self._setup_legacy_attributes()
        
    def _parse_parameters(self) -> Dict:
        """解析参数配置"""
        param_info = {}
        
        for param_name, param_config in self.param_config.items():
            param_type = param_config['type']
            
            if param_type == "continuous list":
                names = param_config['names']
                upper = param_config['upper']
                lower = param_config['lower'] 
                default = param_config['default']
                
                param_info[param_name] = {
                    'type': 'continuous_list',
                    'names': names,
                    'upper': upper,
                    'lower': lower,
                    'default': default,
                    'other_default': param_config.get('other_default', 0.0),
                    'count': len(names)
                }
                
            elif param_type == "continuous value":
                param_info[param_name] = {
                    'type': 'continuous_scalar',
                    'names': param_config['names'],
                    'upper': param_config['upper'],
                    'lower': param_config['lower'],
                    'default': param_config['default'],
                    'count': 1
                }
                
            elif param_type == "categorical list":
                names = param_config['names']
                choices = param_config['choices']
                default = param_config['default']
                
                param_info[param_name] = {
                    'type': 'categorical_list',
                    'names': names,
                    'choices': choices,
                    'default': default,
                    'count': len(names)
                }
        
        return param_info
    
    def _create_bounds(self) -> torch.Tensor:
        """创建参数边界"""
        bounds = []
        
        for param_name, info in self.param_info.items():
            if info['type'] == 'continuous_list':
                for i in range(info['count']):
                    bounds.append([0.0, 1.0])  # 标准化到[0,1]
            elif info['type'] == 'continuous_scalar':
                bounds.append([0.0, 1.0])  # 标准化到[0,1]
            elif info['type'] == 'categorical_list':
                for i in range(info['count']):
                    bounds.append([0.0, 1.0])  # 标准化到[0,1]，后续映射到离散选择
        
        return torch.tensor(bounds).T  # shape: (2, total_dims)
    
    def _setup_legacy_attributes(self):
        """设置兼容原始代码的属性"""
        # 为了兼容原始代码，从配置中提取基础参数
        nmos_width_config = self.param_info.get('nmos_width', {})
        pmos_width_config = self.param_info.get('pmos_width', {})
        length_config = self.param_info.get('length', {})
        
        # 设置默认值
        if nmos_width_config:
            nmos_defaults = nmos_width_config['default']
            if isinstance(nmos_defaults, (list, tuple)):
                pd_default = nmos_defaults[0] if 'pd' in nmos_width_config['names'] else 2.05e-7
                pg_default = nmos_defaults[1] if len(nmos_defaults) > 1 else 1.35e-7
            else:
                pd_default = pg_default = float(nmos_defaults)
        else:
            pd_default, pg_default = 2.05e-7, 1.35e-7
            
        if pmos_width_config:
            pmos_defaults = pmos_width_config['default']
            if isinstance(pmos_defaults, (list, tuple)):
                pu_default = pmos_defaults[0] if pmos_defaults else 9e-8
            else:
                pu_default = float(pmos_defaults)
        else:
            pu_default = 9e-8
            
        if length_config:
            length_default = length_config['default']
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
            if info['type'] == 'continuous_list':
                # 连续型列表参数
                for i, name in enumerate(info['names']):
                    # 从[0,1]映射到实际范围
                    val = info['lower'][i] + x[dim_idx] * (info['upper'][i] - info['lower'][i])
                    
                    if param_name == 'nmos_width':
                        if name == 'pd':
                            params['pd_width'] = float(val)
                        elif name == 'pg':
                            params['pg_width'] = float(val)
                    elif param_name == 'pmos_width':
                        if name == 'pu':
                            params['pu_width'] = float(val)
                    
                    dim_idx += 1
                    
            elif info['type'] == 'continuous_scalar' or info['type'] == 'continuous value':
                # 连续型标量参数
                upper = float(info['upper'])
                lower = float(info['lower'])
                val = lower + x[dim_idx] * (upper - lower)
                
                if param_name == 'length':
                    params['length'] = float(val)
                    params['length_nm'] = float(val * 1e9)
                
                dim_idx += 1
                
            elif info['type'] == 'categorical_list':
                # 分类型列表参数
                for i, name in enumerate(info['names']):
                    # 从[0,1]映射到离散选择索引
                    choice_idx = int(x[dim_idx] * len(info['choices']))
                    choice_idx = min(choice_idx, len(info['choices']) - 1)
                    
                    if param_name == 'nmos_model':
                        params['nmos_model_name'] = info['choices'][choice_idx]
                    elif param_name == 'pmos_model':
                        params['pmos_model_name'] = info['choices'][choice_idx]
                    
                    dim_idx += 1
        
        return params
    
    def print_params(self, params):
        """Print parameter information"""
        print("Parameters / 参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")


def evaluate_sram(params, timeout=120):
    """
    Execute SRAM evaluation with given parameters
    使用给定参数执行SRAM评估
    """
    # 获取项目根目录路径
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
            "DECODER": os.path.join(project_root, "sram_compiler/config_yaml/decoder.yaml")
        }
    )
    try:
        print(f"Starting SRAM evaluation with parameters: {params}")
        print(f"开始使用参数进行SRAM评估: {params}")
        start_time = time.time()
        
        # Set simulation parameters
        # 设置仿真参数
        vdd = 1.0
        pdk_path = os.path.join(project_root, 'tran_models/models_TT.spice')
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
        
        # Create testbench with parameter sweep configuration
        # 创建带有参数扫描配置的测试平台
        mc_testbench = Sram6TCoreMcTestbench(
            sram_config,
            w_rc=True,  # Whether add RC to nets 
            pi_res=100 @ u_Ohm, 
            pi_cap=0.001 @ u_pF,
            vth_std=0.05,  # Process parameter variation is a percentage of its value in model lib 
            custom_mc=False,  # Use your own process params? 
            param_sweep=True,           # 启用参数扫描主开关
            sweep_precharge=True,       # 扫描预充电电路（影响读取性能）
            sweep_senseamp=True,        # 扫描灵敏放大器（影响读取灵敏度）
            sweep_wordlinedriver=True,  # 扫描字线驱动器（影响访问速度）
            sweep_columnmux=False,      # 不扫描列多路复用器
            sweep_writedriver=False,    # 不扫描写驱动器
            sweep_decoder=False,        # 不扫描解码器 
            coner='TT',  # or FF or SS or FS or SF 
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
            # 约束条件 - 只检查最大延迟
            read_delay_feasible = read_delay <= 2e-10  # 200ps
            write_delay_feasible = write_delay <= 2e-10  # 200ps
            
            # Calculate Merit function: Merit = log10(min_snm / (max_power * sqrt(area)))
            # 计算Merit函数: Merit = log10(min_snm / (max_power * sqrt(area)))
            if min_snm > 0 and max_power > 0 and area > 0:
                merit = np.log10(min_snm / (max_power * np.sqrt(area)))
            else:
                merit = -10.0  # Penalty value for invalid cases
            
            # Construct objectives (for multi-objective optimization)
            # 构建目标函数（用于多目标优化）
            objectives = [min_snm, -max_power, -area]  # Maximize SNM, minimize power and area
            
            # Construct constraints (constraint violation)
            # 构建约束（约束违反）
            constraints = []
            if not read_delay_feasible:
                constraints.append(read_delay - 2e-10)
            else:
                constraints.append(0.0)
                
            if not write_delay_feasible:
                constraints.append(write_delay - 2e-10)
            else:
                constraints.append(0.0)
            
            # Construct detailed results
            # 构建详细结果
            result = {
                'hold_snm': hold_snm_val,
                'read_snm': read_snm_val,
                'write_snm': write_snm_val,
                'min_snm': min_snm,
                'read_power': read_power,
                'write_power': write_power,
                'max_power': max_power,
                'read_delay': read_delay,
                'write_delay': write_delay,
                'area': area,
                'merit': merit,
                'read_delay_feasible': read_delay_feasible,
                'write_delay_feasible': write_delay_feasible
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


class BaseOptimizer:
    """
    Base optimizer class
    基础优化器类
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
        self.best_merit = float('-inf')
        self.best_params = None
        self.best_result = None

        # Set initial feasible point Merit and results
        # 设置初始可行点Merit和结果
        if initial_result:
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
                
                self.initial_merit = np.log10(initial_min_snm / (initial_max_power * np.sqrt(initial_area)))
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
            merit = result['merit']
            self.all_merit.append(merit)
            
            # Update best result
            # 更新最佳结果
            if merit > self.best_merit:
                self.best_merit = merit
                self.best_params = self.parameter_space.convert_params(torch.tensor(x, dtype=torch.float32))
                self.best_result = result
                
            # Record iteration data
            # 记录迭代数据
            self.iteration_history.append({
                'iteration': iteration,
                'merit': merit,
                'objectives': objectives,
                'constraints': constraints,
                'success': success,
                'source': source
            })
            
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
        self.log_data.append({
            'iteration': iteration,
            'merit': merit,
            'objectives': objectives,
            'constraints': constraints,
            'success': success
        })
        
    def save_log(self, filename):
        """
        Save log to file
        保存日志到文件
        """
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['iteration', 'merit', 'objectives', 'constraints', 'success']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.log_data:
                writer.writerow(row)


def save_pareto_front(pareto_front, filename):
    """
    Save Pareto front to file
    保存Pareto前沿到文件
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['min_snm', 'max_power', 'area', 'merit']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for point in pareto_front:
            writer.writerow(point)


def save_best_result(best_result, algorithm_name, filename):
    """
    Save best result to file
    保存最佳结果到文件
    """
    result_data = {
        'algorithm': algorithm_name,
        'best_merit': best_result['merit'] if best_result else None,
        'best_params': best_result['params'] if best_result else None,
        'best_result': best_result['result'] if best_result else None
    }
    
    with open(filename, 'w') as jsonfile:
        json.dump(result_data, jsonfile, indent=2, default=str)


def plot_merit_history(merit_history, algorithm_name, filename):
    """
    Plot Merit function history
    绘制Merit函数历史
    """
    plt.figure(figsize=(10, 6))
    plt.plot(merit_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Merit')
    plt.title(f'{algorithm_name} Optimization: Merit vs Iteration')
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pareto_frontier(pareto_front, algorithm_name, filename):
    """
    Plot Pareto frontier
    绘制Pareto前沿
    """
    if len(pareto_front) == 0:
        return
        
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    snm_vals = [p['min_snm'] for p in pareto_front]
    power_vals = [p['max_power'] for p in pareto_front]
    area_vals = [p['area'] for p in pareto_front]
    
    ax.scatter(snm_vals, power_vals, area_vals, c='red', s=50)
    ax.set_xlabel('Min SNM (V)')
    ax.set_ylabel('Max Power (W)')
    ax.set_zlabel('Area (m²)')
    ax.set_title(f'{algorithm_name} Pareto Frontier')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def update_pareto_front(pareto_front, objectives, result):
    """
    Update Pareto front
    更新Pareto前沿
    """
    new_point = {
        'min_snm': objectives[0],
        'max_power': -objectives[1],  # Convert back from negative
        'area': -objectives[2],  # Convert back from negative
        'merit': result['merit']
    }
    
    # Simple Pareto front update (can be optimized)
    # 简单的Pareto前沿更新（可以优化）
    pareto_front.append(new_point)
    return pareto_front


def save_optimization_history(history, algorithm_name, filename):
    """
    Save optimization history
    保存优化历史
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['iteration', 'merit', 'min_snm', 'max_power', 'area', 'success']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for entry in history:
            row = {
                'iteration': entry['iteration'],
                'merit': entry['merit'],
                'min_snm': entry['objectives'][0] if entry['success'] else None,
                'max_power': -entry['objectives'][1] if entry['success'] else None,
                'area': -entry['objectives'][2] if entry['success'] else None,
                'success': entry['success']
            }
            writer.writerow(row)
            
