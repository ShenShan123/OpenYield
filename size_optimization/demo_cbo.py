"""
SRAM Circuit Optimization using Constrained Bayesian Optimization
使用约束贝叶斯优化的SRAM电路优化

This file implements a three-objective constrained Bayesian optimization algorithm 
for SRAM parameter optimization.
该文件实现了用于SRAM参数优化的三目标约束贝叶斯优化算法。
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ConstrainedExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from matplotlib import ticker
import random
import warnings
import csv

# Import test_sram_array function
# 导入test_sram_array函数
import sys
from pathlib import Path

# Get current file directory
# 获取当前文件目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add project root directory to Python path
# 将项目根目录添加到Python路径
project_root = os.path.dirname(current_dir)  # Assume exp_code is directly under project root
sys.path.append(project_root)

# Import utilities from exp_utils
# 从exp_utils导入工具函数
from size_optimization.exp_utils import (
    seed_set, create_directories, evaluate_sram, ModifiedSRAMParameterSpace,
    OptimizationLogger, save_pareto_front, save_best_result, plot_merit_history,
    plot_pareto_frontier, update_pareto_front, save_optimization_history
)
from utils import estimate_bitcell_area

warnings.filterwarnings('ignore')

# Set environment variables
# 设置环境变量
os.environ['PATH'] = os.path.join(os.path.dirname(sys.executable), 'Library', 'bin') + os.pathsep + os.environ['PATH']


# Define parameter space
# 定义参数空间
class SRAMParameterSpace:
    def __init__(self, config_path="config_sram.yaml"):  # 添加config_path参数
        """
        Initialize SRAM parameter space
        初始化SRAM参数空间
        """
        # 添加配置文件支持
        try:
            self._modified_space = ModifiedSRAMParameterSpace(config_path)
            self.bounds = self._modified_space.bounds
            # 从配置中获取参数信息
            self.base_pd_width = self._modified_space.base_pd_width
            self.base_pu_width = self._modified_space.base_pu_width
            self.base_pg_width = self._modified_space.base_pg_width
            self.base_length_nm = self._modified_space.base_length_nm
        except:
            # 如果配置加载失败，使用原始默认值
            print("Warning: Config loading failed, using default parameters")
            # Reference parameter values
            # 参考参数值
            self.base_pd_width = 0.205e-6
            self.base_pu_width = 0.09e-6
            self.base_pg_width = 0.135e-6
            self.base_length_nm = 50

            # Parameter ranges
            # 参数范围
            self.nmos_models = ['NMOS_VTL', 'NMOS_VTG', 'NMOS_VTH']
            self.pmos_models = ['PMOS_VTL', 'PMOS_VTG', 'PMOS_VTH']

            # Width ranges - range coefficients can be modified here
            # 宽度范围 - 范围系数可以在这里修改
            self.pd_width_coef_min = 0.5
            self.pd_width_coef_max = 1.5
            self.pd_width_min = self.base_pd_width * self.pd_width_coef_min
            self.pd_width_max = self.base_pd_width * self.pd_width_coef_max

            self.pu_width_coef_min = 0.5
            self.pu_width_coef_max = 1.5
            self.pu_width_min = self.base_pu_width * self.pu_width_coef_min
            self.pu_width_max = self.base_pu_width * self.pu_width_coef_max

            self.pg_width_coef_min = 0.5
            self.pg_width_coef_max = 1.5
            self.pg_width_min = self.base_pg_width * self.pg_width_coef_min
            self.pg_width_max = self.base_pg_width * self.pg_width_coef_max

            self.length_coef_min = 0.6
            self.length_coef_max = 2.0
            self.length_min = self.base_length_nm * 1e-9 * self.length_coef_min
            self.length_max = self.base_length_nm * 1e-9 * self.length_coef_max

            # Create optimization bounds
            # 创建优化边界
            lower_bounds = [0, 0, 0, 0, 0, 0]
            upper_bounds = [1, 1, 1, 1, 1, 1]
            self.bounds = torch.tensor([lower_bounds, upper_bounds], dtype=torch.float)

    def convert_params(self, x):
        """
        Convert normalized parameters [0,1] to actual parameter values
        将归一化参数[0,1]转换为实际参数值
        """
        # 如果有配置文件支持则使用，否则使用原始逻辑
        if hasattr(self, '_modified_space'):
            return self._modified_space.convert_params(x)
        
        # 原始转换逻辑
        params = {}
        
        # Convert model parameters
        # 转换模型参数
        nmos_idx = int(x[0].item() * len(self.nmos_models))
        nmos_idx = min(nmos_idx, len(self.nmos_models) - 1)
        params['nmos_model_name'] = self.nmos_models[nmos_idx]
        
        pmos_idx = int(x[1].item() * len(self.pmos_models))
        pmos_idx = min(pmos_idx, len(self.pmos_models) - 1)
        params['pmos_model_name'] = self.pmos_models[pmos_idx]
        
        # Convert width parameters
        # 转换宽度参数
        params['pd_width'] = self.pd_width_min + x[2].item() * (self.pd_width_max - self.pd_width_min)
        params['pu_width'] = self.pu_width_min + x[3].item() * (self.pu_width_max - self.pu_width_min)
        params['pg_width'] = self.pg_width_min + x[4].item() * (self.pg_width_max - self.pg_width_min)
        
        # Convert length parameter
        # 转换长度参数
        params['length'] = self.length_min + x[5].item() * (self.length_max - self.length_min)
        params['length_nm'] = params['length'] * 1e9
        
        return params

    def print_params(self, params):
        """
        Print parameter information
        打印参数信息
        """
        # 如果有配置文件支持则使用，否则使用原始逻辑
        if hasattr(self, '_modified_space'):
            return self._modified_space.print_params(params)
        
        # 原始打印逻辑
        print("Parameters / 参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")


# Constrained optimizer
# 约束优化器
class ConstrainedBayesianOptimizer:
    def __init__(self, parameter_space, num_objectives=3, num_constraints=2, initial_result=None, initial_params=None):
        """
        Initialize constrained Bayesian optimizer
        初始化约束贝叶斯优化器
        """
        self.parameter_space = parameter_space
        self.bounds = parameter_space.bounds
        self.dim = self.bounds.shape[1]
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints

        # Initialize training data
        # 初始化训练数据
        self.train_x = torch.zeros((0, self.dim))
        self.train_obj = torch.zeros((0, self.num_objectives))
        self.train_con = torch.zeros((0, self.num_constraints))

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

        # Set initial feasible point Merit and results
        # 设置初始可行点Merit和结果
        if initial_result:
            # Extract values from initial result
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
                
                self.initial_merit = np.log10(initial_min_snm / (initial_max_power * np.sqrt(initial_area)))
                print(f"Initial Merit: {self.initial_merit:.6e}")

        # Initialize Merit tracking state
        # 初始化Merit跟踪状态
        self.best_merit = float('-inf')
        self.best_params = None
        self.best_result = None

        # Initialize history data
        # 初始化历史数据
        self.iteration_history = []
        self.best_history = []

    def suggest(self, n_suggestions=1):
        """
        Generate next batch of points to evaluate
        生成下一批要评估的点
        """
        if len(self.train_x) < 10:
            print(f"Training set size: {len(self.train_x)}, using quasi-random sequence")
            print(f"训练集大小: {len(self.train_x)}, 使用准随机序列")

            soboleng = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True)

            # Add debug information
            # 添加调试信息
            points = soboleng.draw(n_suggestions)
            print(f"Generated points: {points.tolist()}")
            print(f"生成的点: {points.tolist()}")

            # Ensure not returning all-zero vector
            # 确保不返回全零向量
            if torch.allclose(points, torch.zeros_like(points), atol=1e-3):
                print("Detected all-zero vector, switching to uniform random generation")
                print("检测到全零向量，切换到均匀随机生成")
                points = torch.rand(n_suggestions, self.dim)

            return points
        else:
            # Use Bayesian optimization
            # 使用贝叶斯优化
            try:
                # Construct GP model
                # 构建GP模型
                gp = SingleTaskGP(self.train_x, self.train_obj.mean(dim=-1, keepdim=True))
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_model(mll)

                # Use Expected Improvement acquisition function
                # 使用期望改进获取函数
                from botorch.acquisition import ExpectedImprovement
                acq_func = ExpectedImprovement(gp, self.train_obj.mean(dim=-1).max())

                # Optimize acquisition function
                # 优化获取函数
                bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
                candidate, acq_value = optimize_acqf(
                    acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=200,
                )

                print(f"BO suggested point: {candidate.tolist()}")
                print(f"Acquisition value: {acq_value.item():.6e}")
                print(f"BO建议点: {candidate.tolist()}")
                print(f"获取函数值: {acq_value.item():.6e}")

                return candidate

            except Exception as e:
                print(f"BO suggestion failed: {e}, using random generation")
                print(f"BO建议失败: {e}, 使用随机生成")
                return torch.rand(n_suggestions, self.dim)

    def observe(self, x, objectives, constraints, result, success, iteration, opt_type="BO"):
        """
        Record observation results and update model
        记录观察结果并更新模型
        """
        # Record all evaluation results
        # 记录所有评估结果
        self.all_x.append(x)
        self.all_objectives.append(objectives)
        self.all_constraints.append(constraints)
        self.all_results.append(result)
        self.all_success.append(success)

        if success and result:
            merit = result['merit']
            self.all_merit.append(merit)

            # Update training data for BO
            # 为BO更新训练数据
            x_tensor = torch.tensor(x, dtype=torch.float).unsqueeze(0)
            obj_tensor = torch.tensor(objectives, dtype=torch.float).unsqueeze(0)
            con_tensor = torch.tensor(constraints, dtype=torch.float).unsqueeze(0)

            self.train_x = torch.cat([self.train_x, x_tensor])
            self.train_obj = torch.cat([self.train_obj, obj_tensor])
            self.train_con = torch.cat([self.train_con, con_tensor])

            # Update best result
            # 更新最佳结果
            if merit > self.best_merit:
                self.best_merit = merit
                self.best_params = self.parameter_space.convert_params(torch.tensor(x, dtype=torch.float32))
                self.best_result = result

        # Record history
        # 记录历史
        self.best_history.append(self.best_merit if self.best_merit != float('-inf') else float('-inf'))


# Random search phase
# 随机搜索阶段
def random_search(parameter_space, num_iterations=20):
    """
    Random search phase
    随机搜索阶段
    """
    print(f"===== Starting {num_iterations} rounds of random search (RS) =====")
    print(f"===== 开始 {num_iterations} 轮随机搜索 (RS) =====")

    # Initialize optimizer
    # 初始化优化器
    optimizer = ConstrainedBayesianOptimizer(parameter_space)

    for i in range(num_iterations):
        print(f"\n===== RS iteration {i + 1}/{num_iterations} =====")
        print(f"\n===== RS迭代 {i + 1}/{num_iterations} =====")

        # Generate random point
        # 生成随机点
        x = torch.rand(optimizer.dim)

        # Map parameters to actual values
        # 将参数映射到实际值
        params = parameter_space.convert_params(x)
        parameter_space.print_params(params)

        # Evaluate parameter performance
        # 评估参数性能
        start_time = time.time()
        objectives, constraints, result, success = evaluate_sram(params)
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        print(f"评估用时: {end_time - start_time:.2f} 秒")

        if success:
            print("Simulation successful! Results:")
            print("仿真成功！结果:")
            print(f"Min SNM = {-objectives[0]:.6f}")
            print(f"Max power = {objectives[1]:.6e}")
            print(f"Area = {objectives[2]*1e12:.2f} µm²")
            print(f"Constraints: {['satisfied' if c <= 0 else 'violated' for c in constraints]}")
            print(f"约束: {['满足' if c <= 0 else '违反' for c in constraints]}")
            if result:
                print(f"Merit = {result['merit']:.6e}")
        else:
            print("Simulation failed, penalty values assigned")
            print("仿真失败，分配惩罚值")

        # Observe results
        # 观察结果
        optimizer.observe(x, objectives, constraints, result, success, i, "RS")

    print("\n===== Random search completed =====")
    print("\n===== 随机搜索完成 =====")
    return optimizer


# Bayesian optimization phase
# 贝叶斯优化阶段
def bayesian_optimization(optimizer, parameter_space, num_iterations=380):
    """
    Bayesian optimization phase
    贝叶斯优化阶段
    """
    print(f"===== Starting {num_iterations} rounds of Bayesian optimization (BO) =====")
    print(f"===== 开始 {num_iterations} 轮贝叶斯优化 (BO) =====")

    # Track repeated suggestions
    # 跟踪重复建议
    recent_suggestions = []
    max_history = 10
    repeat_count = 0
    max_repeat_allowed = 3

    for i in range(num_iterations):
        print(f"\n===== BO iteration {i + 1}/{num_iterations} =====")
        print(f"\n===== BO迭代 {i + 1}/{num_iterations} =====")

        # Get next evaluation point
        # 获取下一个评估点
        next_x = optimizer.suggest()[0]

        # Check if too similar to recent suggestions
        # 检查是否与最近的建议太相似
        is_repeat = False
        for prev_x in recent_suggestions:
            if torch.norm(next_x - prev_x) < 0.05:  # If Euclidean distance is very small
                repeat_count += 1
                is_repeat = True
                print(f"Warning: Current suggestion very similar to previous suggestions (repeat count: {repeat_count})")
                print(f"警告：当前建议与之前的建议非常相似（重复次数：{repeat_count}）")
                break

        # If too many repeats, force random exploration
        # 如果重复太多次，强制随机探索
        if is_repeat and repeat_count >= max_repeat_allowed:
            print(f"Consecutive repeats {repeat_count} times, switching to random exploration mode...")
            print(f"连续重复 {repeat_count} 次，切换到随机探索模式...")
            next_x = torch.rand(optimizer.dim)
            repeat_count = 0
        elif not is_repeat:
            repeat_count = 0

        # Update suggestion history
        # 更新建议历史
        recent_suggestions.append(next_x)
        if len(recent_suggestions) > max_history:
            recent_suggestions.pop(0)  # Remove oldest record

        # Map parameters to actual values
        # 将参数映射到实际值
        params = parameter_space.convert_params(next_x)
        parameter_space.print_params(params)

        # Evaluate parameter performance
        # 评估参数性能
        start_time = time.time()
        objectives, constraints, result, success = evaluate_sram(params)
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        print(f"评估用时: {end_time - start_time:.2f} 秒")

        if success:
            print("Simulation successful! Results:")
            print("仿真成功！结果:")
            print(f"Min SNM = {-objectives[0]:.6f}")
            print(f"Max power = {objectives[1]:.6e}")
            print(f"Area = {objectives[2]*1e12:.2f} µm²")
            print(f"Constraints: {['satisfied' if c <= 0 else 'violated' for c in constraints]}")
            print(f"约束: {['满足' if c <= 0 else '违反' for c in constraints]}")
            if result:
                print(f"Merit = {result['merit']:.6e}")
        else:
            print("Simulation failed, penalty values assigned")
            print("仿真失败，分配惩罚值")

        # Observe results (index after RS iterations end)
        # 观察结果（RS迭代结束后的索引）
        iteration = i + 20  # Assume first 20 times are RS
        optimizer.observe(next_x, objectives, constraints, result, success, iteration, "BO")

    print("\n===== Bayesian optimization completed =====")
    print("\n===== 贝叶斯优化完成 =====")
    return optimizer


# Main function
# 主函数
def main(config_path="config_sram.yaml"):  # 添加config_path参数
    """
    Main function to run CBO optimization
    运行CBO优化的主函数
    """
    print("===== SRAM optimization using CBO =====")
    print("===== 使用CBO的SRAM优化 =====")

    # Create directories
    # 创建目录
    create_directories()

    # Create parameter space - use original class with config support
    # 创建参数空间 - 使用支持配置的原始类
    parameter_space = SRAMParameterSpace(config_path)  # 支持配置文件

    # Define initial parameters (same as in main.py)
    # 定义初始参数（与main.py中相同）
    initial_params = {
        'nmos_model_name': 'NMOS_VTG',
        'pmos_model_name': 'PMOS_VTG',
        'pd_width': 0.205e-6,
        'pu_width': 0.09e-6,
        'pg_width': 0.135e-6,
        'length': 50e-9,
        'length_nm': 50
    }
    
    print("Running initial point simulation to get baseline performance...")
    print("运行初始点仿真以获得基准性能...")
    
    # Use evaluate_sram function to evaluate initial parameters
    # 使用evaluate_sram函数评估初始参数
    objectives, constraints, initial_result, success = evaluate_sram(initial_params)
    
    if not success:
        print("Warning: Initial point simulation failed, using default values as initial point")
        print("警告：初始点仿真失败，使用默认值作为初始点")
        # Use default values as fallback
        # 使用默认值作为后备
        initial_result = {
            'hold_snm': {'success': True, 'snm': 0.30173446708423357},
            'read_snm': {'success': True, 'snm': 0.12591724230394877},
            'write_snm': {'success': True, 'snm': 0.3732610863628419},
            'read': {'success': True, 'delay': 2.0883543988703797e-10, 'power': 4.024476625792127e-05},
            'write': {'success': True, 'delay': 6.086260190977158e-11, 'power': 3.975272388991992e-05}
        }
    else:
        print(f"Initial point simulation successful!")
        print(f"初始点仿真成功！")
        print(f"SNM: Hold={initial_result['hold_snm']:.4f}, Read={initial_result['read_snm']:.4f}, Write={initial_result['write_snm']:.4f}")
        print(f"Delay: Read={initial_result['read_delay']*1e12:.2f}ps, Write={initial_result['write_delay']*1e12:.2f}ps")
        print(f"Power: Read={initial_result['read_power']:.2e}W, Write={initial_result['write_power']:.2e}W")

    # Run random search phase
    # 运行随机搜索阶段
    optimizer = random_search(parameter_space, num_iterations=2)

    # Run Bayesian optimization phase
    # 运行贝叶斯优化阶段
    optimizer = bayesian_optimization(optimizer, parameter_space, num_iterations=3)

    # Output best results
    # 输出最佳结果
    print("\n===== CBO Optimization Best Results =====")
    print("\n===== CBO优化最佳结果 =====")
    if optimizer.best_merit != float('-inf'):
        print(f"Best Merit: {optimizer.best_merit:.6e}")
        print("Best parameters:")
        print("最佳参数:")
        parameter_space.print_params(optimizer.best_params)
        if optimizer.best_result:
            print(f"Min SNM: {optimizer.best_result['min_snm']:.6f}")
            print(f"Max power: {optimizer.best_result['max_power']:.6e}")
            print(f"Area: {optimizer.best_result['area']*1e12:.2f} µm²")
            print(f"最小SNM: {optimizer.best_result['min_snm']:.6f}")
            print(f"最大功耗: {optimizer.best_result['max_power']:.6e}")
            print(f"面积: {optimizer.best_result['area']*1e12:.2f} µm²")
        
        # Return result
        # 返回结果
        return {
            'params': optimizer.best_params,
            'merit': optimizer.best_merit,
            'result': optimizer.best_result,
            'iteration': 400
        }
    else:
        print("No valid solution found")
        print("未找到有效解")
        return {
            'params': None,
            'merit': None,
            'result': None,
            'iteration': -1
        }


if __name__ == "__main__":
    # Set random seed for reproducibility
    # 设置随机种子以确保可重现性
    SEED = 1
    seed_set(seed=SEED)
    main()
