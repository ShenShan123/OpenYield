"""
SRAM Circuit Optimization using Simulated Annealing Algorithm
使用模拟退火算法的SRAM电路优化

This file implements a Simulated Annealing optimization algorithm for SRAM parameter optimization.
该文件实现了用于SRAM参数优化的模拟退火优化算法。
"""

import os
import time
import numpy as np
import torch
import random
import warnings
from pathlib import Path
import traceback

warnings.filterwarnings('ignore')

# Import path handling
# 导入路径处理
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import utilities from exp_utils
# 从exp_utils导入工具函数
from size_optimization.exp_utils import (
    seed_set, create_directories, evaluate_sram, ModifiedSRAMParameterSpace,
    OptimizationLogger, save_pareto_front, save_best_result, plot_merit_history,
    plot_pareto_frontier, update_pareto_front, save_optimization_history
)
from utils import estimate_bitcell_area


def format_initial_result(result):
    """
    Convert evaluate_sram result format to expected format
    将evaluate_sram结果格式转换为期望格式
    """
    if result is None:
        return None
    
    formatted_result = {
        'hold_snm': {'success': True, 'snm': result['hold_snm']},
        'read_snm': {'success': True, 'snm': result['read_snm']},
        'write_snm': {'success': True, 'snm': result['write_snm']},
        'read': {'success': True, 'delay': result['read_delay'], 
                 'power': abs(result['read_power'])},
        'write': {'success': True, 'delay': result['write_delay'], 
                  'power': abs(result['write_power'])}
    }
    return formatted_result


# SA optimizer class
# SA优化器类
class SAOptimizer:
    def __init__(self, parameter_space, num_objectives=3, num_constraints=2, initial_result=None, initial_params=None):
        """
        Initialize SA optimizer
        初始化SA优化器
        """
        self.parameter_space = parameter_space
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

        # Set initial feasible point Merit and results
        # 设置初始可行点Merit和结果
        if initial_result:
            # Extract values from initial results: handle both formats
            # 从初始结果中提取值
            if isinstance(initial_result.get('hold_snm'), dict):
                # New format: {'hold_snm': {'snm': value}}
                initial_min_snm = min(
                    initial_result['hold_snm']['snm'],
                    initial_result['read_snm']['snm'],
                    initial_result['write_snm']['snm']
                )
                initial_max_power = max(
                    initial_result['read']['power'],
                    initial_result['write']['power']
                )
            else:
                # Direct format: {'hold_snm': value}
                initial_min_snm = min(
                    initial_result['hold_snm'],
                    initial_result['read_snm'],
                    initial_result['write_snm']
                )
                initial_max_power = max(
                    abs(initial_result['read_power']),
                    abs(initial_result['write_power'])
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
        self.best_x = None
        self.best_result = None

        # Initialize history data
        # 初始化历史数据
        self.iteration_history = []  # Record evaluation results for each iteration
        self.best_history = []  # Record best Merit for each iteration

    def evaluate(self, x):
        """
        Evaluate single parameter setting for SA optimization
        评估SA优化的单个参数设置
        """
        # Ensure x is numpy array type
        # 确保x是numpy数组类型
        x = np.array(x) if not isinstance(x, np.ndarray) else x

        # Constrain x to [0,1] range
        # 将x限制在[0,1]范围内
        x = np.clip(x, 0, 1)

        # Convert to torch.Tensor (if needed)
        # 转换为torch.Tensor（如果需要）
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # Convert parameters
        # 转换参数
        params = self.parameter_space.convert_params(x_tensor)

        # Evaluate SRAM performance
        # 评估SRAM性能
        objectives, constraints, result, success = evaluate_sram(params)

        # Calculate optimization objective: Merit = log10(min_snm / (max_power * (area**0.5)))
        # 计算优化目标：Merit = log10(min_snm / (max_power * (area**0.5)))
        if success and result:
            merit = result['merit']

            # Return negative Merit, because SA is a minimization problem
            # 返回负Merit，因为SA是最小化问题
            return -merit
        else:
            # If evaluation fails, return a large penalty value
            # 如果评估失败，返回大的惩罚值
            return 1e9

    def run_optimization(self, max_iter=10, T_max=1000, T_min=1e-7):
        """
        Run SA optimization - modified to count one simulation as one iteration
        运行SA优化 - 修改为一次仿真计为一次迭代
        """
        print(f"===== Starting SA optimization, maximum simulations: {max_iter} =====")
        print(f"===== 开始SA优化，最大仿真次数: {max_iter} =====")

        # Parameter ranges
        # 参数范围
        lb = np.zeros(self.dim)
        ub = np.ones(self.dim)

        # Optimized temperature settings
        # 优化的温度设置
        T = T_max  # Higher initial temperature
        alpha = 0.98  # Slower cooling rate

        # Find better initial point
        # 寻找更好的初始点
        print("Searching for valid initial point...")
        print("搜索有效的初始点...")
        best_init_x = None
        best_init_merit = float('-inf')
        best_init_success = False

        # Try to find a good starting point (try at most 10 initial points)
        # 尝试找到一个好的起始点（最多尝试10个初始点）
        for i in range(min(10, max_iter // 10)):
            init_x = np.random.uniform(lb, ub)
            x_tensor = torch.tensor(init_x, dtype=torch.float32)
            params = self.parameter_space.convert_params(x_tensor)

            print(f"Evaluating initial point {i + 1}/10:")
            print(f"评估初始点 {i + 1}/10:")
            self.parameter_space.print_params(params)

            objectives, constraints, result, success = evaluate_sram(params)

            # If it's a successful simulation, check if it's the best initial point
            # 如果是成功的仿真，检查是否为最佳初始点
            if success and result:
                init_merit = result['merit']
                if init_merit > best_init_merit:
                    best_init_x = init_x.copy()
                    best_init_merit = init_merit
                    best_init_success = True
                    print(f"Found better initial point, Merit = {init_merit:.6e}")
                    print(f"找到更好的初始点，Merit = {init_merit:.6e}")

                    # If a valid point is found, record it first
                    # 如果找到有效点，先记录它
                    self.observe(init_x, objectives, constraints, result, success, i, "SA_init")

        # Determine starting point
        # 确定起始点
        if best_init_success:
            current_x = best_init_x.copy()
            print(f"Using found best initial point, Merit = {best_init_merit:.6e}")
            print(f"使用找到的最佳初始点，Merit = {best_init_merit:.6e}")
        else:
            current_x = np.random.uniform(lb, ub)
            print("No valid initial point found, using random initial point")
            print("未找到有效初始点，使用随机初始点")

        # Set current best solution
        # 设置当前最佳解
        best_x = current_x.copy()

        # Counter and state variables
        # 计数器和状态变量
        sim_counter = 10 if best_init_success else 0  # Consider simulations used for initial point search
        no_improve_count = 0  # Track consecutive non-improvement count

        # Main loop - each simulation counts as one iteration
        # 主循环 - 每次仿真计为一次迭代
        while sim_counter < max_iter:
            print(f"\n===== SA simulation {sim_counter + 1}/{max_iter} =====")
            print(f"Current temperature: {T:.6e}, No improvement count: {no_improve_count}")
            print(f"\n===== SA仿真 {sim_counter + 1}/{max_iter} =====")
            print(f"当前温度: {T:.6e}, 未改进次数: {no_improve_count}")

            # Generate new solution randomly near current solution
            # 在当前解附近随机生成新解
            new_x = current_x + np.random.normal(0, max(0.1, T / T_max), size=self.dim)  # Higher temperature means larger perturbation
            new_x = np.clip(new_x, lb, ub)  # Ensure within parameter range

            # Evaluate new solution
            # 评估新解
            x_tensor = torch.tensor(new_x, dtype=torch.float32)
            params = self.parameter_space.convert_params(x_tensor)

            print("Evaluating new solution:")
            print("评估新解:")
            self.parameter_space.print_params(params)

            start_time = time.time()
            objectives, constraints, result, success = evaluate_sram(params)
            end_time = time.time()
            print(f"Evaluation time: {end_time - start_time:.2f} seconds")
            print(f"评估用时: {end_time - start_time:.2f} 秒")

            # Record results
            # 记录结果
            self.observe(new_x, objectives, constraints, result, success, sim_counter, "SA")

            # Metropolis criterion to decide whether to accept new solution
            # Metropolis准则决定是否接受新解
            accepted = False

            if success and result:
                new_merit = result['merit']
                current_merit = self.best_merit if self.best_merit != float('-inf') else 0.0

                print(f"New solution Merit: {new_merit:.6e}")
                print(f"新解Merit: {new_merit:.6e}")

                # Metropolis acceptance criterion
                # Metropolis接受准则
                if new_merit > current_merit:
                    # Accept better solution
                    # 接受更好的解
                    accepted = True
                    current_x = new_x.copy()
                    no_improve_count = 0
                    print(f"Accept better solution: {new_merit:.6e}")
                    print(f"接受更好的解: {new_merit:.6e}")
                else:
                    # Accept worse solution with probability
                    # 以一定概率接受较差的解
                    delta = (current_merit - new_merit)  # Energy difference (Merit difference)
                    if T > 0:
                        prob = np.exp(-delta / T)
                        if np.random.random() < prob:
                            accepted = True
                            current_x = new_x.copy()
                            print(f"Accept worse solution with probability {prob:.6f}: {new_merit:.6e}")
                            print(f"以概率{prob:.6f}接受较差的解: {new_merit:.6e}")

                if not accepted:
                    no_improve_count += 1
                    print(f"Reject solution: {new_merit:.6e}")
                    print(f"拒绝解: {new_merit:.6e}")

            else:
                print("Simulation failed!")
                print("仿真失败！")
                no_improve_count += 1

            # Decrease temperature - cool down every 5 simulations, slower cooling
            # 降低温度 - 每5次仿真降温一次，较慢的冷却
            if sim_counter % 5 == 0:
                T = max(T * alpha, T_min)

            # Early stopping condition
            # 早停条件
            if T < T_min:
                print(f"Temperature reached minimum {T_min}, stopping optimization")
                print(f"温度达到最小值 {T_min}，停止优化")
                break

            # Restart mechanism: if no improvement for long time, reset state
            # 重启机制：如果长时间无改进，重置状态
            if no_improve_count >= 50:
                print("No improvement for long time, restarting search...")
                print("长时间无改进，重启搜索...")
                current_x = np.random.uniform(lb, ub)  # Re-initialize randomly
                T = T_max * 0.5  # Lower initial temperature, but keep it high to promote exploration
                no_improve_count = 0

            # Update counter
            # 更新计数器
            sim_counter += 1

        print("\n===== SA optimization completed =====")
        print("\n===== SA优化完成 =====")
        
        # Return best result
        # 返回最佳结果
        if self.best_merit != float('-inf'):
            best_params = self.parameter_space.convert_params(torch.tensor(self.best_x, dtype=torch.float32))
            return {
                'params': best_params,
                'merit': self.best_merit,
                'result': self.best_result,
                'iteration': max_iter
            }
        else:
            return {
                'params': None,
                'merit': None,
                'result': None,
                'iteration': -1
            }

    def observe(self, x, objectives, constraints, result, success, iteration, opt_type="SA"):
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

            # Update best result
            # 更新最佳结果
            if merit > self.best_merit:
                self.best_merit = merit
                self.best_x = x.copy()
                self.best_result = result

        # Record history
        # 记录历史
        self.best_history.append(self.best_merit if self.best_merit != float('-inf') else float('-inf'))


# Main function
# 主函数
def main(config_path="config_sram.yaml"):
    """
    Main function to run SA optimization
    运行SA优化的主函数
    """
    print("===== SRAM optimization using SA =====")
    print("===== 使用SA的SRAM优化 =====")

    # Create directories
    # 创建目录
    create_directories()

    # Create parameter space - use modified parameter space class
    # 创建参数空间 - 使用修改的参数空间类
    parameter_space = ModifiedSRAMParameterSpace(config_path)  # 支持配置文件

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
            'hold_snm': 0.30173446708423357,
            'read_snm': 0.12591724230394877,
            'write_snm': 0.3732610863628419,
            'read_delay': 2.0883543988703797e-10,
            'read_power': 4.024476625792127e-05,
            'write_delay': 6.086260190977158e-11,
            'write_power': 3.975272388991992e-05
        }
    else:
        print(f"Initial point simulation successful!")
        print(f"初始点仿真成功！")
        print(f"SNM: Hold={initial_result['hold_snm']:.4f}, Read={initial_result['read_snm']:.4f}, Write={initial_result['write_snm']:.4f}")
        print(f"Delay: Read={initial_result['read_delay']*1e12:.2f}ps, Write={initial_result['write_delay']*1e12:.2f}ps")
        print(f"Power: Read={initial_result['read_power']:.2e}W, Write={initial_result['write_power']:.2e}W")

    # Create SA optimizer, pass initial parameters and results
    # 创建SA优化器，传递初始参数和结果
    optimizer = SAOptimizer(
        parameter_space, 
        initial_result=initial_result,  # Pass direct format
        initial_params=initial_params
    )

    # Run SA optimization
    # 运行SA优化
    best_result = optimizer.run_optimization(max_iter=5, T_max=100, T_min=1e-3)

    # Output best results
    # 输出最佳结果
    print("\n===== SA Optimization Best Results =====")
    print("\n===== SA优化最佳结果 =====")
    if best_result['params'] is not None:
        print(f"Best Merit: {best_result['merit']:.6e}")
        print(f"Iteration count: {best_result['iteration']}")
        print("Best parameters:")
        print("最佳参数:")
        parameter_space.print_params(best_result['params'])
        if best_result['result']:
            print(f"Min SNM: {best_result['result']['min_snm']:.6f}")
            print(f"Max power: {best_result['result']['max_power']:.6e}")
            print(f"Area: {best_result['result']['area']*1e12:.2f} µm²")
            print(f"最小SNM: {best_result['result']['min_snm']:.6f}")
            print(f"最大功耗: {best_result['result']['max_power']:.6e}")
            print(f"面积: {best_result['result']['area']*1e12:.2f} µm²")
    else:
        print("No valid solution found")
        print("未找到有效解")

    return best_result  # 返回结果供总控文件使用


if __name__ == "__main__":
    # Set random seed for reproducibility
    # 设置随机种子以确保可重现性
    SEED = 1
    seed_set(seed=SEED)
    main()
