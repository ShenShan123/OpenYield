"""
SRAM Circuit Optimization using PSO Algorithm
使用PSO算法的SRAM电路优化

This file implements a Particle Swarm Optimization algorithm for SRAM parameter optimization.
该文件实现了用于SRAM参数优化的粒子群优化算法。
"""

import os
import time
import numpy as np
import torch
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Import path handling
# 导入路径处理
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import utilities from exp_utils
# 从exp_utils导入工具函数
from sram_optimization.exp_utils import (
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


# PSO optimizer class
# PSO优化器类
class PSOOptimizer:
    def __init__(self, parameter_space, num_objectives=3, num_constraints=2, initial_result=None, initial_params=None):
        """
        Initialize PSO optimizer
        初始化PSO优化器
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
            # Extract values from initial result
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

        # Initialize history data
        # 初始化历史数据
        self.iteration_history = []  # Record evaluation results for each iteration
        self.best_history = []  # Record best Merit for each iteration

    def evaluate(self, x):
        """
        Evaluate single parameter setting for PSO optimization
        评估PSO优化的单个参数设置
        """
        # Ensure x is numpy array type
        # 确保x是numpy数组类型
        x = np.array(x) if not isinstance(x, np.ndarray) else x

        # Convert to torch.Tensor
        # 转换为torch.Tensor
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

            # Return negative Merit, because PSO is a minimization problem
            # 返回负Merit，因为PSO是最小化问题
            return -merit
        else:
            # If evaluation fails, return a large penalty value
            # 如果评估失败，返回大的惩罚值
            return 1e9

    def run_optimization(self, max_iter=100, population_size=30):
        """
        Run PSO optimization
        运行PSO优化
        """
        print(f"===== Starting PSO optimization, maximum simulations: {max_iter} =====")
        print(f"===== 开始PSO优化，最大仿真次数: {max_iter} =====")

        # Parameter ranges
        # 参数范围
        lb = np.zeros(self.dim)
        ub = np.ones(self.dim)

        # PSO parameters
        # PSO参数
        w = 0.8  # Inertia weight
        c1 = 0.5  # Individual learning factor
        c2 = 0.5  # Social learning factor

        # Initialize particle swarm
        # 初始化粒子群
        particles = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (population_size, self.dim))

        # Initialize fitness
        # 初始化适应度
        p_best = particles.copy()
        p_best_fitness = np.ones(population_size) * 1e9
        g_best = particles[0].copy()
        g_best_fitness = 1e9

        # Best merit tracking
        # 最佳merit跟踪
        best_merit = float('-inf')
        best_params = None
        best_result = None

        # Counter
        # 计数器
        sim_counter = 0

        # Each particle computes one simulation, each simulation counts as one iteration
        # 每个粒子计算一次仿真，每次仿真计为一次迭代
        while sim_counter < max_iter:
            # Randomly select a particle
            # 随机选择一个粒子
            particle_idx = sim_counter % population_size

            # Update particle position and velocity
            # 更新粒子位置和速度
            if sim_counter >= population_size:
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                velocities[particle_idx] = (w * velocities[particle_idx] +
                                            c1 * r1 * (p_best[particle_idx] - particles[particle_idx]) +
                                            c2 * r2 * (g_best - particles[particle_idx]))

                particles[particle_idx] = particles[particle_idx] + velocities[particle_idx]
                particles[particle_idx] = np.clip(particles[particle_idx], lb, ub)

            # Evaluate current particle
            # 评估当前粒子
            print(f"\n===== PSO simulation {sim_counter + 1}/{max_iter} =====")
            print(f"Evaluating particle {particle_idx + 1}/{population_size}")
            print(f"\n===== PSO仿真 {sim_counter + 1}/{max_iter} =====")
            print(f"评估粒子 {particle_idx + 1}/{population_size}")

            x_tensor = torch.tensor(particles[particle_idx], dtype=torch.float32)
            params = self.parameter_space.convert_params(x_tensor)

            print("Current parameters:")
            print("当前参数:")
            self.parameter_space.print_params(params)

            # Detailed evaluation, get complete results
            # 详细评估，获得完整结果
            start_time = time.time()
            objectives, constraints, result, success = evaluate_sram(params)
            end_time = time.time()
            print(f"Evaluation time: {end_time - start_time:.2f} seconds")
            print(f"评估用时: {end_time - start_time:.2f} 秒")

            # Calculate fitness
            # 计算适应度
            if success and result:
                # Updated to use Merit
                # 更新以使用Merit
                merit = result['merit']
                fitness = -merit  # Negative Merit as fitness (minimization problem)

                # Update individual best
                # 更新个体最优
                if fitness < p_best_fitness[particle_idx]:
                    p_best[particle_idx] = particles[particle_idx].copy()
                    p_best_fitness[particle_idx] = fitness

                    # Update global best
                    # 更新全局最优
                    if fitness < g_best_fitness:
                        g_best = particles[particle_idx].copy()
                        g_best_fitness = fitness
                        print(f"Found new global best: {-g_best_fitness:.6e}")
                        print(f"发现新的全局最优: {-g_best_fitness:.6e}")

                # Update overall best tracking
                # 更新整体最佳跟踪
                if merit > best_merit:
                    best_merit = merit
                    best_params = params
                    best_result = result

                # Record results
                # 记录结果
                self.all_x.append(particles[particle_idx])
                self.all_objectives.append(objectives)
                self.all_constraints.append(constraints)
                self.all_results.append(result)
                self.all_success.append(success)
                self.all_merit.append(merit)

                print(f"Current Merit: {merit:.6e}")
                print(f"当前Merit: {merit:.6e}")
                print(f"Best Merit: {-g_best_fitness:.6e}")
                print(f"最佳Merit: {-g_best_fitness:.6e}")
            else:
                # Failed simulations also count towards iteration count, but don't update optimal solutions
                # 失败的仿真也计入迭代次数，但不更新最优解
                self.all_x.append(particles[particle_idx])
                self.all_objectives.append(objectives)
                self.all_constraints.append(constraints)
                self.all_results.append(result)
                self.all_success.append(success)
                print("Evaluation failed")
                print("评估失败")

            # Record current iteration
            # 记录当前迭代
            sim_counter += 1

        print("\n===== PSO optimization completed =====")
        print("\n===== PSO优化完成 =====")
        
        # Return best result
        # 返回最佳结果
        if best_merit != float('-inf'):
            return {
                'params': best_params,
                'merit': best_merit,
                'result': best_result,
                'iteration': max_iter
            }
        else:
            return {
                'params': None,
                'merit': None,
                'result': None,
                'iteration': -1
            }


# Main function
# 主函数
def main(config_path="config_sram.yaml"):  # 添加config_path参数
    """
    Main function to run PSO optimization
    运行PSO优化的主函数
    """
    print("===== SRAM optimization using PSO =====")
    print("===== 使用PSO的SRAM优化 =====")

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

    # Create PSO optimizer, pass initial parameters and results
    # 创建PSO优化器，传递初始参数和结果
    optimizer = PSOOptimizer(
        parameter_space,
        initial_result=initial_result,  # Use direct format
        initial_params=initial_params
    )

    # Run PSO optimization
    # 运行PSO优化
    best_result = optimizer.run_optimization(max_iter=400, population_size=20)

    # Output best results
    # 输出最佳结果
    print("\n===== PSO Optimization Best Results =====")
    print("\n===== PSO优化最佳结果 =====")
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
