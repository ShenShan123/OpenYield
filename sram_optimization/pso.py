"""
SRAM Circuit Optimization using PSO Algorithm
使用PSO算法的SRAM电路优化

This file implements a Particle Swarm Optimization algorithm for SRAM parameter optimization.
该文件实现了用于SRAM参数优化的粒子群优化算法。
"""

import os
import sys
import numpy as np
import torch
import time
import warnings

warnings.filterwarnings('ignore')

# Setup paths first - before importing from sram_optimization
# 首先设置路径 - 在从sram_optimization导入之前
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import utilities from exp_utils
# 从exp_utils导入工具函数
from sram_optimization.exp_utils import (
    seed_set, create_directories, evaluate_sram, ModifiedSRAMParameterSpace,
    BaseOptimizer, get_default_initial_params, run_initial_evaluation
)


class PSOOptimizer(BaseOptimizer):
    """
    PSO optimizer class
    PSO优化器类
    """
    
    def __init__(self, parameter_space, num_objectives=3, num_constraints=2, 
                 initial_result=None, initial_params=None):
        """
        Initialize PSO optimizer
        初始化PSO优化器
        """
        super().__init__(parameter_space, "PSO", num_objectives, num_constraints, 
                         initial_result, initial_params)

    def evaluate(self, x):
        """
        Evaluate single parameter setting for PSO optimization - updated to use Merit
        评估PSO优化的单个参数设置 - 更新以使用Merit
        """
        # Ensure x is numpy array type
        # 确保x是numpy数组类型
        x = np.array(x) if not isinstance(x, np.ndarray) else x

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
            if sim_counter >= population_size:  # Don't update position in first round
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
                fitness = -result['merit']  # Negative Merit as fitness (minimization problem)

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

                # Record results
                # 记录结果
                self.observe(particles[particle_idx], objectives, constraints, result, success, sim_counter, "PSO")

                print(f"Current Merit: {result['merit']:.6e}")
                print(f"Best Merit: {-g_best_fitness:.6e}")
                print(f"当前Merit: {result['merit']:.6e}")
                print(f"最佳Merit: {-g_best_fitness:.6e}")
            else:
                # Failed simulations also count towards iteration count, but don't update optimal solutions
                # 失败的仿真也计入迭代次数，但不更新最优解
                self.observe(particles[particle_idx], objectives, constraints, result, success, sim_counter, "PSO")
                print("Evaluation failed")
                print("评估失败")

            # Record current iteration
            # 记录当前迭代
            sim_counter += 1

            # Save results every 10 simulations
            # 每10次仿真保存一次结果
            if sim_counter % 10 == 0:
                self.save_results()

        # Save final results
        # 保存最终结果
        self.save_results()
        print("\n===== PSO optimization completed =====")
        print("\n===== PSO优化完成 =====")
        return self.get_best_merit()


# Main function
# 主函数
def main():
    """
    Main function to run PSO optimization
    运行PSO优化的主函数
    """
    print("===== SRAM PSO Optimization =====")
    print("===== SRAM PSO优化 =====")

    # Create directories
    # 创建目录
    create_directories()

    # Create parameter space
    # 创建参数空间
    parameter_space = ModifiedSRAMParameterSpace()

    # Get initial parameters and run evaluation
    # 获取初始参数并运行评估
    initial_params = get_default_initial_params()
    initial_result, initial_params = run_initial_evaluation(parameter_space, initial_params)

    # Create PSO optimizer, pass initial parameters and results
    # 创建PSO优化器，传递初始参数和结果
    optimizer = PSOOptimizer(
        parameter_space,
        initial_result=initial_result,
        initial_params=initial_params
    )

    # Run PSO optimization
    # 运行PSO优化
    best_result = optimizer.run_optimization(max_iter=400, population_size=20)

    # Print final results
    # 打印最终结果
    optimizer.print_final_results()


if __name__ == "__main__":
    # Set random seed for reproducibility
    # 设置随机种子以确保可重现性
    SEED = 1
    seed_set(seed=SEED)
    main()
    