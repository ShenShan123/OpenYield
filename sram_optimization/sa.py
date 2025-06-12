"""
SRAM Circuit Optimization using Simulated Annealing Algorithm
使用模拟退火算法的SRAM电路优化

This file implements a Simulated Annealing optimization algorithm for SRAM parameter optimization.
该文件实现了用于SRAM参数优化的模拟退火优化算法。
"""

import os
import sys
import numpy as np
import torch
import time
import random
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


class SAOptimizer(BaseOptimizer):
    """
    SA optimizer class
    SA优化器类
    """
    
    def __init__(self, parameter_space, num_objectives=3, num_constraints=2, 
                 initial_result=None, initial_params=None):
        """
        Initialize SA optimizer
        初始化SA优化器
        """
        super().__init__(parameter_space, "SA", num_objectives, num_constraints, 
                         initial_result, initial_params)

    def evaluate(self, x):
        """
        Evaluate single parameter setting for SA optimization - updated to use Merit
        评估SA优化的单个参数设置 - 更新以使用Merit
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

    def run_optimization(self, max_iter=100, T_max=1000, T_min=1e-7):
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

                print(f"New solution Merit: {new_merit:.6e}, Current best Merit: {current_merit:.6e}")
                print(f"新解Merit: {new_merit:.6e}, 当前最佳Merit: {current_merit:.6e}")

                # If new solution is better, or accept worse solution by probability
                # 如果新解更好，或者按概率接受较差的解
                if new_merit > current_merit:
                    # New solution is better, accept directly
                    # 新解更好，直接接受
                    current_x = new_x.copy()
                    accepted = True
                    no_improve_count = 0
                    print("New solution is better, accepted")
                    print("新解更好，已接受")

                    # Update global optimum
                    # 更新全局最优
                    if new_merit > self.best_merit:
                        self.best_merit = new_merit
                        best_x = new_x.copy()
                        print(f"Update global optimum, Merit = {new_merit:.6e}")
                        print(f"更新全局最优，Merit = {new_merit:.6e}")
                else:
                    # New solution is worse, accept with certain probability
                    # 新解较差，以一定概率接受
                    p = np.exp((new_merit - current_merit) / T)
                    if np.random.random() < p:
                        current_x = new_x.copy()
                        accepted = True
                        print(f"Probabilistically accept worse solution, p = {p:.6f}")
                        print(f"概率性接受较差解，p = {p:.6f}")
                    else:
                        no_improve_count += 1
                        print(f"Reject worse solution, p = {p:.6f}")
                        print(f"拒绝较差解，p = {p:.6f}")
            else:
                print("Simulation failed, reject new solution")
                print("仿真失败，拒绝新解")
                no_improve_count += 1

            # Restart mechanism: if no improvement for long time, reset state
            # 重启机制：如果长时间无改进，重置状态
            if no_improve_count >= 50:
                print("No improvement for long time, restarting search...")
                print("长时间无改进，重启搜索...")
                current_x = np.random.uniform(lb, ub)  # Re-initialize randomly
                T = T_max * 0.5  # Lower initial temperature, but keep it high to promote exploration
                no_improve_count = 0

            # Decrease temperature - cool down every 5 simulations, slower cooling
            # 降低温度 - 每5次仿真降温一次，较慢的冷却
            if sim_counter % 5 == 0:
                T = max(T * alpha, T_min)

            # Update counter
            # 更新计数器
            sim_counter += 1

            # Save results every 10 simulations
            # 每10次仿真保存一次结果
            if sim_counter % 10 == 0:
                self.save_results()

        # Optimization ended, save final results
        # 优化结束，保存最终结果
        self.save_results()

        # If a valid solution is finally found, perform a final optimal parameter evaluation
        # 如果最终找到有效解，进行最终最优参数评估
        if self.best_merit > float('-inf'):
            print("\n===== Final evaluation of optimal solution =====")
            print("\n===== 最优解的最终评估 =====")
            x_tensor = torch.tensor(best_x, dtype=torch.float32)
            params = self.parameter_space.convert_params(x_tensor)

            print("Optimal parameters:")
            print("最优参数:")
            self.parameter_space.print_params(params)

            objectives, constraints, result, success = evaluate_sram(params)
            if success and result:
                print(f"Final optimal Merit: {result['merit']:.6e}")
                print(f"Min SNM: {result['min_snm']:.6f}")
                print(f"Max power: {result['max_power']:.6e}")
                print(f"Area: {result['area']*1e12:.2f} µm²")
                print(f"最终最优Merit: {result['merit']:.6e}")
                print(f"最小SNM: {result['min_snm']:.6f}")
                print(f"最大功耗: {result['max_power']:.6e}")
                print(f"面积: {result['area']*1e12:.2f} µm²")

        print("\n===== SA optimization completed =====")
        print("\n===== SA优化完成 =====")
        return self.get_best_merit()


# Main function
# 主函数
def main():
    """
    Main function to run SA optimization
    运行SA优化的主函数
    """
    print("===== SRAM SA Optimization =====")
    print("===== SRAM SA优化 =====")

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

    # Create SA optimizer, pass initial parameters and results
    # 创建SA优化器，传递初始参数和结果
    optimizer = SAOptimizer(
        parameter_space, 
        initial_result=initial_result,
        initial_params=initial_params
    )

    # Run SA optimization
    # 运行SA优化
    best_result = optimizer.run_optimization(max_iter=400, T_max=100, T_min=1e-7)

    # Print final results
    # 打印最终结果
    optimizer.print_final_results()


if __name__ == "__main__":
    # Set random seed for reproducibility
    # 设置随机种子以确保可重现性
    SEED = 1
    seed_set(seed=SEED)
    main()
    