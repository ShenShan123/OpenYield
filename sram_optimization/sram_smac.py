"""
SRAM Circuit Optimization using SMAC Algorithm
使用SMAC算法的SRAM电路优化

This file implements a SMAC (Sequential Model-based Algorithm Configuration) optimization algorithm 
for SRAM parameter optimization.
该文件实现了用于SRAM参数优化的SMAC（基于序列模型的算法配置）优化算法。
"""

import os
import sys
import time
import numpy as np
import torch
import random
import warnings
import csv
from pathlib import Path
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue, TrialInfo
from ConfigSpace import Configuration, ConfigurationSpace
import ConfigSpace.hyperparameters as CSH

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


class SMACOptimizer(BaseOptimizer):
    """
    SMAC optimizer class
    SMAC优化器类
    """
    
    def __init__(self, parameter_space, num_objectives=3, num_constraints=2, 
                 initial_result=None, initial_params=None):
        """
        Initialize SMAC optimizer
        初始化SMAC优化器
        """
        super().__init__(parameter_space, "SMAC", num_objectives, num_constraints, 
                         initial_result, initial_params)

        # Initialize configuration space
        # 初始化配置空间
        self.cs = self._create_config_space()

        # Create SMAC output directory
        # 创建SMAC输出目录
        Path("sim/opt/smac_output").mkdir(exist_ok=True, parents=True)

    def _create_config_space(self):
        """
        Create SMAC configuration space
        创建SMAC配置空间
        """
        cs = ConfigurationSpace()

        # Add hyperparameters for each dimension
        # 为每个维度添加超参数
        for i in range(self.dim):
            param_name = f'x_{i:03d}'
            param = CSH.UniformFloatHyperparameter(param_name, 0.0, 1.0)
            cs.add_hyperparameter(param)

        return cs


def create_objective_function(parameter_space):
    """
    Create SMAC optimization objective function
    创建SMAC优化目标函数
    """
    def objective_function(config, seed=0):
        """
        SMAC optimization objective function
        SMAC优化目标函数
        """
        # Extract parameters from config
        # 从配置中提取参数
        x = torch.zeros(parameter_space.bounds.shape[1])
        for i in range(len(x)):
            param_name = f'x_{i:03d}'
            x[i] = float(config[param_name])

        # Convert to actual parameters
        # 转换为实际参数
        params = parameter_space.convert_params(x)

        # Evaluate parameter performance
        # 评估参数性能
        objectives, constraints, result, success = evaluate_sram(params)

        # Return main objective (negative min SNM, for minimization)
        # 返回主要目标（负的最小SNM，用于最小化）
        return float(objectives[0])

    return objective_function


# Random search phase
# 随机搜索阶段
def random_search_smac(parameter_space, num_iterations=5, optimizer=None):
    """
    Random search phase for SMAC
    SMAC的随机搜索阶段
    """
    print(f"===== Starting {num_iterations} rounds of random search (RS) =====")
    print(f"===== 开始 {num_iterations} 轮随机搜索 (RS) =====")

    # Create optimizer for recording data, if none passed create new one
    # 创建用于记录数据的优化器，如果没有传递则创建新的
    if optimizer is None:
        optimizer = SMACOptimizer(parameter_space)

    for i in range(num_iterations):
        print(f"\n===== RS iteration {i + 1}/{num_iterations} =====")
        print(f"\n===== RS迭代 {i + 1}/{num_iterations} =====")

        # Generate random parameters
        # 生成随机参数
        x = parameter_space.random_params()

        # Convert to actual parameters
        # 转换为实际参数
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


# SMAC optimization phase
# SMAC优化阶段
def smac_optimization(optimizer, parameter_space, num_iterations=395):
    """
    SMAC optimization phase
    SMAC优化阶段
    """
    print(f"===== Starting {num_iterations} rounds of SMAC optimization =====")
    print(f"===== 开始 {num_iterations} 轮SMAC优化 =====")

    # Create objective function
    # 创建目标函数
    objective_function = create_objective_function(parameter_space)

    # Create scenario
    # 创建场景
    scenario = Scenario(
        optimizer.cs,
        deterministic=True,
        n_trials=num_iterations,
        output_directory="sim/opt/smac_output"
    )

    # Create SMAC optimizer
    # 创建SMAC优化器
    smac = HyperparameterOptimizationFacade(
        scenario,
        objective_function,  # We will define this function below
        overwrite=True
    )

    # Initialize with points from RS phase
    # 用RS阶段的点初始化
    for i in range(len(optimizer.all_x)):
        config = {}
        for j in range(optimizer.dim):
            param_name = f'x_{j:03d}'
            config[param_name] = float(optimizer.all_x[i][j])

        config_obj = Configuration(optimizer.cs, values=config)
        trial_info = TrialInfo(config=config_obj, instance=None, seed=0)
        trial_value = TrialValue(time=0.5, cost=float(optimizer.all_objectives[i][0]))

        smac.tell(trial_info, trial_value)

    # Track repeated suggestions
    # 跟踪重复建议
    recent_suggestions = []
    max_history = 10
    repeat_count = 0
    max_repeat_allowed = 3

    for i in range(num_iterations):
        print(f"\n===== SMAC iteration {i + 1}/{num_iterations} =====")
        print(f"\n===== SMAC迭代 {i + 1}/{num_iterations} =====")

        # Get next evaluation point
        # 获取下一个评估点
        info = smac.ask()

        # Convert configuration to x
        # 将配置转换为x
        x = torch.zeros(optimizer.dim)
        for j in range(optimizer.dim):
            param_name = f'x_{j:03d}'
            x[j] = float(info.config[param_name])

        # Check if too similar to recent suggestions
        # 检查是否与最近的建议太相似
        is_repeat = False
        for prev_x in recent_suggestions:
            if torch.norm(x - prev_x) < 0.05:  # If Euclidean distance is small
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
            x = torch.rand(optimizer.dim)
            repeat_count = 0

            # Update SMAC's config object
            # 更新SMAC的配置对象
            config = {}
            for j in range(optimizer.dim):
                param_name = f'x_{j:03d}'
                config[param_name] = float(x[j])
            info = TrialInfo(config=Configuration(optimizer.cs, values=config), instance=None, seed=0)
        elif not is_repeat:
            repeat_count = 0

        # Update suggestion history
        # 更新建议历史
        recent_suggestions.append(x)
        if len(recent_suggestions) > max_history:
            recent_suggestions.pop(0)  # Remove oldest record

        # Convert to actual parameters
        # 转换为实际参数
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

        # Tell SMAC the results
        # 告诉SMAC结果
        trial_value = TrialValue(time=end_time - start_time, cost=float(objectives[0]))
        smac.tell(info, trial_value)

        # Observe results (RS iterations + SMAC iterations)
        # 观察结果（RS迭代 + SMAC迭代）
        iteration = i + 5  # Assume 5 RS iterations
        optimizer.observe(x, objectives, constraints, result, success, iteration, "SMAC")

        # Save results every 10 iterations
        # 每10次迭代保存一次结果
        if (i + 1) % 10 == 0:
            optimizer.save_results()

    # Save final results
    # 保存最终结果
    optimizer.save_results()

    print("\n===== SMAC optimization completed =====")
    print("\n===== SMAC优化完成 =====")
    return optimizer


# Timing decorator for performance measurement
# 用于性能测量的计时装饰器
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.total_time += (end_time - start_time)
        wrapper.call_count += 1
        return result

    wrapper.total_time = 0.0
    wrapper.call_count = 0
    return wrapper


# Save timing results
# 保存计时结果
def save_timing_results(seed, timing_results):
    """
    Save timing results to CSV
    将计时结果保存到CSV
    """
    timing_file_path = f'sim/opt/results/sram_smac_timing_seed_{seed}.csv'

    with open(timing_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])  # Write header
        for key, value in timing_results.items():
            writer.writerow([key, value])

    print(f"Timing results saved to {timing_file_path}")
    print(f"计时结果已保存到 {timing_file_path}")


# Run timed experiment
# 运行计时实验
def run_timed_experiment(seed):
    """
    Run timed experiment with SMAC
    使用SMAC运行计时实验
    """
    # Declare global variable at the beginning of function
    # 在函数开始时声明全局变量
    global evaluate_sram
    
    print(f"===== Running SMAC experiment with seed {seed} =====")
    print(f"===== 使用种子 {seed} 运行SMAC实验 =====")

    # Create directories
    # 创建目录
    create_directories()

    # Create parameter space
    # 创建参数空间
    parameter_space = ModifiedSRAMParameterSpace()

    # Create timed evaluation functions
    # 创建计时评估函数
    timed_evaluate_rs = timing_decorator(evaluate_sram)
    timed_evaluate_smac = timing_decorator(evaluate_sram)

    # Record start time
    # 记录开始时间
    total_start_time = time.time()

    # Get initial parameters and run evaluation
    # 获取初始参数并运行评估
    initial_params = get_default_initial_params()
    initial_result, initial_params = run_initial_evaluation(parameter_space, initial_params)

    # Create optimizer and pass initial parameters and results
    # 创建优化器并传递初始参数和结果
    optimizer = SMACOptimizer(
        parameter_space, 
        initial_result=initial_result,
        initial_params=initial_params
    )

    # Start random search with timing
    # 开始带计时的随机搜索
    rs_start_time = time.time()

    # Save original evaluate_sram and override with timed version for random search
    # 保存原始evaluate_sram并用计时版本覆盖随机搜索
    original_evaluate_sram = evaluate_sram
    evaluate_sram = timed_evaluate_rs

    # Execute random search
    # 执行随机搜索
    optimizer = random_search_smac(parameter_space, num_iterations=5, optimizer=optimizer)

    # Calculate random search algorithm time (excluding simulation time)
    # 计算随机搜索算法时间（不包括仿真时间）
    rs_algorithm_time = time.time() - rs_start_time - timed_evaluate_rs.total_time

    # Switch to SMAC optimization with timing
    # 切换到带计时的SMAC优化
    smac_start_time = time.time()

    # Override evaluate_sram with timed version for SMAC
    # 用计时版本覆盖SMAC的evaluate_sram
    evaluate_sram = timed_evaluate_smac

    # Execute SMAC optimization
    # 执行SMAC优化
    optimizer = smac_optimization(optimizer, parameter_space, num_iterations=395)

    # Calculate SMAC algorithm time (excluding simulation time)
    # 计算SMAC算法时间（不包括仿真时间）
    smac_algorithm_time = time.time() - smac_start_time - timed_evaluate_smac.total_time

    # Restore original evaluate_sram
    # 恢复原始evaluate_sram
    evaluate_sram = original_evaluate_sram

    # Calculate total time
    # 计算总时间
    total_time = time.time() - total_start_time

    # Calculate total simulation time
    # 计算总仿真时间
    total_simulation_time = timed_evaluate_rs.total_time + timed_evaluate_smac.total_time
    total_simulation_calls = timed_evaluate_rs.call_count + timed_evaluate_smac.call_count

    # Calculate algorithm time (total time minus simulation time)
    # 计算算法时间（总时间减去仿真时间）
    algorithm_time = total_time - total_simulation_time

    # Print timing results
    # 打印计时结果
    print("\n========== Timing Results ==========")
    print("========== 计时结果 ==========")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Total simulation time: {total_simulation_time:.2f} seconds (calls: {total_simulation_calls})")
    print(
        f"  - Random search simulation time: {timed_evaluate_rs.total_time:.2f} seconds (calls: {timed_evaluate_rs.call_count})")
    print(
        f"  - SMAC simulation time: {timed_evaluate_smac.total_time:.2f} seconds (calls: {timed_evaluate_smac.call_count})")
    print(f"Algorithm runtime: {algorithm_time:.2f} seconds")
    print(f"  - Random search algorithm time: {rs_algorithm_time:.2f} seconds")
    print(f"  - SMAC algorithm time: {smac_algorithm_time:.2f} seconds")
    print(f"总运行时间: {total_time:.2f} 秒")
    print(f"总仿真时间: {total_simulation_time:.2f} 秒 (调用次数: {total_simulation_calls})")
    print(f"算法运行时间: {algorithm_time:.2f} 秒")
    print("====================================\n")

    # Timing results dictionary
    # 计时结果字典
    timing_results = {
        'total_time': total_time,
        'simulation_time': total_simulation_time,
        'simulation_calls': total_simulation_calls,
        'algorithm_time': algorithm_time,
        'rs_algorithm_time': rs_algorithm_time,
        'smac_algorithm_time': smac_algorithm_time,
        'rs_simulation_time': timed_evaluate_rs.total_time,
        'rs_simulation_calls': timed_evaluate_rs.call_count,
        'smac_simulation_time': timed_evaluate_smac.total_time,
        'smac_simulation_calls': timed_evaluate_smac.call_count
    }

    # Save timing results
    # 保存计时结果
    save_timing_results(seed, timing_results)

    # Print final results
    # 打印最终结果
    optimizer.print_final_results()

    print("\nExperiment completed!")
    print("\n实验完成！")

    return optimizer, timing_results


# Main function
# 主函数
def main():
    """
    Main function to run SMAC optimization
    运行SMAC优化的主函数
    """
    print("===== SRAM optimization using SMAC =====")
    print("===== 使用SMAC的SRAM优化 =====")

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

    # Create optimizer and pass initial parameters and results
    # 创建优化器并传递初始参数和结果
    optimizer = SMACOptimizer(
        parameter_space, 
        initial_result=initial_result,
        initial_params=initial_params
    )

    # First perform random search
    # 首先执行随机搜索
    optimizer = random_search_smac(parameter_space, num_iterations=5, optimizer=optimizer)

    # Then perform SMAC optimization
    # 然后执行SMAC优化
    optimizer = smac_optimization(optimizer, parameter_space, num_iterations=395)

    # Output Pareto front
    # 输出帕累托前沿
    best_points = optimizer.get_best_points()
    print(f"\n===== Pareto front (found {len(best_points)} non-dominated solutions) =====")
    print(f"\n===== 帕累托前沿 (发现 {len(best_points)} 个非支配解) =====")

    # Print final results
    # 打印最终结果
    optimizer.print_final_results()

    print("\nOptimization completed!")
    print("\n优化完成！")


if __name__ == "__main__":
    # Setup paths and create directories
    # 设置路径并创建目录
    create_directories()

    # Set random seed for reproducibility
    # 设置随机种子以确保可重现性
    SEED = 1
    seed_set(seed=SEED)

    # Run main experiment
    # 运行主实验
    main()

    # For timed experiments, uncomment below
    # 对于计时实验，取消下面的注释
    # optimizer, timing_results = run_timed_experiment(SEED)

    # Run multiple seeds (uncomment below)
    # 运行多个种子（取消下面的注释）
    # for seed in range(1, 6):
    #     print(f"\n\n===== Running seed {seed} =====\n")
    #     seed_set(seed)
    #     optimizer, timing_results = run_timed_experiment(seed)
