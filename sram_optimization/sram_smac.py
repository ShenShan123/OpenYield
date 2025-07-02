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
        
        # Initialize SMAC state
        # 初始化SMAC状态
        self.evaluation_count = 0

    def _create_config_space(self):
        """
        Create configuration space for SMAC
        为SMAC创建配置空间
        """
        cs = ConfigurationSpace()
        
        # Add hyperparameters for each dimension
        # 为每个维度添加超参数
        for i in range(self.dim):
            hp = CSH.UniformFloatHyperparameter(f"x_{i:03d}", 0.0, 1.0, default_value=0.5)
            cs.add_hyperparameter(hp)
        
        return cs

    def smac_objective(self, config: Configuration, seed: int = 0) -> float:
        """
        Objective function for SMAC
        SMAC的目标函数
        """
        # Convert configuration to parameter vector
        # 将配置转换为参数向量
        x = torch.tensor([config[f"x_{i:03d}"] for i in range(self.dim)], dtype=torch.float32)
        
        # Convert parameters
        # 转换参数
        params = self.parameter_space.convert_params(x)
        
        print(f"\n===== SMAC评估 {self.evaluation_count + 1} =====")
        print("当前参数:")
        self.parameter_space.print_params(params)
        
        start_time = time.time()
        objectives, constraints, result, success = evaluate_sram(params)
        end_time = time.time()
        print(f"评估用时: {end_time - start_time:.2f} 秒")
        
        if success and result:
            merit = result['merit']
            
            # Record results
            # 记录结果
            self.observe(x.numpy(), objectives, constraints, result, success, self.evaluation_count, "SMAC")
            
            print(f"Merit: {merit:.6e}")
            print(f"SNM: Hold={result['hold_snm']:.6f}, Read={result['read_snm']:.6f}, Write={result['write_snm']:.6f}")
            print(f"Delay: Read={result['read_delay']*1e12:.2f}ps, Write={result['write_delay']*1e12:.2f}ps")
            print(f"Power: Read={result['read_power']:.2e}W, Write={result['write_power']:.2e}W")
            print(f"Area: {result['area']*1e12:.2f} µm²")
            
            if merit > self.best_merit:
                print(f"发现新的最优解: {merit:.6e}")
            
            # SMAC minimizes, so return negative Merit
            # SMAC是最小化，所以返回负Merit
            self.evaluation_count += 1
            return -merit
        else:
            print("仿真失败！")
            self.evaluation_count += 1
            return 1e6  # Large penalty for failed simulations

    def run_optimization(self, max_iter=400):
        """
        Run SMAC optimization
        运行SMAC优化
        """
        print(f"===== Starting SMAC optimization, maximum simulations: {max_iter} =====")
        print(f"===== 开始SMAC优化，最大仿真次数: {max_iter} =====")

        # Create SMAC scenario
        # 创建SMAC场景
        scenario = Scenario(
            configspace=self.cs,
            deterministic=True,
            n_trials=max_iter,
            seed=1,
        )

        # Initialize SMAC
        # 初始化SMAC
        smac = HyperparameterOptimizationFacade(
            scenario,
            self.smac_objective,
            dask_client=None,  # No parallel execution
        )

        # Run optimization
        # 运行优化
        print("开始SMAC优化循环...")
        incumbent = smac.optimize()

        print(f"\nSMAC优化完成，共进行{self.evaluation_count}次评估")
        print(f"最终配置: {incumbent}")

        # Return best results
        # 返回最佳结果
        if self.best_merit != float('-inf'):
            return {
                'params': self.best_params,
                'merit': self.best_merit,
                'result': self.best_result,
                'iteration': self.evaluation_count
            }
        else:
            return {
                'params': None,
                'merit': None,
                'result': None,
                'iteration': -1
            }


def main(config_path="config_sram.yaml"):
    """
    Main function to run SMAC optimization
    运行SMAC优化的主函数
    """
    print("===== SRAM optimization using SMAC =====")
    print("===== 使用SMAC的SRAM优化 =====")

    # Create directories
    # 创建目录
    create_directories()

    # Create parameter space - use modified parameter space class with config support
    # 创建参数空间 - 使用支持配置的修改参数空间类
    parameter_space = ModifiedSRAMParameterSpace(config_path)

    # Define initial parameters
    # 定义初始参数
    initial_params = get_default_initial_params()
    
    print("Running initial point simulation to get baseline performance...")
    print("运行初始点仿真以获得基准性能...")
    
    # Run initial evaluation
    # 运行初始评估
    initial_result, initial_params = run_initial_evaluation(initial_params)
    
    if initial_result:
        print(f"Initial point simulation successful!")
        print(f"初始点仿真成功！")
        print(f"SNM: Hold={initial_result['hold_snm']['snm']:.4f}, Read={initial_result['read_snm']['snm']:.4f}, Write={initial_result['write_snm']['snm']:.4f}")
        print(f"Delay: Read={initial_result['read']['delay']*1e12:.2f}ps, Write={initial_result['write']['delay']*1e12:.2f}ps")
        print(f"Power: Read={initial_result['read']['power']:.2e}W, Write={initial_result['write']['power']:.2e}W")

    # Create SMAC optimizer
    # 创建SMAC优化器
    optimizer = SMACOptimizer(
        parameter_space,
        initial_result=initial_result,
        initial_params=initial_params
    )

    # Run SMAC optimization
    # 运行SMAC优化
    best_result = optimizer.run_optimization(max_iter=400)

    # Output best results
    # 输出最佳结果
    print("\n===== SMAC Optimization Best Results =====")
    print("\n===== SMAC优化最佳结果 =====")
    if best_result['params'] is not None:
        print(f"Best Merit: {best_result['merit']:.6e}")
        print(f"Evaluation count: {best_result['iteration']}")
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

    return best_result


if __name__ == "__main__":
    # Set random seed for reproducibility
    # 设置随机种子以确保可重现性
    SEED = 1
    seed_set(seed=SEED)
    main()
