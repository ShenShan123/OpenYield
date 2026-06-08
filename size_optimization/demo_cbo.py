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
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ConstrainedExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
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
    CompositeSRAMParameterSpace,
    OptimizationLogger, save_pareto_front, save_best_result, plot_merit_history,
    plot_pareto_frontier, update_pareto_front, save_optimization_history,
    get_default_normalized_vector, estimate_scaled_total_area,
    get_composite_initial_params,
)

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
        # 使用复合参数空间（bitcell + 外围电路）
        try:
            self._modified_space = CompositeSRAMParameterSpace(config_path)
            self.bounds = self._modified_space.bounds
        except Exception:
            # 如果配置加载失败，使用原始默认值
            print("Warning: Config loading failed, using default parameters")
            self._modified_space = None
            lower_bounds = [0, 0, 0, 0, 0, 0]
            upper_bounds = [1, 1, 1, 1, 1, 1]
            self.bounds = torch.tensor([lower_bounds, upper_bounds], dtype=torch.float)

    def convert_params(self, x):
        """
        Convert normalized parameters [0,1] to actual parameter values
        将归一化参数[0,1]转换为实际参数值
        """
        if self._modified_space is not None:
            return self._modified_space.convert_params(x)

        # Fallback: should not reach here in normal operation
        raise RuntimeError("CompositeSRAMParameterSpace not initialized")

    def print_params(self, params):
        """
        Print parameter information
        打印参数信息
        """
        if self._modified_space is not None:
            return self._modified_space.print_params(params)

        print("Parameters / 参数:")
        for key, value in params.items():
            print(f"  {key}: {value}")


# Constrained optimizer
# 约束优化器
class ConstrainedBayesianOptimizer:
    def __init__(self, parameter_space, num_objectives=4, num_constraints=2, initial_result=None, initial_params=None):
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
            initial_max_delay = max(
                initial_result['read']['delay'],
                initial_result['write']['delay']
            )
            
            # Calculate area using initial parameters
            # 使用初始参数计算面积
            if initial_params:
                initial_area = estimate_scaled_total_area(initial_params)
                self.initial_merit = np.log10(initial_min_snm / (initial_max_power * np.sqrt(initial_area) * initial_max_delay))
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
        # 增强：在成功样本较少时加大准随机探索阈值
        if len(self.train_x) < 30:
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
            # Use multi-objective Bayesian optimization (qEHVI)
            # 使用多目标贝叶斯优化 (qEHVI)
            try:
                # Build independent GP for each objective
                # 为每个目标构建独立GP
                models = []
                for i in range(self.num_objectives):
                    gp_i = SingleTaskGP(self.train_x, self.train_obj[:, i:i+1])
                    models.append(gp_i)
                model = ModelListGP(*models)
                mll = SumMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)

                # Reference point: slightly worse than worst observed per objective
                # 参考点: 每个目标略差于观测到的最差值
                ref_point = self.train_obj.max(dim=0).values * 1.1
                # Ensure ref_point is at least as large as penalty values
                ref_point = torch.max(ref_point, torch.tensor([0.0, 10.0, 5e-6, 5e-8]))

                # Use qEHVI acquisition function
                # 使用 qEHVI 获取函数
                partitioning = NondominatedPartitioning(ref_point=ref_point, Y=self.train_obj)
                acq_func = qExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=ref_point.tolist(),
                    partitioning=partitioning,
                )

                # Optimize acquisition function
                # 优化获取函数
                bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
                candidate, acq_value = optimize_acqf(
                    acq_func,
                    bounds=bounds,
                    q=1,
                    num_restarts=20,
                    raw_samples=1024,
                )

                print(f"BO suggested point: {candidate.tolist()}")
                print(f"Acquisition value (EHVI): {acq_value.item():.6e}")
                print(f"BO建议点: {candidate.tolist()}")
                print(f"获取函数值 (EHVI): {acq_value.item():.6e}")

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
def random_search(parameter_space, eval_fn=evaluate_sram, num_iterations=20):
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
        if i == 0:
            default_x = get_default_normalized_vector(parameter_space)
            if default_x:
                for j in range(min(len(default_x), optimizer.dim)):
                    x[j] = default_x[j]

        # Map parameters to actual values
        # 将参数映射到实际值
        params = parameter_space.convert_params(x)
        parameter_space.print_params(params)

        # Evaluate parameter performance
        # 评估参数性能
        start_time = time.time()
        objectives, constraints, result, success = eval_fn(params)
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        print(f"评估用时: {end_time - start_time:.2f} 秒")

        if success:
            print("Simulation successful! Results:")
            print("仿真成功！结果:")
            print(f"Min SNM = {-objectives[0]:.6f}")
            print(f"Max power = {objectives[1]:.6e}")
            print(f"Area = {objectives[2]*1e12:.2f} µm²")
            print(f"Delay = {objectives[3]:.6e} s")
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
def bayesian_optimization(optimizer, parameter_space, eval_fn=evaluate_sram, num_iterations=380, iteration_offset=20):
    """
    Bayesian optimization phase
    贝叶斯优化阶段
    """
    print(f"===== Starting {num_iterations} rounds of Bayesian optimization (BO) =====")
    print(f"===== 开始 {num_iterations} 轮贝叶斯优化 (BO) =====")

    # Track repeated suggestions
    # 跟踪重复建议
    recent_suggestions = []
    # 增强：加长重复建议历史与降低重复容忍度
    max_history = 20
    repeat_count = 0
    max_repeat_allowed = 2

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
            # 增强：更严格的重复检测阈值
            if torch.norm(next_x - prev_x) < 0.03:  # If Euclidean distance is very small
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
        objectives, constraints, result, success = eval_fn(params)
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        print(f"评估用时: {end_time - start_time:.2f} 秒")

        if success:
            print("Simulation successful! Results:")
            print("仿真成功！结果:")
            print(f"Min SNM = {-objectives[0]:.6f}")
            print(f"Max power = {objectives[1]:.6e}")
            print(f"Area = {objectives[2]*1e12:.2f} µm²")
            print(f"Delay = {objectives[3]:.6e} s")
            print(f"Constraints: {['satisfied' if c <= 0 else 'violated' for c in constraints]}")
            print(f"约束: {['满足' if c <= 0 else '违反' for c in constraints]}")
            if result:
                print(f"Merit = {result['merit']:.6e}")
        else:
            print("Simulation failed, penalty values assigned")
            print("仿真失败，分配惩罚值")

        # Observe results (index after RS iterations end)
        # 观察结果（RS迭代结束后的索引）
        iteration = i + iteration_offset  # RS阶段后的索引偏移
        optimizer.observe(next_x, objectives, constraints, result, success, iteration, "BO")

    print("\n===== Bayesian optimization completed =====")
    print("\n===== 贝叶斯优化完成 =====")
    return optimizer


# Main function
# 主函数
def main(config_path="config_sram.yaml", problem=None, max_iter=400):  # 统一接口：接受 problem 与 max_iter
    """
    Main function to run CBO optimization
    运行CBO优化的主函数
    """
    print("===== SRAM optimization using CBO =====")
    print("===== 使用CBO的SRAM优化 =====")

    # Create directories
    # 创建目录
    create_directories()

    # Resolve parameter space and evaluation backend
    # 解析参数空间与评估后端
    if problem is not None:
        try:
            parameter_space, eval_fn, _ = problem
        except Exception:
            parameter_space = SRAMParameterSpace(config_path)
            eval_fn = evaluate_sram
    else:
        parameter_space = SRAMParameterSpace(config_path)
        eval_fn = evaluate_sram

    # Define initial parameters (bitcell + peripheral circuits from YAML)
    # 定义初始参数（从YAML获取bitcell + 外围电路参数）
    initial_params = get_composite_initial_params()
    
    print("Running initial point simulation to get baseline performance...")
    print("运行初始点仿真以获得基准性能...")
    
    # Evaluate initial parameters using selected backend
    # 使用选定评估后端评估初始参数
    objectives, constraints, initial_result, success = eval_fn(initial_params)
    
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

    # Split iterations: small RS + remaining BO
    # 划分迭代：少量随机搜索 + 其余贝优化
    # 增强：增加随机搜索占比并放宽上限
    rs_iters = max(10, min(50, int(max_iter * 0.10)))
    bo_iters = max(1, int(max_iter) - rs_iters)

    # Run random search phase
    # 运行随机搜索阶段
    optimizer = random_search(parameter_space, eval_fn=eval_fn, num_iterations=rs_iters)

    # Run Bayesian optimization phase
    # 运行贝叶斯优化阶段
    optimizer = bayesian_optimization(optimizer, parameter_space, eval_fn=eval_fn, num_iterations=bo_iters, iteration_offset=rs_iters)

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
        ret = {
            'params': optimizer.best_params,
            'merit': optimizer.best_merit,
            'result': optimizer.best_result,
            'iteration': rs_iters + bo_iters
        }
        # 统一返回字段：增加 'fom' 与其他算法一致
        ret['fom'] = ret.get('merit')
        return ret
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
