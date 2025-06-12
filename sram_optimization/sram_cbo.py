"""
SRAM Circuit Optimization using Constrained Bayesian Optimization
使用约束贝叶斯优化的SRAM电路优化

This file implements a three-objective constrained Bayesian optimization algorithm 
for SRAM parameter optimization.
该文件实现了用于SRAM参数优化的三目标约束贝叶斯优化算法。
"""

import os
import sys
import time
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ConstrainedExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
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

# Set random seeds
# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Set environment variables
# 设置环境变量
os.environ['PATH'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Library', 'bin') + os.pathsep + os.environ['PATH']


# Define parameter space
# 定义参数空间
class SRAMParameterSpace:
    def __init__(self):
        """
        Initialize SRAM parameter space
        初始化SRAM参数空间
        """
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

        # Length ranges - range coefficients can be modified here
        # 长度范围 - 范围系数可以在这里修改
        self.length_coef_min = 0.5
        self.length_coef_max = 1.5
        self.length_min_nm = max(int(self.base_length_nm * self.length_coef_min), 5)  # Minimum 5 nanometers
        self.length_max_nm = int(self.base_length_nm * self.length_coef_max)  # Maximum 100 nanometers

        # Discrete model numbers (for parameter conversion)
        # 离散模型编号（用于参数转换）
        self.nmos_model_map = {0: 'NMOS_VTL', 1: 'NMOS_VTG', 2: 'NMOS_VTH'}
        self.pmos_model_map = {0: 'PMOS_VTL', 1: 'PMOS_VTG', 2: 'PMOS_VTH'}

        # Continuous parameter range [0, 1]
        # 连续参数范围[0, 1]
        self.bounds = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ])

        print("Parameter ranges:")
        print("参数范围:")
        print(f"NMOS models: {self.nmos_models}")
        print(f"PMOS models: {self.pmos_models}")
        print(
            f"pd_width: {self.pd_width_min:.2e} - {self.pd_width_max:.2e} (original ratio: {self.pd_width_coef_min} - {self.pd_width_coef_max})")
        print(
            f"pu_width: {self.pu_width_min:.2e} - {self.pu_width_max:.2e} (original ratio: {self.pu_width_coef_min} - {self.pu_width_coef_max})")
        print(
            f"pg_width: {self.pg_width_min:.2e} - {self.pg_width_max:.2e} (original ratio: {self.pg_width_coef_min} - {self.pg_width_coef_max})")
        print(
            f"length: {self.length_min_nm}nm - {self.length_max_nm}nm (original ratio: {self.length_coef_min} - {self.length_coef_max})")

    def convert_params(self, x):
        """
        Convert normalized parameters to actual SRAM parameters
        将归一化参数转换为实际SRAM参数
        """
        # Ensure x is tensor type
        # 确保x是张量类型
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Print original normalized parameter values
        # 打印原始归一化参数值
        print(f"Normalized parameter values: {x.tolist()}")
        print(f"归一化参数值: {x.tolist()}")

        # Discrete parameters (model selection) - properly handle discrete variables
        # 离散参数（模型选择） - 正确处理离散变量
        # Divide [0,1] interval into 3 parts, avoid boundary issues
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

    def print_params(self, params):
        """
        Print SRAM parameters
        打印SRAM参数
        """
        print(f"Parameters: NMOS={params['nmos_model_name']}, PMOS={params['pmos_model_name']}")
        print(f"参数: NMOS={params['nmos_model_name']}, PMOS={params['pmos_model_name']}")
        print(f"pd_width={params['pd_width']:.2e} (original ratio: {params['pd_width'] / self.base_pd_width:.2f})")
        print(f"pu_width={params['pu_width']:.2e} (original ratio: {params['pu_width'] / self.base_pu_width:.2f})")
        print(f"pg_width={params['pg_width']:.2e} (original ratio: {params['pg_width'] / self.base_pg_width:.2f})")
        print(f"length={params['length_nm']}nm (original ratio: {params['length_nm'] / self.base_length_nm:.2f})")

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


# Constrained optimizer
# 约束优化器
class ConstrainedBayesianOptimizer(BaseOptimizer):
    """
    Constrained Bayesian optimizer
    约束贝叶斯优化器
    """
    
    def __init__(self, parameter_space, num_objectives=3, num_constraints=2, 
                 initial_result=None, initial_params=None):
        """
        Initialize constrained Bayesian optimizer
        初始化约束贝叶斯优化器
        """
        super().__init__(parameter_space, "CBO", num_objectives, num_constraints, 
                         initial_result, initial_params)

    def suggest(self, n_suggestions=1):
        """
        Generate next batch of points to evaluate
        生成下一批要评估的点
        """
        if len(self.train_x) < 10:
            print(f"Training set size: {len(self.train_x)}, using quasi-random sequence")
            print(f"训练集大小: {len(self.train_x)}, 使用准随机序列")
            # Fix Sobol engine
            # 修复Sobol引擎
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

        # Add random exploration probability - early iterations explore more, later iterations tend to exploit
        # 添加随机探索概率 - 早期迭代探索更多，后期迭代倾向于利用
        exploration_prob = max(0.3, 1.0 - len(self.train_x) / 100)
        if random.random() < exploration_prob:
            print("Executing random exploration...")
            print("执行随机探索...")
            # Use random perturbation of known training data
            # 使用已知训练数据的随机扰动
            if len(self.train_x) > 0:
                # Randomly select one from existing points as base
                # 从现有点中随机选择一个作为基础
                base_idx = random.randint(0, len(self.train_x) - 1)
                base_x = self.train_x[base_idx].clone()

                # Add random perturbation to each dimension
                # 向每个维度添加随机扰动
                for d in range(self.dim):
                    # Special handling for discrete variables
                    # 对离散变量的特殊处理
                    if d < 2:  # NMOS and PMOS models are discrete
                        # Completely randomly choose a new value
                        # 完全随机选择新值
                        base_x[d] = random.randint(0, 2) / 2.0
                    else:  # Continuous variables
                        # Add Gaussian noise and ensure within range
                        # 添加高斯噪声并确保在范围内
                        delta = torch.randn(1).item() * 0.2  # Standard deviation 0.2
                        base_x[d] = torch.clamp(base_x[d] + delta, 0.0, 1.0)

                return base_x.unsqueeze(0)
            else:
                # Completely random
                # 完全随机
                return torch.rand(1, self.dim)

        # Try using constrained multi-objective optimization
        # 尝试使用约束多目标优化
        try:
            print("Building multi-objective surrogate model...")
            print("构建多目标代理模型...")

            # Check if there are points that satisfy constraints
            # 检查是否有满足约束的点
            feasible_idx = torch.ones(len(self.train_x), dtype=torch.bool)
            for i in range(self.num_constraints):
                feasible_idx = feasible_idx & (self.train_con[:, i] <= 0)

            if not feasible_idx.any():
                print("No constraint-satisfying points found, using single-objective constrained optimization...")
                print("未找到满足约束的点，使用单目标约束优化...")
                return self._suggest_constrained_single_objective(n_suggestions)
            
            # Use multi-objective optimization
            # 使用多目标优化
            return self._suggest_multi_objective(n_suggestions, feasible_idx)

        except Exception as e:
            print(f"Error building surrogate model or optimizing acquisition function: {e}")
            print(f"构建代理模型或优化采集函数时出错: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to random sampling...")
            print("回退到随机采样...")
            return torch.rand(n_suggestions, self.dim)

    def _suggest_constrained_single_objective(self, n_suggestions=1):
        """
        Use constrained single-objective optimization, first focus on finding constraint-satisfying points
        使用约束单目标优化，首先专注于寻找满足约束的点
        """
        print("Using constrained single-objective optimization...")
        print("使用约束单目标优化...")
        
        # Build single-objective SNM model
        # 构建单目标SNM模型
        train_y_obj = self.train_obj[:, 0:1]  # First objective: min SNM
        # Add small amount of noise to improve numerical stability
        # 添加少量噪声以提高数值稳定性
        noise = 1e-6 * torch.randn_like(train_y_obj)
        train_y_obj = train_y_obj + noise

        obj_model = SingleTaskGP(self.train_x, train_y_obj)
        mll = ExactMarginalLogLikelihood(obj_model.likelihood, obj_model)
        fit_gpytorch_model(mll)

        # Build constraint models
        # 构建约束模型
        con_models = []
        for i in range(self.num_constraints):
            train_y = self.train_con[:, i:i+1]
            # Add small amount of noise to improve numerical stability
            # 添加少量噪声以提高数值稳定性
            noise = 1e-6 * torch.randn_like(train_y)
            train_y = train_y + noise

            model = SingleTaskGP(self.train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            con_models.append(model)

        # Create complete joint model
        # 创建完整的联合模型
        from botorch.models.model_list_gp_regression import ModelListGP
        models = [obj_model] + con_models
        model = ModelListGP(*models)

        # Get current best value
        # 获取当前最佳值
        best_f = self.train_obj[:, 0].min().item()
        print(f"Current best SNM value: {best_f}")
        print(f"当前最佳SNM值: {best_f}")

        # Create constraints dictionary
        # 创建约束字典
        constraints_dict = {}
        for i in range(self.num_constraints):
            # Constraint indices start from 1, because 0 is the objective function
            # 约束索引从1开始，因为0是目标函数
            constraints_dict[i + 1] = (None, 0.0)  # (lower_bound, upper_bound)

        print(f"Constraints dictionary: {constraints_dict}")
        print(f"约束字典: {constraints_dict}")

        # Use Constrained Expected Improvement (CEI) acquisition function
        # 使用约束期望改进（CEI）采集函数
        cei = ConstrainedExpectedImprovement(
            model=model,
            best_f=best_f,
            objective_index=0,  # Objective function is always the first output
            constraints=constraints_dict,
            maximize=False  # We are minimizing the objective function (SNM is negative)
        )

        print("Optimizing acquisition function...")
        print("优化采集函数...")
        # Use botorch to optimize acquisition function
        # 使用botorch优化采集函数
        candidates, _ = optimize_acqf(
            acq_function=cei,
            bounds=self.bounds,
            q=n_suggestions,
            num_restarts=40,  # Increase restart count
            raw_samples=1000,  # Increase sample count
        )

        print(f"BO candidate points: {candidates.tolist()}")
        print(f"BO候选点: {candidates.tolist()}")
        return candidates

    def _suggest_multi_objective(self, n_suggestions=1, feasible_idx=None):
        """
        Use multi-objective optimization to generate suggested points
        使用多目标优化生成建议点
        """
        print("Using multi-objective optimization...")
        print("使用多目标优化...")
        
        # If there are constraint-satisfying points, use only these points to build model
        # 如果有满足约束的点，仅使用这些点构建模型
        if feasible_idx is not None and feasible_idx.any():
            train_x = self.train_x[feasible_idx]
            train_y = self.train_obj[feasible_idx]
            print(f"Using {train_x.shape[0]} constraint-satisfying points to build model")
            print(f"使用 {train_x.shape[0]} 个满足约束的点构建模型")
        else:
            train_x = self.train_x
            train_y = self.train_obj
            print(f"Using all {train_x.shape[0]} points to build model")
            print(f"使用所有 {train_x.shape[0]} 个点构建模型")
            
        # Add small amount of noise to improve numerical stability
        # 添加少量噪声以提高数值稳定性
        noise = 1e-6 * torch.randn_like(train_y)
        train_y = train_y + noise
            
        # Build multi-objective model
        # 构建多目标模型
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        
        # Create independent GP models for each objective
        # 为每个目标创建独立的GP模型
        models = []
        for i in range(self.num_objectives):
            model = SingleTaskGP(train_x, train_y[:, i:i+1])
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            models.append(model)
            
        # Create multi-objective model
        # 创建多目标模型
        model = ModelListGP(*models)
        
        # Calculate current hypervolume
        # 计算当前超体积
        from botorch.utils.multi_objective.pareto import is_non_dominated
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        
        # Find Pareto front in current training data
        # 在当前训练数据中找到帕累托前沿
        pareto_mask = is_non_dominated(train_y)
        pareto_y = train_y[pareto_mask]
        
        # Reference point (for hypervolume calculation)
        # 参考点（用于超体积计算）
        ref_point = self.ref_point
        
        # Calculate current hypervolume
        # 计算当前超体积
        try:
            # Only calculate hypervolume when constraint-satisfying Pareto points exist
            # 仅在存在满足约束的帕累托点时计算超体积
            if pareto_y.shape[0] > 0:
                hv = Hypervolume(ref_point=ref_point)
                volume = hv.compute(pareto_y)
                print(f"Current Pareto front hypervolume: {volume:.6f}")
                print(f"当前帕累托前沿超体积: {volume:.6f}")
            else:
                print("No constraint-satisfying Pareto points found")
                print("未找到满足约束的帕累托点")
        except Exception as e:
            print(f"Error calculating hypervolume: {e}")
            print(f"计算超体积时出错: {e}")
            # If hypervolume calculation fails, fall back to constrained single-objective optimization
            # 如果超体积计算失败，回退到约束单目标优化
            return self._suggest_constrained_single_objective(n_suggestions)
        
        # Use Expected Hypervolume Improvement (qEHVI) acquisition function
        # 使用期望超体积改进（qEHVI）采集函数
        try:
            partitioning = DominatedPartitioning(ref_point=ref_point, Y=pareto_y)
            qehvi = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                partitioning=partitioning
            )
            
            print("Optimizing multi-objective acquisition function...")
            print("优化多目标采集函数...")
            # Use botorch to optimize acquisition function
            # 使用botorch优化采集函数
            candidates, _ = optimize_acqf(
                acq_function=qehvi,
                bounds=self.bounds,
                q=n_suggestions,
                num_restarts=40,  # Increase restart count
                raw_samples=1000,  # Increase sample count
            )
            
            print(f"Multi-objective BO candidate points: {candidates.tolist()}")
            print(f"多目标BO候选点: {candidates.tolist()}")
            return candidates
        except Exception as e:
            print(f"Error optimizing multi-objective acquisition function: {e}")
            print(f"优化多目标采集函数时出错: {e}")
            # If multi-objective acquisition function optimization fails, fall back to constrained single-objective optimization
            # 如果多目标采集函数优化失败，回退到约束单目标优化
            return self._suggest_constrained_single_objective(n_suggestions)


# Random search phase
# 随机搜索阶段
def random_search(parameter_space, num_iterations=20, optimizer=None):
    """
    Random search phase
    随机搜索阶段
    """
    print(f"===== Starting {num_iterations} rounds of random search (RS) =====")
    print(f"===== 开始 {num_iterations} 轮随机搜索 (RS) =====")

    # Create optimizer for recording data, if none passed create new one
    # 创建用于记录数据的优化器，如果没有传递则创建新的
    if optimizer is None:
        optimizer = ConstrainedBayesianOptimizer(parameter_space)

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

        # Save results every 10 iterations
        # 每10次迭代保存一次结果
        if (i + 1) % 10 == 0:
            optimizer.save_results()

    # Save final results
    # 保存最终结果
    optimizer.save_results()

    print("\n===== Bayesian optimization completed =====")
    print("\n===== 贝叶斯优化完成 =====")
    return optimizer


# Main function
# 主函数
def main():
    """
    Main function to run CBO optimization
    运行CBO优化的主函数
    """
    print("===== SRAM Three-objective Constrained Optimization =====")
    print("===== SRAM三目标约束优化 =====")

    # Create directories
    # 创建目录
    create_directories()

    # Create parameter space
    # 创建参数空间
    parameter_space = SRAMParameterSpace()

    # Get initial parameters and run evaluation
    # 获取初始参数并运行评估
    initial_params = get_default_initial_params()
    initial_result, initial_params = run_initial_evaluation(parameter_space, initial_params)

    # Create optimizer and pass initial parameters and results
    # 创建优化器并传递初始参数和结果
    optimizer = ConstrainedBayesianOptimizer(
        parameter_space, 
        initial_result=initial_result,
        initial_params=initial_params
    )

    # First perform random search
    # 首先执行随机搜索
    optimizer = random_search(parameter_space, num_iterations=20, optimizer=optimizer)

    # Then perform Bayesian optimization
    # 然后执行贝叶斯优化
    optimizer = bayesian_optimization(optimizer, parameter_space, num_iterations=385)

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
    SEED = 1
    seed_set(seed=SEED)
    main()
    