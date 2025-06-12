"""
SRAM Circuit Optimization using RoSE-Opt Algorithm
使用RoSE-Opt算法的SRAM电路优化

This file implements the RoSE-Opt algorithm for SRAM optimization.
该文件实现了用于SRAM优化的RoSE-Opt算法。

It combines Bayesian Optimization and Reinforcement Learning to find optimal SRAM parameters.
它结合贝叶斯优化和强化学习来寻找最优SRAM参数。
"""

import os
import sys
import time
import numpy as np
import torch
import random
import warnings
import gymnasium as gym
from gymnasium import spaces
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from scipy.stats import norm
from tqdm import tqdm

# Set device
# 设置设备
device = torch.device("cpu")

# Suppress warnings
# 抑制警告
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
    BaseOptimizer, get_default_initial_params, run_initial_evaluation, estimate_bitcell_area
)
from sram_optimization.borl_model.ppo import PPO
from sram_optimization.borl_model.buffer import Memory


class Normalizer:
    """
    Parameter normalizer for optimization
    优化的参数归一化器
    """

    def __init__(self, low, high, param_names=None):
        """
        Initialize normalizer
        初始化归一化器

        Args:
            low: Lower bounds for parameters / 参数的下界
            high: Upper bounds for parameters / 参数的上界
            param_names: Names of parameters / 参数名称
        """
        self.low = np.array(low)
        self.high = np.array(high)
        self.param_names = param_names

    def normalize(self, x):
        """
        Normalize parameters to 0-1 range
        将参数归一化到0-1范围
        """
        # Ensure x is within valid range
        # 确保x在有效范围内
        if isinstance(x, np.ndarray):
            x_clipped = np.clip(x, self.low, self.high)
            if not np.array_equal(x, x_clipped):
                x = x_clipped

        return (x - self.low) / (self.high - self.low + 1e-6)

    def denormalize(self, x_norm):
        """
        Convert normalized parameters (0-1) back to original range
        将归一化参数（0-1）转换回原始范围
        """
        # Ensure normalized values are in 0-1 range
        # 确保归一化值在0-1范围内
        if isinstance(x_norm, np.ndarray):
            x_norm_clipped = np.clip(x_norm, 0, 1)
            if not np.array_equal(x_norm, x_norm_clipped):
                x_norm = x_norm_clipped

        # Perform denormalization
        # 执行去归一化
        x_denorm = x_norm * (self.high - self.low) + self.low

        # Ensure all values are strictly within valid range
        # 确保所有值严格在有效范围内
        if isinstance(x_denorm, np.ndarray):
            np.clip(x_denorm, self.low, self.high, out=x_denorm)

        return x_denorm


class SRAMCircuitEnv(gym.Env):
    """
    Environment for SRAM circuit optimization using RL
    使用RL进行SRAM电路优化的环境
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, parameter_space, max_steps=2):
        """
        Initialize the SRAM environment
        初始化SRAM环境

        Args:
            parameter_space (SRAMParameterSpace): Parameter space for SRAM optimization / SRAM优化的参数空间
            max_steps (int): Maximum steps per episode / 每个回合的最大步数
        """
        super().__init__()
        self.parameter_space = parameter_space

        # Get parameter ranges
        # 获取参数范围
        self.low = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64).to(device)
        self.high = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64).to(device)

        # Set step length for each parameter (can be adjusted)
        # 设置每个参数的步长（可调整）
        self.step_length = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float64).to(device)

        # Parameter names
        # 参数名称
        self.param_names = ['nmos_model', 'pmos_model', 'pd_width', 'pu_width', 'pg_width', 'length']

        # Environment state
        # 环境状态
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None
        self.last_action = None
        self.last_simulation_result = None

        # Circuit type (for compatibility)
        # 电路类型（兼容性）
        self.circuit_type = "sram"

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        将环境重置为初始状态
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.state = torch.zeros(len(self.param_names), dtype=torch.float64).to(device)
        self.last_action = None
        self.last_simulation_result = None
        return self.state, {}

    def step(self, action):
        """
        Execute one step in the environment
        在环境中执行一步

        Args:
            action: Tensor representing parameter changes / 表示参数变化的张量

        Returns:
            tuple: (state, reward, done, objectives, truncated, meets_constraints)
        """
        try:
            self.current_step += 1
            self.last_action = action

            # Ensure all tensors are on the same device
            # 确保所有张量在同一设备上
            action = action.to(device)

            # Apply changes to state
            # 应用状态变化
            step_length = self.step_length.to(device)

            # Ensure state is on the same device
            # 确保状态在同一设备上
            if self.state.device != device:
                self.state = self.state.to(device)

            # Apply action
            # 应用动作
            change = step_length * action

            # Check for NaN or infinity in change
            # 检查变化中的NaN或无穷大
            if torch.isnan(change).any() or torch.isinf(change).any():
                change = torch.nan_to_num(change, nan=0.0, posinf=0.0, neginf=0.0)

            # Update state
            # 更新状态
            self.state = self.state + change

            # Clamp values to valid range
            # 将值限制在有效范围内
            self.state = torch.clamp(self.state, self.low, self.high)

            try:
                # Use modified parameter space conversion function
                # 使用修改的参数空间转换函数
                params = self.parameter_space.convert_params(self.state)

                # Evaluate SRAM performance
                # 评估SRAM性能
                objectives, constraints, result, success = evaluate_sram(params)

                # Store simulation result
                # 存储仿真结果
                self.last_simulation_result = (objectives, constraints, result, success)

                # Calculate reward and check constraints
                # 计算奖励并检查约束
                reward, meets_constraints = self._calculate_reward(objectives, constraints, result)

                # Check if done
                # 检查是否完成
                done = False
                truncated = self.current_step >= self.max_steps

                return self.state, reward, done, objectives, truncated, meets_constraints
            except ValueError as e:
                print(f"Error converting parameters: {e}")
                print(f"参数转换错误: {e}")
                print(f"State shape: {self.state.shape}, State values: {self.state}")

                # Return default values to avoid program crash
                # 返回默认值以避免程序崩溃
                return (self.state, -1000.0, True, [10.0, 1e-3, 1e-10], True, False)

        except Exception as e:
            print(f"Error in environment step: {e}")
            print(f"环境步骤中的错误: {e}")
            import traceback
            traceback.print_exc()
            return (self.state, -1000.0, True, [10.0, 1e-3, 1e-10], True, False)

    def step_without_simulation(self, action):
        """
        Same as regular step but uses existing simulation results
        与常规步骤相同，但使用现有仿真结果

        Args:
            action: Tensor representing parameter changes / 表示参数变化的张量

        Returns:
            tuple: (state, reward, done, objectives, truncated, meets_constraints)
        """
        try:
            self.current_step += 1
            self.last_action = action

            # Ensure all tensors are on the same device
            # 确保所有张量在同一设备上
            action = action.to(device)

            # Apply changes to state
            # 应用状态变化
            step_length = self.step_length.to(device)

            # Ensure state is on the same device
            # 确保状态在同一设备上
            if self.state.device != device:
                self.state = self.state.to(device)

            # Apply action
            # 应用动作
            change = step_length * action

            # Check for NaN or infinity in change
            # 检查变化中的NaN或无穷大
            if torch.isnan(change).any() or torch.isinf(change).any():
                change = torch.nan_to_num(change, nan=0.0, posinf=0.0, neginf=0.0)

            # Update state
            # 更新状态
            self.state = self.state + change

            # Clamp values to valid range
            # 将值限制在有效范围内
            self.state = torch.clamp(self.state, self.low, self.high)

            # Use pre-calculated simulation result if available
            # 如果可用，使用预计算的仿真结果
            if hasattr(self, 'last_simulation_result') and self.last_simulation_result is not None:
                objectives, constraints, result, success = self.last_simulation_result
                reward, meets_constraints = self._calculate_reward(objectives, constraints, result)
                self.last_simulation_result = None  # Clear for next simulation
            else:
                try:
                    # If no pre-calculated result, perform normal simulation
                    # 如果没有预计算结果，执行正常仿真
                    params = self.parameter_space.convert_params(self.state)
                    objectives, constraints, result, success = evaluate_sram(params)
                    self.last_simulation_result = (objectives, constraints, result, success)
                    reward, meets_constraints = self._calculate_reward(objectives, constraints, result)
                except ValueError as e:
                    print(f"Error converting parameters (step_without_simulation): {e}")
                    print(f"参数转换错误 (step_without_simulation): {e}")
                    print(f"State shape: {self.state.shape}, State values: {self.state}")

                    # Return default values to avoid program crash
                    # 返回默认值以避免程序崩溃
                    return (self.state, -1000.0, True, [10.0, 1e-3, 1e-10], True, False)

            # Check if done
            # 检查是否完成
            done = False
            truncated = self.current_step >= self.max_steps

            return self.state, reward, done, objectives, truncated, meets_constraints

        except Exception as e:
            print(f"Error in step_without_simulation: {e}")
            print(f"step_without_simulation中的错误: {e}")
            import traceback
            traceback.print_exc()
            return (self.state, -1000.0, True, [10.0, 1e-3, 1e-10], True, False)

    def _calculate_reward(self, objectives, constraints, result=None):
        """
        Calculate reward from objectives and constraints
        从目标和约束计算奖励

        Args:
            objectives: List of objective values [neg_min_snm, max_power, area] / 目标值列表
            constraints: List of constraint values / 约束值列表
            result: Complete result dictionary, if available / 完整结果字典（如果可用）

        Returns:
            tuple: (reward, meets_constraints)
        """
        # Check if constraints are satisfied
        # 检查约束是否满足
        meets_constraints = all(c <= 0 for c in constraints)

        # Calculate reward - use new merit value as reward
        # 计算奖励 - 使用新的Merit值作为奖励
        if meets_constraints and result is not None:
            # Use merit value from result
            # 使用结果中的Merit值
            reward = result['merit']
        elif meets_constraints:
            # If no result but constraints satisfied, manually calculate merit
            # 如果没有结果但约束满足，手动计算Merit
            # Get values from objectives
            # 从目标获取值
            neg_min_snm, max_power, area = objectives
            min_snm = -neg_min_snm
            
            # Calculate merit
            # 计算Merit
            reward = np.log10(min_snm / (max_power * (area**0.5)))
        else:
            # Apply penalty for constraint violation
            # 对约束违反应用惩罚
            penalty = sum(max(0, c) for c in constraints)
            reward = -penalty

        return reward, meets_constraints

    def render(self):
        """
        Render the environment
        渲染环境
        """
        print(f"Step {self.current_step}: Action = {self.last_action}, State = {self.state}")

    def close(self):
        """
        Close the environment
        关闭环境
        """
        pass


class SRAM_BORL(BaseOptimizer):
    """
    Bayesian Optimization and Reinforcement Learning for SRAM optimization
    用于SRAM优化的贝叶斯优化和强化学习
    """

    def __init__(self, env, parameter_space, objective_function, save_dir='sim/opt/results',
                 hidden_dim=64, lr=3e-4, betas=(0.9, 0.999), gamma=0.99,
                 K_epochs=10, eps_clip=0.2, initial_result=None, initial_params=None):
        """
        Initialize the SRAM_BORL algorithm
        初始化SRAM_BORL算法
        """
        super().__init__(parameter_space, "RoSE_Opt", 3, 2, initial_result, initial_params)
        
        self.env = env
        self.objective_function = objective_function
        
        # Update save directory
        # 更新保存目录
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Get parameter ranges
        # 获取参数范围
        self.env.reset()
        self.param_lower_bound = np.array(self.env.low.cpu().numpy())
        self.param_upper_bound = np.array(self.env.high.cpu().numpy())
        self.param_ranges = np.vstack((self.param_lower_bound, self.param_upper_bound)).T

        # Initialize normalizer
        # 初始化归一化器
        self.normalizer = Normalizer(self.param_lower_bound, self.param_upper_bound, env.param_names)

        # Initialize PPO agent
        # 初始化PPO代理
        state_dim = len(self.param_ranges)
        action_dim = 3  # Increase, decrease, or maintain each parameter

        self.ppo_agent = PPO(state_spec_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                             lr=lr, betas=betas, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip)

        # Ensure all networks are on CPU
        # 确保所有网络都在CPU上
        self.ppo_agent.policy.to(device)
        self.ppo_agent.policy_old.to(device)

        # Make sure PPO network components are on CPU
        # 确保PPO网络组件在CPU上
        for name, param in self.ppo_agent.policy.named_parameters():
            param.data = param.data.to(device)
        for name, param in self.ppo_agent.policy_old.named_parameters():
            param.data = param.data.to(device)

        # Ensure optimizer is on the same device
        # 确保优化器在同一设备上
        for state in self.ppo_agent.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # Initialize tracking variables
        # 初始化跟踪变量
        self.search_time = 0.0
        self.simulation_time = 0.0

        # Tracking optimization history
        # 跟踪优化历史
        self.last_optimal_y = []

        # Add to optimal history if initial result exists
        # 如果存在初始结果，添加到最优历史
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
            else:
                initial_area = estimate_bitcell_area(
                    w_access=self.parameter_space.base_pg_width,
                    w_pd=self.parameter_space.base_pd_width,
                    w_pu=self.parameter_space.base_pu_width,
                    l_transistor=self.parameter_space.base_length_nm * 1e-9
                )

            self.last_optimal_y.append([initial_min_snm, initial_max_power, initial_area])

    def _init_gp_model(self, n_initial_points=20):
        """
        Initialize Gaussian Process model and training data
        初始化高斯过程模型和训练数据
        """
        start_time = time.time()

        # Generate initial random normalized points
        # 生成初始随机归一化点
        X_train_normalized = np.random.rand(n_initial_points, len(self.param_ranges))

        # Convert to actual parameter space
        # 转换到实际参数空间
        X_train = self.normalizer.denormalize(X_train_normalized)

        # Evaluate initial points
        # 评估初始点
        Y_orig = []
        for i in range(n_initial_points):
            try:
                # Convert numpy array to torch tensor for parameter conversion
                # 将numpy数组转换为torch张量进行参数转换
                x_tensor = torch.tensor(X_train[i], dtype=torch.float64)

                # Convert to actual SRAM parameters
                # 转换为实际SRAM参数
                params = self.parameter_space.convert_params(x_tensor)

                # Evaluate SRAM performance
                # 评估SRAM性能
                start_sim = time.time()
                objectives, constraints, result, success = evaluate_sram(params)
                self.simulation_time += time.time() - start_sim

                # Log result
                # 记录结果
                self.logger.write_csv_result(i, "GP-Init", params, success, objectives, constraints, result)

                # Store objectives
                # 存储目标
                if success and result is not None:
                    Y_orig.append(objectives)

                    # Update best result if constraints are satisfied
                    # 如果约束满足，更新最佳结果
                    if all(c <= 0 for c in constraints):
                        merit = result['merit']
                        min_snm = result['min_snm']
                        max_power = result['max_power']
                        area = result['area']

                        is_best = False
                        if merit > self.best_merit:
                            self.best_merit = merit
                            self.best_merit_params = params
                            self.best_merit_result = result
                            self.best_merit_iteration = i
                            is_best = True
                            print(
                                f"[New best Merit!] Iteration {i}: Merit = {merit:.6e}, "
                                f"Min SNM = {min_snm:.6f}, Max power = {max_power:.6e}, Area = {area*1e12:.2f} µm²"
                            )

                        # Write Merit result
                        # 写入Merit结果
                        self.logger.write_merit_result(i, merit, min_snm, max_power, area, is_best)

                        # Add to Pareto front
                        # 添加到帕累托前沿
                        from sram_optimization.exp_utils import update_pareto_front
                        self.pareto_front = update_pareto_front(
                            self.pareto_front,
                            {
                                'params': params,
                                'objectives': {
                                    'min_snm': -float(objectives[0]),  # Convert back to positive SNM
                                    'max_power': float(objectives[1]),
                                    'area': float(objectives[2])
                                },
                                'raw_result': result
                            }
                        )
                else:
                    Y_orig.append([10.0, 1e-3, 1e-10])  # Penalty values for 3 objectives
            except Exception as e:
                print(f"Error evaluating initial point {i}: {e}")
                print(f"评估初始点 {i} 时出错: {e}")
                Y_orig.append([10.0, 1e-3, 1e-10])  # Penalty values for 3 objectives

        Y_orig = np.array(Y_orig)

        # Standardize output
        # 标准化输出
        Y_mean = Y_orig.mean(axis=0)
        Y_std = Y_orig.std(axis=0)
        Y_std[Y_std < 1e-6] = 1e-6  # Prevent division by zero
        Y_train = (Y_orig - Y_mean) / Y_std

        # Fit multi-output Gaussian Process
        # 拟合多输出高斯过程
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False)
        gp.fit(X_train_normalized, Y_train)

        self.search_time += time.time() - start_time

        return X_train_normalized, Y_train, Y_mean, Y_std, gp

    def train(self, num_iterations=400, debug_interval=20):
        """
        Execute RoSE-Opt training, modify data saving format to match other algorithms
        执行RoSE-Opt训练，修改数据保存格式以匹配其他算法
        """
        print(f"===== Starting SRAM RoSE-Opt algorithm optimization =====")
        print(f"===== 开始SRAM RoSE-Opt算法优化 =====")

        # Initialize GP model
        # 初始化GP模型
        print(f"Initializing GP model with random sampling...")
        print(f"使用随机采样初始化GP模型...")
        X_train, Y_train, Y_mean, Y_std, gp = self._init_gp_model()

        # Total iteration counter (including initialization and all iterations)
        # 总迭代计数器（包括初始化和所有迭代）
        total_iterations = len(X_train)  # Start counting from number of initialization samples

        # Track repeated suggestions
        # 跟踪重复建议
        recent_suggestions = []
        max_history = 10
        repeat_count = 0
        max_repeat_allowed = 3

        # Set task ID (1 indicates power optimization)
        # 设置任务ID（1表示功耗优化）
        task_id = 1
        constraints = {
            "max_delay": 5e-10  # 500ps
        }

        # Main loop
        # 主循环
        for iteration in tqdm(range(num_iterations), desc="RoSE-Opt optimization", unit="iteration"):
            try:
                # Start timing
                # 开始计时
                iter_start_time = time.time()

                # ---------- BO step: Use Bayesian optimization to select parameter points ----------
                search_start_time = time.time()

                # Generate candidate points
                # 生成候选点
                X_candidates = np.random.rand(100, X_train.shape[1])

                # Get predictions from GP model
                # 从GP模型获取预测
                mu, sigma = gp.predict(X_candidates, return_std=True)

                # Find current best performance
                # 找到当前最佳性能
                # For power (task_id=1), we want to minimize
                # 对于功耗（task_id=1），我们要最小化
                f_best = Y_train[:, task_id].min()
                mu_task = mu[:, task_id]

                # Ensure sigma dimensions are correct
                # 确保sigma维度正确
                if isinstance(sigma, np.ndarray) and sigma.ndim > 1:
                    sigma_task = sigma[:, task_id]
                else:
                    sigma_task = sigma

                sigma_task = np.maximum(sigma_task, 1e-6)  # Prevent division by zero

                # Calculate Expected Improvement (EI)
                # 计算期望改进（EI）
                z = (f_best - mu_task) / sigma_task
                ei = (f_best - mu_task) * norm.cdf(z) + sigma_task * norm.pdf(z)

                # Unstandardize predictions
                # 反标准化预测
                mu_unstd = mu * Y_std + Y_mean

                # Calculate constraint satisfaction probability
                # 计算约束满足概率
                p_constraint = np.ones_like(ei)

                # Select best candidate point
                # 选择最佳候选点
                best_idx = np.argmax(ei * p_constraint)
                X_next = X_candidates[best_idx].reshape(1, -1)

                # Check if too similar to recent suggestions
                # 检查是否与最近的建议太相似
                is_repeat = False
                for prev_x in recent_suggestions:
                    if np.linalg.norm(X_next - prev_x) < 0.05:  # If Euclidean distance is small
                        repeat_count += 1
                        is_repeat = True
                        print(f"Warning: Current suggestion too similar to previous suggestions (repeat count: {repeat_count})")
                        print(f"警告：当前建议与之前的建议太相似（重复次数：{repeat_count}）")
                        break

                # If too many repeats, force random exploration
                # 如果重复太多次，强制随机探索
                if is_repeat and repeat_count >= max_repeat_allowed:
                    print(f"Consecutive repeats {repeat_count} times, switching to random exploration...")
                    print(f"连续重复 {repeat_count} 次，切换到随机探索...")
                    X_next = np.random.rand(1, X_train.shape[1])
                    repeat_count = 0
                elif not is_repeat:
                    repeat_count = 0

                # Update suggestion history
                # 更新建议历史
                recent_suggestions.append(X_next)
                if len(recent_suggestions) > max_history:
                    recent_suggestions.pop(0)  # Remove oldest record

                self.search_time += time.time() - search_start_time

                # ---------- Convert to actual parameters ----------
                # Convert to tensor
                # 转换为张量
                x_tensor = torch.tensor(X_next.flatten(), dtype=torch.float64).to(device)

                try:
                    # Convert to SRAM parameters
                    # 转换为SRAM参数
                    params = self.parameter_space.convert_params(x_tensor)

                    # Print debug information
                    # 打印调试信息
                    if iteration % debug_interval == 0:
                        print(f"\n----- Iteration {iteration + 1}/{num_iterations} (Total #{total_iterations + 1}) -----")
                        print(f"\n----- 迭代 {iteration + 1}/{num_iterations} (总计 #{total_iterations + 1}) -----")
                        self.parameter_space.print_params(params)

                    # ---------- Evaluate SRAM performance ----------
                    start_sim = time.time()
                    objectives, constraints, result, success = evaluate_sram(params)
                    self.simulation_time += time.time() - start_sim

                    # Key modification: Record BO results using continuous total iteration count
                    # 关键修改：使用连续总迭代计数记录BO结果
                    current_iteration = total_iterations
                    total_iterations += 1  # Update total iteration count

                    # Record BO results
                    # 记录BO结果
                    self.observe(x_tensor, objectives, constraints, result, success, current_iteration, "BORL")

                    # Process and update results (if successful)
                    # 处理和更新结果（如果成功）
                    if success and result is not None and all(c <= 0 for c in constraints):
                        # Update optimization history
                        # 更新优化历史
                        if self.best_merit_result is not None:
                            self.last_optimal_y.append([
                                self.best_merit_result['min_snm'],
                                self.best_merit_result['max_power'],
                                self.best_merit_result['area']
                            ])

                    # ---------- RL step: Use reinforcement learning to improve points ----------
                    # Prepare RL state
                    # 准备RL状态
                    state_norm = torch.tensor(X_next.flatten(), dtype=torch.float64).to(device)

                    # Set environment state
                    # 设置环境状态
                    self.env.state = state_norm.clone()

                    # Store simulation result to avoid repeated simulation
                    # 存储仿真结果以避免重复仿真
                    self.env.last_simulation_result = (objectives, constraints, result, success)

                    # Get RL action
                    # 获取RL动作
                    try:
                        state_for_rl = state_norm.unsqueeze(0) if state_norm.dim() == 1 else state_norm
                        action = self.ppo_agent.policy_old.act(state_for_rl)
                    except Exception as e:
                        print(f"RL action selection error: {e}")
                        print(f"RL动作选择错误: {e}")
                        continue

                    # Execute action but don't repeat simulation
                    # 执行动作但不重复仿真
                    next_state, reward, done, rl_objectives, truncated, meets_constraints = self.env.step_without_simulation(
                        action)

                    # Update PPO memory
                    # 更新PPO内存
                    try:
                        # Ensure reward and done flag are scalars
                        # 确保奖励和完成标志是标量
                        scalar_reward = float(reward) if isinstance(reward, (torch.Tensor, np.ndarray)) else reward
                        scalar_done = bool(done) if isinstance(done, (torch.Tensor, np.ndarray)) else done

                        self.ppo_agent.policy_old.memory.rewards.append(scalar_reward)
                        self.ppo_agent.policy_old.memory.is_terminals.append(scalar_done)

                        # Update PPO
                        # 更新PPO
                        self.ppo_agent.update(self.ppo_agent.policy_old.memory)
                    except Exception as e:
                        print(f"PPO update error: {e}")
                        print(f"PPO更新错误: {e}")
                        continue

                    # ---------- Update GP model with new data ----------
                    try:
                        # Normalize new state
                        # 归一化新状态
                        new_state_np = next_state.cpu().numpy()
                        new_X = self.normalizer.normalize(new_state_np).reshape(1, -1)

                        # Get converted parameters
                        # 获取转换的参数
                        new_params = self.parameter_space.convert_params(next_state)

                        # Re-evaluate if needed (should already be in last_simulation_result)
                        # 如果需要重新评估（应该已经在last_simulation_result中）
                        if self.env.last_simulation_result is None:
                            start_sim = time.time()
                            rl_objectives, rl_constraints, rl_result, rl_success = evaluate_sram(new_params)
                            self.simulation_time += time.time() - start_sim
                        else:
                            rl_objectives, rl_constraints, rl_result, rl_success = self.env.last_simulation_result

                        # Standardize objectives
                        # 标准化目标
                        new_Y = ((np.array(rl_objectives) - Y_mean) / Y_std).reshape(1, -1)

                        # Ensure dimensions match
                        # 确保维度匹配
                        if new_Y.shape[1] != Y_train.shape[1]:
                            if new_Y.shape[1] < Y_train.shape[1]:
                                # Padding
                                # 填充
                                padding = np.zeros((new_Y.shape[0], Y_train.shape[1] - new_Y.shape[1]))
                                new_Y = np.hstack((new_Y, padding))
                            else:
                                # Truncation
                                # 截断
                                new_Y = new_Y[:, :Y_train.shape[1]]

                        # Update training data
                        # 更新训练数据
                        X_train = np.vstack([X_train, new_X])
                        Y_train = np.vstack([Y_train, new_Y])

                        # Re-fit GP model
                        # 重新拟合GP模型
                        gp.fit(X_train, Y_train)

                        # Check and update RL result's current optimal solution
                        # 检查和更新RL结果的当前最优解
                        if rl_success and rl_result is not None and all(c <= 0 for c in rl_constraints):
                            merit = rl_result['merit']
                            
                            # Track Merit
                            # 跟踪Merit
                            self.all_merit.append(merit)

                    except Exception as e:
                        print(f"GP model update or RL result processing error: {e}")
                        print(f"GP模型更新或RL结果处理错误: {e}")
                        import traceback
                        traceback.print_exc()
                except ValueError as e:
                    print(f"Parameter conversion error: {e}")
                    print(f"参数转换错误: {e}")
                    print(f"x_tensor shape: {x_tensor.shape}, values: {x_tensor}")
                    continue

                # Periodically save results
                # 定期保存结果
                if iteration % 10 == 0 or iteration == num_iterations - 1:
                    self.save_results()

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                print(f"迭代 {iteration} 中的错误: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save final results
        # 保存最终结果
        self.save_results()

        print("\n===== RoSE-Opt optimization completed =====")
        print("\n===== RoSE-Opt优化完成 =====")

        return {
            "best_solution": self.best_merit_params,
            "best_metrics": self.best_merit_result,
            "best_iteration": self.best_merit_iteration,
            "pareto_front": self.pareto_front,
            "optimization_history": self.last_optimal_y
        }

    def print_final_results(self):
        """
        Print final optimization results
        打印最终优化结果
        """
        print("\n===== Final RoSE-Opt Results =====")
        print("\n===== 最终RoSE-Opt结果 =====")

        if self.best_merit_params is None:
            print("No valid solution found that satisfies all constraints.")
            print("未找到满足所有约束的有效解。")
            return

        print(f"Best Merit: {self.best_merit:.6e}")
        print(f"Found at iteration: {self.best_merit_iteration}")
        print(f"\nBest parameters:")
        print(f"最佳Merit: {self.best_merit:.6e}")
        print(f"发现于迭代: {self.best_merit_iteration}")
        print(f"\n最佳参数:")
        self.parameter_space.print_params(self.best_merit_params)

        print(f"\nPerformance metrics:")
        print(f"  - Min SNM: {self.best_merit_result['min_snm']:.6f}")
        print(f"  - Max Power: {self.best_merit_result['max_power']:.6e}")
        print(f"  - Area: {self.best_merit_result['area']*1e12:.2f} µm²")
        print(f"  - Read Delay: {self.best_merit_result['read_delay']:.6e}")
        print(f"  - Write Delay: {self.best_merit_result['write_delay']:.6e}")
        print(f"  - Delay constraints satisfied: {self.best_merit_result['read_delay_feasible']} (read), {self.best_merit_result['write_delay_feasible']} (write)")
        print(f"\n性能指标:")
        print(f"  - 最小SNM: {self.best_merit_result['min_snm']:.6f}")
        print(f"  - 最大功耗: {self.best_merit_result['max_power']:.6e}")
        print(f"  - 面积: {self.best_merit_result['area']*1e12:.2f} µm²")
        print(f"  - 读延迟: {self.best_merit_result['read_delay']:.6e}")
        print(f"  - 写延迟: {self.best_merit_result['write_delay']:.6e}")
        print(f"  - 延迟约束满足: {self.best_merit_result['read_delay_feasible']} (读), {self.best_merit_result['write_delay_feasible']} (写)")


# Main function to run the RoSE-Opt algorithm for SRAM optimization
# 运行RoSE-Opt算法进行SRAM优化的主函数
def main():
    """
    Main function to run RoSE-Opt optimization
    运行RoSE-Opt优化的主函数
    """
    print("===== SRAM Optimization using RoSE-Opt Algorithm =====")
    print("===== 使用RoSE-Opt算法的SRAM优化 =====")

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

    # Create SRAM environment
    # 创建SRAM环境
    env = SRAMCircuitEnv(parameter_space, max_steps=2)

    # Create optimizer and pass initial feasible point
    # 创建优化器并传递初始可行点
    optimizer = SRAM_BORL(
        env=env,
        parameter_space=parameter_space,
        objective_function=evaluate_sram,
        save_dir='sim/opt/results',
        hidden_dim=64,
        lr=3e-4,
        betas=(0.9, 0.999),
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        initial_result=initial_result,
        initial_params=initial_params
    )

    # Run RoSE-Opt training
    # 运行RoSE-Opt训练
    results = optimizer.train(num_iterations=400, debug_interval=20)

    # Print final results
    # 打印最终结果
    optimizer.print_final_results()

    # Calculate and print timing statistics
    # 计算并打印时间统计
    print("\n===== Timing Statistics =====")
    print(f"Total simulation time: {optimizer.simulation_time:.2f} seconds")
    print(f"Total search time: {optimizer.search_time:.2f} seconds")
    print("\n===== 时间统计 =====")
    print(f"总仿真时间: {optimizer.simulation_time:.2f} 秒")
    print(f"总搜索时间: {optimizer.search_time:.2f} 秒")

    print("\nOptimization completed!")
    print("\n优化完成！")


if __name__ == "__main__":
    # Set random seed for reproducibility
    # 设置随机种子以确保可重现性
    SEED = 1
    seed_set(seed=SEED)
    main()
    