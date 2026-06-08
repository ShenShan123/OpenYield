import numpy as np
import itertools
import time
#import warnings  # 用于禁用ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from acquisition import (
    extract_pareto_front, 
    monte_carlo_ehvi, 
    create_individuals_from_front
)

class BayesianOptimizer:
    """
    多目标贝叶斯优化器
    """
    
    def __init__(self, problem, n_initial=200, n_iterations=1000, 
                 n_candidates_per_iter=3000, mc_samples=100, ref_point=None):
        """
        参数:
            problem: Problem对象
            n_initial: 初始随机采样数量
            n_iterations: 贝叶斯优化迭代次数
            n_candidates_per_iter: 每次迭代采样的候选点数量（降低以加速）
            mc_samples: 蒙特卡洛采样次数（降低以加速）
            ref_point: 参考点
        """
        self.problem = problem
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_candidates_per_iter = n_candidates_per_iter
        self.mc_samples = mc_samples
        self.ref_point = ref_point
        self.num_objectives = len(problem.objectives)
        
        # 标准化器（关键：处理MRED和PDP的巨大数值差异）
        self.scaler_X = StandardScaler()  # 新增：输入特征标准化
        self.scaler_obj = [StandardScaler() for _ in range(self.num_objectives)]
        
        # 高斯过程模型（针对大数值差异优化：MRED~0.1 vs PDP~250）
        # 使用更宽松的核函数范围和更强的正则化
        # 降低length_scale下界：1e-3 → 1e-4（减少ConvergenceWarning）
        self.gp_models = []
        for _ in range(self.num_objectives):
            kernel = C(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-4, 1e3))
            self.gp_models.append(
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-3,
                    n_restarts_optimizer=1,
                    normalize_y=True,
                    optimizer='fmin_l_bfgs_b'
                )
            )
        
        # 不再预生成所有候选点（避免内存爆炸）
        # 搜索空间：8^9 = 134,217,728 太大！
        # 改为按需动态生成
        print(f"搜索空间大小: 8^{problem.num_of_variables} = {8**problem.num_of_variables:,}")
        print("使用动态采样策略（避免内存爆炸）")
        
    def optimize(self):
        """执行贝叶斯优化"""
        print(f"\n{'='*60}")
        print(f"开始贝叶斯优化（性能优化版）")
        print(f"初始采样: {self.n_initial}次")
        print(f"贝叶斯迭代: {self.n_iterations}次")
        print(f"每次迭代候选点: {self.n_candidates_per_iter}个")
        print(f"蒙特卡洛采样: {self.mc_samples}次")
        print(f"总评估次数: {self.n_initial + self.n_iterations}次")
        print(f"{'='*60}\n")
        
        # 初始采样
        print("第1步: 初始随机采样...")
        X_train, Y_train = self._initial_sampling()
        
        # 设置参考点
        if self.ref_point is None:
            self.ref_point = (np.max(Y_train, axis=0) * 1.2).tolist()
            ref_str = ", ".join([f"{v:.4f}" for v in self.ref_point])
            print(f"自动设置参考点: [{ref_str}]")
        
        # 贝叶斯迭代
        print(f"\n第2步: 贝叶斯优化迭代 ({self.n_iterations}次)...")
        
        for iteration in tqdm(range(self.n_iterations), desc="优化进度"):
            iter_start_time = time.time()
            
            # 智能数据选择策略：保留帕累托前沿 + 随机采样
            max_train_size = 600  # 增加到600个点以提高模型质量
            
            if len(X_train) > max_train_size:
                # 提取当前帕累托前沿索引
                pareto_indices_temp = extract_pareto_front(Y_train)
                n_pareto = len(pareto_indices_temp)
                
                # 计算需要随机采样的数量
                n_random = max_train_size - n_pareto
                
                if n_random > 0:
                    # 随机选择非帕累托点
                    non_pareto_indices = [i for i in range(len(X_train)) 
                                         if i not in pareto_indices_temp]
                    if len(non_pareto_indices) > n_random:
                        random_indices = np.random.choice(
                            non_pareto_indices, 
                            size=n_random, 
                            replace=False
                        )
                        selected_indices = list(pareto_indices_temp) + list(random_indices)
                    else:
                        selected_indices = list(range(len(X_train)))
                else:
                    # 如果帕累托点已经超过max_train_size，只用帕累托点
                    selected_indices = pareto_indices_temp
                
                X_train_selected = X_train[selected_indices]
                Y_train_selected = Y_train[selected_indices]
            else:
                X_train_selected = X_train
                Y_train_selected = Y_train
            
            # 标准化输入特征（重要：输入也需要标准化）
            X_train_scaled = self.scaler_X.fit_transform(X_train_selected)
            
            # 标准化目标值（虽然GP内部有normalize_y，但外部标准化可以提前处理数值差异）
            Y_scaled = []
            for i in range(self.num_objectives):
                y_scaled = self.scaler_obj[i].fit_transform(
                    Y_train_selected[:, i].reshape(-1, 1)
                ).flatten()
                Y_scaled.append(y_scaled)
            
            # 训练GP（带异常处理）
            gp_start = time.time()
            try:
                for i in range(self.num_objectives):
                    self.gp_models[i].fit(X_train_scaled, Y_scaled[i])
            except Exception as e:
                tqdm.write(f"警告: GP训练出现异常 - {str(e)}, 使用上一次模型")
            gp_time = time.time() - gp_start
            
            # 提取帕累托前沿
            pareto_indices = extract_pareto_front(Y_train)
            pareto_front = Y_train[pareto_indices].tolist()
            
            # 找最佳候选点
            acq_start = time.time()
            best_candidate = self._find_best_candidate(X_train, pareto_front)
            acq_time = time.time() - acq_start
            
            # 评估新点
            eval_start = time.time()
            y_new = self._evaluate(best_candidate)
            eval_time = time.time() - eval_start
            
            # 添加到训练集
            X_train = np.vstack([X_train, best_candidate])
            Y_train = np.vstack([Y_train, y_new])
            
            iter_time = time.time() - iter_start_time
            
            # 每10次迭代显示详细信息（改为更频繁的监控）
            if (iteration + 1) % 10 == 0:
                current_pareto = extract_pareto_front(Y_train)
                # 计算帕累托前沿的变化
                pareto_change = len(current_pareto) - len(pareto_indices)
                change_symbol = "+" if pareto_change > 0 else ""
                
                tqdm.write(
                    f"\n迭代 {iteration + 1}/{self.n_iterations}: "
                    f"Pareto前沿 {len(current_pareto)} 个解 ({change_symbol}{pareto_change}) | "
                    f"耗时: {iter_time:.1f}s (GP:{gp_time:.1f}s, ACQ:{acq_time:.1f}s, EVAL:{eval_time:.1f}s)"
                )
        
        # 提取最终帕累托前沿
        print(f"\n第3步: 提取最终帕累托前沿...")
        pareto_indices = extract_pareto_front(Y_train)
        X_pareto = X_train[pareto_indices]
        Y_pareto = Y_train[pareto_indices]
        
        print(f"优化完成！帕累托前沿共有 {len(pareto_indices)} 个解")
        print(f"目标值范围:")
        for i in range(self.num_objectives):
            print(f"  Objective {i + 1}: [{np.min(Y_pareto[:, i]):.6f}, {np.max(Y_pareto[:, i]):.6f}]")
        
        pareto_individuals = create_individuals_from_front(X_pareto, Y_pareto)
        return pareto_individuals
    
    def _initial_sampling(self):
        """初始随机采样"""
        X_train = []
        Y_train = []
        
        for _ in tqdm(range(self.n_initial), desc="初始采样"):
            individual = self.problem.generate_individual()
            x = individual.features
            y = self._evaluate(x)
            X_train.append(x)
            Y_train.append(y)
        
        return np.array(X_train), np.array(Y_train)
    
    def _evaluate(self, x):
        """评估目标值"""
        if self.problem.expand:
            obj_values = [f(*x) for f in self.problem.objectives]
        else:
            obj_values = [f(x) for f in self.problem.objectives]
        return obj_values
    
    def _find_best_candidate(self, X_train, pareto_front):
        """找EHVI最大的候选点（带详细进度）"""
        evaluated_set = set(map(tuple, X_train))
        
        # 采样候选点
        candidates = self._sample_candidates(evaluated_set)
        
        if len(candidates) == 0:
            print("警告: 没有未评估的候选点，随机选择")
            return self.problem.generate_individual().features
        
        candidates = np.array(candidates)
        
        # 标准化候选点（必须使用与训练时相同的标准化）
        candidates_scaled = self.scaler_X.transform(candidates)
        
        mu_list = []
        sigma_list = []
        for i in range(self.num_objectives):
            mu_scaled, sigma_scaled = self.gp_models[i].predict(
                candidates_scaled, return_std=True
            )
            mu = self.scaler_obj[i].inverse_transform(
                mu_scaled.reshape(-1, 1)
            ).flatten()
            sigma = sigma_scaled * self.scaler_obj[i].scale_[0]
            mu_list.append(mu)
            sigma_list.append(sigma)
        
        # 计算EHVI（使用优化的采样次数）
        best_ehvi = -np.inf
        best_idx = 0
        
        for i in range(len(candidates)):
            mu_vec = [mu_list[j][i] for j in range(self.num_objectives)]
            sigma_vec = [sigma_list[j][i] for j in range(self.num_objectives)]
            ehvi = monte_carlo_ehvi(
                mu_vec, sigma_vec,
                pareto_front,
                self.ref_point,
                n_samples=self.mc_samples  # 使用优化的采样次数
            )
            
            if ehvi > best_ehvi:
                best_ehvi = ehvi
                best_idx = i
        
        return candidates[best_idx]
    
    def _sample_candidates(self, evaluated_set):
        """动态随机采样候选点（避免预生成所有候选点）"""
        candidates = []
        max_attempts = self.n_candidates_per_iter * 10  # 最多尝试10倍次数
        attempts = 0
        
        # 动态生成候选点
        while len(candidates) < self.n_candidates_per_iter and attempts < max_attempts:
            # 随机生成一个候选点
            candidate = tuple(
                np.random.randint(
                    self.problem.variables_range[i][0],
                    self.problem.variables_range[i][1] + 1
                )
                for i in range(self.problem.num_of_variables)
            )
            
            # 检查是否已评估过
            if candidate not in evaluated_set and candidate not in [tuple(c) for c in candidates]:
                candidates.append(list(candidate))
            
            attempts += 1
        
        if len(candidates) == 0:
            # 如果实在找不到，就随机生成一个
            candidates = [[
                np.random.randint(
                    self.problem.variables_range[i][0],
                    self.problem.variables_range[i][1] + 1
                )
                for i in range(self.problem.num_of_variables)
            ]]
        
        return candidates
