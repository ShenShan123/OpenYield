"""
MOEAD遗传算法工具
支持连续浮点变量和离散整数变量的混合优化
"""

import numpy as np
import random
from .utils import cpt_tchbycheff


def continuous_sbx_crossover(moead, parent1_features, parent2_features, eta=20):
    """SBX交叉 - 适用于连续变量"""
    var_num = len(parent1_features)
    child1 = []
    child2 = []
    
    for i in range(var_num):
        if moead.problem.variables_type[i] == 'float':
            if random.random() < 0.9:
                p1, p2 = parent1_features[i], parent2_features[i]
                
                if abs(p1 - p2) > 1e-14:
                    lb, ub = moead.problem.variables_range[i]
                    y1 = min(p1, p2)
                    y2 = max(p1, p2)
                    
                    beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))
                    
                    u = random.random()
                    if u <= 1.0 / alpha:
                        beta_q = (u * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    c1 = max(lb, min(c1, ub))
                    c2 = max(lb, min(c2, ub))
                    
                    child1.append(c1)
                    child2.append(c2)
                else:
                    child1.append(p1)
                    child2.append(p2)
            else:
                child1.append(parent1_features[i])
                child2.append(parent2_features[i])
        else:
            child1.append(parent1_features[i])
            child2.append(parent2_features[i])
    
    return child1, child2


def continuous_polynomial_mutation(moead, individual_features, eta=20):
    """多项式变异 - 适用于连续变量"""
    var_num = len(individual_features)
    mutated = individual_features[:]
    
    mutation_rate = 1.0 / var_num
    
    for i in range(var_num):
        if moead.problem.variables_type[i] == 'float':
            if random.random() < mutation_rate:
                y = mutated[i]
                lb, ub = moead.problem.variables_range[i]
                
                delta1 = (y - lb) / (ub - lb)
                delta2 = (ub - y) / (ub - lb)
                
                u = random.random()
                if u <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                    delta_q = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val ** (1.0 / (eta + 1.0))
                
                mutated[i] = y + delta_q * (ub - lb)
                mutated[i] = max(lb, min(mutated[i], ub))
    
    return mutated


def discrete_crossover(moead, parent1_features, parent2_features):
    """单点交叉 - 适用于离散变量"""
    var_num = len(parent1_features)
    child1 = parent1_features[:]
    child2 = parent2_features[:]
    
    if random.random() < 0.9:
        point = random.randint(1, var_num - 1)
        child1 = parent1_features[:point] + parent2_features[point:]
        child2 = parent2_features[:point] + parent1_features[point:]
    
    return child1, child2


def discrete_mutation(moead, individual_features):
    """随机替换变异 - 适用于离散变量"""
    var_num = len(individual_features)
    mutated = individual_features[:]
    
    mutation_rate = 1.0 / var_num
    
    for i in range(var_num):
        if random.random() < mutation_rate:
            lb, ub = moead.problem.variables_range[i]
            current_value = mutated[i]
            candidates = [v for v in range(int(lb), int(ub) + 1) if v != current_value]
            if candidates:
                mutated[i] = random.choice(candidates)
    
    return mutated


def mixed_crossover(moead, parent1_features, parent2_features):
    """混合交叉: 连续用SBX,离散用单点交叉"""
    var_num = len(parent1_features)
    
    child1_cont, child2_cont = continuous_sbx_crossover(moead, parent1_features, parent2_features)
    
    discrete_indices = [i for i in range(var_num) if moead.problem.variables_type[i] == 'int']
    
    if discrete_indices and random.random() < 0.9:
        point = random.randint(1, len(discrete_indices))
        for idx_pos in range(len(discrete_indices)):
            i = discrete_indices[idx_pos]
            if idx_pos < point:
                child1_cont[i] = parent1_features[i]
                child2_cont[i] = parent2_features[i]
            else:
                child1_cont[i] = parent2_features[i]
                child2_cont[i] = parent1_features[i]
    
    return child1_cont, child2_cont


def mixed_mutation(moead, individual_features):
    """混合变异: 连续用多项式,离散用随机替换"""
    var_num = len(individual_features)
    mutated = individual_features[:]
    
    mutation_rate = 1.0 / var_num
    
    for i in range(var_num):
        if random.random() < mutation_rate:
            if moead.problem.variables_type[i] == 'float':
                y = mutated[i]
                lb, ub = moead.problem.variables_range[i]
                
                delta1 = (y - lb) / (ub - lb)
                delta2 = (ub - y) / (ub - lb)
                
                u = random.random()
                eta = 20
                if u <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                    delta_q = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val ** (1.0 / (eta + 1.0))
                
                mutated[i] = y + delta_q * (ub - lb)
                mutated[i] = max(lb, min(mutated[i], ub))
            else:
                lb, ub = moead.problem.variables_range[i]
                current_value = mutated[i]
                candidates = [v for v in range(int(lb), int(ub) + 1) if v != current_value]
                if candidates:
                    mutated[i] = random.choice(candidates)
    
    # 边界检查
    for i in range(var_num):
        lb, ub = moead.problem.variables_range[i]
        if moead.problem.variables_type[i] == 'int':
            mutated[i] = int(max(lb, min(mutated[i], ub)))
        else:
            mutated[i] = max(lb, min(mutated[i], ub))
    
    return mutated


def generate_offspring(moead, gen, wi, parent_idx1, parent_idx2):
    """生成子代 - 自动适配混合变量类型"""
    p1 = moead.Pop[parent_idx1]
    p2 = moead.Pop[parent_idx2]
    
    # 检查变量类型
    has_continuous = any(vt == 'float' for vt in moead.problem.variables_type)
    has_discrete = any(vt == 'int' for vt in moead.problem.variables_type)
    
    if has_continuous and has_discrete:
        child1_features, child2_features = mixed_crossover(moead, p1.features, p2.features)
        child_features = child1_features if random.random() < 0.5 else child2_features
        child_features = mixed_mutation(moead, child_features)
    elif has_continuous:
        child1_features, child2_features = continuous_sbx_crossover(moead, p1.features, p2.features)
        child_features = child1_features if random.random() < 0.5 else child2_features
        child_features = continuous_polynomial_mutation(moead, child_features)
    else:
        child1_features, child2_features = discrete_crossover(moead, p1.features, p2.features)
        child_features = child1_features if random.random() < 0.5 else child2_features
        child_features = discrete_mutation(moead, child_features)
    
    # 创建子代个体
    child = moead.problem.generate_individual()
    child.features = child_features
    moead.problem.calculate_objectives(child)
    
    return child


def envolution(moead):
    """MOEAD主进化循环"""
    from .utils import update_Z, update_BTX, update_EP_By_Individual
    
    print('\n开始MOEAD优化...')
    print(f'种群大小: {moead.Pop_size}, 迭代次数: {moead.max_gen}, 邻域大小: {moead.T_size}')
    print('-' * 70)
    
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print('警告: 未安装tqdm,将使用基础进度显示')
    
    iterator = tqdm(range(moead.max_gen), desc='进化进度') if use_tqdm else range(moead.max_gen)
    
    for gen in iterator:
        moead.gen = gen
        
        for pi in range(moead.Pop_size):
            Bi = moead.W_Bi_T[pi]
            
            k = random.randint(0, moead.T_size - 1)
            l = random.randint(0, moead.T_size - 1)
            while l == k:
                l = random.randint(0, moead.T_size - 1)
            
            ik = Bi[k]
            il = Bi[l]
            
            new_individual = generate_offspring(moead, gen, pi, ik, il)
            update_Z(moead, new_individual)
            update_BTX(moead, Bi, new_individual)
            update_EP_By_Individual(moead, pi, new_individual)
        
        if (gen + 1) % 10 == 0 or gen == 0:
            msg = f'代数 {gen+1:4d}/{moead.max_gen}, Pareto前沿: {len(moead.EP_X_ID):3d}个解'
            if use_tqdm:
                tqdm.write(msg)
            else:
                print(msg)
    
    print('\n' + '-' * 70)
    print(f'优化完成! 最终Pareto前沿: {len(moead.EP_X_ID)}个解')
    
    return moead.EP_X_ID
