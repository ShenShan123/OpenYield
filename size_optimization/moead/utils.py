"""
MOEAD工具函数
适配连续+离散混合变量优化问题
"""

import os
from math import sqrt
import numpy as np
import random


def Load_W(moead):
    """加载权重向量"""
    file = moead.name + '.csv'
    path = os.path.join(moead.csv_file_path, file)
    if not os.path.exists(path):
        print(f'权重文件不存在: {path}')
        from .mean_vector import Mean_vector
        mv = Mean_vector(moead.h, moead.problem.num_of_objectives, path)
        mv.generate()
        print(f'已生成权重文件: {path}')
    W = np.loadtxt(fname=path, delimiter=',')
    moead.Pop_size = W.shape[0]
    moead.W = W
    print(f'加载权重向量: {W.shape[0]}个')
    return W


def cpt_Z(moead):
    """初始化参考点Z(理想点)和Z_max(天底点) - 对于最小化问题"""
    Z = []
    Z_max = []
    for fi in range(moead.problem.num_of_objectives):
        Z.append(1e10)
        Z_max.append(-1e10)
    moead.Z = Z
    moead.Z_max = Z_max
    return Z


def update_Z(moead, individual):
    """根据新个体更新参考点Z(理想点)和Z_max(天底点)"""
    objectives = individual.objectives
    for j in range(moead.problem.num_of_objectives):
        if objectives[j] < moead.Z[j]:
            moead.Z[j] = objectives[j]
        if objectives[j] > moead.Z_max[j]:
            moead.Z_max[j] = objectives[j]


def cpt_W_Bi_T(moead):
    """计算每个权重向量的T个最近邻居"""
    if moead.T_size < 1:
        return -1
    for bi in range(moead.W.shape[0]):
        Bi = moead.W[bi]
        DIS = np.sum((moead.W - Bi) ** 2, axis=1)
        B_T = np.argsort(DIS)
        B_T = B_T[1:moead.T_size + 1]
        moead.W_Bi_T.append(B_T)
    print(f'计算邻域完成,每个个体{moead.T_size}个邻居')


def Tchebycheff_dist(w, f, z):
    """Tchebycheff距离(由cpt_tchbycheff内部调用，已包含归一化)"""
    return w * abs(f - z)


def cpt_tchbycheff(moead, idx, individual):
    """计算Tchebycheff标量化函数值 - 带目标归一化"""
    max_val = -1e10
    ri = moead.W[idx]
    F_X = individual.objectives
    
    for i in range(moead.problem.num_of_objectives):
        # 归一化：除以各目标的动态范围 (Z_max - Z)
        scale = moead.Z_max[i] - moead.Z[i]
        if scale < 1e-20:
            scale = 1.0  # 避免除零
        fi = ri[i] * abs(F_X[i] - moead.Z[i]) / scale
        if fi > max_val:
            max_val = fi
    return max_val


def update_BTX(moead, P_B, new_individual):
    """根据新个体更新邻域内的个体"""
    for j in P_B:
        old_individual = moead.Pop[j]
        d_old = cpt_tchbycheff(moead, j, old_individual)
        d_new = cpt_tchbycheff(moead, j, new_individual)
        
        if d_new < d_old:
            moead.Pop[j].features = new_individual.features[:]
            moead.Pop[j].objectives = new_individual.objectives[:]
            update_EP_By_ID(moead, j, new_individual.objectives)


def is_dominate(objectives1, objectives2):
    """判断objectives1是否支配objectives2 (最小化问题)"""
    better_in_any = False
    for f1, f2 in zip(objectives1, objectives2):
        if f1 > f2:
            return False
        if f1 < f2:
            better_in_any = True
    return better_in_any


def init_EP(moead):
    """初始化Pareto前沿"""
    for pi in range(moead.Pop_size):
        individual = moead.Pop[pi]
        is_pareto = True
        
        for ppi in range(moead.Pop_size):
            if pi != ppi:
                other_individual = moead.Pop[ppi]
                if is_dominate(other_individual.objectives, individual.objectives):
                    is_pareto = False
                    break
        
        if is_pareto:
            moead.EP_X_ID.append(pi)
            moead.EP_X_FV.append(individual.objectives[:])
    print(f'初始Pareto前沿: {len(moead.EP_X_ID)}个解')


def update_EP_By_ID(moead, id, objectives):
    """如果id在EP中,更新其目标值"""
    if id in moead.EP_X_ID:
        position_pi = moead.EP_X_ID.index(id)
        moead.EP_X_FV[position_pi][:] = objectives[:]


def update_EP_By_Individual(moead, id_new, new_individual):
    """根据新个体更新Pareto前沿"""
    objectives_new = new_individual.objectives
    
    # 检查是否被现有EP中的解支配
    dominated_by_ep = False
    for ep_obj in moead.EP_X_FV:
        if is_dominate(ep_obj, objectives_new):
            dominated_by_ep = True
            break
    
    if dominated_by_ep:
        return
    
    # 检查目标值距离(避免重复解)
    epsilon = 0.001
    for ep_obj in moead.EP_X_FV:
        dist = sum((ep_obj[i] - objectives_new[i])**2 for i in range(len(ep_obj)))**0.5
        if dist < epsilon:
            return
    
    # 删除被新解支配的EP中的解
    new_EP_X_ID = []
    new_EP_X_FV = []
    for i in range(len(moead.EP_X_ID)):
        if not is_dominate(objectives_new, moead.EP_X_FV[i]):
            new_EP_X_ID.append(moead.EP_X_ID[i])
            new_EP_X_FV.append(moead.EP_X_FV[i])
    
    moead.EP_X_ID = new_EP_X_ID
    moead.EP_X_FV = new_EP_X_FV
    
    # 添加新解到EP
    moead.EP_X_ID.append(id_new)
    moead.EP_X_FV.append(objectives_new[:])


def Creat_Pop(moead):
    """创建初始种群"""
    Pop = []
    print(f'生成初始种群: {moead.Pop_size}个个体...')
    
    for i in range(moead.Pop_size):
        individual = moead.problem.generate_individual()
        if i == 0:
            default_features = getattr(moead.problem, "default_features", None)
            if default_features:
                individual.features[: len(default_features)] = list(default_features)
            elif len(individual.features) >= 2:
                individual.features[0] = 0
                individual.features[1] = 0
        moead.problem.calculate_objectives(individual)
        Pop.append(individual)
        
        if (i + 1) % 100 == 0:
            print(f'  已生成 {i + 1}/{moead.Pop_size}')
    
    moead.Pop = Pop
    print('初始种群生成完成')
    return Pop
