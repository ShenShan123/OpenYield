"""
MOEAD优化器核心类
支持连续+离散混合变量的多目标优化
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib
from .problem import Problem
from .utils import Load_W, cpt_W_Bi_T, Creat_Pop, cpt_Z, init_EP
from .ga_utils import envolution

matplotlib.use('Agg')


class MOEAD:
    """MOEAD算法类"""
    
    def __init__(self, problem, h=99, max_gen=100, T_size=20, name='sram_moead'):
        """
        参数:
            problem: Problem对象
            h: 权重向量分割数 (2目标时生成h+1个向量)
            max_gen: 最大迭代次数
            T_size: 邻域大小
            name: 问题名称,用于权重文件命名
        """
        self.problem = problem
        self.name = name
        self.h = h
        self.max_gen = max_gen
        self.T_size = T_size
        
        self.Pop_size = -1
        self.Pop = []
        
        self.W = []
        self.W_Bi_T = []
        
        self.Z = []
        
        self.EP_X_ID = []
        self.EP_X_FV = []
        
        # 权重文件路径 - 保存在size_optimization/vector_csv_file/
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_file_path = os.path.join(os.path.dirname(current_dir), 'vector_csv_file')
        os.makedirs(self.csv_file_path, exist_ok=True)
        
        self.gen = 0
        
        print('=' * 70)
        print('MOEAD优化器初始化')
        print(f'问题: {name}')
        print(f'变量维度: {problem.num_of_variables}')
        print(f'目标数量: {problem.num_of_objectives}')
        print(f'权重向量分割数H: {h} (将生成{h+1}个权重向量)')
        print(f'最大迭代次数: {max_gen}')
        print(f'邻域大小: {T_size}')
        print('=' * 70)
    
    def Init_data(self):
        """初始化数据"""
        print('\n初始化MOEAD数据结构...')
        
        Load_W(self)
        cpt_W_Bi_T(self)
        Creat_Pop(self)
        cpt_Z(self)
        
        # 从初始种群更新Z(理想点)和Z_max(天底点)
        from .utils import update_Z
        for individual in self.Pop:
            update_Z(self, individual)
        
        init_EP(self)
        
        print('初始化完成!\n')
    
    def run(self):
        """运行MOEAD优化"""
        t = time.time()
        
        self.Init_data()
        
        EP_X_ID = envolution(self)
        
        dt = time.time() - t
        
        print(f'\n总用时: {dt:.2f}秒 ({dt/60:.2f}分钟)')
        
        return EP_X_ID
    
    def get_pareto_front(self):
        """获取Pareto前沿的解"""
        states = []
        objectives = []
        
        for idx in self.EP_X_ID:
            individual = self.Pop[idx]
            states.append(individual.features[:])
            objectives.append(individual.objectives[:])
        
        return states, objectives
    
    def save_results(self, output_file='moead_pareto_solutions.txt'):
        """保存Pareto前沿结果"""
        states, objectives = self.get_pareto_front()
        
        with open(output_file, 'w') as f:
            f.write('# Pareto Front Solutions\n')
            f.write(f'# Total: {len(states)} solutions\n')
            f.write('# Format: features | objectives\n')
            f.write('state\tobjectives\n')
            for state, obj in zip(states, objectives):
                state_str = ','.join(f'{s:.6e}' if isinstance(s, float) else str(s) for s in state)
                obj_str = ','.join(f'{o:.6e}' for o in obj)
                f.write(f'{state_str}\t{obj_str}\n')
        
        print(f'\nPareto前沿已保存到: {output_file}')
        print(f'共 {len(states)} 个解')
    
    def plot_pareto_front(self, output_file='moead_pareto_front.png', obj_labels=None):
        """绘制Pareto前沿"""
        states, objectives = self.get_pareto_front()
        
        if len(objectives) == 0:
            print('警告: Pareto前沿为空,无法绘图')
            return
        
        if self.problem.num_of_objectives == 2:
            obj1 = [obj[0] for obj in objectives]
            obj2 = [obj[1] for obj in objectives]
            
            plt.figure(figsize=(10, 8))
            plt.scatter(obj1, obj2, alpha=0.6, edgecolor='k', s=50)
            plt.xlabel(obj_labels[0] if obj_labels else 'Objective 1', fontsize=12)
            plt.ylabel(obj_labels[1] if obj_labels else 'Objective 2', fontsize=12)
            plt.title(f'MOEAD Pareto Front (Gen {self.max_gen})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f'Pareto前沿图已保存到: {output_file}')
        
        elif self.problem.num_of_objectives == 3:
            from mpl_toolkits.mplot3d import Axes3D
            obj1 = [obj[0] for obj in objectives]
            obj2 = [obj[1] for obj in objectives]
            obj3 = [obj[2] for obj in objectives]
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(obj1, obj2, obj3, alpha=0.6, edgecolor='k', s=50)
            ax.set_xlabel(obj_labels[0] if obj_labels else 'Objective 1', fontsize=12)
            ax.set_ylabel(obj_labels[1] if obj_labels else 'Objective 2', fontsize=12)
            ax.set_zlabel(obj_labels[2] if obj_labels else 'Objective 3', fontsize=12)
            ax.set_title(f'MOEAD Pareto Front (Gen {self.max_gen})', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f'3D Pareto前沿图已保存到: {output_file}')
        else:
            print(f'不支持{self.problem.num_of_objectives}目标的可视化')
