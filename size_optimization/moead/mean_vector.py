"""
均值向量生成工具
用于生成MOEAD算法的权重向量
"""

import numpy as np
import os


class Mean_vector:
    """生成多目标优化的权重向量"""
    
    def __init__(self, H=99, m=2, path='out.csv'):
        """
        H: 分割数,对于2目标会生成H+1个权重向量
        m: 目标数量
        path: 保存路径
        """
        self.H = H
        self.m = m
        self.path = path
        self.stepsize = 1 / H

    def perm(self, sequence):
        """序列全排列,且无重复"""
        l = sequence
        if len(l) <= 1:
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_mean_vectors(self):
        """生成权重向量"""
        H = self.H
        m = self.m
        sequence = []
        
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        
        ws = []
        pe_seq = self.perm(sequence)
        
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            
            if weight not in ws:
                ws.append(weight)
        
        return ws

    def save_mv_to_file(self, mv):
        """保存权重向量到文件"""
        # 确保路径的目录存在
        dir_path = os.path.dirname(self.path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        f = np.array(mv, dtype=np.float64)
        np.savetxt(fname=self.path, X=f, delimiter=',', fmt='%.6f')
        print(f'权重向量已保存到: {self.path}')
        print(f'权重向量数量: {len(mv)}')

    def generate(self):
        """生成并保存权重向量"""
        m_v = self.get_mean_vectors()
        self.save_mv_to_file(m_v)
        return m_v
