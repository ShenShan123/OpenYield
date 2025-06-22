import numpy as np
import re
import subprocess
import os
import torch.nn as nn


class Spice():
    #write threshold
    threshold = 8.86271e-10
    #read threshold
    #threshold = 6.948e-10
    def __init__(self, feature_num, means=None):
        self.feature_num = feature_num
        if feature_num ==18:
            bound_file_path = '/home/lixy/sram_yield_estimation/bound_lib/model_18_bound.txt'
        elif feature_num ==108:
            bound_file_path = '/home/lixy/sram_yield_estimation/bound_lib/model_108_bound.txt'
        elif feature_num ==1152:
            bound_file_path = '/home/lixy/sram_yield_estimation/bound_lib/model_1152_bound.txt'
        bounds = np.loadtxt(bound_file_path, dtype=np.float64)
        self.up_bounds = bounds[0, :]
        self.low_bounds = bounds[1, :]
        self.bound = np.vstack([self.low_bounds, self.up_bounds])
        if means is None:
            self.means = np.zeros(feature_num)
        else:
            self.means = means

    def get_initial_x(self, N):
        # 协方差矩阵为单位矩阵
        cov = np.eye(self.feature_num)
        # 使用多元正态分布生成样本
        x = np.random.multivariate_normal(self.means, cov, N)
        return x
    def save_y_to_txt(self, y, file_path):
        try:
            if len(y.shape) != 2 or y.shape[1] != 1:
                raise ValueError("输入的数组不是 n*1 的形状。")
            # 以追加模式打开文件
            with open(file_path, 'a') as f:
                # 使用 numpy 的 savetxt 函数将数组保存到 txt 文件，采用科学计数法
                np.savetxt(f, y, fmt='%e')
            #print(f"数组已成功追加保存到 {file_path}。")
        except Exception as e:
            print(f"保存数组时出现错误: {e}")
    def indicator(self, y):
        return (y > self.threshold) | (y < 0)

    def get_yield(self, data_size):
        result = self.get_initial_x(data_size)
        y = self.run_simulation(result)
        y = np.array(y)
        fail_num = self.indicator(y).sum()
        return 1 - (fail_num / data_size)


if __name__ == "__main__":
    netlist_file = 'sram.sp'
    feature_num = 18
    means = np.array([0.322, 0.045, -0.13, -0.3021, 0.02, -0.126, 0.322, 0.045, -0.13, -0.3021, 0.02, -0.126, 0.322,
                      0.045, -0.13, 0.322, 0.045, -0.13])
    simulator = Spice(netlist_file, feature_num, means)

    data_size = 15
    result = simulator.get_initial_x(data_size)
    y = simulator.run_simulation(result)
    # yield_result = simulator.get_yield(data_size)
    # print(f"良率: {yield_result}")
    