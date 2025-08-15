"""
    normal distribution  正态分布
"""

from math import pi
from scipy.special import logsumexp
import mpmath as mp
import numpy as np
from scipy.stats import multivariate_normal

class norm_dist():
    def __init__(self, mu, var):
        """
        :param mu: d
        :param var: k x d
        """
        self.mu = mu
        self.var = var

    #从正态分布中进行采样，返回指定数量的样本
    def sample(self, n):
        return np.random.multivariate_normal(mean=self.mu, cov=self.var, size=n)

    #计算输入数据 x 在正态分布下的对数概率密度函数
    def log_pdf(self, x):
        return multivariate_normal(mean=self.mu, cov=self.var).logpdf(x)

    #计算输入数据 x 在正态分布下的累积分布函数
    def log_cdf(self, x):
        return multivariate_normal(mean=self.mu, cov=self.var).logcdf(x)

    #计算输入数据 x 在正态分布下的概率密度函数值
    def pdf(self, x, mp_format=False):
        log_pdf = self.log_pdf(x)
        mp_exp_broad = np.frompyfunc(mp.exp, 1, 1)
        prob = mp_exp_broad(log_pdf)
        if mp_format==False:
            prob = prob.astype(float)
        return prob

if __name__ == "__main__":

    pass