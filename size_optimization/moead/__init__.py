"""
MOEAD (Multi-Objective Evolutionary Algorithm based on Decomposition) Module
多目标进化算法模块 - 基于分解的多目标优化
"""

from .optimizer import MOEAD
from .problem import Problem
from .individual import Individual
from .population import Population

__all__ = ['MOEAD', 'Problem', 'Individual', 'Population']
