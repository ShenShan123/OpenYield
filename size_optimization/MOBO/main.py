import numpy as np
from problem import Problem
from bayesian_optimizer import BayesianOptimizer

def zdt1_f1(x):
    return x[0]

def zdt1_f2(x):
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    return g * (1 - np.sqrt(x[0] / g))

def main():
    print("Running Generic MOBO on ZDT1-like problem")
    # 5 variables, mapped to [0,1]
    # Using integer variables [0, 100] to be compatible with discrete optimizer
    num_vars = 5
    variables_range = [(0, 100)]
    variables_type = ['int'] * num_vars
    
    def obj1(x):
        x_norm = [xi/100.0 for xi in x]
        return zdt1_f1(np.array(x_norm))
    
    def obj2(x):
        x_norm = [xi/100.0 for xi in x]
        return zdt1_f2(np.array(x_norm))

    problem = Problem(
        objectives=[obj1, obj2],
        num_of_variables=num_vars,
        variables_range=variables_range,
        variables_type=variables_type,
        same_range=True,
        expand=False 
    )
    
    optimizer = BayesianOptimizer(
        problem,
        n_initial=20,
        n_iterations=10,
        n_candidates_per_iter=100
    )
    
    pareto_solutions = optimizer.optimize()
    print(f"Found {len(pareto_solutions)} Pareto solutions.")

if __name__ == "__main__":
    main()
