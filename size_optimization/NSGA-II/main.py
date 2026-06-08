import numpy as np
from problem import Problem
from evolution import Evolution

def zdt1_f1(x):
    return x[0]

def zdt1_f2(x):
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    return g * (1 - np.sqrt(x[0] / g))

def main():
    print("Running Generic NSGA-II on ZDT1-like problem")
    
    num_vars = 10
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
    
    evo = Evolution(
        problem,
        num_of_generations=20,
        num_of_individuals=50
    )
    
    pareto_front = evo.evolve()
    print(f"Found {len(pareto_front)} Pareto solutions.")

if __name__ == "__main__":
    main()
