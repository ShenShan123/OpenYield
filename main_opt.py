import os
import sys

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting SRAM Circuit Optimization Algorithms")
    print("=" * 60)

    config_file = "config_sram.yaml"
    print(f"Using configuration file: {config_file}")

    sys.path.append(os.path.join(os.path.dirname(__file__), 'size_optimization'))

    algorithms = {
        '1':  ('SA',       'Simulated Annealing'),
        '2':  ('PSO',      'Particle Swarm Optimization'),
        '3':  ('CBO',      'Constrained Bayesian Optimization'),
        '4':  ('RoSE_Opt', 'Reinforcement Learning Enhanced Bayesian Optimization'),
        '5':  ('SMAC',     'Sequential Model-based Algorithm Configuration'),
        '6':  ('CMA-ES',   'Covariance Matrix Adaptation Evolution Strategy'),
        '7':  ('MOEAD',    'Multi-Objective Evolutionary Algorithm based on Decomposition'),
        '8':  ('MOBO',     'Multi-Objective Bayesian Optimization'),
        '9':  ('NSGA-II',  'Non-dominated Sorting Genetic Algorithm II'),
        '10': ('tSS-BO',   'Truncated Subspace Sampling Bayesian Optimization'),
        '11': ('CPN',      'TabPFN-based Bayesian Optimization'),
        '12': ('Random',   'Random Search (baseline)'),
    }

    demo_modules = {
        'SA':       'demo_sa',
        'PSO':      'demo_pso',
        'CBO':      'demo_cbo',
        'RoSE_Opt': 'demo_roseopt',
        'SMAC':     'demo_smac',
        'CMA-ES':   'demo_cmaes',
        'MOEAD':    'demo_moead',
        'MOBO':     'demo_mobo',
        'NSGA-II':  'demo_nsgaii',
        'tSS-BO':   'demo_tssbo',
        'CPN':      'demo_cpn',
        'Random':   'demo_random',
    }

    print("\nAvailable optimization algorithms:")
    for num, (algo, desc) in algorithms.items():
        print(f"  {num:>2}. {algo:<12} - {desc}")

    print("\nSelect optimization algorithm(s) to run:")
    print("  Enter number(s) separated by commas (e.g., 1,3,5)")
    print("  Enter 'all' to run all algorithms")
    print("  Enter 'none' to skip")

    user_input = input("Your choice: ").strip().lower()

    if user_input == 'none':
        print("Skipping optimization.")
    else:
        if user_input == 'all':
            selected = [name for name, _ in algorithms.values()]
        else:
            try:
                indices = [x.strip() for x in user_input.split(',')]
                selected = [algorithms[i][0] for i in indices if i in algorithms]
                if not selected:
                    raise ValueError
            except (KeyError, ValueError):
                print("Invalid input. Running SA as default.")
                selected = ['SA']

        for algo in selected:
            print(f"\n{'=' * 20} Running {algo} {'=' * 20}")
            mod_name = demo_modules.get(algo)
            if mod_name is None:
                print(f"Unknown algorithm: {algo}")
                continue
            try:
                import importlib
                mod = importlib.import_module(f'size_optimization.{mod_name}')
                mod.main(config_file)
                print(f"{algo} completed.")
            except ImportError as e:
                print(f"Import error for {algo}: {e}")
                print(f"Make sure size_optimization/{mod_name}.py exists.")
            except Exception as e:
                import traceback
                print(f"Error running {algo}: {e}")
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("SRAM Optimization Session Completed")
    print("=" * 60)
