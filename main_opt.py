from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench
from sram_compiler.testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
import numpy as np
from utils import estimate_bitcell_area

if __name__ == '__main__':
  
    # ===== SRAM Optimization Algorithms =====
    print("\n" + "="*60)
    print("Starting SRAM Circuit Optimization Algorithms")
    print("="*60)
    
    # Configuration file support
    # 配置文件支持
    config_file = "config_sram.yaml"
    print(f"Using configuration file: {config_file}")
    
    # Import optimization algorithms
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'size_optimization'))
        
        # Available optimization algorithms for algorithm name and description
        # 算法名称和描述的可用优化算法
        algorithms = {
            'PSO': 'Particle Swarm Optimization',
            'SA': 'Simulated Annealing',
            'CBO': 'Constrained Bayesian Optimization', 
            'RoSE_Opt': 'Reinforcement Learning Enhanced Bayesian Optimization',
            'SMAC': 'Sequential Model-based Algorithm Configuration'
        }
        
        print("Available optimization algorithms:")
        for i, (algo, desc) in enumerate(algorithms.items(), 1):
            print(f"  {i}. {algo} - {desc}")
        
        print("\nSelect optimization algorithm(s) to run:")
        print("  Enter number(s) separated by commas (e.g., 1,3,5)")
        print("  Enter 'all' to run all algorithms")
        print("  Enter 'comparison' to run automated comparison using run_experiments.py")  # ADDED
        print("  Enter 'none' to skip optimization")
        
        user_input = input("Your choice: ").strip().lower()
        
        if user_input == 'none':
            print("Skipping optimization algorithms.")
        elif user_input == 'comparison':  # Option to run automated comparison
            print("Running automated algorithm comparison...")
            try:
                import run_experiments
                run_experiments.main()
            except ImportError as e:
                print(f"Error importing run_experiments: {e}")
                print("Make sure run_experiments.py exists in the current directory")
            except Exception as e:
                print(f"Error running automated comparison: {e}")
        elif user_input == 'all':
            selected_algos = list(algorithms.keys())
        else:
            try:
                indices = [int(x.strip()) for x in user_input.split(',')]
                selected_algos = [list(algorithms.keys())[i-1] for i in indices if 1 <= i <= len(algorithms)]
            except (ValueError, IndexError):
                print("Invalid input. Running PSO algorithm as default.")
                selected_algos = ['PSO']
        
        # Run selected algorithmsupdated:
        if user_input != 'none' and user_input != 'comparison':
            for algo in selected_algos:
                print(f"\n{'='*20} Running {algo} Algorithm {'='*20}")
                
                try:
                    if algo == 'PSO':
                        from size_optimization import demo_pso
                        demo_pso.main(config_file)  # Pass config file
                        
                    elif algo == 'SA':
                        from size_optimization import demo_sa
                        demo_sa.main(config_file)  # Pass config file
                        
                    elif algo == 'CBO':
                        from size_optimization import demo_cbo
                        demo_cbo.main(config_file)  # Pass config file
                        
                    elif algo == 'RoSE_Opt':  
                        from size_optimization import demo_roseopt
                        demo_roseopt.main(config_file)  # Pass config file
                        
                    elif algo == 'SMAC':
                        from size_optimization import demo_smac 
                        demo_smac.main(config_file)  # Pass config file and correct function call
                    
                    print(f"{algo} optimization completed successfully!")
                    
                except ImportError as e:
                    print(f"Error importing {algo} algorithm: {e}")
                    print(f"Make sure the corresponding file exists in size_optimization/")
                    print(f"Expected files:")
                    expected_files = {
                        'PSO': 'size_optimization/demo_pso.py',
                        'SA': 'size_optimization/demo_sa.py', 
                        'CBO': 'size_optimization/demo_cbo.py',
                        'RoSE_Opt': 'size_optimization/demo_roseopt.py',
                        'SMAC': 'size_optimization/demo_smac.py'
                    }
                    if algo in expected_files:
                        print(f"  - {expected_files[algo]}")
                    
                except Exception as e:
                    print(f"Error running {algo} algorithm: {e}")
                    print("Continuing with next algorithm...")
                    import traceback
                    traceback.print_exc()  # More detailed error info
                    
                print(f"{'='*50}")
        
        print(f"\nAll selected optimization algorithms completed!")
        print("Results are saved in sim/opt/results/ directory")
        print("Plots are saved in sim/opt/plots/ directory")
        
        # Additional information
        print("\nFor comprehensive algorithm comparison, you can also run:")
        print("  python run_experiments.py")
                    
    except Exception as e:
        print(f"Error in optimization algorithms section: {e}")
        print("Continuing without optimization...")
        import traceback
        traceback.print_exc()  # More detailed error info
    
    print("\n" + "="*60)
    print("SRAM Analysis and Optimization Session Completed")
    print("="*60)
