from testbenches.sram_6t_core_testbench import Sram6TCoreTestbench
from testbenches.sram_6t_core_MC_testbench import Sram6TCoreMcTestbench
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
import numpy as np
from utils import estimate_bitcell_area

if __name__ == '__main__':
    vdd = 1.0
    pdk_path = 'model_lib/models.spice'
    nmos_model_name = 'NMOS_VTG'
    pmos_model_name = 'PMOS_VTG'
    pd_width = 0.205e-6
    pu_width = 0.09e-6
    pg_width = 0.135e-6
    length = 50e-9

    # FreePDK45 default transistor sizes
    area = estimate_bitcell_area(
        w_access=pg_width,
        w_pd=pd_width,
        w_pu=pu_width,
        l_transistor=length
    )
    print(f"Estimated 6T SRAM Cell Area: {area*1e12:.2f} µm²")

    num_rows = 64
    num_cols = 4
    num_mc = 1

    print("===== 6T SRAM Array Monte Carlo Simulation Debug Session =====")
    mc_testbench = Sram6TCoreMcTestbench(
        vdd,
        pdk_path, nmos_model_name, pmos_model_name,
        num_rows=num_rows, num_cols=num_cols, 
        pd_width=pd_width, pu_width=pu_width, 
        pg_width=pg_width, length=length,
        w_rc=True, # Whether add RC to nets
        pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
        vth_std=0.05, # Process parameter variation is a percentage of its value in model lib
        custom_mc=False, # Use your own process params?
        q_init_val=0, sim_path='sim',
    )
    # vars = np.random.rand(num_mc,num_rows*num_cols*18)

    # For using DC analysis, operation can be 'write_snm' 'hold_snm' 'read_snm'
    # read_snm = mc_testbench.run_mc_simulation(
    #     operation='write_snm', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, 
    #     vars=None, # Input your data table
    # )

    # For using TRAN analysis, operation can be 'write' or 'read'
    w_delay, w_pavg = mc_testbench.run_mc_simulation(
        operation='read', target_row=num_rows-1, target_col=num_cols-1, mc_runs=num_mc, 
        vars=None, # Input your data table
    )

    print("[DEBUG] Monte Carlo simulation completed")

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
        sys.path.append(os.path.join(os.path.dirname(__file__), 'sram_optimization'))
        
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
                        from sram_optimization import pso
                        pso.main(config_file)  # Pass config file
                        
                    elif algo == 'SA':
                        from sram_optimization import sa
                        sa.main(config_file)  # Pass config file
                        
                    elif algo == 'CBO':
                        from sram_optimization import sram_cbo
                        sram_cbo.main(config_file)  # Pass config file
                        
                    elif algo == 'RoSE_Opt':  
                        from sram_optimization import rose_opt
                        rose_opt.main(config_file)  # Pass config file
                        
                    elif algo == 'SMAC':
                        from sram_optimization import sram_smac 
                        sram_smac.main(config_file)  # Pass config file and correct function call
                    
                    print(f"{algo} optimization completed successfully!")
                    
                except ImportError as e:
                    print(f"Error importing {algo} algorithm: {e}")
                    print(f"Make sure the corresponding file exists in sram_optimization/")
                    print(f"Expected files:")
                    expected_files = {
                        'PSO': 'sram_optimization/pso.py',
                        'SA': 'sram_optimization/sa.py', 
                        'CBO': 'sram_optimization/sram_cbo.py',
                        'RoSE_Opt': 'sram_optimization/rose_opt.py',
                        'SMAC': 'sram_optimization/sram_smac.py'
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
