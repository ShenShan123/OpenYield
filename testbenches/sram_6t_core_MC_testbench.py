import os
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
# Only for yield analysis
from utils import parse_mc_measurements, generate_mc_statistics, save_mc_results, process_simulation_data
from testbenches.sram_6t_core_testbench import SRAM_6T_Array_Testbench
import numpy as np
import PySpice
from PySpice.Spice.Netlist import SubCircuitFactory

class SRAM_6T_Array_MC_Testbench(SRAM_6T_Array_Testbench):
    def __init__(self, pdk_path, nmos_model_name, pmos_model_name, 
                 pd_width, pu_width, pg_width, length,
                 num_rows, num_cols, w_rc, pi_res, pi_cap,
                 vth_std=0.05, custom_mc=False,
                 sim_path='sim'):
        
        super().__init__(
            pdk_path, nmos_model_name, pmos_model_name, 
            pd_width, pu_width, pg_width, length,
            num_rows, num_cols, w_rc, pi_res, pi_cap, 
            custom_mc,
        )

        self.vth_std = vth_std

        # Create temporary model file
        self.sim_path = sim_path
        os.makedirs(self.sim_path, exist_ok=True)

        # Standard MC needs a model lib with variables, 
        # otherwise, process parameters are defined by user
        if not custom_mc:
            self.temp_model_file = self.create_mc_model_file()

    def create_mc_model_file(self):
        """Create temporary model file with Monte Carlo variations"""
        with open(self.pdk_path, 'r') as f:
            model_content = f.read()

        # Add DEV variation to VTH0 parameters for NMOS and PMOS
        modified_content = model_content.replace(# for NMOS
            'vth0 = 0.4106 ',
            f'vth0 = {{AGAUSS(0.4106, 0.4106*{self.vth_std}, 1)}}'
        ).replace(# for NMOS
            'u0 = 0.045 ',
            f'u0 = {{AGAUSS(0.045, 0.045*{self.vth_std}, 1)}}'
        ).replace(# for NMOS
            'voff    = -0.13 ',
            f'voff = {{AGAUSS(-0.13, 0.13*{self.vth_std}, 1)}}'
        ).replace(# for PMOS
            'vth0    = -0.3842 ',
            f'vth0 = {{AGAUSS(-0.3842, 0.3842*{self.vth_std}, 1)}}'
        ).replace(# for PMOS
            'u0      = 0.02 ',
            f'u0 = {{AGAUSS(0.02, 0.02*{self.vth_std}, 1)}}'
        ).replace(# for PMOS
            'voff    = -0.126 ',
            f'voff = {{AGAUSS(-0.126, 0.126*{self.vth_std}, 1)}}'
        )

        # Generate a modified model lib
        temp_model_path = os.path.join(self.sim_path, 'tmp_mc.spice')

        with open(temp_model_path, 'w') as f:
            f.write(modified_content)

        return temp_model_path

    def create_testbench(self, operation, target_row=0, target_col=0):
        """Create testbench with Monte Carlo models"""
        circuit = super().create_testbench(operation, target_row, target_col)

        if not self.custom_mc:
            # Replace original included model lib with new path
            circuit._includes[0] = self.temp_model_file

        return circuit
    
    def add_meas_and_print(self, simulator, init_cond, operation):
        # Internal nodes' names of the target cell
        target_node_q = self.inst_prefix + f'_{self.target_row}_{self.target_col}{self.heir_delimiter}Q'
        target_node_qb = self.inst_prefix + f'_{self.target_row}_{self.target_col}{self.heir_delimiter}QB'

        if operation == 'hold_snm' or operation == 'read_snm' or operation == 'write_snm':
            # Initial V(BL) and V(BLB) for the  cell
            init_cond = {}
            init_cond[f'BL']  = self.vdd @ u_V
            init_cond[f'BLB'] = self.vdd @ u_V
            simulator.initial_condition(**init_cond)
            simulator.measure('DC', 'MAXVD', 'MAX V(VD)')
            simulator.measure('DC', operation.upper(), f"PARAM='1/sqrt(2)*MAXVD'")
            # Add print for SNM 
            simulator.circuit.raw_spice += \
                f'.PRINT DC FORMAT=NOINDEX {{U}} V(V1) V(V2)\n'

        # The read operation
        elif operation == 'read':
            for col in range(self.num_cols):
                # Initial V(BL) and V(BLB) for all columns
                init_cond[f'BL{col}'] = 0 @ u_V
                init_cond[f'BLB{col}'] = 0 @ u_V
            
            simulator.initial_condition(**init_cond)

            # Add measurements for read delay, 
            # which is defined as the time from the WL rise to BL swing to VDD/2
            simulator.measure(
                'TRAN', 'TWL', 
                f'WHEN V(WL{self.target_row})={self.half_vdd} RISE=1 ') # modified for Xyce
            simulator.measure(
                'TRAN', 'TBL', 
                f"WHEN V(BL{self.target_col})='V(BLB{self.target_col})-{self.half_vdd}' FALL=1")
            simulator.measure('TRAN', 'TREAD', f"PARAM='TBL-TWL'")

            # Add print for read delay
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(WL{self.target_row}) V(BL{self.target_col})'+ \
                f' V(BLB{self.target_col}) V({target_node_q}) V({target_node_qb})\n'
        
        # The write operation
        elif operation == 'write':
            # Initial all cell to '0'
            simulator.initial_condition(**init_cond)

            # Add measurements for write delay, 
            # which is defined as the time from the WL rise to data Q rise to 90% VDD.
            simulator.measure(
                'TRAN', 'TWRITE_Q', 
                f'TRIG V(WL{self.target_row})={self.half_vdd} RISE=1', 
                f"TARG V({target_node_q})={float(self.vdd)*0.9:.2f} RISE=1")
            simulator.measure(
                'TRAN', 'TWRITE_QB', 
                f'TRIG V(WL{self.target_row})={self.half_vdd} RISE=1', 
                f"TARG V({target_node_qb})={float(self.vdd)*0.9:.2f} FALL=1")
            
            # Add print for write delay
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(WL{self.target_row}) V({target_node_q}) V({target_node_qb})\n'
        else:
            raise ValueError(f"Invalid operation: {operation}")
        
    def add_xyce_options(self, circuit, mc_runs, operation):
        """ Add options for Xyce """
        pass

    def add_analysis(self, circuit, operation, num_mc):
        """ Add .DC / .TRAN analysis """
        if 'snm' in operation:
            u_tmp = self.vdd / np.sqrt(2)
            circuit.raw_spice += \
                f'.DC U -{u_tmp:.2f} {u_tmp:.2f} 0.001\n'
        else:
            circuit.raw_spice += \
                f'.TRAN {self.t_step:.2e} {float(self.t_period):.2e}\n'
            # Timing interval option is set only in .TRAN analysis.
            circuit.raw_spice += \
                f'.OPTIONS OUTPUT INITIAL_INTERVAL={self.t_step:.2e}\n'
            
        # Whether we use custom MC
        if self.custom_mc:
            # Sweep the each row of the `table`
            circuit.raw_spice += \
                f'.STEP data=table\n'
        else:
            # Use build-in sampling method in Xyce
            circuit.raw_spice += \
                f'.SAMPLING useExpr=true\n.options samples numsamples={num_mc}\n'
        
        print(f"[DEBUG] Custom_MC={self.custom_mc}, numsamples={num_mc}")
                
    def get_table_head(self):
        return self.table_head
    
    def gen_process_params(self, circuit: SubCircuitFactory, 
                           operation: str, num_mc: int, 
                           vars: np.array = None):
        """ Add process parameters' data table for STEP 

        Args:
        ---
            circuit (SubCircuitFactory): simulator's circuits
            operation (string): can be `read`, `write`, `hold_snm`, `read_snm`, `write_snm`
            num_mc (int): number of MC runs
            vars (numpy.ndarray): parameters in data table
        """
        # Order of transistors in bitcell can not be changed
        mos_names = ['PGL', 'PGR', 'PDL', 'PUL', 'PDR', 'PUR']
        # This version only takes 3 params into consideration
        param_names = ['vth0', 'u0', 'voff']
        self.table_head = '.data table\n+ '
        table_content = '\n'
        num_params = 0

        # Define params
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # Only 1 cell's params are generated
                if ('snm' in operation) and (row > 0 or col > 0):
                        continue
                # Each transistor has 6 process variables
                for mos in mos_names:
                    for param in param_names:
                        if mos in ['PUL', 'PUR']:
                            # Parameter definitions for PMOS
                            circuit.raw_spice += \
                                f'.param {param}_{self.pmos_model_name}_{mos}_{row:d}_{col:d}=0.0\n'
                            # Data table head
                            self.table_head += f'{param}_{self.pmos_model_name}_{mos}_{row:d}_{col:d} '
                        else:
                            # Parameter definitions for NMOS
                            circuit.raw_spice += \
                                f'.param {param}_{self.nmos_model_name}_{mos}_{row:d}_{col:d}=0.0\n'
                            # Data table head
                            self.table_head += f'{param}_{self.nmos_model_name}_{mos}_{row:d}_{col:d} '

                        num_params += 1
        # Just for debugging
        if vars is None:
            vars = [0.4106, 0.045, -0.13, # PGL
                    0.4106, 0.045, -0.13, # PGR
                    0.4106, 0.045, -0.13, # PDL
                    -0.3842, 0.02, -0.126,# PUL
                    0.4106, 0.045, -0.13, # PDR
                    -0.3842, 0.02, -0.126]# PUR
            
            if operation == 'read' or operation == 'write':
                vars = np.array([vars*self.num_rows*self.num_cols])
                vars = np.repeat(vars, num_mc, axis=0)
            else:
                vars = np.array([vars])
                vars = np.repeat(vars, num_mc, axis=0)

            print(f"[DEBUG] Generated vars.shape={vars.shape}")
        else:
            assert num_mc == vars.shape[0], f"num_mc={num_mc} mismatches {vars.shape[0]} row number in the data table"
            print(f"[DEBUG] Input vars.shape={vars.shape}")

        assert len(vars.shape) == 2
        assert num_params == vars.shape[1], f'num_params={num_params} mismatches {vars.shape[1]} column number in the data table'

        table_content += "\n".join([
            "+ " + " ".join([f"{x:.4f}" for x in row]) 
            for row in vars
        ])
        
        # Generate and run Xyce netlist
        table_path = os.path.join(self.sim_path, f'mc_{operation}_table.data')
        with open(table_path, 'w') as f:
            f.write(self.table_head+table_content)
        
        circuit.include(table_path)
        print(f'[DEBUG] Data table has been saved to {table_path}')
    
    def run_mc_simulation(self, operation='read', target_row=0, target_col=0, mc_runs=100, vars=None):
        """Run Xyce Monte Carlo simulation"""
        simulator = self.create_testbench(operation, target_row, target_col).simulator()

        # Add some Xyce related commands
        self.add_analysis(simulator.circuit, operation, mc_runs)

        # Add measurements according to the operation
        self.add_meas_and_print(simulator, {}, operation)
        
        # Add process parameters
        if self.custom_mc:
            self.gen_process_params(simulator.circuit, operation, vars=vars, num_mc=mc_runs)

        print("[DEBUG] Printing generated netlists...")
        print(simulator)

        # Generate and run Xyce netlist
        tb_path = os.path.join(self.sim_path, f'mc_{operation}_testbench.sp')
        
        with open(tb_path, 'w') as f:
            f.write(str(simulator))
        # assert 0
        # Execute Xyce and parse results
        try:
            import subprocess
            # command: Xyce <netlist>
            print("[DEBUG] Xyce running ...")
            result = subprocess.run(
                # ['hspice', '-i', tb_path, '-o', self.sim_path],
                ['Xyce', tb_path, '-o', tb_path],
                capture_output=True,
                text=True, check=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Xyce error:\n{result.stderr}, please check the log file {tb_path.replace('.sp', '.lis')}.")
            else:
                print("[DEBUG] Simulation run successfully.")
        finally:
            # plot waveforms of signals in `.PRINT`
            process_simulation_data(
                prn_path=tb_path+'.prn',
                num_mc=mc_runs,
                output=f"{self.sim_path}/mc_{operation}_waveform.png",
            )
            
            # Get all `.mtX` or `.msX` files from MC
            mc_df = parse_mc_measurements(
                netlist_prefix=tb_path,
                file_suffix='ms' if 'snm' in operation else 'mt',
                num_runs=mc_runs,
                # value_threshold=1e-9
            )
            print("[DEBUG] Printing mc_df")
            print(mc_df)
            # assert 0
            # Generate statistics
            stats = generate_mc_statistics(mc_df)
            # Save results
            save_mc_results(
                mc_df, stats, 
                data_file=tb_path.replace('.sp', '.data.csv'), 
                stats_file=tb_path.replace('.sp', '.stats.csv')
            )
            if operation == 'write': return mc_df['TWRITE_Q'].to_numpy()
            elif operation == 'read': return mc_df['TREAD'].to_numpy()
            elif operation == 'hold_snm': return mc_df['HOLD_SNM'].to_numpy()
            elif operation == 'read_snm': return mc_df['READ_SNM'].to_numpy()
            elif operation == 'write_snm': return mc_df['WRITE_SNM'].to_numpy()
            else: raise KeyError(f"Unkonwn operation {operation}")