import os
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF
from sram_6t_core import SRAM_6T_Array  # Assuming SRAM_6T_Cell_RC is defined
from utils import plot_results, measure_delay
import sys
from utils import parse_mt0, analyze_mt0

class Base_Testbench:
    def __init__(self, tb_name, pdk_path, nmos_model_name, pmos_model_name):
        self.name = tb_name
        self.pdk_path = pdk_path

        if os.path.exists(pdk_path):
            print(f"[DEBUG] Transistor model file found: {pdk_path}")
        else:
            raise FileNotFoundError(f"Transistor model file not found: {pdk_path}")
        
        self.nmos_model_name = nmos_model_name
        self.pmos_model_name = pmos_model_name
        self.vdd = 1.0
        self.half_vdd = float(self.vdd) / 2
        ## need to be changed in create_testbench
        self.data_node_prefix = 'X'

        # Define timing parameters for pulse sources
        self.t_rise = 0.1 @ u_ns  # Rise time
        self.t_fall = 0.1 @ u_ns  # Fall time
        self.t_pulse_width = 2 @ u_ns  # Pulse width
        self.t_period = 5 @ u_ns  # Period
        self.t_delay = 2 @ u_ns

    def create_read_periphery(self, circuit):
        """Create read periphery circuitry"""
        raise NotImplementedError("create_read_periphery method not implemented")  
    
    def create_write_periphery(self, circuit):
        """Create write periphery circuitry"""      
        raise NotImplementedError("create_write_periphery method not implemented")

    def create_testbench(self):
        """
        Override this method to create a testbench for the circuit array.
        Create a testbench for the circuit array.
        """
        circuit = Circuit(self.tb_name)
        circuit.include(self.pdk_path)
        
        # Power supply
        circuit.V('VDD', 'VDD', circuit.gnd, self.vdd)
        
        return circuit

    def run_simulation(self):
        """
        Override this method to run a specific test.
        Run specified test and return results
        """
        circuit = self.create_testbench()
        simulator = circuit.simulator()

        # Initialize all internal data nodes (Q and QB) in all cells to 0V
        initial_conditions = {}
        simulator.initial_condition(**initial_conditions)

        # Run transient simulation
        analysis = simulator.transient(step_time=0.01 @ u_ns, end_time=self.t_period)
        
        return {
            'success': True,
            'analysis': analysis,
        }

class SRAM_6T_Array_Testbench(Base_Testbench):
    def __init__(self, pdk_path, nmos_model_name, pmos_model_name, num_rows, num_cols, w_rc=False):
        super().__init__(
            f'SRAM_6T_Array_{num_rows}x{num_cols}_Testbench', 
            pdk_path, nmos_model_name, pmos_model_name)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.inst_prefix = 'X'
        self.w_rc = w_rc

    def create_wl_driver(self, circuit, target_row):
        """Create wordline driver for the target/standby row"""
        # Wordline control & drivers
        for row in range(self.num_rows):
            if row == target_row:
                # Add pulse source for the target row
                circuit.PulseVoltageSource(
                    f'WL{row}_pulse', f'WL{row}', circuit.gnd,
                    initial_value=0 @ u_V, pulsed_value=self.vdd, 
                    delay_time=self.t_delay + 2*self.t_rise,
                    rise_time=self.t_rise, fall_time=self.t_fall, 
                    pulse_width=self.t_pulse_width,
                    period=self.t_period
                )
            else:
                # Tie idle wordlines to ground
                circuit.V(f'WL{row}_gnd', f'WL{row}', circuit.gnd, 0 @ u_V)
        return circuit

    def create_read_periphery(self, circuit):
        """Create read periphery circuitry"""
        # Add precharge circuitry for all columns
        for col in range(self.num_cols):
            circuit.M(f'MP1_{col}', f'BL{col}', 'PRE', 'VDD', 'VDD', model=self.pmos_model_name, w=0.2e-6, l=45e-9)
            circuit.M(f'MP2_{col}', f'BLB{col}', 'PRE', 'VDD', 'VDD', model=self.pmos_model_name, w=0.2e-6, l=45e-9)
            circuit.M(f'MP3_{col}', f'BL{col}', 'PRE', f'BLB{col}', 'VDD', model=self.pmos_model_name, w=0.2e-6, l=45e-9)
        
        # Precharge pulse source to precharge all BL/BLB to VDD
        circuit.PulseVoltageSource(
            'PRE', 'PRE', circuit.gnd, 
            initial_value=self.vdd, pulsed_value=0 @ u_V, 
            delay_time=0 @ u_ns, 
            rise_time=self.t_rise, 
            fall_time=self.t_fall, 
            pulse_width=self.t_pulse_width, 
            period=self.t_period, dc_offset=self.vdd
        )
        return circuit
    
    def create_write_periphery(self, circuit):
        """Create write periphery circuitry"""
        # Write drivers for all columns
        for col in range(self.num_cols):
            circuit.PulseVoltageSource(
                f'BL{col}_pulse', f'BL{col}', circuit.gnd, 
                initial_value=0 @ u_V, pulsed_value=self.vdd, 
                delay_time=self.t_delay, 
                rise_time=self.t_rise, fall_time=self.t_fall, 
                # data hold time, assuming 2X rise times
                pulse_width=self.t_pulse_width + 4*self.t_rise, 
                period=self.t_period)
            
            circuit.PulseVoltageSource(
                f'BLB{col}_pulse', f'BLB{col}', circuit.gnd,
                initial_value=self.vdd, pulsed_value=0 @ u_V, 
                delay_time=self.t_delay, 
                rise_time=self.t_rise, fall_time=self.t_fall, 
                # data hold time, assuming 2X rise times
                pulse_width=self.t_pulse_width + 4*self.t_rise, 
                period=self.t_period, dc_offset=self.vdd)
            
        return circuit

    def create_testbench(self, operation='read', target_row=0, target_col=0):
        """
        Create a testbench for the SRAM array.
        operation: 'read' or 'write'
        target_row: Row index of the target cell
        target_col: Column index of the target cell
        """
        circuit = Circuit(self.name)
        circuit.include(self.pdk_path)
        
        # Power supply
        circuit.V('VDD', 'VDD', circuit.gnd, self.vdd)
        
        sbckt_6t_array = SRAM_6T_Array(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            nmos_model_name=self.nmos_model_name,
            pmos_model_name=self.pmos_model_name,
            w_rc=self.w_rc,
        )
        # Add subcircuit definition to this testbench.
        circuit.subcircuit(sbckt_6t_array)

        # Instantiate the SRAM array.
        circuit.X(sbckt_6t_array.name, sbckt_6t_array.name, 'VDD', circuit.gnd,
                  *[f'BL{i}' for i in range(self.num_cols)],
                  *[f'BLB{i}' for i in range(self.num_cols)],
                  *[f'WL{i}' for i in range(self.num_rows)])
        
        # internal node prefix in the SRAM cell
        self.inst_prefix = 'X' + sbckt_6t_array.name + '.' + sbckt_6t_array.inst_prefix

        self.target_row = target_row
        self.target_col = target_col

        if operation == 'read':
            self.create_read_periphery(circuit)
            
        elif operation == 'write':
            self.create_write_periphery(circuit)

        else:
            raise ValueError(f"Invalid test type {operation}. Use 'read' or 'write'")
        
        self.create_wl_driver(circuit, target_row)
        
        return circuit
    
    def set_initial_conditions(self):
        """Initialize all internal data nodes (Q and QB) in all cells to 0V"""
        initial_conditions = {}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                node_q = self.inst_prefix + f'_{row}_{col}.Q'
                node_qb = self.inst_prefix + f'_{row}_{col}.QB'
                initial_conditions[node_q] = 0 @ u_V
                initial_conditions[node_qb] = self.vdd @ u_V
        return initial_conditions

    def run_simulation(self, operation='read', target_row=0, target_col=0):
        """Run specified test and return results"""
        circuit = self.create_testbench(operation, target_row, target_col)
        simulator = circuit.simulator()

        init_cond = self.set_initial_conditions()

        # Initial voltage conditions (Q=0) for the target cell
        target_node_q = self.inst_prefix + f'_{target_row}_{target_col}.Q'
        target_node_qb = self.inst_prefix + f'_{target_row}_{target_col}.QB'

        if operation == 'read':
            # Initial V(BL) and V(BLB) for the target column
            init_cond[f'BL{target_col}'] = 0 @ u_V
            init_cond[f'BLB{target_col}'] = 0 @ u_V
            simulator.initial_condition(**init_cond)

            print("[DEBUG] Printing generated netlists...")
            print(simulator)

            # Run transient simulation
            analysis = simulator.transient(step_time=0.01 @ u_ns, end_time=self.t_period)

            # Calculate read delay
            delay = measure_delay(
                analysis.time,
                [analysis[f'WL{target_row}'], analysis[f'BL{target_col}']],
                trig_val=self.half_vdd,
                targ_val=self.half_vdd,
                targ_edge_type='fall'
            )

            # Plot waveforms
            plot_results(
                analysis,
                [f'WL{target_row}', 'PRE', f'BL{target_col}', target_node_q],
                fig_name=f'plots/sram_6t_array_{operation}_row{target_row}_col{target_col}.png'
            )


        elif operation == 'write':
            simulator.initial_condition(**init_cond)

            print("[DEBUG] Printing generated netlists...")
            print(simulator)

            # Run transient simulation
            analysis = simulator.transient(step_time=0.01 @ u_ns, end_time=self.t_period)

            # Calculate write delay
            delay = measure_delay(
                analysis.time,
                [analysis[f'WL{target_row}'], analysis[target_node_q]],
                trig_val=self.half_vdd,
                targ_val=0.9 * self.vdd,
                targ_edge_type='rise'
            )

            # Plot waveforms
            plot_results(
                analysis,
                [f'WL{target_row}', f'BL{target_col}', target_node_q, target_node_qb],
                fig_name=f'plots/sram_6t_array_{operation}_row{target_row}_col{target_col}.png'
            )

        else:
            raise ValueError(f"Invalid test type {operation}. Use 'read' or 'write'")
        
        print(f"[DEBUG] {operation} delay for row {target_row}, col {target_col} = {delay:.4e}")

        return {
            'success': True,
            'delay': delay,
            'analysis': analysis,
        }
    

class SRAM_6T_Array_MC_Testbench(SRAM_6T_Array_Testbench):
    def __init__(self, pdk_path, nmos_model_name, pmos_model_name, 
                 num_rows, num_cols, w_rc=False, 
                 vth_std=0.05):
        super().__init__(pdk_path, nmos_model_name, pmos_model_name, 
                         num_rows, num_cols, w_rc)
        self.vth_std = vth_std
        self.temp_model_file = self.create_mc_model_file()

    def create_mc_model_file(self):
        """Create temporary model file with Monte Carlo variations"""
        with open(self.pdk_path, 'r') as f:
            model_content = f.read()

        # Add DEV variation to VTH0 parameters for NMOS and PMOS
        modified_content = model_content.replace(
            'vth0 = 0.322',
            f'vth0 = 0.322 DEV {self.vth_std}'
        ).replace(
            'vth0    = -0.3021',
            f'vth0    = -0.3021 DEV {self.vth_std}'
        )

        # Create temporary model file
        self.sim_path = 'sim'
        os.makedirs(self.sim_path, exist_ok=True)
        temp_model_path = os.path.join(self.sim_path, 'tmp_mc.spice')

        with open(temp_model_path, 'w') as f:
            f.write(modified_content)

        return temp_model_path

    # TODO: Tomorrow I will continue from here
    def create_testbench(self, operation='read', target_row=0, target_col=0):
        """Create testbench with Monte Carlo models"""
        circuit = super().create_testbench(operation, target_row, target_col)

        # Replace original included model lib with new path
        circuit._includes[0] = self.temp_model_file
        return circuit
    
    def run_mc_simulation(self, operation='read', target_row=0, target_col=0, mc_runs=100):
        """Run HSPICE Monte Carlo simulation"""
        circuit = self.create_testbench(operation, target_row, target_col)
        simulator = circuit.simulator()

        init_cond = super().set_initial_conditions()

        # Initial voltage conditions (Q=0) for the target cell
        target_node_q = self.inst_prefix + f'_{target_row}_{target_col}.Q'
        target_node_qb = self.inst_prefix + f'_{target_row}_{target_col}.QB'

        # The read operation
        if operation == 'read':
            # Initial V(BL) and V(BLB) for the target column
            init_cond[f'BL{target_col}'] = 0 @ u_V
            init_cond[f'BLB{target_col}'] = 0 @ u_V
            simulator.initial_condition(**init_cond)

            # Add measurements for read delay, 
            # which is defined as the time from the WL rise to BL swing to VDD/2
            simulator.measure(
                'TRAN', 'TWL', 
                f'WHEN V(WL{self.target_row}) VAL={self.half_vdd} RISE=1 ')
            simulator.measure(
                'TRAN', 'TBL', 
                f"WHEN V(BL{target_col})='V(BLB{target_col})-{self.half_vdd}' FALL=1")
            simulator.measure('TRAN', 'TREAD', f"PARAM='TBL-TWL'")
        
        # The write operation
        elif operation == 'write':
            # Initial all cell to '0'
            simulator.initial_condition(**init_cond)
            # Add measurements for write delay, 
            # which is defined as the time from the WL rise to data Q rise to 90% VDD.
            simulator.measure(
                'TRAN', 'TWRITE_Q', 
                f'TRIG V(WL{self.target_row}) VAL={self.half_vdd} RISE=1', 
                f"TARG V({target_node_q}) VAL={float(self.vdd)*0.9:.2f} RISE=1")
            simulator.measure(
                'TRAN', 'TWRITE_QB', 
                f'TRIG V(WL{self.target_row}) VAL={self.half_vdd} RISE=1', 
                f"TARG V({target_node_qb}) VAL={float(self.vdd)*0.9:.2f} FALL=1")
        else:
            raise ValueError(f"Invalid operation: {operation}")
        
        netlist = str(simulator).replace(
            '.end\n', 
            f'.TRAN {float(self.t_rise)*0.1:.2e} ' + 
            f'{float(self.t_period):.2e} SWEEP MONTE={mc_runs:d}\n' +
            '.END\n')
        
        # Generate and run HSPICE netlist
        tb_path = self.sim_path + '/mc_netlist.sp'
        with open(tb_path, 'w') as f:
            f.write(str(netlist))
        
        # Execute HSPICE and parse results
        try:
            import subprocess
            # command: hspice -i <netlist> -o <output>
            result = subprocess.run(
                ['hspice', '-i', tb_path, '-o', self.sim_path],
                capture_output=True,
                text=True, check=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"HSPICE error:\n{result.stderr}, please check the log file {tb_path.replace('.sp', '.lis')}.")
            
            return self._parse_results(operation)
        finally:
            data = analyze_mt0(self.sim_path + '/mc_netlist.mt0')
            print(f"[DEBUG] Parsed {len(data)} Monte Carlo samples")
            return
        
    
if __name__ == '__main__':
    print("===== 6T SRAM Array Simulation Debug Session =====")
    pdk_path = 'model_lib/models.spice'
    nmos_model_name = 'NMOS_VTL'
    pmos_model_name = 'PMOS_VTL'

    # testbench = SRAM_6T_Array_Testbench(pdk_path, nmos_model_name, pmos_model_name, 2, 2, w_rc=True)
    
    # testbench.run_simulation(operation='read', target_row=1, target_col=1)
    # print("[DEBUG] Simulation of read operation completed")
    
    # testbench.run_simulation(operation='write', target_row=1, target_col=1)
    # print("[DEBUG] Simulation of write operation completed")

    print("===== 6T SRAM Array Monte Carlo Simulation Debug Session =====")
    mc_testbench = SRAM_6T_Array_MC_Testbench(pdk_path, nmos_model_name, pmos_model_name, 2, 2, w_rc=True)

    mc_testbench.run_mc_simulation(operation='read', target_row=1, target_col=1, mc_runs=100)
    print("[DEBUG] Monte Carlo simulation of read operation completed")
    
    mc_testbench.run_mc_simulation(operation='write', target_row=1, target_col=1, mc_runs=100)
    print("[DEBUG] Monte Carlo simulation of write operation completed")