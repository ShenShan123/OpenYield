import os
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF
from sram_6t_core import SRAM_6T_Array, SRAM_6T_Cell  # Assuming SRAM_6T_Cell_RC is defined
from utils import plot_results, measure_delay, calculate_snm, plot_butterfly_with_squares
from utils import parse_mt0, analyze_mt0
from base import Base_Testbench
import numpy as np
from copy import deepcopy

class SRAM_6T_Array_Testbench(Base_Testbench):
    def __init__(self, pdk_path, nmos_model_name, pmos_model_name, 
                 pd_width, pu_width, pg_width, length,
                 num_rows, num_cols, w_rc, pi_res, pi_cap):
        super().__init__(
            f'SRAM_6T_Array_{num_rows}x{num_cols}_Testbench', 
            pdk_path, nmos_model_name, pmos_model_name)
        # transistor size info.
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.length = length
        # array size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.inst_prefix = 'X'
        # add rc?
        self.w_rc = w_rc
        self.pi_res = pi_res
        self.pi_cap = pi_cap

    def create_wl_driver(self, circuit: Circuit, target_row: int):
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
                    pulse_width=self.t_pulse,
                    period=self.t_period
                )
            else:
                # Tie idle wordlines to ground
                circuit.V(f'WL{row}_gnd', f'WL{row}', circuit.gnd, 0 @ u_V)
        return circuit

    def create_read_periphery(self, circuit: Circuit):
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
            pulse_width=self.t_pulse, 
            period=self.t_period, dc_offset=self.vdd
        )
        return circuit
    
    def create_write_periphery(self, circuit: Circuit):
        """Create write periphery circuitry"""
        # Write drivers for all columns
        for col in range(self.num_cols):
            circuit.PulseVoltageSource(
                f'BL{col}_pulse', f'BL{col}', circuit.gnd, 
                initial_value=0 @ u_V, pulsed_value=self.vdd, 
                delay_time=self.t_delay, 
                rise_time=self.t_rise, fall_time=self.t_fall, 
                # data hold time, assuming 2X rise times
                pulse_width=self.t_pulse + 4*self.t_rise, 
                period=self.t_period)
            
            circuit.PulseVoltageSource(
                f'BLB{col}_pulse', f'BLB{col}', circuit.gnd,
                initial_value=self.vdd, pulsed_value=0 @ u_V, 
                delay_time=self.t_delay, 
                rise_time=self.t_rise, fall_time=self.t_fall, 
                # data hold time, assuming 2X rise times
                pulse_width=self.t_pulse + 4*self.t_rise, 
                period=self.t_period, dc_offset=self.vdd)
            
        return circuit

    def create_single_cell_for_snm(self, circuit: Circuit, operation: str):
        """
        Create a single 6T SRAM cell for SNM measurement.
        How to calculate SNM for 6T SRAM cell in SPICE?
        See: https://www.edaboard.com/threads/sram-snm-simulation-hspice.253224/
        """
        # Add U parameter
        # .param U=0
        circuit.parameter('U', 0)

        # Instantiate 6T SRAM cell
        sbckt_6t_cell = SRAM_6T_Cell(
            self.nmos_model_name, self.pmos_model_name,
            self.pd_width, self.pu_width, self.pg_width, self.length,
            w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap, 
            disconncet=True, #NOTE: Key argument to disconnect the internal data nodes!!
        )
        # Add subcircuit definition to this testbench.
        circuit.subcircuit(sbckt_6t_cell)
        circuit.X(sbckt_6t_cell.name, sbckt_6t_cell.name, 'VDD', circuit.gnd, 
                'BL', 'BLB', 'WL')
        # internal node prefix in the SRAM cell
        self.inst_prefix = 'X' + sbckt_6t_cell.name

        if operation == 'hold_snm':
            # For hold_snm measurement, keep WL low and add DC sources to Q/QB
            circuit.V(f'WL_gnd', 'WL', circuit.gnd, 0 @ u_V)

        elif operation == 'read_snm':
            # For read_snm operation, keep WL high and add DC sources to Q/QB
            circuit.V(f'WL_vdd', 'WL', circuit.gnd, self.vdd)
            circuit.V(f'BL_vdd', 'BL', circuit.gnd, self.vdd)
            circuit.V(f'BLB_vdd', 'BLB', circuit.gnd, self.vdd)
        elif operation == 'write_snm':
            # For write_snm operation, keep WL high and add DC sources to Q/QB
            circuit.V(f'WL_vdd', 'WL', circuit.gnd, self.vdd@ u_V)
            circuit.V(f'BL_vdd', 'BL', circuit.gnd, self.vdd @ u_V)
            circuit.V(f'BLB_vdd', 'BLB', circuit.gnd, 0 @ u_V)
        else:
            raise ValueError(f"Invalid operation: {operation}")

        # Add voltage control voltage source for get SNM,
        # The grammar is insane, but it works, fuckin' PySpice,
        # e.g., EV1 V1 0 VOL='U+sqrt(2)*V(XSRAM_6T_CELL.QBD)
        circuit.VCVS(
            'V1', 'V1', '', circuit.gnd, '', 
            **{'raw_spice': f"VOL='U+sqrt(2)*V({self.inst_prefix}.QBD)'"}
        )
        circuit.VCVS(
            'V2', 'V2', '', circuit.gnd, '', 
            **{'raw_spice': f"VOL='-U+sqrt(2)*V({self.inst_prefix}.QD)'"}
        )
        circuit.VCVS(
            'Q', f'{self.inst_prefix}.Q', '', circuit.gnd, '', 
            **{'raw_spice': f" VOL='1/sqrt(2)*U+1/sqrt(2)*V(V1)'"}
        )
        circuit.VCVS(
            'QB', f'{self.inst_prefix}.QB', '', circuit.gnd, '', 
            **{'raw_spice': f" VOL='-1/sqrt(2)*U+1/sqrt(2)*V(V2)'"}
        )
        circuit.VCVS(
            'VD', 'VD', '', circuit.gnd, '', 
            **{'raw_spice': f"VOL='ABS(V(V1)-V(V2))'"}
        )
        return circuit

    def create_testbench(self, operation, target_row, target_col):
        """
        Create a testbench for the SRAM array.
        operation: 'read' or 'write'
        target_row: Row index of the target cell
        target_col: Column index of the target cell
        """
        self.target_row = target_row
        self.target_col = target_col

        circuit = Circuit(self.name)
        circuit.include(self.pdk_path)
        
        # Power supply
        circuit.V('VDD', 'VDD', circuit.gnd, self.vdd)

        # if it is a hold operation for hold SNM
        if 'snm' in operation:
            self.create_single_cell_for_snm(circuit, operation)
            # finish the circuit just return
            return circuit

        # Instantiate 6T SRAM array
        sbckt_6t_array = SRAM_6T_Array(
            self.num_rows, self.num_cols, self.nmos_model_name, self.pmos_model_name,
            self.pd_width, self.pu_width, self.pg_width, self.length,
            w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
        )
        # Add subcircuit definition to this testbench.
        circuit.subcircuit(sbckt_6t_array)
        self.array_subckt_name = sbckt_6t_array.name

        # Instantiate the SRAM array.
        circuit.X(sbckt_6t_array.name, sbckt_6t_array.name, 'VDD', circuit.gnd,
                  *[f'BL{i}' for i in range(self.num_cols)],
                  *[f'BLB{i}' for i in range(self.num_cols)],
                  *[f'WL{i}' for i in range(self.num_rows)])

        # internal node prefix in the SRAM cell
        self.inst_prefix = 'X' + sbckt_6t_array.name + '.' + sbckt_6t_array.inst_prefix

        # For read transient simulation, add pulse source to the array
        if operation == 'read':
            self.create_read_periphery(circuit)
            self.create_wl_driver(circuit, target_row)
        # For write transient simulation, add pulse source to the array
        elif operation == 'write':
            self.create_write_periphery(circuit)
            self.create_wl_driver(circuit, target_row)

        else:
            raise ValueError(f"Invalid test type {operation}. Use 'read' or 'write'")
        
        return circuit
    
    def set_initial_conditions(self):
        """Initialize all internal data nodes (Q=0V and QB=VDD) in all cells"""
        initial_conditions = {}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                node_q = self.inst_prefix + f'_{row}_{col}.Q'
                node_qb = self.inst_prefix + f'_{row}_{col}.QB'
                initial_conditions[node_q] = 0 @ u_V
                initial_conditions[node_qb] = self.vdd @ u_V
        return initial_conditions

    def dc_sweep(self, circuit, sweep_node, sense_node, sweep_steps=1000):
        """Perform DC sweep and return characteristic curve"""
        circuit_tmp = circuit.clone()
        for elem in circuit_tmp.elements:
            if elem.name[0] == 'E':
                elem.PREFIX = '*E'

        # Add DC source to sweep the target node
        Vsrc = circuit_tmp.V(sweep_node+'_src', sweep_node, circuit.gnd, 0 @ u_V)

        simulator = circuit_tmp.simulator()

        # Init BL and BLB voltages
        init_cond = {}
        init_cond[f'BL']  = self.vdd @ u_V
        init_cond[f'BLB'] = self.vdd @ u_V
        simulator.initial_condition(**init_cond)

        # Run dc analysis
        analysis = simulator.dc(**{Vsrc.name : slice(0, self.vdd, self.vdd/sweep_steps)})
        print("[DEBUG] Printing generated netlists...")
        print(simulator)
        
        return (
            np.array(getattr(analysis, sweep_node)),
            np.array(getattr(analysis, sense_node))
        )

    def run_simulation(self, operation='read', target_row=0, target_col=0):
        """Run specified test and return results"""
        circuit = self.create_testbench(operation, target_row, target_col)
        simulator = circuit.simulator()
        init_cond = self.set_initial_conditions()

        if 'snm' in operation:
            target_node_q = self.inst_prefix + '.Q'
            target_node_qb = self.inst_prefix + '.QB'
            target_node_qd = self.inst_prefix + '.QD'
            target_node_qbd = self.inst_prefix + '.QBD'
            # Perform DC sweep for SNM measurement
            vq_sweep, vqb_meas = self.dc_sweep(circuit, target_node_q, target_node_qbd)
            vqb_sweep, vq_meas = self.dc_sweep(circuit, target_node_qb, target_node_qd)

            # Combine results for butterfly curve
            butterfly_curve = np.vstack([
                np.column_stack((vq_sweep, vqb_meas)),
                np.column_stack((vq_meas, vqb_sweep))
            ])

            # plot butterfly curves
            plot_butterfly_with_squares(vq_sweep, vqb_meas, vqb_sweep, vq_meas, 'plots/sram_snm_curve.png')
            
            # get SNM value
            snm = calculate_snm(vq_sweep, vqb_meas, vqb_sweep, vq_meas, operation, 'plots/sram_snm_curve_rotated.png')

            return {
                'success': True,
                'snm': snm,
                'butterfly_curve': butterfly_curve,
            }

        elif operation == 'read':
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

            # Initial voltage conditions (Q=0) for the target cell
            target_node_q = self.inst_prefix + f'_{target_row}_{target_col}.Q'
            target_node_qb = self.inst_prefix + f'_{target_row}_{target_col}.QB'

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
            # Initial voltage conditions (Q=0) for the target cell
            target_node_q = self.inst_prefix + f'_{target_row}_{target_col}.Q'
            target_node_qb = self.inst_prefix + f'_{target_row}_{target_col}.QB'

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
                 pd_width, pu_width, pg_width, length,
                 num_rows, num_cols, w_rc, pi_res, pi_cap,
                 vth_std=0.05):
        
        super().__init__(
            pdk_path, nmos_model_name, pmos_model_name, 
            pd_width, pu_width, pg_width, length,
            num_rows, num_cols, w_rc, pi_res, pi_cap
        )
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

    def create_testbench(self, operation, target_row=0, target_col=0):
        """Create testbench with Monte Carlo models"""
        circuit = super().create_testbench(operation, target_row, target_col)

        # Replace original included model lib with new path
        circuit._includes[0] = self.temp_model_file

        return circuit
    
    def add_init_and_meas(self, simulator, init_cond, operation):
        if operation == 'hold_snm' or operation == 'read_snm':
            # Initial V(BL) and V(BLB) for the  cell
            init_cond = {}
            init_cond[f'BL']  = self.vdd @ u_V
            init_cond[f'BLB'] = self.vdd @ u_V
            simulator.initial_condition(**init_cond)
            simulator.measure('DC', 'MAXVD', 'MAX V(VD)')
            simulator.measure('DC', operation.upper(), f"PARAM='1/sqrt(2)*MAXVD'")

        # The read operation
        elif operation == 'read':
            # Initial V(BL) and V(BLB) for the target column
            init_cond[f'BL{self.target_col}'] = 0 @ u_V
            init_cond[f'BLB{self.target_col}'] = 0 @ u_V
            simulator.initial_condition(**init_cond)

            # Add measurements for read delay, 
            # which is defined as the time from the WL rise to BL swing to VDD/2
            simulator.measure(
                'TRAN', 'TWL', 
                f'WHEN V(WL{self.target_row}) VAL={self.half_vdd} RISE=1 ')
            simulator.measure(
                'TRAN', 'TBL', 
                f"WHEN V(BL{self.target_col})='V(BLB{self.target_col})-{self.half_vdd}' FALL=1")
            simulator.measure('TRAN', 'TREAD', f"PARAM='TBL-TWL'")
        
        # The write operation
        elif operation == 'write':
            # Initial all cell to '0'
            simulator.initial_condition(**init_cond)
            # Initial voltage conditions (Q=0) for the target cell
            target_node_q = self.inst_prefix + f'_{self.target_row}_{self.target_col}.Q'
            target_node_qb = self.inst_prefix + f'_{self.target_row}_{self.target_col}.QB'

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
    
    def run_mc_simulation(self, operation='read', target_row=0, target_col=0, mc_runs=100):
        """Run HSPICE Monte Carlo simulation"""
        circuit = self.create_testbench(operation, target_row, target_col)
        simulator = circuit.simulator()
        # Initial the idle data nodes
        init_cond = super().set_initial_conditions()
        
        # Add measurements according to the operation
        self.add_init_and_meas(simulator, init_cond, operation)

        # Add dc/transient analysis 
        if 'snm' in operation:
            u_tmp = self.vdd / np.sqrt(2)
            netlist = str(simulator).replace(
                '.end\n', 
                f'.DC U -{u_tmp:.2f} {u_tmp:.2f} 0.001 SWEEP MONTE={mc_runs:d}\n' +
                '.END\n')
        else:
            netlist = str(simulator).replace(
                '.end\n', 
                f'.TRAN {float(self.t_rise)*0.1:.2e} ' + 
                f'{float(self.t_period):.2e} SWEEP MONTE={mc_runs:d}\n' +
                '.END\n')
        print("[DEBUG] Printing generated netlists...")
        print(netlist)

        # Generate and run HSPICE netlist
        tb_path = self.sim_path + '/mc_netlist.sp'
        with open(tb_path, 'w') as f:
            f.write(str(netlist))
        
        # Execute HSPICE and parse results
        try:
            import subprocess
            # command: hspice -i <netlist> -o <output>
            print("[DEBUG] HSPICE running ...")

            result = subprocess.run(
                ['hspice', '-i', tb_path, '-o', self.sim_path],
                capture_output=True,
                text=True, check=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"HSPICE error:\n{result.stderr}, please check the log file {tb_path.replace('.sp', '.lis')}.")
            
        finally:
            result_suffix = '.ms0' if 'snm' in operation else '.mt0'
            result_path = self.sim_path + '/mc_netlist' + result_suffix
            data = analyze_mt0(result_path)
            print(f"[DEBUG] Parsed {len(data)} Monte Carlo samples from file {result_path}")
            return
        
    
if __name__ == '__main__':
    pdk_path = 'model_lib/models.spice'
    nmos_model_name = 'NMOS_VTL'
    pmos_model_name = 'PMOS_VTL'
    pd_width=0.1e-6
    pu_width=0.14e-6
    pg_width=0.06e-6
    length=45e-9

    print("===== 6T SRAM Array NgSpice Simulation Debug Session =====")
    testbench = SRAM_6T_Array_Testbench(
        pdk_path, nmos_model_name, pmos_model_name,
        pd_width, pu_width, pg_width, length,
        num_rows=2, num_cols=2, 
        w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF)
    
    testbench.run_simulation(operation='write_snm', target_row=1, target_col=1)
    print(f"[DEBUG] Simulation completed")
    
    # testbench.run_simulation(operation='read', target_row=1, target_col=1)
    # print("[DEBUG] Simulation of read operation completed")
    
    # testbench.run_simulation(operation='write', target_row=1, target_col=1)
    # print("[DEBUG] Simulation of write operation completed")

    # print("===== 6T SRAM Array Monte Carlo Simulation Debug Session =====")
    # mc_testbench = SRAM_6T_Array_MC_Testbench(
    #     pdk_path, nmos_model_name, pmos_model_name,
    #     pd_width=0.1e-6, pu_width=0.2e-6, pg_width=1.5e-6, length=45e-9,
    #     num_rows=2, num_cols=2, 
    #     w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF
    # )
    # mc_testbench.run_mc_simulation(operation='read_snm', target_row=1, target_col=1, mc_runs=100)

    # mc_testbench.run_mc_simulation(operation='read', target_row=1, target_col=1, mc_runs=100)
    # print("[DEBUG] Monte Carlo simulation of read operation completed")
    
    # mc_testbench.run_mc_simulation(operation='write', target_row=1, target_col=1, mc_runs=100)
    # print("[DEBUG] Monte Carlo simulation of write operation completed")