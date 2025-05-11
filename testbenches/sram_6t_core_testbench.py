import os
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
from subcircuits.sram_6t_core_for_yield import (
    SRAM_6T_Array, SRAM_6T_Cell, 
    SRAM_6T_Array_for_Yield, SRAM_6T_Cell_for_Yield  # Assuming SRAM_6T_Cell_RC is defined
)
from utils import plot_results, measure_delay, calculate_snm, plot_butterfly_with_squares
from utils import parse_mt0, analyze_mt0, measure_power
from testbenches.base import Base_Testbench
import numpy as np
from copy import deepcopy

class SRAM_6T_Array_Testbench(Base_Testbench):
    def __init__(self, vdd, pdk_path, nmos_model_name, pmos_model_name, 
                 pd_width, pu_width, pg_width, length,
                 num_rows, num_cols, w_rc, pi_res, pi_cap,
                 custom_mc=False, q_init_val=0):
        super().__init__(
            f'SRAM_6T_Array_{num_rows}x{num_cols}_Testbench', 
            vdd, pdk_path, nmos_model_name, pmos_model_name)
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
        self.heir_delimiter = ':'
        # User defined MC simulation
        self.custom_mc = custom_mc
        self.power_node = 'VDD'
        self.gnd_node = 'VSS'
        # init internal data q
        self.q_init_val = q_init_val

    def create_wl_driver(self, circuit: Circuit, target_row: int):
        """Create wordline driver for the target/standby row"""
        # Wordline control & drivers
        for row in range(self.num_rows):
            if row == target_row:
                # Add pulse source for the target row
                circuit.PulseVoltageSource(
                    f'WL{row}_pulse', f'WL{row}', circuit.gnd,
                    initial_value=0 @ u_V, pulsed_value=self.vdd, 
                    delay_time=self.t_pulse,
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
            circuit.M(f'MP1_{col}', f'BL{col}', 'PRE', 'VDD', 'VDD', model=self.pmos_model_name, w=0.4e-6, l=50e-9)
            circuit.M(f'MP2_{col}', f'BLB{col}', 'PRE', 'VDD', 'VDD', model=self.pmos_model_name, w=0.4e-6, l=50e-9)
            circuit.M(f'MP3_{col}', f'BL{col}', 'PRE', f'BLB{col}', 'VDD', model=self.pmos_model_name, w=0.4e-6, l=50e-9)
        
        # Precharge pulse source to precharge all BL/BLB to VDD
        circuit.PulseVoltageSource(
            'PRE', 'PRE', circuit.gnd, 
            initial_value=self.vdd, pulsed_value=0 @ u_V, 
            delay_time=0 @ u_ns, 
            rise_time=self.t_rise, 
            fall_time=self.t_fall, 
            pulse_width=self.t_pulse-2*self.t_rise, 
            period=self.t_period, dc_offset=self.vdd
        )
        return circuit
    
    def create_write_periphery(self, circuit: Circuit):
        """Create write periphery circuitry, writing `1`s into a row"""
        # Write drivers for all columns
        for col in range(self.num_cols):
            # high voltage on BL
            circuit.PulseVoltageSource(
                f'BL{col}_pulse', f'BL{col}', circuit.gnd, 
                initial_value=0 @ u_V, pulsed_value=self.vdd, 
                # data setup time = t_delay time
                delay_time=self.t_pulse - self.t_delay, 
                rise_time=self.t_rise, fall_time=self.t_fall, 
                # data hold time = t_delay time
                pulse_width=self.t_pulse + 2*self.t_delay, 
                period=self.t_period)
            
            # low voltage on BLB
            circuit.PulseVoltageSource(
                f'BLB{col}_pulse', f'BLB{col}', circuit.gnd,
                initial_value=self.vdd, pulsed_value=0 @ u_V, 
                # data setup time = t_delay time
                delay_time=self.t_pulse - self.t_delay, 
                rise_time=self.t_rise, fall_time=self.t_fall, 
                # data hold time = t_delay time
                pulse_width=self.t_pulse + 2*self.t_delay, 
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

        if self.custom_mc:
            # Instantiate 6T SRAM cell
            sbckt_6t_cell = SRAM_6T_Cell_for_Yield(
                self.vdd, self.nmos_model_name, self.pmos_model_name,
                self.pd_width, self.pu_width, self.pg_width, self.length,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap, 
                disconnect=True, #NOTE: Key argument to disconnect the internal data nodes!!
                suffix='_0_0',
                q_init=False, #NOTE: We can NOT initialize data node in DC analysis
                custom_mc=self.custom_mc,
            )
        else:
            # Instantiate 6T SRAM cell
            sbckt_6t_cell = SRAM_6T_Cell(
                self.vdd, self.nmos_model_name, self.pmos_model_name,
                self.pd_width, self.pu_width, self.pg_width, self.length,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap, 
                disconnect=True, #NOTE: Key argument to disconnect the internal data nodes!!
                q_init=False, #NOTE: We can NOT initialize data node in DC analysis
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
            **{'raw_spice': f"VOL='U+sqrt(2)*V({self.inst_prefix}{self.heir_delimiter}QBD)'"}
        )
        circuit.VCVS(
            'V2', 'V2', '', circuit.gnd, '', 
            **{'raw_spice': f"VOL='-U+sqrt(2)*V({self.inst_prefix}{self.heir_delimiter}QD)'"}
        )
        circuit.VCVS(
            'Q', f'{self.inst_prefix}{self.heir_delimiter}Q', '', circuit.gnd, '', 
            **{'raw_spice': f" VOL='1/sqrt(2)*U+1/sqrt(2)*V(V1)'"}
        )
        circuit.VCVS(
            'QB', f'{self.inst_prefix}{self.heir_delimiter}QB', '', circuit.gnd, '', 
            **{'raw_spice': f" VOL='-1/sqrt(2)*U+1/sqrt(2)*V(V2)'"}
        )
        circuit.VCVS(
            'VD', 'VD', '', circuit.gnd, '', 
            **{'raw_spice': f"VOL='ABS(V(V1)-V(V2))'"}
        )
        # print("[DEBUG] Netlists for SRAM_6T_Cell_for_Yield")
        # print(circuit)
        # assert 0 
        return circuit
    
    def data_init(self):
        init_dict = {}
        vq = self.vdd @ u_V if self.q_init_val else 0 @ u_V
        vqb = 0 @ u_V if self.q_init_val else self.vdd @ u_V

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                q_name = self.inst_prefix + f'_{row}_{col}{self.heir_delimiter}Q'
                qb_name = self.inst_prefix + f'_{row}_{col}{self.heir_delimiter}QB'
                init_dict[q_name] = vq
                init_dict[qb_name] = vqb
                # The target cell always stores '0' by default
                if row == self.target_row and col == self.target_col:
                    init_dict[q_name] = 0 @ u_V
                    init_dict[qb_name] = self.vdd @ u_V
        
        return init_dict

    def create_testbench(self, operation, target_row, target_col):
        """
        Create a testbench for the SRAM array.
        operation: 'read' or 'write'
        target_row: Row index of the target cell
        target_col: Column index of the target cell
        """
        self.target_row = target_row if target_row < self.num_rows else self.num_rows-1
        self.target_col = target_col if target_col < self.num_cols else self.num_cols-1

        circuit = Circuit(self.name)
        circuit.include(self.pdk_path)
        
        # Power supply
        circuit.V('VDD', 'VDD', circuit.gnd, self.vdd)

        # if it is a SNM test
        if 'snm' in operation:
            self.create_single_cell_for_snm(circuit, operation)
            # finish the circuit just return
            return circuit

        # Instantiate 6T SRAM array
        if self.custom_mc:
            sbckt_6t_array = SRAM_6T_Array_for_Yield(
                self.vdd, self.num_rows, self.num_cols, 
                self.nmos_model_name, self.pmos_model_name,
                self.pd_width, self.pu_width, self.pg_width, self.length,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap, 
                q_init_val=self.q_init_val
            )
        else:
            sbckt_6t_array = SRAM_6T_Array(
                self.vdd, self.num_rows, self.num_cols, 
                self.nmos_model_name, self.pmos_model_name,
                self.pd_width, self.pu_width, self.pg_width, self.length,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                q_init_val=self.q_init_val,
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
        self.inst_prefix = 'X' + sbckt_6t_array.name + self.heir_delimiter + sbckt_6t_array.inst_prefix

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

    def dc_sweep(self, circuit, sweep_node, sense_node, sweep_steps=1000):
        """Perform DC sweep and return characteristic curve"""
        circuit_tmp = circuit.clone()
        # The VCVS is only used for MC simulation, 
        # so we comment out these E sources in non-MC simulation
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

        print("[DEBUG] Printing generated netlists...")
        print(simulator)
        # Run dc analysis
        analysis = simulator.dc(**{Vsrc.name : slice(0, self.vdd, self.vdd/sweep_steps)})
        
        # assert 0
        return (
            np.array(getattr(analysis, sweep_node)),
            np.array(getattr(analysis, sense_node))
        )

    def run_simulation(self, operation='read', target_row=0, target_col=0):
        assert target_row < self.num_rows, \
            f"test row index {target_row} >= num_rows {self.num_rows}"
        assert target_col < self.num_cols, \
            f"test col index {target_col} >= num_cols {self.num_cols}"
        
        """Run specified test and return results"""
        circuit = self.create_testbench(operation, target_row, target_col)
        simulator = circuit.simulator()
        init_cond = {}

        # Internal nodes' names of the target cell
        target_node_q = self.inst_prefix + f'_{target_row}_{target_col}.Q'
        target_node_qb = self.inst_prefix + f'_{target_row}_{target_col}.QB'

        if 'snm' in operation:
            # Overwrite internal nodes' names
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
            plot_butterfly_with_squares(
                vq_sweep, vqb_meas, vqb_sweep, vq_meas, 
                f'plots/sram_{operation}_curve.png')
            
            # get SNM value
            snm = calculate_snm(
                vq_sweep, vqb_meas, vqb_sweep, vq_meas, operation, 
                f'plots/sram_{operation}_curve_rotated.png')
            
            print(f"[DEBUG] SNM={snm:.4f}")

            return {
                'success': True,
                'snm': snm,
                'butterfly_curve': butterfly_curve,
            }

        #NOTE: We read a `0` cell in default
        elif operation == 'read':
            # Initial V(BL) and V(BLB) for the target column
            init_cond[f'BL{target_col}'] = 0 @ u_V
            init_cond[f'BLB{target_col}'] = 0 @ u_V
            init_cond.update(self.data_init())
            # init_cond[target_node_q] = 0 @ u_V
            # init_cond[target_node_qb] = self.vdd @ u_V
            
            simulator.initial_condition(**init_cond)

            print("[DEBUG] Printing generated netlists...")
            print(simulator)

            # Run transient simulation
            analysis = simulator.transient(step_time=0.01 @ u_ns, end_time=self.t_period)
            # Calculate power from all voltage sources
            power = measure_power(self.vdd, analysis.branches)
            print(f"[DEBUG] Read Power={power:.4e}")

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

        #NOTE: We write a `1` to a `0` cell in default
        elif operation == 'write':
            init_cond.update(self.data_init())
            simulator.initial_condition(**init_cond)

            print("[DEBUG] Printing generated netlists...")
            print(simulator)

            # Run transient simulation
            analysis = simulator.transient(step_time=0.01 @ u_ns, end_time=self.t_period)
            # Calculate write power from all voltage sources (average value)
            power = measure_power(self.vdd, analysis.branches)
            print(f"[DEBUG] Write Power={power:.4e}")
            
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
            'power': power,
            'analysis': analysis,
        }