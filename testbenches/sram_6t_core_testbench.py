from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA
from subcircuits.sram_6t_core_for_yield import (
    Sram6TCore, Sram6TCell, 
    Sram6TCoreForYield, Sram6TCellForYield  # Assuming SRAM_6T_Cell_RC is defined
)
from subcircuits.wordline_driver import WordlineDriver
from subcircuits.precharge_and_write_driver import Precharge, WriteDriver
from subcircuits.mux_and_sa import ColumnMux, SenseAmp
from utils import parse_spice_models
from testbenches.base_testbench import BaseTestbench


class Sram6TCoreTestbench(BaseTestbench):
    def __init__(self, vdd, pdk_path, nmos_model_name, pmos_model_name, 
                 pd_width, pu_width, pg_width, length,
                 num_rows, num_cols, w_rc, pi_res, pi_cap,
                 custom_mc: bool=False, 
                 q_init_val: int=0
                ):
        super().__init__(
            f'SRAM_6T_CORE_{num_rows}x{num_cols}_TB', 
            vdd, pdk_path, nmos_model_name, pmos_model_name)
        # transistor size info.
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.pg_width = pg_width
        self.length = length
        # array size
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_inst_prefix = 'X'
        self.arr_inst_prefix = 'X'
        # add rc?
        self.w_rc = w_rc
        self.pi_res = pi_res
        self.pi_cap = pi_cap
        self.heir_delimiter = ':'
        # User defined MC simulation
        self.custom_mc = custom_mc
        
        # init internal data q
        self.q_init_val = q_init_val
        # default mux inputs
        self.mux_in = 1

    def create_wl_driver(self, circuit: Circuit, target_row: int):
        """Create wordline driver for the target/standby row"""
        wldrv = WordlineDriver(
            nmos_model_name=self.nmos_model_name, 
            pmos_model_name=self.pmos_model_name,
            num_cols=self.num_cols,
            w_rc=self.w_rc, # default `w_rc` is False
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
        )
        circuit.subcircuit(wldrv)

        # Wordline control & drivers
        for row in range(self.num_rows):
            if row == target_row:
                # Add WL enable source for the target row
                circuit.PulseVoltageSource(
                    f'WLE_{row}', f'WLE{row}', self.gnd_node,
                    initial_value=0 @ u_V, pulsed_value=self.vdd, 
                    delay_time=self.t_pulse,
                    rise_time=self.t_rise, fall_time=self.t_fall, 
                    pulse_width=self.t_pulse,
                    period=self.t_period
                )
                # Add pulse source for the target row
                circuit.X(
                    wldrv.name, wldrv.name,
                    self.power_node, self.gnd_node, f'WLE{row}', 
                    self.power_node, # NOTE: We temporarily set the enable node to VDD
                    f'WL{row}', 
                )
            else:
                # Tie idle wordlines to ground
                circuit.V(f'WL{row}_gnd', f'WL{row}', self.gnd_node, 0 @ u_V)
        return circuit

    def create_read_periphery(self, circuit: Circuit, target_col: int):
        """Create read periphery circuitry"""
        prch = Precharge(
            self.pmos_model_name, 
            w_rc=self.w_rc,  # default `w_rc` is False
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
            num_rows=self.num_rows,
        )
        circuit.subcircuit(prch)
        self.prch_inst_prefix = f"X{prch.name}"

        # Add precharge circuitry for all columns
        for col in range(self.num_cols):
            circuit.X(
                f'{prch.name}_{col}', 
                prch.name, 
                self.power_node, 'PRE', f'BL{col}', f'BLB{col}'
            )

        # Precharge control signal
        circuit.PulseVoltageSource(
            'PRE', 'PRE', self.gnd_node, 
            initial_value=self.vdd, pulsed_value=0 @ u_V, 
            delay_time=0 @ u_ns, 
            rise_time=self.t_rise, 
            fall_time=self.t_fall, 
            pulse_width=self.t_pulse-2*self.t_rise, 
            period=self.t_period, dc_offset=self.vdd
        )

        # we temporarily fix this to 2
        self.mux_in = 2

        # Column Mux
        cmux = ColumnMux(
            self.nmos_model_name, 
            self.pmos_model_name, 
            self.mux_in, w_rc=self.w_rc, 
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
        )
        circuit.subcircuit(cmux)
        self.cmux_inst_prefix = f"X{cmux.name}"

        # Add Column Mux for all columns
        for col in range(self.num_cols // self.mux_in):
            circuit.X(
                f'{cmux.name}_{col}', 
                cmux.name, 
                self.power_node, self.gnd_node, # Power node and GND node
                f'SA_IN{col}',                  # SA inputs are Mux's outputs
                f'SA_INB{col}',                 # SA inputs are Mux's outputs
                # SELect signal, high valid, #SEL = self.mux_in
                *[f'SEL{i}' for i in range(self.mux_in)], 
                # Inputs are BLs, #BLs  = self.mux_in
                *[f'BL{i}' for i in range(col*self.mux_in, (col+1)*self.mux_in)],
                # Inputs are BLBs, #BLBs = self.mux_in
                *[f'BLB{i}' for i in range(col*self.mux_in, (col+1)*self.mux_in)],
            ) 
        # Set SEL signals
        for i in range(self.mux_in):
            if i == target_col % self.mux_in:
                # Pulse setting of select signal is the same as WLE
                circuit.PulseVoltageSource(
                    f'SEL_{i}', f'SEL{i}', self.gnd_node, 
                    initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V, 
                    delay_time=self.t_pulse,
                    rise_time=self.t_rise, fall_time=self.t_fall, 
                    pulse_width=self.t_pulse,
                    period=self.t_period
                )
            else:
                circuit.V(f'SEL_{i}', f'SEL{i}', self.gnd_node, 0 @ u_V)

        # Sense Amplifer
        sa = SenseAmp(
            self.nmos_model_name, 
            self.pmos_model_name, 
            w_rc=self.w_rc, # default `w_rc` is False
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
        )
        circuit.subcircuit(sa)
        self.sa_inst_prefix = f'X{sa.name}'

        # Add SA circuitry for all columns
        for col in range(self.num_cols // self.mux_in):
            circuit.X(
                f'{sa.name}_{col}', 
                sa.name, 
                self.power_node, self.gnd_node,
                'SAE',   # SA Enable signal
                f'SA_IN{col}', f'SA_INB{col}', # Inputs
                f'SA_Q{col}', f'SA_QB{col}',   # Outputs
            )

        # SA enable signal
        circuit.PulseVoltageSource(
            'SAE', 'SAE', self.gnd_node, 
            initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V, 
            delay_time=2*self.t_pulse, 
            rise_time=self.t_rise, 
            fall_time=self.t_fall, 
            pulse_width=self.t_pulse, 
            period=self.t_period,
        )
        
        return circuit
    
    def create_write_periphery(self, circuit: Circuit):
        """Create write periphery circuitry, writing `1`s into a row"""
        write_drv = WriteDriver(
            self.nmos_model_name, self.pmos_model_name,
            # base_nmos_width, base_pmos_width, length,
            w_rc=self.w_rc, # default `w_rc` is False
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
            num_rows=self.num_rows,
        )
        
        circuit.subcircuit(write_drv)
        self.wdrv_inst_name = write_drv.name
        self.wdrv_inst_prefix = f"X{write_drv.name}"

        # Instantiate write drivers for all columns
        for col in range(self.num_cols):
            circuit.X(
                self.wdrv_inst_name+f"_{col}",
                write_drv.name,
                self.power_node, # Power net
                self.gnd_node,   # Ground net
                'WE',            # Write Enable signal
                f'DIN{col}',     # Data In
                f'BL{col}',      # Connect to column bitline
                f'BLB{col}',     # Connect to column bitline bar
            )

        # Write `1` into all columns
        for col in range(self.num_cols):
            circuit.V(f'DIN{col}', f'DIN{col}', self.gnd_node, self.vdd @ u_V)
            # circuit.PulseVoltageSource(
            #     f'DIN{col}', f'DIN{col}', self.gnd_node, 
            #     initial_value=0 @ u_V, pulsed_value=self.vdd, 
            #     # data setup time 
            #     delay_time=0, 
            #     rise_time=self.t_rise, fall_time=self.t_fall, 
            #     # data hold time = 2*t_delay time
            #     pulse_width=2*self.t_pulse + 2*self.t_delay, 
            #     period=self.t_period)
            
        # WE: Write enable signal
        circuit.PulseVoltageSource(
            f'WE', f'WE', self.gnd_node, 
            initial_value=0 @ u_V, pulsed_value=self.vdd, 
            # data on BL/BLB setup time = t_delay time
            delay_time=0, 
            rise_time=self.t_rise, fall_time=self.t_fall, 
            # data on BL/BLB hold time = t_delay time
            pulse_width=2*self.t_pulse + self.t_delay, 
            period=self.t_period
        )

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
            sbckt_6t_cell = Sram6TCellForYield(
                self.nmos_model_name, self.pmos_model_name,
                # This function returns a Dict of MOS models
                parse_spice_models(self.pdk_path),
                self.pd_width, self.pu_width, self.pg_width, self.length,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap, 
                disconnect=True, #NOTE: Key argument to disconnect the internal data nodes!!
                suffix='_0_0',
                custom_mc=self.custom_mc,
            )
        else:
            # Instantiate 6T SRAM cell
            sbckt_6t_cell = Sram6TCell(
                self.nmos_model_name, self.pmos_model_name,
                self.pd_width, self.pu_width, self.pg_width, self.length,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap, 
                disconnect=True, #NOTE: Key argument to disconnect the internal data nodes!!
            )
        # Add subcircuit definition to this testbench.
        circuit.subcircuit(sbckt_6t_cell)
        circuit.X(sbckt_6t_cell.name, sbckt_6t_cell.name, self.power_node, self.gnd_node, 
                'BL', 'BLB', 'WL')
        # internal node prefix in the SRAM cell
        self.cell_inst_prefix = 'X' + sbckt_6t_cell.name

        if operation == 'hold_snm':
            # For hold_snm measurement, keep WL low and add DC sources to Q/QB
            circuit.V(f'WL_gnd', 'WL', self.gnd_node, 0 @ u_V)

        elif operation == 'read_snm':
            # For read_snm operation, keep WL high and add DC sources to Q/QB
            circuit.V(f'WL_vdd', 'WL', self.gnd_node, self.vdd)
            circuit.V(f'BL_vdd', 'BL', self.gnd_node, self.vdd)
            circuit.V(f'BLB_vdd', 'BLB', self.gnd_node, self.vdd)
        elif operation == 'write_snm':
            # For write_snm operation, keep WL high and add DC sources to Q/QB
            circuit.V(f'WL_vdd', 'WL', self.gnd_node, self.vdd@ u_V)
            circuit.V(f'BL_vdd', 'BL', self.gnd_node, self.vdd @ u_V)
            circuit.V(f'BLB_vdd', 'BLB', self.gnd_node, 0 @ u_V)
        else:
            raise ValueError(f"Invalid operation: {operation}")

        # Add voltage control voltage source for get SNM,
        # The grammar is insane, but it works, fuckin' PySpice,
        # e.g., EV1 V1 0 VOL='U+sqrt(2)*V(XSRAM_6T_CELL.QBD)
        circuit.VCVS(
            'V1', 'V1', '', self.gnd_node, '', 
            **{'raw_spice': f"VOL='U+sqrt(2)*V({self.cell_inst_prefix}{self.heir_delimiter}QBD)'"}
        )
        circuit.VCVS(
            'V2', 'V2', '', self.gnd_node, '', 
            **{'raw_spice': f"VOL='-U+sqrt(2)*V({self.cell_inst_prefix}{self.heir_delimiter}QD)'"}
        )
        circuit.VCVS(
            'Q', f'{self.cell_inst_prefix}{self.heir_delimiter}Q', '', self.gnd_node, '', 
            **{'raw_spice': f" VOL='1/sqrt(2)*U+1/sqrt(2)*V(V1)'"}
        )
        circuit.VCVS(
            'QB', f'{self.cell_inst_prefix}{self.heir_delimiter}QB', '', self.gnd_node, '', 
            **{'raw_spice': f" VOL='-1/sqrt(2)*U+1/sqrt(2)*V(V2)'"}
        )
        circuit.VCVS(
            'VD', 'VD', '', self.gnd_node, '', 
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
                # Data Q name is specified by cell_inst_prefix and cell location (row, col)
                q_name = self.cell_inst_prefix + f'_{row}_{col}{self.heir_delimiter}Q'
                qb_name = self.cell_inst_prefix + f'_{row}_{col}{self.heir_delimiter}QB'
                init_dict[q_name] = vq
                init_dict[qb_name] = vqb
                # The target cell always stores '0' by default
                if row == self.target_row and col == self.target_col:
                    init_dict[q_name] = 0 @ u_V
                    init_dict[qb_name] = self.vdd @ u_V

        # initiate the voltage of inputs of SAs, connecting to column muxes
        for col in range(self.num_cols // self.mux_in):        
            init_dict[f'SA_IN{col}'] = self.vdd @ u_V
            init_dict[f'SA_INB{col}'] = self.vdd @ u_V
        
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
        circuit.V(self.power_node, self.power_node, self.gnd_node, self.vdd @ u_V)
        circuit.V(self.gnd_node, self.gnd_node, circuit.gnd, 0 @ u_V)

        # if it is a SNM test
        if 'snm' in operation:
            self.create_single_cell_for_snm(circuit, operation)
            # finish the circuit just return
            return circuit

        # Instantiate 6T SRAM array
        if self.custom_mc:
            sbckt_6t_array = Sram6TCoreForYield(
                self.num_rows, self.num_cols, 
                self.nmos_model_name, self.pmos_model_name,
                # This function returns a Dict of MOS models
                parse_spice_models(self.pdk_path),
                self.pd_width, self.pu_width, 
                self.pg_width, self.length,
                w_rc=self.w_rc, 
                pi_res=self.pi_res, pi_cap=self.pi_cap, 
            )
        else:
            sbckt_6t_array = Sram6TCore(
                self.num_rows, self.num_cols, 
                self.nmos_model_name, self.pmos_model_name,
                self.pd_width, self.pu_width, 
                self.pg_width, self.length,
                w_rc=self.w_rc, 
                pi_res=self.pi_res, pi_cap=self.pi_cap,
            )
        
        # Add subcircuit definition to this testbench.
        circuit.subcircuit(sbckt_6t_array)

        # Instantiate the SRAM array.
        circuit.X(sbckt_6t_array.name, sbckt_6t_array.name, self.power_node, self.gnd_node,
                  *[f'BL{i}' for i in range(self.num_cols)],
                  *[f'BLB{i}' for i in range(self.num_cols)],
                  *[f'WL{i}' for i in range(self.num_rows)])

        # internal node prefix in the SRAM cell
        self.arr_inst_prefix = f'X{sbckt_6t_array.name}'
        self.cell_inst_prefix = self.arr_inst_prefix + self.heir_delimiter + sbckt_6t_array.inst_prefix
        print(f"[DEBUG] self.arr_inst_prefix = {self.arr_inst_prefix}")
        print(f"[DEBUG] self.cell_inst_prefix = {self.cell_inst_prefix} of {self.name}")

        # For read transient simulation, add pulse source to the array
        if operation == 'read':
            self.create_read_periphery(circuit, target_col)
            self.create_wl_driver(circuit, target_row)
        # For write transient simulation, add pulse source to the array
        elif operation == 'write':
            self.create_write_periphery(circuit)
            self.create_wl_driver(circuit, target_row)

        else:
            raise ValueError(f"Invalid test type {operation}. Use 'read' or 'write'")
        
        return circuit