from PySpice.Spice.Netlist import Circuit 
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA 
from subcircuits.sram_6t_core_for_yield import ( # type: ignore
    Sram6TCore, Sram6TCell,
    Sram6TCoreForYield, Sram6TCellForYield  # Assuming SRAM_6T_Cell_RC is defined
)
from subcircuits.wordline_driver import WordlineDriver  # type: ignore
from subcircuits.precharge_and_write_driver import Precharge, WriteDriver  # type: ignore
from subcircuits.mux_and_sa import ColumnMux, SenseAmp  # type: ignore
from utils import parse_spice_models  # type: ignore
from testbenches.base_testbench import BaseTestbench  # type: ignore


class Sram6TCoreTestbench(BaseTestbench):#sram阵列测试平台，继承自BaseTestbench
    def __init__(self, sram_config,
                 w_rc=False, pi_res=10 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 custom_mc: bool = False,
                 q_init_val: int = 0
                 ):
        # 保存配置对象引用
        self.sram_config = sram_config  #包含所有子电路参数
        global_cfg = sram_config.global_config
        cell_cfg = sram_config.sram_6t_cell


        super().__init__(
            f'SRAM_6T_CORE_{global_cfg.num_rows}x{global_cfg.num_cols}_TB',
            global_cfg.vdd, global_cfg.pdk_path
            )
        # transistor size info.
        # self.pd_width = pd_width
        # self.pu_width = pu_width
        # self.pg_width = pg_width
        # self.length = length
        # array size
        self.num_rows = global_cfg.num_rows #从global.yaml中读取行数列数
        self.num_cols =global_cfg.num_cols
        self.cell_inst_prefix = 'X'         #实例前缀
        self.arr_inst_prefix = 'X'
        # add rc?
        self.w_rc = w_rc
        self.pi_res = pi_res
        self.pi_cap = pi_cap
        self.heir_delimiter = ':'
        # User defined MC simulation
        self.custom_mc = custom_mc  #是否启用mc

        # init internal data q
        self.q_init_val = q_init_val
        # default mux inputs
        self.mux_in = 1
        #self.set_vdd(5)

    def create_wl_driver(self, circuit: Circuit, target_row: int):  #创造写驱动电路函数
        """Create wordline driver for the target/standby row"""
        wl_config = self.sram_config.wordline_driver    #从总config类里提取wordline部分参数
        wldrv = WordlineDriver(
            nmos_model_name=wl_config.nmos_model.value[0],
            pmos_model_name=wl_config.pmos_model.value[0],
            base_nand_pmos_width=wl_config.pmos_width.value[0],
            base_nand_nmos_width=wl_config.nmos_width.value[0],
            base_inv_pmos_width=wl_config.pmos_width.value[1],
            base_inv_nmos_width =wl_config.nmos_width.value[1],
            length=wl_config.length.value,
            num_cols=self.num_cols,
            w_rc=self.w_rc,  # default `w_rc` is False
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
        )
        circuit.subcircuit(wldrv)   #添加到主电路

        # Wordline control & drivers
        for row in range(self.num_rows):
            if row == target_row:   #目标行，在MC里初始为第0行
                # Add WL enable source for the target row 添加脉冲电压源控制字线使能
                circuit.PulseVoltageSource(
                    f'WLE_{row}', f'WLE{row}', self.gnd_node,
                    initial_value=0 @ u_V, pulsed_value=self.vdd,
                    delay_time=self.t_pulse,
                    rise_time=self.t_rise, fall_time=self.t_fall,
                    pulse_width=self.t_pulse,
                    period=self.t_period
                )
                # Add pulse source for the target row 实例化字线驱动器
                circuit.X(
                    wldrv.name, wldrv.name,
                    self.power_node, self.gnd_node, f'WLE{row}',
                    self.power_node,  # NOTE: We temporarily set the enable node to VDD
                    f'WL{row}',
                )
            else:
                # Tie idle wordlines to ground 非目标行将字线接地
                circuit.V(f'WL{row}_gnd', f'WL{row}', self.gnd_node, 0 @ u_V)
        return circuit

    def create_read_periphery(self, circuit: Circuit, target_col: int):#创造读外围电路
        """Create read periphery circuitry预充电+多路选择器+感测放大器"""
        prch = Precharge(
            pmos_model_name=self.sram_config.precharge.pmos_model.value,
            base_pmos_width=self.sram_config.precharge.pmos_width.value,
            length=self.sram_config.precharge.length.value,
            w_rc=self.w_rc,  # default `w_rc` is False
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
            num_rows=self.num_rows,
        )
        circuit.subcircuit(prch)    #添加预充电电路到主电路
        self.prch_inst_prefix = f"X{prch.name}"

        # Add precharge circuitry for all columns 为每列都添加预充电电路实例
        for col in range(self.num_cols):
            circuit.X(
                f'{prch.name}_{col}',
                prch.name,
                self.power_node, 'PRE', f'BL{col}', f'BLB{col}'
            )

        # Precharge control signal  添加预充电控制信号（脉冲电压源）
        circuit.PulseVoltageSource(
            'PRE', 'PRE', self.gnd_node,
            initial_value=self.vdd, pulsed_value=0 @ u_V,
            delay_time=0 @ u_ns,
            rise_time=self.t_rise,
            fall_time=self.t_fall,
            pulse_width=self.t_pulse - 2 * self.t_rise, #不一样
            period=self.t_period, dc_offset=self.vdd
        )

        # we temporarily fix this to 2  固定为2路复用
        self.mux_in = 2

        # Column Mux
        cmux = ColumnMux(
            nmos_model_name=self.sram_config.column_mux.nmos_model.value,
            pmos_model_name=self.sram_config.column_mux.pmos_model.value,
            num_in=self.mux_in,
            base_nmos_width=self.sram_config.column_mux.nmos_width.value,
            base_pmos_width=self.sram_config.column_mux.pmos_width.value,
            length=self.sram_config.column_mux.length.value,
            w_rc=self.w_rc,
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
        )
        circuit.subcircuit(cmux)    #添加列多路选择器实例到主电路
        self.cmux_inst_prefix = f"X{cmux.name}"

        # Add Column Mux for all columns    为每组列添加多路复用器实例
        for col in range(self.num_cols // self.mux_in): #//表示整除，即需要几组多路选择器
            circuit.X(
                f'{cmux.name}_{col}',
                cmux.name,
                self.power_node, self.gnd_node,  # Power node and GND node
                f'SA_IN{col}',  # SA inputs are Mux's outputs
                f'SA_INB{col}',  # SA inputs are Mux's outputs
                # SELect signal, high valid, #SEL = self.mux_in
                *[f'SEL{i}' for i in range(self.mux_in)],
                # Inputs are BLs, #BLs  = self.mux_in
                *[f'BL{i}' for i in range(col * self.mux_in, (col + 1) * self.mux_in)],
                # Inputs are BLBs, #BLBs = self.mux_in
                *[f'BLB{i}' for i in range(col * self.mux_in, (col + 1) * self.mux_in)],
            )
            # Set SEL signals   设置选择信号：目标列使用脉冲源，其他列接地
        for i in range(self.mux_in):
            if i == target_col % self.mux_in:   #目标列所在的目标组
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
            nmos_model_name=self.sram_config.senseamp.nmos_model.value,
            pmos_model_name=self.sram_config.senseamp.pmos_model.value,
            base_nmos_width=self.sram_config.senseamp.nmos_width.value,
            base_pmos_width=self.sram_config.senseamp.pmos_width.value,
            length=self.sram_config.senseamp.length.value,
            w_rc=self.w_rc,  # default `w_rc` is False
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
        )
        circuit.subcircuit(sa)  #添加灵敏放大器实例到主电路
        self.sa_inst_prefix = f'X{sa.name}'

        # Add SA circuitry for all columns  #为每组多路选择器下接灵敏放大器
        for col in range(self.num_cols // self.mux_in):
            circuit.X(
                f'{sa.name}_{col}',
                sa.name,
                self.power_node, self.gnd_node,
                'SAE',  # SA Enable signal
                f'SA_IN{col}', f'SA_INB{col}',  # Inputs
                f'SA_Q{col}', f'SA_QB{col}',  # Outputs
            )

        # SA enable signal  添加灵敏放大器使能信号
        circuit.PulseVoltageSource(
            'SAE', 'SAE', self.gnd_node,
            initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V,
            delay_time=2 * self.t_pulse,    #比多路选择器延迟一个t_pulse
            rise_time=self.t_rise,
            fall_time=self.t_fall,
            pulse_width=self.t_pulse,
            period=self.t_period,
        )

        return circuit

    def create_write_periphery(self, circuit: Circuit):#创造写外围电路
        """Create write periphery circuitry, writing `1`s into a row,写驱动"""
        write_drv = WriteDriver(
            nmos_model_name=self.sram_config.write_driver.nmos_model.value,
            pmos_model_name=self.sram_config.write_driver.pmos_model.value,
            base_nmos_width=self.sram_config.write_driver.nmos_width.value,
            base_pmos_width=self.sram_config.write_driver.pmos_width.value,
            length=self.sram_config.write_driver.length.value,
            w_rc=self.w_rc,  # default `w_rc` is False
            # pi_res=self.pi_res, pi_cap=self.pi_cap,
            num_rows=self.num_rows,
        )

        circuit.subcircuit(write_drv)   #添加写驱动子电路实例到主电路
        self.wdrv_inst_name = write_drv.name
        self.wdrv_inst_prefix = f"X{write_drv.name}"

        # Instantiate write drivers for all columns 为每列添加写驱动器实例
        for col in range(self.num_cols):
            circuit.X(
                self.wdrv_inst_name + f"_{col}",
                write_drv.name,
                self.power_node,  # Power net
                self.gnd_node,  # Ground net
                'WE',  # Write Enable signal
                f'DIN{col}',  # Data In
                f'BL{col}',  # Connect to column bitline
                f'BLB{col}',  # Connect to column bitline bar
            )

        # Write `1` into all columns    设置所有数据输入为高电平（写 1）
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

        # WE: Write enable signal   添加写使能信号（脉冲电压源）
        circuit.PulseVoltageSource(
            f'WE', f'WE', self.gnd_node,
            initial_value=0 @ u_V, pulsed_value=self.vdd,
            # data on BL/BLB setup time = t_delay time
            delay_time=0,
            rise_time=self.t_rise, fall_time=self.t_fall,
            # data on BL/BLB hold time = t_delay time
            pulse_width=2 * self.t_pulse + self.t_delay,
            period=self.t_period
        )

        return circuit

    def create_single_cell_for_snm(self, circuit: Circuit, operation: str):
        """
        Create a single 6T SRAM cell for SNM measurement.创建cell单元的静态时序分析
        How to calculate SNM for 6T SRAM cell in SPICE?
        See: https://www.edaboard.com/threads/sram-snm-simulation-hspice.253224/
        """
        # Add U parameter
        # .param U=0
        circuit.parameter('U', 0)

        if self.custom_mc:
            # Instantiate 6T SRAM cell
            sbckt_6t_cell = Sram6TCellForYield(
                self.sram_config.sram_6t_cell.nmos_model.value[0],
                self.sram_config.sram_6t_cell.pmos_model.value,
                self.sram_config.sram_6t_cell.nmos_model.value[1],
                # This function returns a Dict of MOS models
                parse_spice_models(self.sram_config.global_config.pdk_path),
                self.sram_config.sram_6t_cell.nmos_width.value[0],
                self.sram_config.sram_6t_cell.pmos_width.value,
                self.sram_config.sram_6t_cell.nmos_width.value[1],
                self.sram_config.sram_6t_cell.length.value,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                disconnect=True,  # NOTE: Key argument to disconnect the internal data nodes!!
                suffix='_0_0',
                custom_mc=self.custom_mc,
            )
        else:
            # Instantiate 6T SRAM cell
            sbckt_6t_cell = Sram6TCell(
                self.sram_config.sram_6t_cell.nmos_model.value[0],
                self.sram_config.sram_6t_cell.pmos_model.value,
                self.sram_config.sram_6t_cell.nmos_model.value[1],
                self.sram_config.sram_6t_cell.nmos_width.value[0],
                self.sram_config.sram_6t_cell.pmos_width.value,
                self.sram_config.sram_6t_cell.nmos_width.value[1],
                self.sram_config.sram_6t_cell.length.value,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                disconnect=True,  # NOTE: Key argument to disconnect the internal data nodes!!
            )
        # Add subcircuit definition to this testbench.
        circuit.subcircuit(sbckt_6t_cell)   #添加到主电路
        circuit.X(sbckt_6t_cell.name, sbckt_6t_cell.name, self.power_node, self.gnd_node,
                  'BL', 'BLB', 'WL')
        # internal node prefix in the SRAM cell
        self.cell_inst_prefix = 'X' + sbckt_6t_cell.name

        if operation == 'hold_snm':
            # For hold_snm measurement, keep WL low and add DC sources to Q/QB
            #对于hold_snm测量，保持低WL并在Q/QB中添加直流源
            circuit.V(f'WL_gnd', 'WL', self.gnd_node, 0 @ u_V)

        elif operation == 'read_snm':
            # For read_snm operation, keep WL high and add DC sources to Q/QB
            #对于read_snm操作，保持WL高，并在Q/QB中添加DC源
            circuit.V(f'WL_vdd', 'WL', self.gnd_node, self.vdd)
            circuit.V(f'BL_vdd', 'BL', self.gnd_node, self.vdd)
            circuit.V(f'BLB_vdd', 'BLB', self.gnd_node, self.vdd)
        elif operation == 'write_snm':
            # For write_snm operation, keep WL high and add DC sources to Q/QB
            #对于write_snm操作，保持WL高，并在Q/QB中添加DC源
            circuit.V(f'WL_vdd', 'WL', self.gnd_node, self.vdd @ u_V)
            circuit.V(f'BL_vdd', 'BL', self.gnd_node, self.vdd @ u_V)
            circuit.V(f'BLB_vdd', 'BLB', self.gnd_node, 0 @ u_V)
        else:
            raise ValueError(f"Invalid operation: {operation}")

        # Add voltage control voltage source for get SNM,增加电压控制电压源获取SNM；
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

    def data_init(self):    #初始化数据节点 在MC_testbench2里用到
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
                if row == self.target_row and col == self.target_col:   #目标单元Q初始为0，QB为1，即存0
                    init_dict[q_name] = 0 @ u_V
                    init_dict[qb_name] = self.vdd @ u_V

        # initiate the voltage of inputs of SAs, connecting to column muxes 
        # 灵敏放大器输入全置为1
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
        self.target_row = target_row if target_row < self.num_rows else self.num_rows - 1
        self.target_col = target_col if target_col < self.num_cols else self.num_cols - 1

        circuit = Circuit(self.name)
        circuit.include(self.sram_config.global_config.pdk_path)

        # Power supply
        circuit.V(self.power_node, self.power_node, self.gnd_node, self.vdd @ u_V)
        circuit.V(self.gnd_node, self.gnd_node, circuit.gnd, 0 @ u_V)

        # if it is a SNM test   #operation里包含字母snm，operation取决于main2.py里
        #  mc_testbench.run_mc_simulation的输入
        if 'snm' in operation:
            self.create_single_cell_for_snm(circuit, operation)
            # finish the circuit just return
            return circuit

        # Instantiate 6T SRAM array 根据是否使用 MC 创建 SRAM Core
        if self.custom_mc:
            sbckt_6t_array = Sram6TCoreForYield(
                self.num_rows, self.num_cols,
                self.sram_config.sram_6t_cell.nmos_model.value[0],
                self.sram_config.sram_6t_cell.pmos_model.value,
                self.sram_config.sram_6t_cell.nmos_model.value[1],
                # This function returns a Dict of MOS models
                parse_spice_models(self.sram_config.global_config.pdk_path),
                self.sram_config.sram_6t_cell.nmos_width.value[0],
                self.sram_config.sram_6t_cell.pmos_width.value,
                self.sram_config.sram_6t_cell.nmos_width.value[1],
                self.sram_config.sram_6t_cell.length.value,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
            )
        else:
            sbckt_6t_array = Sram6TCore(
                self.num_rows, self.num_cols,
                self.sram_config.sram_6t_cell.nmos_model.value[0],
                self.sram_config.sram_6t_cell.pmos_model.value,
                self.sram_config.sram_6t_cell.nmos_model.value[1],
                self.sram_config.sram_6t_cell.nmos_width.value[0],
                self.sram_config.sram_6t_cell.pmos_width.value,
                self.sram_config.sram_6t_cell.nmos_width.value[1],
                self.sram_config.sram_6t_cell.length.value,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
            )

        # Add subcircuit definition to this testbench.
        circuit.subcircuit(sbckt_6t_array)  #添加到主电路

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
