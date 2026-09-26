from PySpice.Spice.Netlist import Circuit 
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA 
from sram_compiler.subcircuits.time_generate import TaperedBuffer
from sram_compiler.subcircuits.standard_cell import D_latch  # type: ignore
from sram_compiler.testbenches.parameter_factor import (TimeControlFactory,ReplicaColumnFactory,
                                                        DecoderCascadeFactory,WordlineDriverFactory,
                                                        PrechargeFactory,ColumnMuxFactory,SenseAmpFactory,WriteDriverFactory,
                                                        Sram6TCellFactory,Sram6TCoreFactory,Sram10TCellFactory,Sram10TCoreFactory)

from utils import parse_spice_models  # type: ignore
from sram_compiler.testbenches.base_testbench import BaseTestbench  # type: ignore
from math import ceil, log2
from copy import copy
from sram_compiler.interconnect import resolve_interconnect, add_tapped_line, cell_wire_nodes
from sram_compiler.equivalent_modeling import resolve_equivalent
from sram_compiler.sizing import resolve_driver_sizes, resolve_timing
from sram_compiler.sizing.timing import _timing_options, load_timing_lookup, require_envelope
from sram_compiler.sizing.table import physical_context
from sram_compiler.subcircuits.dummy_row_or_column import Dummy_Cell

class Sram6TCoreTestbench(BaseTestbench):#sram阵列测试平台，继承自BaseTestbench
    def __init__(self, sram_config, sram_cell_type="SRAM_6T_CELL",
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 custom_mc: bool = False,sweep_cell: bool = False,sweep_precharge: bool = False,sweep_senseamp: bool = False,sweep_wordlinedriver: bool = False,
                 sweep_columnmux:bool = False,sweep_writedriver:bool = False,sweep_decoder:bool = False,corner="TT",choose_columnmux:bool = True,real_cell_mode:int = None,
                 q_init_val: int = 0, sim_path: str = '', next_row: int = None,
                 driver_sizes=None, timing_config=None, temperature=None, interconnect=None,
                 select_every: int = 1,
                 ):
        # 保存配置对象引用
        self.sram_config = sram_config  #包含所有子电路参数
        global_cfg = sram_config.global_config
        self.temperature = global_cfg.temperature if temperature is None else temperature
        self.interconnect = resolve_interconnect(
            getattr(global_cfg, 'interconnect', None) if interconnect is None else interconnect)
        # Equivalent cells are a simulation input: global.yaml carries the
        # default, an explicit real_cell_mode overrides it (as for interconnect).
        self.equivalent = resolve_equivalent(
            getattr(global_cfg, 'equivalent', None) if real_cell_mode is None else real_cell_mode)
        real_cell_mode = self.equivalent.mode

        super().__init__(
            f'SRAM_6T_CORE_{global_cfg.num_rows}x{global_cfg.num_cols}_TB',
            global_cfg.vdd, global_cfg.pdk_path_TT
            )
        
        self.sram_cell_type = sram_cell_type
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
        self.choose_columnmux=choose_columnmux
        # Resolve from the baseline once. Optimizer/yield callers can inject the
        # same immutable result into each candidate's testbench.
        self.driver_sizes = driver_sizes if driver_sizes is not None else resolve_driver_sizes(
            sram_config, cell_type=sram_cell_type, mux=choose_columnmux,
            physical_context=physical_context(w_rc, float(pi_res), float(pi_cap), real_cell_mode, self.interconnect),
        )
        self.driver_sizes.validate_for(sram_config, sram_cell_type, choose_columnmux,
                                       physical_context(w_rc, float(pi_res), float(pi_cap), real_cell_mode, self.interconnect))
        if self.driver_sizes.source == 'table' and any((sweep_precharge, sweep_senseamp,
                sweep_wordlinedriver, sweep_columnmux, sweep_writedriver, sweep_decoder)):
            raise ValueError('Table-qualified periphery is frozen; use rules_only for peripheral sweeps')
        self.corner=corner#选择工艺角
        self.custom_mc = custom_mc  #是否启用mc
        self.sweep_cell = sweep_cell #cell单元是否用参数扫描
        self.sweep_precharge = sweep_precharge    #预充电电路是否用参数扫描
        self.sweep_senseamp = sweep_senseamp    #灵敏放大器电路是否用参数扫描
        self.sweep_wordlinrdriver = sweep_wordlinedriver     #字线驱动器电路是否用参数扫描
        self.sweep_columnmux = sweep_columnmux  #列多路选择器电路是否用参数扫描
        self.sweep_writedriver = sweep_writedriver  #写驱动电路是否用参数扫描
        self.sweep_decoder = sweep_decoder  #译码器电路是否用参数扫描
        self.real_cell_mode = real_cell_mode  #是否使用等效模型
        # init internal data q
        self.q_init_val = q_init_val
        self.sim_path = sim_path
        # Alternate row captured at the next rising edge (single read/write
        # decks). Clock-low recovery separates the old WL from this update.
        self.next_row = next_row
        # V2.1.6: chip select one cycle in `select_every` (1 = every cycle).  The
        # idle cycles between two single-deck accesses probe the access -> idle
        # -> access boundaries of the precharge and the write slot; single write
        # decks toggle their data at the selected edges, so the second write
        # registers new data (write -> write, or idle -> write).
        if isinstance(select_every, bool) or not isinstance(select_every, int) or select_every < 1:
            raise ValueError('select_every must be a positive integer')
        self.select_every = select_every
        # default mux inputs
        self.mux_in = 1
        self.timing_config = timing_config
        if self.timing_config is None:
            self.timing_config = resolve_timing(sram_config, self.driver_sizes,
                physical_context(w_rc, float(pi_res), float(pi_cap), real_cell_mode, self.interconnect))
        else:
            # V2.2.4: an injected clock does not buy a way past the envelope.
            require_envelope(self.num_rows, self.num_cols, load_timing_lookup(_timing_options(sram_config).get('lookup')))
        if hasattr(self.timing_config, 'validate_for'):
            self.timing_config.validate_for(sram_config, self.driver_sizes)
        self.timing_config.apply(self)
        #self.set_vdd(5)

    def create_time_control_circuit(self, circuit: Circuit,operation: str):
        """Create the TIME_CONTROL block (control-signal generator)"""
        self.operation=operation
        loads = self.driver_sizes.loads
        time_control = TimeControlFactory(
            nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG",
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            operation=self.operation,
            num_sa=loads.num_sa,
            wl_load=loads.wl_load,
            pre_load=loads.pre_load,
            wen_load=loads.wen_load,
            dc_stages=self.driver_sizes.dc_stages,
            effort_buffers=self.driver_sizes.effort_buffers,
            sen_load=loads.sen_load,
            iso_load=loads.iso_load,
            replica_precharge_guard=self.driver_sizes.replica_precharge_guard,
            sen_effort=loads.sen_effort,
            precharge_guard_stages=self.driver_sizes.precharge_guard_stages,
            interconnect=self.interconnect,
            precharge_off_guard=self.driver_sizes.precharge_off_guard,
            precharge_off_guard_stages=self.driver_sizes.precharge_off_guard_stages,
            access_load=loads.access_load,
            precharge_off_tau=self.driver_sizes.precharge_off_tau,
            enable_off_tau=self.driver_sizes.enable_off_tau,
            isolation_tau=self.driver_sizes.isolation_tau,
        ).create()
        circuit.subcircuit(time_control)   # Add to main circuit
        
        # Calculate address bits
        n_bits = ceil(log2(self.num_rows)) if self.num_rows > 1 else 1
        
        # Address nodes
        address_input_nodes = [f'A{i}' for i in range(n_bits)]
        address_output_nodes = [f'A_dff{i}' for i in range(n_bits)]
        
        if operation == 'write' or operation == 'read&write':
            # Data nodes
            data_input_nodes = [f'DIN{i}' for i in range(self.num_cols)]
            data_output_nodes = [f'DIN_dff{i}' for i in range(self.num_cols)]
        
        # All TIME_CONTROL connections
        control_connections = [
            'VDD', 'VSS', 'clk', 'csb', 'web', 'clk_buf', 'clk_bar',
            'cs_bar', 'cs', 'we_bar', 'we', 'gated_clk_bar', 'gated_clk_buf', 'wl_en'
        ]
        
        # Add address nodes
        control_connections.extend(address_input_nodes)
        control_connections.extend(address_output_nodes)
        if operation == 'write' or operation == 'read&write':
            # Add data nodes
            control_connections.extend(data_input_nodes)
            control_connections.extend(data_output_nodes)
        
        # Add remaining nodes.  With RC, the replica bitline reaches the timing
        # block through the same two segments a real bitline sees at its
        # sense-amplifier input (SenseAmp 'IN'), so RBL and BL carry the same
        # wire configuration end to end.
        # The replica sees the same peripheral ladder, mux and sense-input
        # devices as a real column, including when local series RC is disabled.
        rbl_node = ('XREPLICA_SENSEAMP:IN_end' if self.w_rc else
                    ('RBL_MUX' if self.choose_columnmux else self.periphery_tap('BL', None, 'sense')))
        control_connections.extend([rbl_node, 'rbl_delay', 'rbl_delay_bar', 's_en', 'w_en', 'PRE', 'sa_iso'])
        if self.driver_sizes.replica_precharge_guard:
            control_connections.append('RWL_far')
        if self.driver_sizes.precharge_off_guard:
            # The replica precharge sits past the final real column. Observe
            # its transistor gate, including the optional local series stub.
            control_connections.append('XPRECHARGE_RBL:ENB_end' if self.w_rc else 'PRE_line_far')
        
        control_connections.append('XREPLICA_SENSEAMP:ISO_end' if self.w_rc else self.control_tap('sa_iso'))
        control_connections.append('XREPLICA_SENSEAMP:EN_end' if self.w_rc else self.control_tap('s_en'))
        control_connections.append('XREPLICA_WDRV_LOAD:EN_end' if self.w_rc else self.control_tap('w_en'))

        # Instantiate the TIME_CONTROL block (instance XTIME_CONTROL)
        circuit.X(
            'TIME_CONTROL', time_control.NAME,
            *control_connections
        )
        return circuit
    
    def add_cs_startup_clamp(self, circuit: Circuit):
        """上电时把 CS/CS_BAR 强制钳在非使能态，避免 DFF 随机起态。"""
        # Release just *before* the first clock rising edge (1 ns + 0.2*T).  The CS
        # flip-flop's slave node is initialised to the inactive state by `.IC` (see
        # Sram6TCoreMcTestbench.add_meas_and_print), so CS stays low between the release
        # and the edge, and the clamp never fights the DFF output after the capture.
        release_time = 1.0 @ u_ns + 0.2 * self.t_period - 2 * self.t_rise

        # circuit.raw_spice += (
        #     '* Hold CS inactive until just after the first valid clock capture.\n'
        #     'MCSINIT_P cs_bar cs_init_bar VDD VDD PMOS_VTG l=5e-08 w=4.32e-06\n'
        #     'MCSINIT_N cs cs_init VSS VSS NMOS_VTG l=5e-08 w=1.44e-06\n'
        # )
        circuit.M(
            'CSINIT_P', 'cs_bar', 'cs_init_bar', 'VDD', 'VDD',
            model='PMOS_VTG', l=5e-08, w=4.32e-06
        )
        circuit.M(
            'CSINIT_N', 'cs', 'cs_init', 'VSS', 'VSS',
            model='NMOS_VTG', l=5e-08, w=1.44e-06
        )

        # One-shot: once released the clamp must stay off for the whole simulation
        # (the longest run, read&write, is 1 ns + 8*t_period).  With the previous
        # 2*t_period / 4*t_period pulse the clamp re-engaged at ~23 ns and blocked
        # every access in the 3rd and 4th cycles.
        clamp_off = 100 * self.t_period
        circuit.PulseVoltageSource(
            'CSINIT', 'cs_init', self.gnd_node,
            initial_value=self.vdd @ u_V, pulsed_value=0 @ u_V,
            delay_time=release_time,
            rise_time=0.2 * self.t_rise, fall_time=0.2 * self.t_fall,
            pulse_width=clamp_off,
            period=2 * clamp_off
        )

        circuit.PulseVoltageSource(
            'CSINITB', 'cs_init_bar', self.gnd_node,
            initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V,
            delay_time=release_time,
            rise_time=0.2 * self.t_rise, fall_time=0.2 * self.t_fall,
            pulse_width=clamp_off,
            period=2 * clamp_off
        )

    def create_replica_column(self, circuit: Circuit):
        """Create replica column circuitry"""
        # Create Replica Column circuit
        if self.sram_cell_type == 'SRAM_10T_CELL':
            cell_cfg = self.sram_config.sram_10t_cell
            replica_kwargs = {
                'pd_nmos_model': cell_cfg.nmos_model.value[0],
                'pu_pmos_model': cell_cfg.pmos_model.value,
                'pg_nmos_model': cell_cfg.nmos_model.value[1],
                'fd_nmos_model': cell_cfg.nmos_model.value[2],
                'pd_width': cell_cfg.nmos_width.value[0],
                'pu_width': cell_cfg.pmos_width.value,
                'pg_width': cell_cfg.nmos_width.value[1],
                'fd_width': cell_cfg.nmos_width.value[2],
                'pmos_choices': cell_cfg.pmos_model.choices,
                'nmos_choices': cell_cfg.nmos_model.choices,
                'length': cell_cfg.length.value,
            }
        else:
            cell_cfg = self.sram_config.sram_6t_cell
            replica_kwargs = {
                'pd_nmos_model': cell_cfg.nmos_model.value[0],
                'pu_pmos_model': cell_cfg.pmos_model.value,
                'pg_nmos_model': cell_cfg.nmos_model.value[1],
                'pd_width': cell_cfg.nmos_width.value[0],
                'pu_width': cell_cfg.pmos_width.value,
                'pg_width': cell_cfg.nmos_width.value[1],
                'pmos_choices': cell_cfg.pmos_model.choices,
                'nmos_choices': cell_cfg.nmos_model.choices,
                'length': cell_cfg.length.value,
            }

        if self.driver_sizes.replica_matched:
            sizes = self.driver_sizes
            replica_kwargs.update(
                pd_nmos_model=sizes.replica_nmos_models[0],
                pg_nmos_model=sizes.replica_nmos_models[1],
                pu_pmos_model=sizes.replica_pmos_model,
                pd_width=sizes.replica_nmos_widths[0],
                pg_width=sizes.replica_nmos_widths[1],
                pu_width=sizes.replica_pmos_width, length=sizes.replica_length,
            )
            if self.sram_cell_type == 'SRAM_10T_CELL':
                replica_kwargs.update(fd_nmos_model=sizes.replica_nmos_models[2],
                                      fd_width=sizes.replica_nmos_widths[2])
        replica_column = ReplicaColumnFactory(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
            sweep_replica=self.sweep_cell and not self.driver_sizes.replica_matched,
            param_model_file=self.sim_path + '/param_sweep_models.data',
            sram_cell_type=self.sram_cell_type,
            interconnect=self.interconnect,
            active_rows=self.driver_sizes.replica_k,
            **replica_kwargs
        ).create()
        circuit.subcircuit(replica_column)   # Add to main circuit
        self.replica_inst_prefix = f'X{replica_column.name}'
        
        # All Replica Column connections.  Only the RWL cell is an active replica; the
        # other num_rows cells are pure bitline loads with their wordlines tied off, so a
        # real-row access never discharges RBL in parallel with the replica cell.
        k = self.driver_sizes.replica_k
        replica_connections = ['VDD', 'VSS', 'RBL', 'RBLB', *[
            f'RWL_tap{self.num_cols - k + row - (self.num_rows - k)}'
            if row >= self.num_rows - k else self.gnd_node
            for row in range(self.num_rows)]]

        # Instantiate Replica Column circuit
        circuit.X(
            replica_column.NAME, replica_column.NAME,
            *replica_connections
        )
        return circuit
    
    def create_replica_wordline(self, circuit: Circuit):
        """Use the real row driver and match its baseline pass-gate count."""
        driver = self._wordline_driver()
        circuit.subcircuit(driver)
        circuit.X('RWL', driver.NAME, 'VDD', 'VSS', 'VDD', self.control_tap('wl_en'), 'RWL')
        add_tapped_line(circuit, 'RWL', 'RWL', self.num_cols, self.interconnect.wl)
        sizes = self.driver_sizes
        if self.num_cols > sizes.replica_k:
            dummy = Dummy_Cell(
                sizes.replica_nmos_models[0], sizes.replica_pmos_model,
                sizes.replica_nmos_models[1], sizes.replica_nmos_widths[0],
                sizes.replica_pmos_width, sizes.replica_nmos_widths[1],
                sizes.replica_length, w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                cell_pin_rc=self.interconnect.cell_pin_rc,
            )
            circuit.subcircuit(dummy)
            # The K active replica cells already contribute K access-gate pairs.
            # Dummy_Cell keeps its bitline drains disconnected internally.
            for col in range(self.num_cols - sizes.replica_k):
                node = f'RWL_tap{col}'
                circuit.X(f'RWL_LOAD_{col}', dummy.NAME, 'VDD', 'VSS', 'VDD', 'VDD', node)
        return circuit

    def create_decoder(self, circuit: Circuit):
        decoder_config = self.sram_config.decoder    #从总config类里提取decoder部分参数
        decoder = DecoderCascadeFactory(
            nmos_model_inv=decoder_config.nmos_model.value[0],
            pmos_model_inv=decoder_config.pmos_model.value[0],
            nmos_model_nand=decoder_config.nmos_model.value[0],
            pmos_model_nand=decoder_config.pmos_model.value[0],
            num_rows=self.num_rows,
            nand_pmos_width=decoder_config.pmos_width.value[0],
            nand_nmos_width=decoder_config.nmos_width.value[0],
            inv_pmos_width=decoder_config.pmos_width.value[1],
            inv_nmos_width =decoder_config.nmos_width.value[1],
            length=decoder_config.length.value,
            w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
            interconnect=self.interconnect,
            sweep_decoder=self.sweep_decoder,
            output_scale=self.driver_sizes.dec_inv,
            pmos_choices = self.sram_config.senseamp.pmos_model.choices,
            nmos_choices = self.sram_config.senseamp.nmos_model.choices,
            param_model_file =self.sim_path + '/param_sweep_models.data',
        ).create()
        circuit.subcircuit(decoder)   #添加到主电路
        # 计算地址位数
        n_bits = ceil(log2(self.num_rows)) if self.num_rows > 1 else 1
          # 地址节点
        address_nodes = [f'A_dff{i}' for i in range(n_bits)]
        
        # # 设置地址信号（目标行地址）
        # address_nodes = []
        # for bit in range(n_bits):
        #     bit_val = (target_row >> bit) & 1  # 提取每一位的值
        #     node_name = f'A{bit}'
        #     if bit_val:
        #         circuit.V(f'ADDR_{bit}', node_name, self.gnd_node, self.vdd)
        #     else:
        #         circuit.V(f'ADDR_{bit}', node_name, self.gnd_node, 0 @ u_V)
        #     address_nodes.append(node_name)
        
        # 添加使能信号 - 始终使能
        #circuit.V('DEC_EN', 'EN', self.gnd_node, self.vdd @ u_V)
        
        # 字线节点
        wl_nodes = [f'DEC_WL{i}' for i in range(self.num_rows)]
        
        # 实例化译码器
        circuit.X(
            'DECODER', decoder.NAME,
            self.power_node, self.gnd_node,  # VDD, VSS
            *address_nodes,                  # 地址信号
            *wl_nodes                       # 字线输出
        )
         # 保存译码器输出节点供字线驱动器使用
        self.decoder_wl_nodes = wl_nodes

        return circuit
    
    # Control lines that span the array.  Each runs the length of one array
    # dimension in its periphery and drives one load per row or column, so with
    # each is a tapped pi ladder. PRE, w_en, s_en and sa_iso run the
    # array width along the column periphery on the wordline pitch; wl_en runs
    # the array height along the wordline-driver column on the bitline pitch.
    _COLUMN_CONTROLS = ('PRE', 'w_en', 's_en', 'sa_iso')
    _ROW_CONTROLS = ('wl_en',)

    def _add_control_wires(self, circuit: Circuit, operation: str):
        """Each array-spanning control consumer attaches to its own wire tap."""
        self._control_taps = {}
        skip = ()
        for name in self._COLUMN_CONTROLS:
            if name not in skip:
                self._control_taps[name] = add_tapped_line(
                    circuit, f'{name}_line', name, self.num_cols, self.interconnect.wl)
        for name in self._ROW_CONTROLS:
            self._control_taps[name] = add_tapped_line(
                circuit, f'{name}_line', name, self.num_rows, self.interconnect.bl)

    def control_tap(self, name, index=None):
        """Control node at column/row `index`; the far end for the replica."""
        taps = self._control_taps[name]
        return f'{name}_line_far' if index is None else taps[index]

    def _add_periphery_wires(self, circuit):
        """Three peripheral pitches from each array port, mirrored on the replica.

        Placement order from the array is precharge, write driver, then
        mux/sense amplifier. Pitch uses the configured bitline wire geometry.
        """
        for col in [*range(self.num_cols), None]:
            for pin in ('BL', 'BLB'):
                net = f'{pin}{col}' if col is not None else f'R{pin}'
                add_tapped_line(circuit, f'{net}_periph', net, 3, self.interconnect.bl)

    def periphery_tap(self, pin, col, role):
        """BL/BLB connection at a peripheral block; col=None denotes replica."""
        if pin not in ('BL', 'BLB'):
            raise ValueError('Peripheral pin must be BL or BLB')
        index = {'precharge': 0, 'write': 1, 'sense': 2}[role]
        net = f'{pin}{col}' if col is not None else f'R{pin}'
        return f'{net}_periph_tap{index}'

    def _wordline_driver(self):
        """Build the common real/replica wordline driver definition."""
        wl_config = self.sram_config.wordline_driver    #从总config类里提取wordline部分参数
        return WordlineDriverFactory(
            nmos_model=wl_config.nmos_model.value[0],
            pmos_model=wl_config.pmos_model.value[0],
            nand_pmos_width=wl_config.pmos_width.value[0],
            nand_nmos_width=wl_config.nmos_width.value[0],
            inv_pmos_width=wl_config.pmos_width.value[1],
            inv_nmos_width =wl_config.nmos_width.value[1],
            length=wl_config.length.value,
            num_cols=self.num_cols,
            w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
            sweep_wordlinedriver = self.sweep_wordlinrdriver,
            inverter_scale=self.driver_sizes.wl_inv,
            nand_gate_scale=self.driver_sizes.wl_nand,
            pmos_modle_choices = self.sram_config.senseamp.pmos_model.choices,
            nmos_modle_choices = self.sram_config.senseamp.nmos_model.choices,
            param_model_file =self.sim_path + '/param_sweep_models.data',
        ).create()

    def create_wl_driver(self, circuit: Circuit, target_row: int):  #创造字线驱动电路函数
        """Create wordline driver for the target/standby row"""
        wldrv = self._wordline_driver()
        circuit.subcircuit(wldrv)   #添加到主电路

        # Wordline control & drivers
        for row in range(self.num_rows):
            # 使用译码器输出作为使能信号
            decoder_enable = self.decoder_wl_nodes[row]
                # Add pulse source for the target row 实例化字线驱动器
         # 添加字线驱动器
            circuit.X(
                f'WL_DRV_{row}', wldrv.name,
                self.power_node, self.gnd_node, 
                decoder_enable,    # 来自译码器的使能信号
                self.control_tap('wl_en', row),   # 内部使能（始终有效）
                f'WL{row}',        # 输出到SRAM阵列
            )
            # else:
            #     # Tie idle wordlines to ground 非目标行将字线接地
            #     circuit.V(f'WL{row}_gnd', f'WL{row}', self.gnd_node, 0 @ u_V)
        return circuit
    
    def create_D_latch(self, circuit: Circuit, target_col: int):
        """Create D_latch subcircuit and instance"""
        # Create D_latch instance
        d_latch = D_latch(
            nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG")
        
        # Add subcircuit definition to this testbench
        circuit.subcircuit(d_latch)
        
        # Connect D_latch instance to circuit.  With a column mux there is one sense amp
        # per mux group, so the latch input is SA_Q{target_col // mux_in}.
        circuit.X(
            'D_LATCH', d_latch.NAME,
            'VDD', 'VSS', f'SA_Q{target_col // self.mux_in}',
            self.control_tap('s_en', (target_col // self.mux_in) * self.mux_in),
            'OUT', 'OUT_B'
        )
        return circuit

    def create_read_periphery(self, circuit: Circuit, target_col: int):#创造读外围电路
        """Create read periphery circuitry预充电+多路选择器+感测放大器"""

        prch = PrechargeFactory(
            pmos_model=self.sram_config.precharge.pmos_model.value,
            pmos_width=self.sram_config.precharge.pmos_width.value,
            length=self.sram_config.precharge.length.value,
            w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
            num_rows=self.num_rows,
            sweep_precharge = self.sweep_precharge,
            scale=self.driver_sizes.pre,
            pmos_modle_choices = self.sram_config.precharge.pmos_model.choices,
            param_model_file =self.sim_path + '/param_sweep_models.data',
        ).create()
        circuit.subcircuit(prch)    #添加预充电电路到主电路
        self.prch_inst_prefix = f"X{prch.name}"

        # Add precharge circuitry for all columns 为每列都添加预充电电路实例
        for col in range(self.num_cols):
            circuit.X(
                f'{prch.name}_{col}',
                prch.name,
                self.power_node, self.control_tap('PRE', col),
                self.periphery_tap('BL', col, 'precharge'), self.periphery_tap('BLB', col, 'precharge')
            )
        # 新增一列，连接至 RBL 和 RBLB
        circuit.X(
            f'{prch.name}_RBL',
            prch.name,
            self.power_node, self.control_tap('PRE'),
            self.periphery_tap('BL', None, 'precharge'), self.periphery_tap('BLB', None, 'precharge')
        )

        if self.choose_columnmux:
        # we temporarily fix this to 2  固定为2路复用
            self.mux_in = 2
            if self.num_cols % self.mux_in != 0:
                raise ValueError(
                    f"choose_columnmux=True requires num_cols to be a multiple of "
                    f"mux_in={self.mux_in}, got num_cols={self.num_cols}: some columns "
                    f"would have no sense amplifier")
            # The mux generates SELB internally; SELB ports/sources exist only when True.
            use_external_selb = False

            # Column Mux
            cmux = ColumnMuxFactory(
                num_in=self.mux_in,
                nmos_model=self.sram_config.column_mux.nmos_model.value,
                pmos_model=self.sram_config.column_mux.pmos_model.value,
                nmos_width=self.sram_config.column_mux.nmos_width.value,
                pmos_width=self.sram_config.column_mux.pmos_width.value,
                length=self.sram_config.column_mux.length.value,
                w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                sweep_columnmux = self.sweep_columnmux,
                use_external_selb=use_external_selb, #选用哪种多路选择器
                pmos_modle_choices = self.sram_config.senseamp.pmos_model.choices,
                nmos_modle_choices = self.sram_config.senseamp.nmos_model.choices,
                param_model_file =self.sim_path + '/param_sweep_models.data',
            ).create()
            circuit.subcircuit(cmux)    #添加列多路选择器实例到主电路
            circuit.X('REPLICA_MUX', cmux.name, self.power_node, self.gnd_node,
                      'RBL_MUX', 'RBLB_MUX', 'VDD', 'VSS',
                      *(['VSS', 'VDD'] if use_external_selb else []),
                      self.periphery_tap('BL', None, 'sense'), 'VDD',
                      self.periphery_tap('BLB', None, 'sense'), 'VDD')
            self.cmux_inst_prefix = f"X{cmux.name}"
            selects = {f'SEL{i}': add_tapped_line(circuit, f'SEL{i}_line', f'SEL{i}',
                                                 self.num_cols, self.interconnect.wl)
                       for i in range(self.mux_in)}
            if use_external_selb:
                selects.update({f'SELB{i}': add_tapped_line(circuit, f'SELB{i}_line', f'SELB{i}',
                                                           self.num_cols, self.interconnect.wl)
                                for i in range(self.mux_in)})

            # Add Column Mux for all columns    为每组列添加多路复用器实例
            for col in range(self.num_cols // self.mux_in): #//表示整除，即需要几组多路选择器
                circuit.X(
                    f'{cmux.name}_{col}',
                    cmux.name,
                    self.power_node, self.gnd_node,  # Power node and GND node
                    f'SA_IN{col}',  # SA inputs are Mux's outputs
                    f'SA_INB{col}',  # SA inputs are Mux's outputs
                    # SELect signal, high valid, #SEL = self.mux_in
                    *[selects[f'SEL{i}'][col*self.mux_in] for i in range(self.mux_in)],
                    *([selects[f'SELB{i}'][col*self.mux_in] for i in range(self.mux_in)] if use_external_selb else []),
                    # Inputs are BLs, #BLs  = self.mux_in
                    *[self.periphery_tap('BL', i, 'sense') for i in range(col * self.mux_in, (col + 1) * self.mux_in)],
                    # Inputs are BLBs, #BLBs = self.mux_in
                    *[self.periphery_tap('BLB', i, 'sense') for i in range(col * self.mux_in, (col + 1) * self.mux_in)],
                )

            # Set SEL signals.  The column address is static for the whole cycle (the
            # target column never changes), so SEL is a DC level: the target group is
            # selected, the others are off.  This keeps the SA inputs precharged through
            # the mux and does not depend on t_pulse or t_period.
            for i in range(self.mux_in):
                selected = (i == target_col % self.mux_in)   #目标列所在的目标组
                circuit.V(f'SEL_{i}', f'SEL{i}', self.gnd_node,
                          self.vdd @ u_V if selected else 0 @ u_V)
                if use_external_selb:
                    circuit.V(f'SELB_{i}', f'SELB{i}', self.gnd_node,
                              0 @ u_V if selected else self.vdd @ u_V)

        # Sense Amplifer
        sa = SenseAmpFactory(
            nmos_model=self.sram_config.senseamp.nmos_model.value,
            pmos_model=self.sram_config.senseamp.pmos_model.value,
            nmos_width=self.sram_config.senseamp.nmos_width.value,
            pmos_width=self.sram_config.senseamp.pmos_width.value,
            length=self.sram_config.senseamp.length.value,
            w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
            sweep_senseamp = self.sweep_senseamp,
            pmos_modle_choices = self.sram_config.senseamp.pmos_model.choices,
            nmos_modle_choices = self.sram_config.senseamp.nmos_model.choices,
            param_model_file =self.sim_path + '/param_sweep_models.data',
        ).create()
        circuit.subcircuit(sa)  #添加灵敏放大器实例到主电路
        self.sa_inst_prefix = f'X{sa.name}'
        def sa_control(group=None):
            """Control nodes at sense-amplifier `group`; the far end for the replica."""
            column = None if group is None else group * self.mux_in
            return self.control_tap('s_en', column), self.control_tap('sa_iso', column)

        # The replica amplifier sits past the last real column.
        circuit.X('REPLICA_SENSEAMP', sa.name, self.power_node, self.gnd_node,
                  *sa_control(),
                  'RBL_MUX' if self.choose_columnmux else self.periphery_tap('BL', None, 'sense'),
                  'RBLB_MUX' if self.choose_columnmux else self.periphery_tap('BLB', None, 'sense'),
                  'REPLICA_SA_Q', 'REPLICA_SA_QB')

        if self.choose_columnmux:
            # Add SA circuitry for all columns  #为每组多路选择器下接灵敏放大器
            for col in range(self.num_cols // self.mux_in):
                circuit.X(
                    f'{sa.name}_{col}',
                    sa.name,
                    self.power_node, self.gnd_node,
                    *sa_control(col),  # SA enable and input-isolation control
                    f'SA_IN{col}', f'SA_INB{col}',  # Inputs
                    f'SA_Q{col}', f'SA_QB{col}',  # Outputs
                )

        else:
            # 直接连BL和BLB
            for col in range(self.num_cols):
                circuit.X(
                    f'{sa.name}_{col}',
                    sa.name,
                    self.power_node, self.gnd_node,
                    *sa_control(col),  # SA enable and input-isolation control
                    self.periphery_tap('BL', col, 'sense'), self.periphery_tap('BLB', col, 'sense'),  # Inputs
                    f'SA_Q{col}', f'SA_QB{col}',  # Outputs
                )
        return circuit

    def create_write_periphery(self, circuit: Circuit, operation: str = 'write'):#创造写外围电路
        """Create write periphery circuitry, writing `1`s into a row,写驱动"""
        write_drv = WriteDriverFactory(
            nmos_model=self.sram_config.write_driver.nmos_model.value,
            pmos_model=self.sram_config.write_driver.pmos_model.value,
            nmos_width=self.sram_config.write_driver.nmos_width.value,
            pmos_width=self.sram_config.write_driver.pmos_width.value,
            length=self.sram_config.write_driver.length.value,
            w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
            num_rows=self.num_rows,
            sweep_writedriver = self.sweep_writedriver,
            scale=self.driver_sizes.wd_in,
            out_scale=self.driver_sizes.wd_out,
            pmos_modle_choices = self.sram_config.write_driver.pmos_model.choices,
            nmos_modle_choices = self.sram_config.write_driver.nmos_model.choices,
            param_model_file =self.sim_path + '/param_sweep_models.data',
        ).create()

        circuit.subcircuit(write_drv)   #添加写驱动子电路实例到主电路
        self.wdrv_inst_name = write_drv.name
        self.wdrv_inst_prefix = f"X{write_drv.name}"

        if self.driver_sizes.canonical_read:
            # The replica bitline must see the same disabled output-stack drain
            # load as a real bitline, including on large row-scaled drivers.
            circuit.X('REPLICA_WDRV_LOAD', write_drv.name,
                      self.power_node, self.gnd_node, self.control_tap('w_en'), self.gnd_node,
                      self.periphery_tap('BL', None, 'write'), self.periphery_tap('BLB', None, 'write'))

        if operation == 'read':
            # Keep disabled output stacks on both bitlines during read accesses.
            for col in range(self.num_cols):
                circuit.X(f'{write_drv.name}_{col}', write_drv.name,
                          self.power_node, self.gnd_node, self.control_tap('w_en', col),
                          self.gnd_node, self.periphery_tap('BL', col, 'write'),
                          self.periphery_tap('BLB', col, 'write'))
            return circuit

        # The rising-edge data register is stable through clock-high access
        # and clock-low recovery. Buffer its load; no second storage element.
        sizes = self.driver_sizes
        wn = float(self.sram_config.write_driver.nmos_width.value)
        wp = float(self.sram_config.write_driver.pmos_width.value)
        load = ((wn + wp) * (sizes.wd_in + sizes.wd_out) / .36e-6
                + (float(self.pi_cap) / .5e-15 if self.w_rc else 0.0))
        buffer = TaperedBuffer('WRITE_DATA_BUFFER', drive_scale=max(1, ceil(load / 8)),
                               effort_based=sizes.effort_buffers, load_units=load)
        circuit.subcircuit(buffer)

        # Instantiate write drivers for all columns 为每列添加写驱动器实例
        for col in range(self.num_cols):
            circuit.X(
                f'DIN_BUF_{col}', buffer.NAME,
                self.power_node, self.gnd_node, f'DIN_dff{col}', f'DIN_buf{col}',
            )
            circuit.X(
                self.wdrv_inst_name + f"_{col}",
                write_drv.name,
                self.power_node,  # Power net
                self.gnd_node,  # Ground net
                self.control_tap('w_en', col),  # Write Enable signal
                f'DIN_buf{col}',  # Captured data, buffered for the driver load
                self.periphery_tap('BL', col, 'write'),
                self.periphery_tap('BLB', col, 'write'),
            )

        if operation == 'write':
            # Write `1` into all columns in the first selected cycle, `0` in the
            # next (V2.1.6: the second write drives the opposite rails, so the
            # write slot of the following cycle is measured as TWSLOT).
            # With select_every > 1 the data is held through the idle cycles and
            # changes 0.1 T before the next selected edge, so the idle -> write
            # probe also registers new data at that edge (V2.1.7; a change in the
            # idle cycle left the write-data hold latch open long before w_en).
            if self.select_every == 1:
                data_width = 0.2 * self.t_period
            else:
                data_width = self.select_every * self.t_period - self.t_rise
            for col in range(self.num_cols):    
                #circuit.V(f'DIN{col}', f'DIN{col}', self.gnd_node, self.vdd @ u_V)
                circuit.PulseVoltageSource(
                    f'DIN{col}', f'DIN{col}', self.gnd_node,
                        initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V,
                        delay_time=1.0 @ u_ns +0.1 * self.t_period,  # 预充电开始后0.5ns,留1s静默
                        rise_time=self.t_rise,fall_time=self.t_fall,
                        pulse_width=data_width,  # 保持有效
                        period=2 * self.select_every * self.t_period)
        elif operation =="read&write":
            # Write `1` into all columns    设置输入数据为高低转换
            for col in range(self.num_cols):    
                #circuit.V(f'DIN{col}', f'DIN{col}', self.gnd_node, self.vdd @ u_V)
                circuit.PulseVoltageSource(
                    f'DIN{col}', f'DIN{col}', self.gnd_node,
                        initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V,
                        delay_time=1.0 @ u_ns +0.1 * self.t_period,  # 预充电开始后0.5ns，,留1s静默
                        rise_time=self.t_rise,fall_time=self.t_fall,
                        pulse_width=0.2 * self.t_period , # 保持有效
                        period=4*self.t_period)

        # The precharge (all columns + replica column), the column mux and the
        # sense amplifiers are created by create_read_periphery(), which
        # create_testbench() now calls for every transient operation.
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

        if self.sram_cell_type == "SRAM_6T_CELL":
            if self.custom_mc:
                # Instantiate 6T SRAM cell
                sbckt_cell = Sram6TCellFactory(
                    pd_model=self.sram_config.sram_6t_cell.nmos_model.value[0],
                    pu_model=self.sram_config.sram_6t_cell.pmos_model.value,
                    pg_model=self.sram_config.sram_6t_cell.nmos_model.value[1],
                    
                    pd_width=self.sram_config.sram_6t_cell.nmos_width.value[0],
                    pu_width=self.sram_config.sram_6t_cell.pmos_width.value,
                    pg_width=self.sram_config.sram_6t_cell.nmos_width.value[1],
                    length=self.sram_config.sram_6t_cell.length.value,
                    w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                    disconnect=True,  # NOTE: Key argument to disconnect the internal data nodes!!
                    sweep = self.sweep_cell,
                    yield_mode=True,
                    # This function returns a Dict of MOS models
                    model_dict=parse_spice_models(getattr(self.sram_config.global_config, f"pdk_path_{self.corner}")),
                    suffix='_0_0',
                    pmos_modle_choices = self.sram_config.sram_6t_cell.pmos_model.choices,
                    nmos_modle_choices = self.sram_config.sram_6t_cell.nmos_model.choices,
                    param_model_file =self.sim_path + '/param_sweep_models.data',
                ).create()
            else:
                # Instantiate 6T SRAM cell
                sbckt_cell = Sram6TCellFactory(
                    pd_model=self.sram_config.sram_6t_cell.nmos_model.value[0],
                    pu_model=self.sram_config.sram_6t_cell.pmos_model.value,
                    pg_model=self.sram_config.sram_6t_cell.nmos_model.value[1],
                    pd_width=self.sram_config.sram_6t_cell.nmos_width.value[0],
                    pu_width=self.sram_config.sram_6t_cell.pmos_width.value,
                    pg_width=self.sram_config.sram_6t_cell.nmos_width.value[1],
                    length=self.sram_config.sram_6t_cell.length.value,
                    w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                    disconnect=True,  # NOTE: Key argument to disconnect the internal data nodes!!
                    sweep = self.sweep_cell,
                    yield_mode=False,
                    pmos_choices = self.sram_config.sram_6t_cell.pmos_model.choices,
                    nmos_choices = self.sram_config.sram_6t_cell.nmos_model.choices,
                    param_model_file =self.sim_path + '/param_sweep_models.data',
                ).create()
        elif self.sram_cell_type == "SRAM_10T_CELL":
            if self.custom_mc:
            # Instantiate 10T SRAM cell
                sbckt_cell = Sram10TCellFactory(
                    pd_model=self.sram_config.sram_10t_cell.nmos_model.value[0],
                    pu_model=self.sram_config.sram_10t_cell.pmos_model.value,
                    pg_model=self.sram_config.sram_10t_cell.nmos_model.value[1],
                    fd_model=self.sram_config.sram_10t_cell.nmos_model.value[2],
                    pd_width=self.sram_config.sram_10t_cell.nmos_width.value[0],
                    pu_width=self.sram_config.sram_10t_cell.pmos_width.value,
                    pg_width=self.sram_config.sram_10t_cell.nmos_width.value[1],
                    fd_width=self.sram_config.sram_10t_cell.nmos_width.value[2],
                    length=self.sram_config.sram_10t_cell.length.value,
                    w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                    disconnect=True,  # NOTE: Key argument to disconnect the internal data nodes!!
                    sweep = self.sweep_cell,
                    yield_mode=True,
                    model_dict=parse_spice_models(getattr(self.sram_config.global_config, f"pdk_path_{self.corner}")),
                    suffix='_0_0',
                    pmos_modle_choices = self.sram_config.sram_10t_cell.pmos_model.choices,
                    nmos_modle_choices = self.sram_config.sram_10t_cell.nmos_model.choices,
                    param_model_file =self.sim_path + '/param_sweep_models.data',
                ).create()
            else:
                # Instantiate 10T SRAM cell
                sbckt_cell = Sram10TCellFactory(
                    pd_model=self.sram_config.sram_10t_cell.nmos_model.value[0],
                    pu_model=self.sram_config.sram_10t_cell.pmos_model.value,
                    pg_model=self.sram_config.sram_10t_cell.nmos_model.value[1],
                    fd_model=self.sram_config.sram_10t_cell.nmos_model.value[2],
                    pd_width=self.sram_config.sram_10t_cell.nmos_width.value[0],
                    pu_width=self.sram_config.sram_10t_cell.pmos_width.value,
                    pg_width=self.sram_config.sram_10t_cell.nmos_width.value[1],
                    fd_width=self.sram_config.sram_10t_cell.nmos_width.value[2],
                    length=self.sram_config.sram_10t_cell.length.value,
                    w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                    disconnect=True,  # NOTE: Key argument to disconnect the internal data nodes!!
                    sweep = self.sweep_cell,
                    yield_mode=False,
                    pmos_choices = self.sram_config.sram_6t_cell.pmos_model.choices,
                    nmos_choices = self.sram_config.sram_6t_cell.nmos_model.choices,
                    param_model_file =self.sim_path + '/param_sweep_models.data',
                ).create()
        
        # Add subcircuit definition to this testbench.
        circuit.subcircuit(sbckt_cell)   #添加到主电路
        circuit.X(sbckt_cell.name, sbckt_cell.name, self.power_node, self.gnd_node,
                  'BL', 'BLB', 'WL')
        # internal node prefix in the SRAM cell
        self.cell_inst_prefix = 'X' + sbckt_cell.name

        if operation == 'hold_snm':
            # For hold_snm measurement, keep WL low and add DC sources to Q/QB
            #对于hold_snm测量，保持低WL并在Q/QB中添加直流源
            # BL/BLB need real sources: `.IC` is not applied in a .DC sweep and the
            # access transistors are off, so the bitlines would otherwise float.
            circuit.V(f'WL_gnd', 'WL', self.gnd_node, 0 @ u_V)
            circuit.V(f'BL_vdd', 'BL', self.gnd_node, self.vdd @ u_V)
            circuit.V(f'BLB_vdd', 'BLB', self.gnd_node, self.vdd @ u_V)

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

    def data_init(self):    #初始化数据节点 在MC_testbench里用到
        init_dict = {}
        vq = self.vdd @ u_V if self.q_init_val else 0 @ u_V
        vqb = 0 @ u_V if self.q_init_val else self.vdd @ u_V
        # With an equivalent-circuit model (real_cell_mode != 0) only some cells exist.
        core = getattr(self, 'sbckt_array', None)

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if core is not None and not core._should_instantiate_real_cell(row, col):
                    continue
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
        if self.choose_columnmux:
            for col in range(self.num_cols // self.mux_in):
                init_dict[f'SA_IN{col}'] = self.vdd @ u_V
                init_dict[f'SA_INB{col}'] = self.vdd @ u_V

        return init_dict

    def cell_probe(self, pin, row=None, col=None):
        """Actual access-device terminal at the cell's physical wire tap."""
        row = self.target_row if row is None else row
        col = self.target_col if col is None else col
        if pin not in ('BL', 'BLB', 'WL'):
            raise ValueError('Cell probe must be BL, BLB or WL')
        if self.w_rc and self.interconnect.cell_pin_rc:
            return f'{self.cell_inst_prefix}_{row}_{col}:{pin}_end'
        node = cell_wire_nodes(self.sbckt_array, row, col)[('BL', 'BLB', 'WL').index(pin)]
        return f'{self.arr_inst_prefix}:{node}'

    def sense_input_probe(self, pin='IN', col=None):
        col = self.target_col if col is None else col
        if pin not in ('IN', 'INB'):
            raise ValueError('Sense input must be IN or INB')
        if self.w_rc:
            return f'{self.sa_inst_prefix}_{col // self.mux_in}:{pin}_end'
        if self.choose_columnmux:
            return f'SA_IN{"B" if pin == "INB" else ""}{col // self.mux_in}'
        return self.periphery_tap('BLB' if pin == 'INB' else 'BL', col, 'sense')

    def create_testbench(self, operation, target_row, target_col):
        """
        Create a testbench for the SRAM array.
        operation: 'read' or 'write'
        target_row: Row index of the target cell
        target_col: Column index of the target cell
        """
        # Extraction consumes the actual operating point without mutating the
        # baseline configuration shared by optimizer/PVT callers.
        operating_config = copy(self.sram_config.global_config)
        operating_config.corner = self.corner
        operating_config.vdd = float(self.vdd)
        operating_config.temperature = self.temperature
        self.driver_sizes.validate_for(self.sram_config, self.sram_cell_type, self.choose_columnmux,
                                       physical_context(self.w_rc, float(self.pi_res), float(self.pi_cap), self.real_cell_mode, self.interconnect))
        self.target_row = target_row if target_row < self.num_rows else self.num_rows - 1
        self.target_col = target_col if target_col < self.num_cols else self.num_cols - 1
        # Column-mux fan-in (fixed to 2 in create_read_periphery); needed before any
        # SA_Q{col // mux_in} node name is built.
        self.mux_in = 2 if self.choose_columnmux else 1

        circuit = Circuit(self.name)
        circuit.include(getattr(self.sram_config.global_config, f"pdk_path_{self.corner}"))

        # Power supply
        circuit.V(self.power_node, self.power_node, self.gnd_node, self.vdd @ u_V)
        circuit.V(self.gnd_node, self.gnd_node, circuit.gnd, 0 @ u_V)

        # if it is a SNM test   #operation里包含字母snm，operation取决于main_sram.py里
        #  mc_testbench.run_mc_simulation的输入
        if 'snm' in operation:
            self.create_single_cell_for_snm(circuit, operation)
            # finish the circuit just return
            return circuit

        if self.sram_cell_type == 'SRAM_6T_CELL':
            # Instantiate 6T SRAM array 根据是否使用 MC 创建 SRAM Core
            if self.custom_mc:
                sbckt_array = Sram6TCoreFactory(
                    self.num_rows, self.num_cols, target_row, target_col,
                    self.sram_config.sram_6t_cell.nmos_model.value[0],
                    self.sram_config.sram_6t_cell.pmos_model.value,
                    self.sram_config.sram_6t_cell.nmos_model.value[1],
                    self.sram_config.sram_6t_cell.nmos_width.value[0],
                    self.sram_config.sram_6t_cell.pmos_width.value,
                    self.sram_config.sram_6t_cell.nmos_width.value[1],
                    self.sram_config.sram_6t_cell.length.value,
                    w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                    sweep_core=self.sweep_cell,
                    yield_mode=True,
                    real_cell_mode=self.real_cell_mode,
                    write_power_model=(operation == "write"),
                    pmos_choices = self.sram_config.sram_6t_cell.pmos_model.choices,
                    nmos_choices = self.sram_config.sram_6t_cell.nmos_model.choices,
                    param_model_file =self.sim_path + '/param_sweep_models.data',
                    # This function returns a Dict of MOS models
                    model_dict=parse_spice_models(getattr(self.sram_config.global_config, f"pdk_path_{self.corner}")),
                    q_init_val=self.q_init_val,
                    global_config=operating_config,
                    interconnect=self.interconnect,
                ).create()
            else:
                sbckt_array = Sram6TCoreFactory(
                    self.num_rows, self.num_cols,target_row, target_col,
                    self.sram_config.sram_6t_cell.nmos_model.value[0],
                    self.sram_config.sram_6t_cell.pmos_model.value,
                    self.sram_config.sram_6t_cell.nmos_model.value[1],
                    self.sram_config.sram_6t_cell.nmos_width.value[0],
                    self.sram_config.sram_6t_cell.pmos_width.value,
                    self.sram_config.sram_6t_cell.nmos_width.value[1],
                    self.sram_config.sram_6t_cell.length.value,
                    w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                    sweep_core=self.sweep_cell,
                    yield_mode=False,
                    real_cell_mode=self.real_cell_mode,
                    write_power_model=(operation == "write"),
                    pmos_choices = self.sram_config.sram_6t_cell.pmos_model.choices,
                    nmos_choices = self.sram_config.sram_6t_cell.nmos_model.choices,
                    param_model_file =self.sim_path + '/param_sweep_models.data',
                    q_init_val=self.q_init_val,
                    global_config=operating_config,
                    interconnect=self.interconnect,
                ).create()
        elif self.sram_cell_type == 'SRAM_10T_CELL':
            # Instantiate 10T SRAM array 根据是否使用 MC 创建 SRAM Core
            if self.custom_mc:
                sbckt_array = Sram10TCoreFactory(
                    self.num_rows, self.num_cols,target_row, target_col,
                    self.sram_config.sram_10t_cell.nmos_model.value[0],
                    self.sram_config.sram_10t_cell.pmos_model.value,
                    self.sram_config.sram_10t_cell.nmos_model.value[1],
                    self.sram_config.sram_10t_cell.nmos_model.value[2],
                    self.sram_config.sram_10t_cell.nmos_width.value[0],
                    self.sram_config.sram_10t_cell.pmos_width.value,
                    self.sram_config.sram_10t_cell.nmos_width.value[1],
                    self.sram_config.sram_10t_cell.nmos_width.value[2],
                    self.sram_config.sram_10t_cell.length.value,
                    w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                    sweep_core=self.sweep_cell,
                    yield_mode=True,
                    real_cell_mode=self.real_cell_mode,
                    write_power_model=(operation == "write"),
                    pmos_choices = self.sram_config.sram_10t_cell.pmos_model.choices,
                    nmos_choices = self.sram_config.sram_10t_cell.nmos_model.choices,
                    param_model_file =self.sim_path + '/param_sweep_models.data',
                    q_init_val=self.q_init_val,
                    # This function returns a Dict of MOS models
                    model_dict=parse_spice_models(getattr(self.sram_config.global_config, f"pdk_path_{self.corner}")),
                    global_config=operating_config,
                    interconnect=self.interconnect,
                ).create()
            else:
                sbckt_array = Sram10TCoreFactory(
                    self.num_rows, self.num_cols,target_row, target_col,
                    self.sram_config.sram_10t_cell.nmos_model.value[0],
                    self.sram_config.sram_10t_cell.pmos_model.value,
                    self.sram_config.sram_10t_cell.nmos_model.value[1],
                    self.sram_config.sram_10t_cell.nmos_model.value[2],
                    self.sram_config.sram_10t_cell.nmos_width.value[0],
                    self.sram_config.sram_10t_cell.pmos_width.value,
                    self.sram_config.sram_10t_cell.nmos_width.value[1],
                    self.sram_config.sram_10t_cell.nmos_width.value[2],
                    self.sram_config.sram_10t_cell.length.value,
                    w_rc=self.w_rc, pi_res=self.pi_res, pi_cap=self.pi_cap,
                    sweep_core=self.sweep_cell,
                    yield_mode=False,
                    real_cell_mode=self.real_cell_mode,
                    write_power_model=(operation == "write"),
                    pmos_choices = self.sram_config.sram_10t_cell.pmos_model.choices,
                    nmos_choices = self.sram_config.sram_10t_cell.nmos_model.choices,
                    param_model_file =self.sim_path + '/param_sweep_models.data',
                    q_init_val=self.q_init_val,
                    global_config=operating_config,
                    interconnect=self.interconnect,
                ).create()
        else:
            raise ValueError(f"Unknown SRAM cell type: {self.sram_cell_type}")

        # Add subcircuit definition to this testbench.
        circuit.subcircuit(sbckt_array)  #添加到主电路
        self.sbckt_array = sbckt_array   # data_init() asks it which cells really exist

        # Instantiate the SRAM array.
        circuit.X(sbckt_array.name, sbckt_array.name, self.power_node, self.gnd_node,
                  *[f'BL{i}' for i in range(self.num_cols)],
                  *[f'BLB{i}' for i in range(self.num_cols)],
                  *[f'WL{i}' for i in range(self.num_rows)])

        # internal node prefix in the SRAM cell
        self.arr_inst_prefix = f'X{sbckt_array.name}'
        self.cell_inst_prefix = self.arr_inst_prefix + self.heir_delimiter + sbckt_array.inst_prefix
        print(f"[DEBUG] self.arr_inst_prefix = {self.arr_inst_prefix}")
        print(f"[DEBUG] self.cell_inst_prefix = {self.cell_inst_prefix} of {self.name}")

        # Control wires first: the replica, TIME_CONTROL, decoder and periphery blocks
        # below all tap them.
        self._add_control_wires(circuit, operation)
        self._add_periphery_wires(circuit)
        # Create Replica Column 
        self.create_replica_column(circuit)
        self.create_replica_wordline(circuit)
        # Create the TIME_CONTROL block for timing control
        self.create_time_control_circuit(circuit, operation)
        self.add_cs_startup_clamp(circuit)
        # 创建译码器（输出连接到字线驱动器）
        self.create_decoder(circuit)
        
        # 创建字线驱动器（使用译码器输出作为使能）
        self.create_wl_driver(circuit, target_row)

        # Column periphery.  Every transient testbench gets the full column
        # periphery of the macro (precharge on all columns and on the replica
        # column, column mux, sense amplifiers); the write testbenches add the
        # write drivers on top.  Previously the 'write' deck had only the
        # write drivers and an RBL precharge: the bitlines started from the
        # artificial .IC state (BL=0, BLB=VDD) instead of being precharged,
        # floated at the written values after the write pulse, and did not
        # carry the sense-amplifier / mux load, so the write delay came out
        # ~30-40 % shorter than the same write inside the read&write sequence.
        if operation not in ('read', 'write', 'read&write'):
            raise ValueError(f"Invalid test type {operation}. Use 'read', 'write' or 'read&write'")
        self.create_read_periphery(circuit, target_col)
        if operation == 'write':
            self.create_write_periphery(circuit)
        elif operation == 'read&write':
            self.create_write_periphery(circuit, operation)
        elif self.driver_sizes.canonical_read:
            self.create_write_periphery(circuit, 'read')

        # Create the output D latch.  It captures SA_Q, which only exists when a read
        # periphery is present; for a pure write its input would be a floating node.
        if operation in ('read', 'read&write'):
            self.create_D_latch(circuit, target_col)
        
        # 设置目标行地址.  Each address bit is valid around every capture edge
        # (1 ns + 0.2 T + k T), so the register holds `target_row` for the access.
        # With `next_row` set, the bits alternate with a period of 2 T: the
        # register captures `target_row` at the edge that starts the access and
        # `next_row` at the next rising edge, after clock-low recovery.
        n_bits = ceil(log2(self.num_rows)) if self.num_rows > 1 else 1
        if self.select_every != 1 and (operation == 'read&write' or self.next_row is not None):
            raise ValueError("select_every applies to single 'read' / 'write' decks without next_row")
        if self.next_row is None:
            next_row = self.target_row
        else:
            if operation == 'read&write':
                raise ValueError("next_row is only supported for the 'read' and "
                                 "'write' operations (read&write re-accesses the target)")
            if not 0 <= self.next_row < self.num_rows:
                raise ValueError(f"next_row={self.next_row} outside 0..{self.num_rows - 1}")
            next_row = self.next_row
        for bit in range(n_bits):
            bit_val = (self.target_row >> bit) & 1
            next_val = (next_row >> bit) & 1
            node_name = f'A{bit}'
            if bit_val and next_val:
                circuit.PulseVoltageSource(
                    f'ADDR_{bit}', node_name, self.gnd_node,
                    initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V,
                    delay_time=1.0 @ u_ns+0.1 * self.t_period,  # 预充电开始后0.5ns
                    rise_time=self.t_rise,fall_time=self.t_fall,
                    pulse_width=0.2 * self.t_period , # 保持有效
                    period=self.t_period
                )
            elif bit_val or next_val:
                # High around every second capture edge: the even edges for a
                # target-row bit, the odd edges for a next-row bit.
                circuit.PulseVoltageSource(
                    f'ADDR_{bit}', node_name, self.gnd_node,
                    initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V,
                    delay_time=1.0 @ u_ns + 0.1 * self.t_period
                               + (0 if bit_val else 1) * self.t_period,
                    rise_time=self.t_rise, fall_time=self.t_fall,
                    pulse_width=0.2 * self.t_period,
                    period=2 * self.t_period
                )
            else:
                circuit.V(f'ADDR_{bit}', node_name, self.gnd_node, 0 @ u_V)

        # 添加时钟信号源 VCLK
        circuit.PulseVoltageSource(
            'CLK', 'clk', self.gnd_node,
            initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V,
            delay_time=1.0 @ u_ns+0.2 * self.t_period,
            rise_time=self.t_rise, fall_time=self.t_fall,
            pulse_width=0.5 * self.t_period,
            period=self.t_period
        )

        # 添加片选信号源 VCSB
        if self.select_every == 1:
            circuit.PulseVoltageSource(
                'CSB', 'csb', self.gnd_node,
                initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V,
                delay_time=1.0 @ u_ns,
                rise_time=self.t_rise, fall_time=self.t_fall,
                pulse_width=0.1 * self.t_period,
                period=self.t_period
            )
        else:
            # Selected only around every select_every-th capture edge (setup =
            # hold = 0.1 T, like the address bits); high at the other edges.
            circuit.PulseVoltageSource(
                'CSB', 'csb', self.gnd_node,
                initial_value=self.vdd @ u_V, pulsed_value=0 @ u_V,
                delay_time=1.0 @ u_ns + 0.1 * self.t_period,
                rise_time=self.t_rise, fall_time=self.t_fall,
                pulse_width=0.2 * self.t_period,
                period=self.select_every * self.t_period
            )

        if operation == 'read':
        # 添加写使能信号源 VWEB
            circuit.PulseVoltageSource(
                'WEB', 'web', self.gnd_node,
                initial_value=0 @ u_V, pulsed_value=self.vdd @ u_V,
                delay_time=1.0 @ u_ns + 0.1 * self.t_period,
                rise_time=self.t_rise, fall_time=self.t_fall,
                pulse_width=0.98 * self.t_period,
                period=self.t_period
            )
        elif operation == 'write':
        # 添加写使能信号源 VWEB
            circuit.PulseVoltageSource(
                'WEB', 'web', self.gnd_node,
                initial_value=self.vdd @ u_V, pulsed_value=0 @ u_V,
                delay_time=1.0 @ u_ns + 0.1 * self.t_period,
                rise_time=self.t_rise, fall_time=self.t_fall,
                pulse_width=0.98 * self.t_period,
                period=self.t_period
            )
        elif operation == 'read&write':
        # 添加写使能信号源 VWEB
            circuit.PulseVoltageSource(
                'WEB', 'web', self.gnd_node,
                initial_value=self.vdd @ u_V, pulsed_value=0 @ u_V,
                delay_time=1.0 @ u_ns + 0.1 * self.t_period,
                rise_time=self.t_rise, fall_time=self.t_fall,
                pulse_width=0.98 * self.t_period,
                period=2*self.t_period
            )

        return circuit
