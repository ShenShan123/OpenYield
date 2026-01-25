import os
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA 
# Only for yield analysis
from utils import (  # type: ignore
    parse_mc_measurements, generate_mc_statistics,
    save_mc_results, process_simulation_data,
    parse_spice_models, write_spice_models
)
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench  # type: ignore
from sram_compiler.param_sweep_data.sweep_config import SWEEP_CONFIGS
import numpy as np
import csv
from PySpice.Spice.Netlist import SubCircuitFactory
from math import ceil, log2

class Sram6TCoreMcTestbench(Sram6TCoreTestbench):
    def __init__(self, sram_config,
                 w_rc=False, pi_res=10 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 vth_std=0.05, custom_mc=False,sweep_cell=False,sweep_precharge=False,sweep_senseamp=True,sweep_wordlinedriver=False,
                 sweep_columnmux=False,sweep_writedriver=False,sweep_decoder=False,corner='TT',choose_columnmux=True,
                 q_init_val=0, sim_path='sim'):
        """
               蒙特卡洛测试平台初始化
               参数:
                   sram_config: SRAM配置对象（包含所有设计参数）
                   w_rc: 是否添加RC网络
                   pi_res: PI模型电阻
                   pi_cap: PI模型电容
                   vth_std: 阈值电压标准差
                   custom_mc: 是否使用自定义MC参数
                   q_init_val: 初始Q值
                   sim_path: 仿真结果保存路径
               """
        super().__init__(#父类
            sram_config,
            w_rc, pi_res, pi_cap,
            custom_mc, sweep_cell,sweep_precharge,sweep_senseamp,sweep_wordlinedriver,sweep_columnmux,sweep_writedriver,sweep_decoder,corner,choose_columnmux,q_init_val,
        )
        self.choose_columnmux=choose_columnmux
        self.corner=corner
        self.sweep_decoder=sweep_decoder
        self.sweep_writedriver=sweep_writedriver
        self.sweep_columnmux=sweep_columnmux
        self.sweep_wordlinedriver=sweep_wordlinedriver
        self.sweep_senseamp=sweep_senseamp
        self.sweep_precharge=sweep_precharge
        self.sweep_cell =sweep_cell
        self.sram_config = sram_config
        self.vth_std = vth_std
        num_rows = sram_config.global_config.num_rows
        num_cols = sram_config.global_config.num_cols
        self.name = f'SRAM_6T_CORE_{num_rows}x{num_cols}_MC_TB' #根据行列数设置测试平台名称
        self.sim_path = sim_path
        os.makedirs(self.sim_path, exist_ok=True)

    def create_mc_model_file(self):
        """Create temporary model file with Monte Carlo variations创建蒙特卡洛模型文件"""
        #pdk_path = self.sram_config.global_config.pdk_path
        pdk_path = getattr(self.sram_config.global_config, f"pdk_path_{self.corner}")
        model_dict = parse_spice_models(pdk_path)

        for m in model_dict.keys():

            # Get parameters for the model_name
            param_dict = model_dict[m]['parameters']

            for param in param_dict.keys():
                # substitute the default values with user-defined parameters
                if param in ['vth0', 'u0', 'voff']:
                    val = param_dict[param]
                    param_dict[param] = f"{{AGAUSS({val}, {abs(val) * self.vth_std:.5f}, 1)}}"
                    #为关键参数(vth0, u0, voff)添加高斯分布随机变量
        # Generate a modified model lib 生成一个修改过的模型库
        temp_model_path = os.path.join(self.sim_path, 'tmp_mc.spice')
        # Write the model cards back to temporary file
        write_spice_models(model_dict, temp_model_path)

        return temp_model_path

    def create_testbench(self, operation, target_row=0, target_col=0):#定义子类里的create_testbench函数
        """Create testbench with Monte Carlo models"""
        circuit = super().create_testbench(operation, target_row, target_col)
        #调用父类里的create_testbench函数
        # Standard MC needs a model lib with variables,
        # otherwise, process parameters are defined by user
        if not self.custom_mc:  #不需要自定义的MC时
            # Replace original included model lib with new path 替换为包含随机变量的模型文件
            circuit._includes[0] = self.create_mc_model_file()

        return circuit

    def add_meas_and_print(self, simulator, init_cond, operation):
        # Internal nodes' names of the target cell
        target_node_q = self.cell_inst_prefix + f'_{self.target_row}_{self.target_col}{self.heir_delimiter}Q'
        target_node_qb = self.cell_inst_prefix + f'_{self.target_row}_{self.target_col}{self.heir_delimiter}QB'
        #获取目标单元的内部节点名称（Q 和 QB）
        if operation == 'hold_snm' or operation == 'read_snm' or operation == 'write_snm':
            # Initial V(BL) and V(BLB) for the  cell
            init_cond = {}
            init_cond[f'BL'] = self.vdd @ u_V   #设置位线初始电压
            init_cond[f'BLB'] = self.vdd @ u_V
            simulator.initial_condition(**init_cond)    #读入初始电压设置
            simulator.measure('DC', 'MAXVD', 'MAX V(VD)')   #测量最大对角线电压
            simulator.measure('DC', operation.upper(), f"PARAM='1/sqrt(2)*MAXVD'")  #计算静态噪声容限
            # Add print for SNM
            simulator.circuit.raw_spice += \
                f'.PRINT DC FORMAT=NOINDEX {{U}} V(V1) V(V2)\n'

        # The read operation
        elif operation == 'read':
            # .ic conditions
            for col in range(self.num_cols):
                # Initial V(BL) and V(BLB) for all columns
                init_cond[f'BL{col}'] = 0 @ u_V
                init_cond[f'RBL'] = 0 @ u_V
                init_cond[f'SA_Q{col}'] = 0 @ u_V
                init_cond[f'SA_QB{col}'] = self.vdd @ u_V

            # 计算地址位数
            n_bits = ceil(log2(self.num_rows)) if self.num_rows > 1 else 1
            # 为地址锁存器节点设置初始条件
            for bit in range(n_bits):  
                init_cond[f'A_dff{bit}'] = 0 @ u_V
            
            # 为其他控制信号设置初始条件
            init_cond['we'] = self.vdd @ u_V  # we初始化为高电平
            init_cond['cs_bar'] = self.vdd @ u_V  # cs_bar初始化为高电平

            simulator.initial_condition(**init_cond)

            # Measurements for precharge delay (TPRCH), defined by the time from PRE assertion to V(BL)=0.9*VDD
            #预充延迟（TPRCH）的测量，由预充断言到V（BL）的时间定义=0.9*VDD
            simulator.measure(
                'TRAN', 'TPRCH',
                f'TRIG V(PRE)={self.half_vdd} FALL=1 ' +
                f'TARG V(BL{self.target_col})={float(self.vdd) * 0.9} RISE=1')  # modified for Xyce

            # Measurements for wl driver delay (TWLDRV), defined as the time from the WLE assertion to V(WL)=VDD/2
            #测量WLE驱动延迟（TWLDRV），定义为从译码器输出到V(wl)=VDD/2的时间
            simulator.measure(
                'TRAN', 'TDECODER',
                f'TRIG V(A_dff0)={self.half_vdd} RISE=1 ' +
                f'TARG V(DEC_WL{self.target_row})={self.half_vdd} RISE=1')
            simulator.measure(
                'TRAN', 'TWLDRV',
                f'TRIG V(wl_en)={self.half_vdd} RISE=1 ' +
                f'TARG V(WL{self.target_row})={self.half_vdd} RISE=1')

            # Add measurements for read delay (TREAD),读延迟
            # which is defined as the time from the WL rise to BL swing to VDD/2
            simulator.measure(
                'TRAN', 'TWL',
                f'WHEN V(WL{self.target_row})={self.half_vdd} RISE=1 ')  # modified for Xyce
            # Define minimum Vswing = 250mV
            vswing = 0.25
            simulator.measure(
                'TRAN', 'TBL',
                f"WHEN V(BL{self.target_col})='V(BLB{self.target_col})-{vswing}' FALL=1")
            simulator.measure('TRAN', 'TSWING', f"PARAM='TBL-TWL'")

            # Measurements for SA delay (TSA), defined as the time from the SAE assertion to V(Q)=VDD/2
            #SA延迟（TSA）的测量，定义为从SAE断言到V(Q)=VDD/2的时间
            simulator.measure(
                'TRAN', 'TSA',
                f'TRIG V(s_en)={self.half_vdd} RISE=1 ' +
                f'TARG V(SA_Q{self.target_col // self.mux_in})={float(self.vdd) * 0.01} FALL=1')
            
            simulator.measure(
                'TRAN', 'Ts_en',
                f'TRIG V(S_EN)=0.01 RISE=2 ' +
                f'TARG V(S_EN)=0.5 RISE=1')

            # Add measurements for average power, static power and dynamic power    测量功耗(平均、动态、静态)
            simulator.measure(
                'TRAN', 'PAVG',
                f'AVG {{V(VDD)*I(VVDD)}} FROM={float(0.0 @ u_ns)} TO={float(10.0 @ u_ns)}'
            )
            simulator.measure(
                'TRAN', 'PDYN',
                f'AVG {{V(VDD)*I(VVDD)}} FROM={float(4.0 @ u_ns)} TO={float(6.0 @ u_ns)}'
            )
            simulator.measure(
                'TRAN', 'PSTC',
                f'AVG {{V(VDD)*I(VVDD)}} FROM={float(6.0 @ u_ns)} ' +
                f'TO={float(10.0 @ u_ns)}'
            )

             # Add additional print statements for clock and control signals
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(CLK) V(CLK_BUF) V(CLK_BAR) V(CSB)' +\
                f' V(CS_BAR) V(CS) V(WEB) V(WE_BAR) V(WE) V(GATED_CLK_BUF) V(GATED_CLK_BAR) V(WL_EN) V(RWL)\n'
            
            # Add print statement for address and other signals
            address_signals = ' '.join([f'V(A{i}) V(A_DFF{i})' for i in range(ceil(log2(self.num_rows)))])
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX {address_signals} V(RBL) V(RBL_DELAY) V(RBL_DELAY_BAR) V(W_EN) V(PRE)\n'
            
            # Add print for read operation
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(S_EN) V(WL{self.target_row}) ' + \
                f'V(BL{self.target_col}) V(BLB{self.target_col}) ' + \
                f'V({target_node_q}) V({target_node_qb}) \n'
            if self.choose_columnmux:
                simulator.circuit.raw_spice += \
                    f'.PRINT TRAN V(SA_IN{self.target_col // self.mux_in}) ' + \
                    f'V(SA_INB{self.target_col // self.mux_in})\n'      
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN V(SA_Q{self.target_col // self.mux_in}) ' + \
                f'V(SA_QB{self.target_col // self.mux_in})' + \
                f'V(OUT)\n'
            
        # The write operation
        elif operation == 'write':
            # .ic conditions
            for col in range(self.num_cols):
                # Initial V(BL) and V(BLB) for all columns
                init_cond[f'BL{col}'] = 0 @ u_V
                init_cond[f'BLB{col}'] = self.vdd @ u_V
                init_cond[f'RBL'] = 0 @ u_V
                init_cond[f'DIN_dff{col}'] = 0 @ u_V

             # 计算地址位数
            n_bits = ceil(log2(self.num_rows)) if self.num_rows > 1 else 1
            # 为地址锁存器节点设置初始条件
            for bit in range(n_bits):  
                init_cond[f'A_dff{bit}'] = 0 @ u_V
            
            # 为其他控制信号设置初始条件
            init_cond['we'] = self.vdd @ u_V  # we初始化为高电平
            init_cond['cs_bar'] = self.vdd @ u_V  # cs_bar初始化为高电平

            simulator.initial_condition(**init_cond)
            
            #字线驱动延迟
            # Measurements for wl driver delay (TWLDRV), defined as the time from the WLE assertion to V(WL)=VDD/2
            simulator.measure(
                'TRAN', 'TDECODER',
                f'TRIG V(A_dff0)={self.half_vdd} RISE=1 ' +
                f'TARG V(DEC_WL{self.target_row})={self.half_vdd} RISE=1')
            simulator.measure(
                'TRAN', 'TWLDRV',
                f'TRIG V(DEC_WL{self.target_row})={self.half_vdd} RISE=1 ' +
                f'TARG V(WL{self.target_row})={self.half_vdd} RISE=1')
            #写驱动延迟
            # Measurements for write driver delay (TWDRV), defined as the time from the WE assertion to V(BL)=VDD/2
            simulator.measure(
                'TRAN', 'TWDRV',
                f'TRIG V(w_en)={self.half_vdd} RISE=1 ' +
                f'TARG V(BL{self.target_col})={self.half_vdd} RISE=1')
            #写延迟
            # Measurements for write delay (TWRITE_Q/QB),
            # which is defined as the time from the WL rise to data Q rise to 90% VDD.
            simulator.measure(
                'TRAN', 'TWRITE_Q',
                f'TRIG V(WL{self.target_row})={self.half_vdd} RISE=1',
                f"TARG V({target_node_q})={float(self.vdd) * 0.9:.2f} RISE=1")
            simulator.measure(
                'TRAN', 'TWRITE_QB',
                f'TRIG V(WL{self.target_row})={self.half_vdd} RISE=1',
                f"TARG V({target_node_qb})={float(self.vdd) * 0.1:.2f} FALL=1")

            # Add measurements for average power, static power and dynamic power    功耗
            simulator.measure(
                'TRAN', 'PAVG',
                f'AVG {{V(VDD)*I(VVDD)}} FROM={float(0.0 @ u_ns)} TO={float(10.0 @ u_ns)}'
            )
            simulator.measure(
                'TRAN', 'PDYN',
                f'MIN {{V(VDD)*I(VVDD)}} FROM={float(4.0 @ u_ns)} TO={float(6.0 @ u_ns)}'
            )
            simulator.measure(
                'TRAN', 'PSTC',
                f'AVG {{V(VDD)*I(VVDD)}} FROM={float(6.0 @ u_ns)} ' +
                f'TO={float(10.0 @ u_ns)}'
            )
            # Add print for write operation
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(we) V(WL{self.target_row}) V(BL{self.target_col})' + \
                f' V(BLB{self.target_col}) V({target_node_q}) V({target_node_qb})\n'
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(gated_clk_bar) V(DIN0) V(DIN_dff0)' + \
                f' V(w_en) V(RBL) V(RBL_DELAY_BAR)\n'
            
        elif operation == 'read&write':
            # .ic conditions
            for col in range(self.num_cols):
                # Initial V(BL) and V(BLB) for all columns
                init_cond[f'BL{col}'] = 0 @ u_V
                init_cond[f'BLB{col}'] = self.vdd @ u_V
                init_cond[f'RBL'] = 0 @ u_V
                init_cond[f'OUT'] = 0 @ u_V
                init_cond[f'SA_Q{col}'] = 0 @ u_V
                init_cond[f'SA_QB{col}'] = self.vdd @ u_V
                init_cond[f'DIN_dff{col}'] = 0 @ u_V

             # 计算地址位数
            n_bits = ceil(log2(self.num_rows)) if self.num_rows > 1 else 1
            # 为地址锁存器节点设置初始条件
            for bit in range(n_bits):  
                init_cond[f'A_dff{bit}'] = 0 @ u_V
            
            # 为其他控制信号设置初始条件
            init_cond['we'] = self.vdd @ u_V  # we初始化为高电平
            init_cond['cs_bar'] = self.vdd @ u_V  # cs_bar初始化为高电平

            simulator.initial_condition(**init_cond)

             # Add print for read operation
             # Add additional print statements for clock and control signals
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(CLK) V(CLK_BUF) V(CLK_BAR) V(CSB)' +\
                f' V(CS_BAR) V(CS) V(WEB) V(WE_BAR) V(WE) V(GATED_CLK_BUF) V(GATED_CLK_BAR) V(WL_EN) V(RWL)\n'
            
            # Add print statement for address and other signals
            address_signals = ' '.join([f'V(A{i}) V(A_DFF{i})' for i in range(ceil(log2(self.num_rows)))])
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX {address_signals} V(RBL) V(RBL_DELAY) V(RBL_DELAY_BAR) V(W_EN) V(PRE)\n'
            
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(S_EN) V(WL{self.target_row}) ' + \
                f'V(BL{self.target_col}) V(BLB{self.target_col}) ' + \
                f'V({target_node_q}) V({target_node_qb}) \n'
            if self.choose_columnmux:
                simulator.circuit.raw_spice += \
                    f'.PRINT TRAN V(SA_IN{self.target_col // self.mux_in}) ' + \
                    f'V(SA_INB{self.target_col // self.mux_in})\n'      
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN V(SA_Q{self.target_col // self.mux_in}) ' + \
                f'V(SA_QB{self.target_col // self.mux_in})\n'
            
            # Add print for write operation
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX  ' + \
                f'V(DIN0) V(DIN_dff0) V(OUT)\n'

            simulator.measure(
                'TRAN', 'TVOUT_PERIOD',
                f'TRIG V(OUT)={self.half_vdd} RISE=1 ' +
                f'TARG V(OUT)={self.half_vdd} RISE=2')
            simulator.measure(
                'TRAN', 'PAVG',
                f'AVG {{V(VDD)*I(VVDD)}} FROM={float(0.0 @ u_ns)} TO={12*float(self.t_period)}'
            )
        else:
            raise ValueError(f"Invalid operation: {operation}")

    # def add_xyce_options(self, circuit, mc_runs, operation):
    #     """ Add options for Xyce """
    #     pass

    def add_analysis(self, circuit, operation, num_mc):
        """ Add .DC / .TRAN analysis DC 扫描/瞬态分析"""
        if 'snm' in operation:
            u_tmp = self.vdd / np.sqrt(2)
            circuit.raw_spice += \
                f'.DC U -{u_tmp:.2f} {u_tmp:.2f} 0.001\n'
        else:
            if operation == 'read&write':
                circuit.raw_spice += \
                    f'.TRAN {float(self.t_step):.4e} {12*float(self.t_period):.4e}\n'
            else:
                circuit.raw_spice += \
                    f'.TRAN {float(self.t_step):.4e} {2*float(self.t_period):.4e}\n'
            # Timing interval option is set only in .TRAN analysis.
            circuit.raw_spice += \
                f'.OPTIONS OUTPUT INITIAL_INTERVAL={float(self.t_step):.4e}\n'

        # Whether we use custom MC
        if self.custom_mc:
            # Sweep the each row of the `table`
            circuit.raw_spice += \
                f'.STEP data=table\n'
         # Whether we use sweep
        if self.sweep_cell:
            circuit.raw_spice += \
                f'.STEP data=SRAM_6T_CELL\n'
        if self.sweep_precharge:
            circuit.raw_spice += \
                f'.STEP data=PRECHARGE\n'
        if self.sweep_senseamp:
                circuit.raw_spice += \
                f'.STEP data=SENSEAMP\n'
        if self.sweep_wordlinedriver:
            circuit.raw_spice += \
                f'.STEP data=WORDLINEDRIVER\n'
        if self.sweep_columnmux:
            circuit.raw_spice += \
                f'.STEP data=COLUMNMUX\n'
        if self.sweep_writedriver:
            circuit.raw_spice += \
                f'.STEP data=WRITEDRIVER\n'
        if self.sweep_decoder:
            circuit.raw_spice += \
                f'.STEP data=DECODER\n'
        if not self.sweep_cell and not self.sweep_precharge and not self.sweep_senseamp and not self.sweep_wordlinedriver and not self.sweep_columnmux and not self.sweep_writedriver and not self.sweep_decoder:
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
            生成工艺参数数据表
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
                                f'.param {param}_{self.sram_config.sram_6t_cell.pmos_model.value}_{mos}_{row:d}_{col:d}=0.0\n'
                            # Data table head
                            self.table_head += f'{param}_{self.sram_config.sram_6t_cell.pmos_model.value}_{mos}_{row:d}_{col:d} '
                        elif mos in ['PGL','PGR']:
                            circuit.raw_spice += \
                                f'.param {param}_{self.sram_config.sram_6t_cell.nmos_model.value[1]}_{mos}_{row:d}_{col:d}=0.0\n'
                            # Data table head
                            self.table_head += f'{param}_{self.sram_config.sram_6t_cell.nmos_model.value[1]}_{mos}_{row:d}_{col:d} '
                        else:
                            # Parameter definitions for NMOS
                            circuit.raw_spice += \
                                f'.param {param}_{self.sram_config.sram_6t_cell.nmos_model.value[0]}_{mos}_{row:d}_{col:d}=0.0\n'
                            # Data table head
                            self.table_head += f'{param}_{self.sram_config.sram_6t_cell.nmos_model.value[0]}_{mos}_{row:d}_{col:d} '

                        num_params += 1
        # Just for debugging
        if vars is None:
            vars = [0.4106, 0.045, -0.13,  # PGL
                    0.4106, 0.045, -0.13,  # PGR
                    0.4106, 0.045, -0.13,  # PDL
                    -0.3842, 0.02, -0.126,  # PUL
                    0.4106, 0.045, -0.13,  # PDR
                    -0.3842, 0.02, -0.126]  # PUR

            if operation == 'read' or operation == 'write':
                vars = np.array([vars * self.num_rows * self.num_cols])
                vars = np.repeat(vars, num_mc, axis=0)
            else:
                vars = np.array([vars])
                vars = np.repeat(vars, num_mc, axis=0)

            print(f"[DEBUG] Generated vars.shape={vars.shape}")
        else:
            assert num_mc == vars.shape[0], f"num_mc={num_mc} mismatches {vars.shape[0]} row number in the data table"
            print(f"[DEBUG] Input vars.shape={vars.shape}")

        assert len(vars.shape) == 2
        assert num_params == vars.shape[
            1], f'num_params={num_params} mismatches {vars.shape[1]} column number in the data table'

        table_content += "\n".join([
            "+ " + " ".join([f"{x:.4f}" for x in row])
            for row in vars
        ])

        # Generate and run Xyce netlist
        table_path = os.path.join(self.sim_path, f'mc_{operation}_table.data')
        with open(table_path, 'w') as f:
            f.write(self.table_head + table_content)
        circuit.include(table_path)
        print(f'[DEBUG] Data table has been saved to {table_path}')

    def gen_param_sweep_generic(self, circuit: SubCircuitFactory, module_name: str,
                                param_names: list, csv_filename: str, operation: str,
                                ):
        """通用参数扫描方法
        Args:
            circuit: 电路
            module_name: 模块名称（用于数据表标识）
            param_names: 参数名称列表
            csv_filename: CSV文件名
            operation: 操作类型
        """
        csv_path = f"sram_compiler/param_sweep_data/{csv_filename}"
        if csv_path is None:
            raise ValueError("必须指定 csv_path 参数")
        
        vars = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            # 检查表头是否包含需要的列
            missing = set(param_names) - set(reader.fieldnames or [])
            if missing:
                raise KeyError(f"CSV 缺少列: {missing}")
            for row in reader:
                vars.append([float(row[p]) for p in param_names])

         # 构建数据表内容
        table_content = f".data {module_name}\n"
        table_content += "+ " + " ".join(param_names) + "\n"

        # 在网表中添加参数定义
        for param in param_names:
            circuit.raw_spice += f'.param {param}=0.0\n'
        
        # 验证参数维度
        expected_len = len(param_names)
        for i, row in enumerate(vars):
            assert len(row) == expected_len, f"Row {i} has {len(row)} values, expected {expected_len}"

        for row in vars:
            # 使用科学计数法格式化
            formatted_row = [f"{x:.3e}" for x in row]
            table_content += "+ " + " ".join(formatted_row) + "\n"

        # 生成并保存参数表文件
        include_path = os.path.join(self.sim_path, f'param_sweep_{module_name}.data')
        with open(include_path, 'w') as f:
            f.write(table_content)
    
        # 在网表中包含参数表
        circuit.include(include_path)
        print(f'[DEBUG] Parameter sweep table saved to {include_path}')


    def run_mc_simulation(self, operation='read', target_row=0, target_col=0, mc_runs=100, temperature=27,vars=None):
        """Run Xyce Monte Carlo simulation"""
        circuit = self.create_testbench(operation, target_row, target_col)
        simulator = circuit.simulator(
        temperature=temperature,           # 通过 **kwargs 传递
        nominal_temperature=27    # 通过 **kwargs 传递
        )

        # Add some Xyce related commands
        self.add_analysis(simulator.circuit, operation, mc_runs)

        # Add measurements according to the operation
        self.add_meas_and_print(simulator, self.data_init(), operation)

        # Add process parameters
        if self.custom_mc:
            self.gen_process_params(simulator.circuit, operation, vars=vars, num_mc=mc_runs)
        
        # 定义扫描开关与配置键名的对应关系
        # 元组结构: (Config Key, Boolean Switch)
        # 这里的 Boolean Switch 来自 __init__ 中定义的 self.sweep_xxx
        check_list = [
            ('cell', self.sweep_cell),
            ('precharge', self.sweep_precharge),
            ('senseamp', self.sweep_senseamp),
            ('wordlinedriver', self.sweep_wordlinedriver),
            ('columnmux', self.sweep_columnmux),
            ('writedriver', self.sweep_writedriver),
            ('decoder', self.sweep_decoder)
        ]

        # 遍历检查并在需要时调用通用方法
        for key, is_enabled in check_list:
            if is_enabled:
                # 从外部文件导入的配置中获取参数
                if key not in SWEEP_CONFIGS:
                    print(f"[WARNING] 配置 '{key}' 在 sram_mc_config.py 中未定义，跳过。")
                    continue            
                cfg = SWEEP_CONFIGS[key]        
                self.gen_param_sweep_generic(
                    circuit=simulator.circuit,
                    operation=operation,
                    module_name=cfg['name'],
                    param_names=cfg['params'],
                    csv_filename=cfg['csv'],
                )

        print("[DEBUG] Printing generated netlists...")
        print(simulator)
        init = '_q1' if self.q_init_val > 0 else ''

        # Generate and run Xyce netlist
        tb_path = os.path.join(
            self.sim_path,
            f'mc_{operation}_{self.num_rows}x{self.num_cols}_rc{self.w_rc:d}{init}_tb.sp')

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
                raise RuntimeError(
                    f"Xyce error:\n{result.stderr}, please check the log file {tb_path.replace('.sp', '.lis')}.")
            else:
                print("[DEBUG] Simulation run successfully.")
        finally:
            # 根据操作类型选择要在波形图中显示的信号
            if operation == 'read':
                selected_columns = [
                    f'V(S_EN)',
                    f'V(WL{target_row})',
                    f'V(BL{target_col})',
                    f'V(BLB{target_col})',
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:Q)',
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:QB)',
                    f'V(SA_Q{target_col})',
                    f'V(SA_QB{target_col})',
                ]
            elif operation == 'write':
                selected_columns = [
                    f'V(WE)',
                    f'V(WL{target_row})',
                    f'V(BL{target_col})',
                    f'V(BLB{target_col})',
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:Q)',
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:QB)',
                ]
            elif operation == 'read&write':
                selected_columns = [
                    f'V(OUT)'
                ]
            elif operation in ['hold_snm', 'write_snm', 'read_snm']:
                selected_columns = [
                    f'{{U}}',f'V(V1)',f'V(V2)'
                ]  
            # plot waveforms of signals in `selected_columns`
            process_simulation_data(
                prn_path=tb_path + '.prn',
                num_mc=mc_runs,
                output=f"{self.sim_path}/mc_{operation}_waveform.png",
                selected_columns=selected_columns
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
            # Generate statistics
            stats = generate_mc_statistics(mc_df)
            # Save results
            save_mc_results(
                mc_df, stats,
                data_file=tb_path.replace('.sp', '.data.csv'),
                stats_file=tb_path.replace('.sp', '.stats.csv')
            )
            if operation == 'read' or operation == 'write':
                # Calculate and print half clock cycle time (1/2CLK)
                # Read the statistics CSV file
                stats_csv_path = tb_path.replace('.sp', '.stats.csv')
                import pandas as pd
                
                # Load the statistics data
                stats_df = pd.read_csv(stats_csv_path, index_col=0)
                
                # Define the parameters we need
                params = ['TSA', 'TSWING', 'TS_EN', 'TWLDRV']
                
                # Initialize sum for 1/2CLK calculation
                half_clk_sum = 0.0
                
                # Extract mean and std for each parameter and add to sum
                for param in params:
                    if param in stats_df.index:
                        mean_val = stats_df.loc[param, 'mean']
                        std_val = stats_df.loc[param, 'std']
                        #如果标准差大于0.1ns，则只加均值，否则均值和标准差都加上
                        if std_val > 0.1e-9 or pd.isna(std_val) or std_val == float('inf') or std_val == float('-inf'):  # 0.1ns = 0.1e-9 seconds
                            half_clk_sum += mean_val
                            print(f"[DEBUG] {param} - Mean: {mean_val:.3e}, Std: {std_val:.3e} (>0.1ns or (mc=1 so no std, only mean used)")
                        else:
                            half_clk_sum += mean_val + std_val
                            print(f"[DEBUG] {param} - Mean: {mean_val:.3e}, Std: {std_val:.3e}")
                    else:
                        print(f"[WARNING] Parameter {param} not found in statistics")
                
                # Print the calculated 1/2CLK value
                print(f"[INFO] Calculated 1/2CLK : {half_clk_sum:.3e}")
                print(f"[INFO] CLK(min) in this size and PVT : {((half_clk_sum+0.1e-9)*2):.3e}")

            data_csv_path=tb_path.replace('.sp', '.data.csv')
           # Reture the performance metrics for yield analysis and sizing optimization
            if operation == 'write' or operation == 'read&write' or operation == 'read':
                return data_csv_path
            elif operation == 'hold_snm':
                return mc_df['HOLD_SNM'].to_numpy()
            elif operation == 'read_snm':
                return mc_df['READ_SNM'].to_numpy()
            elif operation == 'write_snm':
                return mc_df['WRITE_SNM'].to_numpy()
            else:
                raise KeyError(f"Unkonwn operation {operation}")
