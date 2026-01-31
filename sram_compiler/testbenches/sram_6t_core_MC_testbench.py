import os
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA 
# Only for yield analysis
from utils import (  # type: ignore
    parse_mc_measurements, generate_mc_statistics,
    save_mc_results, process_simulation_data,
    parse_spice_models, write_spice_models
)
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench  # type: ignore
from sram_compiler.config_yaml.sweep_config import SWEEP_CONFIGS
import numpy as np
from PySpice.Spice.Netlist import SubCircuitFactory
from math import ceil, log2

class Sram6TCoreMcTestbench(Sram6TCoreTestbench):
    def __init__(self, sram_config, sram_cell_type="SRAM_6T_CELL",
                 w_rc=False, pi_res=10 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 vth_std=0.05, custom_mc=False,sweep_cell=False,sweep_precharge=False,sweep_senseamp=True,sweep_wordlinedriver=False,
                 sweep_columnmux=False,sweep_writedriver=False,sweep_decoder=False,corner='TT',choose_columnmux=True,
                 q_init_val=0, sim_path='sim'):
        """
               蒙特卡洛测试平台初始化
               参数:
                   sram_config: SRAM配置对象(包含所有设计参数)
                   w_rc: 是否添加RC网络
                   pi_res: PI模型电阻
                   pi_cap: PI模型电容
                   vth_std: 阈值电压标准差
                   custom_mc: 是否使用自定义MC参数
                   q_init_val: 初始Q值
                   sim_path: 仿真结果保存路径
               """
        super().__init__(#父类
            sram_config, sram_cell_type,
            w_rc, pi_res, pi_cap,
            custom_mc, sweep_cell,sweep_precharge,sweep_senseamp,sweep_wordlinedriver,sweep_columnmux,sweep_writedriver,sweep_decoder,
            corner,choose_columnmux,q_init_val,sim_path
        )
        self.sram_cell_type=sram_cell_type
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
            if self.sram_cell_type == 'SRAM_6T_CELL':
                circuit.raw_spice += \
                    f'.STEP data=SRAM_6T_CELL\n'
            elif self.sram_cell_type == 'SRAM_10T_CELL':
                circuit.raw_spice += \
                    f'.STEP data=SRAM_10T_CELL\n'
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
        if not self.custom_mc and not self.sweep_cell and not self.sweep_precharge and not self.sweep_senseamp and not self.sweep_wordlinedriver and not self.sweep_columnmux and not self.sweep_writedriver and not self.sweep_decoder:
            # Use build-in sampling method in Xyce
            circuit.raw_spice += \
                f'.SAMPLING useExpr=true\n.options samples numsamples={num_mc}\n'

        print(f"[DEBUG] Custom_MC={self.custom_mc}, numsamples={num_mc}")

    def get_table_head(self):
        return self.table_head

    def gen_process_params(self, circuit: SubCircuitFactory,
                           operation: str, num_mc: int,
                           vars: np.array = None):
        """ 
        统一生成工艺参数数据表 (支持 6T 和 10T)
        """
        param_names = ['vth0', 'u0', 'voff']
        self.table_head = '.data table\n+ '
        table_content = '\n'
        num_params = 0

        # --- 1. 根据 Cell 类型配置差异化参数 ---
        if self.sram_cell_type == 'SRAM_10T_CELL':
            cell_config = self.sram_config.sram_10t_cell
            # 晶体管顺序不能变
            mos_names = ['PGL', 'PGR', 'PDL1', 'PDL2', 'PUL', 'PDR1', 'PDR2', 'PUR', 'FD_L', 'FD_R']
            # 定义模型映射集合
            pmos_set = {'PUL', 'PUR'}
            pg_nmos_set = {'PGL', 'PGR'}
            fd_nmos_set = {'FD_L', 'FD_R'} # 10T 特有
            # 默认调试数据 (30个值)
            default_vars_list = [
                0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, # N
                -0.3842, 0.02, -0.126, 0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, -0.3842, 0.02, -0.126, # P/N/N/P
                0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13 # FD
            ]
        else: # 默认为 SRAM_6T_CELL
            cell_config = self.sram_config.sram_6t_cell
            mos_names = ['PGL', 'PGR', 'PDL', 'PUL', 'PDR', 'PUR']
            pmos_set = {'PUL', 'PUR'}
            pg_nmos_set = {'PGL', 'PGR'}
            fd_nmos_set = set() # 空集合
            # 默认调试数据 (18个值)
            default_vars_list = [
                0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13, # PGL, PGR, PDL
                -0.3842, 0.02, -0.126, 0.4106, 0.045, -0.13, -0.3842, 0.02, -0.126 # PUL, PDR, PUR
            ]

        # --- 2. 统一生成参数逻辑 ---
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if ('snm' in operation) and (row > 0 or col > 0):
                    continue
                
                for mos in mos_names:
                    # 确定当前晶体管使用的模型值字符串
                    if mos in pmos_set:
                        model_val = cell_config.pmos_model.value
                    elif mos in pg_nmos_set:
                        model_val = cell_config.nmos_model.value[1] # PG 位于 index 1
                    elif mos in fd_nmos_set:
                        model_val = cell_config.nmos_model.value[2] # FD 位于 index 2
                    else:
                        model_val = cell_config.nmos_model.value[0] # PD 位于 index 0 (默认)

                    for param in param_names:
                        # 写入 SPICE 参数定义
                        param_def = f'{param}_{model_val}_{mos}_{row:d}_{col:d}'
                        circuit.raw_spice += f'.param {param_def}=0.0\n'
                        self.table_head += f'{param_def} '
                        num_params += 1

        # --- 3. 处理 Vars 数据 ---
        if vars is None:
            vars = default_vars_list
        vars = np.array(vars)

        # 根据行列数扩展
        if operation in ['read', 'write', 'read&write']:
            vars = np.tile(vars, self.num_rows * self.num_cols)
        
        vars = np.array([vars]) # 转为二维
        vars = np.repeat(vars, num_mc, axis=0) # 扩展 MC 次数
        print(f"[DEBUG] Generated vars.shape={vars.shape}")

        assert len(vars.shape) == 2
        assert num_params == vars.shape[1], f'Cols mismatch: expected {num_params}, got {vars.shape[1]}'

        # 格式化数据表
        table_content += "\n".join([
            "+ " + " ".join([f"{x:.4f}" for x in row])
            for row in vars
        ])

        # 保存并包含
        table_path = os.path.join(self.sim_path, f'mc_{operation}_table.data')
        with open(table_path, 'w') as f:
            f.write(self.table_head + table_content)
        circuit.include(table_path)
        print(f'[DEBUG] Data table has been saved to {table_path}')


    def _extract_sweep_rows(self, module_config, module_name, target_params):
            sweep_lists = []
            for req_param in target_params:
                matched = False
                for yaml_key, param_obj in module_config.parameters.items():
                    yaml_names = param_obj.instance_names if isinstance(param_obj.instance_names, list) else [param_obj.instance_names]
                    yaml_sweeps = param_obj.value_sweep if isinstance(param_obj.instance_names, list) else [param_obj.value_sweep]
                    
                    for idx, inst_name in enumerate(yaml_names):
                        if req_param == f"{yaml_key}_{inst_name}" or req_param == yaml_key:
                            val = yaml_sweeps[idx]
                            sweep_lists.append(val if isinstance(val, list) else [val])
                            matched = True
                            break
                    if matched: break
                if not matched: raise ValueError(f"模块 {module_name} 中未找到参数 {req_param}")

            # 长度校验与自动对齐
            lens = [len(x) for x in sweep_lists]
            max_len = max(lens)
            if len(set(lens)) > 1:
                sweep_lists = [x * max_len if len(x) == 1 else x for x in sweep_lists]
                
            return list(zip(*sweep_lists)) # 返回对齐后的行数据 [(val1, val2), ...]

        # 2. 修改原方法：只处理普通参数
    def gen_param_sweep_generic(self, circuit: SubCircuitFactory, module_name: str,
                                param_names: list, module_config=None):
        """处理普通参数，生成独立的模块数据文件.data"""
        if not param_names: return
        
        if module_config is None and hasattr(self.sram_config, module_name):
            module_config = getattr(self.sram_config, module_name)

        rows = self._extract_sweep_rows(module_config, module_name, param_names)
        
        # 写入模块独立文件
        content = f".data {module_name}\n+ " + " ".join(param_names) + "\n"
        for row in rows:
            content += "+ " + " ".join([f"{float(x):.4e}" for x in row]) + "\n"
        
        path = os.path.join(self.sim_path, f'param_sweep_{module_name}.data')
        with open(path, 'w') as f: f.write(content)
        circuit.include(path)
        
        for p in param_names: 
            circuit.raw_spice += f'.param {p}=0.0\n'

    # 3. 新增方法：独立处理模型参数 (全局累积)
    def gen_model_sweep_generic(self, module_name: str, 
                                param_model_names: list, module_config=None):
        """处理模型参数，生成 param_sweep_models.data 文件"""
        if not param_model_names: return

        if module_config is None and hasattr(self.sram_config, module_name):
            module_config = getattr(self.sram_config, module_name)

        # 初始化全局缓存
        if not hasattr(self, '_global_model_data'):
            self._global_model_data = {'names': [], 'cols': []}

        # 提取数据
        rows = self._extract_sweep_rows(module_config, module_name, param_model_names)
        cols = list(zip(*rows)) # 转置为列以便追加

        # 更新缓存
        self._global_model_data['names'].extend(param_model_names)
        self._global_model_data['cols'].extend(cols)

        # 重写文件 (包含所有已收集的列)
        full_names = self._global_model_data['names']
        full_rows = list(zip(*self._global_model_data['cols']))
        
        content = f" ".join(full_names) + "\n"
        for row in full_rows:
            content += " ".join([str(x) for x in row]) + "\n"

        path = os.path.join(self.sim_path, 'param_sweep_models.data')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f: f.write(content)



    def run_mc_simulation(self, operation='read', target_row=0, target_col=0, mc_runs=100, temperature=27,vars=None):
        """Run Xyce Monte Carlo simulation"""
        # 定义扫描开关与配置键名的对应关系
        # 元组结构: (Config Key, Boolean Switch)
        # 这里的 Boolean Switch 来自 __init__ 中定义的 self.sweep_xxx
        check_list = [
            ('cell', self.sweep_cell) if self.sram_cell_type == 'SRAM_6T_CELL' else ('cell_10T', self.sweep_cell),
            ('precharge', self.sweep_precharge),
            ('senseamp', self.sweep_senseamp),
            ('wordlinedriver', self.sweep_wordlinedriver),
            ('columnmux', self.sweep_columnmux),
            ('writedriver', self.sweep_writedriver),
            ('decoder', self.sweep_decoder)
        ]
        for key, is_enabled in check_list:
            if is_enabled:
                # 从外部文件导入的配置中获取参数
                if key not in SWEEP_CONFIGS:
                    print(f"[WARNING] 配置 '{key}' 在 sram_mc_config.py 中未定义，跳过。")
                    continue            
                cfg = SWEEP_CONFIGS[key]        
                self.gen_model_sweep_generic(
                    module_name=cfg['name'],
                    param_model_names=cfg['model_params']
                )
        
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
                    module_name=cfg['name'],
                    param_names=cfg['params'],
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
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:Q)' if self.sram_cell_type == 'SRAM_6T_CELL' else f'V(XSRAM_10T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_10T_CELL_{target_row}_{target_col}:Q)',
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:QB)'if self.sram_cell_type == 'SRAM_6T_CELL' else f'V(XSRAM_10T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_10T_CELL_{target_row}_{target_col}:QB)',
                    f'V(SA_Q{target_col})',
                    f'V(SA_QB{target_col})',
                ]
            elif operation == 'write':
                selected_columns = [
                    f'V(WE)',
                    f'V(WL{target_row})',
                    f'V(BL{target_col})',
                    f'V(BLB{target_col})',
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:Q)'if self.sram_cell_type == 'SRAM_6T_CELL' else f'V(XSRAM_10T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_10T_CELL_{target_row}_{target_col}:Q)',
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:QB)'if self.sram_cell_type == 'SRAM_6T_CELL' else f'V(XSRAM_10T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_10T_CELL_{target_row}_{target_col}:QB)',
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
