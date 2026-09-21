import os
import hashlib
import json
import shutil
from pathlib import Path
from sram_compiler.per_device_mc.netlist import specialize_netlist
from PySpice.Unit import u_V, u_ns, u_Ohm, u_pF, u_A, u_mA 
# Only for yield analysis
from utils import (  # type: ignore
    parse_mc_measurements, generate_mc_statistics,
    save_mc_results, process_simulation_data,
    parse_spice_models, write_spice_models
)
from sram_compiler.testbenches.snm import process_xyce_montecarlo_prn
from sram_compiler.testbenches.sram_6t_core_testbench import Sram6TCoreTestbench  # type: ignore
from sram_compiler.config_yaml.sweep_config import SWEEP_CONFIGS
from sram_compiler.version import VERSION
import numpy as np
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from math import ceil, log2

# Fraction of the period by which a read's data must be on its rail and a
# written cell settled, counted from the capture edge of that cycle. The
# independent waveform screen (tests/spice/phased_waveforms.py) uses the same
# number, so a runtime pass and a screen pass mean the same thing.
ACCESS_DEADLINE = 0.68


def cycle_plan(operation, select_every=1, cycles=None):
    """Clock cycles of a transient deck: (transient length in periods, [(kind, data), ...]).

    `kind` is 'read', 'write' or 'idle' and `data` the value the target cell holds
    after the cycle.  Single decks select one cycle in `select_every` (V2.1.6:
    the idle cycles probe the write -> idle -> write boundary) and their write
    data alternates 1, 0, ...; the sequence writes 1, reads, writes 0, reads.
    """
    if isinstance(select_every, bool) or not isinstance(select_every, int) or select_every < 1:
        raise ValueError('select_every must be a positive integer')
    if operation == 'read&write':
        if select_every != 1:
            raise ValueError('The read&write sequence selects every cycle')
        cycles = 8 if cycles is None else cycles
        if cycles not in (4, 8):
            raise ValueError('Sequence decks have four or eight cycles')
        plan = [('write' if k % 2 == 0 else 'read', int((k // 2) % 2 == 0)) for k in range(cycles + 1)]
        return cycles + .7, plan
    if operation not in ('read', 'write'):
        raise ValueError(f"Invalid operation: {operation}")
    if cycles is not None:
        raise ValueError('cycles applies to the read&write sequence')
    span = 2 if select_every == 1 else select_every + 2
    plan = []
    for k in range(span):
        if k % select_every:
            plan.append(('idle', None))
        else:
            index = k // select_every
            plan.append((operation, int(index % 2 == 0) if operation == 'write' else 0))
    return span, plan


def access_cycles(operation, select_every=1, cycles=None):
    """(cycle, kind, data, next_cycle, next_kind, next_data) per checked access.

    Check accesses with their complete recovery phase inside the transient
    (1 ns + (cycle + 1.2) T). Data is due at cycle + ACCESS_DEADLINE T. The next_*
    fields identify a following selected access inside the simulated span.
    """
    span, plan = cycle_plan(operation, select_every, cycles)
    entries = []
    for cycle, (kind, data) in enumerate(plan):
        if kind == 'idle' or cycle + 1.2 > span + 1e-9:
            continue
        following = next(((j, jk, jd) for j, (jk, jd) in enumerate(plan)
                          if j > cycle and jk != 'idle'), None)
        if following is not None and following[0] + .7 > span + 1e-9:
            following = None
        entries.append((cycle, kind, data) + (following if following else (None, None, None)))
    return entries


class Sram6TCoreMcTestbench(Sram6TCoreTestbench):
    def __init__(self, sram_config, sram_cell_type="SRAM_6T_CELL",
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 vth_std=0.05, mc=True, enable_mc=None, custom_mc=False,
                 sweep_cell=False, sweep_precharge=False, sweep_senseamp=False, sweep_wordlinedriver=False,
                 sweep_columnmux=False, sweep_writedriver=False, sweep_decoder=False,
                 corner='TT', choose_columnmux=True, real_cell_mode=None,
                 q_init_val=0, sim_path='sim', enable_waveform=True,
                 mc_seed=None, xyce_options=None, t_max_step=None, next_row=None,
                 driver_sizes=None, timing_config=None, variation_mode=None, temperature=None,
                 interconnect=None, select_every=1):
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
                   mc_seed: Xyce 随机数种子 (None = 每次运行随机).  With a seed a
                            Monte Carlo sweep (mc_runs > 1) is reproducible.
                   xyce_options: extra netlist lines (e.g. ['.OPTIONS TIMEINT ERROPTION=1'])
                            appended to every transient deck.  Not needed for the
                            arrays validated in V2.0.1; a few 512-row decks stop with a
                            Xyce "time step too small" (Newton loop oscillating with a
                            1e-12 A residual) and complete with ERROPTION=1, at the
                            price of 2-15 % delay shifts on decks that converge anyway.
                   t_max_step: maximum transient time step in seconds (4th field of
                            .TRAN); None keeps Xyce's adaptive default.  A small value
                            (e.g. 2e-11) is the accuracy-preserving alternative for the
                            convergence problem above (delays within 0.5 %, ~1.8x the
                            time steps); it is applied automatically as a one-time
                            retry when Xyce reports "Time step too small".
                   next_row: row address captured at the next rising clock edge,
                            after low-phase recovery ('read' / 'write' only).
                            None = the address stays
                            at target_row; another row exercises the address-change
                            hold path (old wordline off before the new decoder output
                            rises).  The wordline and the cell (next_row, target_col)
                            are added to the .PRINT so the hold margin can be checked.
               """
        super().__init__(#父类
            sram_config, sram_cell_type,
            w_rc, pi_res, pi_cap,
            custom_mc, sweep_cell,sweep_precharge,sweep_senseamp,sweep_wordlinedriver,sweep_columnmux,sweep_writedriver,sweep_decoder,
            corner,choose_columnmux,real_cell_mode,q_init_val,sim_path,
            next_row=next_row,
            driver_sizes=driver_sizes,
            timing_config=timing_config,
            temperature=temperature,
            interconnect=interconnect,
            select_every=select_every,
        )
        self.sram_cell_type=sram_cell_type
        # enable_mc is an alias for mc (backward compatibility with experiment.py)
        if enable_mc is not None:
            mc = enable_mc
        if variation_mode is None:
            variation_mode = 'custom' if custom_mc else ('per-device' if mc else 'nominal')
        if variation_mode not in ('nominal', 'shared', 'custom', 'per-device'):
            raise ValueError('Unknown variation_mode')
        if custom_mc != (variation_mode == 'custom'):
            raise ValueError('custom_mc and variation_mode must agree')
        self.variation_mode = variation_mode
        self.variation_summary = {'variation_mode': variation_mode}
        mc = variation_mode in ('shared', 'per-device')
        self.mc=mc
        self.enable_mc = mc  # backward-compat alias
        self.enable_waveform = enable_waveform
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
        if self.driver_sizes.source == 'table' and variation_mode == 'per-device' and vth_std != .05:
            raise ValueError('Table-qualified timing requires the recorded 5% local mismatch model')
        self.mc_seed = mc_seed
        self.xyce_options = list(xyce_options) if xyce_options else []
        self.t_max_step = t_max_step
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
        if self.variation_mode in ('nominal', 'per-device'):
            pdk_path = getattr(self.sram_config.global_config, f"pdk_path_{self.corner}")
            pdk_path = Path(pdk_path).expanduser()
            if not pdk_path.is_absolute():
                pdk_path = Path(__file__).resolve().parents[2] / pdk_path
            pdk_path = str(pdk_path.resolve())
            circuit._includes[0] = pdk_path
        else:
            if not self.custom_mc:  #不需要自定义的MC时
                # Replace original included model lib with new path 替换为包含随机变量的模型文件
                circuit._includes[0] = self.create_mc_model_file()

        if self.variation_mode == 'per-device':
            # Specialize before returning the circuit, so direct deck exporters
            # and run_mc_simulation use exactly the same independent devices.
            # Instance/node names and width expressions are retained verbatim.
            source = str(circuit)
            digest = hashlib.sha256(source.encode() + Path(pdk_path).read_bytes()
                                    + repr(self.vth_std).encode()).hexdigest()[:16]
            audit_path = Path(self.sim_path) / f'model_audit_{digest}.csv'
            deck, self.variation_summary = specialize_netlist(
                source, base_model_path=Path(pdk_path),
                model_output_path=Path(self.sim_path) / f'models_per_device_{digest}.spice',
                audit_path=audit_path,
                mc_runs=None, vth_std=self.vth_std,
                deck_base_dir=Path(__file__).resolve().parents[2],
            )
            self.variation_summary['audit_file'] = str(audit_path.resolve())
            specialized = Circuit(circuit.title)
            specialized.raw_spice = deck.split('\n', 1)[1]
            circuit = specialized

        return circuit

    def add_meas_and_print(self, simulator, init_cond, operation):
        # Internal nodes' names of the target cell
        target_node_q = self.cell_inst_prefix + f'_{self.target_row}_{self.target_col}{self.heir_delimiter}Q'
        target_node_qb = self.cell_inst_prefix + f'_{self.target_row}_{self.target_col}{self.heir_delimiter}QB'
        local_wl = f'WL{self.target_row}'
        local_bl, local_blb = f'BL{self.target_col}', f'BLB{self.target_col}'
        sense_bl, sense_blb = local_bl, local_blb
        if operation in ('read', 'write', 'read&write'):
            local_wl = self.cell_probe('WL')
            local_bl, local_blb = self.cell_probe('BL'), self.cell_probe('BLB')
            sense_bl, sense_blb = self.sense_input_probe('IN'), self.sense_input_probe('INB')
        #获取目标单元的内部节点名称（Q 和 QB）
        if operation == 'hold_snm' or operation == 'read_snm' or operation == 'write_snm':
            # Initial V(BL) and V(BLB) for the  cell
            init_cond = {}
            init_cond[f'BL'] = self.vdd @ u_V   #设置位线初始电压
            init_cond[f'BLB'] = self.vdd @ u_V
            simulator.initial_condition(**init_cond)    #读入初始电压设置
            #simulator.measure('DC', 'MAXVD', 'MAX V(VD)')   #测量最大对角线电压
            #simulator.measure('DC', operation.upper(), f"PARAM='1/sqrt(2)*MAXVD'")  #计算静态噪声容限
            # Add print for SNM
            simulator.circuit.raw_spice += \
                f'.PRINT DC FORMAT=NOINDEX {{U}} V(V1) V(V2)\n'

        # The read operation
        elif operation == 'read':
            # .ic conditions: the precharged clock-low state (V2.2.2).
            self._init_precharged_columns(init_cond, sense=True)
            # Preset the output latch to VDD: the target cell stores 0, so a completed
            # read is the *falling* edge of OUT (used by TSA / TREAD_TOTAL below).
            init_cond['OUT'] = self.vdd @ u_V
            self._init_control_path(init_cond)

            simulator.initial_condition(**init_cond)

            # Measurements for precharge delay (TPRCH), defined by the time from PRE assertion to V(BL)=0.9*VDD
            #预充延迟（TPRCH）的测量，由预充断言到V（BL）的时间定义=0.9*VDD
            simulator.measure(
                'TRAN', 'TPRCH',
                f'TRIG V(PRE)={self.half_vdd} FALL=1 TD={1e-9 + .65 * float(self.t_period):.12g} ' +
                f'TARG V({local_bl})={float(self.vdd) * 0.9} RISE=1 TD={1e-9 + .65 * float(self.t_period):.12g}')  # modified for Xyce

            # Decoder delay (TDECODER): address capture -> decoder output
            self._add_decoder_measure(simulator)

            # Measurements for wl driver delay (TWLDRV), defined as the time from the WLE assertion to V(WL)=VDD/2
            #测量WLE驱动延迟（TWLDRV），定义为从译码器输出到V(wl)=VDD/2的时间
            simulator.measure(
                'TRAN', 'TWLDRV',
                f'TRIG V(wl_en)={self.half_vdd} RISE=1 ' +
                f'TARG V({local_wl})={self.half_vdd} RISE=1')

            # Add measurements for read delay (TREAD),读延迟
            # which is defined as the time from the WL rise to BL swing to VDD/2
            # Mux/input RC can cross the differential threshold during initial
            # condition release. Measure only the first clock-high access;
            # otherwise TBL can be a few ps and TSWING several ns negative.
            access_start = 1e-9 + .2 * float(self.t_period)
            access_stop = 1e-9 + .7 * float(self.t_period)
            access_window = f'TD={access_start:.12g} TO={access_stop:.12g}'
            simulator.measure(
                'TRAN', 'TWL',
                f'WHEN V({local_wl})={self.half_vdd} RISE=1 {access_window}')  # modified for Xyce
            # Define minimum Vswing = 250mV
            vswing = 0.25
            simulator.measure(
                'TRAN', 'TBL',
                f"WHEN V({sense_bl})='V({sense_blb})-{vswing}' FALL=1 {access_window}")
            simulator.measure('TRAN', 'TSWING', f"PARAM='TBL-TWL'")

            # Sense-amp delay (TSA): sense-enable assertion -> data output valid.  OUT is
            # preset to VDD and the target cell stores 0, so the read completes on the
            # first falling crossing of OUT.  The latch is opaque while S_EN is low, so
            # this event cannot precede the trigger.  (SA_Q itself tracks BL through the
            # SA pass gates before S_EN and reaches any threshold *before* the enable;
            # its old 0.01*VDD target also sat on the `.IC` parking level.)
            #SA延迟（TSA）的测量，定义为从SAE断言到输出锁存器 OUT 翻转的时间
            simulator.measure(
                'TRAN', 'TSA',
                f'TRIG V(s_en)={self.half_vdd} RISE=1 ' +
                f'TARG V(OUT)={self.half_vdd} FALL=1')

            # s_en rise time (20% -> 80%), both crossings on the same (first) rising
            # edge, measured only from the start of the clock-high (access) phase:
            # when the first precharge fires (~3.3 ns) all bitlines rise together and
            # couple through the sense-amp pass gates into s_en, a bump that reaches
            # 0.2-0.3 V with 32-128 sense amplifiers and would otherwise be taken as
            # the 20 % crossing (TS_EN then read ~5.3 ns).
            t_acc0 = float(1.0 @ u_ns) + 0.2 * float(self.t_period)
            simulator.measure(
                'TRAN', 'Ts_en',
                f'TRIG V(S_EN)={float(self.vdd) * 0.2} RISE=1 TD={t_acc0} ' +
                f'TARG V(S_EN)={float(self.vdd) * 0.8} RISE=1 TD={t_acc0}')

            #总读延迟: wl_en assertion -> data output valid
            simulator.measure(
                'TRAN', 'TREAD_TOTAL',
                f'TRIG V(WL_EN)={self.half_vdd} RISE=1 ' +
                f'TARG V(OUT)={self.half_vdd} FALL=1')
            # Clock-to-wl_en, decoder settle and bitline restore for the
            # minimum-period estimate (the read discharges BL{target_col}).
            self._add_period_measures(simulator, 'read', f'BL{self.target_col}')
            # Add measurements for average power, static power and dynamic power    测量功耗(平均、动态、静态)
            self._add_power_measures(simulator, 'EREAD')

             # Add additional print statements for clock and control signals
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(CLK) V(CLK_BUF) V(CLK_BAR) V(CSB)' +\
                f' V(CS_BAR) V(CS) V(WEB) V(WE_BAR) V(WE) V(GATED_CLK_BUF) V(GATED_CLK_BAR) V(WL_EN) V(RWL)\n'
            
            # Add print statement for address and other signals
            address_signals = ' '.join([f'V(A{i}) V(A_DFF{i})' for i in range(ceil(log2(self.num_rows)))])
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX {address_signals} V(RBL) V(RBL_DELAY) V(RBL_DELAY_BAR) V(W_EN) V(PRE) V(SA_ISO)\n'
            
            # Add print for read operation
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(S_EN) V({local_wl}) V(DEC_WL{self.target_row}) ' + \
                f'V({local_bl}) V({local_blb}) ' + \
                f'V({target_node_q}) V({target_node_qb}) \n'
            if self.choose_columnmux:
                simulator.circuit.raw_spice += \
                    f'.PRINT TRAN V(SA_IN{self.target_col // self.mux_in}) ' + \
                    f'V(SA_INB{self.target_col // self.mux_in})\n'      
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN V(SA_Q{self.target_col // self.mux_in}) ' + \
                f'V(SA_QB{self.target_col // self.mux_in})' + \
                f' V(OUT)\n'
            self._add_next_row_print(simulator)

        # The write operation
        elif operation == 'write':
            # .ic conditions: the precharged clock-low state (V2.2.2).
            self._init_precharged_columns(init_cond, sense=False)
            for col in range(self.num_cols):
                init_cond[f'DIN_dff{col}'] = 0 @ u_V
            self._init_control_path(init_cond)

            simulator.initial_condition(**init_cond)

            #字线驱动延迟
            self._add_decoder_measure(simulator)
            # Measurements for wl driver delay (TWLDRV), defined as the time from the WLE assertion to V(WL)=VDD/2
            simulator.measure(
                'TRAN', 'TWLDRV',
                f'TRIG V(WL_EN)={self.half_vdd} RISE=1 ' +
                f'TARG V({local_wl})={self.half_vdd} RISE=1')
            #写驱动延迟
            # Write driver delay (TWDRV): w_en assertion -> the driven bitline
            # reaches VDD/2.  The bitlines are precharged to VDD before the
            # write and a '1' is written, so the write driver pulls BLB low
            # while BL stays high (the old BL RISE target relied on the
            # bitlines starting at the .IC value 0 V with no precharge).
            simulator.measure(
                'TRAN', 'TWDRV',
                f'TRIG V(w_en)={self.half_vdd} RISE=1 ' +
                f'TARG V({local_blb})={self.half_vdd} FALL=1')
            #写延迟
            # Measurements for write delay (TWRITE_Q/QB),
            # which is defined as the time from the WL rise to data Q rise to 90% VDD.
            simulator.measure(
                'TRAN', 'TWRITE_Q',
                f'TRIG V({local_wl})={self.half_vdd} RISE=1',
                f"TARG V({target_node_q})={float(self.vdd) * 0.9:.2f} RISE=1")
            simulator.measure(
                'TRAN', 'TWRITE_QB',
                f'TRIG V({local_wl})={self.half_vdd} RISE=1',
                f"TARG V({target_node_qb})={float(self.vdd) * 0.1:.2f} FALL=1")
            #总延迟
            simulator.measure(
                'TRAN', 'TWRITE_TOTAL',
                f'TRIG V(WL_EN)={self.half_vdd} RISE=1',
                f"TARG V({target_node_q})={float(self.vdd) * 0.9:.2f} RISE=1")
            # Minimum-period measures; the next selected cycle writes '0', so
            # its write slot pulls BL{target_col} to the low rail (TWSLOT).
            self._add_period_measures(simulator, 'write', f'BL{self.target_col}')
            # Add measurements for average power, static power and dynamic power    功耗
            self._add_power_measures(simulator, 'EWRITE')
            # Add print for write operation
            address_signals = ' '.join([f'V(A{i}) V(A_DFF{i})' for i in range(ceil(log2(self.num_rows)))])
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX {address_signals} V(we) V({local_wl}) V(DEC_WL{self.target_row}) V({local_bl})' + \
                f' V({local_blb}) V({target_node_q}) V({target_node_qb})\n'
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(cs) V(clk_buf) V(clk_bar) V(gated_clk_bar) V(DIN0) V(DIN_dff0)' + \
                f' V(w_en) V(wl_en) V(web) V(RBL) V(RBL_DELAY_BAR) V(PRE) V(SA_ISO)\n'
            self._add_next_row_print(simulator)

        elif operation == 'read&write':
            # .ic conditions: the precharged clock-low state (V2.2.2).
            self._init_precharged_columns(init_cond, sense=True)
            for col in range(self.num_cols):
                init_cond[f'DIN_dff{col}'] = 0 @ u_V
            init_cond['OUT'] = 0 @ u_V
            self._init_control_path(init_cond)

            simulator.initial_condition(**init_cond)

             # Add print for read operation
             # Add additional print statements for clock and control signals
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(CLK) V(CLK_BUF) V(CLK_BAR) V(CSB)' +\
                f' V(CS_BAR) V(CS) V(WEB) V(WE_BAR) V(WE) V(GATED_CLK_BUF) V(GATED_CLK_BAR) V(WL_EN) V(RWL)\n'
            
            # Add print statement for address and other signals
            address_signals = ' '.join([f'V(A{i}) V(A_DFF{i})' for i in range(ceil(log2(self.num_rows)))])
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX {address_signals} V(RBL) V(RBL_DELAY) V(RBL_DELAY_BAR) V(W_EN) V(PRE) V(SA_ISO)\n'
            
            simulator.circuit.raw_spice += \
                f'.PRINT TRAN FORMAT=NOINDEX V(S_EN) V({local_wl}) ' + \
                f'V({local_bl}) V({local_blb}) ' + \
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
            # Average power over one complete write-1 / read / write-0 / read
            # pattern (DIN has a period of 4*t_period), starting at the first
            # access so the start-up transient is excluded; ends well before
            # the .TRAN end (1 ns + 8.7*t_period, see _analysis_stop).
            t_from = float(1.0 @ u_ns) + 0.2 * float(self.t_period)
            simulator.measure(
                'TRAN', 'PAVG',
                f'AVG {{-V(VDD)*I(VVDD)}} FROM={t_from} ' +
                f'TO={t_from + 4 * float(self.t_period)}'
            )
            self._add_static_power_measures(simulator)
        else:
            raise ValueError(f"Invalid operation: {operation}")

        if operation in ('read', 'write', 'read&write'):
            self._add_precharge_safety_measures(simulator, operation)
            self._add_access_checks(simulator, operation, target_node_q, target_node_qb)
        if operation in ('read', 'write', 'read&write'):
            self._add_interconnect_print(simulator)

    def _analysis_stop(self, operation):
        """End of the transient analysis (see add_analysis).

        Single decks simulate two selected cycles (plus the idle cycles of
        `select_every`), the sequence its eight cycles and the following restore
        phase.  Round up onto the output interval (t_step): Xyce prints on that
        grid, and an off-grid stop (1 ns + 8.7 * 4.75 ns = 42.325 ns on 2 ps,
        V2.1.4) ended some traces with the final time printed twice.
        """
        cycles = cycle_plan(operation, self.select_every)[0]
        step = float(self.t_step)
        return ceil((1e-9 + cycles * float(self.t_period)) / step - 1e-6) * step

    def _add_access_checks(self, simulator, operation, q, qb):
        """Check data by clock fall, retention until the next capture, and recovery.

        Xyce TRIG/TARG delays can report a crossing after the clock deadline;
        a TO option does not bound that target in Xyce 7.4. Check the actual
        data at an explicit time and invalidate late crossings in the caller.
        These targeted checks complement independent waveform screening.
        """
        vdd, period = float(self.vdd), float(self.t_period)
        wl = self.cell_probe('WL')
        pre = (f'{self.prch_inst_prefix}_{self.target_col}:ENB_end' if self.w_rc
               else self.control_tap('PRE', self.target_col))
        enable = (f'{self.wdrv_inst_prefix}_{self.target_col}:EN_end' if self.w_rc
                  else self.control_tap('w_en', self.target_col))
        sense = (f'{self.sa_inst_prefix}_{self.target_col // self.mux_in}:EN_end' if self.w_rc
                 else self.control_tap('s_en', (self.target_col // self.mux_in) * self.mux_in))
        isolation = (f'{self.sa_inst_prefix}_{self.target_col // self.mux_in}:ISO_end' if self.w_rc
                     else self.control_tap('sa_iso', (self.target_col // self.mux_in) * self.mux_in))
        bitlines = [f'{self.arr_inst_prefix}:BL{self.target_col}_far',
                    f'{self.arr_inst_prefix}:BLB{self.target_col}_far',
                    f'{self.replica_inst_prefix}:RBL_far']
        for cycle, kind, expected, following, next_kind, next_data in access_cycles(operation, self.select_every):
            error = f'MAX(ABS(V({q})-{expected * vdd:.12g}),ABS(V({qb})-{(1-expected) * vdd:.12g}))'
            if kind == 'read':
                error = f'MAX({error},ABS(V(OUT)-{expected * vdd:.12g}))'
            # V2.2.3: the data deadline is the waveform checker's k + 0.68 T,
            # not the looser k + 0.7 T these cards used through V2.2.2. The
            # 0.02 T difference is 180 ps at a 9 ns clock and 555 ps at
            # 27.75 ns, and it is the gap that let a V2.2.1 8x512 read pass
            # these measures while still on the wrong rail at the checker's
            # deadline. Runtime and screen now accept exactly the same traces.
            simulator.measure('TRAN', f'VACCESS_ERROR_{cycle}',
                              f'FIND {{{error}}} AT={1e-9 + (cycle + ACCESS_DEADLINE) * period:.12g}')
            simulator.measure('TRAN', f'VHOLD_ERROR_{cycle}',
                              f'MAX {{{error}}} FROM={1e-9 + (cycle + ACCESS_DEADLINE) * period:.12g} '
                              f'TO={1e-9 + (cycle + 1.2) * period:.12g}')
            # PRE is active low. A correct final datum does not excuse opening
            # the cell or write driver before local precharge has released.
            pre_error = f'ABS(V({pre})-{vdd:.12g})'
            simulator.measure('TRAN', f'VPRE_ACCESS_ERROR_{cycle}',
                              f'MAX {{MAX(IF(V({wl})>{.1*vdd:.12g},{pre_error},0),'
                              f'IF(V({enable})>{.5*vdd:.12g},{pre_error},0))}} '
                              f'FROM={1e-9 + (cycle + .2) * period:.12g} '
                              f'TO={1e-9 + (cycle + 1.2) * period:.12g}')
            if kind == 'write':
                # V2.1.9: the drivers stay fully on while the wordline is on.  The
                # target cell's wordline and the far end of its row are below
                # 0.1 VDD whenever the target driver's enable is below 0.9 VDD.
                wl_level = f'MAX(ABS(V({wl})),ABS(V({self.arr_inst_prefix}:WL{self.target_row}_far)))'
                simulator.measure('TRAN', f'VWEN_ACCESS_ERROR_{cycle}',
                                  f'MAX {{IF(V({enable})<{.9*vdd:.12g},{wl_level},0)}} '
                                  f'FROM={1e-9 + (cycle + .2) * period:.12g} '
                                  f'TO={1e-9 + (cycle + 1.2) * period:.12g}')
            roles = f'MAX(IF(V({sense})>{.1*vdd:.12g},ABS(V({wl})),0),IF(V({enable})>{.1*vdd:.12g},ABS(V({sense})),0))'
            iso_error = f'IF(MAX(V({sense}),V({enable}))>{.1*vdd:.12g},ABS(V({isolation})-{vdd:.12g}),0)'
            for phase, expression in (('ROLE', roles), ('ISO', iso_error)):
                simulator.measure('TRAN', f'V{phase}_ERROR_{cycle}',
                                  f'MAX {{{expression}}} FROM={1e-9 + (cycle + .2) * period:.12g} '
                                  f'TO={1e-9 + (cycle + 1.2) * period:.12g}')
            boundary = f'MAX(ABS(V({wl})),MAX(ABS(V({enable})),MAX(ABS(V({sense})),ABS(V({isolation})))))'
            for node in ('XTIME_CONTROL:wordline_busy', 'XTIME_CONTROL:iso_ready',
                         'XTIME_CONTROL:access_request', 'XTIME_CONTROL:access_settled',
                         'XTIME_CONTROL:read_done', 'rbl_delay'):
                boundary = f'MAX({boundary},ABS(V({node})))'
            boundary = f'MAX({boundary},ABS(V(XTIME_CONTROL:enables_off)-{vdd:.12g}))'
            simulator.measure('TRAN', f'VBOUNDARY_ERROR_{cycle}',
                              f'FIND {{{boundary}}} AT={1e-9 + (cycle + .2) * period:.12g}')
            if following is None:
                continue
            # End of this cycle's recovery: every bitline is precharged,
            # including after a write. Check before the next capture edge.
            targets = [vdd, vdd, vdd]
            restore_error = f'ABS(V({bitlines[0]})-{targets[0]:.12g})'
            for node, target in zip(bitlines[1:], targets[1:]):
                restore_error = f'MAX({restore_error},ABS(V({node})-{target:.12g}))'
            simulator.measure('TRAN', f'VRESTORE_ERROR_{cycle}',
                              f'FIND {{{restore_error}}} AT={1e-9 + (cycle + 1.1) * period:.12g}')

    @staticmethod
    def access_validity(measurements, operation, vdd, select_every=1, cycles=None):
        entries = access_cycles(operation, select_every, cycles)
        names = [f'V{phase}_ERROR_{cycle}' for cycle, *_ in entries
                 for phase in ('ACCESS', 'HOLD', 'PRE_ACCESS', 'BOUNDARY', 'ROLE', 'ISO')]
        # V2.1.9: a write's drivers stay on while its wordline is on.
        names += [f'VWEN_ACCESS_ERROR_{cycle}' for cycle, kind, *_ in entries if kind == 'write']
        limits = [.1 * vdd] * len(names)
        restore = [f'VRESTORE_ERROR_{cycle}' for cycle, kind, data, following, *_ in entries
                   if following is not None]
        names += restore
        limits += [.02 * vdd] * len(restore)
        values = measurements.reindex(columns=names).to_numpy(dtype=float)
        return np.isfinite(values).all(axis=1) & (np.abs(values) <= limits).all(axis=1)

    def _add_precharge_safety_measures(self, simulator, operation):
        """Physical WL must be off throughout each clock-low precharge.

        Check the observed entry edge and any subsequent capacitive rebound,
        with every measurement bounded by this cycle's recovery interval.

        Without the replica guard `wordline_off` falls back to the logical
        `wl_en_bar`, so PRE no longer waits for the physical wordline and this
        measurement would compare the precharge against a signal that cannot
        report the distributed tail.  Refuse to emit it rather than report a
        safety margin the controller does not enforce.
        """
        period = float(self.t_period)
        vdd = float(self.vdd)
        analysis_stop = float(self._analysis_stop(operation))
        sizes = self.driver_sizes
        if not (sizes.precharge_off_guard and sizes.replica_precharge_guard):
            raise ValueError('The precharge safety measures need the replica and precharge-off guards')
        probes = {'FAR': f'{self.arr_inst_prefix}:WL{self.target_row}_far',
                  'LOCAL': self.cell_probe('WL')}
        peak = f'MAX(ABS(V({probes["FAR"]})),ABS(V({probes["LOCAL"]})))'
        for cycle, kind, data, following, next_kind, next_data in access_cycles(operation, self.select_every):
            if following is None:
                continue
            start = 1e-9 + (cycle + .7) * period
            # Never claim a rebound interval beyond the simulated transient.
            stop = min(1e-9 + (cycle + 1.2) * period, analysis_stop)
            name, event, active = 'PRE', f'V(PRE)={.9 * vdd:.12g} FALL=1', f'V(PRE)<{.9 * vdd:.12g}'
            for location, node in probes.items():
                simulator.measure('TRAN', f'VWL_{name}_{location}_{cycle}',
                                  f'FIND V({node}) WHEN {event} TD={start:.12g} TO={stop:.12g}')
            # Bitline restoration or the write drivers can capacitively lift an
            # already falling WL.  Check the whole entry interval as well.
            simulator.measure('TRAN', f'VWL_{name}_PEAK_{cycle}',
                              f'MAX {{IF({active},{peak},0)}} FROM={start:.12g} TO={stop:.12g}')

    def _check_distributed_precharge(self, measurements, operation):
        self.validate_distributed_precharge(measurements, operation, float(self.vdd),
                                            select_every=self.select_every)

    @staticmethod
    def validate_distributed_precharge(measurements, operation, vdd, select_every=1, cycles=None):
        """Refuse unsafe/incomplete samples while preserving their raw measures."""
        names = [f'VWL_PRE_{location}_{cycle}'
                 for cycle, kind, data, following, next_kind, next_data
                 in access_cycles(operation, select_every, cycles)
                 if following is not None
                 for location in ('FAR', 'LOCAL', 'PEAK')]
        values = measurements.reindex(columns=names).to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1) & (np.abs(values) <= .1 * vdd).all(axis=1)
        if measurements.empty or not valid.all():
            failed = list(measurements.index[~valid])
            raise RuntimeError('SRAM precharge overlaps an active wordline or its '
                               f'entry measurement is missing in samples {failed}. '
                               'Inspect VWL_PRE_* and the waveforms; this wire/timing '
                               'configuration cannot supply valid SRAM metrics.')

    def _add_interconnect_print(self, simulator):
        """Export physical wire endpoints and sense inputs for waveform scoring."""
        probes = ['RWL', 'RWL_far', self.sense_input_probe('IN'), self.sense_input_probe('INB')]
        probes += ['XTIME_CONTROL:access_request', 'XTIME_CONTROL:pre_off_ready',
                   'XTIME_CONTROL:wordline_request', 'XTIME_CONTROL:access_settled',
                   'XTIME_CONTROL:write_window',
                   'XTIME_CONTROL:s_en_bar', 'XTIME_CONTROL:enables_off',
                   'XTIME_CONTROL:wordline_busy', 'XTIME_CONTROL:wordline_idle',
                   'XTIME_CONTROL:read_done', 'XTIME_CONTROL:iso_ready',
                   'XTIME_CONTROL:local_enables_off', 'XTIME_CONTROL:far_enables_off',
                   'XTIME_CONTROL:far_enables_ready']
        probes += [f'{self.replica_inst_prefix}:RBL_far', f'{self.replica_inst_prefix}:RBLB_far']
        for name, taps in self._control_taps.items():
            probes += [name, taps[0], taps[-1], self.control_tap(name)]
        for bit in range(max(1, ceil(log2(self.num_rows)))):
            probes += [f'A_dff{bit}', f'XDECODER:A{bit}_line_far']
        if self.operation in ('write', 'read&write'):
            probes += ['CLK_BUF', 'XTIME_CONTROL:Xdff_buf_data:CLK_line_far']
            for col in sorted({0, self.target_col, self.num_cols - 1}):
                probes += [f'XTIME_CONTROL:Xdff_buf_data:CLK_line_tap{col}',
                           f'DIN{col}', f'DIN_dff{col}', f'DIN_buf{col}']
        for col in [0, self.target_col, self.num_cols - 1, None]:
            for pin in ('BL', 'BLB'):
                probes += [self.periphery_tap(pin, col, role)
                           for role in ('precharge', 'write', 'sense')]
        for row in range(self.num_rows):
            probes += [f'WL{row}', f'{self.arr_inst_prefix}:WL{row}_far']
        for col in range(self.num_cols):
            for pin in ('BL', 'BLB'):
                probes += [f'{pin}{col}', f'{self.arr_inst_prefix}:{pin}{col}_far']
        if self.w_rc:
            probes += [f'{self.cell_inst_prefix}_{self.target_row}_{self.target_col}:{pin}_end'
                       for pin in ('Q', 'QB')]
        signals = ' '.join(f'V({node})' for node in dict.fromkeys(probes))
        simulator.circuit.raw_spice += f'.PRINT TRAN FORMAT=NOINDEX {signals}\n'

    def _add_next_row_print(self, simulator):
        """Wordline, decoder output and cell of `next_row` (address-change hold check)."""
        if self.next_row is None or self.next_row == self.target_row:
            return
        row, col = self.next_row, self.target_col
        wl = f'{self.arr_inst_prefix}:WL{row}_far'
        signals = f'V({wl}) V(DEC_WL{row})'
        core = getattr(self, 'sbckt_array', None)
        if core is None or core._should_instantiate_real_cell(row, col):
            q = self.cell_inst_prefix + f'_{row}_{col}{self.heir_delimiter}Q'
            qb = self.cell_inst_prefix + f'_{row}_{col}{self.heir_delimiter}QB'
            signals += f' V({q}) V({qb})'
        simulator.circuit.raw_spice += f'.PRINT TRAN FORMAT=NOINDEX {signals}\n'

    def _init_precharged_columns(self, init_cond, sense):
        """Start every bitline pair, the replica bitline and the sense nodes at VDD.

        At t = 0 the clock is low, so the controller holds PRE active and the
        sense-amplifier pass gates open: the physical state is every bitline,
        RBL and both sense nodes at VDD.  Until V2.2.1 the decks forced BL and
        RBL to 0 V (and SA_Q to 0 V) against the active precharge, an
        inconsistent operating point that Xyce 7.4 could not solve for one
        per-device FF 10T mux read (`8x4_10t_mux_FF_read_pd`, V2.2.1 record).
        The precharged bias is the state clock-low recovery returns to, so
        no measurement window changes; the startup charge crossing disappears.
        """
        for col in range(self.num_cols):
            init_cond[f'BL{col}'] = self.vdd @ u_V
            init_cond[f'BLB{col}'] = self.vdd @ u_V
        init_cond['RBL'] = self.vdd @ u_V
        init_cond['RBLB'] = self.vdd @ u_V
        if sense:
            for sa in range(self.num_cols // self.mux_in):
                # One sense amp per mux group; both nodes follow the precharged inputs.
                init_cond[f'SA_Q{sa}'] = self.vdd @ u_V
                init_cond[f'SA_QB{sa}'] = self.vdd @ u_V
        return init_cond

    def _init_control_path(self, init_cond):
        """Deterministic start-up state of the control path (shared by read / write)."""
        n_bits = ceil(log2(self.num_rows)) if self.num_rows > 1 else 1
        for bit in range(n_bits):
            # address register output (inside TIME_CONTROL) and the held / buffered copy
            init_cond[f'XTIME_CONTROL:A_reg{bit}'] = 0 @ u_V
            init_cond[f'A_dff{bit}'] = 0 @ u_V
        init_cond['we'] = self.vdd @ u_V       # we初始化为高电平
        init_cond['cs_bar'] = self.vdd @ u_V   # cs_bar初始化为高电平
        # Slave node of the CS flip-flop (cs = ~qint).  Without this the slave latch
        # powers up in a random state; when it comes up "selected" the start-up clamp
        # fights the DFF output inverter (~300 uA) until the clamp releases, i.e. inside
        # the EREAD / EWRITE window.
        init_cond['XTIME_CONTROL:Xdff_buf:qint'] = self.vdd @ u_V
        if self.operation in ('write', 'read&write'):
            # Initialize the data buffer and the register feedback nodes
            # consistently instead of letting DC Newton choose their state.
            for col in range(self.num_cols):
                init_cond[f'DIN_buf{col}'] = 0 @ u_V
                # At clk=0 the master is transparent with DIN=0; the slave
                # holds zero. Initialize every feedback node consistently.
                prefix = f'XTIME_CONTROL:Xdff_buf_data:Xdff_{col}'
                for node in ('D_b', 'z1', 'z3', 'z4', 'z5', 'QB'):
                    init_cond[f'{prefix}:{node}'] = self.vdd @ u_V
                init_cond[f'{prefix}:z2'] = 0 @ u_V
        return init_cond

    def _print_min_period(self, stats_csv_path, operation):
        """Diagnostic 50%-duty estimate from access/decode and recovery.

        This measured estimate is not the lookup qualification rule. It uses
        mean + one standard deviation and 10% margin, and excludes access tail
        risk not represented by these scalar measurements.
        """
        import pandas as pd
        stats_df = pd.read_csv(stats_csv_path, index_col=0)

        def worst(name):
            if name not in stats_df.index:
                return None
            mean_val = float(stats_df.loc[name, 'mean'])
            std_val = stats_df.loc[name, 'std']
            if pd.isna(std_val) or not np.isfinite(std_val):
                return mean_val
            return mean_val + float(std_val)

        access = worst('TREAD_TOTAL' if operation == 'read' else 'TWRITE_TOTAL')
        t_wlen = worst('TCLK_WLEN')
        # Every operation restores in clock-low, including writes.
        t_restore = worst('TRESTORE')
        t_dec = worst('TCLK_DEC')
        if access is None or t_wlen is None or t_restore is None:
            print("[WARNING] minimum-period estimate skipped: TCLK_WLEN / TRESTORE / "
                  "access delay missing in the statistics")
            return None
        half_high = max(t_wlen + access, t_dec or 0.0)
        half_low = t_restore
        t_min = 2.0 * max(half_low, half_high) * 1.1
        print(f"[INFO] clock-high access/decode : {half_high:.3e} s")
        print(f"[INFO] clock-low recovery      : {half_low:.3e} s")
        print(f"[INFO] CLK(min) estimate in this size and PVT (50 % duty, +10 %) : {t_min:.3e} s")
        return t_min

    def _add_period_measures(self, simulator, operation, bitline):
        """TCLK_WLEN: rising capture edge to wordline enable.

        TCLK_DEC: capture to decode. TRESTORE: falling edge to restored bitline,
        for both reads and writes. TWSLOT remains a write diagnostic: next
        rising edge to low write rail; it is part of clock-high access now.
        """
        node = f'{self.arr_inst_prefix}:{bitline}_far'
        vdd = float(self.vdd)
        following = access_cycles(operation, self.select_every)[0][3]
        if following is None:
            raise ValueError('The period measures need a following selected cycle')
        t_edge = 1e-9 + (following + .2) * float(self.t_period)   # edge starting the next selected cycle
        simulator.measure(
            'TRAN', 'TCLK_WLEN',
            f'TRIG V(clk)={self.half_vdd} RISE=1 ' +
            f'TARG V(wl_en)={self.half_vdd} RISE=1')
        if self.target_row != 0:
            simulator.measure(
                'TRAN', 'TCLK_DEC',
                f'TRIG V(clk)={self.half_vdd} RISE=1 ' +
                f'TARG V(DEC_WL{self.target_row})={self.half_vdd} RISE=1')
        # Recovery always precharges, after reads and writes alike.
        # BL is low after reading 0; BLB is low after the first write of 1.
        restore_node = node if operation == 'read' else f'{self.arr_inst_prefix}:BLB{self.target_col}_far'
        td = 1e-9 + .65 * float(self.t_period)
        simulator.measure(
            'TRAN', 'TRESTORE',
            f'TRIG V(clk)={self.half_vdd} FALL=1 TD={td:.4e} ' +
            f'TARG V({restore_node})={0.9 * vdd:.3f} RISE=1 TD={td:.4e}')
        if operation == 'write':
            td = t_edge - 1e-10
            simulator.measure(
                'TRAN', 'TWSLOT',
                f'TRIG V(clk)={self.half_vdd} RISE=1 TD={td:.4e} ' +
                f'TARG V({node})={0.1 * vdd:.3f} FALL=1 TD={td:.4e}')

    def _add_decoder_measure(self, simulator):
        """TDECODER: address-bit capture -> decoder output of the target row."""
        if self.target_row == 0:
            # Address 0 is also the start-up state: no address bit toggles and DEC_WL0 is
            # already high, so the decoder delay cannot be observed with this stimulus.
            print("[WARNING] TDECODER not measured: target_row=0 toggles no address bit")
            return
        bit = (self.target_row & -self.target_row).bit_length() - 1   # lowest set bit
        simulator.measure(
            'TRAN', 'TDECODER',
            f'TRIG V(A_dff{bit})={self.half_vdd} RISE=1 ' +
            f'TARG V(DEC_WL{self.target_row})={self.half_vdd} RISE=1')

    def _add_static_power_measures(self, simulator):
        """PSTC over a quiescent window and PDYN = PAVG - PSTC.

        The window 1 ns + [1.1, 1.15]*t_period is at the end of clock-low
        recovery, before the next capture at 1 ns + 1.2*T. Startup
        precharge is outside this window even for the short lookup clocks.
        """
        t_period = float(self.t_period)
        t0 = float(1.0 @ u_ns)
        # Use the end of the restore phase after the first access. This is
        # inside the same full cycle as PAVG and excludes startup charging.
        simulator.measure(
            'TRAN', 'PSTC',
            f'AVG {{-V(VDD)*I(VVDD)}} FROM={t0 + 1.1 * t_period} ' +
            f'TO={t0 + 1.15 * t_period}'
        )
        simulator.measure(
            'TRAN', 'PDYN',
            f'PARAM={{PAVG-PSTC}}'
        )

    def _add_power_measures(self, simulator, energy_name):
        """Energy of one full clock period, PAVG, PSTC and PDYN.

        The window starts at first capture (clock rise begins at 1 ns + 0.2*T)
        and ends one period later, so it contains one selected capture-to-capture
        cycle: wordline access, sensing / writing, the low-phase precharge that
        restores the bitlines, and the idle time up to the next access.  The
        previous 2 ns .. 2 ns + T window instead contained the start-up
        precharge that charges every bitline from its 0 V initial condition
        (about half of the measured "read energy" on an 8x4 array) and cut the
        access off 1 ns before the wordline falls.
        """
        t_period = float(self.t_period)
        t_from = float(1.0 @ u_ns) + 0.2 * t_period
        simulator.measure(
            'TRAN', energy_name,
            f'INTEG {{-V(VDD)*I(VVDD)}} FROM={t_from} TO={t_from + t_period}'
        )
        simulator.measure(
            'TRAN', 'PAVG',
            f'PARAM={{{energy_name}/{t_period}}}'
        )
        self._add_static_power_measures(simulator)

    # def add_xyce_options(self, circuit, mc_runs, operation):
    #     """ Add options for Xyce """
    #     pass

    def add_analysis(self, circuit, operation, num_mc):
        """ Add .DC / .TRAN analysis DC 扫描/瞬态分析"""
        if self.variation_mode == 'per-device' and any((
                self.sweep_cell, self.sweep_precharge, self.sweep_senseamp,
                self.sweep_wordlinedriver, self.sweep_columnmux,
                self.sweep_writedriver, self.sweep_decoder)):
            raise ValueError('Local mismatch requires a separate deck per geometry; '
                             'Xyce 7.4 does not execute the combined .STEP/.SAMPLING grid')
        if 'snm' in operation:
            u_tmp = self.vdd / np.sqrt(2)
            circuit.raw_spice += \
                f'.DC U -{u_tmp:.2f} {u_tmp:.2f} 0.001\n'
        else:
            t_stop = self._analysis_stop(operation)
            # .4e keeps 1 ps below 100 ns; print more digits only when it would move the stop.
            stop_text = f'{t_stop:.4e}' if abs(float(f'{t_stop:.4e}') - t_stop) < 1e-18 else f'{t_stop:.9e}'
            max_step = '' if self.t_max_step is None else f' 0 {float(self.t_max_step):.4e}'
            circuit.raw_spice += f'.TRAN {float(self.t_step):.4e} {stop_text}{max_step}\n'
            # Timing interval option is set only in .TRAN analysis.
            circuit.raw_spice += \
                f'.OPTIONS OUTPUT INITIAL_INTERVAL={float(self.t_step):.4e}\n'
            # Write failed measures as "FAILED" instead of Xyce's default -1.0, which
            # would otherwise be parsed as a valid (negative) delay.
            circuit.raw_spice += '.OPTIONS MEASURE MEASFAIL=1\n'
            for line in self.xyce_options:
                circuit.raw_spice += line.rstrip('\n') + '\n'

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
            if not self.mc:
                circuit.raw_spice += \
                f'.options samples numsamples={num_mc}\n'
            elif num_mc > 1 or self.variation_mode == 'per-device':
                # Latin-hypercube sampling of the AGAUSS(...) model parameters.
                # A fixed seed makes the sweep reproducible; without one Xyce
                # picks a new seed every run (it is printed in the .log).
                seed = '' if self.mc_seed is None else f' seed={int(self.mc_seed)}'
                circuit.raw_spice += \
                f'.SAMPLING useExpr=true\n.options samples numsamples={num_mc}{seed}\n'
            else:
                # A single run is the *nominal* point: without .SAMPLING Xyce
                # evaluates every AGAUSS(...) at its mean, so the result is
                # deterministic.  (Previously a single run was one random
                # process sample with a random seed, so two identical calls
                # returned different delay/power numbers and a "nominal"
                # characterisation could randomly fail.)
                print("[DEBUG] mc_runs=1: no .SAMPLING, model parameters at their nominal values")

        print(f"[DEBUG] Custom_MC={self.custom_mc}, numsamples={num_mc}")

    def gen_process_params(self, circuit: SubCircuitFactory,
                        operation: str, num_mc: int,
                        vars: np.array = None):
        """
        统一生成工艺参数数据表 (支持 6T 和 10T)

        说明：
        1. vars 必须为二维数组，shape = (num_mc, n)
        2. 每一行对应一次 MC 的参数
        3. n 可以是：
        - 单个 cell 的参数数目
            6T: 18
            10T: 30
        - 所有 active cell 展开后的参数总数
            6T: 18 * active_cell_num
            10T: 30 * active_cell_num
        4. 若传入的是单 cell 参数（每行 18 或 30 个值），
        对于多 cell 阵列会自动复制到所有 active cell
        """
        param_names = ['vth0', 'u0', 'voff']
        self.table_head = '.data table\n+ '
        table_content = '\n'
        num_params = 0

        # --- 1. 根据 Cell 类型配置差异化参数 ---
        if self.sram_cell_type == 'SRAM_10T_CELL':
            cell_config = self.sram_config.sram_10t_cell
            mos_names = ['PGL', 'PGR', 'PDL1', 'PDL2', 'PUL',
                        'PDR1', 'PDR2', 'PUR', 'FD_L', 'FD_R']
            pmos_set = {'PUL', 'PUR'}
            pg_nmos_set = {'PGL', 'PGR'}
            fd_nmos_set = {'FD_L', 'FD_R'}

            # 默认调试数据：1组 10T 参数，注意这里只是 1 行
            default_vars_list = [[
                0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13,
                0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13,
                -0.3842, 0.02, -0.126,
                0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13,
                -0.3842, 0.02, -0.126,
                0.4106, 0.045, -0.13, 0.4106, 0.045, -0.13
            ]]
        else:
            cell_config = self.sram_config.sram_6t_cell
            mos_names = ['PGL', 'PGR', 'PDL', 'PUL', 'PDR', 'PUR']
            pmos_set = {'PUL', 'PUR'}
            pg_nmos_set = {'PGL', 'PGR'}
            fd_nmos_set = set()

            # 默认调试数据：1组 6T 参数，注意这里只是 1 行
            default_vars_list = [[
                0.4106, 0.045, -0.13,
                0.4106, 0.045, -0.13,
                0.4106, 0.045, -0.13,
                -0.3842, 0.02, -0.126,
                0.4106, 0.045, -0.13,
                -0.3842, 0.02, -0.126
            ]]

        # --- 2. 统一生成参数表头 ---
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if ('snm' in operation) and (row > 0 or col > 0):
                    continue

                for mos in mos_names:
                    if mos in pmos_set:
                        model_val = cell_config.pmos_model.value
                    elif mos in pg_nmos_set:
                        model_val = cell_config.nmos_model.value[1]  # PG
                    elif mos in fd_nmos_set:
                        model_val = cell_config.nmos_model.value[2]  # FD
                    else:
                        model_val = cell_config.nmos_model.value[0]  # PD

                    for param in param_names:
                        param_def = f'{param}_{model_val}_{mos}_{row:d}_{col:d}'
                        circuit.raw_spice += f'.param {param_def}=0.0\n'
                        self.table_head += f'{param_def} '
                        num_params += 1

        # --- 3. 处理 vars 数据 ---
        if vars is None:
            vars = default_vars_list

        vars = np.array(vars, dtype=float)

        if vars.ndim != 2:
            raise ValueError(
                f'vars must be a 2D array with shape (num_mc, n), got ndim={vars.ndim}'
            )

        if vars.shape[0] != num_mc:
            raise ValueError(
                f'vars row mismatch: expected num_mc={num_mc}, got {vars.shape[0]}'
            )

        params_per_cell = len(mos_names) * len(param_names)

        if 'snm' in operation:
            active_cell_num = 1
        else:
            active_cell_num = self.num_rows * self.num_cols

        expected_total_params = params_per_cell * active_cell_num

        # 每行只给 1 个 cell 的参数：自动复制到所有 active cell
        if vars.shape[1] == params_per_cell:
            if active_cell_num > 1:
                vars = np.tile(vars, (1, active_cell_num))

        # 每行已经给了所有 active cell 展开后的完整参数：直接使用
        elif vars.shape[1] == expected_total_params:
            pass

        else:
            raise ValueError(
                f'vars col mismatch: expected {params_per_cell} '
                f'(one cell) or {expected_total_params} (all active cells), '
                f'got {vars.shape[1]}'
            )

        print(f"[DEBUG] Generated vars.shape={vars.shape}")

        assert len(vars.shape) == 2
        assert num_params == vars.shape[1], \
            f'Cols mismatch: expected {num_params}, got {vars.shape[1]}'

        # --- 4. 格式化数据表 ---
        table_content += "\n".join([
            "+ " + " ".join([f"{x:.4f}" for x in row_data])
            for row_data in vars
        ])

        # --- 5. 保存并包含 ---
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
        
        self.temperature = temperature
        circuit = self.create_testbench(operation, target_row, target_col)
        simulator = circuit.simulator(
        simulator='xyce-serial',
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
        with open(tb_path + '.variation.json', 'w') as f:
            json.dump({**self.variation_summary, 'seed': self.mc_seed, 'samples': mc_runs,
                       'compiler_version': VERSION, 'xyce': shutil.which('Xyce'),
                       'corner': self.corner,
                       'temperature': temperature, 'vdd': float(self.vdd),
                       'model_sha256': hashlib.sha256(Path(getattr(
                           self.sram_config.global_config, f'pdk_path_{self.corner}')).read_bytes()).hexdigest(),
                       'driver_sizes': self.driver_sizes.to_dict(),
                       'timing': self.timing_config.to_dict(),
                       'equivalent': self.equivalent.to_dict(),
                       'full_device_coverage': self.variation_mode == 'per-device'
                                               and self.real_cell_mode == 0}, f, indent=2)
        # assert 0
        # Execute Xyce and parse results
        try:
            log_path = tb_path.replace('.sp', '.log')

            def _run_xyce():
                # command: Xyce <netlist>
                print("[DEBUG] Xyce running ...")
                from utils.xyce import execute_xyce
                res = execute_xyce(
                    tb_path,
                    # ['hspice', '-i', tb_path, '-o', self.sim_path],
                    ['Xyce', tb_path, '-o', tb_path],
                    log_path=log_path,
                )
                return res

            result = _run_xyce()

            if result.returncode != 0:
                raise RuntimeError(
                    f"Xyce simulation failed (exit {result.returncode}):\n{result.stderr.strip()}\n"
                    f"Check log: {log_path}")
            else:
                print(f"[DEBUG] Simulation run successfully (Xyce log: {log_path}).")
        except Exception:
            raise
        else:
            # 根据操作类型选择要在波形图中显示的信号
            if operation == 'read':
                selected_columns = [
                    f'V(S_EN)',
                    f'V(WL{target_row})',
                    f'V(BL{target_col})',
                    f'V(BLB{target_col})',                   
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:Q)' if self.sram_cell_type == 'SRAM_6T_CELL' else f'V(XSRAM_10T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_10T_CELL_{target_row}_{target_col}:Q)',
                    f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:QB)'if self.sram_cell_type == 'SRAM_6T_CELL' else f'V(XSRAM_10T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_10T_CELL_{target_row}_{target_col}:QB)',
                    f'V(SA_Q{target_col // self.mux_in})',
                    f'V(SA_QB{target_col // self.mux_in})',
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
                    f'V(CLK)',
                    # f'V(CSB)',
                    # f'V(WEB)',
                    # f'V(A0)',
                    # f'V(A_DFF0)',
                    f'V(DIN0)',
                    f'V(DIN_DFF0)',
                    f'V(OUT)',
                    # f'V(S_EN)',
                    # f'V(WL{target_row})',
                    # f'V(BL{target_col})',
                    # f'V(BLB{target_col})',                   
                    # f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:Q)' if self.sram_cell_type == 'SRAM_6T_CELL' else f'V(XSRAM_10T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_10T_CELL_{target_row}_{target_col}:Q)',
                    # f'V(XSRAM_6T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_6T_CELL_{target_row}_{target_col}:QB)'if self.sram_cell_type == 'SRAM_6T_CELL' else f'V(XSRAM_10T_CORE_{self.num_rows}X{self.num_cols}:XSRAM_10T_CELL_{target_row}_{target_col}:QB)',
                    # f'V(SA_Q{target_col})',
                    # f'V(SA_QB{target_col})',
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

            if operation == 'read' or operation == 'write' or operation == 'read&write':
                # Get all `.mtX` or `.msX` files from MC
                mc_df = parse_mc_measurements(
                    netlist_prefix=tb_path,
                    file_suffix='ms' if 'snm' in operation else 'mt',
                    num_runs=mc_runs,
                    # value_threshold=1e-9
                )
                print("[DEBUG] Printing mc_df")
                print(mc_df)
                # Keep evidence even when timing safety rejects the run.
                mc_df.to_csv(tb_path + '.raw.data.csv')
                mc_df.to_csv(tb_path.replace('.sp', '.data.csv'))
                self._check_distributed_precharge(mc_df, operation)
                valid = self.access_validity(mc_df, operation, float(self.vdd),
                                             select_every=self.select_every)
                # Preserve the raw CSV above. Failed samples remain in the
                # returned ensemble, as NaN, so optimizers/yield count them.
                delay_name = {'read': 'TREAD_TOTAL', 'write': 'TWRITE_TOTAL',
                              'read&write': 'TVOUT_PERIOD'}[operation]
                mc_df.loc[~valid, delay_name] = np.nan
                # Generate statistics
                stats = generate_mc_statistics(mc_df)
                # Save results
                save_mc_results(
                    mc_df, stats,
                    data_file=tb_path.replace('.sp', '.data.csv'),
                    stats_file=tb_path.replace('.sp', '.stats.csv')
                )
            elif operation in ['hold_snm', 'write_snm', 'read_snm']:
                snm_df_data, _ = process_xyce_montecarlo_prn(
                    prn_path=tb_path + '.prn',
                    metric_name=operation,
                    operation=operation,
                    vdd=float(self.vdd),
                )

            if operation == 'read' or operation == 'write':
                self._print_min_period(tb_path.replace('.sp', '.stats.csv'), operation)

            data_csv_path = tb_path.replace('.sp', '.data.csv')

            # ── Return format compatible with experiment.py / exp_utils.py ──
            # write/read  → (delay, pavg, pstc, pdyn)  all numpy scalar arrays
            # SNM         → scalar SNM value
            import pandas as pd   # numpy is the module-level `np`

            if operation in ('write', 'read', 'read&write'):
                df = pd.read_csv(data_csv_path)

                def _col(name):
                    """Fail loudly instead of turning a missing/failed measure into 0.0."""
                    if name not in df.columns or df[name].isna().all():
                        raise RuntimeError(
                            f"Measurement {name} is missing or FAILED for '{operation}' on "
                            f"the {self.num_rows}x{self.num_cols} array (see {tb_path}.mt0). "
                            f"The access probably did not complete inside the clock window; "
                            f"increase t_period with set_timing_parameters().")
                    n_failed = int(df[name].isna().sum())
                    if n_failed:
                        print(f"[WARNING] {name} FAILED in {n_failed}/{len(df)} MC runs "
                              f"(kept as NaN)")
                    return df[name].values

                # End-to-end access delays.  Summing TDECODER + TPRCH + TSA + ... over-
                # counted: the segments overlap, and decoder / precharge are not on the
                # wl_en -> data path (they resolve in the other clock half).
                delay_name = {'read': 'TREAD_TOTAL',
                              'write': 'TWRITE_TOTAL',
                              'read&write': 'TVOUT_PERIOD'}[operation]
                delay = _col(delay_name)
                pavg = _col('PAVG')
                pstc = _col('PSTC')
                pdyn = _col('PDYN')
                return delay, pavg, pstc, pdyn

            elif operation in ('hold_snm', 'read_snm', 'write_snm'):
                # snm_df_data was returned from process_xyce_montecarlo_prn above
                if operation in snm_df_data.columns:
                    snm_vals = snm_df_data[operation].values
                else:
                    snm_vals = np.array([float('nan')])
                return snm_vals

            else:
                raise KeyError(f"Unknown operation {operation}")
