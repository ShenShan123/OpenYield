from PySpice.Unit import u_Ohm, u_pF
from .base_subcircuit import BaseSubcircuit
from math import ceil, log2
from .standard_cell import Pinv,AND2,PNAND2,PNOR2,AND3,PNAND3,D_latch


class TaperedBuffer(BaseSubcircuit):
    """Non-inverting inverter chain for a large fan-out.

    The output stage is `drive_scale` unit inverters (0.09 / 0.27 um) and the
    stages are tapered geometrically from the input, so every stage sees a
    fan-out of drive_scale ** (1 / n_stages): two stages up to a scale of 16
    (fan-out <= 4 per stage), four stages above.  `name` must be unique per
    scope because PySpice keeps one subcircuit definition per name.
    """
    NODES = ('VDD', 'VSS', 'A', 'Z')

    def __init__(self, name, drive_scale=1.0,
                 nmos_model="NMOS_VTG", pmos_model="PMOS_VTG", length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):
        self.NAME = name
        super().__init__(
            nmos_model, pmos_model,
            0.09e-6, 0.27e-6, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        drive_scale = max(1.0, float(drive_scale))
        n_stages = 2 if drive_scale <= 16 else 4
        self.drive_scale = drive_scale
        self.n_stages = n_stages
        prev = 'A'
        for k in range(n_stages):
            s = drive_scale ** ((k + 1) / n_stages)
            inv = Pinv(nmos_model, pmos_model, 0.09e-6 * s, 0.27e-6 * s, length,
                       num=f'_{name}_{k}')
            self.subcircuit(inv)
            out = 'Z' if k == n_stages - 1 else f'b{k}'
            self.X(f'inv{k}', inv.NAME, 'VDD', 'VSS', prev, out)
            prev = out


class D_latch_addr(D_latch):
    """D_LATCH with its own subcircuit name (address hold latch of TIME)."""
    NAME = "D_LATCH_ADDR"


class TransmissionGate(BaseSubcircuit):
    """
    传输门 (Transmission Gate)
    由一对NMOS和PMOS晶体管组成,实现信号传输
    """
    NAME = "TRANSMISSION_GATE"
    NODES = ('VDD', 'VSS', 'IN', 'OUT', 'CTR_P', 'CTR_N')
    
    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 pmos_width=5e-07, nmos_width=2.5e-07, length=5e-08,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF
                 ):
        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        self.nmos_model= nmos_model
        self.pmos_model= pmos_model
        self.nmos_width=nmos_width
        self.pmos_width=pmos_width
        self.length=length
        self.add_transmission_gate()
    
    def add_transmission_gate(self):
        """添加传输门晶体管"""
        
        # PMOS晶体管 - 由CTR_P控制
        self.M('transpmos', 'OUT', 'CTR_P', 'IN', 'VDD',
               model=self.pmos_model, w=self.pmos_width, l=self.length)
        # NMOS晶体管 - 由CTR_N控制
        self.M('transnmos', 'IN', 'CTR_N', 'OUT', 'VSS',
               model=self.nmos_model, w=self.nmos_width, l=self.length)
   

class pdrive(BaseSubcircuit):  # ////////缓冲器链，由一系列尺寸逐渐增大的反相器组成，增强时钟信号的驱动能力，提供陡峭的时钟边沿，并减少时钟 skew

    NAME = "pdrive"
    NODES = ('VDD', 'VSS', 'A', 'Z')

    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 pmos_width=0.27e-6, nmos_width=0.18e-6,
                 length=0.05e-6,
                 drive_scale=1.0,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        drive_scale = max(1.0, float(drive_scale))
        s1 = drive_scale ** 0.25
        s2 = drive_scale ** 0.50
        s3 = drive_scale ** 0.75
        s4 = drive_scale
        # 创建不同尺寸的反相器
        self.inv1 = Pinv(nmos_model, pmos_model,0.09e-6 * s1,0.27e-6 * s1,0.05e-6,num=1)
        self.inv2 = Pinv(nmos_model, pmos_model,0.27e-6 * s2,0.81e-6 * s2,0.05e-6,num=2)
        self.inv3 = Pinv(nmos_model, pmos_model,0.91e-6 * s3,2.43e-6 * s3,0.05e-6,num=3)
        self.inv4 = Pinv(nmos_model, pmos_model,2.43e-6 * s4,7.29e-6 * s4,0.05e-6,num=4)
        
        # 添加子电路
        self.subcircuit(self.inv1)
        self.subcircuit(self.inv2)
        self.subcircuit(self.inv3)
        self.subcircuit(self.inv4)
        
        # 添加内部节点
        # self.node('zb1_node')
        # self.node('zb2_node')
        # self.node('zb3_node')
        
        # 构建缓冲器链
        self.add_buffer_chain()

    def add_buffer_chain(self):
        """构建四级缓冲器链"""
        # 第一级: 尺寸=1
        self.X('buf_inv1', self.inv1.NAME,
               'VDD', 'VSS', 'A', 'zb1_node')
        
        # 第二级: 尺寸=3
        self.X('buf_inv2', self.inv2.NAME,
               'VDD', 'VSS', 'zb1_node', 'zb2_node')
        
        # 第三级: 尺寸=8
        self.X('buf_inv3', self.inv3.NAME,
               'VDD', 'VSS', 'zb2_node', 'zb3_node')
        
        # 第四级: 尺寸=25
        self.X('buf_inv4', self.inv4.NAME,
               'VDD', 'VSS', 'zb3_node', 'Z')
        
class pdrive2_for_pre(BaseSubcircuit):  # ////////缓冲器链，由一系列尺寸逐渐增大的反相器组成，增强时钟信号的驱动能力，提供陡峭的时钟边沿，并减少时钟 skew

    NAME = "pdrive2_for_pre"
    NODES = ('VDD', 'VSS', 'A', 'Z')

    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 pmos_width=0.27e-6, nmos_width=0.18e-6,
                 length=0.05e-6,
                 drive_scale=1.0,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )

        drive_scale = max(1.0, float(drive_scale))
        stage1_scale = max(1.0, drive_scale ** 0.5)
        stage2_scale = drive_scale
        
        # 创建不同尺寸的反相器
        self.inv1 = Pinv(nmos_model, pmos_model,0.09e-6 * stage1_scale,0.27e-6 * stage1_scale,0.05e-6,num=1)
        self.inv2 = Pinv(nmos_model, pmos_model,0.27e-6 * stage2_scale,0.81e-6 * stage2_scale,0.05e-6,num=2)
        
        # 添加子电路
        self.subcircuit(self.inv1)
        self.subcircuit(self.inv2)
     
        # 构建缓冲器链
        self.add_buffer_chain()

    def add_buffer_chain(self):
        """构建两级缓冲器链"""
        # 第一级: 尺寸=1
        self.X('buf_inv1', self.inv1.NAME,
               'VDD', 'VSS', 'A', 'zb1_node')
        
        # 第二级: 尺寸=3
        self.X('buf_inv2', self.inv2.NAME,
               'VDD', 'VSS', 'zb1_node', 'Z')
        
class wl_pdrive(BaseSubcircuit):  # ////////用于字线驱动的缓冲器

    NAME = "wl_pdrive"
    NODES = ('VDD', 'VSS', 'A', 'Z')

    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 # Base widths for NAND gate transistors
                 pmos_width=0.27e-6, nmos_width=0.18e-6,
                 length=0.05e-6,
                 drive_scale=1.0,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )

        # Both stages scale with the load (num_rows wordline-driver NAND2 gates):
        # with the fixed 1.35/0.45 um output stage the wl_en edge took 280 ps to
        # rise and 600 ps to fall at 512 rows (40/90 ps at 64 rows).
        drive_scale = max(1.0, float(drive_scale))
        # 创建不同尺寸的反相器
        self.inv1 = Pinv(nmos_model, pmos_model,0.09e-6 * drive_scale,0.27e-6 * drive_scale,0.05e-6,num=1)
        self.inv2 = Pinv(nmos_model, pmos_model,0.45e-06 * drive_scale,1.35e-06 * drive_scale,0.05e-6,num=2)
        
        # 添加子电路
        self.subcircuit(self.inv1)
        self.subcircuit(self.inv2)

        # 构建缓冲器链
        self.add_buffer_chain()

    def add_buffer_chain(self):
        """构建2级缓冲器链"""
        # 第一级: 尺寸=1
        self.X('buf_inv1', self.inv1.NAME,
               'VDD', 'VSS', 'A', 'zb1_node')
        
        # 第二级: 尺寸=3
        self.X('buf_inv2', self.inv2.NAME,
               'VDD', 'VSS', 'zb1_node', 'Z')

        
class dff(BaseSubcircuit):  # ////////构建传输门型触发器

    NAME = "DFF"
    NODES = ('VDD', 'VSS', 'D', 'Q','CLK')

    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 # Base widths for NAND gate transistors
                 pmos_width=5e-07, nmos_width=2.5e-07,
                 length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        ) 
        self.inv_dff = Pinv(nmos_model,pmos_model,2.5e-07, 5e-07,0.05e-6,num=1)
        self.subcircuit(self.inv_dff)

        self.trans_dff = TransmissionGate(
            nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG"
        )
        self.subcircuit(self.trans_dff)
        # 构建传输门型触发器
        self.add_dff()      

    def add_dff(self):
        """构建传输门D触发器"""
        # 1. 时钟反相器 - 生成clk_b
        self.X('inv1_clk', self.inv_dff.NAME,
               'VDD', 'VSS', 'CLK', 'CLKB')
        # 2. 主锁存器 - 第一部分 
        #    信号反相器 - 生成D_b
        self.X('inv2_D', self.inv_dff.NAME,
               'VDD', 'VSS', 'D', 'D_b')
        # T1传输门 - 当CLK=0时导通
        self.X('tg1', self.trans_dff.NAME,
               'VDD', 'VSS', 'D_b', 'z1', 'CLK', 'CLKB')
        
        # 反相器3
        self.X('inv3', self.inv_dff.NAME,
               'VDD', 'VSS', 'z1', 'z2')
        
        # 反相器4 (反馈)
        self.X('inv4', self.inv_dff.NAME,
               'VDD', 'VSS', 'z2', 'z3')
        #T2传输门（反馈）
        self.X('tg2', self.trans_dff.NAME,
               'VDD', 'VSS', 'z3', 'z1', 'CLKB', 'CLK')
        
        # 3. 第二部分 (T3和T4)
         # 反相器5 
        self.X('inv5', self.inv_dff.NAME,
               'VDD', 'VSS', 'z2', 'z4')
        # T3传输门 - 当CLK=1时导通 (反馈路径)
        self.X('tg3', self.trans_dff.NAME,
               'VDD', 'VSS', 'z4', 'z5', 'CLKB', 'CLK')
        
        # 反相器6
        self.X('inv6', self.inv_dff.NAME,
               'VDD', 'VSS', 'z5', 'Q')
        
        # 反相器7 (反馈)
        self.X('inv7', self.inv_dff.NAME,
               'VDD', 'VSS', 'Q', 'QB')
        
        # T4传输门 - 当CLK=0时导通 (反馈路径)
        self.X('tg4', self.trans_dff.NAME,
               'VDD', 'VSS', 'QB', 'z5', 'CLK', 'CLKB')
        
class DFF_BUF(BaseSubcircuit):
    """D Flip-Flop with output buffers"""#加两级缓冲增加驱动能力
    NAME = "DFF_BUF"
    NODES = ('VDD', 'VSS', 'D', 'Q', 'QB', 'CLK')
    
    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 pmos_width=5e-07, nmos_width=2.5e-07,
                 length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        ) 
        
        # 创建DFF和反相器
        self.dff1 = dff(nmos_model, pmos_model, length)
        self.inv1 = Pinv(nmos_model, pmos_model,0.18e-6, 0.54e-6,length=0.05e-6,num=1)  # 尺寸2
        self.inv2 = Pinv(nmos_model, pmos_model,0.36e-6, 1.08e-6,length=0.05e-6,num=2)  # 尺寸4
        
        self.subcircuit(self.dff1)
        self.subcircuit(self.inv1)
        self.subcircuit(self.inv2)
        
        self.add_dff_buf()
    
    def add_dff_buf(self):
        """构建带缓冲的DFF"""
        # DFF
        self.X('dff', self.dff1.NAME, 
               'VDD', 'VSS', 'D', 'qint', 'CLK')
        
        # 第一个反相器（尺寸2）- 产生QB
        self.X('inv1', self.inv1.NAME,
               'VDD', 'VSS', 'qint', 'QB')
        
        # 第二个反相器（尺寸4）- 产生缓冲后的Q
        self.X('inv2', self.inv2.NAME,
               'VDD', 'VSS', 'QB', 'Q')
        
class DelayChain(BaseSubcircuit):#用于复制位线延迟的延迟链
    """
    延迟链电路 (sram_delay_chain)
    输入: in, VDD, VSS
    输出: out
    """
    NAME = "delay_chain"
    NODES = ('VDD', 'VSS', 'in', 'out')
    
    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 pmos_width=5e-07, nmos_width=2.5e-07,
                 length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        ) 
        
        # 创建基本反相器单元
        self.inv = Pinv(nmos_model, pmos_model,0.9e-07,2.7e-07, length=0.05e-6,num=1)
        self.subcircuit(self.inv)
        
        # 添加内部节点
        #self.add_internal_nodes()
        
        # 构建延迟链
        self.add_delay_chain()
    
    def add_delay_chain(self):
        """构建延迟链电路"""
        # 第一级反相器
        self.X('dinv0', self.inv.NAME, 'VDD', 'VSS', 'in', 'dout_1')
        
        # 第一级的4个负载
        for j in range(4):
            self.X(f'dload_0_{j}', self.inv.NAME, 
                   'VDD', 'VSS', 'dout_1', f'n_0_{j}')
        
        # 中间7级反相器 (第2级到第8级)
        for i in range(1, 8):
            # 反相器
            self.X(f'dinv{i}', self.inv.NAME, 
                   'VDD', 'VSS', f'dout_{i}', f'dout_{i+1}')
            
            # 负载
            for j in range(4):
                self.X(f'dload_{i}_{j}', self.inv.NAME, 
                       'VDD', 'VSS', f'dout_{i+1}', f'n_{i}_{j}')
        
        # 最后一级反相器 (第9级)
        self.X('dinv8', self.inv.NAME, 'VDD', 'VSS', 'dout_8', 'out')
        
        # 最后一级的4个负载
        for j in range(4):
            self.X(f'dload_8_{j}', self.inv.NAME, 
                   'VDD', 'VSS', 'out', f'n_8_{j}')
            
class WenDelayChain(BaseSubcircuit):
    NAME = "wen_delay_chain"
    NODES = ('VDD', 'VSS', 'in', 'out')

    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 length=0.05e-6, stages=4, loads_per_stage=4,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):

        stages = max(2, int(stages))
        if stages % 2:
            stages += 1

        self.stages = stages
        self.loads_per_stage = loads_per_stage

        super().__init__(
            nmos_model, pmos_model,
            0.09e-6, 0.27e-6, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )

        self.inv = Pinv(
            nmos_model, pmos_model,
            nmos_width=0.09e-6,
            pmos_width=0.27e-6,
            length=length,
            num='_wen_delay'
        )
        self.subcircuit(self.inv)
        self.add_delay_chain()

    def add_delay_chain(self):
        prev = 'in'

        for i in range(self.stages):
            out = 'out' if i == self.stages - 1 else f'wen_dly_{i + 1}'

            self.X(f'winv{i}', self.inv.NAME,
                   'VDD', 'VSS', prev, out)

            for j in range(self.loads_per_stage):
                self.X(f'wload_{i}_{j}', self.inv.NAME,
                       'VDD', 'VSS', out, f'wn_{i}_{j}')

            prev = out

class ADDR_DFF(BaseSubcircuit):
    """D Flip-Flop for address"""
    NAME = "ADDR_DFF"
    #NODES = ('VDD', 'VSS', 'D', 'Q', 'QB', 'CLK')
    
    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 # Base widths for NAND gate transistors
                 pmos_width=5e-07, nmos_width=2.5e-07,
                 length=0.05e-6,num_rows=16,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):
        self.num_rows = num_rows
        n_bits = ceil(log2(self.num_rows)) if self.num_rows > 1 else 1
        # 动态生成节点：包括所有地址输入、输出
        nodes = ['VDD', 'VSS', 'CLK']
        # 添加地址输入节点 A0, A1, A2, ..
        nodes.extend([f'A{i}' for i in range(n_bits)])
        # 添加地址输出节点 Q0, Q1, Q2, ...
        nodes.extend([f'A_dff{i}' for i in range(n_bits)]) 
        self.NODES = nodes

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        ) 
        n_bits = ceil(log2(self.num_rows)) if self.num_rows > 1 else 1
        self.dff_addr = dff(nmos_model, pmos_model)
        self.subcircuit(self.dff_addr)
         # 构建地址DFF阵列
        self.add_addr_dff_array(n_bits)
    
    def add_addr_dff_array(self, n_bits):
        # 创建DFF和反相器
        for i in range(n_bits):
            self.X(f'dff_{i}', self.dff_addr.NAME, 
                   'VDD', 'VSS', f'A{i}', f'A_dff{i}', 'CLK')
            
class DATA_DFF(BaseSubcircuit):
    """D Flip-Flop for data"""
    NAME = "DATA_DFF"
    # NODES会动态生成
    
    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 pmos_width=5e-07, nmos_width=2.5e-07,
                 length=0.05e-6, num_cols=8,  # 默认16x8结构，8列数据
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):
        self.num_cols = num_cols
        
        # 动态生成节点：包括所有数据输入、输出和电源/时钟
        nodes = ['VDD', 'VSS', 'CLK']
        # 添加数据输入节点 DIN0, DIN1, ...
        nodes.extend([f'DIN{i}' for i in range(num_cols)])
        # 添加数据输出节点 DIN_dff0, DIN_dff1, ...
        nodes.extend([f'DIN_dff{i}' for i in range(num_cols)])
        self.NODES = nodes

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        # 创建单个DFF实例作为模板
        self.dff_data = dff(nmos_model, pmos_model, length)
        self.subcircuit(self.dff_data)
        
        # 构建数据DFF阵列
        self.add_data_dff_array(num_cols)
    
    def add_data_dff_array(self, num_cols):
        # 为每个数据位创建DFF
        for i in range(num_cols):
            self.X(f'dff_{i}', self.dff_data.NAME, 
                   'VDD', 'VSS', f'DIN{i}', f'DIN_dff{i}', 'CLK')


class AND2_WEN(AND2):
    """AND2 with its own subcircuit name (write-enable gate of TIME)."""
    NAME = "AND2_WEN"


class TIME(BaseSubcircuit):
    """
    时序信号生成
    输入:VDD, VSS, clk
    输出:clk_buf
    """
    NAME = "TIME"
    #NODES = ('VDD', 'VSS', 'clk', 'csb','web', 'clk_buf','clk_bar','cs','gated_clk_bar','gated_clk_buf','wl_en')

    def __init__(self, nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                 # Base widths for NAND gate transistors
                 pmos_width=0.27e-6, nmos_width=0.18e-6,
                 length=0.05e-6,num_rows=16,num_cols=8,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,operation='read',
                 num_sa=None, wl_load=None, pre_load=None, wen_load=None,
                 ):
        """
        num_sa:   number of sense amplifiers driven by s_en (num_cols / mux_in);
                  default num_cols.
        wl_load:  load on wl_en in units of a 0.18/0.27 um NAND2 gate, i.e.
                  num_rows * (wordline-driver NAND2 scale); default num_rows.
        pre_load: load on PRE in unit (0.09/0.27 um) inverter inputs, i.e.
                  (num_cols + 1) * 3 * precharge PMOS width / 0.36 um; default
                  assumes the 0.27 um base width scaled with max(0.5, rows/16).
        wen_load: load on w_en in unit inverter inputs: per column the write
                  driver's EN inverter and two EN-gated NMOS (3 * nmos_w +
                  pmos_w, row-scaled) plus the testbench's w_en_bar inverter;
                  default assumes the 0.18/0.36 um base widths scaled with
                  max(8, rows)/16.
        """
        # 计算需要的地址位数
        n_bits = ceil(log2(num_rows)) if num_rows > 1 else 1
        num_sa = num_cols if num_sa is None else int(num_sa)
        wl_load = float(num_rows) if wl_load is None else float(wl_load)
        if pre_load is None:
            pre_load = (num_cols + 1) * 3 * 0.27e-6 * max(0.5, num_rows / 16.0) / 0.36e-6
        pre_load = float(pre_load)
        if wen_load is None:
            wen_load = num_cols * (3 * 0.18e-6 + 0.36e-6) * max(8, num_rows) / 16.0 / 0.36e-6
        wen_load = float(wen_load)
         # 动态生成节点
        nodes = ['VDD', 'VSS', 'clk', 'csb', 'web', 'clk_buf', 'clk_bar', 
                'cs_bar','cs', 'we_bar','we','gated_clk_bar', 'gated_clk_buf', 'wl_en']
        
        # 添加地址输入输出节点
        nodes.extend([f'A{i}' for i in range(n_bits)])
        nodes.extend([f'A_dff{i}' for i in range(n_bits)])
        if operation == 'write' or operation == 'read&write':
            # 添加数据输入输出节点
            nodes.extend([f'DIN{i}' for i in range(num_cols)])
            nodes.extend([f'DIN_dff{i}' for i in range(num_cols)])

        nodes += ['rbl','rbl_delay','rbl_delay_bar','s_en','w_en','PRE','sa_iso']
        self.NODES = nodes

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.num_rows=num_rows
        self.num_cols=num_cols
        self.n_bits = n_bits
        self.num_sa = num_sa
        self.wl_load = wl_load
        self.pre_load = pre_load
        self.wen_load = wen_load
        #触发器在时钟上升沿触发地址信号
        dff_buf_addr=ADDR_DFF(nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG",num_rows=self.num_rows)
        self.subcircuit(dff_buf_addr)
            # 构建地址DFF连接列表
        addr_dff_connections = ['VDD', 'VSS', 'clk_buf']  # 基本连接
            # 添加地址输入连接
        for i in range(self.n_bits):
            addr_dff_connections.append(f'A{i}')
            # 添加地址输出连接 (register outputs; A_dff{i} is the held / buffered
            # address below)
        for i in range(self.n_bits):
            addr_dff_connections.append(f'A_reg{i}')
            # 实例化DFF
        self.X('dff_buf_addr',
               dff_buf_addr.NAME, *addr_dff_connections)

        # Address hold latch + fan-out buffer -> A_dff{i} (decoder input).
        #
        # The register A_reg updates ~100-150 ps after the clock edge that ends
        # an access, while wl_en (and with it the old wordline) is still
        # falling; a changed address therefore raised the *new* row's wordline
        # for the tail of the old access and wrote the old bitline data into
        # that row (measured at 64-512 rows).  The latch is transparent while
        # wl_en is low and holds the address while the wordline is on, so a
        # new decoder output can only rise after wl_en_bar is high again, i.e.
        # after the wordline driver has been disabled.  (Same scheme as the
        # write-data hold latch of the testbench.)
        #
        # The decoder input load grows with the row count (5 gate inputs per
        # last-level 3-to-8 decoder for the low address bits: 320 gates at
        # 512 rows, 650 ps register edge before this change), so the latch
        # output is buffered for a fan-out of ~8 per stage.
        addr_latch = D_latch_addr(nmos_model="NMOS_VTG", pmos_model="PMOS_VTG")
        self.subcircuit(addr_latch)
        addr_fanout_units = 5 * ceil(self.num_rows / 8.0)      # 0.09/0.27 um gate equivalents
        addr_scale = max(1, ceil(addr_fanout_units / 8.0 / 3.0))  # 3-unit output per 8 loads
        addr_buf = TaperedBuffer('ABUF', drive_scale=addr_scale)
        self.subcircuit(addr_buf)
        for i in range(self.n_bits):
            self.X(f'addr_hold_{i}', addr_latch.NAME,
                   'VDD', 'VSS', f'A_reg{i}', 'wl_en_bar', f'A_lat{i}', f'A_latb{i}')
            self.X(f'addr_buf_{i}', addr_buf.NAME,
                   'VDD', 'VSS', f'A_lat{i}', f'A_dff{i}')

        if operation == 'write' or operation == 'read&write':
            #触发器在时钟上升沿触发数据信号
            dff_buf_data=DATA_DFF(nmos_model="NMOS_VTG",
                pmos_model="PMOS_VTG",num_cols=self.num_cols)  # 对于16x8结构，有8位数据
            self.subcircuit(dff_buf_data)
            
            # 构建数据DFF连接列表
            data_dff_connections = ['VDD', 'VSS', 'clk_buf']  # 基本连接
            # 添加数据输入连接
            for i in range(self.num_cols):  
                data_dff_connections.append(f'DIN{i}')
            # 添加数据输出连接
            for i in range(self.num_cols):  
                data_dff_connections.append(f'DIN_dff{i}')
            # 实例化DFF
            self.X('dff_buf_data',
                dff_buf_data.NAME, *data_dff_connections)


        #产生内部时钟
        # 让 clkbuf 按实际 DFF 负载自动放大
        ref_rows = 16
        ref_cols = 16
        ref_bits = ceil(log2(ref_rows))

        clk_dff_count = self.n_bits + 2  # 地址DFF + CS_DFF + WE_DFF
        ref_dff_count = ref_bits + 2

        if operation == 'write' or operation == 'read&write':
            clk_dff_count += self.num_cols
            ref_dff_count += ref_cols

        clk_drive_scale = max(1.0, clk_dff_count / ref_dff_count)

        clkbuf = pdrive(
            nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG",
            drive_scale=clk_drive_scale
        )
        self.subcircuit(clkbuf)
        self.X('clkbuf',
               clkbuf.NAME,
               'VDD', 'VSS', 'clk', 'clk_buf')
        #产生内部主时钟的反信号
        inv_clk_bar = Pinv(
            nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG",
            nmos_width=0.09e-6,
            pmos_width=0.27e-6,
            length=0.05e-6,
        )
        self.subcircuit(inv_clk_bar)
        self.X('inv_clk_bar',
               inv_clk_bar.NAME,
               'VDD', 'VSS', 'clk_buf', 'clk_bar')
        #触发器在时钟上升沿触发片选信号
        dff_buf=DFF_BUF(nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG")
        self.subcircuit(dff_buf)
        self.X('dff_buf',
               dff_buf.NAME,
               'VDD', 'VSS', 'csb', 'cs_bar','cs','clk_buf')
        #触发器在时钟上升沿触发写信号
        dff_buf1=DFF_BUF(nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG")
        self.subcircuit(dff_buf1)
        self.X('dff_buf1',
               dff_buf1.NAME,
               'VDD', 'VSS', 'web', 'we_bar','we','clk_buf')
        #门控时钟（反相）
        and2_gated_clk_bar=AND2(nmos_model_nand="NMOS_VTG",
                                pmos_model_nand="PMOS_VTG",
                                nmos_model_inv="NMOS_VTG",
                                pmos_model_inv="PMOS_VTG",
                                nand_pmos_width=0.27e-6,
                                nand_nmos_width=0.18e-6,
                                inv_pmos_width=1.62e-6,
                                inv_nmos_width=0.54e-6,
                                length=0.05e-6,
                                w_rc=w_rc
                                )
        self.subcircuit(and2_gated_clk_bar)
        self.X('and2_gated_clk_bar',
            and2_gated_clk_bar.NAME,
            'VDD', 'VSS', 'cs', 'clk_bar','gated_clk_bar')
        #门控时钟
        and2_gated_clk_buf=AND2(nmos_model_nand="NMOS_VTG",
                                pmos_model_nand="PMOS_VTG",
                                nmos_model_inv="NMOS_VTG",
                                pmos_model_inv="PMOS_VTG",
                                nand_pmos_width=0.27e-6,
                                nand_nmos_width=0.18e-6,
                                inv_pmos_width=1.62e-6,
                                inv_nmos_width=0.54e-6,
                                length=0.05e-6,
                                w_rc=w_rc
                                )
        self.subcircuit(and2_gated_clk_buf)
        self.X('and2_gated_clk_buf',
               and2_gated_clk_buf.NAME,
               'VDD', 'VSS', 'cs', 'clk_buf','gated_clk_buf')
        #字线使能，在clk的低电平
        # wl_en drives one NAND2 input per row in the wordline drivers (plus the
        # replica-wordline AND2).  The output stage (1.35/0.45 um per unit)
        # drives 4 unit NAND2 loads at a fan-out of 1, so one unit per 32 loads
        # keeps the fan-out <= 8: unchanged up to 16x16, 4x at 64x16, 16x at
        # 512x4.  Without it the wl_en edge was 280 ps (rise) / 600 ps (fall)
        # at 512 rows, which is also what opened the address-change hazard.
        wl_en_scale = max(1, ceil(self.wl_load / 32.0))
        wl_en=wl_pdrive(drive_scale=wl_en_scale)
        self.subcircuit(wl_en)
        self.X('wl_en',
               wl_en.NAME,
               'VDD', 'VSS', 'gated_clk_bar', 'wl_en')
        # wl_en_bar enables the address hold latches (2 NAND2 inputs per bit)
        # and the precharge NAND3; size it for that fan-out.
        wlb_scale = max(1, ceil((2 * self.n_bits + 1) / 5.0))
        inv_wl_en_bar = Pinv(
            nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG",
            nmos_width=0.09e-6 * wlb_scale,
            pmos_width=0.27e-6 * wlb_scale,
            length=0.05e-6,
            num='_wl_en_bar'
        )
        self.subcircuit(inv_wl_en_bar)
        self.X('inv_wl_en_bar',
            inv_wl_en_bar.NAME,
            'VDD', 'VSS', 'wl_en', 'wl_en_bar')

        #复制位线延迟链
        delaychain=DelayChain()
        self.subcircuit(delaychain)
        self.X('delaychain',
               delaychain.NAME,
               'VDD', 'VSS', 'rbl', 'rbl_delay')
        #复制位线延迟反相
        inv_rbl_delay_bar = Pinv(
            nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG",
            nmos_width=0.09e-6,
            pmos_width=0.27e-6,
            length=0.05e-6,
        )
        self.subcircuit(inv_rbl_delay_bar)
        self.X('inv_rbl_delay_bar',
               inv_rbl_delay_bar.NAME,
               'VDD', 'VSS', 'rbl_delay', 'rbl_delay_bar')
        
        #产生写使能: w_en = gated_clk_bar & we, i.e. the write drivers stay on
        # for the whole clock-low (wordline) phase, exactly like the wordline.
        #
        # Previously w_en was also gated by rbl_delay_bar, so the write pulse
        # ended as soon as the *replica cell* had discharged the replica
        # bitline (~250 ps).  That replica path (cell pull-down through the
        # pass gate) is stronger than the write path (row-scaled write driver,
        # optionally through the column-mux transmission gate), so the pulse
        # only had ~30 % margin nominally and Monte Carlo samples with a weak
        # NMOS left the bitline at 0.3-0.4 V when w_en ended: the cell kept
        # its old data.  The hard-coded 16x512 WenDelayChain that used to
        # lengthen the pulse for one array size is no longer needed.
        # w_en buffer.  The AND2's 4-unit inverter (1.08/0.36 um) drives up to
        # 32 unit loads directly (fan-out <= 8); above that a TaperedBuffer
        # sized for a fan-out of ~8 follows it.  V2.0.1 scaled the inverter
        # with columns/64 and rows/16, which still left a fan-out of ~40-60:
        # the w_en edge was 130-180 ps (10-90 %) from 64x16 up to 16x512 and
        # the driven bitline reached VDD/2 only 120-240 ps after gated_clk_bar,
        # 40-60 ps after the wordline (the write waited for w_en).
        w_en=AND2_WEN(nmos_model_nand="NMOS_VTG",
                pmos_model_nand="PMOS_VTG",
                nmos_model_inv="NMOS_VTG",
                pmos_model_inv="PMOS_VTG",
                nand_pmos_width=0.27e-6,
                nand_nmos_width=0.18e-6,
                inv_pmos_width=1.08e-6,
                inv_nmos_width=0.36e-6,
                length=0.05e-6,
                w_rc=w_rc
                )
        self.subcircuit(w_en)
        if self.wen_load <= 32:
            wen_src = 'w_en'
            self.X('w_en',
                   w_en.NAME,
                   'VDD','VSS' , 'gated_clk_bar' ,'we', 'w_en' )
        else:
            wen_src = 'w_en_unbuf'
            self.X('w_en',
                   w_en.NAME,
                   'VDD','VSS' , 'gated_clk_bar' ,'we', 'w_en_unbuf' )
            wen_buf = TaperedBuffer('WEN_BUF', drive_scale=ceil(self.wen_load / 8.0))
            self.subcircuit(wen_buf)
            self.X('w_en_buf', wen_buf.NAME, 'VDD', 'VSS', 'w_en_unbuf', 'w_en')
        #产生灵敏放大器
        # s_en enables the sense-amplifier footers (one 0.27 um NMOS gate, ~0.75
        # unit loads per amplifier) and the output latch; sa_iso (below) drives
        # the input pass gates.  The AND3's 4-unit inverter drives up to 32 unit
        # loads directly (fan-out <= 8), above that a TaperedBuffer follows.
        # V2.0.1 scaled one 4-unit inverter per 64 columns for footer *and*
        # pass gates (fan-out ~75): s_en edge 190-230 ps (10-90 %) at >= 64
        # columns and a 0.2-0.3 V precharge-coupling bump on s_en.
        s_en=AND3(nmos_model_nand="NMOS_VTG",
                pmos_model_nand="PMOS_VTG",
                nmos_model_inv="NMOS_VTG",
                pmos_model_inv="PMOS_VTG",
                nand_pmos_width=0.27e-6,
                nand_nmos_width=0.18e-6,
                inv_pmos_width=1.08e-6,
                inv_nmos_width=0.36e-6,
                length=0.05e-6,
                w_rc=w_rc
            )
        self.subcircuit(s_en)
        sen_load = 0.75 * self.num_sa + 2.5 + 1.0   # footers + output latch + NOR below
        if sen_load <= 32:
            sen_src = 's_en'
            self.X('s_en',
                   s_en.NAME,
                   'VDD','VSS' ,'rbl_delay', 'gated_clk_bar' ,'we_bar' ,'s_en' )
        else:
            sen_src = 's_en_unbuf'
            self.X('s_en',
                   s_en.NAME,
                   'VDD','VSS' ,'rbl_delay', 'gated_clk_bar' ,'we_bar' ,'s_en_unbuf' )
            sen_buf = TaperedBuffer('SEN_BUF', drive_scale=ceil(sen_load / 8.0))
            self.subcircuit(sen_buf)
            self.X('s_en_buf', sen_buf.NAME, 'VDD', 'VSS', 's_en_unbuf', 's_en')

        # Sense-amplifier input isolation: sa_iso = s_en | w_en (PMOS pass
        # gates off while the amplifier is fired *or* the write drivers are
        # on).  Before V2.0.2 the pass gates were driven by s_en alone, so the
        # cross-coupled PMOS pair of the amplifier stayed connected to the
        # bitlines during a write and acted as a keeper: the write only
        # succeeded while w_en rose before the wordline (60 ps margin at 2x128
        # in V2.0.1; with the faster wordline path of V2.0.2 the order flipped
        # and the 2x128 write deadlocked, BL 0.27 V / BLB 0.9 V).  Two pass
        # gates (4/3 x 0.54 um) per amplifier = ~4 unit loads.
        sa_iso_nor = PNOR2(nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                           nmos_width=0.09e-6, pmos_width=0.54e-6, length=0.05e-6,
                           w_rc=w_rc)
        self.subcircuit(sa_iso_nor)
        self.X('sa_iso_nor', sa_iso_nor.NAME, 'VDD', 'VSS', sen_src, wen_src, 'sa_iso_bar')
        sa_iso_inv = Pinv(nmos_model="NMOS_VTG", pmos_model="PMOS_VTG",
                          nmos_width=0.36e-6, pmos_width=1.08e-6, length=0.05e-6,
                          num='_sa_iso')
        self.subcircuit(sa_iso_inv)
        iso_load = 4.0 * self.num_sa
        if iso_load <= 32:
            self.X('sa_iso_inv', sa_iso_inv.NAME, 'VDD', 'VSS', 'sa_iso_bar', 'sa_iso')
        else:
            self.X('sa_iso_inv', sa_iso_inv.NAME, 'VDD', 'VSS', 'sa_iso_bar', 'sa_iso_unbuf')
            iso_buf = TaperedBuffer('ISO_BUF', drive_scale=ceil(iso_load / 8.0))
            self.subcircuit(iso_buf)
            self.X('sa_iso_buf', iso_buf.NAME, 'VDD', 'VSS', 'sa_iso_unbuf', 'sa_iso')

        #产生预充电使能
        # PRE (active low) = NAND3(clk_buf, cs, wl_en_bar): the bitlines are
        # precharged for the whole clock-high phase of a selected cycle and
        # released as soon as the clock falls, ~4 gate delays before the
        # wordline rises (the wl_en_bar term keeps the precharge off while any
        # wordline is on).
        #
        # Previously PRE = NAND3(gated_clk_buf, rbl_delay, wl_en_bar) was a
        # self-timed pulse of ~300 ps that ended when the replica bitline had
        # been recharged; afterwards every bitline floated for the rest of the
        # cycle and leaked through the off pass gates of the cells storing a
        # 0 on that side.  Measured before the change (nominal, 8x4 / 16x16):
        # RBL 0.89 V at the next access with a 50 ns clock, 0.76 V with 100 ns
        # (TT 25 C); with the default 10 ns clock RBL 0.87-0.90 V at TT 125 C
        # and 0.74-0.77 V at FF 125 C, BL 0.88-0.92 V against BLB 0.99-1.00 V
        # (an 80-120 mV offset before the read).  The replica-timed sensing
        # still resolved every read, but the bitline level at the start of an
        # access depended on the cycle time and the corner.  Holding the
        # bitlines costs no dynamic energy (the charge that leaked away had to
        # be replaced by the next pulse anyway); the PSTC window now includes
        # the bitline leakage, which is supplied through the precharge devices.
        pre_unbuf = PNAND3(nmos_model="NMOS_VTG",
                   pmos_model="PMOS_VTG",
                   nmos_width=0.27e-6,
                   pmos_width=0.27e-6,
                   length=0.05e-6,
                   w_rc=w_rc
                   )
        self.subcircuit(pre_unbuf)
        self.X('pre_unbuf',
            pre_unbuf.NAME,
            'VDD', 'VSS', 'clk_buf', 'cs', 'wl_en_bar', 'PRE_UNBUF')

        # PRE buffer sized for its load: 3 PMOS gates per precharge cell,
        # num_cols + 1 cells (replica column included), PMOS width scaled with
        # the row count by PrechargeFactory; `pre_load` is that load in unit
        # (0.09/0.27 um) inverter inputs.  The previous 2-stage buffer had a
        # fan-out of ~49 per unit (PRE only reached 0.05-0.08 V on the largest
        # arrays and its edge was 120-150 ps).
        pre_scale = max(1, ceil(self.pre_load / 8.0))
        pre = TaperedBuffer('PRE_BUF', drive_scale=pre_scale)
        self.subcircuit(pre)
        self.X('pre',
               pre.NAME,
               'VDD','VSS', 'PRE_UNBUF', 'PRE')
       


        
        
if __name__ == '__main__':
    top = TIME()
    with open('time.sp', 'w') as f:
        f.write(str(top))
    print("已生成 time.sp")
    print(top)
