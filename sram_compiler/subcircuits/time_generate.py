from PySpice.Unit import u_Ohm, u_pF
from .base_subcircuit import BaseSubcircuit
from math import ceil, log2
from .standard_cell import Pinv,AND2,PNAND2,AND3
        
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
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        # 创建不同尺寸的反相器
        self.inv1 = Pinv(nmos_model, pmos_model,0.09e-6,0.27e-6,0.05e-6,num=1)
        self.inv2 = Pinv(nmos_model, pmos_model,0.27e-6,0.81e-6,0.05e-6,num=2)
        self.inv3 = Pinv(nmos_model, pmos_model,0.36e-6,1.08e-6,0.05e-6,num=3)
        self.inv4 = Pinv(nmos_model, pmos_model,0.56e-6,1.69e-6,0.05e-6,num=4)
        
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
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        # 创建不同尺寸的反相器
        self.inv1 = Pinv(nmos_model, pmos_model,0.09e-6,0.27e-6,0.05e-6,num=1)
        self.inv2 = Pinv(nmos_model, pmos_model,0.27e-6,0.81e-6,0.05e-6,num=2)
        
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
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        # 创建不同尺寸的反相器
        self.inv1 = Pinv(nmos_model, pmos_model,0.09e-6,0.27e-6,0.05e-6,num=1)
        self.inv2 = Pinv(nmos_model, pmos_model,0.45e-06,1.35e-06,0.05e-6,num=2)
        
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
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,operation='read'
                 ):
        # 计算需要的地址位数
        n_bits = ceil(log2(num_rows)) if num_rows > 1 else 1
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

        nodes += ['rbl','rbl_delay','rbl_delay_bar','s_en','w_en','PRE']
        self.NODES = nodes

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        self.num_rows=num_rows
        self.num_cols=num_cols
        self.n_bits = n_bits
        #触发器在时钟上升沿触发地址信号
        dff_buf_addr=ADDR_DFF(nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG",num_rows=self.num_rows)
        self.subcircuit(dff_buf_addr)
            # 构建地址DFF连接列表
        addr_dff_connections = ['VDD', 'VSS', 'clk_buf']  # 基本连接
            # 添加地址输入连接
        for i in range(self.n_bits):
            addr_dff_connections.append(f'A{i}')
            # 添加地址输出连接
        for i in range(self.n_bits):
            addr_dff_connections.append(f'A_dff{i}')
            # 实例化DFF
        self.X('dff_buf_addr',
               dff_buf_addr.NAME, *addr_dff_connections)

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
        clkbuf = pdrive(
            nmos_model="NMOS_VTG",
            pmos_model="PMOS_VTG"
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
        wl_en=wl_pdrive()
        self.subcircuit(wl_en)
        self.X('wl_en',
               wl_en.NAME,
               'VDD', 'VSS', 'gated_clk_bar', 'wl_en')
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
        
         #产生写使能
        w_en=AND3(nmos_model_nand="NMOS_VTG",
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
        self.X('w_en',
               w_en.NAME,
               'VDD','VSS' ,'rbl_delay_bar', 'gated_clk_bar' ,'we', 'w_en' )
        #产生灵敏放大器
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
        self.X('s_en',
               s_en.NAME,
               'VDD','VSS' ,'rbl_delay', 'gated_clk_bar' ,'we_bar' ,'s_en' )

        #产生预充电使能
        pre_unbuf=PNAND2(nmos_model="NMOS_VTG",
                        pmos_model="PMOS_VTG",
                        nmos_width=0.18e-6,
                        pmos_width=0.27e-6,
                        length=0.05e-6,
                        w_rc=w_rc
                        )
        self.subcircuit(pre_unbuf)
        self.X('pre_unbuf',
               pre_unbuf.NAME,
               'VDD','VSS', 'gated_clk_buf', 'rbl_delay', 'PRE_UNBUF')
        pre=pdrive2_for_pre()
        self.subcircuit(pre)
        self.X('pre',
               pre.NAME,
               'VDD','VSS', 'PRE_UNBUF', 'PRE')
       
