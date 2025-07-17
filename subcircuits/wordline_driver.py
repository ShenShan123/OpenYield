from PySpice.Spice.Netlist import SubCircuitFactory, SubCircuit
from PySpice.Unit import u_Ohm, u_pF, u_um, u_m

# Import BaseSubcircuit from the specified location
from .base_subcircuit import BaseSubcircuit

class Pinv(BaseSubcircuit): #单个反相器,不分pd,pu,pg
    """
    CMOS Inverter based on sram_16x4_pinv netlist.
    Widths can be dynamically scaled based on num_cols.宽度可以根据列的数量动态缩放
    列数多,同一条字线上连的cell单元就多,驱动能力要强,晶体管宽度就要大
    """
    NAME = "PINV"
    NODES = ('VDD', 'VSS', 'A', 'Z') 

    def __init__(self, nmos_model_name, pmos_model_name,
                 base_pmos_width=0.27e-6, base_nmos_width=0.09e-6, length=0.05e-6,
                 num_cols=4, # Number of columns in the SRAM array configuration    #列
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF
                 ):

        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nmos_width, base_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )

        self.num_cols = num_cols
        # Calculate dynamic widths before calling super().__init__ if BaseSubcircuit uses them
        # or store base widths and calculate actuals after super()
        self.pmos_width = self.calculate_dynamic_width(base_pmos_width, self.num_cols)
        self.nmos_width = self.calculate_dynamic_width(base_nmos_width, self.num_cols)
        # self.transistor_length = length
        
        self.add_inverter_transistors() #添加反相器单元函数

    def calculate_dynamic_width(self, base_width, num_cols_config):
        """
        Dynamically adjust the transistor width based on the number of columns.
        Reference configuration is 4 columns.根据列数动态调整晶体管宽度。
        参考配置为 4 列
        """
        # Apply a minimum scaling factor to prevent excessively small transistors
        num_cols_config = 4 if num_cols_config < 4 else num_cols_config #最小按4列去配置宽度

        scaling_factor = num_cols_config / 4.0 # Reference is 4 columns #缩放因子
        return base_width * scaling_factor

    def add_inverter_transistors(self): #反相器即一个pmos+一个nmos
        # Mpinv_pmos Z A vdd vdd pmos_vtg m=1 w=0.27u l=0.05u
        self.M('pinv_pmos', 'Z', 'A', 'VDD', 'VDD',
               model=self.pmos_pdk_model,
               w=self.pmos_width, l=self.length)
        # Mpinv_nmos Z A gnd gnd nmos_vtg m=1 w=0.09u l=0.05u
        self.M('pinv_nmos', 'Z', 'A', 'VSS', 'VSS',
               model=self.nmos_pdk_model,
               w=self.nmos_width, l=self.length)

class PNAND2(BaseSubcircuit):   #单个2输入与非门
    """
    CMOS NAND2 gate based on sram_16x4_pnand2 netlist in OpenRAM.
    Widths can be dynamically scaled based on num_cols.
    不需要晶体管宽度根据列数调整，因为不起驱动作用
    """
    NAME = "PNAND2"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')

    def __init__(self, nmos_model_name, pmos_model_name,
                 base_pmos_width=0.27e-6, base_nmos_width=0.18e-6, length=0.05e-6,
                 num_cols=4, # Number of columns in the SRAM array configuration
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):
        
        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nmos_width, base_pmos_width, length, 
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )

        self.add_nand2_transistors()
    #不需要尺寸调整函数
    # def calculate_dynamic_width(self, base_width, num_cols_config):
    #     """
    #     Dynamically adjust the transistor width based on the number of columns.
    #     Reference configuration is 4 columns.
    #     """
    #     if num_cols_config <= 2:        #和PINV里的定义不一样？
    #         num_cols_config = 4
            
    #     scaling_factor = num_cols_config / 4.0 
    #     return base_width * scaling_factor

    def add_nand2_transistors(self):
        self.M('pnand2_pmos1', 'VDD', 'A', 'Z', 'VDD',
               model=self.pmos_pdk_model,
               w=self.base_pmos_width, l=self.length)
        self.M('pnand2_pmos2', 'Z', 'B', 'VDD', 'VDD',
               model=self.pmos_pdk_model,
               w=self.base_pmos_width, l=self.length)
        
        self.M('pnand2_nmos1', 'Z', 'B', 'net1_nand', 'VSS', 
               model=self.nmos_pdk_model,
               w=self.base_nmos_width, l=self.length)
        self.M('pnand2_nmos2', 'net1_nand', 'A', 'VSS', 'VSS',
               model=self.nmos_pdk_model,
               w=self.base_nmos_width, l=self.length)


class WordlineDriver(BaseSubcircuit):   #总的字线驱动器=一个与非门加一个反相器
    """
    Wordline driver circuit based on sram_16x4_wordline_driver netlist.
    It consists of a NAND2 gate followed by an Inverter.
    The sizes of  Inverter can be scaled based on num_cols.
    INV的尺寸可以根据num_cols进行缩放。
    """
    NAME = "WORDLINEDRIVER"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')  

    def __init__(self, nmos_model_name, pmos_model_name,
                 # Base widths for NAND gate transistors
                 base_nand_pmos_width=0.27e-6, base_nand_nmos_width=0.18e-6,
                 # Base widths for Inverter transistors
                 base_inv_pmos_width=0.27e-6, base_inv_nmos_width=0.09e-6,
                 length=0.05e-6, 
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 num_cols=4, # Number of columns this driver is intended for
                 ):
        
        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nand_nmos_width, base_nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        self.num_cols = num_cols

        # This is the nand gate
        self.nand_gate = PNAND2(nmos_model_name=nmos_model_name, 
                                pmos_model_name=pmos_model_name,
                                base_pmos_width=base_nand_pmos_width,
                                base_nmos_width=base_nand_nmos_width,
                                length=length,
                                num_cols=self.num_cols) # Pass num_cols for dynamic sizing
        self.subcircuit(self.nand_gate) #添加与非门电路
        
        # This is the inverter for driving WLs
        self.inv_driver = Pinv(nmos_model_name=nmos_model_name,
                               pmos_model_name=pmos_model_name,
                               base_pmos_width=base_inv_pmos_width,
                               base_nmos_width=base_inv_nmos_width,
                               length=length,
                               num_cols=self.num_cols) # Pass num_cols for dynamic sizing
        self.subcircuit(self.inv_driver)    #添加反相器电路

        self.add_driver_components()

    def add_driver_components(self):
        if self.w_rc:                                               #字线要考虑是否添加rc网络，
            a_node = self.add_rc_networks_to_node('A', num_segs=2)  #调用base里的rc网络函数
            b_node = self.add_rc_networks_to_node('B', num_segs=2)  #4条线每条分成两段加rc
            zb_node = self.add_rc_networks_to_node('zb_int', num_segs=2)
            z_node = self.add_rc_networks_to_node('Z', num_segs=2)
        else:
            a_node = 'A'
            b_node = 'B'
            zb_node = 'zb_int'
            z_node = 'Z'

        """ Instantiate the `PNAND2` and `Pinv` gates """       #实例化
        self.X(self.nand_gate.name, self.nand_gate.name, 
               'VDD', 'VSS', a_node, a_node, 'zb_int')          #两个输入都是A
        self.X(self.inv_driver.name, self.inv_driver.name,
               'VDD', 'VSS', zb_node, z_node)
