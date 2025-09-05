from PySpice.Spice.Netlist import SubCircuitFactory, SubCircuit
from PySpice.Unit import u_Ohm, u_pF, u_um, u_m

# Import BaseSubcircuit from the specified location
from .base_subcircuit import BaseSubcircuit
# 从标准单元库导入
from .standard_cell import  PNAND2_for_wordline_driver,Pinv_for_wordline_driver # type: ignore



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
                 sweep_wordlinedriver = False,
                pmos_modle_choices = 'PMOS_VTG',
                nmos_modle_choices = 'MOS_VTG'
                 ):
        
        super().__init__(
            nmos_model_name, pmos_model_name,
            base_nand_nmos_width, base_nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        self.num_cols = num_cols
        self.sweep_wordlinedriver= sweep_wordlinedriver
        self.pmos_modle_choices=pmos_modle_choices
        self.nmos_modle_choices=nmos_modle_choices

        # This is the nand gate
        self.nand_gate =  PNAND2_for_wordline_driver(nmos_model_name=nmos_model_name, 
                                pmos_model_name=pmos_model_name,
                                base_pmos_width=base_nand_pmos_width,
                                base_nmos_width=base_nand_nmos_width,
                                length=length,
                                num_cols=self.num_cols,
                                sweep_wordlinedriver=self.sweep_wordlinedriver,
                                pmos_modle_choices = self.pmos_modle_choices,
                                nmos_modle_choices = self.nmos_modle_choices
                                ) # Pass num_cols for dynamic sizing
        self.subcircuit(self.nand_gate) #添加与非门电路
        
        # This is the inverter for driving WLs
        self.inv_driver = Pinv_for_wordline_driver(nmos_model_name=nmos_model_name,
                               pmos_model_name=pmos_model_name,
                               base_pmos_width=base_inv_pmos_width,
                               base_nmos_width=base_inv_nmos_width,
                               length=length,
                               num_cols=self.num_cols,
                               sweep_wordlinedriver=self.sweep_wordlinedriver,
                               pmos_modle_choices = self.pmos_modle_choices,
                               nmos_modle_choices = self.nmos_modle_choices
                               ) # Pass num_cols for dynamic sizing
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
