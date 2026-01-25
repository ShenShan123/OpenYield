from PySpice.Unit import u_Ohm, u_pF
# Import BaseSubcircuit from the specified location
from .base_subcircuit import BaseSubcircuit
# 从标准单元库导入
from .standard_cell import Pinv,PNAND2 # type: ignore

class WordlineDriver(BaseSubcircuit):   #总的字线驱动器=一个与非门加一个反相器
    """
    Wordline driver circuit based on sram_16x4_wordline_driver netlist.
    It consists of a NAND2 gate followed by an Inverter.
    The sizes of  Inverter can be scaled based on num_cols.
    """
    NAME = "WORDLINEDRIVER"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')  

    def __init__(self, nmos_model_inv, pmos_model_inv,nmos_model_nand, pmos_model_nand,
                 # Base widths for NAND gate transistors
                 nand_pmos_width=0.27e-6, nand_nmos_width=0.18e-6,
                 # Base widths for Inverter transistors
                 inv_pmos_width=0.27e-6, inv_nmos_width=0.09e-6,
                 length=0.05e-6, 
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):
        
        super().__init__(
            nmos_model_inv, pmos_model_inv,
            nand_nmos_width, nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        self.nmos_model_inv = nmos_model_inv
        self.pmos_model_inv = pmos_model_inv
        self.nmos_model_nand = nmos_model_nand
        self.pmos_model_nand = pmos_model_nand
        self.inv_pmos_width = inv_pmos_width
        self.inv_nmos_width = inv_nmos_width
        self.nand_nmos_width = nand_nmos_width
        self.nand_pmos_width = nand_pmos_width
        self.length = length
        self.w_rc = w_rc    
        
        # This is the nand gate
        self.nand_gate =  PNAND2(nmos_model=self.nmos_model_nand, 
                                pmos_model=self.pmos_model_nand,
                                pmos_width=self.nand_pmos_width,
                                nmos_width=self.nand_nmos_width,
                                length=self.length,
                                ) 
        self.subcircuit(self.nand_gate) 
        
        # This is the inverter for driving WLs
        self.inv_driver = Pinv(nmos_model=self.nmos_model_inv,
                               pmos_model=self.pmos_model_inv,
                               pmos_width=self.inv_pmos_width,
                               nmos_width=self.inv_nmos_width,
                               length=self.length,
                               ) 
        self.subcircuit(self.inv_driver)  

        self.add_driver_components()

    def add_driver_components(self):
        if self.w_rc:         #The line should consider whether to add the rc network.
            a_node = self.add_rc_networks_to_node('A', num_segs=2)  
            b_node = self.add_rc_networks_to_node('B', num_segs=2)  
            zb_node = self.add_rc_networks_to_node('zb_int', num_segs=2)
            z_node = self.add_rc_networks_to_node('Z', num_segs=2)
        else:
            a_node = 'A'
            b_node = 'B'
            zb_node = 'zb_int'
            z_node = 'Z'


        """ Instantiate the `PNAND2` and `Pinv` gates """       
        self.X(self.nand_gate.name, self.nand_gate.name, 
               'VDD', 'VSS', a_node, b_node, 'zb_int')         
        self.X(self.inv_driver.name, self.inv_driver.name,
               'VDD', 'VSS', zb_node, z_node)
