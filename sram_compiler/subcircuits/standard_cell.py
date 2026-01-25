from PySpice.Spice.Netlist import SubCircuitFactory, SubCircuit
from PySpice.Unit import u_Ohm, u_pF, u_um, u_m
from .base_subcircuit import BaseSubcircuit

class Pinv(BaseSubcircuit):
    """
    Standard CMOS Inverter
    NODES: VDD, VSS, A (Input), Z (Output)
    """
    NODES = ('VDD', 'VSS', 'A', 'Z')

    def __init__(self, nmos_model, pmos_model, 
                 nmos_width, pmos_width, length,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,num=''):
        
        self.NAME = f"PINV{num}"
        super().__init__(
            nmos_model, pmos_model, 
            nmos_width, pmos_width, length, 
            w_rc, pi_res, pi_cap)
        
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length
        
        self.add_inverter_transistors()

    def add_inverter_transistors(self):
        self.M('pinv_pmos', 'Z', 'A', 'VDD', 'VDD', 
            model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M('pinv_nmos', 'Z', 'A', 'VSS', 'VSS', 
            model=self.nmos_model, w=self.nmos_width, l=self.length)
        
class PNAND2(BaseSubcircuit):
    """
    Standard CMOS 2-input NAND Gate
    NODES: VDD, VSS, A, B, Z
    """
    NAME = "PNAND2"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')

    def __init__(self, nmos_model, pmos_model, 
                 nmos_width, pmos_width, length,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):
        
        super().__init__(
            nmos_model, pmos_model, 
            nmos_width, pmos_width, length, 
            w_rc, pi_res, pi_cap
        )

        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length
        
        self.add_nand2_transistors()

    def add_nand2_transistors(self):
        # PMOS (Parallel)
        self.M('pnand2_pmos1', 'Z', 'A', 'VDD', 'VDD', 
               model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M('pnand2_pmos2', 'Z', 'B', 'VDD', 'VDD', 
               model=self.pmos_model, w=self.pmos_width, l=self.length)
        # NMOS (Series)
        self.M('pnand2_nmos1', 'Z', 'B', 'net1', 'VSS', 
               model=self.nmos_model, w=self.nmos_width, l=self.length)
        self.M('pnand2_nmos2', 'net1', 'A', 'VSS', 'VSS', 
               model=self.nmos_model, w=self.nmos_width, l=self.length)
        
class PNAND3(BaseSubcircuit):
    """
    Standard CMOS 3-input NAND Gate
    NODES: VDD, VSS, A, B, C, Z
    """
    NAME = "PNAND3"
    NODES = ('VDD', 'VSS', 'A', 'B', 'C', 'Z')

    def __init__(self, nmos_model, pmos_model, 
                 nmos_width, pmos_width, length,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):
        
        super().__init__(
            nmos_model, pmos_model, 
            nmos_width, pmos_width, length, 
            w_rc, pi_res, pi_cap
        )
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length

        self.add_nand3_transistors()

    def add_nand3_transistors(self):
        # PMOS (Parallel)
        self.M('pnand3_pmos1', 'Z', 'A', 'VDD', 'VDD', 
               model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M('pnand3_pmos2', 'Z', 'B', 'VDD', 'VDD', 
               model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M('pnand3_pmos3', 'Z', 'C', 'VDD', 'VDD', 
               model=self.pmos_model, w=self.pmos_width, l=self.length)
        # NMOS (Series)
        self.M('pnand3_nmos1', 'Z', 'A', 'net1', 'VSS', 
               model=self.nmos_model, w=self.nmos_width, l=self.length)
        self.M('pnand3_nmos2', 'net1', 'B', 'net2', 'VSS', 
               model=self.nmos_model, w=self.nmos_width, l=self.length)
        self.M('pnand3_nmos3', 'net2', 'C', 'VSS', 'VSS', 
               model=self.nmos_model, w=self.nmos_width, l=self.length)

class Pbuff(BaseSubcircuit):  # 两个反相器级联构成的缓冲器
    """
    CMOS Buffer (2-stage inverter chain) based on PINV.
    """
    NAME = "PBUFF"
    NODES = ('VDD', 'VSS', 'A', 'Z')

    def __init__(self, nmos_model, pmos_model,
                 nmos_width, pmos_width, length,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):
        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.pmos_width = pmos_width
        self.nmos_width = nmos_width
        self.length = length

        self.add_buffer_transistors()

    def add_buffer_transistors(self):
        # The internal node is named Z_int (first-level output / second-level input)
        # First-level inverter: Input A, Output Z_int
        self.M('buff_pmos_1', 'Z_int', 'A', 'VDD', 'VDD',
               model=self.pmos_model,
               w=self.pmos_width, l=self.length)

        self.M('buff_nmos_1', 'Z_int', 'A', 'VSS', 'VSS',
               model=self.nmos_model,
               w=self.nmos_width, l=self.length)

        # Second-level inverter: Input Z_int, Output Z
        self.M('buff_pmos_2', 'Z', 'Z_int', 'VDD', 'VDD',
               model=self.pmos_model,
               w=self.pmos_width, l=self.length)

        self.M('buff_nmos_2', 'Z', 'Z_int', 'VSS', 'VSS',
               model=self.nmos_model,
               w=self.nmos_width, l=self.length)

class AND2(BaseSubcircuit):
    """
    AND2 gate  generation based on SPICE netlist.
    Consists of a PNAND2 followed by a PINV.
    """
    NAME = "AND2"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')

    def __init__(self, nmos_model_nand, pmos_model_nand,
                 nmos_model_inv, pmos_model_inv,
                 # Base widths for NAND gate transistors
                 nand_pmos_width=0.27e-6, nand_nmos_width=0.18e-6,
                 # Base widths for Inverter transistors
                 inv_pmos_width=0.27e-6, inv_nmos_width=0.09e-6, length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF
                 ):

        super().__init__(
            nmos_model_nand, pmos_model_nand,
            nand_nmos_width, nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
   
        self.nmos_model_nand = nmos_model_nand
        self.pmos_model_nand = pmos_model_nand
        self.nmos_model_inv = nmos_model_inv
        self.pmos_model_inv = pmos_model_inv

        self.nand_pmos_width = nand_pmos_width
        self.nand_nmos_width = nand_nmos_width
        self.inv_pmos_width = inv_pmos_width
        self.inv_nmos_width = inv_nmos_width
        self.length = length

        self.w_rc = w_rc

        # This is the nand gate
        self.nand_gate = PNAND2(nmos_model=self.nmos_model_nand,
                                pmos_model=self.pmos_model_nand,
                                pmos_width=self.nand_pmos_width,
                                nmos_width=self.nand_nmos_width,
                                length=self.length,                                
                                )
        self.subcircuit(self.nand_gate)  # Add a NAND gate circuit

        # This is the inverter for driving WLs
        self.inv_driver =  Pinv(nmos_model=self.nmos_model_inv,
                                pmos_model=self.pmos_model_inv,
                                pmos_width=self.inv_pmos_width,
                                nmos_width=self.inv_nmos_width,
                                length=self.length,
                                )
        self.subcircuit(self.inv_driver)  # Add an inverter circuit

        self.add_and3_components()

    def add_and3_components(self):
        if self.w_rc:  # 字线要考虑是否添加rc网络，
            a_node = self.add_rc_networks_to_node('A', num_segs=2)  # 调用base里的rc网络函数
            b_node = self.add_rc_networks_to_node('B', num_segs=2)  # 4条线每条分成两段加rc
            zb_node = self.add_rc_networks_to_node('zb_int', num_segs=2)
            z_node = self.add_rc_networks_to_node('Z', num_segs=2)
        else:
            a_node = 'A'
            b_node = "B"
            zb_node = "zb_int"
            z_node = "Z"
        """ Instantiate the `PNAND3` and `Pinv` gates """ 
        self.X(f'PNAND3', self.nand_gate.name,
               'VDD', 'VSS', a_node,b_node, zb_node)
        self.X(f'PINV', self.inv_driver.name,
               'VDD', 'VSS', zb_node,z_node)

class AND3(BaseSubcircuit):
    """
    AND3 gate generation based on SPICE netlist.
    Consists of a PNAND3 followed by a PINV.
    """
    NAME = "AND3"
    NODES = ('VDD', 'VSS', 'A', 'B', 'C', 'Z')

    def __init__(self, nmos_model_nand, pmos_model_nand,
                 nmos_model_inv, pmos_model_inv,
                 # Base widths for NAND gate transistors
                 nand_pmos_width=0.27e-6, nand_nmos_width=0.18e-6,
                 # Base widths for Inverter transistors
                 inv_pmos_width=0.27e-6, inv_nmos_width=0.09e-6, length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF
                 ):

        super().__init__(
            nmos_model_nand, pmos_model_nand,
            nand_nmos_width, nand_pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
   
        self.nmos_model_nand = nmos_model_nand
        self.pmos_model_nand = pmos_model_nand
        self.nmos_model_inv = nmos_model_inv
        self.pmos_model_inv = pmos_model_inv
        self.nand_pmos_width = nand_pmos_width
        self.nand_nmos_width = nand_nmos_width
        self.inv_pmos_width = inv_pmos_width
        self.inv_nmos_width = inv_nmos_width
        self.length = length

        # This is the 3-input nand gate
        self.nand3_gate = PNAND3(nmos_model=self.nmos_model_nand,
                                 pmos_model=self.pmos_model_nand,
                                 pmos_width=self.nand_pmos_width,
                                 nmos_width=self.nand_nmos_width,
                                 length=self.length,                                
                                 )
        self.subcircuit(self.nand3_gate)  # Add a 3-input NAND gate circuit

        # This is the inverter to convert NAND to AND
        self.inv_driver =  Pinv(nmos_model=self.nmos_model_inv,
                                pmos_model=self.pmos_model_inv,
                                pmos_width=self.inv_pmos_width,
                                nmos_width=self.inv_nmos_width,
                                length=self.length,
                                )
        self.subcircuit(self.inv_driver)  # Add an inverter circuit

        self.add_and3_components()

    def add_and3_components(self):
        """ Instantiate the `PNAND3` and `Pinv` gates """ 
        self.X('PNAND3', self.nand3_gate.name,
               'VDD', 'VSS', 'A', 'B', 'C', "zb_int")
        self.X('PINV', self.inv_driver.name,
               'VDD', 'VSS',"zb_int", 'Z')
        
class D_latch(BaseSubcircuit):
    """
    D latch implemented using Pinv_for_latch and PNAND2_for_latch components.
    This is a transparent latch that captures the data when CLK is high.
    """
    NAME = "D_LATCH"
    NODES = ('VDD', 'VSS', 'D', 'EN', 'Q', 'QB')

    def __init__(self, nmos_model, pmos_model,
                 pmos_width=0.27e-6, nmos_width=0.18e-6, length=0.05e-6,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                ):
        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc, pi_res, pi_cap,
        )
        
        # Store parameters for subcircuits
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.pmos_width = pmos_width
        self.nmos_width = nmos_width
        self.length = length

        
        # Create the required subcircuits
        # NAND gates for the master latch
        self.nand1 = PNAND2(
                            nmos_model=self.nmos_model,
                            pmos_model=self.pmos_model,
                            pmos_width=self.pmos_width,
                            nmos_width=self.nmos_width,
                            length=self.length,
                            )
        
        # Two inverters for the slave latch
        self.inv1  =     Pinv(
                            nmos_model=self.nmos_model,
                            pmos_model=self.pmos_model,
                            pmos_width=self.pmos_width,
                            nmos_width=self.nmos_width,
                            length=self.length,
                            )
        
        # Add subcircuits to the main circuit
        self.subcircuit(self.nand1)
        self.subcircuit(self.inv1)
        # Connect the components
        self.add_d_latch_components()

    def add_d_latch_components(self):
        # Define node names
        d_node = 'D'
        db_node = 'DB'
        en_node = 'EN'
        q_node = 'Q'
        qb_node = 'QB'
        int1_node = 'INT1'  # Internal node between nand1 and nand2
        int2_node = 'INT2'  # Internal node between nand2 and inv1
        
        # Instantiate the NAND gates
        # First NAND: inputs D and CLK, output INT1
                # First inverter: input INT2, output Q
        self.X('INV1', self.inv1.name, 'VDD', 'VSS', d_node, db_node)
        self.X('NAND1', self.nand1.name, 'VDD', 'VSS', d_node, en_node, int1_node)
        self.X('NAND2', self.nand1.name, 'VDD', 'VSS', db_node, en_node, int2_node)
        self.X('NAND3', self.nand1.name, 'VDD', 'VSS', int1_node, qb_node, q_node)
        self.X('NAND4', self.nand1.name, 'VDD', 'VSS', int2_node, q_node, qb_node)
