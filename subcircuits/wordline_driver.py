from PySpice.Spice.Netlist import SubCircuitFactory, SubCircuit
from PySpice.Unit import u_Ohm, u_pF, u_um, u_m

# Import BaseSubcircuit from the specified location
from subcircuits.base_subcircuit import BaseSubcircuit

class Pinv(BaseSubcircuit):
    """
    CMOS Inverter based on sram_16x4_pinv netlist.
    Widths can be dynamically scaled based on num_cols.
    """
    NAME = "PINV"
    NODES = ('VDD', 'VSS', 'A', 'Z') 

    def __init__(self, nmos_model_name, pmos_model_name,
                 base_pmos_width=0.27e-6, base_nmos_width=0.09e-6, length=0.05e-6,
                 num_cols=4, # Number of columns in the SRAM array configuration
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF
                 ):

        super().__init__(
            # name=self.NAME, 
            # nodes=self.NODES, 
            nmos_model_name=nmos_model_name, 
            pmos_model_name=pmos_model_name,
            base_nmos_width=base_nmos_width, # Pass actual width to BaseSubcircuit if it uses it
            base_pmos_width=base_pmos_width, # Pass actual width to BaseSubcircuit if it uses it
            length=length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )

        self.num_cols = num_cols
        # Calculate dynamic widths before calling super().__init__ if BaseSubcircuit uses them
        # or store base widths and calculate actuals after super()
        self.pmos_width = self.calculate_dynamic_width(base_pmos_width, self.num_cols)
        self.nmos_width = self.calculate_dynamic_width(base_nmos_width, self.num_cols)
        # self.transistor_length = length
        
        self.add_inverter_transistors()

    def calculate_dynamic_width(self, base_width, num_cols_config):
        """
        Dynamically adjust the transistor width based on the number of columns.
        Reference configuration is 4 columns.
        """
        # Apply a minimum scaling factor to prevent excessively small transistors
        num_cols_config = 4 if num_cols_config < 4 else num_cols_config

        scaling_factor = num_cols_config / 4.0 # Reference is 4 columns
        return base_width * scaling_factor

    def add_inverter_transistors(self):
        # Mpinv_pmos Z A vdd vdd pmos_vtg m=1 w=0.27u l=0.05u
        self.M('pinv_pmos', 'Z', 'A', 'VDD', 'VDD',
               model=self.pmos_pdk_model,
               w=self.pmos_width, l=self.length)
        # Mpinv_nmos Z A gnd gnd nmos_vtg m=1 w=0.09u l=0.05u
        self.M('pinv_nmos', 'Z', 'A', 'VSS', 'VSS',
               model=self.nmos_pdk_model,
               w=self.nmos_width, l=self.length)

class PNAND2(BaseSubcircuit):
    """
    CMOS NAND2 gate based on sram_16x4_pnand2 netlist in OpenRAM.
    Widths can be dynamically scaled based on num_cols.
    """
    NAME = "PNAND2"
    NODES = ('VDD', 'VSS', 'A', 'B', 'Z')

    def __init__(self, nmos_model_name, pmos_model_name,
                 base_pmos_width=0.27e-6, base_nmos_width=0.18e-6, length=0.05e-6,
                 num_cols=4, # Number of columns in the SRAM array configuration
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):
        
        super().__init__(
            # name=self.NAME, 
            # nodes=self.NODES, 
            nmos_model_name=nmos_model_name, 
            pmos_model_name=pmos_model_name,
            base_nmos_width=base_nmos_width, 
            base_pmos_width=base_pmos_width, 
            length=length, 
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
        )
        
        self.add_nand2_transistors()

    def calculate_dynamic_width(self, base_width, num_cols_config):
        """
        Dynamically adjust the transistor width based on the number of columns.
        Reference configuration is 4 columns.
        """
        if num_cols_config <= 2:
            num_cols_config = 4
            
        scaling_factor = num_cols_config / 4.0 
        return base_width * scaling_factor

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


class WordlineDriver(BaseSubcircuit):
    """
    Wordline driver circuit based on sram_16x4_wordline_driver netlist.
    It consists of a NAND2 gate followed by an Inverter.
    The sizes of the NAND and Inverter can be scaled based on num_cols.
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
            # name=self.NAME, 
            # nodes=self.NODES, 
            nmos_model_name=nmos_model_name, 
            pmos_model_name=pmos_model_name,
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
        self.subcircuit(self.nand_gate)
        
        # This is the inverter for driving WLs
        self.inv_driver = Pinv(nmos_model_name=nmos_model_name,
                               pmos_model_name=pmos_model_name,
                               base_pmos_width=base_inv_pmos_width,
                               base_nmos_width=base_inv_nmos_width,
                               length=length,
                               num_cols=self.num_cols) # Pass num_cols for dynamic sizing
        self.subcircuit(self.inv_driver)

        self.add_driver_components()

    def add_driver_components(self):
        if self.w_rc:
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
               'VDD', 'VSS', a_node, a_node, 'zb_int')
        self.X(self.inv_driver.name, self.inv_driver.name,
               'VDD', 'VSS', zb_node, z_node)