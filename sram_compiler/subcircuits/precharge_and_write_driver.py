from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF, u_V, u_ns
from .base_subcircuit import BaseSubcircuit        

class Precharge(BaseSubcircuit):
    """
    Standard Precharge Circuit (Topology Only).
    """
    NAME = "PRECHARGE"
    NODES = ('VDD', 'ENB', 'BL', 'BLB') 

    def __init__(self, pmos_model, pmos_width, length, 
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):
        
        # Precharge 只有 PMOS，NMOS 传 None 即可
        super().__init__(
            None, pmos_model,
            pmos_width, pmos_width, length,
            w_rc, pi_res, pi_cap,
        )
        
        self.pmos_model = pmos_model
        self.pmos_width = pmos_width
        self.length = length
        self.w_rc = w_rc
        
        self.add_precharge_transistors()

    def add_precharge_transistors(self):

        if self.w_rc:
            bl_node = self.add_rc_networks_to_node('BL', 2)
            blb_node = self.add_rc_networks_to_node('BLB', 2)
            enb_node = self.add_rc_networks_to_node('ENB', 1)
        else:
            bl_node, blb_node, enb_node = 'BL', 'BLB', 'ENB'
        
        # M1: Precharge BL
        self.M(1, bl_node, enb_node, 'VDD', 'VDD',
               model=self.pmos_model, w=self.pmos_width, l=self.length)
        
        # M2: Precharge BLB
        self.M(2, blb_node, enb_node, 'VDD', 'VDD',
               model=self.pmos_model, w=self.pmos_width, l=self.length)
        
        # M3: Equalization (BL <-> BLB)
        self.M(3, bl_node, enb_node, blb_node, 'VDD',
               model=self.pmos_model, w=self.pmos_width, l=self.length)

            
class WriteDriver(BaseSubcircuit):
    """
    Standard Write Driver (Topology Only).
    """
    NAME = "WRITEDRIVER"
    NODES = ('VDD', 'VSS', 'EN', 'DIN', 'BL', 'BLB') 

    def __init__(self, nmos_model, pmos_model,
                 nmos_width, pmos_width, length,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):
        
        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc, pi_res, pi_cap,
        )
        
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length
        self.w_rc = w_rc
        
        self.add_driver_transistors()

    def add_driver_transistors(self):

        if self.w_rc:
            d_node = self.add_rc_networks_to_node('DIN', 1)
            db_node = self.add_rc_networks_to_node('DINB', 1) 
            bl_node = self.add_rc_networks_to_node('BL', 2)
            blb_node = self.add_rc_networks_to_node('BLB', 2)
            en_node = self.add_rc_networks_to_node('EN', 1)
            enb_node = self.add_rc_networks_to_node('ENB', 1)
        else:
            d_node, db_node = 'DIN', 'DINB'
            bl_node, blb_node = 'BL', 'BLB'
            en_node, enb_node = 'EN', 'ENB'
        
        # --- Inverters for Data and Enable ---
        # DIN -> DINB
        self.M(1, 'DINB', d_node, 'VDD', 'VDD', model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M(2, 'DINB', d_node, 'VSS', 'VSS', model=self.nmos_model, w=self.nmos_width, l=self.length)
        # EN -> ENB
        self.M(3, 'ENB', en_node, 'VDD', 'VDD', model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M(4, 'ENB', en_node, 'VSS', 'VSS', model=self.nmos_model, w=self.nmos_width, l=self.length)

        # --- Tristate Driver for BL ---
        self.M(5, 'int1', db_node, 'VDD', 'VDD', model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M(6, bl_node, enb_node, 'int1', 'VDD', model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M(7, bl_node, en_node, 'int2', 'VSS', model=self.nmos_model, w=self.nmos_width, l=self.length)
        self.M(8, 'int2', db_node, 'VSS', 'VSS', model=self.nmos_model, w=self.nmos_width, l=self.length)

        # --- Tristate Driver for BLB ---
        self.M(9, 'int3', d_node, 'VDD', 'VDD', model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M(10, blb_node, enb_node, 'int3', 'VDD', model=self.pmos_model, w=self.pmos_width, l=self.length)
        self.M(11, blb_node, en_node, 'int4', 'VSS', model=self.nmos_model, w=self.nmos_width, l=self.length)
        self.M(12, 'int4', d_node, 'VSS', 'VSS', model=self.nmos_model, w=self.nmos_width, l=self.length)
