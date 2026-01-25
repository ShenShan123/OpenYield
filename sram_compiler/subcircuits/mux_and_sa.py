from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF, u_V, u_ns
from .base_subcircuit import BaseSubcircuit

class ColumnMux(BaseSubcircuit):
    """
    Standard Column Multiplexer (Topology Only).
    supports two modes:
    1. Generates anti-signal internally (use_external_selb=False)
    2.  Receives anti-signal from the outside (use_external_selb=True)
    """
    
    def __init__(self, 
                 nmos_model, pmos_model,
                 nmos_width, pmos_width, length,
                 num_in,
                 use_external_selb=False, # <--- The key flag for the control mode
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF):
        
        self.NAME = f'COLUMNMUX{num_in}'
        # --- 1. Dynamic definition of nodes ---
        nodes = ['VDD', 'VSS', 'SA_IN', 'SA_INB']
        # Select signals
        nodes += [f'SEL{i}' for i in range(num_in)]
        # If an external SELB is used, add SELB nodes
        if use_external_selb:
            nodes += [f'SELB{i}' for i in range(num_in)]
        # Bitline signals
        nodes += [f'BL{i}' for i in range(num_in)]
        nodes += [f'BLB{i}' for i in range(num_in)]
        self.NODES = tuple(nodes)

        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap
        )
        
        # 保存参数
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.nmos_width = nmos_width
        self.pmos_width = pmos_width
        self.length = length
        self.num_in = num_in
        self.use_external_selb = use_external_selb
        self.w_rc = w_rc
        
        self.add_switch_transistors()
    
    def add_switch_transistors(self):
        #merge x columns, generate x identical structures.
        for i in range(self.num_in):
            if self.w_rc:
                sel_node = self.add_rc_networks_to_node(f'SEL{i}', num_segs=1)
                selb_node = self.add_rc_networks_to_node(f'SELB{i}', num_segs=1)
            else:
                sel_node = f'SEL{i}'
                selb_node = f'SELB{i}'

            # Only when an external SELB is not used, should the internal inverter be added.---
            if not self.use_external_selb:
                # Inverters for select signal 
                self.M(f'Invp_{i}', selb_node, sel_node, 'VDD', 'VDD',
                       model=self.pmos_model, w=self.pmos_width, l=self.length)
                self.M(f'Invn_{i}', selb_node, sel_node, 'VSS', 'VSS',
                       model=self.nmos_model, w=self.nmos_width, l=self.length)

            # Transmission Gates ---
            # Bitline pass gate
            # NMOS (controlled by SEL)
            self.M(f'Muxn_BL_{i}', f'BL{i}', sel_node, 'SA_IN', 'VSS', 
                   model=self.nmos_model, w=self.nmos_width, l=self.length)
            # PMOS (controlled by SELB)
            self.M(f'Muxp_BL_{i}', f'BL{i}', selb_node, 'SA_IN', 'VDD', 
                   model=self.pmos_model, w=self.pmos_width, l=self.length)
            # Bitline bar pass gate
            # NMOS (controlled by SEL)
            self.M(f'Muxn_BLB_{i}', f'BLB{i}', sel_node, 'SA_INB', 'VSS',
                   model=self.nmos_model, w=self.nmos_width, l=self.length)
            # PMOS (controlled by SELB)
            self.M(f'Muxp_BLB_{i}', f'BLB{i}', selb_node, 'SA_INB', 'VDD',
                   model=self.pmos_model, w=self.pmos_width, l=self.length)

class SenseAmp(BaseSubcircuit):
    """Differential sense amplifier with enable signal.带使能的差分感测放大器"""
    
    NAME = 'SENSEAMP'
    # Input: VDD, VSS, SA_BL, SA_BLB, EN
    # Output: OUT, OUTB, 
    NODES = ('VDD', 'VSS', 'EN', 'IN', 'INB', 'Q', 'QB')

    def __init__(self, nmos_model, pmos_model, 
                 nmos_width, pmos_width, length,#design parameters
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF#parasititic parameters
                 ):
        """
        Args:
            nmos_model_name (str): NMOS model name.
            pmos_model_name (str): PMOS model name.
            nmos_width (float/str): Width of NMOS transistors.
            pmos_width (float/str): Width of PMOS transistors.
            length (float/str): Length of transistors.
        """
        super().__init__(
            nmos_model, pmos_model,
            nmos_width, pmos_width, length,
            w_rc, pi_res, pi_cap,
        )
        
        #Save the parameters for use by add_transistors
        self.nmos_model = nmos_model
        self.pmos_model = pmos_model
        self.pmos_width = pmos_width
        #self.pmos_width_pass = self.pmos_width / 3 * 4  #Specifically designed for the two transmission tubes on the left and right sides
        self.nmos_width = nmos_width
        self.length = length
        self.w_rc = w_rc
        if isinstance(pmos_width, str):
            # In parameter sweep mode, store the string and handle in spice netlist generation
            self.pmos_width_pass = self.pmos_width
        else:
            # In normal mode, calculate the derived width
            self.pmos_width_pass = pmos_width / 3 * 4 

        self.add_sense_transistors()
    
    def add_sense_transistors(self):
        if self.w_rc:                                    #考虑是否加rc参数，修改节点名
            en_node = self.add_rc_networks_to_node('EN', num_segs=2)
            in_node = self.add_rc_networks_to_node('IN', num_segs=2)
            inb_node = self.add_rc_networks_to_node('INB', num_segs=2)
            q_node = self.add_rc_networks_to_node('Q', num_segs=2)
            qb_node = self.add_rc_networks_to_node('QB', num_segs=2)
            net1_node = self.add_rc_networks_to_node('net1', num_segs=1)
        else:
            en_node = 'EN'
            in_node = 'IN'
            inb_node = 'INB'
            q_node = 'Q'
            qb_node = 'QB'
            net1_node = 'net1'
        
        # Cross-coupled inverters for positive feedback
        self.M(1, 'Q', qb_node, 'net1', 'VSS', model=self.nmos_model,
            w=self.nmos_width, l=self.length)  # NMOS
        self.M(2, 'Q', qb_node, 'VDD', 'VDD', model=self.pmos_model,
            w=self.pmos_width, l=self.length)  # PMOS
        self.M(3, 'QB', q_node, 'net1', 'VSS', model=self.nmos_model,
            w=self.nmos_width, l=self.length)  # NMOS
        self.M(4, 'QB', q_node, 'VDD', 'VDD', model=self.pmos_model,
            w=self.pmos_width, l=self.length)  # PMOS

        # Sense enable transistors 注意左右两个传输管的参数不用base
        self.M(5, 'Q', en_node, in_node, 'VDD', model=self.pmos_model,
            w=self.pmos_width_pass, l=self.length)  # PMOS (wider for lower resistance)
        self.M(6, 'QB', en_node, inb_node, 'VDD', model=self.pmos_model,
            w=self.pmos_width_pass, l=self.length)  # PMOS (wider for lower resistance)
        self.M(7, net1_node, en_node, 'VSS', 'VSS', model=self.nmos_model,
            w=self.nmos_width, l=self.length)  # NMOS
        
