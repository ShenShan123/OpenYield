from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF, u_V, u_ns
from subcircuits.base_subcircuit import BaseSubcircuit


class ColumnMux(BaseSubcircuit):
    """Column multiplexer for SRAM array using NMOS pass gates."""
    
    def __init__(self, mux_ratio, nmos_model_name, pmos_model_name, 
                 base_nmos_width, base_pmos_width, length,
                 w_rc=False, pi_res=10 @ u_Ohm, pi_cap=0.001 @ u_pF
                 ):
        """
        Args:
            mux_ratio (int): Number of columns multiplexed into one output.
            nmos_model_name (str): NMOS model name.
            width (float): Width of the NMOS pass gates.
            length (float): Length of the NMOS pass gates.
        """
        self.NAME = f'ColumnMux_{mux_ratio}'
        # Power Nodes: VDD, VSS,
        # Output: SA_BL (sense amplifier bitline), SA_BLB (sense amplifier bitline bar),
        # Input: SEL (select signals), BL/BLB (bitlines).
        nodes = ['VDD', 'VSS', 'SA_BL', 'SA_BLB']
        nodes += [f'SEL{i}' for i in range(mux_ratio)]  # input: Select signals
        nodes += [f'BL{i}' for i in range(mux_ratio)]   # input: Bitlines
        nodes += [f'BLB{i}' for i in range(mux_ratio)]  # input: Bitline bars
        self.NODES = tuple(nodes)

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
        
        self.mux_ratio = mux_ratio

        # Add NMOS pass transistors for each column in the mux
        for i in range(mux_ratio):
            # Bitline pass gate
            self.M(f'Muxn_BL_{i}', f'BL{i}', f'SEL{i}', 'SA_BL', 'VSS', 
                   model=nmos_model_name, w=base_nmos_width, l=length)
            self.M(f'Muxp_BL_{i}', f'BL{i}', f'SEL{i}', 'SA_BL', 'VDD', 
                   model=pmos_model_name, w=base_pmos_width, l=length)
            # Bitline bar pass gate
            self.M(f'Muxn_BLB_{i}', f'BLB{i}', f'SEL{i}', 'SA_BLB', 'VSS',
                   model=nmos_model_name, w=base_nmos_width, l=length)
            self.M(f'Muxp_BLB_{i}', f'BLB{i}', f'SEL{i}', 'SA_BLB', 'VDD',
                   model=pmos_model_name, w=base_pmos_width, l=length)

        # Performance considerations:
        # 1. NMOS width should be large enough to reduce resistance (~2-3x access transistor width).
        # 2. Ensure only one SEL signal is active at a time to avoid contention.

class SenseAmp(BaseSubcircuit):
    """Differential sense amplifier with enable signal."""
    
    NAME = 'SENSEAMP'
    # Input: VDD, VSS, SA_BL, SA_BLB, EN
    # Output: OUT, OUTB, 
    NODES = ('VDD', 'VSS', 'EN', 'IN', 'INB', 'Q', 'QB')

    def __init__(self, nmos_model_name, pmos_model_name, 
                 base_nmos_width=0.27e-6, base_pmos_width=0.54e-6, length=50e-9,
                 w_rc=False, pi_res=10 @ u_Ohm, pi_cap=0.001 @ u_pF
                 ):
        """
        Args:
            nmos_model_name (str): NMOS model name.
            pmos_model_name (str): PMOS model name.
            base_nmos_width (float): Width of NMOS transistors.
            base_pmos_width (float): Width of PMOS transistors.
            length (float): Length of transistors.
        """
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
        self.pmos_width = self.base_pmos_width / 3 * 4
        self.nmos_width = self.base_nmos_width
        self.add_sense_transistors()

    def add_sense_transistors(self):
        if self.w_rc:
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
        self.M(1, 'Q', qb_node, 'net1', 'VSS', model=self.nmos_pdk_model,
               w=self.base_nmos_width, l=self.length)  # NMOS
        self.M(2, 'Q', qb_node, 'VDD', 'VDD', model=self.pmos_pdk_model,
               w=self.base_pmos_width, l=self.length)  # PMOS
        self.M(3, 'QB', q_node, 'net1', 'VSS', model=self.nmos_pdk_model,
               w=self.base_nmos_width, l=self.length)  # NMOS
        self.M(4, 'QB', q_node, 'VDD', 'VDD', model=self.pmos_pdk_model,
               w=self.base_pmos_width, l=self.length)  # PMOS

        # Sense enable transistors
        self.M(5, 'Q', en_node, in_node, 'VDD', model=self.pmos_pdk_model,
               w=self.pmos_width, l=self.length)  # NMOS (wider for lower resistance)
        self.M(6, 'QB', en_node, inb_node, 'VDD', model=self.pmos_pdk_model,
               w=self.pmos_width, l=self.length)  # NMOS (wider for lower resistance)
        self.M(7, net1_node, en_node, 'VSS', 'VSS', model=self.nmos_pdk_model,
               w=self.nmos_width, l=self.length)  # NMOS

        # Performance considerations:
        # 1. Cross-coupled inverters should be balanced for symmetric operation.
        # 2. Sense enable transistors should be wider to minimize resistance.
        # 3. Ensure SE signal is timed correctly to avoid premature sensing.