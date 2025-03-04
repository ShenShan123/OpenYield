from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF, u_V, u_ns

class ColumnMux(SubCircuitFactory):
    """Column multiplexer for SRAM array using NMOS pass gates."""
    
    def __init__(self, mux_ratio, nmos_model_name, width, length):
        """
        Args:
            mux_ratio (int): Number of columns multiplexed into one output.
            nmos_model_name (str): NMOS model name.
            width (float): Width of the NMOS pass gates.
            length (float): Length of the NMOS pass gates.
        """
        self.NAME = f'ColumnMux_{mux_ratio}'
        self.mux_ratio = mux_ratio

        # Nodes: SA_BL (sense amplifier bitline), SA_BLB (sense amplifier bitline bar),
        # VSS (ground), SEL (select signals), BL/BLB (bitlines).
        nodes = ['SA_BL', 'SA_BLB', 'VSS']
        nodes += [f'SEL{i}' for i in range(mux_ratio)]  # Select signals
        nodes += [f'BL{i}' for i in range(mux_ratio)]   # Bitlines
        nodes += [f'BLB{i}' for i in range(mux_ratio)]  # Bitline bars
        self.NODES = tuple(nodes)
        super().__init__()

        # Add NMOS pass transistors for each column in the mux
        for i in range(mux_ratio):
            # Bitline pass gate
            self.M(f'Mux_BL_{i}', f'BL{i}', f'SEL{i}', 'SA_BL', 'VSS', 
                   model=nmos_model_name, w=width, l=length)
            # Bitline bar pass gate
            self.M(f'Mux_BLB_{i}', f'BLB{i}', f'SEL{i}', 'SA_BLB', 'VSS',
                   model=nmos_model_name, w=width, l=length)

        # Performance considerations:
        # 1. NMOS width should be large enough to reduce resistance (~2-3x access transistor width).
        # 2. Ensure only one SEL signal is active at a time to avoid contention.

class SenseAmp(SubCircuitFactory):
    """Differential sense amplifier with enable signal."""
    
    NAME = 'SenseAmp'
    NODES = ('SA_BL', 'SA_BLB', 'OUT', 'OUTB', 'VDD', 'VSS', 'SE')

    def __init__(self, nmos_model_name, pmos_model_name, 
                 nmos_width, pmos_width, length):
        """
        Args:
            nmos_model_name (str): NMOS model name.
            pmos_model_name (str): PMOS model name.
            nmos_width (float): Width of NMOS transistors.
            pmos_width (float): Width of PMOS transistors.
            length (float): Length of transistors.
        """
        super().__init__()

        # Cross-coupled inverters for positive feedback
        self.M(1, 'OUT', 'OUTB', 'VSS', 'VSS', model=nmos_model_name,
               w=nmos_width, l=length)  # NMOS
        self.M(2, 'OUT', 'OUTB', 'VDD', 'VDD', model=pmos_model_name,
               w=pmos_width, l=length)  # PMOS
        self.M(3, 'OUTB', 'OUT', 'VSS', 'VSS', model=nmos_model_name,
               w=nmos_width, l=length)  # NMOS
        self.M(4, 'OUTB', 'OUT', 'VDD', 'VDD', model=pmos_model_name,
               w=pmos_width, l=length)  # PMOS

        # Sense enable transistors
        self.M(5, 'OUT', 'SE', 'SA_BL', 'VSS', model=nmos_model_name,
               w=nmos_width*2, l=length)  # NMOS (wider for lower resistance)
        self.M(6, 'OUTB', 'SE', 'SA_BLB', 'VSS', model=nmos_model_name,
               w=nmos_width*2, l=length)  # NMOS (wider for lower resistance)

        # Performance considerations:
        # 1. Cross-coupled inverters should be balanced for symmetric operation.
        # 2. Sense enable transistors should be wider to minimize resistance.
        # 3. Ensure SE signal is timed correctly to avoid premature sensing.