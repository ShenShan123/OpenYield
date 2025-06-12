from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF

class SRAM_6T_Cell(SubCircuitFactory):
    """ 6T SRAM Cell SubCircuitFactory with debug capabilities """
    NAME = 'SRAM_6T_CELL'
    # The first and second nodes are always power and ground nodes,VDD and VSS
    NODES = ('VDD', 'VSS', 'BL', 'BLB', 'WL')
    
    def __init__(self, nmos_model_name, pmos_model_name,
                 pd_width, pu_width, 
                 pg_width, length,
                 w_rc=False,
                 pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF,
                 disconncet=False,
                 ):
        if disconncet:
            self.NAME += '_DISCONNECT'
            
        super().__init__()
        print(f"\n[DEBUG] Creating SRAM_6T_Cell with models: "
              f"NMOS={nmos_model_name}, PMOS={pmos_model_name}")
        
        self.nmos_model_name = nmos_model_name
        self.pmos_model_name = pmos_model_name

        # Transistor Sizes (FreePDK45 uses nanometers)
        self.pg_width = pg_width
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.length = length
        self.disconncet = disconncet

        if not w_rc:
            bl_node  = self.NODES[2]
            blb_node = self.NODES[3]
            wl_node  = self.NODES[4]

        else:
            # Add L-shape RC networks for BL, BLB, and WL
            bl_node  = self.add_rc_networks_to_node(self.NODES[2], pi_res, pi_cap, 2)
            blb_node = self.add_rc_networks_to_node(self.NODES[3], pi_res, pi_cap, 2)
            wl_node  = self.add_rc_networks_to_node(self.NODES[4], pi_res, pi_cap, 2)

        self.add_6T_cell(bl_node, blb_node, wl_node)
        
    def add_rc_networks_to_node(self, in_node, pi_r, pi_c, num_segs=1):
        """ Add RC networks to the SRAM cell """
        start_node = in_node
        end_node = start_node

        for i in range(num_segs):
            if num_segs-1 == i:
                end_node = in_node + '_end'
            else:
                end_node = start_node + f'_seg{i}'

            self.R(f'R_{in_node}_{i}',  start_node, end_node, pi_r)
            self.C(f'Cg_{in_node}_{i}', end_node, self.gnd, pi_c)
            start_node = end_node
        
        return end_node
    
    def add_6T_cell(self, bl_node, blb_node, wl_node):
        """ Add 6T cell to the SRAM cell """
        if self.disconncet:
            data_q = 'QD'
            data_qb = 'QBD'
        else:
            data_q = 'Q'
            data_qb = 'QB'
        # Access transistors
        self.M(1, bl_node,  wl_node, data_q,  self.NODES[1], model=self.nmos_model_name, w=self.pg_width, l=self.length)
        self.M(2, blb_node, wl_node, data_qb, self.NODES[1], model=self.nmos_model_name, w=self.pg_width, l=self.length)
        print(f"[DEBUG] M1-M2: Access transistorsNMOS ({self.nmos_model_name}) W={self.pg_width} L={self.length})")

        # Cross-coupled inverters
        self.M(3, data_q, 'QB', self.NODES[1], self.NODES[1], model=self.nmos_model_name, w=self.pd_width, l=self.length)
        self.M(4, data_q, 'QB', self.NODES[0], self.NODES[0], model=self.pmos_model_name, w=self.pu_width, l=self.length)
        self.M(5, data_qb, 'Q', self.NODES[1], self.NODES[1], model=self.nmos_model_name, w=self.pd_width, l=self.length)
        self.M(6, data_qb, 'Q', self.NODES[0], self.NODES[0], model=self.pmos_model_name, w=self.pu_width, l=self.length)

        # NOTE: Add small load cap to data nodes to balance the write time.
        self.C(f'g_Q', 'Q', self.gnd, 0.001 @ u_pF)
        self.C(f'g_QB', 'QB', self.gnd, 0.001 @ u_pF)

        print(f"[DEBUG] M3-M6: Cross-coupled inverters (NMOS={self.nmos_model_name} W={self.pd_width} L={self.length}"+
              f"      PMOS={self.pmos_model_name} W={self.pu_width} L={self.length})")  
        
    def estimate_bitcell_area(
        self,
        # FreePDK45 Design Rules (default values in meters)
        CPP: float = 0.18e-6,          # Contacted Poly Pitch
        MMP: float = 0.16e-6,          # Minimum Metal Pitch
        diffusion_spacing: float = 0.08e-6,    # Diffusion-to-diffusion
        gate_contact_spacing: float = 0.05e-6, # Gate-to-contact
        metal_overhang: float = 0.03e-6,       # Metal over active
    ) -> float:
        """
        Estimates 6T SRAM bitcell area for FreePDK45nm technology.
        Methodology:
        1. Horizontal dimension: Based on contacted poly pitch (CPP) and transistor widths
        2. Vertical dimension: Based on metal pitch (MMP) and peripheral routing
        3. Includes spacing/overhang from design rules
        """
        
        # --- Horizontal Dimension (Width) ---
        # Effective width per cell column (sum of parallel transistors)
        access_width = self.pg_width + 2 * gate_contact_spacing
        pd_pu_width = max(self.pd_width, self.pu_width) + 2 * diffusion_spacing
        
        # Total cell width = (3 columns * CPP) + scaling from transistor widths
        width_columns = 3 * CPP  # 2 access + 1 inverter pair
        width_scaling = (access_width + pd_pu_width) / (0.135e-6 + 0.205e-6)  # Normalize to PDK45 ref
        cell_width = width_columns * width_scaling + 2 * metal_overhang

        # --- Vertical Dimension (Height) ---
        # Height dominated by metal routing (2 tracks) + n-well spacing
        cell_height = 2 * MMP + 2 * diffusion_spacing + 4 * metal_overhang
        
        # --- Area Calculation ---
        area = cell_width * cell_height
        print(f"Estimated 6T SRAM Cell Area: {area*1e12:.2f} µm²")
        return area

class SRAM_6T_Array(SubCircuitFactory):
    """
    SRAM Array SubCircuitFactory class.
    Configurable number of rows and columns.
    """

    def __init__(self, num_rows, num_cols, 
                 nmos_model_name, pmos_model_name, 
                 pd_width, pu_width, 
                 pg_width, length,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF,):
                #  disconnect=False, target_row=None, target_col=None):
        
        self.NAME = f"SRAM_6T_ARRAY_{num_rows}x{num_cols}"
        # Define nodes
        self.NODES = (
            'VDD',  # Power supply
            'VSS',  # Ground
            *[f'BL{i}' for i in range(num_cols)],  # Bitlines
            *[f'BLB{i}' for i in range(num_cols)],  # Bitline bars
            *[f'WL{i}' for i in range(num_rows)],  # Wordlines
        )
        super().__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.nmos_model_name = nmos_model_name
        self.pmos_model_name = pmos_model_name

        subckt_6t_cell = SRAM_6T_Cell(
            nmos_model_name, pmos_model_name, 
            pd_width, pu_width, pg_width, length, 
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap
        )

        # define the cell subcircuit
        self.subcircuit(subckt_6t_cell)
        self.inst_prefix = "X" + subckt_6t_cell.name
        self.cell_name = subckt_6t_cell.name

        # Build the array
        self.build_array(num_rows, num_cols)

    def build_array(self, num_rows, num_cols):
        # Generate SRAM cells
        for row in range(num_rows):
            for col in range(num_cols):
                self.X(
                    self.cell_name+f"_{row}_{col}",
                    self.cell_name,
                    self.NODES[0],          # Power net
                    self.NODES[1],          # Ground net
                    f'BL{col}',     # Connect to column bitline
                    f'BLB{col}',    # Connect to column bitline bar
                    f'WL{row}',     # Connect to row wordline
                )