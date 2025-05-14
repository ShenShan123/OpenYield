from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF
from subcircuits.base import BaseSubcircuit

class Sram6TCell(BaseSubcircuit):
    ###6T SRAM Cell SubCircuitFactory with debug capabilities###
    NAME = 'SRAM_6T_CELL'
    # The first and second nodes are always power and ground nodes,VDD and VSS
    NODES = ('VDD', 'VSS', 'BL', 'BLB', 'WL')
    
    def __init__(self, vdd,
                 nmos_model_name, pmos_model_name,
                 pd_width, pu_width, 
                 pg_width, length,
                 w_rc=False,
                 pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF,
                 disconnect=False, 
                 ):
        # Modify the name of this subcircuit before call parent class.__init__()
        if disconnect:
            self.NAME += '_DISCONNECT'
            
        super().__init__(
            nmos_model_name, pmos_model_name, 
            pd_width, pu_width, length, 
            w_rc, pi_res, pi_cap
        )
        print(f"\n[DEBUG] Creating {self.name} with models: "
              f"NMOS={nmos_model_name}, PMOS={pmos_model_name}")
        self.vdd = vdd

        # Transistor Sizes (FreePDK45 uses nanometers)
        self.pg_width = pg_width
        self.pd_width = pd_width
        self.pu_width = pu_width

        self.disconnect = disconnect

        if not self.w_rc:
            bl_node  = self.NODES[2]
            blb_node = self.NODES[3]
            wl_node  = self.NODES[4]
            q_node = 'Q'
            qb_node = 'QB'
        else:
            # Add L-shape RC networks for BL, BLB, and WL
            bl_node  = self.add_rc_networks_to_node(self.NODES[2], 2)
            blb_node = self.add_rc_networks_to_node(self.NODES[3], 2)
            wl_node  = self.add_rc_networks_to_node(self.NODES[4], 2)
            # Add L-shape RC networks for Q and QB
            q_node  = self.add_rc_networks_to_node('Q', 1)
            qb_node = self.add_rc_networks_to_node('QB', 1)

        self.add_6T_cell(bl_node, blb_node, wl_node, q_node, qb_node)
        
    def add_6T_cell(self, bl_node, blb_node, wl_node, q_node, qb_node):
        ###Add 6T cell to the SRAM cell, initializaed with `0` at Q###

        if self.disconnect:
            data_q = 'QD'
            data_qb = 'QBD'
        else:
            data_q = q_node
            data_qb = qb_node
        # Access transistors
        self.M(1, bl_node,  wl_node, data_q,  self.NODES[1], model=self.nmos_pdk_model, w=self.pg_width, l=self.length)
        self.M(2, blb_node, wl_node, data_qb, self.NODES[1], model=self.nmos_pdk_model, w=self.pg_width, l=self.length)
        print(f"[DEBUG] M1-M2: Access transistors NMOS={self.nmos_pdk_model} W={self.pg_width} L={self.length})")

        # Cross-coupled inverters
        self.M(3, data_q, 'QB', self.NODES[1], self.NODES[1], model=self.nmos_pdk_model, w=self.pd_width, l=self.length)
        self.M(4, data_q, 'QB', self.NODES[0], self.NODES[0], model=self.pmos_pdk_model, w=self.pu_width, l=self.length)
        self.M(5, data_qb, 'Q', self.NODES[1], self.NODES[1], model=self.nmos_pdk_model, w=self.pd_width, l=self.length)
        self.M(6, data_qb, 'Q', self.NODES[0], self.NODES[0], model=self.pmos_pdk_model, w=self.pu_width, l=self.length)

        # NOTE: Add small load cap to data nodes to balance the write time.
        # self.C(f'g_Q', 'Q', self.gnd, 0.001 @ u_pF)
        # self.C(f'g_QB', 'QB', self.gnd, 0.001 @ u_pF)

        print(f"[DEBUG] M3-M6: Cross-coupled inverters (NMOS={self.nmos_pdk_model} W={self.pd_width} L={self.length}"+
              f"      PMOS={self.pmos_pdk_model} W={self.pu_width} L={self.length})")  

class Sram6TCellForYield(Sram6TCell):
    ###6T SRAM Cell SubCircuitFactory with debug capabilities###
    NAME = 'SRAM_6T_CELL'
    
    def __init__(self, vdd,
                 nmos_model_name, pmos_model_name,
                 pd_width, pu_width, 
                 pg_width, length,
                 w_rc=False,
                 pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF,
                 disconnect=False,
                 suffix='',
                #  q_init=True, q_init_val=0,
                 custom_mc=False,
                 ):
        # Modify the name of this subcircuit before call parent class.__init__()
        if disconnect:
            assert suffix == '_0_0', 'using disconnected cell in an array'
        
        self.NAME += suffix
        # Suffix of user defined model name
        self.suffix = suffix
        # Whether use local process parameters
        self.custom_mc = custom_mc

        # Call parent class's __init__()
        super().__init__(
            vdd,
            nmos_model_name, pmos_model_name, 
            pd_width, pu_width, pg_width, length, 
            w_rc, pi_res, pi_cap, disconnect, 
        )
        
    
    def add_6T_cell(self, bl_node, blb_node, wl_node, q_node, qb_node):
        ###Add 6T cell to the SRAM cell###
        if self.disconnect:
            data_q = 'QD'
            data_qb = 'QBD'
        else:
            data_q = q_node
            data_qb = qb_node

        # Access transistors
        pgl_udf_model = self.nmos_pdk_model+'_PGL'+self.suffix
        self.M(1, bl_node,  wl_node, data_q,  self.NODES[1], 
               model=pgl_udf_model, 
               w=self.pg_width, l=self.length)
        self.add_usrdefine_mos_model(pgl_udf_model)

        pgr_udf_model = self.nmos_pdk_model+'_PGR'+self.suffix
        self.M(2, blb_node, wl_node, data_qb, self.NODES[1], 
               model=pgr_udf_model, 
               w=self.pg_width, l=self.length)
        self.add_usrdefine_mos_model(pgr_udf_model)

        print(f"[DEBUG] M1-M2: Access transistors NMOS={pgl_udf_model} W={self.pg_width} L={self.length})")

        # Cross-coupled inverters
        # Left-side inverter
        pdl_udf_model = self.nmos_pdk_model+'_PDL'+self.suffix
        self.M(3, data_q, 'QB', self.NODES[1], self.NODES[1], 
               model=pdl_udf_model, 
               w=self.pd_width, l=self.length)
        self.add_usrdefine_mos_model(pdl_udf_model)
        
        pul_udf_model = self.pmos_pdk_model+'_PUL'+self.suffix
        self.M(4, data_q, 'QB', self.NODES[0], self.NODES[0], 
               model=pul_udf_model, 
               w=self.pu_width, l=self.length)
        self.add_usrdefine_mos_model(pul_udf_model)
        
        # Right-side inverter
        pdr_udf_model = self.nmos_pdk_model+'_PDR'+self.suffix
        self.M(5, data_qb, 'Q', self.NODES[1], self.NODES[1], 
               model=pdr_udf_model, 
               w=self.pd_width, l=self.length)
        self.add_usrdefine_mos_model(pdr_udf_model)

        pur_udf_model = self.pmos_pdk_model+'_PUR'+self.suffix
        self.M(6, data_qb, 'Q', self.NODES[0], self.NODES[0], 
               model=pur_udf_model, 
               w=self.pu_width, l=self.length)
        self.add_usrdefine_mos_model(pur_udf_model)

        print(f"[DEBUG] M3-M6: Cross-coupled inverters (NMOS={pdr_udf_model} W={self.pd_width} L={self.length}"+
              f"      PMOS={pur_udf_model} W={self.pu_width} L={self.length})")  
        
    def add_usrdefine_mos_model(self, udf_model_name):
        # Define parameters in user defined model
        if 'NMOS_VTG' in udf_model_name:
            params = f'.param vth0_{udf_model_name}=0.4106\n.param u0_{udf_model_name}=0.045\n.param voff_{udf_model_name}=-0.13\n'
            basic_mos_type = 'nmos'
        elif 'PMOS_VTG' in udf_model_name:
            params = f'.param vth0_{udf_model_name}=-0.3842\n.param u0_{udf_model_name}=0.02\n.param voff_{udf_model_name}=-0.126\n'
            basic_mos_type = 'pmos'
        else:
            raise ValueError(f"Can't find default process parameters for user defined model {udf_model_name}")

        # Update the parameters in circuit, only for debugging
        # self.raw_spice += params 

        # Add model definition
        self.raw_spice += f'.model {udf_model_name} {basic_mos_type} ' + \
            f'level=54 vth0=vth0_{udf_model_name} u0=u0_{udf_model_name} voff=voff_{udf_model_name}\n'


class Sram6TCore(SubCircuitFactory):
    ###
    # SRAM Array SubCircuitFactory class.
    # Configurable number of rows and columns.
    ###

    def __init__(self, vdd, num_rows, num_cols, 
                 nmos_model_name, pmos_model_name, 
                 pd_width=0.205e-6, pu_width=0.09e-6, 
                 pg_width=0.135e-6, length=50e-9,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF,
                 ):
                #  disconnect=False, target_row=None, target_col=None):
        
        self.NAME = f"SRAM_6T_CORE_{num_rows}x{num_cols}"
        # Define nodes
        self.NODES = (
            'VDD',  # Power supply
            'VSS',  # Ground
            *[f'BL{i}' for i in range(num_cols)],  # Bitlines
            *[f'BLB{i}' for i in range(num_cols)],  # Bitline bars
            *[f'WL{i}' for i in range(num_rows)],  # Wordlines
        )
        super().__init__()
        self.vdd = vdd
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.nmos_pdk_model = nmos_model_name
        self.pmos_pdk_model = pmos_model_name
        # Transistor Sizes (FreePDK45 uses nanometers)
        self.pg_width = pg_width
        self.pd_width = pd_width
        self.pu_width = pu_width
        self.length = length
        # other config
        self.w_rc = w_rc
        self.pi_res = pi_res
        self.pi_cap = pi_cap

        subckt_6t_cell = Sram6TCell(
            vdd,
            nmos_model_name, pmos_model_name, 
            pd_width, pu_width, pg_width, length, 
            w_rc=w_rc, pi_res=pi_res, pi_cap=pi_cap,
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

class Sram6TCoreForYield(Sram6TCore):
    ###
    # SRAM Array SubCircuitFactory class.
    # Configurable number of rows and columns.
    ###

    def __init__(self, vdd, num_rows, num_cols, 
                 nmos_model_name, pmos_model_name, 
                 pd_width=0.205e-6, pu_width=0.09e-6, 
                 pg_width=0.135e-6, length=50e-9,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF,
                 ):
                #  disconnect=False, target_row=None, target_col=None):
        super().__init__(vdd, num_rows, num_cols, 
                         nmos_model_name, pmos_model_name, 
                         pd_width, pu_width, pg_width, length, 
                         w_rc, pi_res, pi_cap, 
                         )

    def build_array(self, num_rows, num_cols):
        # Generate SRAM cells
        for row in range(num_rows):
            for col in range(num_cols):
                # instantiate the bitcell's subcircuit for each bit
                subckt_6t_cell = Sram6TCellForYield(
                    self.vdd,
                    self.nmos_pdk_model, self.pmos_pdk_model, 
                    self.pd_width, self.pu_width, 
                    self.pg_width, self.length, 
                    w_rc=self.w_rc, 
                    pi_res=self.pi_res, pi_cap=self.pi_cap,
                    suffix=f'_{row}_{col}',
                )

                # add the cell subcircuit to this circuit
                self.subcircuit(subckt_6t_cell)
                # self.inst_prefix = "X" + subckt_6t_cell.name
                # cell_name = subckt_6t_cell.name

                self.X(
                    subckt_6t_cell.name,
                    subckt_6t_cell.name,
                    self.NODES[0],  # Power net
                    self.NODES[1],  # Ground net
                    f'BL{col}',     # Connect to column bitline
                    f'BLB{col}',    # Connect to column bitline bar
                    f'WL{row}',     # Connect to row wordline
                )

if __name__ == '__main__':
    pdk_path = 'model_lib/models.spice'
    nmos_model_name = 'NMOS_VTG'
    pmos_model_name = 'PMOS_VTG'
    pd_width=0.205e-6
    pu_width=0.09e-6
    pg_width=0.135e-6
    length=50e-9

#     # bc = SRAM_6T_Cell_for_Yield(
#     #     nmos_model_name, pmos_model_name,
#     #     pd_width=0.1e-6, pu_width=0.2e-6, pg_width=1.5e-6, length=45e-9,
#     #     suffix='_0_0', disconnect=True,
#     # )
#     # print(bc)

    array = Sram6TCoreForYield(
        2, 2, 
        nmos_model_name, pmos_model_name,
        pd_width=pd_width, pu_width=pu_width, pg_width=pg_width, length=length,
        w_rc=True, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF
    )
    print(array)