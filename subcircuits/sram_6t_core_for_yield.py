from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF
from subcircuits.base_subcircuit import BaseSubcircuit
# from utils import model_dict2str
from typing import Dict, Any, Union

class Sram6TCell(BaseSubcircuit):
    ###6T SRAM Cell SubCircuitFactory with debug capabilities###
    NAME = 'SRAM_6T_CELL'
    # The first and second nodes are always power and ground nodes,VDD and VSS
    NODES = ('VDD', 'VSS', 'BL', 'BLB', 'WL')
    
    def __init__(self, 
                 nmos_model_name: str, pmos_model_name: str,
                 pd_width: float, pu_width: float, 
                 pg_width: float, length: float,
                 w_rc=False,
                 pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
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
            bl_node  = self.add_rc_networks_to_node(self.NODES[2], 1)
            blb_node = self.add_rc_networks_to_node(self.NODES[3], 1)
            wl_node  = self.add_rc_networks_to_node(self.NODES[4], 1)
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
        self.M('PGL', bl_node,  wl_node, data_q,  self.NODES[1], model=self.nmos_pdk_model, w=self.pg_width, l=self.length)
        self.M('PGR', blb_node, wl_node, data_qb, self.NODES[1], model=self.nmos_pdk_model, w=self.pg_width, l=self.length)
        print(f"[DEBUG] M1-M2: Access transistors NMOS={self.nmos_pdk_model} W={self.pg_width} L={self.length})")

        # Cross-coupled inverters
        self.M('PDL', data_q, 'QB', self.NODES[1], self.NODES[1], model=self.nmos_pdk_model, w=self.pd_width, l=self.length)
        self.M('PUL', data_q, 'QB', self.NODES[0], self.NODES[0], model=self.pmos_pdk_model, w=self.pu_width, l=self.length)
        self.M('PDR', data_qb, 'Q', self.NODES[1], self.NODES[1], model=self.nmos_pdk_model, w=self.pd_width, l=self.length)
        self.M('PUR', data_qb, 'Q', self.NODES[0], self.NODES[0], model=self.pmos_pdk_model, w=self.pu_width, l=self.length)

        print(f"[DEBUG] M3-M6: Cross-coupled inverters (NMOS={self.nmos_pdk_model} W={self.pd_width} L={self.length}"+
              f"        PMOS={self.pmos_pdk_model} W={self.pu_width} L={self.length})")  

class Sram6TCellForYield(Sram6TCell):
    ###6T SRAM Cell SubCircuitFactory with debug capabilities###
    NAME = 'SRAM_6T_CELL'
    
    def __init__(self,
                 nmos_model_name: str, pmos_model_name: str,
                 model_dict: Dict[str, Dict[str, Any]],
                 pd_width: float, pu_width: float, 
                 pg_width: float, length: float,
                 w_rc=False,
                 pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 disconnect=False,
                 suffix='',
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
        # Model parameters are in this dict.
        self.model_dict = model_dict 

        # Call parent class's __init__()
        super().__init__(
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
        pgl_udf_model = self.nmos_pdk_model + '_PGL' + self.suffix
        self.M('PGL', bl_node,  wl_node, data_q,  self.NODES[1], 
               model=pgl_udf_model, 
               w=self.pg_width, l=self.length)
        self.add_usrdefine_mos_model(self.nmos_pdk_model, pgl_udf_model)

        pgr_udf_model = self.nmos_pdk_model + '_PGR' + self.suffix
        self.M('PGR', blb_node, wl_node, data_qb, self.NODES[1], 
               model=pgr_udf_model, 
               w=self.pg_width, l=self.length)
        self.add_usrdefine_mos_model(self.nmos_pdk_model, pgr_udf_model)

        print(f"[DEBUG] M1-M2: Access transistors NMOS={pgl_udf_model} W={self.pg_width} L={self.length})")

        # Cross-coupled inverters
        # Left-side inverter
        pdl_udf_model = self.nmos_pdk_model + '_PDL' + self.suffix
        self.M('PDL', data_q, 'QB', self.NODES[1], self.NODES[1], 
               model=pdl_udf_model, 
               w=self.pd_width, l=self.length)
        self.add_usrdefine_mos_model(self.nmos_pdk_model, pdl_udf_model)
        
        pul_udf_model = self.pmos_pdk_model + '_PUL' + self.suffix
        self.M('PUL', data_q, 'QB', self.NODES[0], self.NODES[0], 
               model=pul_udf_model, 
               w=self.pu_width, l=self.length)
        self.add_usrdefine_mos_model(self.pmos_pdk_model, pul_udf_model)
        
        # Right-side inverter
        pdr_udf_model = self.nmos_pdk_model + '_PDR' + self.suffix
        self.M('PDR', data_qb, 'Q', self.NODES[1], self.NODES[1], 
               model=pdr_udf_model, 
               w=self.pd_width, l=self.length)
        self.add_usrdefine_mos_model(self.nmos_pdk_model, pdr_udf_model)

        pur_udf_model = self.pmos_pdk_model + '_PUR' + self.suffix
        self.M('PUR', data_qb, 'Q', self.NODES[0], self.NODES[0], 
               model=pur_udf_model, 
               w=self.pu_width, l=self.length)
        self.add_usrdefine_mos_model(self.pmos_pdk_model, pur_udf_model)

        print(f"[DEBUG] M3-M6: Cross-coupled inverters (NMOS={pdr_udf_model} W={self.pd_width} L={self.length}"+
              f"        PMOS={pur_udf_model} W={self.pu_width} L={self.length})")  
        
    def add_usrdefine_mos_model(self, pdk_model_name, udf_model_name):
        model_data = self.model_dict[pdk_model_name]
        mos_type = model_data['type']
        self.raw_spice += f'.model {udf_model_name} {mos_type} '
        params = model_data['parameters']
        for param_name, param_value in params.items():
            # Format parameter value
            if isinstance(param_value, float):
                # Use scientific notation for very small/large numbers
                if abs(param_value) < 1e-3 or abs(param_value) > 1e6:
                    param_str = f"{param_value:.3e}"
                else:
                    param_str = str(param_value)
            else:
                param_str = str(param_value)
            # substitute the default values with user-defined parameters
            if param_name in ['vth0', 'u0', 'voff']:
                param_str = f"'{param_name}_{udf_model_name}'"
            # Write parameter 
            self.raw_spice += f"{param_name}={param_str} "
        self.raw_spice += "\n"
        # print(f"[DEBUG] {self.raw_spice}")
        # assert 0


class Sram6TCore(SubCircuitFactory):
    ###
    # SRAM Array SubCircuitFactory class.
    # Configurable number of rows and columns.
    ###

    def __init__(self, num_rows: int, num_cols: int, 
                 nmos_model_name: str, pmos_model_name: str, 
                 pd_width=0.205e-6, pu_width=0.09e-6, 
                 pg_width=0.135e-6, length=50e-9,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
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
        
        # Build the array
        self.build_array(num_rows, num_cols)
        # set instance prefix and the name of 6t cell
        self.inst_prefix = "XSRAM_6T_CELL"  
        

    def build_array(self, num_rows, num_cols):
        # Generate SRAM cells
        subckt_6t_cell = Sram6TCell(
            self.nmos_pdk_model, self.pmos_pdk_model, 
            self.pd_width, self.pu_width, 
            self.pg_width, self.length, 
            w_rc=self.w_rc, 
            pi_res=self.pi_res, pi_cap=self.pi_cap,
        )

        # define the cell subcircuit
        self.subcircuit(subckt_6t_cell)

        # Instantiate SRAM cells
        for row in range(num_rows):
            for col in range(num_cols):
                self.X(
                    subckt_6t_cell.name+f"_{row}_{col}",
                    subckt_6t_cell.name,
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

    def __init__(self, num_rows: int, num_cols: int, 
                 nmos_model_name: str, pmos_model_name: str, 
                 model_dict: Dict[str, Dict[str, Any]],
                 pd_width=0.205e-6, pu_width=0.09e-6, 
                 pg_width=0.135e-6, length=50e-9,
                 w_rc=False, pi_res=100 @ u_Ohm, pi_cap=0.001 @ u_pF,
                 ):
        # add model_dict for customizing transistor `.model` in each cell
        # set `model_dict` before super().__init__()
        self.model_dict = model_dict

        super().__init__(num_rows, num_cols, 
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
                    self.nmos_pdk_model, self.pmos_pdk_model, 
                    self.model_dict,
                    self.pd_width, self.pu_width, 
                    self.pg_width, self.length, 
                    w_rc=self.w_rc, 
                    pi_res=self.pi_res, pi_cap=self.pi_cap,
                    suffix=f'_{row}_{col}',
                )

                # add the cell subcircuit to this circuit
                self.subcircuit(subckt_6t_cell)

                # Instantiate SRAM cells
                self.X(
                    subckt_6t_cell.name,
                    subckt_6t_cell.name,
                    self.NODES[0],  # Power net
                    self.NODES[1],  # Ground net
                    f'BL{col}',     # Connect to column bitline
                    f'BLB{col}',    # Connect to column bitline bar
                    f'WL{row}',     # Connect to row wordline
                )

# if __name__ == '__main__':
#     pdk_path = 'model_lib/models.spice'
#     nmos_model_name = 'NMOS_VTG'
#     pmos_model_name = 'PMOS_VTG'
#     pd_width=0.205e-6
#     pu_width=0.09e-6
#     pg_width=0.135e-6
#     length=50e-9

# #     # bc = SRAM_6T_Cell_for_Yield(
# #     #     nmos_model_name, pmos_model_name,
# #     #     pd_width=0.1e-6, pu_width=0.2e-6, pg_width=1.5e-6, length=45e-9,
# #     #     suffix='_0_0', disconnect=True,
# #     # )
# #     # print(bc)

#     array = Sram6TCoreForYield(
#         2, 2, 
#         nmos_model_name, pmos_model_name,
#         pd_width=pd_width, pu_width=pu_width, pg_width=pg_width, length=length,
#         w_rc=True, pi_res=100 @ u_Ohm, pi_cap=0.010 @ u_pF
#     )
#     print(array)