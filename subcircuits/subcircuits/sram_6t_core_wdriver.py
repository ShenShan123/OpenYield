from PySpice.Spice.Netlist import SubCircuitFactory, Circuit
from PySpice.Unit import u_Ohm, u_pF
from subcircuits.sram_6t_core_for_yield import Sram6TCore, Sram6TCoreForYield
from subcircuits.precharge_and_write_driver import Precharge, WriteDriver
from subcircuits.base import BaseSubcircuit

class Sram6TCoreWdriver(BaseSubcircuit):
    ###
    # SRAM Array SubCircuitFactory class.
    # Configurable number of rows and columns.
    ###

    def __init__(self, vdd, num_rows, num_cols, 
                 nmos_model_name, pmos_model_name, 
                 subckt_design_params: dict={},
                 ):
                #  disconnect=False, target_row=None, target_col=None):
        
        self.NAME = f"SRAM_6T_ARRAY_WDRIVER_{num_rows}x{num_cols}"
        # Define nodes
        self.NODES = (
            'VDD',  # Power supply
            'VSS',  # Ground
            # 'PRE',  # Precharge signal
            'WE',   # Write Enable
            *[f'DIN{i}' for i in range(num_cols)],  # Data on each column
            *[f'WL{i}' for i in range(num_rows)],  # Wordlines
        )
        super().__init__(nmos_model_name, pmos_model_name, )

        self.vdd = vdd
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.nmos_pdk_model = nmos_model_name
        self.pmos_pdk_model = pmos_model_name
        self.heir_delimiter = ':'

        if subckt_design_params.get('core'):
            (pd_width, pu_width, pg_width, length,
            w_rc, pi_res, pi_cap) = subckt_design_params.get('core')
            # define the cell subcircuit
            sbckt_6t_core = Sram6TCore(
                self.vdd, self.num_rows, self.num_cols, 
                self.nmos_pdk_model, self.pmos_pdk_model,
                pd_width, pu_width, pg_width, length,
                w_rc, pi_res, pi_cap,
            )
        else:
            # define the cell subcircuit
            sbckt_6t_core = Sram6TCore(
                self.vdd, self.num_rows, self.num_cols, 
                self.nmos_pdk_model, self.pmos_pdk_model,
            )
        self.subcircuit(sbckt_6t_core)
        self.arr_inst_name = sbckt_6t_core.name
        self.arr_inst_prefix = \
            f"X{self.arr_inst_name}{self.heir_delimiter}{sbckt_6t_core.inst_prefix}"
        
        # if subckt_design_params.get('pre'):
        #     pass
        # # define precharge PMOS
        # precharger = Precharge(self.pmos_pdk_model, num_rows=num_rows)
        # self.subcircuit(precharger)
        # self.pre_inst_name = precharger.name
        # self.pre_inst_prefix = f"X{precharger.name}"

        if subckt_design_params.get('wdrv'):
            (base_nmos_width, base_pmos_width, length,
            w_rc, pi_res, pi_cap) = subckt_design_params.get('wdrv')
            # define write driver
            write_drv = WriteDriver(
                self.nmos_pdk_model, self.pmos_pdk_model,
                base_nmos_width, base_pmos_width, length,
                w_rc, pi_res, pi_cap, num_rows=num_rows,
            )
        else:
            write_drv = WriteDriver(
                self.nmos_pdk_model, self.pmos_pdk_model,
                # base_nmos_width, base_pmos_width, length,
                # w_rc, pi_res, pi_cap, num_rows=num_rows,
            )
        self.subcircuit(write_drv)
        self.wdrv_inst_name = write_drv.name
        self.wdrv_inst_prefix = f"X{write_drv.name}"

        # Build the array
        self.build_array(num_rows, num_cols)

    def build_array(self, num_rows, num_cols):
        # Instantiate the SRAM array.
        self.X(self.arr_inst_name, self.arr_inst_name, 'VDD', self.gnd,
               *[f'BL{i}' for i in range(self.num_cols)],
               *[f'BLB{i}' for i in range(self.num_cols)],
               *[f'WL{i}' for i in range(self.num_rows)])
        
        for col in range(num_cols):
            # self.X(
            #     self.pre_inst_name+f"_{col}",
            #     self.pre_inst_name,
            #     self.NODES[0],  # Power net
            #     'PRE',          # Precharge enable signal
            #     f'BL{col}',     # Connect to column bitline
            #     f'BLB{col}',    # Connect to column bitline bar
            # )
            self.X(
                self.wdrv_inst_name+f"_{col}",
                self.wdrv_inst_name,
                self.NODES[0],  # Power net
                self.NODES[1],  # Ground net
                'WE',           # Write Enable
                f'DIN{col}',    # Data In
                f'BL{col}',     # Connect to column bitline
                f'BLB{col}',    # Connect to column bitline bar
            )